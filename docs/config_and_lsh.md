# Deep Dive: `syckpt/config.py` & `syckpt/hash.py`

This document deconstructs the hyperparameter tracking engine inside `syckpt`. 

Specifically, we will thoroughly explain the complex mathematics underlying **Locality-Sensitive Hashing (LSH)** and **Distance-Sensitive Hashing (DSH)**, mathematically proving *why* this hashing technique allows `syckpt` to automatically detect structurally identical model checkpoints for delta-compression.

---

## 1. The Hashing Problem in MLOps

Traditional package managers and version control systems use **Cryptographic Hash Functions** like SHA-256. 

A cryptographic hash is designed for an **Avalanche Effect**: If a single bit in the input changes, the resulting hash string becomes completely and utterly unrecognizably different.
*   `SHA256(lr=0.01)` $\rightarrow$ `8d1b...`
*   `SHA256(lr=0.010001)` $\rightarrow$ `f4c9...`

**The MLOps Delta-Compression Problem:**
When performing hyperparameter searches across hundreds of concurrent GPUs, `syckpt` uses delta-compression to save 90% of storage space by computing $W_t - W_{base}$. But how does `syckpt` *find* $W_{base}$ in a sea of millions of checkpoint blobs? 

If `syckpt` relies on SHA256, an algorithm searching with `lr=0.010001` will generate a hash that is mathematically entirely alienated from an existing `lr=0.01` baseline checkpoint. It will fail to pair them, thus failing delta compression entirely.

We need a hashing algorithm where **similar inputs produce similar hashes**.

---

## 2. Locality-Sensitive Hashing (LSH)

Locality-Sensitive Hashing (LSH) solves this by maximizing collisions. Instead of an avalanche effect, if two inputs are "close" in high-dimensional space, their resulting output hashes will structurally collide or share incredibly high Hamming similarity.

### The Mathematics: Random Projection Hyperplanes

`syckpt` utilizes a specific LSH technique based on Random Projections (often called SimHash), which mathematically approximates the **Cosine Distance** between two high-dimensional vectors.

#### Step 1: Vectorization
First, `syckpt` flattens all hyperparameters into a high-dimensional vector $v \in \mathbb{R}^d$.
```python
# From _get_factor_vector:
vector = [lr, batch_size, momentum, weight_decay, ...]
```
This vector is standardized to unit-length (normalized to the unit sphere $||v||_2 = 1$).

#### Step 2: Hyperplane Generation
When `syckpt` initializes the `LSHHashGenerator`, it generates multiple random normal matrices $P$:
```python
self._projection_matrices = [
    np.random.randn(16, len(self.factors)) for _ in range(num_bands)
]
```
Mathematically, each row $r_i$ inside $P$ represents the normal vector of a random hyperplane intersecting the origin in $\mathbb{R}^d$. 

#### Step 3: Projection and Thresholding (The Proof)
To compute the hash of the hyperparameter vector $v$, we calculate the dot product between $v$ and the random hyperplane normal $r_i$:

$$h_i(v) = \text{sign}(r_i \cdot v) = \begin{cases} 1 & \text{if } r_i \cdot v > 0 \\ 0 & \text{if } r_i \cdot v \le 0 \end{cases}$$

**Why does this approximate similarity?**
If two hyperparameters $v_1$ and $v_2$ are nearly identical, the angle $\theta$ between them is incredibly small. 
For the random hyperplane to separate them (meaning $h_i(v_1) \neq h_i(v_2)$), the hyperplane must physically slice precisely between the tiny angle $\theta$.

Because the hyperplanes are generated uniformly at random, the probability that a hyperplane splits them is directly proportional to their angle:
$$P[h_i(v_1) \neq h_i(v_2)] = \frac{\theta}{\pi}$$

Therefore, the **Hamming Distance** between the generated binary hash strings mathematically mirrors the True Cosine Distance between the raw hyperparameter setups!

### The Code Implementation in `syckpt`
```python
def _compute_band_hashes(self, vector: np.ndarray) -> List[int]:
    band_hashes = []
    for matrix in self._projection_matrices:
        projected = np.dot(matrix, vector)
        band_hash = int("".join(["1" if p > 0 else "0" for p in projected]), 2)
        band_hashes.append(band_hash)
    return band_hashes
```
By concatenating the 1s and 0s generated against the 16 random hyperplanes into an integer, `syckpt` builds a collision bucket identifier.

---

## 3. Distance-Sensitive Continuous Quantization

Furthermore, hyperparameters are rarely standard floats; they operate on log-scales. 

When searching `lr=0.01` vs `lr=0.009`, their Cosine Distance might still falsely trigger an LSH failure if the random hyperplanes get exceedingly lucky and split them anyway. To guarantee perfect base-finding for delta-compression, `syckpt` aggressively enforces log-scale continuous quantization *before* taking the dot products.

### `quantize_value`
```python
def quantize_value(value: float, scales: List[float] = None) -> float:
```
This acts as a Distance-Sensitive Hashing (DSH) precursor. 
1. Computes the magnitude and sign.
2. Checks against a predefined scale map: `[1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]`
3. Evaluates: `closest = min(scales, key=lambda s: abs(magnitude - s))`

By collapsing `0.011` and `0.009` securely into the exact `1e-2` bin floating point standard, we artificially force their mathematical vectors $v_1 / v_2$ to an angle $\theta = 0$. This guarantees an identical LSH response over that attribute axis, enabling massive scale delta-compression across massive continuous hyperparameter search grids.

---

## 4. `HyperConfig`: Dynamic State Proxying

To handle these nested hyperparameter configurations transparently to the user, `config.py` provides the `HyperConfig` mapping subclass.

### Flattening Nested Layouts
Nested dictionaries break single-layer `dict` loops and strict mapping type annotations. 
The `HyperConfig` class automatically un-nests data recursively using dot-notation string tracking:

```python
def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
```
*   `{"model": {"layers": 12}}` internally flattens entirely into: `{"model.layers": 12}`.
*   This acts as a high-speed runtime dictionary, avoiding memory allocations.

### Transparent Attribute Routing
```python
def __getattr__(self, name: str) -> Any:
```
If a user tries to access `config.model` where `model` was a nested dictionary, `__getattr__` intercepts the call, parses the flattened internal `.data` store, identifies that multiple keys resolve to the `model` prefix, and dynamically builds and returns a completely new `HyperConfig(val)` referencing the sub-root context on-the-fly. This allows infinite levels of `a.b.c.d` traversal safely.
