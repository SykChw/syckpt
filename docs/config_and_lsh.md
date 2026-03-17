# Deep Dive: `syckpt/config.py` & `syckpt/hash.py` — Configuration, LSH & Distance-Sensitive Hashing

This document is a complete, line-by-line examination of `config.py` (162 lines) and `hash.py` (250 lines). These modules implement:

- **Locality-Sensitive Hashing (LSH)**: A hashing technique where **similar inputs produce similar outputs** — the exact opposite of cryptographic hashing.
- **Distance-Sensitive Hashing (DSH)**: Log-scale quantization that ensures numerically close hyperparameters (like `lr=0.009` and `lr=0.011`) collapse into the same hash bucket.
- **`HyperConfig`**: A transparent dict proxy with dot-notation attribute access for nested hyperparameter configurations.

---

## Table of Contents

1. [The Hashing Problem: Why Not SHA-256?](#1-the-hashing-problem-why-not-sha-256)
2. [Locality-Sensitive Hashing (LSH) — Theory](#2-locality-sensitive-hashing-lsh--theory)
3. [Distance-Sensitive Continuous Quantization (DSH)](#3-distance-sensitive-continuous-quantization-dsh)
4. [The `LSHHashGenerator` Class — Line-by-Line](#4-the-lshhashgenerator-class--line-by-line)
5. [The `HyperConfig` Class — Line-by-Line](#5-the-hyperconfig-class--line-by-line)

---

## 1. The Hashing Problem: Why Not SHA-256?

Cryptographic hash functions (SHA-256, MD5, BLAKE2) are designed for the **Avalanche Effect**: changing a single bit in the input completely and unpredictably changes the output.

```
SHA-256("lr=0.01")  → "8d1b3f...c42a"
SHA-256("lr=0.011") → "f4c91a...7e3b"   ← Completely different!
```

This is a feature for security (making it impossible to reverse-engineer inputs from outputs), but it's a **disaster** for `syckpt`'s delta compression.

### Why `syckpt` Needs Similarity-Preserving Hashes

`syckpt` uses delta compression to save 10–50× storage by computing $\Delta W = W_{\text{current}} - W_{\text{base}}$. To compute this delta, `syckpt` must find a suitable **base checkpoint** — one that used similar hyperparameters and therefore has similar weights.

If `syckpt` used SHA-256 to hash its configs:
- An experiment with `lr=0.01` produces hash `"8d1b..."`.
- Another experiment with `lr=0.010001` (nearly identical) produces hash `"f4c9..."` (completely different).
- `syckpt` cannot pair them for delta compression because their hashes have no structural similarity.
- Delta compression fails. Full checkpoints are stored. Storage savings = 0.

**The solution:** Use a hash function where **similar inputs produce similar (or identical) outputs**. This is exactly what Locality-Sensitive Hashing provides.

---

## 2. Locality-Sensitive Hashing (LSH) — Theory

### Definition

A hash family $\mathcal{H}$ is **$(d_1, d_2, p_1, p_2)$-sensitive** if for any two points $u, v$:
- If $\text{dist}(u, v) \leq d_1$ then $\Pr[h(u) = h(v)] \geq p_1$ (similar points collide with high probability)
- If $\text{dist}(u, v) \geq d_2$ then $\Pr[h(u) = h(v)] \leq p_2$ (distant points collide with low probability)

Where $d_1 < d_2$ and $p_1 > p_2$.

### The Random Hyperplane Method (SimHash)

`syckpt` uses a specific LSH technique called **SimHash** (Charikar, 2002), which approximates the **cosine distance** between high-dimensional vectors using random hyperplanes.

#### Geometric Intuition

Imagine two hyperparameter configurations $v_1$ and $v_2$ as vectors in $d$-dimensional space (where $d$ is the number of hyperparameters). The angle $\theta$ between them represents their dissimilarity.

Now, drop a random hyperplane through the origin. This hyperplane divides the space into two half-spaces (one "positive", one "negative"). Each vector falls into one of the two half-spaces.

**Key insight:** The probability that the hyperplane **separates** $v_1$ and $v_2$ (puts them on different sides) is exactly:

$$P[h(v_1) \neq h(v_2)] = \frac{\theta}{\pi}$$

This is because the hyperplane's normal vector $r$ is uniformly random in all directions. The hyperplane separates $v_1$ and $v_2$ if and only if $r$ falls within the "wedge" angle $\theta$ between them. Since $r$ is uniform on the $(d-1)$-sphere, this probability is $\theta / \pi$.

**Proof sketch:**
1. Let $r \in \mathbb{R}^d$ be the hyperplane's random normal vector.
2. Define $h_r(v) = \text{sign}(r \cdot v) = \begin{cases} 1 & \text{if } r \cdot v > 0 \\ 0 & \text{if } r \cdot v \leq 0 \end{cases}$
3. $h_r(v_1) \neq h_r(v_2)$ when $\text{sign}(r \cdot v_1) \neq \text{sign}(r \cdot v_2)$.
4. This happens when $r$ lies in the "band" of angles between the two hyperplanes orthogonal to $v_1$ and $v_2$.
5. The angular width of this band is $\theta$, and the full circle is $\pi$ (we only consider one hemisphere due to sign symmetry).
6. Therefore: $P[h_r(v_1) \neq h_r(v_2)] = \theta / \pi$.

#### Multiple Hyperplanes → Binary Hash

A single hyperplane gives a single bit of information (which half-space). To get a richer hash, `syckpt` uses **16 random hyperplanes per band**:

```
Hyperplane 1: r₁ · v > 0 → 1
Hyperplane 2: r₂ · v > 0 → 0
Hyperplane 3: r₃ · v > 0 → 1
...
Hyperplane 16: r₁₆ · v > 0 → 1
```

Concatenating these bits gives a 16-bit binary hash: `1011...1`. Converting to an integer gives the band hash.

#### Bands and Amplification

A single 16-bit hash might occasionally miscategorize similar vectors. To reduce false negatives, `syckpt` uses **multiple bands** (default: 4). Each band uses a **different set of 16 random hyperplanes**. Two configs are in the same "bucket" if they collide in **any** band.

This is the standard **banding technique** from the LSH literature. With $b$ bands, the probability that two similar vectors fail to collide in any band drops exponentially.

---

## 3. Distance-Sensitive Continuous Quantization (DSH)

Even with LSH, there's still a risk: hyperparameters like learning rates operate on **logarithmic scales**. The difference between `lr=0.01` and `lr=0.009` is negligible in practice, but in the raw vector space, these produce non-zero angular separation that could cause LSH to split them into different buckets.

`syckpt` solves this with a **pre-quantization step**: before computing the LSH hash, continuous hyperparameters are snapped to the nearest value on a predefined log-scale grid.

### `quantize_value` — Line-by-Line

```python
def quantize_value(value: float, scales: List[float] = None) -> float:
```

**Purpose:** Map a continuous float to the nearest value on a log-scale grid.

```python
    if scales is None:
        if value == 0:
            return 0
        magnitude = abs(value)
        sign = 1 if value > 0 else -1
```
Handle zero and negative values separately.

```python
        # Common learning rate scales
        scales = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
```
The default scale grid covers the full range of learning rates commonly used in deep learning (from `1e-5` for fine-tuning large models to `1.0` for SGD with warmup).

```python
        closest = min(scales, key=lambda s: abs(magnitude - s))
        return sign * closest
```
Snap to the nearest scale value using minimum absolute distance. This ensures:
- `lr=0.009` → `1e-2` (0.01)
- `lr=0.011` → `1e-2` (0.01)
- `lr=0.006` → `5e-3` (0.005)

After quantization, both `0.009` and `0.011` map to `0.01`, so their LSH vectors will be identical in that dimension — guaranteeing a hash collision.

```python
    return min(scales, key=lambda s: abs(value - s))
```
With custom scales, snap directly without magnitude/sign separation.

### `quantize_dict` — Line-by-Line

```python
def quantize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in data.items():
        if isinstance(value, float):
            result[key] = quantize_value(value)
        elif isinstance(value, dict):
            result[key] = quantize_dict(value)
        elif isinstance(value, list):
            result[key] = [
                quantize_value(v) if isinstance(v, float) else v for v in value
            ]
        else:
            result[key] = value
    return result
```

Recursively quantizes all floats in a nested config dict. Non-float values (ints, strings, bools) pass through unchanged.

**Example:**

```python
config = {"lr": 0.0095, "batch_size": 32, "betas": [0.9, 0.999], "arch": "resnet50"}

quantize_dict(config)
# → {"lr": 0.01, "batch_size": 32, "betas": [1.0, 1.0], "arch": "resnet50"}
```

---

## 4. The `LSHHashGenerator` Class — Line-by-Line

### `__init__`

```python
class LSHHashGenerator:
    def __init__(self, hash_length: int = 8, num_bands: int = 4,
                 factors: Optional[Set[str]] = None):
        self.hash_length = hash_length
        self.num_bands = num_bands
        self.factors = factors or DEFAULT_HASH_FACTORS
```

- **`hash_length`** — Length of the final hex hash string (8 → 32 bits of hash space).
- **`num_bands`** — Number of independent LSH bands (more bands = finer locality grouping).
- **`factors`** — Which config keys to include in the hash. Defaults to `DEFAULT_HASH_FACTORS`:

```python
DEFAULT_HASH_FACTORS: Set[str] = {
    "lr", "learning_rate", "seed", "batch_size", "num_epochs",
    "weight_decay", "momentum", "beta1", "beta2", "eps",
}
```

These are the hyperparameters most likely to affect weight similarity.

```python
        np.random.seed(42)  # Fixed seed for reproducibility
        self._projection_matrices = [
            np.random.randn(16, len(self.factors)) for _ in range(num_bands)
        ]
```

Generate the random hyperplane normal vectors. Each band gets a `16 × d` matrix where $d = |\text{factors}|$. Each of the 16 rows is a random vector sampled from $\mathcal{N}(0, 1)^d$ — this produces uniformly random directions on the unit sphere (after normalization, which the dot product effectively handles).

**Critical:** The seed is fixed to `42`. This ensures the same hyperplanes are used every time — necessary for reproducibility across different training runs and machines.

### `_get_factor_vector`

```python
    def _get_factor_vector(self, config: Dict[str, Any]) -> np.ndarray:
        sorted_factors = sorted(self.factors)
        values = []
```
Sort factors alphabetically to ensure consistent vector ordering regardless of dict insertion order.

```python
        for factor in sorted_factors:
            value = config.get(factor, 0)
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.append(hash(str(value)) % 1000 / 100.0)
```
- Numeric values are used directly.
- Non-numeric values (strings, etc.) are hashed to a float in [0, 10). This allows things like `optimizer_type="adam"` to contribute to the hash vector.

```python
        arr = np.array(values, dtype=np.float32)

        # Normalize to unit sphere
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        return arr
```
**Unit normalization.** The cosine similarity between two vectors $u$ and $v$ is $\frac{u \cdot v}{\|u\| \|v\|}$. By normalizing both vectors to unit length ($\|v\| = 1$), the dot product $r \cdot v$ directly measures the cosine, and the random hyperplane hash functions correctly approximate cosine distance.

### `_compute_band_hashes`

```python
    def _compute_band_hashes(self, vector: np.ndarray) -> List[int]:
        band_hashes = []
        for matrix in self._projection_matrices:
            projected = np.dot(matrix, vector)
```
For each band, project the config vector onto all 16 random hyperplane normals simultaneously via matrix multiplication. `projected` is a 16-element vector where each element is $r_i \cdot v$.

```python
            band_hash = int("".join(["1" if p > 0 else "0" for p in projected]), 2)
```
Threshold each projection: positive → "1", non-positive → "0". Concatenate into a 16-character binary string, then convert to an integer. This is the band hash — a 16-bit integer representing which side of each hyperplane the config vector falls on.

```python
            band_hashes.append(band_hash)
        return band_hashes
```

### `generate`

```python
    def generate(self, config: Dict[str, Any]) -> str:
        # Step 1: Quantize continuous values (DSH)
        quantized = quantize_dict(config)

        # Step 2: Extract factor vector and normalize
        vector = self._get_factor_vector(quantized)

        # Step 3: Compute band hashes (LSH)
        band_hashes = self._compute_band_hashes(vector)

        # Step 4: Create final hash
        combined = "".join(str(h) for h in band_hashes)
        full_hash = hashlib.sha256(combined.encode()).hexdigest()
        return full_hash[:self.hash_length]
```

The pipeline:
1. **Quantize** → snap continuous values to log-scale grid (DSH).
2. **Vectorize** → convert config to a unit vector in $\mathbb{R}^d$.
3. **Project** → compute band hashes via random hyperplane thresholding (LSH).
4. **Finalize** → concatenate band hashes, SHA-256 for uniform distribution, truncate to desired length.

The final SHA-256 is not for locality-sensitivity — it's for **uniform distribution** of the hash string. The LSH locality properties are already embedded in the band hashes; the SHA-256 just converts the band hash integers into a clean hex string.

### `generate_from_components`

```python
    def generate_from_components(self, config, model=None, optimizer=None):
```

Enriches the config dict with structural information from the model and optimizer before generating the LSH hash:

```python
        if model is not None:
            num_params = sum(p.numel() for p in model.parameters())
            config["_num_params"] = num_params

            layer_types = []
            for m in model.modules():
                if len(list(m.children())) == 0:
                    layer_types.append(type(m).__name__)
            config["_layers"] = "_".join(sorted(set(layer_types))[:5])
```
- **`_num_params`** — Total number of parameters. Two models with vastly different parameter counts are unlikely to have compatible weights for delta compression.
- **`_layers`** — A signature of the unique leaf module types (e.g., `"BatchNorm2d_Conv2d_Linear_ReLU"`). This ensures models with different architectures don't accidentally collide.

```python
        if optimizer is not None:
            opt_type = type(optimizer).__name__
            config["_opt_type"] = opt_type
            if optimizer.param_groups:
                pg = optimizer.param_groups[0]
                config["_opt_lr"] = pg.get("lr", 0)
                config["_opt_momentum"] = pg.get("momentum", 0)
```
- **`_opt_type`** — Optimizer class name (e.g., `"Adam"`, `"SGD"`). Different optimizers produce fundamentally different weight trajectories.
- **`_opt_lr`**, **`_opt_momentum`** — Extracted directly from the optimizer's first param group for finer-grained hashing.

### `similarity`

```python
    def similarity(self, hash1: str, hash2: str) -> float:
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        max_len = max(len(bin1), len(bin2))
        bin1 = bin1.zfill(max_len)
        bin2 = bin2.zfill(max_len)

        matches = sum(c1 == c2 for c1, c2 in zip(bin1, bin2))
        return matches / max_len
```

Computes the **Hamming similarity** between two hash strings. This converts both hashes to binary representations, pads to equal length, and counts the fraction of matching bits.

Because the LSH hash was constructed by thresholding against random hyperplanes, the Hamming distance between two hashes is proportional to the cosine distance between their input vectors:

$$\text{Hamming Distance} \approx \frac{\theta}{\pi} \times \text{num\_bits}$$

Therefore `similarity ≈ 1 - θ/π`, providing a meaningful measure of config closeness.

### `find_similar_configs`

```python
    def find_similar_configs(self, config, existing_configs, top_k=5):
        target_hash = self.generate(config)
        similarities = []
        for existing in existing_configs:
            existing_hash = self.generate(existing)
            sim = self.similarity(target_hash, existing_hash)
            similarities.append((existing, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
```

Finds the `top_k` most similar configs from a list. This is used for finding the best base checkpoint for delta compression across hyperparameter sweeps.

---

## 5. The `HyperConfig` Class — Line-by-Line

`HyperConfig` is a `collections.abc.Mapping` subclass that provides transparent dot-notation access to nested configuration dictionaries. Internally, it stores everything as a **flat dictionary** with dot-notation keys.

### Why Flat Storage?

Nested dictionaries are natural for humans to read:
```python
{"model": {"layers": 12, "hidden_dim": 768}}
```

But flat dictionaries are faster for lookup, iteration, and serialization:
```python
{"model.layers": 12, "model.hidden_dim": 768}
```

`HyperConfig` presents the **nested interface** to the user while maintaining **flat storage** internally.

### `__init__`

```python
class HyperConfig(Mapping):
    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs):
        self._data: Dict[str, Any] = {}
        if data:
            self._data = self._flatten_dict(data) if isinstance(data, dict) else {}
        self._data.update(kwargs)
```

On construction, if a nested dict is provided, it's immediately flattened to dot-notation.

### `_flatten_dict` — Recursive Flattening

```python
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

**Example:**
```python
_flatten_dict({"model": {"layers": 12, "attention": {"heads": 8}}})
# → {"model.layers": 12, "model.attention.heads": 8}
```

Recursively walks nested dicts, concatenating keys with `.` separator. Non-dict values become leaf entries.

### `_unflatten_dict` — Reconstruction

```python
    def _unflatten_dict(self, d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d_obj = result
            for part in parts[:-1]:
                if part not in d_obj:
                    d_obj[part] = {}
                d_obj = d_obj[part]
            d_obj[parts[-1]] = value
        return result
```

**Example:**
```python
_unflatten_dict({"model.layers": 12, "model.attention.heads": 8})
# → {"model": {"layers": 12, "attention": {"heads": 8}}}
```

Splits each dot-notation key and builds nested dicts level by level.

### `__getattr__` — Transparent Dot-Notation Access

```python
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        unflattened = self._unflatten_dict(self._data)
        if name in unflattened:
            val = unflattened[name]
            if isinstance(val, dict) and all(
                isinstance(k, str) and not any("." in k for k in v.keys())
                if isinstance(v, dict) else True
                for k, v in val.items()
                if isinstance(v, dict)
            ):
                return HyperConfig(val)
            return val
```

When a user accesses `config.model`:

1. Skip private attributes (starting with `_`).
2. Unflatten the internal flat dict.
3. Look up `"model"` in the unflattened structure.
4. If it's a dict, return a **new `HyperConfig`** wrapping that sub-tree. This enables chained access: `config.model.attention.heads`.
5. If it's a primitive, return the value directly.

```python
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
```

Fallback: check if the name exists as a flat key (e.g., `config["model.layers"]` → `12`).

### `__setattr__` — Transparent Nested Assignment

```python
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, dict):
                for k, v in self._flatten_dict({name: value}).items():
                    self._data[k] = v
            else:
                self._data[name] = value
```

Setting `config.model = {"layers": 12}` flattens the dict and stores `{"model.layers": 12}` internally. Setting `config.lr = 0.01` stores `{"lr": 0.01}` directly.

### Mapping Protocol

`HyperConfig` implements the full `collections.abc.Mapping` protocol:

```python
    def __getitem__(self, key):     # config["key"]
    def __setitem__(self, key, val) # config["key"] = val
    def __delitem__(self, key)      # del config["key"]
    def __contains__(self, key)     # "key" in config
    def __iter__(self)              # for k in config (iterates unflattened top-level keys)
    def __len__(self)               # len(config) (number of unflattened top-level keys)
```

All of these unflatten the internal dict before operating, providing a consistent nested-dict interface.

### Serialization

```python
    def to_dict(self) -> Dict[str, Any]:
        return self._unflatten_dict(self._data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperConfig":
        return cls(data)
```

`to_dict()` returns a nested dict for JSON serialization (stored in commit metadata). `from_dict()` creates a new `HyperConfig` from a nested dict (used during checkpoint loading).

### Example Usage

```python
config = HyperConfig({
    "model": {"layers": 12, "hidden_dim": 768, "attention": {"heads": 8}},
    "training": {"lr": 0.001, "batch_size": 32}
})

# Dot-notation access (returns nested HyperConfig for dicts)
config.model.layers           # → 12
config.model.attention.heads  # → 8
config.training.lr            # → 0.001

# Dict-style access
config["model.layers"]        # → 12

# Setting values
config.training.lr = 0.01     # Updates "training.lr" in flat storage

# Serialization
config.to_dict()
# → {"model": {"layers": 12, "hidden_dim": 768, "attention": {"heads": 8}},
#    "training": {"lr": 0.01, "batch_size": 32}}
```
