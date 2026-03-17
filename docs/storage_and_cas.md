# Deep Dive: `syckpt/storage.py`

This document provides a comprehensive, file-centric examination of `storage.py`. This module forms the backbone of `syckpt`, managing zero-copy PyTorch serialization, strict nested dictionary flattening, delta-compression arithmetic, and Content-Addressable Storage (CAS) over physical file systems.

Below is a detailed walkthrough of its logic, functions, and object-oriented abstractions.

---

## 1. Safetensors and the Flattening Algorithm

To eliminate out-of-memory errors and malicious code execution inherently present in Python's `pickle` (`torch.save`), `syckpt` utilizes "Zero-Copy" **Safetensors**. Safetensors strictly requires a 1D mapping of `str -> torch.Tensor`. Because PyTorch states are heavily nested (`e.g. {'state': {0: {'momentum': tensor}}}`), `storage.py` implements a destructive recursive flattening algorithm.

### `flatten_state`
```python
def flatten_state(state: Any, prefix: str = "", tensors: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Any, Dict[str, torch.Tensor]]:
```
* **Purpose:** Separates a nested Python dictionary/list/tuple structure into a lightweight JSON-serializable structure mapping, and a flat 1D dictionary of raw `torch.Tensor` objects.
* **Arguments:**
    * `state (Any)`: The current node in the recursive depth tree (a dict, list, tuple, primitive, or `torch.Tensor`).
    * `prefix (str)`: The string path taken to get to this node (e.g., `"model.layer1.weight"`).
    * `tensors (Optional[Dict])`: The accumulator dictionary collecting the un-nested vectors.
* **Mechanism:**
    * **Base Accumulator Initialization:** If `tensors` is `None` (first call), it initializes an empty dictionary.
    * **Tensor Leaf Condition:** If `isinstance(state, torch.Tensor)`, it stores the tensor in the accumulator dictionary under the `prefix` key. It returns a metadata pointer: `{"__tensor__": prefix}` replacing the array in the structure tree.
    * **Dictionary Recursion:** If it evaluates a `dict`, it iterates over keys, appending `k` to the `prefix` path, and recursively calls `flatten_state` on the values.
    * **List/Tuple Recursion:** Iterates over items, appending `[i]` to the prefix, converting `tuple` structures natively into list arrays flagged by a `{"__tuple__": [...]}` wrapper mapping.
    * **Primitive Leaf Condition:** If the node is standard floats, ints, or strings, it returns them unmodified.
* **Return:** `(structure_map, flat_tensors)`, isolating the massive GPU vectors away from the nested integer metadata.

### `unflatten_state`
```python
def unflatten_state(structure: Any, tensors: Dict[str, torch.Tensor]) -> Any:
```
* **Purpose:** The inverse of `flatten_state`. Upon restoring a checkpoint, this algorithm crawls the lightweight JSON metadata map and re-assigns the heavy floating-point arrays back into their correct deeply-nested historical positions.
* **Mechanism:** 
    * If a dictionary contains the key `__tensor__`, it queries the massive memory-mapped `tensors` object and extracts the physical array dynamically.
    * If it finds the `__tuple__` keyword wrapper, it enforces strict Python tuple generation on the extracted objects instead of returning lists.

---

## 2. Delta Compression Arithmetic

In Machine Learning, a 5GB weight tensor $W$ changes *slightly* at every training step $t$. A traditional cryptographic hash function over the tensor $H(W_t)$ would completely change, forcing standard MLOps platforms to save redundant gigabytes every epoch. 

Instead of treating the tensor as an opaque binary blob, `syckpt` implements element-wise mathematical patches directly in memory.

### The Mathematics of Gradient Updates

In standard Stochastic Gradient Descent (SGD), the weight update rule from step $t-1$ to $t$ is defined as:

$$W_t = W_{t-1} - \eta \nabla L(W_{t-1})$$

Where:
*   $W_t$ is the current weight matrix.
*   $W_{t-1}$ is the base weight matrix from the previous checkpoint.
*   $\eta$ is the learning rate (typically a very small scalar like $10^{-4}$ or $10^{-5}$).
*   $\nabla L$ is the gradient of the Loss function.

### `compute_delta`: The Sparse Difference

If we rearrange the SGD equation, we isolate the difference between checkpoints:

$$\Delta W = W_t - W_{t-1} = -\eta \nabla L(W_{t-1})$$

Because learning rates ($\eta$) are infinitesimally small, the resulting matrix $\Delta W$ is entirely populated by values clustered extremely close to $0.0$ (e.g., $0.00001$). 

When this highly-localized probability distribution of near-zeros is passed into binary serialization (Safetensors) or standard compression algorithms, the entropy of the file crashes. The resulting delta file shrinks by upwards of 90% compared to saving $W_t$ explicitly.

```python
def compute_delta(current: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
```
* **Mechanism:**
    * Iterates through all tensors in the `current` state.
    * If the exact same tensor string-pointer exists in the `base` state, and their shapes/dtypes match completely, it executes the $\Delta W$ pure mathematical subtraction: `delta[k] = v - base[k]`.
    * If the tensor is completely new (architecture changed) or dimensions warped, it writes the raw matrix verbatim.

### `apply_delta`: Exact Reconstitution

```python
def apply_delta(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
```
* **Purpose:** Perfect state resumption. Given a base tensor $W_{t-1}$ and a saved delta-tensor $\Delta W$, reconstitute the exact mathematical `current` state $W_t$.
* **Mechanism:** 
    * To retrieve the exact original weights, the loader simply performs: $W_t = W_{t-1} + \Delta W$.
    * First, it clones `base` matrices fully to prevent memory-address mutation tracking collisions inside the active PyTorch loop. 
    * Then it iterates through the delta dictionary. If dimensions match, it applies mathematical addition: `reconstructed[k] = reconstructed[k] + d`. 

---

## 3. The `CASStorage` Context Layer

This class is the core physical filesystem bridge. It coordinates the flat tensors, metadata abstractions, and `fsspec` atomic writes. It forces the local OS to act identical to a standard `.git` worktree storage directory.

### Initialization & Properties
```python
class CASStorage:
    def __init__(self, root: str):
```
* **`self.root`:** The URI (e.g. `s3://bucket/`, `./local`). Handled seamlessly by `fsspec.core.url_to_fs(root)`.
* **`.syckpt/objects/`:** The Blob Database storing Content-Addressable payload payloads.
* **`.syckpt/refs/heads/`:** The Branch pointer logs (tracking what branch is resolving to what blob).
* **HEAD Generation:** If `HEAD` doesn't exist upon initialization, it explicitly hardcodes `ref: refs/heads/main`.

### Atomic JSON & I/O Internal Utilities
* **`_atomic_write_json(self, data, path)`**:
    Writing large files directly to S3 across the internet risks corrupted blobs if the connection drops. This intercepts JSON payloads and dumps them into local `tempfile.NamedTemporaryFile`. Once finalized safely on the local OS, it executes `fsspec.put_file()` over the network atomically, completely ensuring zero corruption possibilities inside the Git tracking database.
* **`_read_json(self, path)`**: Simplistic wrapper dynamically reading string files over `fsspec.open()`.

### Git-Native Refs Management
* **`write_head(branch_name)`**: Explicitly constructs the internal `ref: refs/heads/...` symbolic string standard to track the active checking out state structure.
* **`read_head()`**: Replaces the active session dynamically based on text resolution. 
* **`write_ref(branch_name, commit_hash)`**: Drops a 40-character LSH string into the specific branch file. This is how `syckpt` tracks the head of a trial without storing 5GB duplicate objects.
* **`read_ref(branch_name)`**: Reads the string `commit_hash` natively from the branch pointer text block.
* **`list_branches()`**: Returns all string filenames present in `.syckpt/refs/heads/`.
* **`delete_ref(branch_name)`**: Executes an `fsspec.rm()` destruction operation explicitly on the branch log natively.

### Git-Native Object Management
* **`save_commit_metadata(commit_hash, commit_data)`**: Executes `_atomic_write_json` specifically routing to `.syckpt/objects/[hash].json`. This stores the metrics, step values, parent tree pointers, and nested tensor structure maps safely.
* **`load_commit(commit_hash)`**: Loads the object `.json` file.
* **`check_commit_exists(commit_hash)`**: Extremely fast boolean metadata lookup verifying existence inside `objects/`.

### Safetensors Cloud Abstraction Pipeline
* **`_save_safetensors_fsspec(tensors, path)`**, **`_load_safetensors_fsspec(path)`**:
    Because the Rust backend of `<safetensors>` operates on pure OS `mmap` syscalls, it completely fails to understand remote cloud endpoints like `gcs://`. These functions bridge the gap: they execute `save_file` and `load_file` exclusively on local OS `tempfiles`, utilizing `fsspec` network wrappers to dynamically sync the raw temporary binary blocks across the network interface correctly.

### High-Level API Methods
```python
def save_tensors(self, tensors: Dict[str, torch.Tensor], blob_hash: str, base_tensors: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
```
* **Purpose:** The primary execution gate for saving checkpoints.
* **Mechanism:**
    1. Instantiates tracking `metadata = {"is_delta": False}`.
    2. If `base_tensors` exists, it triggers `delta_tensors = compute_delta()` on the arrays, flips `is_delta = True`, and routes to `_save_safetensors_fsspec`. 
    3. If `base_tensors` is omitted, it routes the raw arrays entirely.
    4. Returns the boolean dictionary explicitly instructing the JSON tree object whether or not mathematical patch reconstitution is actively required on resumption.

```python
def load_tensors(self, blob_hash: str, base_tensors: Optional[Dict[str, torch.Tensor]] = None, is_delta: bool = False) -> Dict[str, torch.Tensor]:
```
* **Purpose:** Resumption interface. 
* **Mechanism:** Pulls the safetensors mapping block across `_load_safetensors_fsspec`. If the object metadata demands `is_delta`, it forcefully runs `apply_delta(base, loaded_tensors)` to instantly reconstitute the actual required float outputs mathematically perfect to how they existed prior to extraction.
