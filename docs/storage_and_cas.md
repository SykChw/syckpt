# Deep Dive: `syckpt/storage.py`

This document provides a comprehensive, file-centric examination of `storage.py`. This module forms the backbone of `syckpt`, managing zero-copy PyTorch serialization, strict nested dictionary flattening, delta-compression arithmetic, and Content-Addressable Storage (CAS) over physical file systems.

---

## 1. Safetensors and the Flattening Algorithm

To eliminate out-of-memory errors and malicious code execution inherently present in Python's `pickle` (`torch.save`), `syckpt` utilizes "Zero-Copy" **Safetensors**. Safetensors strictly requires a 1D mapping of `str -> torch.Tensor`. Because PyTorch states are heavily nested (`e.g. {'state': {0: {'momentum': tensor}}}`), `storage.py` implements a destructive recursive flattening algorithm.

### `flatten_state`
```python
def flatten_state(state: Any, prefix: str = "", tensors: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Any, Dict[str, torch.Tensor]]:
```
* **Purpose:** Separates a nested Python dictionary/list/tuple structure into a lightweight JSON-serializable structure mapping, and a flat 1D dictionary of raw `torch.Tensor` objects.
* **Mechanism:** 
    * Isolates massive GPU vectors away from the nested integer metadata.
    * Replaces tensors with a pointer: `{"__tensor__": prefix}`.
    * Handles tuples by wrapping in a `{"__tuple__": [...]}` flag.

### `unflatten_state`
```python
def unflatten_state(structure: Any, tensors: Dict[str, torch.Tensor]) -> Any:
```
* **Purpose:** The inverse of `flatten_state`. It crawls the JSON metadata map and re-assigns the heavy floating-point arrays back into their deeply-nested positions.

---

## 2. Delta Compression Arithmetic

In Machine Learning, a 5GB weight tensor $W$ changes *slightly* at every training step $t$. Instead of treating the tensor as an opaque binary blob, `syckpt` implements element-wise mathematical patches directly in memory.

### The Mathematics of Sparse Differences

In standard Stochastic Gradient Descent (SGD):
$$W_t = W_{t-1} - \eta \nabla L(W_{t-1})$$

Rearranging to isolate the change:
$$\Delta W = W_t - W_{t-1}$$

Because learning rates ($\eta$) are small, $\Delta W$ is highly sparse (clustered near 0). Compressing this sparse difference is significantly more efficient than saving $W_t$.

---

## 3. Sub-Layer Freezing & Hard-Linking

Modern training (like Transfer Learning or Adapter-based tuning) often involves "frozen" backbone layers where `requires_grad=False`. In these cases, $\Delta W = 0$.

### The Hard-Linking Paradigm

Instead of computing a delta of zeros and writing a redundant file, `syckpt` detects if a layer is identical to its parent reference. It then injects a **virtual hard-link** into the metadata:

```json
"frozen_links": {
    "backbone.layer0.weight": "parent_commit_hash"
}
```

This functions identically to Docker layer caching or Git worktrees. During resumption, `syckpt` detects this link and simply points the reconstructed state back to the existing memory-address of the base commit, achieving **Zero Storage Cost** for frozen parameters.

---

## 4. The `CASStorage` Context Layer

This class manages the physical filesystem bridge, coordinating flat tensors and `fsspec` atomic writes.

### Git-Native Architecture
* **`.syckpt/objects/`**: The Blob Database.
* **`.syckpt/refs/heads/`**: Branch pointer logs.
* **Atomic Writes**: Uses `tempfile` and `fsspec.put_file()` to prevent corrupted blobs on cloud storage.

### Resumption Lifecycle
1. **`load_commit`**: Fetches the JSON metadata tree.
2. **Delta Resolution**: If `is_delta=True`, it fetches the `base_hash` recursively.
3. **Hard-Link Injection**: Re-injects `frozen_links` into the delta map.
4. **`apply_delta`**: Reconstitutes the final weights: $W_t = W_{t-1} + \Delta W$.

By combining mathematical deltas with architectural hard-links, `syckpt` achieves state-of-the-art storage efficiency for deep learning at scale.
