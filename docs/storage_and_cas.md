# Deep Dive: `syckpt/storage.py` — Content-Addressable Storage & Delta Compression

This document is a complete, line-by-line examination of `storage.py` (275 lines). This module forms the backbone of `syckpt`: it manages zero-copy PyTorch serialization via Safetensors, strict nested dictionary flattening, element-wise delta-compression arithmetic, frozen-layer hard-linking, and a Git-like Content-Addressable Storage (CAS) filesystem layer with `fsspec` atomic writes.

---

## Table of Contents

1. [Background: Git Work-Trees and Merkle Trees](#1-background-git-work-trees-and-merkle-trees)
2. [Safetensors and the Flattening Algorithm](#2-safetensors-and-the-flattening-algorithm)
3. [Delta Compression Arithmetic](#3-delta-compression-arithmetic)
4. [Sub-Layer Freezing & Hard-Linking](#4-sub-layer-freezing--hard-linking)
5. [The `CASStorage` Class](#5-the-casstorage-class)
6. [Hash Generation: From Config to Address](#6-hash-generation-from-config-to-address)

---

## 1. Background: Git Work-Trees and Merkle Trees

Before diving into code, it's important to understand the two data structures that `syckpt` borrows from Git.

### What is a Git Work-Tree?

A Git repository stores data in two layers:

1. **The Working Tree** — the visible files and directories you edit.
2. **The Object Database** (`.git/objects/`) — an immutable, content-addressed store of every version of every file ever committed.

Git's object database contains three types of objects:
- **Blob** — raw file contents (analogous to `syckpt`'s `.safetensors` files).
- **Tree** — a directory listing mapping filenames to blob hashes (analogous to `syckpt`'s `components_structure` JSON).
- **Commit** — metadata (author, message, timestamp) plus a pointer to a tree hash and a `parent` commit hash.

`syckpt` mirrors this exactly:

| Git Concept | `syckpt` Equivalent | Location |
|---|---|---|
| `.git/objects/` | `.syckpt/objects/` | Immutable blob store |
| Blob object (SHA-1 of file) | `<hash>.safetensors` | Raw tensor data |
| Tree object | `components_structure` inside `<hash>.json` | JSON mapping tensor names → pointers |
| Commit object | `<hash>.json` | Step, epoch, parent, config, RNG, blob_metadata |
| `.git/refs/heads/main` | `.syckpt/refs/heads/main` | Branch tip pointer |
| `.git/HEAD` | `.syckpt/HEAD` | Symbolic ref to active branch |

### What is a Merkle Tree?

A **Merkle Tree** (named after Ralph Merkle, 1979) is a tree data structure where:
- Every **leaf node** is labelled with the cryptographic hash of a data block.
- Every **non-leaf node** is labelled with the hash of the concatenation of its children's labels.

This means the **root hash** uniquely identifies the entire tree. If any single byte changes anywhere in the tree, the root hash changes.

In `syckpt`, commits form a **Merkle DAG** (Directed Acyclic Graph, not strictly a tree because branches can exist):

```
Commit c3 (hash: "a3f8...")
├── parent: "b7e2..."  ──→  Commit c2 (hash: "b7e2...")
│                             ├── parent: "d4c1..."  ──→  Commit c1 (hash: "d4c1...")
│                             │                             └── parent: null (root)
│                             └── blob_hash: "b7e2..."
├── blob_hash: "a3f8..."
└── frozen_links: {"model.backbone.weight": "model.backbone.weight"}
```

The `blob_hash` identifies which `.safetensors` file holds this commit's tensor data. The `parent` pointer chains commits together. To reconstruct any historical state, `syckpt` walks backwards through the parent chain — this is exactly `git log`.

---

## 2. Safetensors and the Flattening Algorithm

### Why Safetensors?

`torch.save()` uses Python's `pickle` protocol internally. Pickle has two critical problems:

1. **Arbitrary Code Execution** — A malicious `.pt` file can execute arbitrary Python code when loaded. This is a known security vulnerability (CVE-2019-16370 and others).
2. **Memory Overhead** — Pickle serializes tensors by copying them into Python byte buffers, causing 2× memory spikes.

[Safetensors](https://github.com/huggingface/safetensors) (by Hugging Face) solves both: it uses a zero-copy memory-mapped format with no code execution. However, it strictly requires a **flat `Dict[str, Tensor]`** — no nesting, no non-tensor values.

PyTorch state dictionaries are deeply nested:

```python
# model.state_dict() might look like:
{
    "layers.0.weight": tensor([...]),  # Already flat — easy
    "layers.0.bias": tensor([...]),
}

# But optimizer.state_dict() is deeply nested:
{
    "state": {
        0: {
            "step": tensor(100),
            "exp_avg": tensor([...]),
            "exp_avg_sq": tensor([...]),
        },
        1: { ... }
    },
    "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999), ...}]
}
```

`syckpt` must flatten this arbitrarily nested structure into a single `Dict[str, Tensor]` for Safetensors, while preserving the structure so it can be unflattened on load.

### `flatten_state` — Line-by-Line

```python
def flatten_state(
    state: Any,
    prefix: str = "",
    tensors: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[Any, Dict[str, torch.Tensor]]:
```

**Parameters:**
- `state` — Any Python object: a dict, list, tuple, tensor, or primitive.
- `prefix` — The dot-notation path accumulated so far (e.g., `"optimizer.state.0"`).
- `tensors` — The accumulator dictionary being built up. Shared across all recursive calls.

**Returns:** `(structure, tensors)` where `structure` is a JSON-serializable tree with tensors replaced by pointers.

```python
    if tensors is None:
        tensors = {}
```
On the first call, initialize the shared accumulator. Subsequent recursive calls pass this same dictionary.

```python
    if isinstance(state, torch.Tensor):
        tensors[prefix] = state
        return {"__tensor__": prefix}, tensors
```
**Base case: Tensor.** Store the tensor in the flat dictionary keyed by its dot-notation path. Replace the tensor in the structure with a JSON pointer `{"__tensor__": "optimizer.state.0.exp_avg"}`. This pointer is what allows `unflatten_state` to reconnect tensors later.

```python
    elif isinstance(state, dict):
        result = {}
        for k, v in state.items():
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            result[k], _ = flatten_state(v, sub_prefix, tensors)
        return result, tensors
```
**Recursive case: Dict.** Walk every key-value pair, extending the prefix with the key name. The `str(k)` handles integer keys from optimizer state dicts (e.g., `{0: {...}, 1: {...}}`).

```python
    elif isinstance(state, list):
        result = []
        for i, v in enumerate(state):
            sub_prefix = f"{prefix}[{i}]"
            res, _ = flatten_state(v, sub_prefix, tensors)
            result.append(res)
        return result, tensors
```
**Recursive case: List.** Index-based prefix (e.g., `"param_groups[0]"`). Lists are preserved as JSON arrays.

```python
    elif isinstance(state, tuple):
        result = []
        for i, v in enumerate(state):
            sub_prefix = f"{prefix}[{i}]"
            res, _ = flatten_state(v, sub_prefix, tensors)
            result.append(res)
        return {"__tuple__": result}, tensors
```
**Recursive case: Tuple.** JSON has no native tuple type, so tuples are wrapped in a sentinel `{"__tuple__": [...]}`. This is critical because PyTorch optimizer `param_groups` contain tuples (e.g., `betas=(0.9, 0.999)`) and these must be restored as tuples, not lists.

```python
    else:
        # Primitives: str, int, float, bool, None
        return state, tensors
```
**Base case: Primitive.** Strings, ints, floats, booleans, and `None` are already JSON-serializable. Pass through unchanged.

#### Worked Example

Input:
```python
state = {
    "model": {"weight": tensor([1.0, 2.0]), "bias": tensor([0.1])},
    "optimizer": {"state": {0: {"step": tensor(100)}}, "param_groups": [{"lr": 0.001}]}
}
```

Output `structure`:
```python
{
    "model": {
        "weight": {"__tensor__": "model.weight"},
        "bias": {"__tensor__": "model.bias"}
    },
    "optimizer": {
        "state": {0: {"step": {"__tensor__": "optimizer.state.0.step"}}},
        "param_groups": [{"lr": 0.001}]
    }
}
```

Output `tensors`:
```python
{
    "model.weight": tensor([1.0, 2.0]),
    "model.bias": tensor([0.1]),
    "optimizer.state.0.step": tensor(100)
}
```

### `unflatten_state` — Line-by-Line

```python
def unflatten_state(structure: Any, tensors: Dict[str, torch.Tensor]) -> Any:
```

This is the exact inverse of `flatten_state`. It walks the JSON structure and replaces every `{"__tensor__": key}` pointer with the actual tensor from the flat dictionary.

```python
    if isinstance(structure, dict):
        if "__tensor__" in structure:
            return tensors[structure["__tensor__"]]
```
If the dict contains a `__tensor__` key, look up the tensor from the flat dictionary and return it directly.

```python
        elif "__tuple__" in structure:
            return tuple(unflatten_state(v, tensors) for v in structure["__tuple__"])
```
If the dict contains a `__tuple__` key, recursively unflatten each element and wrap in a Python `tuple`.

```python
        else:
            return {k: unflatten_state(v, tensors) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [unflatten_state(v, tensors) for v in structure]
    else:
        return structure
```
Otherwise, recurse into dicts and lists, or return primitives as-is.

---

## 3. Delta Compression Arithmetic

### The Mathematics of Why Deltas Are Sparse

In Standard Stochastic Gradient Descent (SGD), the update rule at training step $t$ is:

$$W_t = W_{t-1} - \eta \nabla L(W_{t-1})$$

Where:
- $W_t \in \mathbb{R}^{m \times n}$ — the weight matrix at step $t$
- $\eta$ — the learning rate (typically $10^{-3}$ to $10^{-5}$)
- $\nabla L(W_{t-1})$ — the gradient of the loss with respect to $W_{t-1}$

The **delta** (difference) between consecutive checkpoints is:

$$\Delta W = W_t - W_{t-1} = -\eta \nabla L(W_{t-1})$$

Because $\eta$ is small, $\|\Delta W\|$ is orders of magnitude smaller than $\|W\|$. The delta tensor is **numerically sparse**: most values cluster tightly around zero. Serialization formats like Safetensors compress near-zero floating-point patterns far more efficiently than diverse weight values, often achieving **10–50× storage reduction**.

For Adam optimizer (which most modern models use), the update is:

$$W_t = W_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates. The key property remains: $\eta$ keeps the per-element changes small relative to the magnitude of $W$.

### `compute_delta` — Line-by-Line

```python
def compute_delta(
    current: Dict[str, torch.Tensor],
    base: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
```

Takes two flat tensor dictionaries (current checkpoint vs. previous "base" checkpoint) and returns a delta map.

```python
    delta = {}
    for k, v in current.items():
```
Iterate over every tensor key in the current state.

```python
        if k in base and v.shape == base[k].shape and v.dtype == base[k].dtype:
```
**Guard clause for Shape & Dtype Safety:** Delta computation is only mathematically valid when the base contains exactly the same tensor topology. `syckpt` inherently protects against:
- **Architecture mutations:** If you add a new layer, `v` won't exist in `base`, triggering the fallback.
- **Precision mismatch:** If you resume training in `bf16` after starting in `fp32`, `v.dtype == base[k].dtype` fails. `syckpt` will securely flush the full `bf16` tensor rather than risk mathematically corrupting precision through an `fp32 - bf16` delta downcast.
If this guard falls through, we enter the raw-tensor fallback branch below.

```python
            if torch.equal(v, base[k]):
                delta[k] = {"__frozen__": k}
```
**Frozen layer detection.** `torch.equal()` is a C++ kernel that compares two tensors element-by-element and short-circuits on the first mismatch — it's extremely fast. If a layer is completely unchanged (e.g., a frozen backbone with `requires_grad=False`), there's no point computing or storing a delta of zeros. Instead, emit a sentinel `{"__frozen__": key}` that tells the storage layer to create a virtual hard-link.

```python
            else:
                delta[k] = v - base[k]
```
**Delta computation.** Element-wise subtraction produces the $\Delta W$ tensor. This is the core of the space-saving: instead of storing a 600 MB weight matrix, we store a 30 MB delta.

```python
        else:
            delta[k] = v
```
**Fallback: raw tensor.** If the key doesn't exist in the base, or the shape/dtype changed, store the full tensor. This handles new layers or architecture changes gracefully.

```python
    return delta
```

### `apply_delta` — Line-by-Line

```python
def apply_delta(
    base: Dict[str, torch.Tensor],
    delta: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
```

Reconstructs full tensors by applying a delta map to a base snapshot: $W_t = W_{\text{base}} + \Delta W$.

```python
    reconstructed = {}
    for k, d in delta.items():
        if isinstance(d, dict) and "__frozen__" in d:
            reconstructed[k] = base[d["__frozen__"]].clone()
```
**Frozen link resolution.** If the delta entry is a `{"__frozen__": key}` sentinel, clone the tensor from the base. The `.clone()` creates an independent copy so that subsequent in-place operations (e.g., optimizer steps) don't corrupt the base.

```python
        elif k in base and torch.is_tensor(d) and d.shape == base[k].shape and d.dtype == base[k].dtype:
            reconstructed[k] = base[k].clone() + d
```
**Delta patch.** The core reconstruction: $W_t = W_{\text{base}} + \Delta W$. Clone-then-add prevents aliasing bugs.

```python
        else:
            reconstructed[k] = d
```
**Raw new tensor.** If there's no base to patch against, use the delta entry directly — it's a full tensor.

```python
    return reconstructed
```

---

## 4. Sub-Layer Freezing & Hard-Linking

Modern training regimes frequently freeze large portions of the model:

- **Transfer learning**: Freeze a pre-trained ResNet/BERT backbone, train only the classification head.
- **Adapter tuning (LoRA, QLoRA)**: Freeze all original weights, train only small adapter matrices.
- **Progressive unfreezing**: Gradually unfreeze layers during training.

In all these cases, the frozen parameters satisfy $\Delta W = 0$ exactly (no gradient flows through them, so no update occurs). Computing and storing a tensor of zeros would waste both CPU time and disk space.

### How `syckpt` Handles Frozen Layers

1. **Detection** — `compute_delta()` calls `torch.equal(current, base)`. This is an optimized C++ comparison that checks byte-level equality and short-circuits at the first differing element. For truly frozen layers, it returns `True` immediately.

2. **Metadata Encoding** — Instead of writing a zeros tensor to the `.safetensors` file, `syckpt` records the frozen layer in the commit's JSON metadata:

```json
{
    "blob_metadata": {
        "is_delta": true,
        "frozen_links": {
            "backbone.layer0.weight": "backbone.layer0.weight",
            "backbone.layer0.bias": "backbone.layer0.bias",
            "backbone.layer1.weight": "backbone.layer1.weight"
        }
    }
}
```

3. **Separation** — In `CASStorage.save_tensors()`, the frozen sentinels are separated from the pure tensor deltas:

```python
pure_tensors = {}
for k, v in delta_map.items():
    if isinstance(v, dict) and "__frozen__" in v:
        metadata["frozen_links"][k] = v["__frozen__"]  # Goes to JSON
    else:
        pure_tensors[k] = v  # Goes to .safetensors
```

The `.safetensors` file contains **only the layers that actually changed**. Frozen layers cost zero bytes.

4. **Reconstruction** — During `load_tensors()`, the frozen links are re-injected into the delta map before calling `apply_delta()`:

```python
delta_map = dict(loaded_tensors)
if frozen_links:
    for k, frozen_key in frozen_links.items():
        delta_map[k] = {"__frozen__": frozen_key}
return apply_delta(base_tensors, delta_map)
```

### Storage Savings Calculation

For a 150M-parameter ResNet-50 where you freeze the first 140M parameters (the convolutional backbone) and fine-tune only the 10M-parameter classification head:

| Approach | Per-checkpoint Size | 50 Checkpoints |
|---|---|---|
| Naïve `torch.save` | ~600 MB | ~30 GB |
| Delta only (no freeze detection) | ~600 MB delta (zeros) | ~30 GB |
| Delta + freeze detection (`syckpt`) | ~40 MB (head deltas only) | ~2 GB |

---

## 5. The `CASStorage` Class

`CASStorage` is the filesystem abstraction layer that manages the `.syckpt/` directory structure. It supports both local filesystems and cloud storage (S3, GCS) via `fsspec`.

### `__init__` — Line-by-Line

```python
class CASStorage:
    def __init__(self, root: str):
        self.root = root
        self.fs, self.fs_path = fsspec.core.url_to_fs(root)
```
`fsspec.core.url_to_fs()` is the key abstraction. Given a path:
- `"./experiments"` → returns `(LocalFileSystem, "/absolute/path/to/experiments")`
- `"s3://bucket/experiments"` → returns `(S3FileSystem, "bucket/experiments")`

Every subsequent file operation uses `self.fs` (the filesystem implementation) and `self.fs_path` (the root path on that filesystem). This means the exact same code works for local disk, S3, GCS, HDFS, etc.

```python
        self.syckpt_dir = f"{self.fs_path}/.syckpt"
        self.objects_dir = f"{self.syckpt_dir}/objects"
        self.refs_dir = f"{self.syckpt_dir}/refs/heads"

        self.fs.makedirs(self.objects_dir, exist_ok=True)
        self.fs.makedirs(self.refs_dir, exist_ok=True)
```
Create the Git-like directory structure. `exist_ok=True` makes this idempotent.

```python
        head_path = f"{self.syckpt_dir}/HEAD"
        if not self.fs.exists(head_path):
            self.write_head("main")
```
Initialize `HEAD` to point to the `main` branch, just like `git init`.

### `_atomic_write_json` — Crash-Safe JSON Writes

```python
    def _atomic_write_json(self, data: Any, path: str):
        class TensorEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
```
A custom JSON encoder that handles PyTorch tensors and NumPy arrays. Single-element tensors (like optimizer step counters) are converted to scalars via `.item()`. Multi-element tensors and arrays are converted to Python lists.

```python
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp, cls=TensorEncoder)
            tmp_path = tmp.name
        try:
            self.fs.put_file(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
```
**The Atomic Write Pattern:**
1. Write the JSON to a **temporary local file** first. If the process crashes mid-write, only the temp file is corrupted — the real commit file is untouched.
2. Use `fsspec.put_file()` to **atomically move** the completed temp file to the final destination. On local filesystems, this is typically an `os.rename()` (atomic on most POSIX systems). On cloud storage, this is a single PUT operation.
3. Clean up the temp file in a `finally` block.

This pattern prevents corrupted commit metadata even if the machine loses power mid-write.

### `save_tensors` — The Delta Save Pipeline

```python
    def save_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        blob_hash: str,
        base_tensors: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
```

```python
        blob_path = f"{self.objects_dir}/{blob_hash}.safetensors"
        metadata = {"blob_hash": blob_hash, "is_delta": False, "frozen_links": {}}
```
Initialize metadata assuming no delta. If base tensors are provided, we'll upgrade to delta mode.

```python
        if base_tensors is not None:
            delta_map = compute_delta(tensors, base_tensors)
```
Compute element-wise differences and detect frozen layers.

```python
            pure_tensors = {}
            for k, v in delta_map.items():
                if isinstance(v, dict) and "__frozen__" in v:
                    metadata["frozen_links"][k] = v["__frozen__"]
                else:
                    pure_tensors[k] = v
```
**Critical separation.** Safetensors can only store `Dict[str, Tensor]`, not arbitrary dicts. The `{"__frozen__": key}` sentinels must be routed to the JSON metadata, and only pure tensor deltas go to the `.safetensors` file.

```python
            self._save_safetensors_fsspec(pure_tensors, blob_path)
            metadata["is_delta"] = True
```
Write only the changed tensor deltas. Mark this blob as a delta so the loader knows to apply `apply_delta()`.

```python
        else:
            self._save_safetensors_fsspec(tensors, blob_path)
```
No base → store the full snapshot.

```python
        return metadata
```

### `load_tensors` — The Delta Resolution Pipeline

```python
    def load_tensors(
        self,
        blob_hash: str,
        base_tensors: Optional[Dict[str, torch.Tensor]] = None,
        is_delta: bool = False,
        frozen_links: Optional[Dict[str, str]] = None
    ) -> Dict[str, torch.Tensor]:
```

```python
        blob_path = f"{self.objects_dir}/{blob_hash}.safetensors"
        if not self.fs.exists(blob_path):
            raise FileNotFoundError(f"Blob {blob_hash} not found in CAS storage.")
        loaded_tensors = self._load_safetensors_fsspec(blob_path)
```
Load the raw tensor data from disk via Safetensors.

```python
        if is_delta:
            if base_tensors is None:
                raise ValueError(...)
            delta_map = dict(loaded_tensors)
            if frozen_links:
                for k, frozen_key in frozen_links.items():
                    delta_map[k] = {"__frozen__": frozen_key}
            return apply_delta(base_tensors, delta_map)
```
**Re-injection:** The frozen links were stored in JSON, not in the `.safetensors` file. Re-inject them as `{"__frozen__": key}` sentinels so that `apply_delta()` can process everything uniformly.

```python
        else:
            return loaded_tensors
```
Not a delta → return the raw tensors as-is.

### Branch and Ref Operations

The remaining methods mirror Git's ref management:

```python
    def write_head(self, branch_name: str):
        # Writes "ref: refs/heads/main" to .syckpt/HEAD

    def read_head(self) -> str:
        # Parses HEAD, returns branch name (e.g., "main")

    def write_ref(self, branch_name: str, commit_hash: str):
        # Writes commit hash to .syckpt/refs/heads/<branch>

    def read_ref(self, branch_name: str) -> Optional[str]:
        # Reads commit hash from .syckpt/refs/heads/<branch>

    def list_branches(self) -> List[str]:
        # Lists all files in .syckpt/refs/heads/

    def delete_ref(self, branch_name: str) -> bool:
        # Removes a branch ref file

    def save_commit(self, commit_hash: str, commit_data: Dict[str, Any]):
        # Atomically writes commit JSON to .syckpt/objects/<hash>.json

    def load_commit(self, commit_hash: str) -> Dict[str, Any]:
        # Reads and parses commit JSON

    def check_commit_exists(self, commit_hash: str) -> bool:
        # Checks if .syckpt/objects/<hash>.json exists
```

---

## 6. Hash Generation: From Config to Address

`syckpt` uses **Locality-Sensitive Hashing (LSH)** rather than cryptographic hashing (like SHA-256) to generate commit addresses. This is a deliberate design choice: two experiments with nearly identical hyperparameters (e.g., `lr=0.01` vs `lr=0.011`) should produce similar or identical hashes, enabling delta compression across experiments.

The process:

1. `LSHHashGenerator._get_factor_vector(config)` → Flatten hyperparameters into a normalized vector $v \in \mathbb{R}^d$
2. `quantize_value()` → Snap continuous values to a log-scale grid (DSH)
3. `_compute_band_hashes(vector)` → Project $v$ onto random hyperplanes, threshold to binary
4. Concatenate band hashes → SHA-256 → truncate to `hash_length` characters

The full mathematical details are in [config_and_lsh.md](./config_and_lsh.md).

> **Note:** Because LSH is designed to produce collisions for similar configs, `syckpt` appends a UUID suffix when a new commit would collide with its parent: `current_hash = f"{current_hash}-{uuid.uuid4().hex[:6]}"`. This prevents overwriting the parent's`.safetensors` file.
