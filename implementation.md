# Implementation Guide: Architecting `ckpt` v0.0.1

Welcome to the comprehensive implementation guide for `ckpt` v0.0.1. This document serves as a deep-dive educational resource explaining every computer science concept, mathematical mechanism, and engineering design choice utilized to build a robust, Git-native checkpointing system for PyTorch.

## Table of Contents
1. [Content-Addressable Storage (CAS) & Delta Compression](#1-content-addressable-storage-cas--delta-compression)
2. [Locality-Sensitive Hashing (LSH)](#2-locality-sensitive-hashing-lsh)
3. [Safetensors: Zero-Copy and Memory Mapping](#3-safetensors-zero-copy-and-memory-mapping)
4. [Filesystem Abstraction with `fsspec`](#4-filesystem-abstraction-with-fsspec)
5. [Exact Mathematical Resumption](#5-exact-mathematical-resumption)
6. [Ecosystem Compatibility (`.ckpt` Export)](#6-ecosystem-compatibility-ckpt-export)
7. [Code Structure Walkthrough](#7-code-structure-walkthrough)

---

## 1. Content-Addressable Storage (CAS) & Delta Compression

Standard MLOps tools frequently suffer from disk-space exhaustion because they save massive, redundant 5GB `.pt` files at every epoch. We solve this using **Content-Addressable Storage (CAS)** and **Delta Compression**.

### How Traditional Git Works
In Git, files are hashed (using SHA-1) and stored in `.git/objects/`. If you commit the exact same file twice, Git only stores the bytes once. This is CAS.
However, in Machine Learning, the 5GB weight tensor changes *slightly* at every training step. A traditional SHA-256 hash would completely change, forcing the system to store a brand new 5GB blob every step. 

### Delta Compression Mechanism
Instead of saving the entire model at step `T`, `ckpt` identifies the most similar previous model structure (step `T-1`). 
It then performs an element-wise mathematical subtraction:
`delta_tensor = current_tensor - base_tensor`
Because training steps represent infinitesimally small gradient updates, the resulting `delta_tensor` consists of values clustered extremely close to zero. These highly-compressible deltas are saved instead of full weights. When you "checkout" a branch, `ckpt` fetches the original base tensor, fetches the delta, and perfectly reconstructs: `current_tensor = base_tensor + delta_tensor`.

---

## 2. Locality-Sensitive Hashing (LSH)

If Delta Compression requires finding a "similar" base model, how do we search for similar experiments across thousands of branches instantly? 
We use **Locality-Sensitive Hashing (LSH)**.

While cryptographic hashes (like SHA-256) are designed to avoid collisions, LSH is designed to purposefully *maximize* collisions for similar items. 
1. **Quantization:** We bucket continuous hyperparameters. A learning rate of `0.009` and `0.011` both quantize to `0.01`.
2. **Random Projections:** We convert hyperparameter keys (like `batch_size`, `seed`, `lr`) into continuous vectors and project them across random mathematical hyperplanes.
3. **Bucketing:** Configurations that fall on the same side of multiple hyperplanes generate the exact same binary hash prefix.

When you call `ckpt.save()`, the manager generates an LSH hash. If that hash already exists, the project knows you are running a mathematically similar trial, enabling rapid similarity searches and delta-base target finding.

---

## 3. Safetensors: Zero-Copy and Memory Mapping

Historically, PyTorch relied on Python's `pickle` module (`torch.save`). `pickle` is notoriously insecure (capable of executing arbitrary malicious code) and extremely memory inefficient. Loading a 10GB pickled model requires 10GB of system RAM just to hold the binary string before it gets moved to GPU memory, risking `Out Of Memory (OOM)` errors.

### The Safetensors Solution
To circumvent this, `ckpt` v0.0.1 exclusively utilizes **Safetensors**. Safetensors is a format built in Rust that utilizes **Memory Mapping (mmap)**. 
When `ckpt` loads a `safetensors` file, the operating system maps the file on the hard drive directly into the Python process's virtual memory space safely, bypassing the need to load the bytes into RAM entirely. It’s "Zero-Copy", meaning the GPU fetches the bytes directly from SSD to VRAM.

### Binarization and Flat Dictionaries
Safetensors strictly requires a **flat dictionary** of `str -> torch.Tensor`. It rejects nested dictionaries, integers, strings, or Python lists. 
Because PyTorch optimizer states contain heavily nested metadata combined with tensors (`e.g., {'state': {0: {'momentum': tensor}}}`), `ckpt` implements a recursive flattening algorithm in `storage.py`.
1. It transverses the Python structure.
2. Extracts tensors and places them in a flat dictionary: `{"optimizer.state.0.momentum": tensor}`.
3. Saves a metadata map ("components_structure") detailing exactly where the tensors belong.
4. Serializes the flat tensors to disk, and the metadata structure natively alongside the commit.

---

## 4. Filesystem Abstraction with `fsspec`

Machine Learning engineers demand the ability to run locally on a laptop, but scale to Amazon S3 or Google Cloud Storage with zero code changes.

`ckpt` manages this using `fsspec` (Filesystem Spec). When you initialize `CheckpointManager("s3://my-bucket/experiments")`, `fsspec` dynamically loads the AWS storage drivers. 

### Why Temporary Files are Critical for Atomic Cloud Writes
If an internet connection drops while writing a 5GB file to an S3 object, the checkpoint corrupts. To guarantee **Atomic** guarantees, `storage.py` writes the `safetensors` block locally to an OS-level temporary file (`tempfile.NamedTemporaryFile`) first. Only when the serialization completely successfully finishes does `fsspec.put_file()` stream the finished blob to the target filesystem. 

---

## 5. Exact Mathematical Resumption

Resuming an experiment involves more than restoring model weights. If random states or dataloaders fall out of sync, the training loss curve will spike aggressively upon resumption. 

### State-of-the-Art RNG Synchronization 
`ckpt` captures:
* **PyTorch Global / CUDA**: Serialized natively.
* **Modern Numpy Generators**: `np.random.Generator` states are captured via their `bit_generator.state` API.
* **Distributed Data Parallel (DDP)**: In massive multi-GPU topologies, `manager.py` utilizes PyTorch Distributed barriers (`dist.barrier()`) and explicitly gathers the RNG state dicts from every rank to the main rank using `dist.gather_object()`. When resumed, it broadcasts them back to their respective hardware.

### Fast-Forwarding DataLoaders
PyTorch `DataLoader` objects notoriously contain no native internal state tracking. If resumed mid-epoch, a naive standard loop will serve identical data points!
To fix this, we implemented `StatefulDataLoader` (`dataloader.py`). It traps the `batch_idx` and deterministically reseeds its internal `torch.Generator` against the epoch. By intercepting the `RandomSampler` index permutations, it slices the yielded array `_indices[items_to_skip:]`. This guarantees that resumption immediately processes the exact mathematically required next batch without manually running thousands of empty `next()` iterations.

---

## 6. Ecosystem Compatibility (`.ckpt` Export)

While generating a local Git-Tree `.syckpt` repository allows for instant branching and zero-copy version control, the broader PyTorch ecosystem expects standard monolithic `.ckpt` or `.pt` files for inference deployment on platforms like HuggingFace.

`ckpt` implements a specific Ecosystem Escape Hatch: `export_ckpt(hash, output_path)`.
Under the hood, this method:
1. Queries the CAS system and recursively builds the delta-compressed `safetensors`.
2. Reloads the nested Python metadata structure.
3. Repopulates the flat tensors into their nested dictionary positions.
4. Serializes the monolithic object safely into a portable standard `.ckpt` file.

---

## 7. Line-by-Line Code Walkthrough

### `storage.py` 

**State Flattening:**
```python
def flatten_state(state: Any, prefix: str = "", tensors: Optional[Dict[str, torch.Tensor]] = None):
    # Recursively traverses a dictionary. If it finds a tensor, it intercepts it, references its 
    # specific prefix path (e.g. `model.layers.0.weight`), and stores it flat.
    if isinstance(state, torch.Tensor):
        tensors[prefix] = state
        return {"__tensor__": prefix}, tensors
```

**Git-Native `.syckpt` Backend:**
```python
    def save_tensors(self, tensors: Dict[str, torch.Tensor], blob_hash: str, base_tensors: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        # If the CheckpointManager passes a 'base_tensor' from a previous commit, 
        # compute_delta is run. The resulting delta dict is routed through the atomic fsspec writer.
```

### `manager.py`

**DDP Saftey Architecture within `save()`:**
```python
    import torch.distributed as dist
    # Blocks all GPU ranks until they reach the exact identical point in the code.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        is_main = dist.get_rank() == 0
    # ...
    # The LSH hash generated on the main process is broadcast to all compute nodes.
    # This ensures that even though only the main process writes to disk, every GPU 
    # remembers the exact string hash representing their current state.
    dist.broadcast_object_list(hash_list, src=0)
```
