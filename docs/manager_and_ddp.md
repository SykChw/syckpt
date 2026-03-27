# Deep Dive: `syckpt/manager.py` — Orchestration, Async Saves & DDP Synchronization

This document is a complete, line-by-line examination of `manager.py`, the core orchestration engine of `syckpt`. It covers:

- How PyTorch Distributed Data Parallel (DDP) works at the process level
- How `CheckpointManager.save()` synchronizes a multi-GPU cluster, then forks a GIL-free background process
- How the `__setattr__` proxy makes registration transparent
- How `fcntl` file locking prevents concurrent writes
- How `export_ckpt` reassembles CAS fragments into a monolithic `.ckpt`

---

## Table of Contents

1. [Background: How DDP Works](#1-background-how-ddp-works)
2. [The `Lock` Class: Distributed File Safety](#2-the-lock-class-distributed-file-safety)
3. [The `Commit` Data Class](#3-the-commit-data-class)
4. [The `CheckpointManager` Class](#4-the-checkpointmanager-class)
5. [The `save()` Method: A Complete Network Dance](#5-the-save-method-a-complete-network-dance)
6. [The `load()` and Auto-Resume Path](#6-the-load-and-auto-resume-path)
7. [Branching, Logging, and Diffing](#7-branching-logging-and-diffing)
8. [Exporting to Standard PyTorch Format](#8-exporting-to-standard-pytorch-format)
9. [Utilities: `step_up`, `loop`, Context Manager](#9-utilities-step_up-loop-context-manager)
10. [Mega-Hash Squashing](#10-mega-hash-squashing)
11. [Future: Hierarchical Mega-Hashes](#11-future-hierarchical-mega-hashes)

---

## 1. Background: How DDP Works

When training on a single GPU becomes infeasible (model too large, dataset too big, training too slow), PyTorch provides **Distributed Data Parallel (DDP)** to spread computation across multiple GPUs — potentially on different physical machines.

### The Process Model

Unlike Python's `threading` (which shares memory and is bottlenecked by the GIL), DDP spawns entirely separate **OS processes**, each with its own Python interpreter, its own GPU, and its own copy of the model weights in VRAM.

```
Machine 1                          Machine 2
┌─────────────┐                    ┌─────────────┐
│  Rank 0     │                    │  Rank 2     │
│  (Master)   │                    │  GPU 2      │
│  GPU 0      │  ◄── NVLink/   ── │  Model copy │
│  Model copy │      NCCL net     │             │
├─────────────┤                    ├─────────────┤
│  Rank 1     │                    │  Rank 3     │
│  GPU 1      │                    │  GPU 3      │
│  Model copy │                    │  Model copy │
└─────────────┘                    └─────────────┘
```

- **Rank 0 (Master)**: The primary process. Handles logging, evaluation, disk I/O, and checkpoint saving.
- **Ranks 1..N (Workers)**: Identical copies. Compute gradients independently on different mini-batch slices.

### The All-Reduce Synchronization

Each GPU computes gradients $\nabla L_i$ on its own data slice. If these gradients are applied independently, the model copies will diverge after a single step.

DDP solves this by hijacking `loss.backward()`. As gradients flow through the computation graph, PyTorch initiates an **All-Reduce** operation over the NVLink/NCCL network. This mathematically averages the gradients across all GPUs:

$$\nabla L_{\text{avg}} = \frac{1}{N} \sum_{i=0}^{N-1} \nabla L_i$$

After All-Reduce completes, every GPU holds the **identical** averaged gradient. When the optimizer steps, every GPU arrives at the **identical** weight matrix — perfect synchronization without explicit weight copying.

### The Checkpointing Danger

Because every GPU holds identical weights, we want **only Rank 0** to save the checkpoint. If all 8 GPUs try to write `weights.safetensors` to the same file path simultaneously:
- **Race condition**: Multiple processes opening the same file for writing.
- **Binary corruption**: Partial writes interleave, producing an unreadable file.
- **Cloud storage failure**: S3/GCS PUT operations are not atomic across concurrent uploads.

`syckpt` prevents this through a precise synchronization protocol in `save()`.

---

## 2. The `Lock` Class: Distributed File Safety

```python
class Lock:
    __slots__ = ("lock_path", "timeout", "_fd")

    def __init__(self, lock_path: Path, timeout: int = 30):
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd = None
```

The `Lock` class uses **POSIX `fcntl` advisory file locking** to prevent concurrent access to the `.syckpt/` directory from multiple processes on the same machine.

### How `fcntl` File Locking Works

`fcntl.flock()` is a Linux system call that requests an advisory lock on an open file descriptor. Unlike Windows mandatory locks, POSIX advisory locks are cooperative — both readers and writers must check the lock file. The lock is automatically released when the file descriptor is closed or the process terminates.

```python
    def acquire(self) -> bool:
        import time
        start = time.time()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
```
Ensure the lock file's parent directory exists.

```python
        while True:
            try:
                self._fd = open(self.lock_path, "w")
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
```
- `LOCK_EX` — **Exclusive lock**. No other process can hold any lock (shared or exclusive) on this file.
- `LOCK_NB` — **Non-blocking**. If the lock is held by another process, raise `IOError` immediately instead of waiting forever.

```python
                self._fd.write(str(os.getpid()))
                self._fd.flush()
                return True
```
Write the current PID to the lock file for debugging (allows identifying which process holds the lock).

```python
            except (IOError, OSError):
                if self._fd:
                    self._fd.close()
                if time.time() - start > self.timeout:
                    return False
                time.sleep(0.1)
```
If the lock is held, retry every 100ms until the timeout expires.

```python
    def release(self):
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
            except Exception:
                pass
            finally:
                self._fd = None
```
Release the lock by explicitly calling `LOCK_UN` and closing the file descriptor.

The class also implements `__enter__`/`__exit__` for `with` statement usage.

> **Note:** The lock is only created for local filesystem paths. Cloud storage paths (starting with `s3://`, `gcs://`, etc.) skip locking entirely, relying instead on the cloud provider's eventual consistency model and the fact that only Rank 0 writes.

---

## 3. The `Commit` Data Class

```python
class Commit:
    __slots__ = (
        "hash", "parent", "message", "step", "epoch",
        "config", "metric", "timestamp", "blob_hash",
        "blob_metadata", "components_structure", "rng", "deterministic"
    )
```

A lightweight data class representing a saved checkpoint — analogous to a Git commit. Uses `__slots__` for memory efficiency (prevents creation of a per-instance `__dict__`).

Key fields:
- **`hash`** — The LSH-derived commit identifier.
- **`parent`** — The hash of the previous commit (`None` for the first commit in a session).
- **`blob_hash`** — Points to the `.safetensors` file in `.syckpt/objects/`.
- **`blob_metadata`** — Dict with `is_delta` (bool) and `frozen_links` (dict of frozen layer keys).
- **`components_structure`** — JSON-serializable schema describing the nested structure of the state dict (needed by `group_commits` to create Mega-Hash metadata without re-reading disk).
- **`rng`** / **`deterministic`** — Captured PRNG and cuDNN state at save time, also cached in-memory to prevent race conditions in `group_commits` when async workers haven't yet written their `.json` files.

The `to_dict()` / `from_dict()` methods serialize to/from JSON for storage.

---

## 4. The `CheckpointManager` Class

### Initialization

```python
class CheckpointManager:
    __slots__ = (
        "root", "storage", "max_to_keep", "maximize", "auto_resume",
        "save_rng", "_lock", "_locked", "state_manager", "_config",
        "_step", "_epoch", "_batch_idx", "_hash", "_current_branch",
        "_commits", "_session_commits", "_session_start_hash",
        "_top_k_metrics", "run_mode", "_bg_processes",
    )
```

Again uses `__slots__` — this is a performance-conscious design since the manager is a long-lived singleton that persists for the entire training run.

```python
    def __init__(self, dirpath, max_to_keep=5, maximize=False,
                 auto_resume=True, save_rng=True, lock_timeout=30, hash_length=8):
        self.root = str(dirpath)
        self.storage = CASStorage(self.root)
```
Creates the `CASStorage` instance, which initializes the `.syckpt/` directory structure.

```python
        self.state_manager = StateManager()
        self._config = HyperConfig()
```
- `StateManager` — the component registry that duck-types `.state_dict()` on registered objects.
- `HyperConfig` — the nested dict proxy that tracks hyperparameters for LSH hashing.

```python
        self.run_mode = run_mode
        self._session_commits: List[str] = []
        self._session_start_hash: Optional[str] = None
        self._top_k_metrics: List[Tuple[float, str]] = []
        self._bg_processes: list = []

        self._hash: Optional[str] = None
        self._current_branch = self.storage.read_head()
        self._commits: Dict[str, Commit] = {}
```
- `run_mode` — controls branching behaviour on each `with` invocation.
- `_session_commits` — hashes collected during the current `loop()` / `with` block, used by `group_commits` to build the Mega-Hash.
- `_bg_processes` — list of background `multiprocessing.Process` objects. Joined before `group_commits` is called so that workers can't overwrite the branch ref after the Mega-Hash has been written.

```python
        branch_hash = self.storage.read_ref(self._current_branch)
        if branch_hash and self.storage.check_commit_exists(branch_hash):
            self._hash = branch_hash
            try:
                data = self.storage.load_commit(branch_hash)
                if not data.get("is_mega"):
                    self._load_commit_into_cache(branch_hash)
            except Exception:
                pass
```

On startup, cache the branch tip — but **skip Mega-Hash commits** (they have no tensor blob and exist only as UI grouping nodes).

### The `__setattr__` Proxy — Magic Registration

One of `syckpt`'s distinctive features is that users register components via simple attribute assignment:

```python
ckpt = CheckpointManager("./experiments")
ckpt.model = model        # Registers model with StateManager
ckpt.optimizer = optimizer  # Registers optimizer with StateManager
```

This works because `CheckpointManager` overrides `__setattr__`:

```python
    def __setattr__(self, name: str, value: Any):
        if name.startswith("_") or name in (
            "config", "step", "epoch", "batch_idx", "hash", "branch",
            "storage", "root", "max_to_keep", "maximize", "auto_resume",
            "save_rng", "state_manager"
        ):
            object.__setattr__(self, name, value)
        else:
            self.state_manager.register(**{name: value})
```

**Logic:**
1. If the attribute name starts with `_` (private) or is one of the known internal attributes, use the default Python `object.__setattr__` to set it normally.
2. Otherwise, redirect the assignment to `StateManager.register()`. This means `ckpt.foo = bar` actually calls `self.state_manager.register(foo=bar)`.

The corresponding `__getattr__` intercepts attribute reads:

```python
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        comp = self.state_manager.get(name)
        if comp is not None:
            return comp
        raise AttributeError(...)
```

This means `ckpt.model` transparently returns the registered model object.

### Hash Generation

```python
    def _generate_hash(self) -> str:
        gen = LSHHashGenerator(hash_length=8)
        config_dict = self._config.to_dict() if self._config else {}
        components = {}
        for name in self.state_manager.list_components():
            components[name] = self.state_manager.get(name)
        return gen.generate_from_components(
            config_dict, components.get("model"), components.get("optimizer")
        )
```

The hash is generated from three sources:
1. **Hyperparameter config** — learning rate, batch size, etc.
2. **Model architecture signature** — number of parameters, unique layer types.
3. **Optimizer signature** — optimizer type name, learning rate from param groups.

These are combined into a vector, projected through LSH random hyperplanes, and truncated to 8 hex characters. See [config_and_lsh.md](./config_and_lsh.md) for the full mathematics.

---

## 5. The `save()` Method: A Complete Network Dance

This is the most complex function in the entire codebase. Let's trace it line by line.

### Step 1: DDP Detection and Barrier

```python
    def save(self, metric: Optional[float] = None, message: str = "") -> str:
        self._lock_acquire()
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                is_main = dist.get_rank() == 0
                world_size = dist.get_world_size()
                is_dist = True
            else:
                is_main = True
                is_dist = False
```

`dist.barrier()` is a **collective operation**: every process in the group must call it before any process can proceed. This ensures all GPUs have completed the same training step before checkpointing begins.

If `syckpt.save()` is called at slightly different wall-clock times by different GPUs (which is normal due to data loading variance), the barrier synchronizes them.

### Step 2: Hash Generation and Broadcast

```python
            current_hash = self._generate_hash() if is_main else ""

            if is_dist:
                hash_list = [current_hash]
                dist.broadcast_object_list(hash_list, src=0)
                current_hash = hash_list[0]
```

**Why broadcast the hash?** Each GPU could independently generate a hash, but due to floating-point non-determinism in the LSH computation, they might produce slightly different strings. By having Rank 0 generate the authoritative hash and broadcasting it, all ranks agree on a single commit identifier.

`dist.broadcast_object_list()` serializes the Python object on Rank 0, sends it over the NCCL network, and deserializes it on all other ranks.

### Step 3: RNG State Gathering

```python
            rng_state = get_rng_state() if self.save_rng else None

            if is_dist and self.save_rng:
                if is_main:
                    gathered_rngs = [None for _ in range(world_size)]
                    dist.gather_object(rng_state, gathered_rngs, dst=0)
                    rng_state = gathered_rngs
                else:
                    dist.gather_object(rng_state, dst=0)
```

For perfect DDP resumption, we need the PRNG state of **every GPU independently**. Each GPU processes different data slices, so their CUDA random states (used for dropout, data augmentation, etc.) have diverged.

`dist.gather_object()` collects each rank's `rng_state` dict and places them in an array on Rank 0. After this call, `rng_state` on Rank 0 is a list: `[rank0_rng, rank1_rng, ..., rankN_rng]`.

### Step 4: Worker Exit

```python
            if not is_main:
                if is_dist:
                    dist.barrier()
                self._hash = current_hash
                return current_hash
```

**All non-master ranks exit immediately.** They don't need to compute deltas, write files, or wait for disk I/O. They simply store the new hash locally and return to the training loop, ready for the next forward pass.

The second `dist.barrier()` here ensures workers don't race ahead and attempt another `save()` call before Rank 0 has finished setting up the async write.

### Step 5: Anti-Collision Guard

```python
            base_hash = self._hash

            base_c_hash = current_hash
            while self.storage.check_commit_exists(current_hash) or current_hash in self._commits:
                import uuid
                current_hash = f"{base_c_hash}-{uuid.uuid4().hex[:6]}"

            # If the base has never been committed, treat this save as a root.
            if base_hash == current_hash or not self.storage.check_commit_exists(base_hash):
                base_hash = None
```

LSH is designed to produce **identical hashes** for similar configs — that's its purpose. But if the same config is used across consecutive checkpoints (same model, same optimizer, same hyperparams), the LSH hash would be identical to the parent's hash. This would:
1. Overwrite the parent's `.safetensors` file, destroying the base needed for delta resolution.
2. Create a **self-referential parent** (`parent == hash`), breaking tree traversal.

The fix uses a `while` loop (not a simple `if`) to guarantee uniqueness against **all existing objects on disk and in the in-memory cache**. The UUID suffix preserves the LSH locality property in the shared prefix (e.g. `9af15b33-24c3f9`).

The second guard sets `base_hash = None` when the base commit has never actually been written to disk (e.g. the very first save in a session, where `__enter__` generates a placeholder hash that is never committed). This prevents the first commit from having a phantom parent.

### Step 6: Build Commit Data

```python
            metadata, flat_tensors = self._build_commit_data()

            if self.save_rng:
                metadata["rng"] = rng_state
```

`_build_commit_data()` calls `StateManager.build_state()` to gather all registered components' state dicts, then runs `flatten_state()` to separate the nested structure from the flat tensor dictionary, and captures RNG + deterministic states.

### Step 7: CPU Migration and Async Fork

```python
            blob_hash = current_hash
            commit_data = {
                "hash": current_hash,
                "parent": base_hash,
                "message": message or "Update",
                "metric": metric,
                "blob_hash": blob_hash,
                **metadata
            }
```

Build the full commit JSON by merging the hash, parent pointer, user message, metric, and the metadata from `_build_commit_data()`.

```python
            cpu_tensors = {
                k: v.to("cpu", non_blocking=True).clone()
                for k, v in flat_tensors.items()
            }
```

**Critical memory operation.** This line does two things:
1. `.to("cpu", non_blocking=True)` — Initiates an asynchronous DMA transfer from GPU VRAM to CPU RAM. The `non_blocking=True` flag means the GPU doesn't wait for the transfer to complete before continuing to the next operation.
2. `.clone()` — Creates an independent copy of the CPU tensor. This **severs the autograd graph** — the background process will hold a tensor that is completely detached from the live training computation graph. Without `.clone()`, the background process and the training loop could share the same memory buffer, causing race conditions.

```python
            def _async_save_worker(comp_tensors, c_hash, b_hash, c_data, b_branch, fs_storage):
                import logging
                logger = logging.getLogger(__name__)

                logger.info(f"Async Save Started [PID: {os.getpid()}] processing blob {c_hash}")
                b_tensors = None
                if b_hash and fs_storage.check_commit_exists(b_hash):
                    # Skip Mega-Hash commits — they are UI containers with no tensor blob.
                    try:
                        b_meta = fs_storage.load_commit(b_hash)
                        if not b_meta.get("is_mega"):
                            b_tensors = fs_storage.load_tensors(b_hash, is_delta=False)
                    except Exception:
                        b_tensors = None
```

The worker loads base tensors for delta compression, **but skips Mega-Hash parents** — Mega-Hashes are metadata containers with no `.safetensors` blob. Attempting to load tensors from one would raise a `FileNotFoundError`.

```python
                blob_meta = fs_storage.save_tensors(comp_tensors, c_hash, base_tensors=b_tensors)
                c_data["blob_metadata"] = blob_meta

                fs_storage.save_commit(c_hash, c_data)
                fs_storage.write_ref(b_branch, c_hash)
                fs_storage.write_head(b_branch)
                logger.info(f"Async Save Finished [PID: {os.getpid()}]")
```

The worker performs all the heavy I/O:
1. Compute deltas, detect frozen layers, write `.safetensors`.
2. Write the commit JSON with blob metadata.
3. Update the branch ref to point to the new commit.
4. Update HEAD.

```python
            import multiprocessing
            p = multiprocessing.Process(
                target=_async_save_worker,
                args=(cpu_tensors, current_hash, base_hash, commit_data,
                      self._current_branch, self.storage)
            )
            p.start()
```

### Why `multiprocessing.Process` Instead of `threading.Thread`?

Python's **Global Interpreter Lock (GIL)** is a mutex that protects access to Python objects, preventing true parallel execution of Python bytecode. Even on a 64-core machine, only one Python thread can execute at a time.

`threading.Thread` shares the GIL with the main training thread. If the save worker is doing CPU-bound work (computing deltas, serializing tensors), it steals CPU time from PyTorch's C++ backend allocator, the autograd engine, and the dataloader's prefetcher — all of which release the GIL for C++ execution but reacquire it for Python-level bookkeeping.

`multiprocessing.Process` creates a **new OS process** with its own PID, its own Python interpreter, its own GIL, and its own address space. The parent and child processes are completely isolated — the child can saturate its CPU core for delta computation and disk I/O without affecting the parent's training throughput at all.

```python
            self._hash = current_hash
            self._commits[current_hash] = Commit.from_dict(commit_data)
            return current_hash
```

The parent process updates its in-memory state immediately and returns — **the GPU is unblocked in milliseconds**, while the background process may continue writing for seconds.

> **Key design note:** The background process is appended to `self._bg_processes`. Before `group_commits` is called (to write the Mega-Hash as the branch tip), all workers in this list are **joined** (`p.join(timeout=120)`). This ordering is critical: if workers were allowed to run past `group_commits`, they would overwrite the branch ref with the last sub-commit's hash, clobbering the Mega-Hash pointer.

---

## 6. The `load()` and Auto-Resume Path

### Explicit Loading

```python
    def load(self, hash: Optional[str] = None) -> Dict:
        ...
        commit_data = self.storage.load_commit(hash)

        # Mega-hash commits are UI grouping containers with no tensor blob.
        # Transparently resolve to the last real sub-commit.
        if commit_data.get("is_mega") and commit_data.get("sub_commits"):
            last_sub = commit_data["sub_commits"][-1]
            if self.storage.check_commit_exists(last_sub):
                commit_data = self.storage.load_commit(last_sub)
                hash = last_sub

        flat_tensors = self._fetch_tensors(hash)
        self._restore_commit_data(commit_data, flat_tensors)
```

`load()` transparently resolves Mega-Hash commits to their last real sub-commit. This means you can safely call `ckpt.load("mega_xxxx")` or `ckpt.goto("mega_xxxx")` — the actual weights are loaded from the final epoch checkpoint embedded inside.

`_fetch_tensors()` is a recursive delta resolver:

```python
    def _fetch_tensors(self, commit_hash: str) -> Dict[str, torch.Tensor]:
        commit_data = self.storage.load_commit(commit_hash)
        blob_metadata = commit_data.get("blob_metadata", {})
        blob_hash = blob_metadata.get("blob_hash", commit_hash)

        if blob_metadata.get("is_delta"):
            base_hash = commit_data.get("parent")
            # Guard against infinite loops
            if not base_hash:
                raise ValueError(...)
            if base_hash == commit_hash:
                raise ValueError(...)

            # Recurse! Walk the Merkle chain backwards
            base_tensors = self._fetch_tensors(base_hash)
            return self.storage.load_tensors(
                blob_hash, base_tensors=base_tensors,
                is_delta=True, frozen_links=blob_metadata.get("frozen_links", {})
            )
        else:
            return self.storage.load_tensors(blob_hash, is_delta=False)
```

This recursion walks backwards through the parent chain, loading and applying deltas at each step, until it reaches a non-delta (full snapshot) commit. The final result is the fully reconstructed tensor dictionary for the target commit.

### `_restore_commit_data` — Rebuilding the Training State

```python
    def _restore_commit_data(self, metadata, flat_tensors):
        self._step = metadata.get("step", 0)
        self._epoch = metadata.get("epoch", 0)
        self._batch_idx = metadata.get("batch_idx", 0)
        self._current_branch = metadata.get("branch", "main")
        self._config = HyperConfig.from_dict(metadata.get("config", {}))
```
Restore all scalar training metadata.

```python
        if "components_structure" in metadata:
            components_state = unflatten_state(metadata["components_structure"], flat_tensors)
            self.state_manager.restore_state(components_state)
```
Reconstruct the nested state dicts from the JSON structure + flat tensors, then call `.load_state_dict()` on each registered component.

```python
        if self.save_rng and "rng" in metadata and metadata["rng"]:
            rng_state = metadata["rng"]
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and isinstance(rng_state, list):
                rank = dist.get_rank()
                if rank < len(rng_state):
                    set_rng_state(rng_state[rank])
                else:
                    set_rng_state(rng_state[0])
            else:
                if isinstance(rng_state, list):
                    set_rng_state(rng_state[0])
                else:
                    set_rng_state(rng_state)
```
**DDP-aware RNG restoration.** The saved `rng` is a list `[rank0_state, rank1_state, ...]`. Each rank restores **its own** RNG state — Rank 2 restores `rng_state[2]`, not `rng_state[0]`### `__enter__` — Run Mode Dispatch

```python
    def __enter__(self) -> "CheckpointManager":
        self._lock_acquire()
        self._session_commits = []

        if self.run_mode == "overwrite":
            self.storage.delete_ref(self._current_branch)
            self._hash = self._generate_hash()
            self._session_start_hash = self._hash

        elif self.run_mode == "new_branch":
            existing = self.storage.read_ref(self._current_branch)
            if existing:
                try:
                    self.load(existing)  # warm-start weights from last run
                except Exception as e:
                    logger.warning(f"Could not pre-load state: {e}")
            # Reset counters — new_branch is a fresh run, not a continuation
            self._epoch = 0
            self._step  = 0
            self._batch_idx = 0
            new_branch_name = f"{base}_continue_{uuid.uuid4().hex[:4]}"
            self._current_branch = new_branch_name
            self.storage.write_head(new_branch_name)
            self._hash = self._generate_hash()
            self._session_start_hash = self._hash

        else:  # append
            if self.auto_resume:
                latest = self.storage.read_ref(self._current_branch)
                if latest and self.storage.check_commit_exists(latest):
                    try:
                        self.load(latest)
                    except Exception as e:
                        logger.warning(f"Failed to resume: {e}")
            if self._hash is None:
                self._hash = self._generate_hash()
            self._session_start_hash = self._hash

        return self
```

**`new_branch` always resets `_epoch`, `_step`, `_batch_idx` to 0** even if weights are loaded from a prior checkpoint. This is because `new_branch` is a *fresh experiment*, not a continuation — `ckpt.loop(epochs=50)` should always yield epochs 0–49 regardless of what the prior branch achieved.

### `__exit__` — Cleanup Without Implicit Save

```python
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.save(message=f"[FAILED] ❌ {exc_type.__name__}")
        except Exception as e:
            logger.warning(f"Failed to save on exit: {e}")
        finally:
            # group_commits is already called by loop()'s finally block.
            # Only invoke here if the user saved manually without ckpt.loop().
            if self._session_commits:
                self.group_commits(message="Training Loop Mega-Hash")
            # Join all pending async workers before printing so the tree is fully populated.
            for p in self._bg_processes:
                p.join(timeout=120)
            self._bg_processes.clear()
            self._lock_release()
            self.print_tree()
        return False
```

Key changes from earlier versions:
- **No implicit `self.save()` on clean exit** — `loop()`'s `finally` block already handles grouping. An extra save after a Mega-Hash was written would cause async worker crashes.
- **Workers are joined** before `print_tree()` so the tree is fully populated when displayed.

```python
        if "deterministic" in metadata:
            set_deterministic_state(metadata["deterministic"])
```
Restore `torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark` flags.

### Auto-Resume via Context Manager

```python
    def __enter__(self) -> "CheckpointManager":
        self._lock_acquire()

        if self.auto_resume:
            latest = self.storage.read_ref(self._current_branch)
            if latest and self.storage.check_commit_exists(latest):
                try:
                    self.load(latest)
                    logger.info(f"Resumed from step {self._step}, batch {self._batch_idx}")
                except Exception as e:
                    logger.warning(f"Failed to resume from {latest}: {e}")
            else:
                self._hash = self._generate_hash()
                logger.info(f"Initialized new branch {self._current_branch} with {self._hash}")
        else:
            self._hash = self._generate_hash()

        return self
```

When used as `with CheckpointManager(...) as ckpt:`, the `__enter__` method:
1. Acquires the file lock.
2. Checks the `run_mode` argument (`append` by default):
    - If `run_mode == "overwrite"`, it can optionally purge the active branch if `overwrite=True` or ask the user interactively.
    - If `run_mode == "new_branch"`, it creates a fresh branch before proceeding.
3. Checks if the current branch has any commits.
4. If yes, loads the latest checkpoint (restoring model, optimizer, PRNG states, step, epoch, batch_idx).
5. If no, initializes a fresh hash.

```python
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.save(message="[FAILED] \u274c")
                logger.error(f"Training loop failed. Saved failure state to {self._hash}")
            else:
                self.save()
            
            # Print the tree visualization at the end of the run
            self.print_tree()
        except Exception as e:
            logger.warning(f"Failed to save on exit: {e}")
        finally:
            self._lock_release()
        return False
```

On exit, auto-save the final state (or a `[FAILED] \u274c` state if an exception was raised), print the rich tree visualization of all branches, and release the lock. The `return False` ensures exceptions propagate normally.

---

## 7. Branching, Logging, and Diffing

### `create_branch`

```python
    def create_branch(self, name: str, message: str = "") -> str:
```

Creates a named experiment branch — like `git branch`. If the branch already exists, it simply switches to it (like `git checkout`). Otherwise, it creates a new commit with the current state, writes a new ref file, and updates HEAD.

### `checkout_branch`

```python
    def checkout_branch(self, name: str) -> bool:
```

Switches to an existing branch. Reads the branch ref, loads the tip commit, restores model/optimizer/PRNG states — like `git checkout`.

### `goto`

```python
    def goto(self, hash_or_branch: str) -> bool:
```

Flexible navigation: accepts either a branch name or a commit hash. First checks if it's a branch ref; if not, treats it as a direct commit hash — like `git checkout <hash>`.

### `log`

```python
    def log(self, n: int = 10) -> list:
```

Walks the parent chain backwards from the current commit, returning up to `n` `Commit` objects — like `git log -n 10`.

### `diff`

```python
    def diff(self, hash1: str, hash2: str) -> Dict:
```

Compares the `config` dicts of two commits and returns the differing keys — like a lightweight `git diff` for hyperparameters.

---

## 8. Exporting to Standard PyTorch Format

```python
    def export_ckpt(self, hash_or_branch: str, output_path: Union[str, Path]) -> None:
```

CAS stores data as fragmented `.safetensors` blobs and JSON commits — great for storage efficiency, but external tools (HuggingFace Hub, TorchServe, ONNX export) expect a single monolithic file.

```python
        target_hash = self.storage.read_ref(hash_or_branch) or hash_or_branch
        commit_data = self.storage.load_commit(target_hash)
        flat_tensors = self._fetch_tensors(target_hash)
        components = unflatten_state(commit_data.get("components_structure", {}), flat_tensors)
```

Resolve the target (branch name or hash), recursively resolve all deltas, and reconstruct the full nested state dict.

```python
        monolithic_state = {
            "hash": target_hash,
            "step": commit_data.get("step", 0),
            "epoch": commit_data.get("epoch", 0),
            "batch_idx": commit_data.get("batch_idx", 0),
            "branch": commit_data.get("branch", "main"),
            "config": commit_data.get("config", {}),
            "components": components,
            "rng": commit_data.get("rng"),
            "deterministic": commit_data.get("deterministic")
        }
        torch.save(monolithic_state, str(output_path))
```

Package everything into a single dict and save with `torch.save()` (which uses `pickle` internally). This produces a standard `.ckpt` file compatible with any PyTorch loading code.

---

## 9. Utilities: `step_up`, `loop`, Context Manager

### `step_up`

```python
    def step_up(self):
        self._step += 1
```
Increment the global step counter. Call this after each gradient update to keep the manager in sync with training progress.

### `loop`

```python
    def loop(self, epochs: int, steps_per_epoch: Optional[int] = None):
        start = self._epoch
        self._session_start_hash = self._hash
        self._session_commits = []
        try:
            for ep in range(start, epochs):
                self._epoch = ep
                if steps_per_epoch is None:
                    yield ep
                else:
                    for st in range(steps_per_epoch):
                        self._step = ep * steps_per_epoch + st
                        yield ep, st
        finally:
            # Join background workers BEFORE group_commits.
            # Workers write the branch ref to the last sub-commit hash.
            # We must let them finish first, then overwrite that ref with the Mega-Hash.
            for p in self._bg_processes:
                p.join(timeout=120)
            self._bg_processes.clear()
            if self._session_commits:
                self.group_commits(message=f"Loop Mega-Hash ({epochs} epochs)")
```

A generator that yields epoch numbers starting from `self._epoch` (which is 0 in `new_branch` mode, or the last saved epoch in `append` mode). The `finally` block runs when the `for epoch in ckpt.loop()` iteration completes (or if interrupted by an exception).

> **Why workers are joined here, not in `__exit__`:** The `finally` in `loop()` runs while we're still inside the `with` block, before `__exit__` fires. Joining here ensures all `.safetensors` and `.json` files are on disk before `group_commits` writes the Mega-Hash as the branch tip.

### `load_into_*` Methods

```python
    def load_into_model(self, model: nn.Module, hash: Optional[str] = None) -> None:
    def load_into_optimizer(self, optimizer, hash: Optional[str] = None) -> None:
    def load_into_scheduler(self, scheduler, hash: Optional[str] = None) -> None:
    def load_into_dataloader(self, dataloader, hash: Optional[str] = None) -> int:
    def load_into_config(self, hash: Optional[str] = None) -> HyperConfig:
```

These convenience methods load a specific commit and extract just one component's state. Useful when you want to load weights into a different model architecture or pick up only the optimizer state from a previous experiment.

### `create_checkpoint` Factory

```python
def create_checkpoint(dirpath: Union[str, Path], **kwargs) -> CheckpointManager:
    return CheckpointManager(dirpath=dirpath, **kwargs)
```

A module-level convenience factory function.

---

## 10. Mega-Hash Squashing

At the end of every `ckpt.loop()` call (or `__exit__` for manual saves), `group_commits()` is invoked to squash all session commits into a single **Mega-Hash** commit:

```python
def group_commits(self, message: str = "Mega-Hash"):
    if not self._session_commits or len(self._session_commits) <= 1:
        return

    mega_hash = f"mega_{uuid.uuid4().hex[:8]}"
    last_data = self._commits[self._session_commits[-1]]

    mega_commit_data = {
        "hash": mega_hash,
        "parent": self._session_start_hash,
        "is_mega": True,
        "sub_commits": self._session_commits,
        "message": message,
        "step": self._step,
        "epoch": self._epoch,
        "metric": last_data.metric,
        "components_structure": last_data.components_structure or {},
        "rng": last_data.rng,
        "deterministic": last_data.deterministic
    }

    self.storage.save_commit(mega_hash, mega_commit_data)
    self.storage.write_ref(self._current_branch, mega_hash)
    self.storage.write_head(self._current_branch)
```

### What is a Mega-Hash commit?

A Mega-Hash is a commit JSON file with `"is_mega": true` and a `"sub_commits"` list. It has **no corresponding `.safetensors` blob** — it is a pure metadata container.

When you call `ckpt.load("mega_xxxx")` or `ckpt.goto("mega_xxxx")`, `syckpt` transparently resolves to `sub_commits[-1]` (the last real sub-commit) and loads tensors from there.

### LSH prefix clustering and Mega-Hash identity

Each epoch's LSH hash is derived from your model architecture + hyperparameter config. Since these don't change between epochs of the same run, **all sub-commits share the same 8-character LSH prefix**:

```
9af15b33        ← epoch 0 (first save, no collision)
9af15b33-24c3f9 ← epoch 1 (collision-resolved)
9af15b33-a03e0c ← epoch 2 (collision-resolved)
```

This shared prefix is a natural fingerprint of the experiment. Any two runs with different hyperparameters (different learning rate, different model size) will produce hashes with **different prefixes**, making the tree immediately scannable without reading commit messages.

### `print_tree` rendering

Mega-Hash commits are rendered with their sub-commits nested inline:

```
--- Syckpt Tree ---
├── mega_9b2 (main_continue_3d32): [MEGA-HASH] 3 sub-commits | Loop Mega-Hash (3 epochs) [Epoch 2]
│   ├── 9af15b33: epoch-0 [Epoch 0]
│   ├── 9af15b33-24c3f9: epoch-1 [Epoch 1]
│   └── 9af15b33-e390dc: epoch-2 [Epoch 2]
└── mega_4ae (HEAD, *main_continue_64e2*): [MEGA-HASH] 3 sub-commits | Loop Mega-Hash (3 epochs) [Epoch 2]
    └── ...
```

Sub-commits are **excluded from top-level tree roots** (they don't appear as independent nodes), keeping the view clean regardless of how many epochs you train.

---

## 11. Future: Hierarchical Mega-Hashes

The current Mega-Hash system is **single-level**: one Mega-Hash wraps all epoch commits from one run. As training scales to foundational models spanning months, the true evolution is **Hierarchical Mega-Hashes** — nesting Mega-Hashes inside larger Mega-Hashes:

1. **Epoch Mega-Hashes**: Squash 10,000 granular gradient-step checkpoints into a single `Epoch-1` node.
2. **Phase Mega-Hashes**: Squash 100 `Epoch-X` epochs into a `Warmup-Phase` or `Cooldown-Phase` node.
3. **Experiment Mega-Hashes**: Combine multiple parallel phases into a single `Run-V2` root.

```
--- Syckpt Tree ---
└── mega_run_v2 (HEAD, *main*) [MEGA-HASH]
    ├── mega_warmup [MEGA-HASH]
    │   ├── mega_epoch_1 [MEGA-HASH]
    │   │   ├── 1a2b3c4d [Step 100]
    │   │   ├── ...
    ...
```

Instead of flat lists, Hierarchical Mega-Hashes will contain **DAGs of sub-mega-hashes**, providing infinite zoom-in/zoom-out capability for deep training histories without losing step-level rewindability.

This requires:
- An interactive tree visualization engine beyond terminal output.
- A lazy tensor-fetching strategy for recursive delta resolution across nested trees.

Targeted for `v2.0`.
