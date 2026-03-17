# Deep Dive: `syckpt/manager.py` — Orchestration, Async Saves & DDP Synchronization

This document is a complete, line-by-line examination of `manager.py` (855 lines), the core orchestration engine of `syckpt`. It covers:

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
        "config", "metric", "timestamp", "blob_hash", "blob_metadata"
    )
```

A lightweight data class representing a saved checkpoint — analogous to a Git commit. Uses `__slots__` for memory efficiency (prevents creation of a per-instance `__dict__`).

Key fields:
- **`hash`** — The LSH-derived commit identifier.
- **`parent`** — The hash of the previous commit (forms the Merkle DAG chain).
- **`blob_hash`** — Points to the `.safetensors` file in `.syckpt/objects/`.
- **`blob_metadata`** — Dict with `is_delta` (bool) and `frozen_links` (dict of frozen layer keys).

The `to_dict()` / `from_dict()` methods serialize to/from JSON for storage.

---

## 4. The `CheckpointManager` Class

### Initialization

```python
class CheckpointManager:
    __slots__ = (
        "root", "storage", "max_to_keep", "maximize", "auto_resume",
        "save_rng", "_lock", "_locked", "state_manager", "_config",
        "_step", "_epoch", "_batch_idx", "_hash", "_current_branch", "_commits",
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
        self._hash: Optional[str] = None
        self._current_branch = self.storage.read_head()
        self._commits: Dict[str, Commit] = {}
```
- `_hash` — the hash of the currently active commit (or `None` if uninitialized).
- `_current_branch` — read from `.syckpt/HEAD` on startup.
- `_commits` — in-memory cache of loaded commits to avoid repeated disk reads.

```python
        branch_hash = self.storage.read_ref(self._current_branch)
        if branch_hash:
            self._hash = branch_hash
            self._load_commit_into_cache(branch_hash)
```
On startup, check if the current branch has any commits. If so, cache the tip commit.

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

            if base_hash and current_hash == base_hash:
                import uuid
                current_hash = f"{current_hash}-{uuid.uuid4().hex[:6]}"
```

LSH is designed to produce **identical hashes** for similar configs — that's its purpose. But if the same config is used across consecutive checkpoints (same model, same optimizer, same hyperparams), the LSH hash will be identical to the parent's hash. This would overwrite the parent's `.safetensors` and `.json` files, destroying the base needed for delta resolution.

The UUID suffix ensures uniqueness while preserving the LSH locality property in the prefix.

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
                    b_tensors = fs_storage.load_tensors(b_hash, is_delta=False)
```

The worker loads the base tensors from the parent commit's `.safetensors` file. It loads them with `is_delta=False` because in this context, it needs the raw tensor data from the file — the delta resolution for the base itself was already handled when the base was originally saved.

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

---

## 6. The `load()` and Auto-Resume Path

### Explicit Loading

```python
    def load(self, hash: Optional[str] = None) -> Dict:
        self._lock_acquire()
        try:
            if hash is None:
                hash = self._hash
            if not hash:
                raise ValueError("No hash provided and manager is uninitialized.")
            if not self.storage.check_commit_exists(hash):
                raise ValueError(f"Commit not found: {hash}")
```
Load a specific commit by hash. If no hash is given, reload the current commit.

```python
            commit_data = self.storage.load_commit(hash)
            flat_tensors = self._fetch_tensors(hash)
            self._restore_commit_data(commit_data, flat_tensors)
```

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
**DDP-aware RNG restoration.** The saved `rng` is a list `[rank0_state, rank1_state, ...]`. Each rank restores **its own** RNG state — Rank 2 restores `rng_state[2]`, not `rng_state[0]`. This ensures each GPU's dropout masks, data augmentation, etc., resume from exactly where they left off.

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
2. Checks if the current branch has any commits.
3. If yes, loads the latest checkpoint (restoring model, optimizer, PRNG states, step, epoch, batch_idx).
4. If no, initializes a fresh hash.

```python
    def __exit__(self, *args):
        try:
            self.save()
        except Exception as e:
            logger.warning(f"Failed to save on exit: {e}")
        finally:
            self._lock_release()
        return False
```

On exit, auto-save the final state and release the lock. The `return False` ensures exceptions propagate normally.

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
        for ep in range(start, epochs):
            self._epoch = ep
            if steps_per_epoch is None:
                yield ep
            else:
                for st in range(steps_per_epoch):
                    self._step = ep * steps_per_epoch + st
                    yield ep, st
            if self.auto_resume:
                self.save()
```

A generator that yields epoch numbers (or `(epoch, step)` tuples) starting from the last saved epoch. If `auto_resume=True`, it auto-saves at the end of each epoch. This means re-running the script after a crash will resume from the correct epoch automatically.

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
