# Deep Dive: `syckpt/manager.py` & Asynchronous DDP

This document explores `manager.py`, the core orchestration engine of `syckpt`. 
We will deconstruct how PyTorch Distributed Data Parallel (DDP) training fundamentally operates under the hood, and explicitly walk through how the `CheckpointManager` class leverages message passing and **Asynchronous Multiprocessing** to guarantee atomic, unified, zero-stall saves across massive clusters.

---

## 1. What is DDP and How Does it Work?

When scaling deep learning models, training on a single GPU becomes infeasible. PyTorch handles this via **Distributed Data Parallel (DDP)**.

### The Physics of DDP
Unlike threading (which shares memory), DDP spawns entirely separate **Python OS processes**, often running on different physical machines across the network.
*   **Rank 0 (Main/Master):** The primary process. Usually handles logging, evaluation, and disk I/O.
*   **Ranks 1..N:** Worker processes acting in total isolation. 

Each GPU holds a fully physically distinct copy of the model weights in its VRAM. 

### The Synchronization Problem
If every GPU calculates gradients ($\nabla L$) independently on different slices of the data, their model weights will drift apart after a single step. 

DDP solves this by hijacking the backward pass (`loss.backward()`). As gradients execute, PyTorch initiates an **All-Reduce** operation over the NVLink/NCCL network. It mathematically averages the gradients across all GPUs simultaneously, ensuring that when the optimizer steps, every single isolated GPU mathematically arrives at the identical weight matrix.

**The Danger of Checkpointing:**
Because every GPU has the identical weights, we only want **Rank 0** to save the checkpoint to disk. If all 8 GPUs try to write `weights.safetensors` to the same physical file simultaneously, you trigger an OS race condition resulting in irrecoverable binary corruption.

---

## 2. `CheckpointManager`: Asynchronous Safe Orchestration

The `CheckpointManager` acts as the firewall preventing this corruption. It intercepts the user's `save()` command, forces network synchronization, assigns tasks strictly to Rank 0, broadcasts results back, and most importantly, natively forks an isolated background process to prevent GPU stalls.

### The `save()` Mechanism: A Line-by-Line Network Dance

Let's dissect the core save routine.

#### Step 1: The Network Barrier
```python
if dist.is_available() and dist.is_initialized():
    dist.barrier()
    is_main = dist.get_rank() == 0
    world_size = dist.get_world_size()
    is_dist = True
```
If `syckpt.save()` is called at slightly different times by different GPUs, the system will crash. `dist.barrier()` forces all GPUs to halt execution until every single GPU reaches this exact line of code across the entire cluster.

#### Step 2: The Hash Broadcast
```python
current_hash = self._generate_hash() if is_main else ""

if is_dist:
    hash_list = [current_hash]
    dist.broadcast_object_list(hash_list, src=0)
    current_hash = hash_list[0]
```
Rather than letting each GPU generate slightly disparate LSH strings, `syckpt` forces the Main process to generate the authoritative tracking string. It then uses `dist.broadcast_object_list()` to serialize that string and beam it over the network to Ranks 1-N.

#### Step 3: RNG State Gathering
If you want to resume DDP perfectly, you must save the CPU/CUDA random seeds of *every* GPU independently.
```python
if is_dist and self.save_rng:
    if is_main:
        gathered_rngs = [None for _ in range(world_size)]
        dist.gather_object(rng_state, gathered_rngs, dst=0)
        rng_state = gathered_rngs
    else:
        dist.gather_object(rng_state, dst=0)
```
`dist.gather_object()` forces all remote nodes to package their local PRNG matrices and beam them directly into Rank 0's RAM (`gathered_rngs`), acting as a unified collection array.

#### Step 4: Worker Exile
```python
if not is_main:
    self._hash = current_hash
    return current_hash
```
All GPUs that are not Rank 0 are immediately kicked out of the function. They natively proceed to execute the next forward pass on the GPU without waiting for disk IO!

#### Step 5: Master Asynchronous Forking
Rank 0 must compute heavy CPU mathematical deltas ($\Delta W = W_t - W_{t-1}$) and serialize massive byte streams to the SSD. If performed natively, the Main wrapper GPU will stall, forcing Workers 1-N to lock up waiting on the next batch's collective `All-Reduce`.

```python
# Unblock the GPU by shifting arrays identically into isolated CPU RAM
cpu_tensors = {k: v.to("cpu", non_blocking=True).clone() for k, v in flat_tensors.items()}

import multiprocessing
p = multiprocessing.Process(
    target=_async_save_worker,
    args=(cpu_tensors, current_hash, ... , self.storage)
)
p.start()

# GPU instantly returns to the training loop unblocked
return current_hash
```
`threading.Thread` in Python is famously deadly because the **Global Interpreter Lock (GIL)** artificially bottlenecks execution, stealing CPU cycles from PyTorch's backend allocator.

Instead, `syckpt` invokes an OS-level `multiprocessing.Process`. By cloning the tensors `to("cpu")` actively, the system cleanly severs the memory bridge graph. A dedicated detached Linux sub-system PID spins up in isolation, allocating its own RAM, performing the heavy IO math delta extraction, and dumping to Safetensors, completely independent from the core PyTorch training loop.

---

## 3. Core File Abstractions

Outside of DDP orchestration, `manager.py` bridges standard Machine Learning concepts into Git interfaces.

### `__setattr__` Proxy and State Tracking
To feel "magical", `syckpt` registers attributes natively:
```python
manager = CheckpointManager("./mnt")
manager.model = model
manager.optimizer = optimizer
```
It intercepts this Python capability dynamically:
```python
def __setattr__(self, name: str, value: Any):
    if name.startswith("_") or name in (... core properties):
        object.__setattr__(self, name, value)
    else:
        self.state_manager.register(**{name: value})
```
Any component attached to the manager is routed straight into the `StateManager` architecture automatically.

### Monolithic Exporting (`export_ckpt`)
```python
def export_ckpt(self, hash_or_branch: str, output_path: Union[str, Path]) -> None:
```
Because Content-Addressable Storage (CAS) is highly fractured, external pipelines (like HuggingFace Hub) cannot natively use it. This pipeline dynamically builds the entire nested node layout utilizing `unflatten_state` alongside raw `torch.save()`, dropping a legacy `.ckpt` pickle file for deployment endpoints.
