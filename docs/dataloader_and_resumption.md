# Deep Dive: `syckpt/dataloader.py`

This document deconstructs `StatefulRandomSampler`, the mechanism allowing `syckpt` to definitively guarantee **Exact Mathematical Resumption** in $O(1)$ time complexity.

---

## 1. The Catastrophic Forgetting Problem

When standard PyTorch training scripts crash, engineers often reload the model weights and the optimizer state, and simply restart the script.

**The Problem:**
1. The standard `DataLoader` index resets to $0$. 
2. The PyTorch RNG seed generator often resets identically to when the script first launched originally.
3. The model structurally "re-sees" data from Epoch 1 that it has already trained on, completely altering the gradient trajectory.

Visually, this manifests as a massive, instantaneous "Spike" in the loss curve upon resuming a checkpoint, followed by the model fighting itself to re-descend the manifold. `syckpt` solves this by forcing the DataLoader to fast-forward into the precise microscopic state it died at.

---

## 2. Stateful Determinism

To force determinism, the script wrests control of the random layout entirely away from PyTorch's backend abstractions. Instead of wrapping the iterator sequentially, `syckpt` subclasses `torch.utils.data.Sampler` itself.

### Overriding Initialization
```python
class StatefulRandomSampler(Sampler):
    def __init__(self, data_source, batch_size: int, base_seed: int = 42, epoch: int = 0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.epoch = epoch
        self.batch_idx = 0
        self._generator = torch.Generator()
```
Instead of allowing the standard global context to dictate shuffling, `syckpt` spins up an entirely isolated `torch.Generator()` dedicated *solely* to this explicit Sampler instance.

### `__iter__` Setup
When `iter()` is called on the Sampler (e.g., at the start of a new epoch):
```python
def __iter__(self):
    self._generator.manual_seed(self.base_seed + self.epoch)
    
    n = len(self.data_source)
    # Generate the full deterministic epoch permutation exactly once!
    self._indices = torch.randperm(n, generator=self._generator).tolist()
```
1. It seeds the isolated generator dynamically based on the current `self.epoch`. This guarantees Epoch 1 and Epoch 5 uniformly generate vastly different arrays, but if Epoch 4 crashes, resuming Epoch 4 will re-seed its isolated generator with the identically correct integer.

2. By feeding the explicitly seeded `self._generator` directly into `randperm()`, it mathematically forces the exact same randomized shuffle array to occur deterministically upon crash resumption.

---

## 3. $O(1)$ List-Slicing Resumption Mathematics

Older generation checkpointing modules (and legacy versions of this codebase) relied on standard iterators, fast-forwarding them dynamically via a `next()` `while` loop:
```python
for _ in range(batch_idx * batch_size): 
    next(iterator) # Legacy O(N) Skip
```
If a crash occurred at Batch $300,000$ on a massive image dataset, this forced PyTorch to actively load $300,000$ batches (including CPU/disk bottlenecks) to throw them away, stalling the GPU for several minutes on startup.

`syckpt` bypasses this entirely using **$O(1)$ native memory slicing**:

```python
    # Fast-forward instantaneously in O(1) time
    items_to_skip = self.batch_idx * self.batch_size
    
    # Python list slicing resolves instantly via pointer offset
    yield from self._indices[items_to_skip:]
```

Because python list slices `[items_to_skip:]` are evaluated at the C-level pointer offset within a continuous block of memory, the sampler skips $300,000$ batches in nanoseconds exactly. The parent DataLoader simply grabs this sliced yield and immediately retrieves precisely the next mathematical integer representing the dataset.

This achieves mathematically provable Exact Resumption without scaling penalties for massive petabyte-era operations.
