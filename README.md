# Syckpt v0.0.1

**Git-like experiment tracking for deep learning with exact computational resumption, zero-copy safetensors memory-mapping, and delta-compression.**

`syckpt` is a lightweight, local-first experiment version control system designed to perfectly reconstruct massive computational states—model weights, optimizer momentum, mixed-precision GradScalers, Random Number Generators, and Stateful DataLoaders—without perturbing the loss curve.

---

## How `syckpt` Works (The Architecture)

When training massive Deep Learning models, saving a full checkpoint at every epoch typically results in gigabytes of duplicated disk space and high latency. `syckpt` solves this by treating machine learning checkpoints like a Git repository.

1. **Content-Addressable Storage (CAS) & Delta-Compression**:
   Instead of saving full 5GB `.pt` weight files at every step, `syckpt` finds the most mathematically similar historical checkpoint and computes the pure `float32` difference (`delta = current - base`). Because gradient steps are small, this delta is highly compressible. `syckpt` stores these deltas in a hidden `.syckpt/objects/` directory, saving up to 90% of disk space.

2. **Locality-Sensitive Hashing (LSH)**:
   To instantly find the "most similar" historical checkpoint, `syckpt` uses LSH to hash your hyperparameters (like learning rate, batch size, and seed). Similar hyperparameters mathematically collide to produce identical hash prefixes, allowing the system to rapidly query the Git-tree.

3. **Zero-Copy memory mapping via Safetensors**:
   `syckpt` bypasses Python's insecure and memory-heavy `pickle` module. It uses Rust-backed `safetensors` to memory-map the delta-blobs directly from your SSD into the GPU's VRAM ("Zero-Copy"), completely eliminating CPU RAM Out-Of-Memory (OOM) errors during loading.

4. **Exact Mathematical Resumption**:
   Standard PyTorch training loops suffer from "resumption spikes" in the loss curve because the DataLoader indices and Random Number Generators (RNG) get reset. `syckpt` intercepts PyTorch, CUDA, and Numpy generators, as well as preserving the internal `RandomSampler` permutations of your DataLoaders. When you resume, it is mathematically identical to if the process was never interrupted.

## Installation

We utilize the Rust-accelerated `uv` package manager.

```bash
pip install syckpt
# Or using uv 
uv pip install syckpt
```

## Quick Start

```python
import torch
import torch.nn as nn
import torch.optim as optim
from syckpt import CheckpointManager
from syckpt.dataloader import StatefulDataLoader
from torch.utils.data import DataLoader, TensorDataset

# Typical PyTorch components
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)

dummy_data = TensorDataset(torch.randn(100, 10), torch.randn(100, 2))
# Wrap standard non-deterministic DataLoader internally
loader = StatefulDataLoader(DataLoader(dummy_data, batch_size=32, shuffle=True)) 

# Specify an S3 or Local URL: Atomic locks handles concurrency natively
with CheckpointManager("s3://my-experiments-bucket/.syckpt") as ckpt:
    # 1. Register dynamic objects (Automatically mapped via flattening)
    ckpt.model = model
    ckpt.optimizer = optimizer
    ckpt.dataloader = loader
    
    # 2. Hyperparameters automatically generate the unique LSH Hash
    ckpt.config.lr = 0.01
    ckpt.config.batch_size = 32
    
    # 3. Training Loop inherently traps the step and epoch parameters 
    for epoch in ckpt.loop(epochs=10):
        for batch_idx, batch in enumerate(loader):
            loss = torch.randn(1) # Fake loss
            ckpt.step_up()
            
        # Delta-Compression kicks in automatically
        if epoch % 2 == 0:
            ckpt.save(metric=loss.item())
            
    print(f"Mathematical execution saved at LSH Commit: {ckpt.hash}")
```

## Feature Reference

### Exporting Monolithic Assets (`.ckpt`)
If you deploy your model and no longer need `.syckpt` branching, you can securely collapse the Git-tree into a standard monolithic PyTorch `.ckpt` file for Hugging Face or deployment:
```python
with CheckpointManager("./experiments") as ckpt:
    # Recursively loads flat delta-tensors and reconstitutes standard dict
    ckpt.export_ckpt(hash_or_branch="main", output_path="final-model.ckpt")
```

### Full Distributed Resumption (DDP)
`syckpt` seamlessly broadcasts LSH hashes and uses `dist.gather` to collect highly volatile RNG seeds across your entire multi-GPU cluster.
```python
import numpy as np

with CheckpointManager("./") as ckpt:
    # Simply register your Modern Numpy generator and the state_manager intercepts the memory bytes
    ckpt.numpy_rng = np.random.default_rng() 
    ckpt.save()
```

---

## Architectural Deep-Dive
Curious how `syckpt v0.0.1` leverages Git pointers, `fsspec` atomic cloud mechanisms, manages PyTorch tensors, and accelerates training via Zero-Copy Safetensors? 

Read the definitive educational walkthrough: [Implementation Guide (`implementation.md`)](./implementation.md).
