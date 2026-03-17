# Syckpt

**Git-like experiment tracking for deep learning with exact computational resumption, zero-copy safetensors memory-mapping, and delta-compression.**

`syckpt` is a lightweight, local-first experiment version control system designed to perfectly reconstruct massive computational states—model weights, optimizer momentum, mixed-precision GradScalers, Random Number Generators, and Stateful DataLoaders—without perturbing the loss curve.

---

## How `syckpt` Works (The Architecture)

`syckpt` solves checkpoint bloat by treating machine learning parameters exactly like source code in a Git repository.

1. **Content-Addressable Storage (CAS) & Delta-Compression**: Computes purely elemental arrays of `delta = current - base` to save disk space over epochs.
2. **Sub-Layer Freezing**: Identifies immutable layers (e.g. frozen backbones) and uses virtual hard-links to achieve zero-cost storage.
3. **Locality-Sensitive Hashing (LSH)**: Hyperparameters hash into unique prefixes. Mathematically similar training runs intentionally collide into the identical hash bucket to instantly query the Git tree.
4. **Asynchronous Multiprocessing**: Offloads I/O and delta math to background systems to prevent GPU stalling and bypass the GIL.
5. **Exact Mathematical Resumption**: Restores RNG bit-states specifically across PyTorch, CUDA, Numpy, and isolated `StatefulDataLoaders` with $O(1)$ efficiency.

---

## Installation

```bash
pip install syckpt
```

---

## Workflows and Example Usages

### 1. Integrating into an Existing Training Flow

If you have a massive Vision Transformer (ViT) looping over a billion-image dataset on S3, slotting in `syckpt` takes 4 lines.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from syckpt import CheckpointManager
from syckpt.dataloader import StatefulRandomSampler
from torch.utils.data import DataLoader, TensorDataset

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)
dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 2))

# Use StatefulRandomSampler for O(1) resumption
sampler = StatefulRandomSampler(dataset, batch_size=32)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

with CheckpointManager("./.syckpt") as ckpt:
    ckpt.model = model
    ckpt.optimizer = optimizer
    ckpt.sampler = sampler
    
    for epoch in ckpt.loop(epochs=10):
        for batch_idx, (x, y) in enumerate(loader):
            # Training happens here...
            ckpt.step_up()
            
        ckpt.save()
```

### 2. Hyperparameter Search Efficiency

Evaluating thousands of configurations linearly is deeply inefficient. Because `syckpt` utilizes **Locality-Sensitive Hashing** on the `config` object, running an efficient grid search is incredibly fast.

```python
def evaluate_run(learning_rate: float):
    with CheckpointManager("./ml_logs/.syckpt") as ckpt:
        ckpt.model = ResNet50()
        ckpt.config.lr = learning_rate
        
        # Resumes directly from similar runs if hashes collide in LSH space
        for epoch in ckpt.loop(epochs=50):
            train_one_epoch()
            ckpt.save()
```

### 3. Distributed Resumption (PyTorch DDP)
`syckpt` seamlessly integrates with massive GPU topologies, employing `dist.barrier()` natively to guarantee exact write structures.

### 4. Exporting Monolithic Assets (`.ckpt`)
When the checkpoint has achieved optimal convergence, collapse the internal Git-Tree branches into a single flattened standard portable format.

```python
with CheckpointManager("./experiments") as ckpt:
    ckpt.export_ckpt(hash_or_branch="main", output_path="final-model.ckpt")
```

---

## Architectural Deep-Dive

Curious how `syckpt` leverages Git pointers, `fsspec` atomic cloud mechanisms, manages PyTorch tensors, and accelerates training via Zero-Copy Safetensors? 

Read our comprehensive documentation reports across the `docs/` suite:

*   **[Implementation Architecture Summary](docs/implementation.md)**
*   **[Storage, CAS & Delta Compression](docs/storage_and_cas.md)**
*   **[Checkpoint Manager & DDP Synchronization](docs/manager_and_ddp.md)**
*   **[Configuration & LSH Bucketings](docs/config_and_lsh.md)**
*   **[DataLoader & O(1) Resumption](docs/dataloader_and_resumption.md)**
*   **[State Aggregation & RNG Determinism](docs/state_aggregation.md)**
