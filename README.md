# Syckpt

**Git-like experiment tracking for deep learning with exact computational resumption, zero-copy safetensors memory-mapping, and delta-compression.**

`syckpt` is a lightweight, local-first experiment version control system designed to perfectly reconstruct massive computational states—model weights, optimizer momentum, mixed-precision GradScalers, Random Number Generators, and Stateful DataLoaders—without perturbing the loss curve.

---

## How `syckpt` Works (The Architecture)

`syckpt` solves checkpoint bloat by treating machine learning parameters exactly like source code in a Git repository.

1. **Content-Addressable Storage (CAS) & Delta-Compression**: Computes purely elemental arrays of `delta = current - base` to save disk space over epochs.
2. **Locality-Sensitive Hashing (LSH)**: Hyperparameters hash into unique prefixes. Mathematically similar training runs intentionally collide into the identical hash bucket to instantly query the Git tree.
3. **Zero-Copy memory mapping via Safetensors**: Bypasses `pickle` insecurity and natively relies on `mmap` inside Rust to move files from SSD directly into GPU VRAM. 
4. **Exact Mathematical Resumption**: Restores RNG bit-states specifically across PyTorch, CUDA, Numpy, and isolated `StatefulDataLoaders`. 

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
from syckpt.dataloader import StatefulDataLoader
from torch.utils.data import DataLoader, TensorDataset

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loader = StatefulDataLoader(DataLoader(TensorDataset(torch.randn(100, 10)), batch_size=32)) 

# 1. Initialize to a local path or cloud directory natively (via fsspec)
with CheckpointManager("s3://experiments/.syckpt") as ckpt:
    
    # 2. Register variables and hyperparameters dynamically
    ckpt.model = model
    ckpt.optimizer = optimizer
    ckpt.dataloader = loader
    ckpt.config.update({"lr": 0.01, "batch_size": 32})
    
    # 3. The context manager loops with exact resumption guarantees
    for epoch in ckpt.loop(epochs=10):
        for batch_idx, batch in enumerate(loader):
            loss = torch.randn(1)
            ckpt.step_up()
            
        # 4. Save deltas without OOM lockups
        ckpt.save(metric=loss.item())
```

### 2. Hyperparameter Search Efficiency

Evaluating thousands of configurations linearly is deeply inefficient. Because `syckpt` utilizes **Locality-Sensitive Hashing** on the `config` object, running an efficient grid search is incredibly fast.

When you transition from a broad learning rate search to a highly precise grid, `syckpt` automatically computes the hash, recognizes it has already solved the broad baseline natively inside the Object Store, and resumes directly from the cached branch without re-evaluating baseline iterations.

```python
def evaluate_run(learning_rate: float, branch: str):
    with CheckpointManager("./ml_logs/.syckpt") as ckpt:
        ckpt.model = ResNet50()
        
        # Hyperparameters dictate the branch mapping natively 
        ckpt.config.lr = learning_rate
        ckpt.config.optimizer = "adamw"
        
        # If the LSH hash is mathematically identical to a previous run,
        # ckpt.save() will pull the float blocks via Delta Compression!
        for epoch in ckpt.loop(epochs=50):
            train_loss()
            ckpt.save()

# Broad Search (Epoch 0 -> 25)
evaluate_run(learning_rate=0.01, branch="broad_search")

# Precise Grid Search inherently resumes and compresses deltas 
# against the broad baseline because their quantized ranges overlap!
for lr in [0.009, 0.011, 0.012]:
    evaluate_run(learning_rate=lr, branch=f"precise_{lr}")
```

### 3. Distributed Resumption (PyTorch DDP)
`syckpt` seamlessly integrates with massive GPU topologies, employing `dist.barrier()` natively to guarantee exact write structures.

```python
import numpy as np

with CheckpointManager("./") as ckpt:
    # Captures global torch and cuda limits. Intercept custom frameworks natively!
    ckpt.numpy_rng = np.random.default_rng() 
    ckpt.save()
```

### 4. Exporting Monolithic Assets (`.ckpt`)
When the checkpoint has achieved optimal convergence, collapse the internal Git-Tree branches into a single flattened standard portable format.

```python
with CheckpointManager("./experiments") as ckpt:
    # Builds delta-safetensors, un-flattens arrays, and constructs a .ckpt payload
    ckpt.export_ckpt(hash_or_branch="main", output_path="final-model.ckpt")
```

---

## Architectural Deep-Dive

Curious how `syckpt v0.0.1` leverages Git pointers, `fsspec` atomic cloud mechanisms, manages PyTorch tensors, and accelerates training via Zero-Copy Safetensors? 

Read our comprehensive documentation reports across the `docs/` suite:

*   **[Implementation Architecture Summary](docs/implementation.md)**
*   [01. Content-Addressable Storage (CAS) and Worktrees](docs/01_overview_and_cas.md)
*   [02. Delta Compression Mechanics](docs/02_delta_compression.md)
*   [03. Safetensors and the Recursive Flattening Algorithm](docs/03_safetensors_and_flattening.md)
*   [04. Locality-Sensitive Hashing (LSH) for Hyperparameters](docs/04_lsh_and_hyperparameters.md)
*   [05. Distributed Data Parallel (DDP) Mechanics](docs/05_ddp_synchronization.md)
*   [06. Exact Mathematical Resumption and DataLoaders](docs/06_resumption_dataloader.md)

---

## Advanced Concepts & Future Avenues

We maintain deeply pedagogical documentation regarding advanced topics (like asynchronous threading, mathematical sub-layer freezing, and exact sampler resumption hooks) for developers looking to optimize or contribute to `syckpt`. 

*   **[Read the Advanced Concepts & Future Pipelines Guide](docs/advanced_concepts.md)**
