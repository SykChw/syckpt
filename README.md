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

## Competitive Analysis & The MLOps Ecosystem

The fundamental checkpointing challenge in deep learning is managing the exorbitant cost of physically storing 5GB to 100GB matrices repeatedly over epoch intervals. Below is a comparative analysis detailing where `syckpt` positions itself in the broader storage paradigm.

### 1. The HuggingFace Hub (Standard Git+LFS)
HuggingFace utilizes standard `git`. It tracks massive models using `git-lfs` pointers mapping to AWS S3.
*   **How it Works:** Treats every tensor completely transparently as a massive opaque binary file. It uses Git Object models, but utilizes SHA-256 for chunking.
*   **The Difference:** If an entire 10GB model is passed down a gradient trajectory, literally every floating-point variable inside the `.safetensors` file mutates slightly. Git LFS fails to compress this entirely, viewing it as a brand new 10GB block, and forces massive continuous upload speeds. 
*   **Where `syckpt` Wins:** `syckpt` operates inside PyTorch directly before serialization, ripping the tensor strings out natively and executing element-wise subtractive patch $\Delta W$ mathematics. This shrinks the required bandwidth artificially to ~5%. `syckpt` focuses on the high-speed volatile development phase rather than the finalized storage of the end-product. 

### 2. Weights & Biases (W&B Artifacts)
W&B functions as a primary experiment tracking dashboard. It allows users to log massive model directories.
*   **How it Works:** Executes standard SHA-1 deduplication. If `model.bin` exists inside Artifact $v0$, and you upload an identical `model.bin` inside Artifact $v1$, it simply references the initial string pointer without actively uploading the duplicate bytes.
*   **The Difference:** Again, tracking training trajectories inherently modifies *all* bytes simultaneously. Thus W&B cannot deduplicate training runs gracefully locally unless the layers are explicitly frozen.
*   **Where `syckpt` Wins:** `syckpt` utilizes LSH (Locality-Sensitive Hashing). Even if the user initiates a brand new run, tweaking the learning rate slightly, `syckpt` detects the hyperparameter collision automatically at execution time across local caches and dynamically forces the engine to Delta-Compress against the closest related experiment branch, saving massive local disk thresholds.

### 3. MosaicML Composer
Composer focuses entirely on training optimization, and introduces a robust `ObjectStore` backend natively handling S3 streaming checkpoint loading dynamically to prevent node disk-fill limits.
*   **How it Works:** Periodically packages classical DDP model shards dynamically across multiple cloud buckets asynchronously in the background.
*   **The Difference & Where `syckpt` Wins:** Composer utilizes brute-force cloud network speeds to circumvent local disk requirements (essentially throwing money and I/O at the problem). `syckpt` leverages CPU mathematical efficiency to minimize the objects entirely *before* network execution is even required via mathematical tensor geometry. 

### Potential Future Avenues for `syckpt`
*   **Sub-Layer Freezing:** Instead of Delta-Compressing the entire 10GB graph automatically, `syckpt` could explicitly evaluate which PyTorch layers mutated (e.g. tracking `requires_grad=False`), forcing instant hard-linking for un-mutated blocks similar to standard Docker layering.
*   **Sub-classing Samplers:** The current `dataloader.py` method executes highly inefficient `next()` skipping to achieve exact mathematical resumption. `syckpt` should override `torch.utils.data.Sampler` natively in future releases to execute slice indexing over arrays identically to `[current_batch:]`. 
*   **Asynchronous Storage Threads:** Moving the Delta-Compression operations (which are heavily CPU and memory bound) directly into an asynchronous background thread during the user loop to unblock GPU execution immediately.
