# 1. Content-Addressable Storage (CAS) and Worktrees

Standard MLOps tools frequently suffer from disk-space exhaustion. A modern LLM training run might save a massive 5GB `.pt` PyTorch file at every single epoch. If you run 1,000 epochs, you have consumed 5 Terabytes of storage simply for intermediate states. 

`syckpt` bypasses this linear storage scaling by implementing a Git-native architecture under the hood: **Content-Addressable Storage (CAS)** mapped to hidden object directories.

## The Git Paradigm for Checkpoints

In traditional Git, whenever you commit a source code file, Git hashes the exact contents of that file (using SHA-1). It then saves the compressed contents into the hidden `.git/objects/` folder, using the hash as the filename.
If you commit the exact same file twice (perhaps across two different branches), Git does not duplicate the file on your hard drive. It looks up the hash, sees it already exists, and simply points the new commit metadata to the existing blob. 

### Enter `.syckpt/`

When you initialize `CheckpointManager(dirpath="s3://my-bucket/experiments")`, `syckpt` initializes a Git-like worktree at the target destination. 

1. **`.syckpt/objects/`**: This directory acts as the CAS backend. All tensors, regardless of which training run or epoch generated them, are pooled together here. Their filenames are derived deterministically using Locality-Sensitive Hashing based on their metadata and hyperparameters.
2. **`.syckpt/refs/heads/`**: This directory contains branch pointers. Instead of naming a file `model_epoch5_lr0.01.pt`, `syckpt` updates a branch reference (e.g., `main` or `run-alpha`) to point to the specific commit metadata hash. 
3. **Commit Metadata**: Inside the object store, `syckpt` saves lightweight JSON artifacts (functioning identically to Git Tree and Git Commit objects) that tie specific metrics, timestamps, and tensor blobs together securely. 

## Benefits of CAS in Deep Learning

*   **Atomic Cloud Operations:** Because `syckpt` integrates natively with `fsspec`, writing to `s3://` or `gcs://` is atomic. Half-written blobs are written to local temporary OS files first. Because filenames are hashes of their content (CAS), a corrupted overwrite is mathematically impossible. 
*   **Branching is Instant and Free:** If you want to test a new Learning Rate scheduling midway through an epoch, you don't need to copy a 5GB weights file to a new folder. `syckpt` simply creates a new 40-byte text pointer in `.syckpt/refs/heads/new_experiment` pointing to the same CAS tensor. 
*   **Deduplication:** Duplicated layers, optimizer momentums that haven't shifted, or identical dataset indices are automatically deduplicated. 

This hidden topology allows `syckpt` to provide infinite version control checkpoints without the storage overhead penalties normally associated with machine learning pipelines.
