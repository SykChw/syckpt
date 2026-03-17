# Syckpt Internal Documentation

Welcome to the internal documentation suite for `syckpt`. 

Each underlying python file has been deeply analyzed and mathematically deconstructed to serve as both an architectural reference and an educational resource explaining the concepts powering the package (Content Addressable Storage, Delta Compression, PRNG Sequences, Distributed Training Barriers, and Hash Hyperplanes).

Please explore the following monolithic guides depending on your focus:

## Component Deep-Dives

1. **[`syckpt/storage.py`: Content Addressable Storage & Delta Compression](./storage_and_cas.md)**
   * Educational deep dive into Content-Addressable Storage (CAS), Git Work-Trees, and Safetensors flat-dictionary requirements.
   * Explores the mathematics of Stochastic Gradient Descent (SGD) and *why* mathematically delta-compression operates so efficiently on weight arrays.
   * Full method breakdown for binary serialization, atomic writes via `fsspec`, and tree reconstruction.

2. **[`syckpt/manager.py`: Orchestration & PyTorch DDP Synchronization](./manager_and_ddp.md)**
   * Explains how Distributed Data Parallel (DDP) functions across multiple distinct OS environments.
   * Deconstructs exactly how `dist.barrier()` and `broadcast_object_list()` are leveraged to prevent severe node-write corruption during checkpointing.
   * Line breakdown of context bounding `save()`, `load()`, and `export_ckpt()`.

3. **[`syckpt/config.py` & `syckpt/hash.py`: Configuration & LSH Bucketing](./config_and_lsh.md)**
   * Explores the mathematical geometry of Locality-Sensitive Hashing (LSH) random hyperplanes and proofs of cosine-distance mapping.
   * Analyzes Distance-Sensitive continuous quantization algorithms ensuring similar hyperparameter configs logically collide.
   * Breakdown of the `HyperConfig` dot-notation dictionary proxy mapping.

4. **[`syckpt/dataloader.py`: Slicing Iterators & Fast-Forwarding](./dataloader_and_resumption.md)**
   * Explores how catastrophic forgetting and loss-curve spiking occurs when DataLoaders are not deterministically preserved across crashes.
   * Analyzes the exact mechanics (and current loop-based inefficiency limitations) of fast-forwarding PyTorch `_iterator` object states dynamically.

5. **[`syckpt/state.py`: PRNG Aggregation & The Seeds Of Randomness](./state_aggregation.md)**
   * Educational breakdown of Pseudo-Random Number Generators (PRNGs), Linear Congruential Generators, and how temporal seeds function mathematically.
   * Exploration of how CUDA, NumPy, Python, and PyTorch PRNGs structurally differ across physical arrays and CPU bounds.
   * Deconstructs `StateManager`, exploring component routing and dynamic property assignments.
