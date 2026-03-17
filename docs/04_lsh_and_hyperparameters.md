# 4. Locality-Sensitive Hashing (LSH) for Hyperparameters

If `syckpt` performs Delta-Compression by finding a parent tensor to subtract against, how does the system find the optimal parent efficiently across hundreds of complex experiments and active trials?

Instead of relying purely on timestamp chronologies (which fail across disconnected computing nodes), `syckpt` utilizes **Locality-Sensitive Hashing (LSH)** on the experiment configuration.

## What is LSH?

Standard cryptographic hash functions (like SHA-256) are structurally designed to maximize the Avalanche Effect: changing a single digit (a Learning Rate from `0.009` to `0.010`) results in a completely unrecognizable output string.

LSH reverses this paradigm. It is designed to purposely *maximize* collisions for inputs that are mathematically similar to each other.

### The Quantization Mechanism
In `syckpt`, we bucket continuous values into fixed regions.

When learning rates, beta values, or batch sizes are submitted into `CheckpointManager()`, the deterministic hider engine quantizes them:
1. Continuous data streams are binned: a learning rate of `0.009` and `0.015` both quantize identically into a rounded logarithmic bucket around `0.01`.
2. Categorical items (Architecture types) maintain hard string boundaries.

### Branch Generation

When `ckpt.save()` triggers, the LSH algorithm evaluates the quantized values, projects them against deterministic random hyperplanes, and evaluates which bounded zone the configuration falls into exactly.
Run configurations grouped on the precise same side of multiple boundary metrics output the **exact same string hash prefix**.

If you launch an experiment with learning rate `0.01` and another with `0.012`, their resulting hashes map to the identical bucket. 
`syckpt` understands instantaneously this new configuration is attempting to explore identical loss landscapes, enabling it to retrieve the `0.01` checkpoint immediately as the base reference structure for mathematical Delta Compression operations.

This generates a local cache mapping entirely driven by metric similarity, not arbitrary user-titled save folders, maximizing deduplication automatically.
