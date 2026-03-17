# 2. Delta Compression Mechanics

While Content-Addressable Storage (see previous section) handles deduplication of *identical* files perfectly, Machine Learning weights are rarely identical.

During training, a 5GB weight tensor changes *slightly* at every training step. A traditional cryptographic hash function over the tensor would completely change, forcing the system to store a brand new 5GB blob every single step, defeating the purpose of CAS.

## The Mathematical Base Difference

To achieve Git-like efficiency for shifting tensors, `syckpt` applies **Delta Compression**. 

When a checkpoint is invoked, `syckpt` identifies the most mathematically similar structure (usually the direct parent commit `step T-1`).
Instead of serializing the new matrices, it intercepts the `float32` arrays in memory and calculates an element-wise mathematical subtraction across the flattened parameter dicts:

```python
def compute_delta(current: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for key, current_tensor in current.items():
        if key in base and current_tensor.shape == base[key].shape:
            # Generate pure sparse diff
            delta[key] = current_tensor - base[key]
        else:
            delta[key] = current_tensor
    return delta
```

## Why Delta Tensors are Highly Compressible

In Stochastic Gradient Descent (SGD) and Adam, training steps represent infinitesimally small gradient updates multiplied by a small learning rate.
Therefore, `current - base` results in a matrix entirely populated by numbers clustered extremely close to `0.000000` (e.g. `1e-6`).

When these highly localized arrays of zeros are passed through generic compression algorithms or safetensors compression libraries natively, the entropy drops drastically. The result is a mathematically exact representation of the epoch shift that takes megabytes, rather than gigabytes.

## Exact Reconstitution on Load

When a user calls `ckpt.load("main")` or initializes a new model with auto-resume, `syckpt` must do the inverse:

1. Identify the parent commit.
2. Load the original base tensor into RAM.
3. Load the compressed Delta Tensor into RAM.
4. Mathematically apply the delta patch: `current_tensor = base_tensor + delta_tensor`.

Because it relies on straightforward floating-point Tensor arithmetic, it executes instantly on the CPU/GPU, eliminating the storage overhead of redundant data points across consecutive check-ins while perfectly reconstructing the exact nested PyTorch states.
