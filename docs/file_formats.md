# Precision & File Formats

Syckpt inherently operates on the tensor level, isolating binary storage from PyTorch-specific metadata serialization.

## Numerical Formats

Syckpt is completely dtype-agnostic—it serializes pure bytes according to the tensor's underlying precision shape:

- **`fp64` (Double)**: Typically avoided in deep learning due to compute/memory footprint, but seamlessly tracked by Syckpt. Delta compression handles double precision perfectly.
- **`fp32` (Float)**: The standard precision mode.
- **`bf16` / `fp16` (Half)**: The standard for mixed-precision training. Syckpt preserves exact half-precision bit representations during delta computations without unintended upcasting to `fp32`.
- **`nvfp4` / `int8` (Quantized formats)**: Because Syckpt uses byte-level hashing (LSH) and direct binary I/O, quantized tensors, custom sparse tensors, or dynamically scaled integer types are natively supported as long as they provide a continuous memory view via `.numpy()`.

Because delta compression acts on exact shapes and dtypes, transferring a branch from `fp32` to `bf16` will correctly record the `bf16` cast as a full parameter shift, preventing silent precision drift during resumptions!

## Storage Container Formats

The underlying binary blobs are managed within a Content-Addressable Storage (CAS). 

### Supported Backends
1. **`.safetensors` (Default)**: 
   Syckpt natively uses HuggingFace's `safetensors.torch` to safely serialize and zero-copy read tensors. It stores raw blobs directly on the disk completely detached from Python `pickle`, avoiding security vulnerabilities.

2. **`.ckpt` (Pickle-based)**:
   Standard PyTorch format. Not recommended for Syckpt due to `pickle` vulnerabilities and the inability to lazily map individual tensor slices into memory.

3. **`.gguf` / `.ggml`**:
   While primarily designed for LLM unified deployment across C++ inferencers, you can natively implement a format converter or a custom encoder.

### Building Custom Formats

If you need to define a new numerical storage backend (e.g. distributed sharded Zarr or direct memory mapped custom files), you simply replace the `TensorEncoder` logic in `storage.py` and implement your own serialization protocol:

```python
# Customizing the Storage layer
class MyCustomStorage(CASStorage):
    def save_tensors(self, t_dict, hash_id, base_tensors=None):
        # Implement custom differential serialization here
        # e.g., storing to .gguf containers!
        pass

    def load_tensors(self, hash_id, is_delta=False):
        # Load from your custom container format
        pass
```
