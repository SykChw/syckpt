# Deep Dive: `syckpt/state.py` — PRNG Aggregation & The Seeds of Randomness

This document is a complete, line-by-line examination of `state.py` (203 lines). This module handles:

- **Pseudo-Random Number Generators (PRNGs)**: What they are, how they work mathematically, and why saving their state is critical for exact resumption.
- **Multi-backend state capture**: Independently freezing and restoring Python, NumPy, PyTorch CPU, PyTorch CUDA, and `torch.compile` RNG states.
- **The `StateManager` engine**: Dynamic duck-typed component registration and state serialization.

---

## Table of Contents

1. [The Illusion of Randomness: What Are PRNGs?](#1-the-illusion-of-randomness-what-are-prngs)
2. [The Four Independent RNG Backends](#2-the-four-independent-rng-backends)
3. [`get_rng_state` — Line-by-Line](#3-get_rng_state--line-by-line)
4. [`set_rng_state` — Line-by-Line](#4-set_rng_state--line-by-line)
5. [Deterministic State & Seed Control](#5-deterministic-state--seed-control)
6. [The `StateManager` Engine — Line-by-Line](#6-the-statemanager-engine--line-by-line)

---

## 1. The Illusion of Randomness: What Are PRNGs?

True randomness — like measuring radioactive decay or atmospheric noise — is computationally impossible on a deterministic silicon CPU. Every "random" number your program generates is actually produced by a **Pseudo-Random Number Generator (PRNG)**: a mathematical function that takes an initial starting point (the **seed**) and outputs a sequence of numbers that *appear* statistically random, but are actually 100% rigidly predetermined by the seed.

### The Linear Congruential Generator (LCG)

The simplest PRNG is the **Linear Congruential Generator**, whose entire state machine fits in one equation:

$$X_{n+1} = (a \cdot X_n + c) \mod m$$

Where:
- $m$ — The **modulus** (maximum value + 1). Determines the period of the sequence.
- $a$ — The **multiplier**. Carefully chosen to maximize sequence quality.
- $c$ — The **increment**. When $c = 0$, this becomes a "multiplicative congruential generator".
- $X_n$ — The **current state**. This single integer *is* the internal state of the PRNG.
- $X_{n+1}$ — The generated random number, which also becomes the new state.

**Example:** With $m = 16$, $a = 5$, $c = 3$, seed $X_0 = 7$:

| Step | Computation | $X_n$ |
|---|---|---|
| 0 | (seed) | 7 |
| 1 | $(5 \times 7 + 3) \mod 16$ | 6 |
| 2 | $(5 \times 6 + 3) \mod 16$ | 1 |
| 3 | $(5 \times 1 + 3) \mod 16$ | 8 |
| 4 | $(5 \times 8 + 3) \mod 16$ | 11 |

The key insight: **the sequence is a chain**. If you know $X_n$ (the state at step 100), you can perfectly predict every subsequent value. If a training script crashes at step 100 and you restart without restoring $X_{100}$, the PRNG resets to $X_0$. The sequence diverges completely, producing alien dropout masks, data augmentation transforms, and weight initializations.

### Modern PRNGs Used in Deep Learning

Real-world frameworks use far more sophisticated PRNGs:

| Backend | Algorithm | State Size | Period |
|---|---|---|---|
| **Python `random`** | Mersenne Twister (MT19937) | 624 × 32-bit words + index | $2^{19937} - 1$ |
| **NumPy `np.random`** | PCG64 (Permuted Congruential Generator) | 128-bit state + 128-bit increment | $2^{128}$ |
| **PyTorch CPU** | Mersenne Twister (modified) | ~5 KB byte tensor | $2^{19937} - 1$ |
| **PyTorch CUDA** | Philox4x32-10 (per-GPU) | 64-bit counter + 64-bit key | $2^{128}$ per GPU |

Notice that each backend uses a **different algorithm** with a **different internal state structure**. They do not share state, and consuming random numbers from one backend does not affect the others. To perfectly freeze a training script in time, you must independently capture **all four** (plus the `torch.compile` JIT RNG if applicable).

### The Mersenne Twister

Python's `random` module and PyTorch's CPU RNG both use the **Mersenne Twister** (MT19937), named after the Mersenne prime $2^{19937} - 1$ that defines its period.

The internal state is an array of 624 32-bit integers, plus an index indicating the current position in that array. At each step:
1. If the index reaches 624, the entire array is **twisted** (mixed) using a linear recurrence over $\text{GF}(2)$ (the Galois field with 2 elements).
2. The current element is **tempered** (bit-mixed) to improve statistical quality.
3. The index is incremented.

Saving this state means saving the 624-integer array plus the index — enough to perfectly resume the sequence.

### PCG64 (NumPy)

NumPy's modern default RNG uses **PCG64** (Permuted Congruential Generator), which is faster and statistically superior to the Mersenne Twister. Its state is just two 128-bit integers (state + increment), making it extremely compact to serialize.

### Philox (CUDA)

GPU random number generation uses the **Philox4x32-10** counter-based RNG. Unlike recursive PRNGs (where each state depends on the previous), Philox is a **stateless** function: given a counter value and a key, it produces a deterministic output. This makes it parallelizable across thousands of GPU threads — each thread uses a different counter offset.

The "state" to save is the current counter value and key for each GPU device.

---

## 2. The Four Independent RNG Backends

A typical PyTorch training script uses all four PRNG backends simultaneously:

```python
import random
import numpy as np
import torch

# Python random — used in:
#   - random.shuffle() for dataset splitting
#   - random.random() for probabilistic data augmentation
random.random()

# NumPy random — used in:
#   - np.random.choice() for class-balanced sampling
#   - np.random.normal() for noise injection
np.random.randn(10)

# PyTorch CPU — used in:
#   - torch.randn() for weight initialization
#   - dropout layers (sampling mask on CPU)
torch.randn(10)

# PyTorch CUDA — used in:
#   - torch.randn(..., device='cuda') for GPU operations
#   - cuDNN kernel selection (when benchmark=True)
#   - dropout masks computed on GPU
if torch.cuda.is_available():
    torch.randn(10, device='cuda')
```

These four RNG backends operate **completely independently**. Consuming random numbers from NumPy does not affect PyTorch's RNG, and vice versa. To achieve exact resumption, `syckpt` must capture and restore all four states independently.

---

## 3. `get_rng_state` — Line-by-Line

```python
def get_rng_state() -> Dict[str, Any]:
    state = {}
```

Initialize an empty dict to hold all RNG states.

```python
    state["torch_rng"] = torch.get_rng_state()
```

`torch.get_rng_state()` returns a `ByteTensor` containing the entire Mersenne Twister state (the 624 × 32-bit words array + metadata). This is approximately 5 KB of data.

```python
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
```

`torch.cuda.get_rng_state_all()` returns a **list** of `ByteTensor`s — one per GPU device. On a machine with 8 GPUs, this is a list of 8 byte tensors, each containing the Philox counter + key for that device. The `_all` suffix is critical: it captures **every GPU**, not just the current device.

```python
    state["numpy_rng"] = np.random.get_state()
```

`np.random.get_state()` returns a tuple: `('MT19937', array_of_624_uint32, position, has_gauss, cached_gauss)` for the legacy API, or the PCG64 bit_generator state for the new API. Either way, it contains everything needed to resume the NumPy random sequence.

```python
    state["python_rng"] = random.getstate()
```

`random.getstate()` returns a tuple: `(version, tuple_of_625_ints, gauss_next)`. The 625 integers are the Mersenne Twister internal state (624 words + index as a separate entry). The `gauss_next` is a cached value for the Box-Muller transform used by `random.gauss()`.

```python
    try:
        state["torch_compile_rng"] = torch._C._get_graph_execution_based_rng_state()
    except AttributeError:
        pass
```

PyTorch 2.0+ introduced `torch.compile()`, which JIT-compiles the computation graph. The compiled graph may have its own internal RNG state for operations like dropout. This captures that state if available, with a graceful fallback for older PyTorch versions.

```python
    return state
```

The returned dict contains **all five** potential RNG states, totaling approximately 10–20 KB of data. This is serialized into the commit's JSON metadata.

---

## 4. `set_rng_state` — Line-by-Line

This function restores the saved RNG states. The complexity comes from handling **JSON deserialization**: when the commit JSON was saved, tensors and tuples were converted to Python lists. On load, these lists must be converted back to the proper types.

```python
def set_rng_state(state: Dict[str, Any]) -> None:
```

```python
    if "torch_rng" in state:
        val = state["torch_rng"]
        if isinstance(val, list):
            val = torch.tensor(val, dtype=torch.uint8)
        torch.set_rng_state(val)
```

`torch.set_rng_state()` expects a `ByteTensor`. If the state was serialized through JSON (which converts tensors to lists via the `TensorEncoder` in `CASStorage._atomic_write_json()`), we must convert the list back to a `uint8` tensor.

```python
    if "cuda_rng" in state and torch.cuda.is_available():
        val = state["cuda_rng"]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            val = [torch.tensor(v, dtype=torch.uint8) for v in val]
        torch.cuda.set_rng_state_all(val)
```

For CUDA states, the value is a **list of lists** (one list per GPU), each needing conversion to a `ByteTensor`. `torch.cuda.set_rng_state_all()` restores the Philox counter + key for every GPU simultaneously.

```python
    if "numpy_rng" in state:
        val = state["numpy_rng"]
        if isinstance(val, list):
            val = (val[0], np.array(val[1], dtype=np.uint32), val[2], val[3], val[4])
        np.random.set_state(val)
```

NumPy's `set_state()` expects a tuple in the exact format `('MT19937', ndarray, position, has_gauss, cached_gauss)`. JSON serialization converts the tuple to a list and the ndarray to a nested list. This reconstructs the original format.

```python
    if "python_rng" in state:
        val = state["python_rng"]
        if isinstance(val, list):
            val = (val[0], tuple(val[1]), val[2]) if len(val) == 3 else tuple(val)
        random.setstate(val)
```

Python's `setstate()` expects a tuple `(version, tuple_of_ints, gauss_next)`. The inner element must be a tuple (not a list), hence the `tuple(val[1])` conversion.

```python
    if "torch_compile_rng" in state and hasattr(torch, "_C"):
        try:
            torch._C._set_graph_execution_based_rng_state(state["torch_compile_rng"])
        except AttributeError:
            pass
```

Restore `torch.compile` RNG state if available. The `try/except` handles version mismatches (e.g., saving on PyTorch 2.1, loading on 2.0).

---

## 5. Deterministic State & Seed Control

### `get_deterministic_state` / `set_deterministic_state`

```python
def get_deterministic_state() -> Dict[str, Any]:
    state = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    return state
```

cuDNN (NVIDIA's deep neural network library) has two relevant flags:

- **`deterministic`** — When `True`, cuDNN selects deterministic (but potentially slower) algorithms. When `False`, cuDNN may select non-deterministic algorithms that are faster but produce slightly different results on each run.
- **`benchmark`** — When `True`, cuDNN benchmarks multiple algorithms for each convolution input size and caches the fastest one. When `False`, cuDNN uses a default algorithm.

These flags affect reproducibility and must be saved/restored alongside model weights and RNG states.

```python
def set_deterministic_state(state: Dict[str, Any]) -> None:
    if "cudnn_deterministic" in state:
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    if "cudnn_benchmark" in state:
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
```

### `set_seed` — Universal Seeding

```python
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```

Sets the seed for **all five** RNG backends simultaneously:

1. `random.seed(seed)` — Python's Mersenne Twister.
2. `np.random.seed(seed)` — NumPy's legacy RNG.
3. `torch.manual_seed(seed)` — PyTorch CPU Mersenne Twister.
4. `torch.cuda.manual_seed(seed)` — Philox on the current CUDA device.
5. `torch.cuda.manual_seed_all(seed)` — Philox on **all** CUDA devices.

```python
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

When `deterministic=True`, force cuDNN into fully reproducible mode. This may slow down training by ~10% for convolution-heavy models but guarantees bit-for-bit reproducibility.

```python
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
```

`PYTHONHASHSEED` controls the randomization of Python's built-in `hash()` function. Since Python 3.3, hash randomization is enabled by default for security. Setting this environment variable makes `hash()` deterministic, which affects:
- `set()` iteration order.
- `dict()` iteration order (in some Python implementations).
- Any code that relies on `hash()` for bucketing.

---

## 6. The `StateManager` Engine — Line-by-Line

The `StateManager` is the component registry that the `CheckpointManager` delegates to for gathering and restoring state from user-registered objects (models, optimizers, schedulers, samplers, etc.).

### Class Definition

```python
class StateManager:
    __slots__ = ("_components", "_custom_handlers")

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._custom_handlers: Dict[str, Callable] = {}
```

- **`_components`** — Maps string names to registered objects: `{"model": <Linear>, "optimizer": <Adam>}`.
- **`_custom_handlers`** — Maps string names to custom functions for objects that don't follow standard serialization APIs.
- **`__slots__`** — Prevents dynamic attribute creation, reducing memory footprint.

### Registration

```python
    def register(self, **kwargs) -> None:
        self._components.update(kwargs)

    def unregister(self, *names: str) -> None:
        for name in names:
            self._components.pop(name, None)
```

Components are registered by keyword argument:
```python
state_manager.register(model=model, optimizer=optimizer)
# Internally: _components = {"model": model, "optimizer": optimizer}
```

This is what `CheckpointManager.__setattr__` delegates to when you write `ckpt.model = model`.

### `build_state` — Extracting State from All Components

```python
    def build_state(self) -> Dict[str, Any]:
        state = {}
        for name, obj in self._components.items():
            state[name] = self._get_state(obj, name)
        return state
```

Iterates over all registered components and extracts their serializable state.

### `_get_state` — Duck-Typed State Extraction

```python
    def _get_state(self, obj: Any, name: str) -> Any:
        if callable(getattr(obj, "state_dict", None)):
            return obj.state_dict()
```

**Priority 1:** If the object has a `state_dict()` method, call it. This handles:
- `torch.nn.Module` (models)
- `torch.optim.Optimizer` (optimizers)
- `torch.optim.lr_scheduler._LRScheduler` (schedulers)
- `StatefulRandomSampler` (syckpt's custom sampler)

```python
        elif callable(getattr(obj, "state", None)):
            return obj.state() if callable(obj.state) else obj.state
```

**Priority 2:** If the object has a `state` attribute or method, use it. This handles custom objects that follow a `state()`/`load_state()` API instead of PyTorch's `state_dict()`/`load_state_dict()`.

```python
        elif type(obj).__name__ == "Generator" and hasattr(obj, "bit_generator"):
            return obj.bit_generator.state
```

**Priority 3:** Special case for NumPy `Generator` objects. NumPy's new-style `np.random.Generator` wraps a `bit_generator` (like PCG64) whose state must be accessed through `.bit_generator.state`.

```python
        elif name in self._custom_handlers:
            return self._custom_handlers[name](obj)
```

**Priority 4:** If a custom handler was registered for this component name, call it. This allows users to handle arbitrary objects:

```python
state_manager.register_handler("my_custom_ds", lambda ds: {"offset": ds.current_offset})
```

```python
        else:
            logger.warning(f"Component '{name}' has no state_dict() method, skipping")
            return None
```

**Fallback:** If none of the above patterns match, log a warning and skip. The component is registered but its state won't be serialized.

### `restore_state` — Restoring State to All Components

```python
    def restore_state(self, state: Dict[str, Any]) -> None:
        for name, obj in self._components.items():
            if name in state:
                self._set_state(obj, state[name], name)
```

Iterates over all registered components and restores their state from the loaded dict.

### `_set_state` — Duck-Typed State Restoration

```python
    def _set_state(self, obj: Any, state: Any, name: str) -> None:
        if state is None:
            return
        if callable(getattr(obj, "load_state_dict", None)):
            obj.load_state_dict(state)
```

**Priority 1:** Calls `.load_state_dict()` (models, optimizers, schedulers, samplers).

```python
        elif callable(getattr(obj, "load_state", None)):
            obj.load_state(state)
```

**Priority 2:** Calls `.load_state()` for custom objects.

```python
        elif type(obj).__name__ == "Generator" and hasattr(obj, "bit_generator"):
            obj.bit_generator.state = state
```

**Priority 3:** Directly sets the NumPy Generator's bit_generator state.

```python
        elif name in self._custom_handlers:
            logger.warning(f"Custom handler for '{name}' cannot restore state")
        else:
            logger.warning(f"Component '{name}' has no load_state_dict() method, skipping")
```

Custom handlers are one-directional (extract only). Restoration for custom objects must be handled manually.

### The Complete State Lifecycle

```
Registration:     ckpt.model = model
                       ↓
                  StateManager.register(model=model)
                       ↓
Save:             StateManager.build_state()
                       ↓
                  _get_state(model, "model")
                       ↓
                  model.state_dict()
                       ↓
                  {"model": {"weight": tensor([...]), "bias": tensor([...])}}
                       ↓
                  flatten_state(...)  → flat tensors + JSON structure
                       ↓
                  CASStorage.save_tensors(...)  → .safetensors blob
                       ↓
                  CASStorage.save_commit(...)  → .json commit

Load:             CASStorage.load_commit(hash)  → JSON metadata
                       ↓
                  _fetch_tensors(hash)  → recursively resolve deltas
                       ↓
                  unflatten_state(structure, tensors)  → nested state dict
                       ↓
                  StateManager.restore_state(components)
                       ↓
                  _set_state(model, state["model"], "model")
                       ↓
                  model.load_state_dict(state["model"])
```
