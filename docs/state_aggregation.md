# Deep Dive: `syckpt/state.py`

This document deconstructs Pseudo-Random Number Generators (PRNGs), explaining what they are, how their deterministic mathematics functions, and how `syckpt` captures these moving parts collectively to guarantee exact system resumption.

---

## 1. The Illusion of Randomness

True randomness (like measuring radioactive decay) is computationally impossible on a deterministic silicon CPU. Therefore, programming languages utilize **Pseudo-Random Number Generators (PRNGs)**. 

A PRNG is a mathematical function that takes an initial starting point (the **Seed**), and outputs a sequence of numbers that *appear* statistically random, but are actually $100\%$ rigidly predetermined by the seed itself.

### The Mathematics: Linear Congruential Generators
While modern systems use vastly more complex models (like the Mersenne Twister), the foundational math of a PRNG can be understood via the classical LCG equation:

$$X_{n+1} = (a \cdot X_n + c) \mod m$$

Where:
*   **$m$**: The modulus (max number limit)
*   **$a$**: The multiplier
*   **$c$**: The increment
*   **$X_n$**: The *Current Mathematical State* 
*   **$X_{n+1}$**: The drawn random number (which mathematically becomes the new State).

**Notice the crucial detail:** The sequence is a chain. If you know $X_n$ (the State at step 100), you can instantly predict step 101 perfectly. If a PyTorch training loop crashes at step 100, and you reload the model but *fail* to reload $X_{100}$, PyTorch's $n$ value resets to $0$. The model will then draw totally alien "random" dropout masks and noise vectors compared to its original trajectory, destroying the loss curve seamlessly.

---

## 2. Capturing the Beast: Multi-Layer Aggregation

To perfectly freeze a system in time, we must capture every single mathematical state variable ($X_n$) currently governing the script. 

A Deep Learning script uses four structurally distinct PRNG architectures that do not communicate with each other:

1.  **Python (`random`)**: Often used in simple augmentations or dataset splitting. (Mersenne Twister logic).
2.  **NumPy (`np.random`)**: Highly optimized C++ array randomization. (PCG64 architecture).
3.  **PyTorch CPU (`torch.manual_seed`)**: Native tensors tracking operations on system RAM.
4.  **PyTorch GPU (`torch.cuda`)**: Massively parallel grid-generation routines operating explicitly on physical GPU VRAM.

### `get_rng_state()`
```python
def get_rng_state() -> Dict[str, Any]:
    state = {}
    state["torch_rng"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    state["numpy_rng"] = np.random.get_state()
    state["python_rng"] = random.getstate()
    return state
```
This function traverses the OS and forces every standalone backend library to halt and dump its internal $X_n$ mathematical arrays (which are often vast binary matrices, not simple integers like the LCG example). The `get_rng_state_all()` command explicitly aggregates the states natively across every single physical GPU device strapped to the system bus.

### `set_rng_state()`
When `syckpt` resumes, it recursively fires these exact massive arrays backward into the C++ bindings of each library natively:
```python
def set_rng_state(state: Dict[str, Any]) -> None:
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    ...
```
This brutally overrides whatever the OS initialized, explicitly dialing the temporal variables of the mathematical LCG sequences definitively back to the exact millisecond the checkpoint was captured.

---

## 3. The `StateManager` Engine

With the core PRNG states captured globally, `syckpt` still must handle user-defined objects programmatically. 

### Dynamic Duct Tape
The `StateManager` acts as a dynamic dictionary proxy.
```python
def __init__(self):
    self._components: Dict[str, Any] = {}
```

When iterating over registered classes (like the Model or Optimizer), it attempts to organically locate standard PyTorch serialization routines rather than enforcing inheritance:
```python
def _get_state(self, obj: Any, name: str) -> Any:
    if callable(getattr(obj, "state_dict", None)):
        return obj.state_dict()
    elif callable(getattr(obj, "state", None)):
        return obj.state() if callable(obj.state) else obj.state
```

The resulting dictionary is then mathematically flattened by `storage.py` and converted securely to Safetensors, maintaining the hyper-strict state integrity globally.
