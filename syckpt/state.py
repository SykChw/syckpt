"""State management utilities for checkpointing various components."""

import random
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """Manages the state dict collection from various components."""

    __slots__ = ("_components", "_custom_handlers")

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._custom_handlers: Dict[str, Callable] = {}

    def register(self, **kwargs) -> None:
        """Register components by name.

        Args:
            **kwargs: Named components to register (e.g., model=model, optimizer=optimizer)
        """
        self._components.update(kwargs)

    def unregister(self, *names: str) -> None:
        """Unregister components by name."""
        for name in names:
            self._components.pop(name, None)

    def clear(self) -> None:
        """Clear all registered components."""
        self._components.clear()

    def get(self, name: str) -> Optional[Any]:
        """Get a registered component by name."""
        return self._components.get(name)

    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())

    def register_handler(self, name: str, handler: Callable[[Any], Dict]) -> None:
        """Register a custom state handler for a component type.

        Args:
            name: Identifier for the handler
            handler: Function that takes an object and returns its state dict
        """
        self._custom_handlers[name] = handler

    def build_state(self) -> Dict[str, Any]:
        """Build complete state dict from all registered components."""
        state = {}

        for name, obj in self._components.items():
            state[name] = self._get_state(obj, name)

        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state to all registered components."""
        for name, obj in self._components.items():
            if name in state:
                self._set_state(obj, state[name], name)

    def _get_state(self, obj: Any, name: str) -> Any:
        """Get state from an object using appropriate method."""
        if callable(getattr(obj, "state_dict", None)):
            return obj.state_dict()
        elif callable(getattr(obj, "state", None)):
            return obj.state() if callable(obj.state) else obj.state
        elif type(obj).__name__ == "Generator" and hasattr(obj, "bit_generator"):
            return obj.bit_generator.state
        elif name in self._custom_handlers:
            return self._custom_handlers[name](obj)
        else:
            logger.warning(f"Component '{name}' has no state_dict() method, skipping")
            return None

    def _set_state(self, obj: Any, state: Any, name: str) -> None:
        """Set state to an object using appropriate method."""
        if state is None:
            return
        if callable(getattr(obj, "load_state_dict", None)):
            obj.load_state_dict(state)
        elif callable(getattr(obj, "load_state", None)):
            obj.load_state(state)
        elif type(obj).__name__ == "Generator" and hasattr(obj, "bit_generator"):
            obj.bit_generator.state = state
        elif name in self._custom_handlers:
            logger.warning(f"Custom handler for '{name}' cannot restore state")
        else:
            logger.warning(
                f"Component '{name}' has no load_state_dict() method, skipping"
            )


def get_rng_state() -> Dict[str, Any]:
    """Capture RNG state from all sources.

    Returns:
        Dict containing torch, cuda, numpy, and python random states
    """
    state = {}

    state["torch_rng"] = torch.get_rng_state()

    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()

    state["numpy_rng"] = np.random.get_state()

    state["python_rng"] = random.getstate()

    try:
        state["torch_compile_rng"] = torch._C._get_graph_execution_based_rng_state()
    except AttributeError:
        pass

    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from saved state dict.

    Args:
        state: Dict containing RNG states
    """
    if "torch_rng" in state:
        val = state["torch_rng"]
        if isinstance(val, list):
            val = torch.tensor(val, dtype=torch.uint8)
        torch.set_rng_state(val)

    if "cuda_rng" in state and torch.cuda.is_available():
        val = state["cuda_rng"]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            val = [torch.tensor(v, dtype=torch.uint8) for v in val]
        torch.cuda.set_rng_state_all(val)

    if "numpy_rng" in state:
        val = state["numpy_rng"]
        if isinstance(val, list):
            val = (val[0], np.array(val[1], dtype=np.uint32), val[2], val[3], val[4])
        np.random.set_state(val)

    if "python_rng" in state:
        val = state["python_rng"]
        if isinstance(val, list):
            # Python random state format: (version, tuple(internal_state), gauss_next)
            val = (val[0], tuple(val[1]), val[2]) if len(val) == 3 else tuple(val)
        random.setstate(val)

    if "torch_compile_rng" in state and hasattr(torch, "_C"):
        try:
            torch._C._set_graph_execution_based_rng_state(state["torch_compile_rng"])
        except AttributeError:
            pass


def get_deterministic_state() -> Dict[str, Any]:
    """Get deterministic settings for reproducibility."""
    state = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    return state


def set_deterministic_state(state: Dict[str, Any]) -> None:
    """Restore deterministic settings."""
    if "cudnn_deterministic" in state:
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    if "cudnn_benchmark" in state:
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seed for all RNG sources.

    Args:
        seed: The random seed
        deterministic: Whether to also set cudnn to deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
