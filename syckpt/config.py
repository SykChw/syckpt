"""Configuration system with attribute and dict access."""

import copy
from typing import Any, Dict, Optional, Union
from collections.abc import Mapping


class HyperConfig(Mapping):
    """A configuration object that supports both attribute and dict access.

    Supports nested configuration via dot notation (e.g., config.a.b.c)
    and provides full dict-like access (config['key'], config.get('key')).
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs):
        self._data: Dict[str, Any] = {}
        if data:
            self._data = self._flatten_dict(data) if isinstance(data, dict) else {}
        self._data.update(kwargs)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dict into dot-notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _unflatten_dict(self, d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
        """Unflatten dot-notation keys back to nested dict."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d_obj = result
            for part in parts[:-1]:
                if part not in d_obj:
                    d_obj[part] = {}
                d_obj = d_obj[part]
            d_obj[parts[-1]] = value
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        unflattened = self._unflatten_dict(self._data)
        if name in unflattened:
            val = unflattened[name]
            if isinstance(val, dict) and all(
                isinstance(k, str) and not any("." in k for k in v.keys())
                if isinstance(v, dict)
                else True
                for k, v in val.items()
                if isinstance(v, dict)
            ):
                return HyperConfig(val)
            return val
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, dict):
                for k, v in self._flatten_dict({name: value}).items():
                    self._data[k] = v
            else:
                self._data[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self._data:
            del self._data[name]
        unflattened = self._unflatten_dict(self._data)
        if name in unflattened:
            del unflattened[name]
            self._data = self._flatten_dict(unflattened)
            return
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in self._flatten_dict({key: value}).items():
                self._data[k] = v
        else:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._data

    def __iter__(self):
        return iter(self._unflatten_dict(self._data))

    def __len__(self) -> int:
        return len(self._unflatten_dict(self._data))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._unflatten_dict(self._data)})"

    def __str__(self) -> str:
        import json

        return json.dumps(self._unflatten_dict(self._data), indent=2)

    def __bool__(self) -> bool:
        return bool(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(
        self, other: Union[Dict[str, Any], "HyperConfig"], **kwargs
    ) -> "HyperConfig":
        """Update config with new values."""
        if other:
            if isinstance(other, HyperConfig):
                self._data.update(other._data)
            elif isinstance(other, dict):
                self._data.update(self._flatten_dict(other))
        self._data.update(self._flatten_dict(kwargs))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return unflattened dict representation."""
        return self._unflatten_dict(self._data)

    def copy(self) -> "HyperConfig":
        """Return a shallow copy."""
        return HyperConfig(copy.copy(self._data))

    def deep_copy(self) -> "HyperConfig":
        """Return a deep copy."""
        return HyperConfig(copy.deepcopy(self._data))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperConfig":
        """Create config from dict."""
        return cls(data)

    def items(self):
        """Return unflattened items."""
        return self._unflatten_dict(self._data).items()

    def keys(self):
        """Return unflattened keys."""
        return self._unflatten_dict(self._data).keys()

    def values(self):
        """Return unflattened values."""
        return self._unflatten_dict(self._data).values()
