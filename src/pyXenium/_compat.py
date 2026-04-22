from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any, Callable


def deprecated_symbol(
    name: str,
    *,
    target_module: str,
    public_names: set[str],
    old_namespace: str,
    new_namespace: str,
) -> Any:
    if name not in public_names:
        raise AttributeError(f"module {old_namespace!r} has no attribute {name!r}")

    warnings.warn(
        f"`{old_namespace}.{name}` is deprecated; use `{new_namespace}.{name}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    module = import_module(target_module)
    return getattr(module, name)


def deprecated_callable(
    func: Callable[..., Any],
    *,
    old_path: str,
    new_path: str,
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"`{old_path}` is deprecated; use `{new_path}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__name__ = getattr(func, "__name__", "wrapper")
    wrapper.__doc__ = getattr(func, "__doc__", None)
    wrapper.__module__ = getattr(func, "__module__", __name__)
    return wrapper
