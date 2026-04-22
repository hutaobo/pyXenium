from __future__ import annotations

import warnings
from importlib import import_module

_TARGET_MODULE = "pyXenium.multimodal.analysis.tabnet_tools"


def __getattr__(name: str):
    module = import_module(_TARGET_MODULE)
    if not hasattr(module, name):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"`{__name__}.{name}` is deprecated; use `{_TARGET_MODULE}.{name}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(module, name)


def __dir__() -> list[str]:
    module = import_module(_TARGET_MODULE)
    return sorted(set(globals()) | {name for name in dir(module) if not name.startswith("_")})


__all__: list[str] = []
