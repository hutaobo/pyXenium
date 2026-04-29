from __future__ import annotations

from pyXenium._compat import deprecated_symbol

_PUBLIC_NAMES = {"cci_topology_analysis"}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.cci",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.cci",
    )


__all__ = ["cci_topology_analysis"]
