from __future__ import annotations

from pyXenium._compat import deprecated_symbol

_PUBLIC_NAMES = {"ligand_receptor_topology_analysis"}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.ligand_receptor",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.ligand_receptor",
    )


__all__ = ["ligand_receptor_topology_analysis"]
