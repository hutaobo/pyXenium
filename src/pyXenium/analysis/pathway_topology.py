from __future__ import annotations

from pyXenium._compat import deprecated_symbol

_PUBLIC_NAMES = {"compute_pathway_activity_matrix", "pathway_topology_analysis"}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.pathway",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.pathway",
    )


__all__ = ["compute_pathway_activity_matrix", "pathway_topology_analysis"]
