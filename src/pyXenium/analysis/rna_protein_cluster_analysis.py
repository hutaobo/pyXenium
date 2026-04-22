from __future__ import annotations

from pyXenium._compat import deprecated_callable, deprecated_symbol
from pyXenium.multimodal.analysis.rna_protein_cluster_analysis import (
    rna_protein_cluster_analysis as _rna_protein_cluster_analysis,
)

_PUBLIC_NAMES = {"ProteinModelResult"}

rna_protein_cluster_analysis = deprecated_callable(
    _rna_protein_cluster_analysis,
    old_path="pyXenium.analysis.rna_protein_cluster_analysis",
    new_path="pyXenium.multimodal.rna_protein_cluster_analysis",
)


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal.analysis.rna_protein_cluster_analysis",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal",
    )


__all__ = ["ProteinModelResult", "rna_protein_cluster_analysis"]
