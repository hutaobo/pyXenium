from __future__ import annotations

from pyXenium._compat import deprecated_callable
from pyXenium.multimodal.loading import load_rna_protein_anndata


load_xenium_gene_protein = deprecated_callable(
    load_rna_protein_anndata,
    old_path="pyXenium.io.load_xenium_gene_protein",
    new_path="pyXenium.multimodal.load_rna_protein_anndata",
)

load_xenium_gene_protein.__doc__ = """
Deprecated compatibility wrapper for :func:`pyXenium.multimodal.load_rna_protein_anndata`.
"""

__all__ = ["load_xenium_gene_protein"]
