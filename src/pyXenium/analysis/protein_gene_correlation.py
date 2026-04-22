from __future__ import annotations

from pyXenium._compat import deprecated_callable
from pyXenium.multimodal.analysis.protein_gene_correlation import (
    protein_gene_correlation as _protein_gene_correlation,
)


protein_gene_correlation = deprecated_callable(
    _protein_gene_correlation,
    old_path="pyXenium.analysis.protein_gene_correlation",
    new_path="pyXenium.multimodal.protein_gene_correlation",
)

__all__ = ["protein_gene_correlation"]
