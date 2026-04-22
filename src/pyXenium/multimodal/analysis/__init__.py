from __future__ import annotations

from importlib import import_module

from .plotting import plot_auc_heatmap, plot_DE_volcano, plot_model_diagnostics, plot_topk_per_cluster
from .protein_gene_correlation import protein_gene_correlation
from .protein_microenvironment import ProteinMicroEnv
from .rna_protein_cluster_analysis import ProteinModelResult, rna_protein_cluster_analysis
from .scoring import write_model_scores

_MODULE_EXPORTS = {
    "differential",
    "microenv_analysis",
    "plotting",
    "scoring",
    "tabnet_model",
    "tabnet_pipeline",
    "tabnet_reports",
    "tabnet_tools",
}


def __getattr__(name: str):
    if name in _MODULE_EXPORTS:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MODULE_EXPORTS)

__all__ = [
    "ProteinMicroEnv",
    "ProteinModelResult",
    "differential",
    "microenv_analysis",
    "plot_auc_heatmap",
    "plot_DE_volcano",
    "plot_model_diagnostics",
    "plot_topk_per_cluster",
    "plotting",
    "protein_gene_correlation",
    "rna_protein_cluster_analysis",
    "scoring",
    "tabnet_model",
    "tabnet_pipeline",
    "tabnet_reports",
    "tabnet_tools",
    "write_model_scores",
]
