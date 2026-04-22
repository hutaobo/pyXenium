from __future__ import annotations

from pyXenium._compat import deprecated_callable, deprecated_symbol
from pyXenium.multimodal import (
    plot_auc_heatmap as _plot_auc_heatmap,
    plot_DE_volcano as _plot_DE_volcano,
    plot_model_diagnostics as _plot_model_diagnostics,
    plot_topk_per_cluster as _plot_topk_per_cluster,
    protein_gene_correlation as _protein_gene_correlation,
    rna_protein_cluster_analysis as _rna_protein_cluster_analysis,
    write_model_scores as _write_model_scores,
)

from . import differential, plotting, scoring
from .ligand_receptor_topology import ligand_receptor_topology_analysis
from .pathway_topology import compute_pathway_activity_matrix, pathway_topology_analysis

protein_gene_correlation = deprecated_callable(
    _protein_gene_correlation,
    old_path="pyXenium.analysis.protein_gene_correlation",
    new_path="pyXenium.multimodal.protein_gene_correlation",
)
rna_protein_cluster_analysis = deprecated_callable(
    _rna_protein_cluster_analysis,
    old_path="pyXenium.analysis.rna_protein_cluster_analysis",
    new_path="pyXenium.multimodal.rna_protein_cluster_analysis",
)
write_model_scores = deprecated_callable(
    _write_model_scores,
    old_path="pyXenium.analysis.write_model_scores",
    new_path="pyXenium.multimodal.write_model_scores",
)
plot_auc_heatmap = deprecated_callable(
    _plot_auc_heatmap,
    old_path="pyXenium.analysis.plot_auc_heatmap",
    new_path="pyXenium.multimodal.plot_auc_heatmap",
)
plot_DE_volcano = deprecated_callable(
    _plot_DE_volcano,
    old_path="pyXenium.analysis.plot_DE_volcano",
    new_path="pyXenium.multimodal.plot_DE_volcano",
)
plot_model_diagnostics = deprecated_callable(
    _plot_model_diagnostics,
    old_path="pyXenium.analysis.plot_model_diagnostics",
    new_path="pyXenium.multimodal.plot_model_diagnostics",
)
plot_topk_per_cluster = deprecated_callable(
    _plot_topk_per_cluster,
    old_path="pyXenium.analysis.plot_topk_per_cluster",
    new_path="pyXenium.multimodal.plot_topk_per_cluster",
)

_DEPRECATED_PUBLIC_NAMES = {
    "DEFAULT_BRANCH_MODELS",
    "DEFAULT_MARKER_PAIRS",
    "DEFAULT_PATHWAY_MARKERS",
    "DEFAULT_RESISTANT_NICHES",
    "DEFAULT_STATE_HIERARCHY",
    "DEFAULT_STATE_SIGNATURES",
    "MarkerPair",
    "ProteinMicroEnv",
    "ProteinModelResult",
    "aggregate_multi_sample_study",
    "annotate_joint_cell_states",
    "build_spatial_niches",
    "compute_rna_protein_discordance",
    "score_immune_resistance_program",
}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal",
        public_names=_DEPRECATED_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal",
    )


__all__ = [
    "compute_pathway_activity_matrix",
    "differential",
    "ligand_receptor_topology_analysis",
    "pathway_topology_analysis",
    "plot_auc_heatmap",
    "plot_DE_volcano",
    "plot_model_diagnostics",
    "plot_topk_per_cluster",
    "plotting",
    "protein_gene_correlation",
    "rna_protein_cluster_analysis",
    "scoring",
    "write_model_scores",
]
