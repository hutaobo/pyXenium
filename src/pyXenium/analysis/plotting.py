from __future__ import annotations

from pyXenium._compat import deprecated_callable
from pyXenium.multimodal.analysis.plotting import (
    plot_auc_heatmap as _plot_auc_heatmap,
    plot_DE_volcano as _plot_DE_volcano,
    plot_model_diagnostics as _plot_model_diagnostics,
    plot_topk_per_cluster as _plot_topk_per_cluster,
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

__all__ = [
    "plot_auc_heatmap",
    "plot_DE_volcano",
    "plot_model_diagnostics",
    "plot_topk_per_cluster",
]
