from __future__ import annotations

from pyXenium._compat import deprecated_callable
from pyXenium.multimodal.analysis.scoring import write_model_scores as _write_model_scores


write_model_scores = deprecated_callable(
    _write_model_scores,
    old_path="pyXenium.analysis.write_model_scores",
    new_path="pyXenium.multimodal.write_model_scores",
)

__all__ = ["write_model_scores"]
