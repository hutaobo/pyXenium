from __future__ import annotations

from pyXenium._compat import deprecated_symbol

_PUBLIC_NAMES = {
    "DEFAULT_BRANCH_MODELS",
    "DEFAULT_MARKER_PAIRS",
    "DEFAULT_PATHWAY_MARKERS",
    "DEFAULT_RESISTANT_NICHES",
    "DEFAULT_STATE_HIERARCHY",
    "DEFAULT_STATE_SIGNATURES",
    "MarkerPair",
    "aggregate_multi_sample_study",
    "annotate_joint_cell_states",
    "build_spatial_niches",
    "compute_rna_protein_discordance",
    "score_immune_resistance_program",
}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal.immune_resistance",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal",
    )


__all__ = sorted(_PUBLIC_NAMES)
