from __future__ import annotations

from pyXenium._compat import deprecated_callable, deprecated_symbol
from pyXenium.multimodal import (
    build_cohort_handoff_spec as _build_cohort_handoff_spec,
    build_panel_gap_table as _build_panel_gap_table,
    build_serializable_pilot_summary as _build_serializable_pilot_summary,
    build_summary as _build_summary,
    build_top_hypotheses_table as _build_top_hypotheses_table,
    extract_ranked_patches as _extract_ranked_patches,
    render_markdown_report as _render_markdown_report,
    render_renal_immune_resistance_report as _render_renal_immune_resistance_report,
    run_renal_immune_resistance_pilot as _run_renal_immune_resistance_pilot,
    run_validated_renal_ffpe_smoke as _run_validated_renal_ffpe_smoke,
    validate_summary as _validate_summary,
    write_output_artifacts as _write_output_artifacts,
    write_renal_immune_resistance_artifacts as _write_renal_immune_resistance_artifacts,
)

from .atera_wta_breast_topology import (
    DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS,
    DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
    DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID,
    DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR,
    DEFAULT_LR_SMOKE_PANEL,
    DEFAULT_PATHWAY_PANEL,
    build_serializable_breast_topology_summary,
    render_atera_wta_breast_topology_report,
    run_atera_wta_breast_topology,
    write_atera_wta_breast_topology_artifacts,
)

build_summary = deprecated_callable(
    _build_summary,
    old_path="pyXenium.validation.build_summary",
    new_path="pyXenium.multimodal.workflows.build_summary",
)
validate_summary = deprecated_callable(
    _validate_summary,
    old_path="pyXenium.validation.validate_summary",
    new_path="pyXenium.multimodal.workflows.validate_summary",
)
render_markdown_report = deprecated_callable(
    _render_markdown_report,
    old_path="pyXenium.validation.render_markdown_report",
    new_path="pyXenium.multimodal.workflows.render_markdown_report",
)
write_output_artifacts = deprecated_callable(
    _write_output_artifacts,
    old_path="pyXenium.validation.write_output_artifacts",
    new_path="pyXenium.multimodal.workflows.write_output_artifacts",
)
run_validated_renal_ffpe_smoke = deprecated_callable(
    _run_validated_renal_ffpe_smoke,
    old_path="pyXenium.validation.run_validated_renal_ffpe_smoke",
    new_path="pyXenium.multimodal.run_validated_renal_ffpe_smoke",
)
build_cohort_handoff_spec = deprecated_callable(
    _build_cohort_handoff_spec,
    old_path="pyXenium.validation.build_cohort_handoff_spec",
    new_path="pyXenium.multimodal.workflows.build_cohort_handoff_spec",
)
build_panel_gap_table = deprecated_callable(
    _build_panel_gap_table,
    old_path="pyXenium.validation.build_panel_gap_table",
    new_path="pyXenium.multimodal.workflows.build_panel_gap_table",
)
build_serializable_pilot_summary = deprecated_callable(
    _build_serializable_pilot_summary,
    old_path="pyXenium.validation.build_serializable_pilot_summary",
    new_path="pyXenium.multimodal.workflows.build_serializable_pilot_summary",
)
build_top_hypotheses_table = deprecated_callable(
    _build_top_hypotheses_table,
    old_path="pyXenium.validation.build_top_hypotheses_table",
    new_path="pyXenium.multimodal.workflows.build_top_hypotheses_table",
)
extract_ranked_patches = deprecated_callable(
    _extract_ranked_patches,
    old_path="pyXenium.validation.extract_ranked_patches",
    new_path="pyXenium.multimodal.workflows.extract_ranked_patches",
)
render_renal_immune_resistance_report = deprecated_callable(
    _render_renal_immune_resistance_report,
    old_path="pyXenium.validation.render_renal_immune_resistance_report",
    new_path="pyXenium.multimodal.workflows.render_renal_immune_resistance_report",
)
run_renal_immune_resistance_pilot = deprecated_callable(
    _run_renal_immune_resistance_pilot,
    old_path="pyXenium.validation.run_renal_immune_resistance_pilot",
    new_path="pyXenium.multimodal.run_renal_immune_resistance_pilot",
)
write_renal_immune_resistance_artifacts = deprecated_callable(
    _write_renal_immune_resistance_artifacts,
    old_path="pyXenium.validation.write_renal_immune_resistance_artifacts",
    new_path="pyXenium.multimodal.workflows.write_renal_immune_resistance_artifacts",
)

_DEPRECATED_PUBLIC_NAMES = {
    "DEFAULT_DATASET_PATH",
    "EXPECTED_CELLS",
    "EXPECTED_PROTEIN_MARKERS",
    "EXPECTED_RNA_FEATURES",
}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal.workflows",
        public_names=_DEPRECATED_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal.workflows",
    )


__all__ = [
    "DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS",
    "DEFAULT_ATERA_WTA_BREAST_DATASET_PATH",
    "DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID",
    "DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_LR_SMOKE_PANEL",
    "DEFAULT_PATHWAY_PANEL",
    "EXPECTED_CELLS",
    "EXPECTED_PROTEIN_MARKERS",
    "EXPECTED_RNA_FEATURES",
    "build_cohort_handoff_spec",
    "build_panel_gap_table",
    "build_serializable_breast_topology_summary",
    "build_serializable_pilot_summary",
    "build_summary",
    "build_top_hypotheses_table",
    "extract_ranked_patches",
    "render_atera_wta_breast_topology_report",
    "render_markdown_report",
    "render_renal_immune_resistance_report",
    "run_atera_wta_breast_topology",
    "run_renal_immune_resistance_pilot",
    "run_validated_renal_ffpe_smoke",
    "validate_summary",
    "write_atera_wta_breast_topology_artifacts",
    "write_output_artifacts",
    "write_renal_immune_resistance_artifacts",
]
