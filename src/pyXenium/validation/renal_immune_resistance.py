from __future__ import annotations

from pyXenium._compat import deprecated_callable
from pyXenium.multimodal.workflows.renal_immune_resistance import (
    build_cohort_handoff_spec as _build_cohort_handoff_spec,
    build_panel_gap_table as _build_panel_gap_table,
    build_serializable_pilot_summary as _build_serializable_pilot_summary,
    build_top_hypotheses_table as _build_top_hypotheses_table,
    extract_ranked_patches as _extract_ranked_patches,
    render_renal_immune_resistance_report as _render_renal_immune_resistance_report,
    run_renal_immune_resistance_pilot as _run_renal_immune_resistance_pilot,
    write_renal_immune_resistance_artifacts as _write_renal_immune_resistance_artifacts,
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

__all__ = [
    "build_cohort_handoff_spec",
    "build_panel_gap_table",
    "build_serializable_pilot_summary",
    "build_top_hypotheses_table",
    "extract_ranked_patches",
    "render_renal_immune_resistance_report",
    "run_renal_immune_resistance_pilot",
    "write_renal_immune_resistance_artifacts",
]
