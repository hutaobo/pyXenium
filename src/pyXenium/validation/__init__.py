from __future__ import annotations

from importlib import import_module

from pyXenium._compat import deprecated_callable, deprecated_symbol

_ATTRIBUTE_EXPORTS = {
    "DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS": (".atera_wta_breast_topology", "DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS"),
    "DEFAULT_ATERA_WTA_BREAST_DATASET_PATH": (".atera_wta_breast_topology", "DEFAULT_ATERA_WTA_BREAST_DATASET_PATH"),
    "DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID": (".atera_wta_breast_topology", "DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID"),
    "DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR": (".atera_wta_breast_topology", "DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR"),
    "DEFAULT_CCI_SMOKE_PANEL": (".atera_wta_breast_topology", "DEFAULT_CCI_SMOKE_PANEL"),
    "DEFAULT_PATHWAY_PANEL": (".atera_wta_breast_topology", "DEFAULT_PATHWAY_PANEL"),
    "build_serializable_breast_topology_summary": (".atera_wta_breast_topology", "build_serializable_breast_topology_summary"),
    "render_atera_wta_breast_topology_report": (".atera_wta_breast_topology", "render_atera_wta_breast_topology_report"),
    "run_atera_wta_breast_topology": (".atera_wta_breast_topology", "run_atera_wta_breast_topology"),
    "write_atera_wta_breast_topology_artifacts": (".atera_wta_breast_topology", "write_atera_wta_breast_topology_artifacts"),
    "DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS"),
    "DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY"),
    "DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH"),
    "DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY"),
    "DEFAULT_ATERA_WTA_CERVICAL_LR_PANEL": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_LR_PANEL"),
    "DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL"),
    "DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL"),
    "DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID"),
    "DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR": (".atera_wta_cervical_end_to_end", "DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR"),
    "build_atera_wta_cervical_bio6_structures": (".atera_wta_cervical_end_to_end", "build_atera_wta_cervical_bio6_structures"),
    "build_serializable_cervical_end_to_end_summary": (".atera_wta_cervical_end_to_end", "build_serializable_cervical_end_to_end_summary"),
    "render_atera_wta_cervical_end_to_end_report": (".atera_wta_cervical_end_to_end", "render_atera_wta_cervical_end_to_end_report"),
    "run_atera_wta_cervical_end_to_end": (".atera_wta_cervical_end_to_end", "run_atera_wta_cervical_end_to_end"),
    "run_sfplot_tbc_table_bundle": (".sfplot_tbc_bridge", "run_sfplot_tbc_table_bundle"),
}

_DEPRECATED_CALLABLE_EXPORTS = {
    "build_summary": "build_summary",
    "validate_summary": "validate_summary",
    "render_markdown_report": "render_markdown_report",
    "write_output_artifacts": "write_output_artifacts",
    "run_validated_renal_ffpe_smoke": "run_validated_renal_ffpe_smoke",
    "build_cohort_handoff_spec": "build_cohort_handoff_spec",
    "build_panel_gap_table": "build_panel_gap_table",
    "build_serializable_pilot_summary": "build_serializable_pilot_summary",
    "build_top_hypotheses_table": "build_top_hypotheses_table",
    "extract_ranked_patches": "extract_ranked_patches",
    "render_renal_immune_resistance_report": "render_renal_immune_resistance_report",
    "run_renal_immune_resistance_pilot": "run_renal_immune_resistance_pilot",
    "write_renal_immune_resistance_artifacts": "write_renal_immune_resistance_artifacts",
}

_DEPRECATED_PUBLIC_NAMES = {
    "DEFAULT_DATASET_PATH",
    "EXPECTED_CELLS",
    "EXPECTED_PROTEIN_MARKERS",
    "EXPECTED_RNA_FEATURES",
}


def __getattr__(name: str):
    if name in _ATTRIBUTE_EXPORTS:
        module_name, attribute_name = _ATTRIBUTE_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    if name in _DEPRECATED_CALLABLE_EXPORTS:
        target_name = _DEPRECATED_CALLABLE_EXPORTS[name]
        multimodal = import_module("pyXenium.multimodal")
        value = deprecated_callable(
            getattr(multimodal, target_name),
            old_path=f"pyXenium.validation.{name}",
            new_path=f"pyXenium.multimodal.workflows.{target_name}",
        )
        globals()[name] = value
        return value
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal.workflows",
        public_names=_DEPRECATED_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal.workflows",
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ATTRIBUTE_EXPORTS) | set(_DEPRECATED_CALLABLE_EXPORTS) | _DEPRECATED_PUBLIC_NAMES)


__all__ = [
    "DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS",
    "DEFAULT_ATERA_WTA_BREAST_DATASET_PATH",
    "DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID",
    "DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR",
    "DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS",
    "DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY",
    "DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH",
    "DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY",
    "DEFAULT_ATERA_WTA_CERVICAL_LR_PANEL",
    "DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL",
    "DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL",
    "DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID",
    "DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_CCI_SMOKE_PANEL",
    "DEFAULT_PATHWAY_PANEL",
    "EXPECTED_CELLS",
    "EXPECTED_PROTEIN_MARKERS",
    "EXPECTED_RNA_FEATURES",
    "build_atera_wta_cervical_bio6_structures",
    "build_cohort_handoff_spec",
    "build_panel_gap_table",
    "build_serializable_breast_topology_summary",
    "build_serializable_cervical_end_to_end_summary",
    "build_serializable_pilot_summary",
    "build_summary",
    "build_top_hypotheses_table",
    "extract_ranked_patches",
    "render_atera_wta_breast_topology_report",
    "render_atera_wta_cervical_end_to_end_report",
    "render_markdown_report",
    "render_renal_immune_resistance_report",
    "run_atera_wta_breast_topology",
    "run_atera_wta_cervical_end_to_end",
    "run_renal_immune_resistance_pilot",
    "run_sfplot_tbc_table_bundle",
    "run_validated_renal_ffpe_smoke",
    "validate_summary",
    "write_atera_wta_breast_topology_artifacts",
    "write_output_artifacts",
    "write_renal_immune_resistance_artifacts",
]
