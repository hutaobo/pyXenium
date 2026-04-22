from __future__ import annotations

from pyXenium._compat import deprecated_callable, deprecated_symbol
from pyXenium.multimodal.workflows.renal_ffpe_protein import (
    build_summary as _build_summary,
    render_markdown_report as _render_markdown_report,
    run_validated_renal_ffpe_smoke as _run_validated_renal_ffpe_smoke,
    validate_summary as _validate_summary,
    write_output_artifacts as _write_output_artifacts,
)

_PUBLIC_NAMES = {
    "DEFAULT_DATASET_PATH",
    "EXPECTED_CELLS",
    "EXPECTED_PROTEIN_MARKERS",
    "EXPECTED_RNA_FEATURES",
}

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


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal.workflows.renal_ffpe_protein",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal.workflows",
    )


__all__ = sorted(_PUBLIC_NAMES) + [
    "build_summary",
    "render_markdown_report",
    "run_validated_renal_ffpe_smoke",
    "validate_summary",
    "write_output_artifacts",
]
