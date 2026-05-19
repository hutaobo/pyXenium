# pyXenium.pathway Nature Figure Package

This package contains two R-rendered Nature-style figures for the new pyXenium.pathway morphopathway suite.

Primary figures:
- `figures/figure1_breast_discovery_morphopathway.svg|pdf|tiff|png`
- `figures/figure2_cross_cancer_stability_validation.svg|pdf|tiff|png`

Traceability:
- Per-panel source tables are under `source_data/`.
- H&E image integrity and scale calibration are recorded in `source_data/he_image_integrity_manifest.csv`.
- H&E display contrast is documented in `source_data/he_display_adjustment_manifest.csv`; raw source pixels are unchanged.
- QA is recorded in `reports/figure_qa_report.md` and `reports/figure_qa_report.json`.
- A100 fallback instructions are recorded in `a100/a100_fallback_manifest.json`.

Evidence boundary: this package uses only the new pyXenium.pathway morphopathway result bundle and raw Atera H&E/WTA inputs.
