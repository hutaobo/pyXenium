# Changelog

All notable changes to pyXenium are documented here.

## Unreleased

- Document AI-Driven Spatial Pathologist via the external `spatho` package as pyXenium's eighth feature area, without adding new pyXenium runtime code or dependencies.

## 0.4.1 - 2026-04-26

- Complete the PDC Dardel Slurm validation for the contour-GMI Atera S1/S5 workflow across all 8 stages.
- Add final contour-GMI biological readout: QC20, RNA-only, and no-coordinate stages support an S5/DCIS RNA program led by `NIBAN1` and `SORL1`.
- Document spatial-only and all-nonempty sensitivity behavior, including composition-driven contour context and QC sensitivity to low-cell contours.
- Add PDC validation summaries and release documentation for the GitHub-only `v0.4.1` release.

## 0.4.0 - 2026-04-26

- Add `pyXenium.mechanostress` as the seventh canonical public surface for morphology-derived mechanical stress analysis.
- Support HNSCC-style prefixed Xenium artifacts, including `*_cell_feature_matrix.h5`, cells parquet, boundary parquet, and transcript parquet variants.
- Add cohort-level mechanostress workflow APIs and CLI output summaries.
- Add `pyarrow` as a runtime dependency so PyPI installs can read parquet-based Xenium exports directly.

## 0.3.0 - 2026-04-25

- Promote `pyXenium.gmi` to the sixth canonical public surface for contour-level GMI modeling.
- Add contour-GMI API, tutorial, user-guide, and Atera reproducibility workflow documentation.
- Expose core GMI types and workflow functions from the top-level `pyXenium` namespace.
- Keep the vendored local `Gmi` R source as the only runtime installation source.

## 0.2.3 - 2026-04-22

- Add missing runtime dependencies required by the public import surface in clean environments, including `seaborn`, `statsmodels`, `scanpy`, and `PyYAML`.
- Validate a clean-environment `import pyXenium` path after the rapid `0.2.2` hotfix.

## 0.2.2 - 2026-04-22

- Fix a cross-platform `SyntaxError` in `pyXenium.io.xenium_artifacts.join_path(...)` caused by a backslash-containing f-string expression.
- Restore Linux CI importability immediately after the `0.2.1` metadata/docs sync release.

## 0.2.1 - 2026-04-22

- Add `pyXenium.contour.expand_contours(...)` with overlap-preserving and Voronoi-exclusive contour expansion modes.
- Migrate the documentation site to `Sphinx + pydata-sphinx-theme`.
- Rebuild the homepage and API landing pages around canonical public namespaces.
- Add pyXenium branding assets for Read the Docs and GitHub surfaces.
- Synchronize GitHub, Read the Docs, and PyPI cross-links, metadata, and build/version status displays.
