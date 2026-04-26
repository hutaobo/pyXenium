# Changelog

All notable changes to pyXenium are documented here.

## Unreleased

- No unreleased changes yet.

## 0.4.0 - 2026-04-26

- Add `pyXenium.mechanostress` as a canonical beta public surface for morphology-derived mechanical stress analysis.
- Support HNSCC-style prefixed Xenium artifacts, including `*_cell_feature_matrix.h5`, cells parquet, boundary parquet, and transcript parquet variants.
- Add cohort-level mechanostress workflow APIs and CLI output summaries.
- Add `pyarrow` as a runtime dependency so PyPI installs can read parquet-based Xenium exports directly.

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
