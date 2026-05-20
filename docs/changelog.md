# Changelog

All notable changes to pyXenium are documented here.

## Unreleased

## 0.4.6 - 2026-05-20

- Add `pyXenium.perturb` as the lightweight SpatialPerturb Bridge for optional Perturb-seq reference projection handoffs without adding a core runtime dependency.
- Document AI-Driven Spatial Pathologist via the external `spatho` package as an optional workflow bridge, without adding new pyXenium runtime code or dependencies.
- Packaging cleanup: stop shipping `_vendor/Gmi/.github/**` workflow files inside the wheel; narrow `_vendor/Gmi/*` `package-data` to the R sources actually consumed at runtime.
- Pin `aiohttp>=3.9` (previously unpinned) to match the rest of the runtime-dependency hygiene.
- Tighten supported Python range to `>=3.10` so classifiers, runtime, and CI matrix agree (scanpy 1.10 / scientific-stack wheels already exclude 3.8/3.9 in practice).
- Expand CI Python matrix to `3.10`–`3.13` and remove the duplicate `test.yml` workflow.
- Fix bare `except:` in `multimodal.analysis.differential.get_rna_expr_df` so non-`AttributeError` failures surface.
- Plug fsspec handle leaks in `io.xenium_artifacts.open_text` (release on TextIOWrapper/GzipFile construction error) and in `read_cell_feature_matrix_h5` (close the fsspec fileobj when `h5py.File(fileobj, ...)` fails before falling back to a local re-open).
- Seed `np.random.random` in the SCILD CCI adapter; benchmark runs are now reproducible via a `seed` kwarg.
- Add `logger.warning(...)` / `logger.debug(...)` to 15 silent `except Exception:` sites across `multimodal/histoseg_lazyslide.py`, `multimodal/immune_resistance.py`, and `_topology_core.safe_to_parquet`.
- README: replace the self-contradicting `pyXenium.spatho` row in the features table with the actual entry point (`build_spatho_manifest` + external `spatho` package).

## 0.4.5 - 2026-05-08

- Switch the repository and package metadata from the prior non-commercial license to `AGPL-3.0-only`.
- Add `SPATHO AB` as the declared maintainer in public package metadata and user-facing project docs.
- Refresh scverse submission drafts and manuscript availability text so license and maintainer metadata stay consistent with the release.

## 0.4.4 - 2026-05-05

- Finalize the `XeniumSlide` rename across the public I/O surface, examples, tests, and docs while removing the legacy `XeniumSData` alias.
- Refresh GitHub/PyPI-facing version text and installation snippets so the published package metadata matches the new slide-native API.

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
