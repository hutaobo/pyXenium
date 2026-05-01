# `pyXenium.gmi`

`pyXenium.gmi` is the canonical contour-native GMI surface. It builds
contour-level pseudo-bulk design matrices from Xenium data, combines RNA and
numeric contour features, runs the vendored local `Gmi` R package, and writes
report-ready artifacts for main effects, interactions, controls, and
within-label heterogeneity.

The API is public, and the Atera S1-vs-S5 workflow has a completed PDC Dardel
validation in `v0.4.1`. Statistical and biological interpretation still keeps a
beta caveat: use the bundled controls, cross-validation, and sensitivity runs
before making biological claims on new datasets.

```{eval-rst}
.. currentmodule:: pyXenium.gmi

.. autosummary::
   :toctree: generated
   :nosignatures:

   ContourGmiConfig
   ContourGmiDataset
   ContourGmiResult
   GmiModuleConfig
   GmiModuleResult
   build_contour_gmi_dataset
   run_contour_gmi
   run_atera_breast_contour_gmi
   build_gmi_effect_graph
   discover_gmi_modules
   score_gmi_modules
   render_gmi_module_report
   render_contour_gmi_report
```

## Runtime notes

- `Gmi` is installed only from the vendored source snapshot under
  `pyXenium._vendor.Gmi`; normal runtime paths never install from GitHub.
- Required R packages are `cPCG`, `MASS`, `Rcpp`, and `RcppEigen`.
- Backwards-compatible `SpatialGmi*` aliases remain importable, but they point
  to the contour implementation and do not construct spatial tiles.

## Spatial gene modules

`discover_gmi_modules(...)` adds a supervised module layer on top of an existing
GMI output directory. Selected or bootstrap-stable GMI effects seed each module,
then correlated features, contour-neighborhood spatial-lag correlations, and
GMI interaction partners expand the module. The output bundle includes
`spatial_modules.tsv`, `module_features.tsv`, `module_scores.tsv.gz`,
`module_enrichment.tsv`, `module_interactions.tsv`,
`module_spatial_autocorr.tsv`, a Markdown report, and optional contour score
maps.

The WTA breast PDC module validation generated an S5/DCIS `NIBAN1`/`SORL1`
module in the primary QC20, RNA-only, and no-coordinate runs, plus spatial-only
composition modules and QC sensitivity maps documented in the GMI spatial
module tutorial.
