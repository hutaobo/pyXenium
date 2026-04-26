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
   build_contour_gmi_dataset
   run_contour_gmi
   run_atera_breast_contour_gmi
   render_contour_gmi_report
```

## Runtime notes

- `Gmi` is installed only from the vendored source snapshot under
  `pyXenium._vendor.Gmi`; normal runtime paths never install from GitHub.
- Required R packages are `cPCG`, `MASS`, `Rcpp`, and `RcppEigen`.
- Backwards-compatible `SpatialGmi*` aliases remain importable, but they point
  to the contour implementation and do not construct spatial tiles.
