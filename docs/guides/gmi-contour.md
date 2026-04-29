# Contour-GMI guide

`pyXenium.gmi` is pyXenium’s sixth canonical public surface. It adapts the
vendored `Gmi` R package to contour-level spatial transcriptomics, with each
independent contour polygon treated as one pseudo-bulk sample.

## Data model

The canonical workflow starts from a `XeniumSData` object and a contour layer,
usually imported from Xenium Explorer GeoJSON with:

```python
from pyXenium.contour import add_contours_from_geojson

add_contours_from_geojson(
    sdata,
    geojson_path="xenium_explorer_annotations.s1_s5.generated.geojson",
    key="s1_s5_contours",
    id_key="name",
    pixel_size_um=0.2125,
)
```

For the Atera breast workflow, `S1` is the positive invasive tumor/CAF label
and `S5` is the negative apocrine-luminal DCIS label.

## Feature blocks

GMI receives one combined design matrix:

- RNA features are raw counts aggregated inside each retained contour and
  transformed to contour-level logCPM.
- Spatial features are numeric contour descriptors from
  `build_contour_feature_table(...)`, including geometry, composition, context,
  rim, gradient, edge-contrast, and cell-cell interaction summaries when available.
- `feature_metadata.tsv` records `feature_block = rna|spatial` for every
  design-matrix column.

The default feature budget is 500 RNA features plus 100 spatial contour
features. This keeps the model tractable for small contour sample sizes while
preserving the option to run sensitivity analyses.

## Fitting and artifacts

```python
from pyXenium.gmi import ContourGmiConfig, run_atera_breast_contour_gmi

result = run_atera_breast_contour_gmi(
    dataset_root="/path/to/WTA_Preview_FFPE_Breast_Cancer_outs",
    output_dir="pyxenium_gmi_outputs/atera_s1_s5",
    config=ContourGmiConfig(
        feature_count=500,
        spatial_feature_count=100,
        spatial_cv_folds=5,
        bootstrap_repeats=10,
    ),
)
```

The R fit uses the local vendored Gmi source snapshot and writes
`gmi_fit.rds`, `main_effects.tsv`, `interaction_effects.tsv`, predictions,
CV metrics, stability tables, heterogeneity tables, figures, `summary.json`,
and `report.md`.

## Controls and sensitivity runs

Use the controls before interpreting a selected effect:

- RNA-only: checks whether selected genes are sufficient.
- Spatial-only: checks whether contour geometry or tissue context predicts the
  endpoint without RNA.
- No-coordinate: removes direct position/context coordinate features.
- Label permutation: checks whether signal disappears when labels are broken.
- Coordinate shuffle: breaks cell-to-contour spatial membership.
- Spatial feature shuffle: preserves RNA while disrupting the spatial block.
- All-nonempty sensitivity: includes contours with at least one cell, but keeps
  QC20 as the primary result.

## Within-label heterogeneity

After the main S1-vs-S5 fit, GMI computes a contour heterogeneity score inside
each label using standardized RNA and spatial features. When enough contours
exist, the top tertile versus bottom tertile becomes a within-label binary GMI
task; the middle tertile is held out of fitting.

## Interpretation status

`pyXenium.gmi` is canonical and public, with a beta caveat for statistical and
biological interpretation. The `v0.4.1` Atera validation completed on PDC Dardel
as an 8-stage Slurm chain. The primary QC20 result used 80 retained contours
from 131 S1/S5 endpoint contours and selected `NIBAN1` and `SORL1`.

The validation pattern supports this readout:

- RNA-only QC20 selected `NIBAN1` and `SORL1`, so the primary signal is carried
  by expression features.
- No-coordinate QC20 again selected `NIBAN1` and `SORL1`, arguing against a
  direct centroid or slide-position artifact.
- Spatial-only QC20 selected luminal-like amorphous DCIS composition features
  with no coordinate main effects, so spatial predictive signal is mostly
  endpoint composition/context.
- Top1000 QC20 kept `SORL1`, introduced `EFHD1`, and retained bootstrap support
  for `NIBAN1`, so the expanded RNA budget is supportive but less sparse-stable
  than the primary model.
- All-nonempty sensitivity switched to 11q13 invasive tumor cell composition
  features, showing that low-cell contours can alter the sparse model.

For biological interpretation, treat QC20 as the primary result: S1 invasive
tumor/CAF versus S5 apocrine-luminal DCIS is best summarized as an S5/DCIS RNA
program led by `NIBAN1` and `SORL1`. CAF/ECM remodeling, angiogenesis/pericyte,
myeloid-vascular context, Notch, IGF/MAPK, Wnt, and TGF-beta programs were not
selected as primary sparse drivers in this validation, though they remain
candidate axes for larger datasets and follow-up biology.
