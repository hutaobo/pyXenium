# pyXenium.gmi tutorial

## Overview

`pyXenium.gmi` uses independent contour polygons as samples for sparse GMI
modeling. The first canonical workflow reuses the Atera WTA breast S1/S5
contours: `S1` is the invasive tumor/CAF endpoint and `S5` is the
apocrine-luminal DCIS endpoint.

The workflow is contour-first. It does not build spatial tiles.

## Biological question

The reference task asks which RNA programs and numeric contour features
separate S1 invasive tumor/CAF contours from S5 apocrine-luminal DCIS contours,
and whether the same feature space can describe within-label contour
heterogeneity.

## Setup

Create the S1/S5 contour GeoJSON with the contour tutorial, then run GMI from
the Xenium export root:

```bash
pyxenium gmi run \
  --dataset-root /path/to/WTA_Preview_FFPE_Breast_Cancer_outs \
  --output-dir pyxenium_gmi_outputs/atera_s1_s5 \
  --rna-feature-count 500 \
  --spatial-feature-count 100 \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10 \
  --label-permutation-control \
  --spatial-feature-shuffle-control
```

If the default generated contour file is not present beside the dataset, pass
it explicitly:

```bash
pyxenium gmi run \
  --dataset-root /path/to/WTA_Preview_FFPE_Breast_Cancer_outs \
  --contour-geojson /path/to/xenium_explorer_annotations.s1_s5.generated.geojson \
  --output-dir pyxenium_gmi_outputs/atera_s1_s5
```

## Core workflow

GMI builds one sample per retained S1/S5 contour. RNA counts are aggregated
inside each contour, normalized to contour-level logCPM, and combined with
numeric contour features from `build_contour_feature_table(...)`. Feature
metadata marks every column as `rna` or `spatial`.

The default QC keeps contours with at least 20 cells and nonzero library size.
Dropped endpoint contours remain in `sample_metadata.tsv` with `retained` and
`drop_reason` fields so QC can be visualized and audited.

## Outputs

Each run writes:

- `design_matrix.tsv.gz`
- `sample_metadata.tsv`
- `feature_metadata.tsv`
- `gmi_fit.rds`
- `main_effects.tsv`
- `interaction_effects.tsv`
- `groups.tsv`
- `cv_metrics.tsv`
- `stability.tsv`
- `heterogeneity.tsv`
- `summary.json`
- `report.md`
- `figures/` contour overlays, QC maps, prediction maps, and gene logCPM maps

## Controls

Use RNA-only, spatial-only, no-coordinate, label-permutation,
coordinate-shuffle, and spatial-feature-shuffle runs to separate expression
programs from spatial layout artifacts. The PDC workflow encodes these
presets as reproducible Slurm stages.

## Biological interpretation

The PDC Dardel validation for `v0.4.1` completed all 8 stages. The primary
QC20 model retained 80 of 131 endpoint contours and selected the RNA features
`NIBAN1` and `SORL1`, with train AUC 1.0 and 5-fold stratified spatial CV mean
AUC 1.0 in the stability stage. RNA-only and no-coordinate validations also
selected `NIBAN1` and `SORL1`, supporting the interpretation that the main
S1/S5 separation is an S5/DCIS RNA expression program rather than a direct
centroid or slide-position artifact.

The spatial-only validation selected luminal-like amorphous DCIS composition
features, not coordinate features. This means spatial context is predictive, but
the sparse spatial signal is mainly endpoint composition rather than an
independent CAF/ECM, vascular/pericyte, immune, Notch, IGF/MAPK, Wnt, or
TGF-beta axis. The top1000 sensitivity kept `SORL1` in the main model and had
bootstrap support for `NIBAN1`, but introduced `EFHD1`, so expanded RNA feature
space should be interpreted as a sensitivity result.

The all-nonempty sensitivity retained 102 contours and switched to
`11q13 invasive tumor cell` composition features. This is useful as a QC warning:
low-cell contours can move GMI toward label-composition structure, so QC20
remains the primary biological result.

## Caveats

GMI is sparse and sample-size sensitive. For contour pseudo-bulk analysis, keep
QC20 as the primary model and use all-nonempty contours only as sensitivity
analysis. Zero-cell contours should not enter model fitting.

## Next steps

Archive the PDC scratch artifacts if they need long-term retention, then compare
new datasets against the same QC20, RNA-only, spatial-only, no-coordinate,
top1000, and all-nonempty template before promoting additional selected
features to biological claims.
