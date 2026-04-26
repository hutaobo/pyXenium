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
programs from spatial layout artifacts. The A100 workflow encodes these
presets as reproducible stages.

## Biological interpretation

The current A100 validation is still running for the promoted canonical beta
surface. Earlier contour-GMI runs pointed to an S5/DCIS expression signal led by
`NIBAN1` and `SORL1`, while spatial contour features were not yet selected as
primary drivers. Treat that as provisional until the new 8-stage validation
completes.

## Caveats

GMI is sparse and sample-size sensitive. For contour pseudo-bulk analysis, keep
QC20 as the primary model and use all-nonempty contours only as sensitivity
analysis. Zero-cell contours should not enter model fitting.

## Next steps

Compare the full, RNA-only, spatial-only, no-coordinate, top1000, and
all-nonempty outputs before promoting any selected feature to a biological
claim.
