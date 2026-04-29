# Atera contour boundary ecology workflow

This workflow demonstrates contour-native multimodal discovery on the Atera WTA
FFPE breast Xenium sample. It combines each contour's H&E patch with cell-level
spatial transcriptomics to score tumor-boundary ecology programs, cluster
contour ecotypes, match controls, and rank biological hypotheses.

Notebook:

- [RTD tutorial](../tutorials/contour_boundary_ecology)
- Source notebook: `docs/tutorials/contour_boundary_ecology.ipynb`

## Dataset

The reference tutorial is parameterized for:

```text
Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs
```

It expects the standard Xenium export plus:

- `WTA_Preview_FFPE_Breast_Cancer_he_image.ome.tif`
- `WTA_Preview_FFPE_Breast_Cancer_he_alignment.csv`
- `xenium_explorer_annotations.generated.geojson`
- `cells.parquet`
- `cell_feature_matrix.h5`

## What the workflow does

1. Loads the Atera Xenium sample with H&E image alignment metadata.
2. Imports a tutorial-sized subset of Xenium Explorer contours from GeoJSON.
3. Cuts one level-0 H&E patch per contour into `XeniumSData.contour_images`.
4. Builds a contour study table with geometry, pathomics, cell-level
   pseudobulk RNA, pathway scores, cell-cell interaction summaries, and context
   features.
5. Scores the six default boundary programs:
   `immune_exclusion`, `myeloid_vascular_belt`, `emt_invasive_front`,
   `stromal_encapsulation`, `tls_adjacent_activation`, and
   `necrotic_hypoxic_rim`.
6. Clusters contour ecotypes, pairs exemplar contours with matched controls,
   and writes a report-ready discovery package.

## Outputs

The checked tutorial artifact bundle is stored under:

```text
manuscript/atera_contour_boundary_ecology/
```

Primary files:

- `summary.json`
- `report.md`
- `contour_features.csv`
- `program_scores.csv`
- `ecotype_assignments.csv`
- `matched_exemplars.csv`
- `program_feature_deltas.csv`
- `hypothesis_ranking.csv`
- `exemplar_montage.png`
- `atera_tutorial_contour_subset.geojson`

## Biology readout

The current tutorial run analyzes 28 real Atera contours across seven Explorer
structures and 170,057 cells. It identifies four contour ecotypes and ranks all
six boundary programs. The top-program distribution is:

- `immune_exclusion`: 10 contours
- `stromal_encapsulation`: 6 contours
- `emt_invasive_front`: 3 contours
- `tls_adjacent_activation`: 3 contours
- `necrotic_hypoxic_rim`: 3 contours
- `myeloid_vascular_belt`: 3 contours

The result should be treated as a tutorial-scale discovery package: it is meant
to show how contour H&E evidence, molecular evidence, and matched controls are
assembled into hypotheses. Larger production reruns should use more contours
and may enable streamed transcript-gradient profiling.
