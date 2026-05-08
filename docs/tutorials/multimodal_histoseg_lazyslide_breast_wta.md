# Breast WTA HistoSeg + LazySlide image features

## Overview

This tutorial documents the upgraded RNA + HistoSeg structure + H&E image
workflow for the Atera breast WTA Xenium sample. HistoSeg provides tissue
structure contours, LazySlide provides WSI tile embeddings and vision-language
scores, and `pyXenium.multimodal` aggregates those outputs into structure-level
image/RNA association tables.

The pyXenium side is intentionally a thin integration layer. It does not run
HistoSeg segmentation and it does not vendor LazySlide image-model code.

## Biological question

For each HistoSeg structure in the breast WTA sample:

- Which H&E image features distinguish this structure from the other
  structures?
- Which PLIP/CONCH text labels are enriched in the tiles assigned to the
  structure?
- Do structure-level H&E features align with Xenium RNA programs and boundary
  hypotheses?

## Workflow boundary

| Package | Responsibility |
| --- | --- |
| HistoSeg | Structure segmentation, contour/ROI GeoJSON, mask QC |
| LazySlide | WSI opening, H&E tiling, PLIP/CONCH feature extraction, spatial tile domains |
| pyXenium | Xenium/H&E alignment, tile-to-structure assignment, structure aggregation, RNA/image association |

## Python API

```python
from pyXenium.multimodal import run_histoseg_lazyslide_structure_workflow

result = run_histoseg_lazyslide_structure_workflow(
    "/path/to/WTA_Preview_FFPE_Breast_Cancer_outs",
    contour_geojson="/path/to/xenium_explorer_annotations.s1_s5.generated.geojson",
    contour_key="histoseg_structures",
    output_dir="/path/to/a100/run",
    model="plip",
    tile_px=224,
    mpp=0.5,
    device="cuda",
    batch_size=64,
    table_format="parquet",
)
```

The workflow writes:

```text
image_contours.parquet
tile_features.parquet
tile_assignments.parquet
structure_image_features.parquet
structure_differential_features.parquet
structure_rna_summary.parquet
structure_program_scores.parquet
rna_image_associations.parquet
program_image_associations.parquet
run_manifest.json
```

## A100 run

Use the A100 runner in `benchmarking/lazyslide_a100/`:

```bash
export PYXENIUM_REPO=/path/to/pyXenium
export A100_ENV_DIR=/path/to/envs/pyxenium-lazyslide
bash benchmarking/lazyslide_a100/scripts/bootstrap_a100_env.sh

export PYXENIUM_ATERA_DATASET=/path/to/WTA_Preview_FFPE_Breast_Cancer_outs
export HISTOSEG_GEOJSON=/path/to/xenium_explorer_annotations.s1_s5.generated.geojson
export A100_OUTPUT_DIR=/path/to/runs/histoseg_lazyslide_breast_wta_plip
export LAZYSLIDE_MODEL=plip

bash benchmarking/lazyslide_a100/scripts/run_a100_histoseg_lazyslide.sh \
  --max-tiles 2000
```

The direct WSI LazySlide path is implemented as the preferred production path.
On the current Atera breast WTA OME-TIFF, the full WSI run was killed by memory
pressure because the H&E image is a non-pyramidal 17 GB OME-TIFF. To avoid
reporting a failed WSI run as biology, the A100 result below uses the existing
HistoSeg contour patch corpus as the tiling input and runs full PLIP inference
over every patch.

The patch fallback is still a real image-model run: HistoSeg supplies the
structure patches, PLIP supplies the image embeddings and text-image scores,
and pyXenium summarizes the resulting image features by HistoSeg structure.
It does not rerun segmentation and it does not make synthetic image features.

The full PLIP patch command used for this RTD snapshot was:

```bash
CUDA_VISIBLE_DEVICES=7 \
/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/envs/plip-patch/bin/python \
  benchmarking/lazyslide_a100/scripts/run_histoseg_patch_plip_workflow.py \
  --manifest /data/taobo.hu/projects/stgpt_l3_20260504/data/xenium_slides/WTA_Preview_FFPE_Breast_Cancer_outs/contour_patches_manifest.json \
  --output-dir /data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/runs/patch_plip_full \
  --batch-size 64 \
  --table-format parquet
```

## A100 PLIP result snapshot

The committed RTD artifacts in this page come from a completed A100 run on GPU
7 using `vinid/plip` through `transformers.CLIPModel`.

| Field | Value |
| --- | --- |
| Workflow | `histoseg_patch_plip_structure_features` |
| GPU | NVIDIA A100-SXM4-40GB |
| Torch | `2.6.0+cu124` |
| Model | `vinid/plip` |
| HistoSeg patches | 2,606 |
| HistoSeg structures | 7 |
| Embedding dimensions | 512 |
| Runtime | 61.5 seconds |

The PLIP text terms used for zero-shot scoring were:

```text
ductal epithelium
invasive carcinoma
in situ carcinoma
fibrotic stroma
immune infiltrate
necrosis
adipose tissue
vascular stroma
lumen or secretion
```

### Structure-level labels

| HistoSeg structure | Tiles | Top mean PLIP term | Top tile-label mode | Enriched text-similarity terms |
| --- | ---: | --- | --- | --- |
| 11q13 Invasive Tumor Cells | 136 | vascular stroma | adipose tissue | invasive carcinoma, in situ carcinoma, immune infiltrate |
| Apocrine Cells | 18 | lumen or secretion | lumen or secretion | lumen or secretion, ductal epithelium |
| Basal-like Structured DCIS Cells | 548 | lumen or secretion | lumen or secretion | ductal epithelium, lumen or secretion, in situ carcinoma |
| Endothelial Cells | 735 | adipose tissue | adipose tissue | fibrotic stroma, vascular stroma, adipose tissue |
| Luminal-like Amorphous DCIS Cells | 121 | adipose tissue | adipose tissue | invasive carcinoma, in situ carcinoma |
| Macrophages | 695 | vascular stroma | adipose tissue | vascular stroma, fibrotic stroma |
| Plasma Cells | 353 | vascular stroma | lumen or secretion | immune infiltrate, necrosis |

The enriched terms are one-vs-rest positive PLIP text-similarity features with
FDR < 0.05 when available. They should be interpreted as image-language
features, not as diagnostic labels. The score range is narrow, so the most
useful signal is the structure-to-structure contrast rather than the absolute
name of a single top label.

### Visual outputs

![PLIP tile embedding UMAP](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/tile_embedding_umap.png)

![Structure-level PLIP text similarity heatmap](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/structure_text_similarity_heatmap.png)

![Spatial tile map](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/spatial_tile_map.png)

![Representative tile montage](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/representative_tile_montage.png)

### Artifact files

The copied A100 snapshot is stored in:

```text
docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/
```

Key files are:

- [`run_manifest.json`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/run_manifest.json)
- [`structure_text_summary.csv`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/structure_text_summary.csv)
- [`structure_image_features.csv`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/structure_image_features.csv)
- [`structure_differential_features.csv`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/structure_differential_features.csv)
- [`tile_feature_summary.csv`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/tile_feature_summary.csv)
- [`tile_embedding_umap.csv`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/tile_embedding_umap.csv)
- [`tile_features.parquet`](../_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/tile_features.parquet)

## Interpretation rules

- `structure_image_features` is the main table for asking what image signatures
  each HistoSeg structure carries.
- `structure_differential_features` ranks one-vs-rest image features per
  structure.
- The current RTD snapshot reports the full PLIP image-feature layer. RNA/image
  coupling should be generated by joining these structure IDs back to the
  pyXenium contour feature table and boundary program scores.
- PLIP is the first required A100 result. CONCH is a comparison result only when
  the model can be loaded with valid credentials.

## Current implementation status

The pyXenium API, optional dependency boundary, A100 runner, patch fallback,
artifact schema, and full A100 PLIP image-feature snapshot are implemented.
The next production hardening step is to convert or cache the breast WTA H&E
image as a pyramidal WSI so the direct LazySlide WSI backend can tile the same
HistoSeg structures without memory pressure.
