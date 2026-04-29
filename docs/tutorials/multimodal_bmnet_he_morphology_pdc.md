# BM-Net/H&E morphology increment on PDC

## Overview

This tutorial shows how to run a BM-Net-style H&E morphology increment pilot on
the aligned breast Xenium RNA + H&E contour workflow. The goal is not simply to
add another image model. The goal is to ask a more specific question:

> Does H&E-derived contour morphology add information beyond Xenium-native DAPI,
> cell-boundary, and nucleus-boundary morphology?

The workflow extends the existing RNA + contour + H&E tutorial with three extra
pieces:

- a named pathology backend that can emit BM-Net-style breast pathology features
  such as `bmnet__whole__invasive_prob`
- a PDC runner that crops aligned H&E patches for each contour and writes
  contour-level model features
- an increment test that compares H&E morphology blocks against
  Xenium-native morphology blocks

The implementation is intentionally downstream-only. It does not change the
default behavior of `run_contour_boundary_ecology_pilot`.

```{warning}
The completed PDC run documented below used the `deterministic-smoke` backend.
It validates contour cropping, schema, Slurm execution, artifact writing, and
increment-analysis plumbing. It is not biological evidence and should not be
interpreted as trained BM-Net prediction.
```

## Biological Question

The breast Xenium export already contains DAPI-derived nucleus segmentation and
stain-informed cell segmentation. H&E may still add information because it
captures tissue architecture, stromal texture, lumen/duct organization,
necrosis-like appearance, tumor-front morphology, and eosin/hematoxylin contrast
that are not fully represented by Xenium-native cell and nucleus geometries.

This tutorial therefore treats H&E morphology as a candidate increment, not as a
replacement for Xenium-native morphology.

## Model Choices

The BM-Net paper describes a breast whole-slide image classifier with four
diagnostic classes: normal, benign, in situ carcinoma, and invasive carcinoma
([Bioengineering 2022](https://www.mdpi.com/2306-5354/9/6/261)). Public weights
were not confirmed during setup, so the PDC scaffold supports three backend
levels:

| Backend | Purpose | Output semantics |
| --- | --- | --- |
| `deterministic-smoke` | Dependency-light smoke test for PDC and artifact schema | BM-Net-like feature names, no biological meaning |
| `bmnet-local` | Use when a compatible trained BM-Net checkpoint is available | `bmnet__...` four-class probabilities |
| `hf-pathology-backbone` | Use a pathology foundation/surrogate model when BM-Net weights are unavailable | `pathology__...` embedding features, not BM-Net probabilities |

Two useful Hugging Face candidates discovered during setup were
[`1aurent/vit_small_patch8_224.lunit_dino`](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino)
and [`wisdomik/QuiltNet-B-32`](https://huggingface.co/wisdomik/QuiltNet-B-32).
The default surrogate in the runner is the Lunit DINO ViT model because it can
be loaded through the current `timm`/Hugging Face path.

## Inputs

The completed smoke run used the Atera WTA FFPE breast cancer Xenium export on
PDC:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs
```

The expected input files are:

- `cell_feature_matrix.h5`
- `cells.parquet`
- `WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv`
- `xenium_explorer_annotations.s1_s5.generated.geojson`
- an aligned H&E OME-TIFF named
  `WTA_Preview_FFPE_Breast_Cancer_he_image.ome.tif`
- the matching H&E alignment/keypoint files when available

For the first smoke run, a downsampled aligned H&E OME-TIFF was staged on PDC to
avoid uploading the full 17.7 GB H&E image. The downsample was used only to
validate the workflow.

## PDC Setup

From the staged pyXenium repository on PDC:

```bash
export PDC_ROOT=/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04
export PDC_XENIUM_ROOT=/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs

bash benchmarking/bmnet_pdc/scripts/bootstrap_pdc_env.sh
```

The bootstrap script creates the optional BM-Net/PDC environment with the image
model dependencies used by the real backends.

## Smoke Run

Run a small deterministic smoke job first:

```bash
bash benchmarking/bmnet_pdc/scripts/submit_pdc_bmnet_pilot.sh \
  --backend deterministic-smoke \
  --smoke-max-contours 5
```

The completed smoke run used:

| Field | Value |
| --- | --- |
| Slurm job | `20143908` |
| Backend | `deterministic-smoke` |
| Run directory | `/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04/runs/bmnet_smoke5_deterministic_smoke_v2` |
| Contours | 5 |
| Status | `COMPLETED` |
| Elapsed time | `00:05:52` |
| Max RSS | about `7.7 GB` |

The retained contour IDs were `S1 S1 #1.1` through `S1 S1 #5.1`.

## Output Artifacts

Each run writes a compact artifact bundle:

```text
bmnet_patch_predictions.csv
bmnet_pdc_run_summary.json
contour_features_with_bmnet.csv
feature_redundancy.csv
he_morphology_features.csv
incremental_prediction.csv
matched_review_table.csv
morphology_increment_summary.json
partial_associations.csv
program_scores.csv
xenium_native_morphology.csv
```

The smoke run summary contained the expected high-level checks:

```json
{
  "n_contours": 5,
  "contour_key": "s1_s5_contours",
  "n_he_morphology_features": 139,
  "n_xenium_native_morphology_features": 10,
  "has_bmnet_features": true,
  "evaluation_mode": "in_sample_small_n"
}
```

`bmnet_patch_predictions.csv` contains BM-Net-style named features, including:

```text
bmnet__whole__normal_prob
bmnet__whole__benign_prob
bmnet__whole__in_situ_prob
bmnet__whole__invasive_prob
bmnet__outer_rim__invasive_prob
bmnet__outer_minus_inner__invasive_prob
edge_contrast__bmnet__whole__invasive_prob
```

For a trained BM-Net checkpoint, these columns would represent contour-level
breast pathology probabilities and rim/region contrasts. In the smoke backend,
they are deterministic H&E color/texture proxies with the same schema.

## Increment Analysis

The increment module writes six analysis artifacts:

| Artifact | Meaning |
| --- | --- |
| `xenium_native_morphology.csv` | Morphology derived from Xenium-native cell and nucleus boundaries |
| `he_morphology_features.csv` | H&E pathomics, BM-Net, and named pathology features |
| `feature_redundancy.csv` | Correlation/redundancy between H&E and Xenium-native feature blocks |
| `incremental_prediction.csv` | Nested prediction models comparing baseline, Xenium-native, H&E, and combined blocks |
| `partial_associations.csv` | Feature associations after adjusting for selected covariates |
| `matched_review_table.csv` | Contour-level review table for manual inspection |

For the five-contour smoke run, `incremental_prediction.csv` is only a schema and
plumbing check because the evaluation is explicitly `in_sample_small_n`. A
biological interpretation requires a larger contour set and a real trained or
validated pathology backend.

## Real Model Runs

When a compatible BM-Net checkpoint is available, run:

```bash
bash benchmarking/bmnet_pdc/scripts/submit_pdc_bmnet_pilot.sh \
  --backend bmnet-local \
  --checkpoint /cfs/klemming/scratch/h/hutaobo/models/bmnet/bmnet.pt \
  --include-full
```

When BM-Net weights are unavailable, run a Hugging Face pathology surrogate:

```bash
bash benchmarking/bmnet_pdc/scripts/submit_pdc_bmnet_pilot.sh \
  --backend hf-pathology-backbone \
  --hf-model 1aurent/vit_small_patch8_224.lunit_dino \
  --smoke-max-contours 20
```

The surrogate backend writes `pathology__...` features instead of pretending to
be BM-Net. This distinction is important for downstream reports.

## Python API

The same workflow can be called directly:

```python
from pyXenium.multimodal import run_bmnet_morphology_increment_pilot

result = run_bmnet_morphology_increment_pilot(
    dataset_root="/path/to/WTA_Preview_FFPE_Breast_Cancer_outs",
    output_dir="/path/to/output/bmnet_smoke",
    contour_geojson="/path/to/xenium_explorer_annotations.s1_s5.generated.geojson",
    backend="deterministic-smoke",
    max_contours=5,
    program_library="breast_boundary_bmnet_v1",
)

print(result["summary"]["artifact_files"])
```

The runner performs four steps:

1. load the Xenium export, cell table, clusters, contours, and aligned H&E image
2. crop whole-contour, inner, and outer-rim H&E patches
3. write named H&E pathology features into the contour feature table
4. compare H&E morphology against Xenium-native morphology with redundancy,
   nested prediction, partial association, and shuffle-control outputs

## Troubleshooting Notes From The Smoke Run

Several practical issues were resolved during the first PDC run:

- If no H&E image is detected, confirm that the OME-TIFF and alignment files are
  staged under the Xenium export root and that `include_images=True` can discover
  them.
- If GeoJSON contours do not contain `polygon_id`, the runner falls back to the
  contour `name` field.
- If `adata.obsm["spatial"]` is absent, pass or stage `cells.parquet` so the
  runner can reconstruct Xenium-native morphology.
- Tiny smoke patches can collapse to a one-pixel RGB array; the current image
  conversion path handles that case.

## Interpretation Rules

Use the output in three tiers:

- Smoke-only result: validates software, files, schema, and PDC orchestration.
- Surrogate pathology result: useful for exploring whether H&E representation
  adds signal, but should be labeled as a surrogate.
- Trained BM-Net result: can support BM-Net-specific biological interpretation
  when the checkpoint, training data, inference date, and labels are recorded in
  `morphology_increment_summary.json`.

For a publishable analysis, the next full run should use at least the full S1/S5
contour set, keep the shuffle control, and report whether H&E/BM-Net blocks add
held-out predictive value beyond Xenium-native cell and nucleus morphology.
