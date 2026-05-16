# mTM WTA NBT replication

This directory contains manuscript-level replication instructions for the mTM WTA Brief Communication package. It is intended to sit inside the pyXenium repository and to complement the method text in the submission.

## Repositories

- HistoSeg contour-generation software: https://github.com/hutaobo/HistoSeg
- pyXenium analysis, figure and submission-package code: https://github.com/hutaobo/pyXenium

## Replication levels

Two levels are supported.

1. Full GPU analysis from raw inputs. This requires local access to the Atera WTA outputs, HistoSeg contour GeoJSON files and H&E WSI files. It regenerates direct WSI LazySlide features and contour-level mTM outputs.
2. Submission-package rebuild from existing manuscript source tables and figures. This regenerates the Nature-style composite figure, Online Methods/SI docx files and upload directory without rerunning PLIP/UNI.
3. Reviewer-side processed-data validation. This uses deposited source-data CSVs and sanitized processed summary tables to validate the main reported numbers without raw 10x files or remote compute access.

Raw 10x Genomics Atera WTA input files are not redistributed in this repository. Provide them through the environment variables documented in `replica_manifest.yaml`.

## Quick start

For a Linux/A100 run:

```bash
cd /path/to/pyXenium
export PYTHON=/path/to/python
export BREAST_DATASET_ROOT=/path/to/WTA_Preview_FFPE_Breast_Cancer_outs/spatialdata.zarr
export BREAST_HISTOSEG_GEOJSON=/path/to/xenium_explorer_annotations.s1_s5.generated.geojson
export BREAST_HE_OME_TIF=/path/to/WTA_Preview_FFPE_Breast_Cancer_he_image.ome.tif
export BREAST_WORK_DIR=/path/to/mtm_breast_wta_run
export CUDA_VISIBLE_DEVICES=7
bash manuscript/mtm_wta_nbt_replication/run_full_replica.sh
```

The manuscript-facing result tables were generated on `sscb-a100.scilifelab.se`. The A100 source-of-truth run roots are:

- Breast: `/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507`
- Breast PLIP mTM: `runs/direct_lazyslide_plip_full_text_mtm_wta_programs_20260509`
- Breast UNI mTM: `runs/direct_lazyslide_uni_full_mtm_wta_20260509`
- Cervical: `/data/taobo.hu/pyxenium_lazyslide_cervical_wta_20260511`
- Cervical PLIP/UNI mTM: `runs/direct_lazyslide_plip_full_mtm_wta` and `runs/direct_lazyslide_uni_full_mtm_wta`

PDC paths are dataset storage and transfer provenance for the breast Atera WTA input copy. They are not the source of the NBT/mTM result tables.

For a local Windows rebuild of figures and submission files:

```powershell
cd D:\GitHub\pyXenium
powershell -ExecutionPolicy Bypass -File manuscript\mtm_wta_nbt_replication\run_full_replica.ps1
```

For a reviewer-facing processed-data archive and main-number validation:

```bash
cd /path/to/pyXenium
python3 manuscript/mtm_wta_nbt_replication/build_processed_data_archive.py
python3 manuscript/mtm_wta_nbt_replication/recompute_main_numbers.py
```

This creates `manuscript/mtm_wta_nbt_replication/processed_data_archive_20260516/` and a sibling zip file for DOI deposition. Generated archive files are ignored by git; the builder and validation script are tracked.

For spatial sensitivity checks after a full mTM run has produced
`contour_multimodal_summary.parquet`:

```bash
python3 manuscript/mtm_wta_nbt_replication/run_spatial_sensitivity.py \
  --run-dir /path/to/runs/direct_lazyslide_plip_full_text_mtm_wta_programs_20260509 \
  --out-dir manuscript/mtm_wta_nbt_replication/spatial_sensitivity_20260516
```

The runner produces leave-one-spatial-block-out, local mismatched-pair and
centroid-jitter sensitivity tables. Generated sensitivity outputs are ignored by
git until they are reviewed and explicitly promoted into the manuscript package.

For component-gene validation of candidate WTA programs after a full mTM run:

```bash
python3 manuscript/mtm_wta_nbt_replication/run_gene_component_validation.py \
  --dataset-root /path/to/WTA_Preview_FFPE_Breast_Cancer_outs \
  --contour-geojson /path/to/xenium_explorer_annotations.s1_s5.generated.geojson \
  --contour-multimodal /path/to/runs/direct_lazyslide_plip_full_text_mtm_wta_programs_20260509/contour_multimodal_summary.parquet \
  --candidates docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1c_Spatial_Permutation_Source_Data.csv \
  --dataset breast \
  --model plip \
  --out-dir manuscript/mtm_wta_nbt_replication/gene_component_validation_20260516
```

This script audits whether candidate H&E embedding axes track the component
genes of the nominated WTA programs. It is a biological sanity check and does
not replace IHC/protein validation.

## Expected manuscript values

The rebuild should preserve the following reported values:

- Breast S3 luminal estrogen response partial Spearman rho: `-0.639`
- Breast S3 unfolded protein response partial Spearman rho: `0.515`
- Breast S3 oxidative phosphorylation partial Spearman rho: `0.531`
- Spatial permutation empirical P for the strongest defended programs: `9.999 x 10^-5`

## Notes

The replication scripts do not claim IHC/protein validation, causality, directional boundary effects or morphology-only clinical deployment. They reproduce a paired H&E-WTA contour-constrained residual decoding analysis.
