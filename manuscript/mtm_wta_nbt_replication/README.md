# mTM WTA NBT replication

This directory contains manuscript-level replication instructions for the mTM WTA Brief Communication package. It is intended to sit inside the pyXenium repository and to complement the method text in the submission.

## Repositories

- HistoSeg contour-generation software: https://github.com/hutaobo/HistoSeg
- pyXenium analysis, figure and submission-package code: https://github.com/hutaobo/pyXenium

## Replication levels

Two levels are supported.

1. Full GPU analysis from raw inputs. This requires local access to the Atera WTA outputs, HistoSeg contour GeoJSON files and H&E WSI files. It regenerates direct WSI LazySlide features and contour-level mTM outputs.
2. Submission-package rebuild from existing manuscript source tables and figures. This regenerates the Nature-style composite figure, Online Methods/SI docx files and upload directory without rerunning PLIP/UNI.

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

For a local Windows rebuild of figures and submission files:

```powershell
cd D:\GitHub\pyXenium
powershell -ExecutionPolicy Bypass -File manuscript\mtm_wta_nbt_replication\run_full_replica.ps1
```

## Expected manuscript values

The rebuild should preserve the following reported values:

- Breast S3 luminal estrogen response partial Spearman rho: `-0.639`
- Breast S3 unfolded protein response partial Spearman rho: `0.515`
- Breast S3 oxidative phosphorylation partial Spearman rho: `0.531`
- Spatial permutation empirical P for the strongest defended programs: `9.999 x 10^-5`

## Notes

The replication scripts do not claim IHC/protein validation, causality, directional boundary effects or morphology-only clinical deployment. They reproduce a paired H&E-WTA contour-constrained residual decoding analysis.
