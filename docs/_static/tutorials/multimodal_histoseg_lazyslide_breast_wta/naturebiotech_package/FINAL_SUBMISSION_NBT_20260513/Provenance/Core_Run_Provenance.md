# Core Run Provenance

Checked UTC: 2026-05-16T17:45:00Z

## Source of truth for manuscript results

The core mTM manuscript results were generated on the A100 host:

`sscb-a100.scilifelab.se`

The remote A100 directories record the raw runtime and intermediate outputs. The submission package in this repository contains the synchronized manuscript-facing figures, source-data tables and document files.

## Breast Atera WTA runs

Breast root:

`/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507`

Key run directories:

- `runs/direct_lazyslide_plip_full_text`
- `runs/direct_lazyslide_plip_full_text_mtm_wta_programs_20260509`
- `runs/direct_lazyslide_uni_full_mtm_wta_20260509`
- `runs/uni_smoke_20260509`

The local breast `run_manifest.json` records a PLIP direct-WSI run with CUDA available on eight NVIDIA A100-SXM4-40GB devices, 3,115 total tiles, 3,114 assigned tiles and 1,578 contours.

## Cervical Atera WTA runs

Cervical root:

`/data/taobo.hu/pyxenium_lazyslide_cervical_wta_20260511`

Key run directories:

- `runs/direct_lazyslide_plip_full_mtm_wta`
- `runs/direct_lazyslide_uni_full_mtm_wta`

The cervical runs provide the cross-cancer stress-test and PLIP/UNI program-family comparison summarized in Fig. 1e and the source-data tables.

## PDC role

PDC is not the source of the NBT/mTM result tables. PDC records provide breast Atera WTA dataset storage and transfer provenance, including the copied breast dataset under:

`/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs`

Those PDC paths are useful for reproducibility triage but should not be cited as the core analysis runtime for this manuscript.

## Public package boundary

The upload package is self-contained for editorial review: manuscript files, source-data CSVs, figures and supplementary figures are present locally. Internal provenance notes remain in the final package tree for reproducibility triage and are not included in the upload folder. Raw 10x Genomics Atera WTA/H&E files and A100 intermediate outputs are not redistributed in the GitHub repository.
