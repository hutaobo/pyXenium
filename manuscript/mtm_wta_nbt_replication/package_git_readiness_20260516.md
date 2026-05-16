# Package/git readiness audit, 2026-05-16

Scope: one-figure NBT initial submission package under
`docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/`.

## Current git behavior

- `.gitignore` line 27 ignores the full `naturebiotech_package/` directory.
- Files already tracked inside that ignored directory still appear as modified or deleted.
- Newly created files inside that ignored directory do not appear in normal `git status` and require `git add -f` if they should be committed.

## Intended tracked changes

These modified tracked files are part of the package hardening work and can be staged with normal `git add`:

- `benchmarking/lazyslide_a100/scripts/prepare_nbt_initial_submission_upload.py`
- `docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Main_Text.md`
- `docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Online_Methods.md`
- `docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Supplementary_Information.md`
- `docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Cover_Letter.md`
- regenerated upload markdown and DOCX files under `NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/`
- `manuscript/mtm_wta_nbt_replication/README.md`
- `manuscript/mtm_wta_nbt_replication/expected_outputs.md`
- `manuscript/mtm_wta_nbt_replication/replica_manifest.yaml`

## Intentional source-data renaming

The stale pre-one-figure source-data names were removed from git and replaced by
the current Fig. 1 panel names plus supplementary-table robustness files. This
keeps the upload package from exposing old multi-figure numbering.

## New ignored files that should be force-added if committing the package

The renamed one-figure source-data CSVs are currently ignored by `.gitignore` and should be force-added:

```bash
git add -f \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1b_Hero_Patches_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1c_Spatial_Permutation_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1c_BlockBootstrap_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1d_MAZ_QC_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1e_CrossCancer_Signature_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Supplementary_Table_5_SpatialSensitivity_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Supplementary_Table_6_GeneComponent_Summary_Source_Data.csv \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Supplementary_Table_6_GeneComponent_Long_Source_Data.csv
```

The internal audit notes are also ignored. Force-add them only if the internal audit trail should be committed:

```bash
git add -f \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Provenance/Claim_Audit_20260516.md \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Provenance/Reference_Audit_20260516.md \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Provenance/SI_SourceData_Audit_20260516.md
```

## Suggested staging sequence

```bash
git add \
  benchmarking/lazyslide_a100/scripts/prepare_nbt_initial_submission_upload.py \
  manuscript/mtm_wta_nbt_replication/README.md \
  manuscript/mtm_wta_nbt_replication/expected_outputs.md \
  manuscript/mtm_wta_nbt_replication/replica_manifest.yaml \
  manuscript/mtm_wta_nbt_replication/package_git_readiness_20260516.md

git add \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Cover_Letter.md \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Main_Text.md \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Online_Methods.md \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/FINAL_SUBMISSION_NBT_20260513/Supplementary_Information.md

git add \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/*.md \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/*.docx

git add -u \
  docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data
```

Then run the `git add -f` command for the renamed source-data and robustness
CSV files above.

## Do not stage by default

- `notebook_outputs/`
- `src/pyXenium/notebooks/notebook_outputs/`
- `NBT_INTERNAL_ADMIN_20260515/`
- any unreviewed cluster/runtime files

## Final package inventory

The regenerated upload folder contains:

- manuscript, online methods, cover letter and supplementary information in both markdown and DOCX form
- one main figure in SVG/PDF/PNG form
- supplementary figure files
- five one-figure source-data CSVs plus three supplementary robustness source-data CSVs

The regenerated upload folder does not include a `Provenance/` subdirectory.
