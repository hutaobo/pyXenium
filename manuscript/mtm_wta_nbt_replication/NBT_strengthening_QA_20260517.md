# NBT strengthening QA, 2026-05-17

## A100 run status

- Host alias: `a100`.
- Remote output root: `/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/runs/nbt_strengthening_20260517`.
- Local full output mirror: `manuscript/mtm_wta_nbt_replication/nbt_strengthening_20260517` (ignored from git because it contains logs and 49 MB of blinded patch images).
- Automated monitor: `run_nbt_strengthening_pipeline.py`, default poll interval 600 seconds.

## Completed analyses

- Registration perturbation: 13 PLIP candidates, 23 perturbations each. All candidates retained the base association sign; breast maximum absolute delta was 0.054 rho and cervical maximum absolute delta was 0.137 rho.
- Nested spatial holdout: 23 PLIP/UNI candidates, 16 folds each. Primary breast PLIP signs were recovered in 75.0-93.8% of folds. Weaker cross-cancer/model stress-test candidates are reported conservatively.
- Blinded morphology panel: 24 breast S3 H&E patches exported across luminal estrogen-response, unfolded-protein-response and oxidative-phosphorylation axes. No pathologist scores are reported yet.

## Package updates

- Added Supplementary Tables 7 and 8 to the Supplementary Information.
- Added registration-perturbation and nested-holdout source-data CSVs to the one-figure upload package.
- Regenerated manuscript, online methods, supplementary information and cover-letter DOCX files from markdown using Pandoc.
- Updated processed-data archive builder and recomputation checker so the DOI-ready archive includes the new source-data tables.

## Checks

- Manuscript title: 76 characters, 10 words.
- Abstract: 63 words.
- Main text body before references: 1,024 words.
- Required statements present: author, affiliation, correspondence, funding, competing interests, data availability, code availability and ethics/data-use.
- Source-data mapping: 12 listed CSV files, 0 missing, 0 unlisted.
- Old `Figure_2`/`Figure_3`/`Figure_4` source-data labels: none found in the initial upload package.
- `recompute_main_numbers.py`: passed after archive rebuild.
- Script compilation: passed for all new and updated replication scripts.

## DOCX QA

DOCX files were regenerated with Pandoc and normalized to US Letter, 1-inch margins using `python-docx`. Structural checks passed with `unzip -t` and `textutil` text extraction, and the new registration/nested-holdout content was confirmed in the DOCX text. Visual render QA could not be completed on this machine because LibreOffice/`soffice` is unavailable to the document renderer.
