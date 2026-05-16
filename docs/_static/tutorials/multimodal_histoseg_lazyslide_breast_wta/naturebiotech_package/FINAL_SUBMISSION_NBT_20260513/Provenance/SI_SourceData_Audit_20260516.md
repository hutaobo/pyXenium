# Supplementary information and source-data audit, 2026-05-16

Scope: `Supplementary_Information.md` and the one-figure upload package `Source_Data/` CSV files.

## Result

- Supplementary Table 1 now reports the same rounded `reported_partial_rho` values used in the main text for the three breast S3 programs:
  - luminal estrogen response: `-0.639`
  - unfolded protein response: `0.515`
  - oxidative phosphorylation: `0.531`
- The source-data CSV still preserves both `reported_partial_rho` and `recomputed_partial_rho` columns for transparency.
- Supplementary Table 2 matches `Figure_1c_BlockBootstrap_Source_Data.csv` after rounding medians and confidence intervals to three decimals.
- Supplementary Table 3 matches `Figure_1e_CrossCancer_Signature_Source_Data.csv` after rounding maximum absolute partial rho values to three decimals.
- Supplementary Table 4 matches `Figure_1b_Hero_Patches_Source_Data.csv` after rounding displayed WTA and H&E z-scores to two decimals.
- Stale source-data names using `Figure_2`, `Figure_3`, `Figure_4`, `Supplementary_BlockBootstrap` or `Supplementary_Spatial_Permutation` are absent from the regenerated upload package.

## Checks run

- Parsed all markdown supplementary tables and compared rows against the regenerated upload CSV files.
- Verified `mismatch_count 0` for Supplementary Tables 1-4 under the declared rounding conventions.
- Verified upload source-data files:
  - `Figure_1b_Hero_Patches_Source_Data.csv`
  - `Figure_1c_Spatial_Permutation_Source_Data.csv`
  - `Figure_1c_BlockBootstrap_Source_Data.csv`
  - `Figure_1d_MAZ_QC_Source_Data.csv`
  - `Figure_1e_CrossCancer_Signature_Source_Data.csv`
- Rebuilt DOCX files after edits.
