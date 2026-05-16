# A100 Spatial Sensitivity Summary, 2026-05-16

This file records a first-pass spatial robustness analysis for the mTM residual-decoding candidate associations. The analysis was run on A100 and summarized locally for manuscript-strengthening work. These results are not yet incorporated into the current NBT initial-submission manuscript, Supplementary Information or DOCX package.

## Run Context

- Remote host alias: `a100`
- Remote working directory: `/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/runs/spatial_sensitivity_20260516`
- Script: `manuscript/mtm_wta_nbt_replication/run_spatial_sensitivity.py`
- PLIP candidate source-data table: `docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1c_Spatial_Permutation_Source_Data.csv`
- UNI candidate source-data table: `docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data/Figure_1e_CrossCancer_Signature_Source_Data.csv`
- UNI image-feature candidates were reconstructed by matching Figure 1e program-level rho values to A100 `contour_image_molecular_associations.parquet` rows.
- Breast PLIP and UNI runs summarized from A100 full-run contour tables.
- Cervical PLIP and UNI runs summarized from A100 full-run contour tables.

## Checks Performed

- Leave-one-spatial-block-out: split contours into a 4 x 4 spatial grid and recomputed candidate partial Spearman correlations after excluding each block.
- Local mismatch controls: rematched expression profiles to nearby non-identical contours and compared observed absolute rho with 1,000 local mismatch iterations.
- Centroid jitter sensitivity: perturbed centroid covariates up to 1% of slide span and recomputed correlations over 200 iterations per scale.

## Compact Result Table

| Dataset | Model | Program | n | Observed rho | Reported rho | LOO blocks | LOO sign stability | Local mismatch q95 | Observed > q95 | Max centroid jitter | Min jitter sign stability |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| breast | PLIP | luminal_estrogen_response | 157 | -0.651885 | -0.638985 | 16 | 1.00 | 0.356310 | True | 0.01 | 1.00 |
| breast | PLIP | unfolded_protein_response | 157 | 0.523104 | 0.515438 | 16 | 1.00 | 0.256315 | True | 0.01 | 1.00 |
| breast | PLIP | oxidative_phosphorylation | 157 | 0.536257 | 0.531128 | 16 | 1.00 | 0.306003 | True | 0.01 | 1.00 |
| cervical | PLIP | myofibroblast_caf_activation | 215 | -0.554637 | -0.554637 | 16 | 1.00 | 0.126210 | True | 0.01 | 1.00 |
| cervical | PLIP | emt_invasive_front | 215 | 0.551991 | 0.551991 | 16 | 1.00 | 0.241824 | True | 0.01 | 1.00 |
| cervical | PLIP | tls_adjacent_activation | 215 | 0.550630 | 0.550630 | 16 | 1.00 | 0.170149 | True | 0.01 | 1.00 |
| cervical | PLIP | collagen_ecm_organization | 215 | -0.545507 | -0.545507 | 16 | 1.00 | 0.140831 | True | 0.01 | 1.00 |
| cervical | PLIP | immune_exclusion | 215 | 0.543928 | 0.543928 | 16 | 1.00 | 0.145019 | True | 0.01 | 1.00 |
| cervical | PLIP | stromal_encapsulation | 215 | 0.540767 | 0.540767 | 16 | 1.00 | 0.146446 | True | 0.01 | 1.00 |
| cervical | PLIP | epithelial_identity | 215 | 0.522184 | 0.522184 | 16 | 1.00 | 0.190561 | True | 0.01 | 1.00 |
| cervical | PLIP | immune_activation | 215 | 0.504868 | 0.504868 | 16 | 1.00 | 0.228906 | True | 0.01 | 1.00 |
| cervical | PLIP | oxidative_phosphorylation | 215 | -0.487491 | -0.487491 | 16 | 1.00 | 0.147390 | True | 0.01 | 1.00 |
| cervical | PLIP | emt_invasion | 215 | 0.478659 | 0.478659 | 16 | 1.00 | 0.160595 | True | 0.01 | 1.00 |
| breast | UNI | luminal_estrogen_response | 300 | -0.586359 | -0.586359 | 16 | 1.00 | 0.276549 | True | 0.01 | 1.00 |
| breast | UNI | oxidative_phosphorylation | 300 | -0.456251 | -0.456251 | 16 | 1.00 | 0.181975 | True | 0.01 | 1.00 |
| breast | UNI | collagen_ecm_organization | 300 | -0.376897 | -0.376897 | 16 | 1.00 | 0.165080 | True | 0.01 | 1.00 |
| breast | UNI | t_cell_exhaustion_checkpoint | 300 | 0.390930 | 0.390930 | 16 | 1.00 | 0.104856 | True | 0.01 | 1.00 |
| breast | UNI | emt_invasion | 300 | 0.303753 | 0.303753 | 16 | 1.00 | 0.173340 | True | 0.01 | 1.00 |
| cervical | UNI | epithelial_identity | 215 | -0.527315 | -0.527315 | 16 | 1.00 | 0.123890 | True | 0.01 | 1.00 |
| cervical | UNI | oxidative_phosphorylation | 215 | 0.554715 | 0.554715 | 16 | 1.00 | 0.136469 | True | 0.01 | 1.00 |
| cervical | UNI | stromal_encapsulation | 215 | -0.614523 | -0.614523 | 16 | 1.00 | 0.128435 | True | 0.01 | 1.00 |
| cervical | UNI | immune_exclusion | 215 | -0.633250 | -0.633250 | 16 | 1.00 | 0.134007 | True | 0.01 | 1.00 |
| cervical | UNI | emt_invasion | 215 | 0.533734 | 0.533734 | 16 | 1.00 | 0.138684 | True | 0.01 | 1.00 |

## Interpretation

All 23 candidate associations retained their sign across all 16 leave-one-spatial-block-out recomputations. All observed absolute correlations exceeded the 95th percentile of the local mismatch-control distribution. Centroid jitter up to 1% of slide span did not flip any candidate association.

This strengthens the residual-decoding result by arguing that the strongest reported contour-level associations are not explained by a single spatial block, nearest-neighbor rematching, or small centroid-coordinate perturbations.

## Limits

These checks do not replace a full registration-perturbation analysis, an independent external cohort, protein/IHC validation, or morphology-only diagnostic benchmarking. The breast PLIP observed rho values differ slightly from the submitted source-data values because this script recomputes partial Spearman correlations directly from the contour table; the manuscript should continue to use the locked source-data values unless the source-data package is intentionally updated. The UNI values match the Figure 1e source-data values after reconstructing the image-feature candidates from the full-run association tables.
