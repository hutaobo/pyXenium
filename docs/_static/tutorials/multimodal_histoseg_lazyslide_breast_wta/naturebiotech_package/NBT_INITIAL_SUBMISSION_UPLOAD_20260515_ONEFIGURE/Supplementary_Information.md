# Supplementary Information

## Supplementary Methods

### Contour provenance

HistoSeg contours were generated from Atera WTA cell-coordinate tables and cluster or cell-group definitions. The contour-generation workflow corresponds to `histoseg.contour` in HistoSeg (https://github.com/hutaobo/HistoSeg). H&E images were not used to draw the breast S1-S5 contours or the cervical tissue contours. HistoSeg also includes H&E segmentation utilities, but those utilities were not used to define the contours analyzed here. H&E whole-slide images entered only after contour generation, through direct-WSI LazySlide PLIP/UNI tile embeddings assigned to registered contour geometry. Atera WTA programs were summarized within the same contour polygons, making each contour the shared unit for image features, molecular features and residual statistical testing. The image-feature and WTA aggregation workflows are implemented in pyXenium (https://github.com/hutaobo/pyXenium).

### Non-independence of contour construction and residual estimand

The spatial-omics origin of the HistoSeg contours means that they are not independent histological labels. The analysis therefore estimates within-label residual association rather than independent contour discovery. We denote the spatial-omics-derived discrete contour label as `C`, spatial covariates including centroid and boundary-distance bins as `S`, the continuous WTA program score as `Y` and the H&E foundation-model feature as `X`. The reported association is computed between residualized `X | C,S` and residualized `Y | C,S`. This does not prove standalone morphology-only clinical prediction and does not exclude all contour-construction-induced coupling, but it argues against the simpler explanation that mTM merely rediscovers the discrete HistoSeg label.

### Symbol table for the residual model

| Symbol | Meaning |
|---|---|
| `i` | contour index |
| `C_i` | spatial-omics-derived discrete contour label |
| `S_i` | spatial covariates, including centroid coordinates and boundary-distance bins |
| `X_ik` | H&E foundation-model feature `k` in contour `i` |
| `Y_i` | continuous WTA program score in contour `i` |
| `Z_i` | covariate design vector `[1, one-hot(C_i), S_i]` |
| `r^X_k` | residualized rank-transformed H&E feature |
| `r^Y` | residualized rank-transformed WTA program score |
| `rho_k` | residual partial Spearman association between `r^X_k` and `r^Y` |

### Residual statistical model

Contour-level molecular and image features were rank transformed and residualized against specified covariates: spatial-omics-derived HistoSeg contour label, centroid x/y coordinates and available boundary-distance summaries. For the breast S3 analysis, the contour label was constant and was omitted from the model matrix. In matrix notation, `Z_i = [1, one-hot(C_i), S_i]`, `r^X_k = (I - P_Z) rank(X_k)`, `r^Y = (I - P_Z) rank(Y)`, and `rho_k = cor(r^X_k, r^Y)`, where `P_Z = Z(Z'Z)^-1Z'`. Associations were reported as partial Spearman's rho between residualized image and WTA program vectors.

### Spatial permutation and block bootstrap

For the spatial-null defense, molecular residuals were permuted within strata preserving contour label, centroid-position bins and boundary-distance bins. This retains coarse compartmental and spatial organization while breaking contour-wise H&E-to-WTA pairing. Empirical P values were calculated as `(1 + sum_b 1{|rho_b^null| >= |rho_obs|}) / (1 + B)`. The permutation mitigates coarse spatial-autocorrelation explanations but does not exclude all fine-scale spatial dependence, registration uncertainty or biological coupling induced by the original spatial-omics contour construction. Spatial block bootstrap resampled centroid x/y quartile blocks to estimate confidence intervals.

Additional A100 sensitivity checks were run after the locked source-data package was assembled. Candidate associations were recomputed after leaving out each of 16 spatial blocks, compared with 1,000 local mismatched-pair controls and recalculated under centroid-covariate jitter up to 1% of slide span. Existing tile embeddings were also reassigned under 23 small registration perturbations, including translations, rotations, scale changes and mild combined shifts, before recomputing candidate contour-level embedding means and residual associations. In a nested spatial holdout check, embedding features were selected using only training spatial blocks and evaluated on held-out blocks. Component-gene audits recomputed contour-level means for genes inside the selected breast S3 WTA programs and tested whether the reported H&E embedding axis tracked these component genes under the same residual framework. These checks strengthen transcript-level and statistical consistency but do not replace independent IHC/protein validation.

### Boundary co-variation analysis

Boundary profiles were computed in distance rings around selected tissue interfaces. Candidate molecularly active zones were interpreted only as ring-level H&E-WTA co-variation. Causal, temporal or directional boundary interpretations were not assigned.

### Source-data mapping

| File | Contents |
|---|---|
| `Figure_1b_Hero_Patches_Source_Data.csv` | contour IDs, WTA z-scores and oriented H&E embedding z-scores for breast S3 hero patches |
| `Figure_1c_Spatial_Permutation_Source_Data.csv` | spatial-null permutation results for reported candidate programs |
| `Figure_1c_BlockBootstrap_Source_Data.csv` | spatial block-bootstrap confidence intervals |
| `Figure_1d_MAZ_QC_Source_Data.csv` | conservative boundary co-variation summaries used for the MAZ panel and Supplementary Fig. 3 |
| `Figure_1e_CrossCancer_Signature_Source_Data.csv` | program-family signature summary across breast/cervical and PLIP/UNI |
| `Supplementary_Table_5_SpatialSensitivity_Source_Data.csv` | A100 leave-one-block, local mismatch and centroid-jitter sensitivity summary |
| `Supplementary_Table_6_GeneComponent_Summary_Source_Data.csv` | breast S3 PLIP component-gene audit summary |
| `Supplementary_Table_6_GeneComponent_Long_Source_Data.csv` | per-gene component audit values for selected breast S3 PLIP programs |
| `Supplementary_Table_7_RegistrationPerturbation_Summary_Source_Data.csv` | A100 registration-perturbation summary by candidate |
| `Supplementary_Table_7_RegistrationPerturbation_Long_Source_Data.csv` | per-perturbation registration-sensitivity values |
| `Supplementary_Table_8_NestedSpatialHoldout_Summary_Source_Data.csv` | nested spatial holdout summary by candidate |
| `Supplementary_Table_8_NestedSpatialHoldout_Long_Source_Data.csv` | per-fold nested holdout values |

## Supplementary Tables

### Supplementary Table 1. Spatial permutation defense

| Dataset | Model | Program | Program family | n contours | Observed partial Spearman's rho | Permutations | Empirical P | 99% null threshold |
|---|---|---|---|---:|---:|---:|---:|---:|
| Breast | PLIP | luminal_estrogen_response | endocrine/epithelial identity | 157 | -0.639 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.306 |
| Breast | PLIP | unfolded_protein_response | metabolic/stress | 157 | 0.515 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.232 |
| Breast | PLIP | oxidative_phosphorylation | metabolic/stress | 157 | 0.531 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.290 |
| Cervical | PLIP | myofibroblast_caf_activation | stromal-remodeling/CAF/ECM | 215 | -0.555 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.450 |
| Cervical | PLIP | emt_invasive_front | invasion/boundary/EMT | 215 | 0.552 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.430 |
| Cervical | PLIP | tls_adjacent_activation | immune ecology/TLS/immune exclusion | 215 | 0.551 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.465 |
| Cervical | PLIP | collagen_ecm_organization | stromal-remodeling/CAF/ECM | 215 | -0.546 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.461 |
| Cervical | PLIP | immune_exclusion | immune ecology/TLS/immune exclusion | 215 | 0.544 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.446 |
| Cervical | PLIP | stromal_encapsulation | stromal-remodeling/CAF/ECM | 215 | 0.541 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.461 |
| Cervical | PLIP | epithelial_identity | endocrine/epithelial identity | 215 | 0.522 | 10,000 | 0.0023 | 0.497 |
| Cervical | PLIP | immune_activation | immune ecology/TLS/immune exclusion | 215 | 0.505 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.448 |
| Cervical | PLIP | oxidative_phosphorylation | metabolic/stress | 215 | -0.487 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.396 |
| Cervical | PLIP | emt_invasion | invasion/boundary/EMT | 215 | 0.479 | 10,000 | 9.999 × 10<sup>-5</sup> | 0.407 |

### Supplementary Table 2. Spatial block-bootstrap confidence intervals

| Dataset | Model | Program | Bootstrap samples | Median rho | 95% CI |
|---|---|---|---:|---:|---|
| Breast | PLIP | luminal_estrogen_response | 2,000 | -0.652 | -0.742 to -0.528 |
| Breast | PLIP | unfolded_protein_response | 2,000 | 0.519 | 0.358 to 0.658 |
| Breast | PLIP | oxidative_phosphorylation | 2,000 | 0.531 | 0.404 to 0.657 |
| Cervical | PLIP | myofibroblast_caf_activation | 2,000 | -0.553 | -0.628 to -0.484 |
| Cervical | PLIP | emt_invasive_front | 2,000 | 0.552 | 0.425 to 0.646 |
| Cervical | PLIP | tls_adjacent_activation | 2,000 | 0.551 | 0.475 to 0.631 |
| Cervical | PLIP | collagen_ecm_organization | 2,000 | -0.543 | -0.624 to -0.469 |
| Cervical | PLIP | immune_exclusion | 2,000 | 0.544 | 0.431 to 0.656 |
| Cervical | PLIP | stromal_encapsulation | 2,000 | 0.539 | 0.440 to 0.628 |
| Cervical | PLIP | epithelial_identity | 2,000 | 0.524 | 0.386 to 0.616 |
| Cervical | PLIP | immune_activation | 2,000 | 0.504 | 0.405 to 0.597 |
| Cervical | PLIP | oxidative_phosphorylation | 2,000 | -0.488 | -0.631 to -0.302 |
| Cervical | PLIP | emt_invasion | 2,000 | 0.482 | 0.404 to 0.558 |

### Supplementary Table 3. Program-family signatures by cancer and model

| Dataset | Program family | Model | Maximum absolute partial Spearman's rho | Top WTA program | Support |
|---|---|---|---:|---|---|
| Breast | endocrine/epithelial identity | PLIP | 0.597 | luminal_estrogen_response | strong |
| Breast | endocrine/epithelial identity | UNI | 0.586 | luminal_estrogen_response | strong |
| Breast | metabolic/stress | PLIP | 0.491 | unfolded_protein_response | strong |
| Breast | metabolic/stress | UNI | 0.456 | oxidative_phosphorylation | strong |
| Breast | stromal-remodeling/CAF/ECM | PLIP | 0.330 | collagen_ecm_organization | weak |
| Breast | stromal-remodeling/CAF/ECM | UNI | 0.377 | collagen_ecm_organization | moderate |
| Breast | immune ecology/TLS/immune exclusion | PLIP | 0.409 | tls_b_cell_plasma | moderate |
| Breast | immune ecology/TLS/immune exclusion | UNI | 0.391 | t_cell_exhaustion_checkpoint | moderate |
| Breast | invasion/boundary/EMT | PLIP | NA | not_detected | not_detected |
| Breast | invasion/boundary/EMT | UNI | 0.304 | emt_invasion | weak |
| Cervical | endocrine/epithelial identity | PLIP | 0.522 | epithelial_identity | strong |
| Cervical | endocrine/epithelial identity | UNI | 0.527 | epithelial_identity | strong |
| Cervical | metabolic/stress | PLIP | 0.487 | oxidative_phosphorylation | strong |
| Cervical | metabolic/stress | UNI | 0.555 | oxidative_phosphorylation | strong |
| Cervical | stromal-remodeling/CAF/ECM | PLIP | 0.555 | myofibroblast_caf_activation | strong |
| Cervical | stromal-remodeling/CAF/ECM | UNI | 0.615 | stromal_encapsulation | strong |
| Cervical | immune ecology/TLS/immune exclusion | PLIP | 0.551 | tls_adjacent_activation | strong |
| Cervical | immune ecology/TLS/immune exclusion | UNI | 0.633 | immune_exclusion | strong |
| Cervical | invasion/boundary/EMT | PLIP | 0.552 | emt_invasive_front | strong |
| Cervical | invasion/boundary/EMT | UNI | 0.534 | emt_invasion | strong |

### Supplementary Table 4. Final luminal estrogen-response hero patch pairs

| Pair | Polarity | Contour ID | WTA program z-score | Oriented H&E embedding z-score |
|---:|---|---|---:|---:|
| 1 | high | S3 S3 #424.1 | 3.57 | 1.50 |
| 1 | low | S3 S3 #550.1 | -1.14 | -1.22 |
| 2 | high | S3 S3 #445.1 | 2.30 | 2.53 |
| 2 | low | S3 S3 #516.1 | -1.07 | -1.30 |
| 3 | high | S3 S3 #82.1 | 2.23 | 2.71 |
| 3 | low | S3 S3 #533.1 | -0.93 | -1.61 |
| 4 | high | S3 S3 #287.1 | 1.82 | 2.65 |
| 4 | low | S3 S3 #543.1 | -0.83 | -1.81 |

### Supplementary Table 5. A100 spatial sensitivity summary

| Dataset | Model | Candidate programs | Leave-one-block sign-stable | Exceed local mismatch q95 | Minimum centroid-jitter sign stability |
|---|---|---:|---:|---:|---:|
| Breast | PLIP | 3 | 3/3 | 3/3 | 1.00 |
| Breast | UNI | 5 | 5/5 | 5/5 | 1.00 |
| Cervical | PLIP | 10 | 10/10 | 10/10 | 1.00 |
| Cervical | UNI | 5 | 5/5 | 5/5 | 1.00 |

### Supplementary Table 6. Breast S3 component-gene audit

| Program | Program genes found | Effective image-gene n | Reported program-image rho | Median image-gene rho | Image-gene sign-match fraction | Program high-low positive fraction | Strongest component gene |
|---|---:|---:|---:|---:|---:|---:|---|
| luminal_estrogen_response | 9 | 157 | -0.639 | -0.560 | 1.00 | 1.00 | XBP1 |
| unfolded_protein_response | 9 | 157 | 0.515 | 0.285 | 0.78 | 1.00 | XBP1 |
| oxidative_phosphorylation | 8 | 157 | 0.531 | 0.341 | 0.88 | 1.00 | COX6C |

### Supplementary Table 7. Registration perturbation sensitivity

| Dataset | Model | Program | Perturbations | Base rho | Median perturbed rho | Max abs delta rho | Sign-stable fraction | Min assignment fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Breast | PLIP | luminal_estrogen_response | 23 | -0.652 | -0.652 | 0.054 | 1.000 | 0.999 |
| Breast | PLIP | unfolded_protein_response | 23 | 0.523 | 0.520 | 0.038 | 1.000 | 0.999 |
| Breast | PLIP | oxidative_phosphorylation | 23 | 0.536 | 0.536 | 0.041 | 1.000 | 0.999 |
| Cervical | PLIP | myofibroblast_caf_activation | 23 | -0.555 | -0.548 | 0.077 | 1.000 | 0.999 |
| Cervical | PLIP | emt_invasive_front | 23 | 0.552 | 0.552 | 0.031 | 1.000 | 0.999 |
| Cervical | PLIP | tls_adjacent_activation | 23 | 0.551 | 0.546 | 0.034 | 1.000 | 0.999 |
| Cervical | PLIP | collagen_ecm_organization | 23 | -0.546 | -0.543 | 0.066 | 1.000 | 0.999 |
| Cervical | PLIP | immune_exclusion | 23 | 0.544 | 0.534 | 0.062 | 1.000 | 0.999 |
| Cervical | PLIP | stromal_encapsulation | 23 | 0.541 | 0.535 | 0.060 | 1.000 | 0.999 |
| Cervical | PLIP | epithelial_identity | 23 | 0.522 | 0.509 | 0.137 | 1.000 | 0.999 |
| Cervical | PLIP | immune_activation | 23 | 0.505 | 0.519 | 0.045 | 1.000 | 0.999 |
| Cervical | PLIP | oxidative_phosphorylation | 23 | -0.487 | -0.492 | 0.071 | 1.000 | 0.999 |
| Cervical | PLIP | emt_invasion | 23 | 0.479 | 0.465 | 0.070 | 1.000 | 0.999 |

### Supplementary Table 8. Nested spatial holdout

| Dataset | Model | Program | Folds | Base locked rho | Median held-out selected rho | Selected sign-match fraction | Locked sign-match fraction | Locked-feature reuse fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Breast | PLIP | luminal_estrogen_response | 16 | -0.652 | -0.637 | 0.938 | 0.938 | 1.000 |
| Breast | PLIP | unfolded_protein_response | 16 | 0.523 | 0.462 | 0.750 | 0.750 | 1.000 |
| Breast | PLIP | oxidative_phosphorylation | 16 | 0.536 | 0.367 | 0.750 | 0.812 | 0.000 |
| Cervical | PLIP | myofibroblast_caf_activation | 16 | -0.555 | -0.486 | 0.750 | 0.812 | 0.875 |
| Cervical | PLIP | emt_invasive_front | 16 | 0.552 | 0.626 | 0.750 | 0.750 | 1.000 |
| Cervical | PLIP | tls_adjacent_activation | 16 | 0.551 | 0.516 | 0.625 | 0.625 | 0.688 |
| Cervical | PLIP | collagen_ecm_organization | 16 | -0.546 | -0.531 | 0.562 | 0.625 | 0.750 |
| Cervical | PLIP | immune_exclusion | 16 | 0.544 | 0.593 | 0.688 | 0.688 | 0.938 |
| Cervical | PLIP | stromal_encapsulation | 16 | 0.541 | 0.430 | 0.750 | 0.750 | 0.875 |
| Cervical | PLIP | epithelial_identity | 16 | 0.522 | 0.395 | 0.688 | 0.688 | 1.000 |
| Cervical | PLIP | immune_activation | 16 | 0.505 | 0.383 | 0.562 | 0.562 | 0.188 |
| Cervical | PLIP | oxidative_phosphorylation | 16 | -0.487 | -0.188 | 0.562 | 0.688 | 0.750 |
| Cervical | PLIP | emt_invasion | 16 | 0.479 | 0.729 | 0.875 | 0.875 | 0.875 |
| Breast | UNI | luminal_estrogen_response | 16 | -0.586 | -0.422 | 0.812 | 0.938 | 0.812 |
| Breast | UNI | oxidative_phosphorylation | 16 | -0.456 | -0.493 | 0.938 | 0.938 | 0.938 |
| Breast | UNI | collagen_ecm_organization | 16 | -0.377 | -0.259 | 0.750 | 0.750 | 1.000 |
| Breast | UNI | t_cell_exhaustion_checkpoint | 16 | 0.391 | 0.442 | 0.875 | 0.875 | 1.000 |
| Breast | UNI | emt_invasion | 16 | 0.304 | 0.157 | 0.750 | 0.750 | 1.000 |
| Cervical | UNI | epithelial_identity | 16 | -0.527 | -0.288 | 0.688 | 0.688 | 0.938 |
| Cervical | UNI | oxidative_phosphorylation | 16 | 0.555 | 0.562 | 0.625 | 0.625 | 0.688 |
| Cervical | UNI | stromal_encapsulation | 16 | -0.615 | -0.412 | 0.688 | 0.750 | 0.750 |
| Cervical | UNI | immune_exclusion | 16 | -0.633 | -0.631 | 0.750 | 0.750 | 1.000 |
| Cervical | UNI | emt_invasion | 16 | 0.534 | -0.146 | 0.375 | 0.688 | 0.500 |

## Supplementary Figure Captions

**Supplementary Fig. 1 | Spatial permutation defense.** Program residuals were permuted within compartment-aware strata defined by spatial-omics-derived contour labels, centroid-position bins and boundary-distance bins. Observed residual associations are compared with the compartment-aware spatial null; this test mitigates coarse spatial-autocorrelation explanations but does not exclude all fine-scale spatial dependence.

**Supplementary Fig. 2 | Expanded breast S3 luminal estrogen-response examples.** High- and low-program S3 contours are shown as expanded H&E examples for Fig. 1b. These patches share the same spatial-omics-derived contour label and were selected after statistical ranking; they are not a blinded visual diagnostic claim.

**Supplementary Fig. 3 | Expanded boundary co-variation profiles.** Ring-level profiles show H&E-WTA co-variation at selected tissue interfaces. These examples illustrate conservative boundary co-variation and do not imply causality, temporal ordering or directional boundary effects.

## Source Data

Source Data files accompany Fig. 1, Supplementary Figs. 1-3 and the supplementary robustness tables. The upload package names figure-panel files by the relevant Fig. 1 panel and names robustness tables by supplementary-table number to avoid stale multi-figure numbering. They include the full spatial permutation table, block-bootstrap confidence intervals, cross-cancer program-family signature table, MAZ quality-control table, hero-patch metadata, spatial-sensitivity summary, component-gene audit tables, registration-perturbation tables and nested spatial holdout tables. No IHC or protein validation is claimed.
