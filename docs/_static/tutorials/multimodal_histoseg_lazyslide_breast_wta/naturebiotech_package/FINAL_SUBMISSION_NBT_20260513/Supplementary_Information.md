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

### Boundary co-variation analysis

Boundary profiles were computed in distance rings around selected tissue interfaces. Candidate molecularly active zones were interpreted only as ring-level H&E-WTA co-variation. Causal, temporal or directional boundary interpretations were not assigned.

### Source-data mapping

| File | Contents |
|---|---|
| `Figure_1b_Hero_Patches_Source_Data.csv` | contour IDs, WTA z-scores, oriented H&E embedding z-scores and contour-size metadata for QC-filtered breast S3 hero patches |
| `Figure_1c_Spatial_Permutation_Source_Data.csv` | spatial-null permutation results for reported candidate programs |
| `Figure_1c_BlockBootstrap_Source_Data.csv` | spatial block-bootstrap confidence intervals |
| `Figure_1d_MAZ_QC_Source_Data.csv` | conservative boundary co-variation summaries used for the MAZ panel and Supplementary Fig. 3 |
| `Figure_1e_CrossCancer_Signature_Source_Data.csv` | program-family signature summary across breast/cervical and PLIP/UNI |

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

| Pair | Polarity | Contour ID | WTA program z-score | Oriented H&E embedding z-score | n_tiles | n_cells |
|---:|---|---|---:|---:|---:|---:|
| 1 | high | S3 S3 #417.1 | 1.03 | 0.78 | 7 | 508 |
| 1 | low | S3 S3 #519.1 | -0.82 | -1.06 | 3 | 211 |
| 2 | high | S3 S3 #139.1 | 0.52 | 1.28 | 6 | 328 |
| 2 | low | S3 S3 #56.1 | -0.68 | -0.59 | 6 | 120 |
| 3 | high | S3 S3 #351.1 | 0.82 | 0.92 | 10 | 138 |
| 3 | low | S3 S3 #188.1 | -0.83 | -0.93 | 6 | 59 |
| 4 | high | S3 S3 #173.1 | 1.74 | 0.43 | 5 | 566 |
| 4 | low | S3 S3 #285.1 | -0.61 | -0.84 | 3 | 78 |

### Supplementary Table 9. Contour-size sensitivity

The machine-readable table contains full breast and cervical PLIP/UNI candidate results under the full set, n_tiles-only, n_cells-only and combined size filters. The compact view below shows the primary breast S3 PLIP programs under the full set and the combined n_tiles >= 3 and n_cells >= 50 filter.

| Program | Filter | n contours | Recomputed partial Spearman's rho |
|---|---|---:|---:|
| luminal_estrogen_response | full set | 157 | -0.652 |
| luminal_estrogen_response | n_tiles >= 3; n_cells >= 50 | 49 | -0.656 |
| unfolded_protein_response | full set | 157 | 0.523 |
| unfolded_protein_response | n_tiles >= 3; n_cells >= 50 | 49 | 0.672 |
| oxidative_phosphorylation | full set | 157 | 0.536 |
| oxidative_phosphorylation | n_tiles >= 3; n_cells >= 50 | 49 | 0.647 |

## Supplementary Figure Captions

**Supplementary Fig. 1 | Spatial permutation defense.** Program residuals were permuted within compartment-aware strata defined by spatial-omics-derived contour labels, centroid-position bins and boundary-distance bins. **A,** observed residual associations after structure/spatial residualization. **B,** observed absolute partial correlations compared with the 95th percentile of the stratified spatial null. This test mitigates coarse spatial-autocorrelation explanations but does not exclude all fine-scale spatial dependence.

**Supplementary Fig. 2 | Expanded breast S3 luminal estrogen-response examples.** High- and low-program S3 contours are shown as expanded H&E examples for Fig. 1b. All displayed patches pass n_tiles >= 3, n_cells >= 50, edge-proximity and area QC filters. They share the same spatial-omics-derived contour label and were selected as representative concordant examples after statistical ranking; they are not a blinded visual diagnostic claim.

**Supplementary Fig. 3 | Expanded boundary co-variation profiles.** Ring-level profiles show H&E-WTA co-variation at selected tissue interfaces. These examples illustrate conservative boundary co-variation and do not imply causality, temporal ordering or directional boundary effects.

## Source Data

Source Data files accompany Fig. 1 and Supplementary Figs. 1-3. The upload package names these files by the relevant Fig. 1 panel to avoid stale multi-figure numbering. They include the full spatial permutation table, block-bootstrap confidence intervals, cross-cancer program-family signature table, MAZ quality-control table and hero-patch metadata. No IHC or protein validation is claimed.
