# pyXenium.pathway morphopathway evidence notes

Run directory: `D:\GitHub\pyXenium\benchmarking\morphopathway_atera\results\he_texture_block_6k_smoke_20260512_1758`

## Current claim boundary

This evidence package supports a conservative H&E+WTA pathway-family stress test. It does not support a direct cervical replication claim for the top breast pathway because recovery is evaluated at pathway and family level, and breast discovery effect sizes remain modest.

## Inputs and scale

- Breast discovery observations: 343 spatial blocks from 6000 sampled cells.
- Cervical validation observations: 365 spatial blocks from 6000 sampled cells.
- H&E features: aligned low-resolution image pyramid sampled around Xenium cell centroids, then averaged into spatial blocks.
- WTA features: curated pathway activity scores from Xenium `cell_feature_matrix.h5`.

## Key results

- Breast top association: `unfolded_protein_response` / `image__he_hematoxylin_proxy`, abs partial Spearman rho = 0.1345.
- Cervical top association: `collagen_ecm_organization` / `image__he_b_mean`, abs partial Spearman rho = 0.2722.
- Cross-cancer pathway/family recovery: 5/10.
- Pathway coverage: breast 10/10; cervical 10/10.

## Statistics

Associations are residual partial Spearman correlations after adjustment for coarse spatial structure, x/y coordinate ranks, boundary distance, and log total counts. Spatial nulls permute residual pathway activity within spatial strata. Negative controls use expression-matched random gene sets for the same image feature. Current smoke settings use limited permutations and negative controls, so p-values are gate checks rather than final inferential values.

## Residual risks

- H&E descriptors are deterministic color/texture features, not PLIP/UNI embeddings.
- Breast signal is stable enough for smoke testing but too modest for a direct discovery claim.
- Cervical validation should be described as pathway-family stress testing unless stronger H&E embeddings recover direct pathway signals.
