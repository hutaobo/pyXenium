# pyXenium.pathway morphopathway evidence notes

Run directory: `D:\GitHub\pyXenium\benchmarking\morphopathway_atera\results\plip_smoke_3000_d64_seed29_bins12_null12_20260512_1924`

## Current claim boundary

This evidence package supports a conservative H&E+WTA pathway-family stress test. It does not support a direct cervical replication claim for the top breast pathway because recovery is evaluated at pathway and family level, and breast discovery effect sizes remain modest.

## Inputs and scale

- Breast discovery observations: 133 spatial blocks from 3000 sampled cells.
- Cervical validation observations: 137 spatial blocks from 3000 sampled cells.
- H&E features: aligned low-resolution image pyramid sampled around Xenium cell centroids, then averaged into spatial blocks.
- Breast H&E embedding backend status: `transformers_clip:vinid/plip`; cervical H&E embedding backend status: `transformers_clip:vinid/plip`.
- WTA features: curated pathway activity scores from Xenium `cell_feature_matrix.h5`.

## Key results

- Breast top association: `unfolded_protein_response` / `embedding__plip_028`, abs partial Spearman rho = 0.2775.
- Cervical top association: `epithelial_identity` / `embedding__plip_007`, abs partial Spearman rho = 0.3293.
- Cross-cancer pathway/family recovery: 9/10.
- Pathway coverage: breast 10/10; cervical 10/10.

## Statistics

Associations are residual partial Spearman correlations after adjustment for coarse spatial structure, x/y coordinate ranks, boundary distance, and log total counts. Spatial nulls permute residual pathway activity within spatial strata. Negative controls use expression-matched random gene sets for the same image feature. Current smoke settings use limited permutations and negative controls, so p-values are gate checks rather than final inferential values.

## Residual risks

- H&E descriptors are deterministic color/texture/projection features in this run, not PLIP/UNI foundation-model embeddings unless a manifest explicitly records a PLIP/UNI backend.
- Breast signal is stable enough for smoke testing but too modest for a direct discovery claim.
- Cervical validation should be described as pathway-family stress testing unless stronger H&E embeddings recover direct pathway signals.
