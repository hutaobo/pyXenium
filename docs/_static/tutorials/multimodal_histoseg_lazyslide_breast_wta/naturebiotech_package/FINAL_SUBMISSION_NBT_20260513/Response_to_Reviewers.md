# Response to Likely Reviewers v2

## Concern 1: The result may be driven by spatial autocorrelation.

Response: We added stratified spatial permutation and spatial block bootstrap. Molecular residuals are shuffled within structure, centroid and boundary-distance strata after rank residualization. 13 candidate associations pass the 95% spatial-null gate. This does not prove causality, but it shows the strongest results exceed coarse spatial-position effects.

## Concern 2: This is a small cohort.

Response: We frame breast WTA as the discovery dataset and cervical Atera WTA as a cross-cancer stress test in a second epithelial cancer context. Cervical is not a direct luminal/ER replication and does not establish population-level generalization. It tests whether the Contour-constrained residual decoding workflow can recover stromal, immune and invasion program families in another tissue context.

## Concern 3: Why no IHC/protein validation?

Response: We make no protein-level claim. Atera WTA provides an 18,000-gene discovery view that a single IHC marker cannot replace, but transcript evidence does not substitute for protein validation. Matched IHC is a next experiment, not a claimed result.

## Concern 4: Are PLIP and UNI dimensions interpretable?

Response: No individual embedding dimension is assigned a universal meaning. Robustness is defined at the WTA program-family level. PLIP and UNI may rotate or flip latent axes, so model-agnostic agreement is recovery of the same program, not matching coordinate sign.

## Concern 5: Does MAZ imply causality or directional boundary effects?

Response: No. MAZ is presented as conservative boundary coupling and hypothesis generation. The manuscript should state that morphology and WTA gradients co-localize at selected interfaces, not that either causes or temporally precedes the other.

## Concern 6: Are the HistoSeg contours independent morphology labels?

Response: No. This is now stated explicitly. The HistoSeg contours used here are generated from Atera WTA cell-coordinate and cluster/cell-group information, not from H&E image segmentation. This creates a non-independence/circularity concern that the manuscript treats directly. The label is discrete and is included as a covariate, or held constant in the breast S3 analysis. The reported associations are continuous residual program associations after label and spatial covariates are fixed. Thus, mTM does not show that H&E rediscovers HistoSeg; it shows that H&E maps residual molecular position within HistoSeg-defined classes. We therefore do not frame mTM as a standalone morphology-only predictor; we frame it as a conservative paired H&E-WTA residual decoding test.
