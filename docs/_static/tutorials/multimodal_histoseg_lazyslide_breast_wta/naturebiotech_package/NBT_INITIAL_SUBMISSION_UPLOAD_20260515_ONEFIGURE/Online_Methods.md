# Online Methods

## Datasets and registration

Breast Atera WTA was used as the discovery dataset and cervical Atera WTA as a second epithelial cancer stress test. Each dataset was analyzed as paired spatial whole-transcriptome (WTA) and H&E whole-slide image data. Atera WTA coordinate metadata in the SpatialData object were used to register H&E-derived image features and WTA measurements into a shared contour coordinate system. The analysis did not redistribute raw 10x input files; source data tables for the reported figures are provided with the submission package.

## Contour provenance

HistoSeg contours were generated from Atera WTA spatial-transcriptomics inputs, including cell-coordinate tables and cluster or cell-group definitions. The contour workflow corresponds to the `histoseg.contour` functionality in [HistoSeg](https://github.com/hutaobo/HistoSeg), which supports contour synthesis from spatial cell coordinates, clustered cell labels and spatial-transcriptomics contour exports. HistoSeg also contains separate H&E image-segmentation functionality, but that functionality was not used to draw the breast S1-S5 contours or the cervical tissue contours in this study. H&E images were not used to draw contours. H&E images entered only after contour generation, through assignment of LazySlide image features to the registered contour geometry. This provenance makes the test conservative: the contour label already captures coarse molecular-spatial tissue structure, and mTM asks whether H&E morphology contributes residual WTA information after that structure has been controlled.

## H&E WSI embedding

The original H&E OME-TIFF was converted into a tiled pyramidal BigTIFF readable by the direct WSI workflow. LazySlide was run on the converted WSI with PLIP for the primary analysis and UNI for model-sensitivity analysis where available. Tile-level image embeddings were assigned to HistoSeg contours using registered spatial geometry, and contour-level image summaries were computed from the tiles assigned to each polygon. No H&E-derived contour segmentation was introduced during this step. The direct WSI and contour aggregation code is implemented in [pyXenium](https://github.com/hutaobo/pyXenium), primarily through the `pyXenium.multimodal` workflow and manuscript-level replication scripts in the repository.

## WTA program scoring

Atera WTA measurements were summarized within the same contour polygons used for image-feature aggregation. Program scores were assembled for biologically interpretable families including endocrine/epithelial identity, metabolic/stress, stromal-remodeling/CAF/ECM, immune ecology/TLS/immune exclusion and invasion/boundary/EMT. Breast analyses focused on luminal estrogen response, unfolded protein response and oxidative phosphorylation in S3 contours. Cervical analyses tested stromal, immune and invasion programs as a cross-cancer stress test. Program scores were treated as continuous contour-level WTA readouts rather than as categorical labels.

For component-gene audit analyses, contour-level normalized means were recomputed for the genes comprising each selected breast S3 WTA program. The reported H&E embedding axis was then tested against each component gene using the same rank-residualized partial Spearman framework and the same spatial covariates. These analyses were used as transcript-level sanity checks only; they are not independent IHC or protein validation.

## Contour-level aggregation

For contour \(i\), let \(T_i\) be the set of LazySlide tiles assigned to that contour and \(G_i\) the cells or transcripts assigned to the same contour polygon. H&E image features were summarized as contour-level means or related summaries across \(T_i\), and WTA program scores were summarized across \(G_i\). These paired summaries define a contour-level table in which each row has the same geometry, the same spatial-omics-derived contour label and matched H&E/WTA measurements.

## Non-independence of contour construction and residual estimand

The HistoSeg contours are not independent histological labels. We therefore define the estimand as a within-label residual association rather than an independent label-prediction task. For contour \(i\), let \(C_i\) denote the spatial-omics-derived discrete contour label, \(S_i\) denote spatial covariates including centroid coordinates and boundary-distance bins, \(X_{ik}\) denote H&E foundation-model feature \(k\), and \(Y_i\) denote a continuous WTA program score. The covariate design matrix is

\[
Z_i = [1,\mathrm{onehot}(C_i),S_i].
\]

For breast S3 analyses, \(C_i\) is constant by design, so the contour-label one-hot term is omitted. Rank-transformed image and WTA variables are residualized against \(Z\):

\[
r^X_k = (I - P_Z)\,\mathrm{rank}(X_k), \qquad
r^Y = (I - P_Z)\,\mathrm{rank}(Y),
\]

where \(P_Z = Z(Z^\top Z)^{-1}Z^\top\) is the projection matrix. The reported residual association is

\[
\rho_k = \mathrm{cor}(r^X_k,r^Y),
\]

computed as a Spearman correlation after rank residualization. This design cannot establish standalone morphology-only prediction and does not exclude every form of contour-construction-induced coupling, but it directly tests whether H&E features map continuous WTA program variation beyond simply rediscovering the discrete HistoSeg label.

## Residual decoding and program selection

For each candidate WTA program, image features were screened by absolute residual association within the relevant contour set. The primary breast results were restricted to S3 contours and reported the strongest luminal estrogen-response, unfolded-protein-response and oxidative-phosphorylation associations. Cervical analyses used the same residual-decoding logic but interpreted positive findings as a cross-cancer stress test rather than direct replication of the breast luminal phenotype. Individual PLIP or UNI embedding dimensions were not assigned universal biological meaning because latent axes can rotate, flip or change scale across foundation models.

## Spatial permutation and block bootstrap

For the spatial-null defense, molecular residuals were permuted within strata preserving contour label, centroid-position bins and boundary-distance bins. This retains coarse compartmental and spatial organization while breaking contour-wise H&E-to-WTA pairing. For \(B\) permutations, empirical two-sided enrichment was calculated as

\[
P_{\mathrm{emp}} =
\frac{1 + \sum_{b=1}^{B} \mathbf{1}(|\rho^{\mathrm{null}}_b| \ge |\rho_{\mathrm{obs}}|)}
{1 + B}.
\]

The permutation mitigates coarse spatial-autocorrelation explanations but does not exclude all fine-scale spatial dependence, registration uncertainty or biological coupling induced by the original spatial-omics contour construction. Spatial block bootstrap resampled centroid x/y spatial blocks and recomputed \(\rho\) to estimate 95% confidence intervals. Additional sensitivity checks recomputed candidate associations after leaving out each spatial block, compared observed pairings with local mismatched-pair controls and jittered centroid covariates by up to 1% of slide span.

For registration perturbation, existing tile-level embeddings were reassigned to contour polygons after 23 small coordinate perturbations: x/y translations of 8-32 pixels, rotations of 0.25-0.5 degrees, 0.5% scale changes and mild combined shifts. Candidate contour-level embedding means were recomputed after each reassignment and tested against the same WTA program values and covariates. This test probes whether the reported associations are brittle to small tile-to-contour registration changes without rerunning the H&E encoder.

For nested spatial holdout, contours were partitioned into a 4 by 4 centroid grid. In each fold, the embedding dimension was selected using only the training spatial blocks, and the selected feature and the locked manuscript feature were then evaluated on the held-out block. This stricter check is reported as fold-level sign stability, not as a new discovery screen, because small held-out blocks can be noisy for weaker cross-cancer/model stress-test candidates.

## Boundary co-variation

Boundary profiles were computed in distance rings around selected tissue interfaces. For each selected program-feature pair, WTA program values and oriented H&E feature values were summarized across signed distance bins from contour boundaries. Candidate molecularly active zones were described only as ring-level H&E-WTA co-variation. No causal, temporal or directional boundary interpretation is claimed.

## Software and reproducibility

HistoSeg contour-generation software is maintained at [https://github.com/hutaobo/HistoSeg](https://github.com/hutaobo/HistoSeg). The mTM analysis, direct WSI embedding, contour aggregation, statistical defense, figure composition and submission-package preparation are implemented in [pyXenium](https://github.com/hutaobo/pyXenium). Manuscript-level full replication scripts are provided under `manuscript/mtm_wta_nbt_replication/` in the pyXenium repository. They define the expected raw input paths, run order, GPU/Linux commands, local packaging commands and expected output files.

## Data availability

Source data are provided for Fig. 1 panels, spatial permutation defense, block-bootstrap summaries, cross-cancer program-family summaries, MAZ quality-control table, hero-patch metadata, spatial-sensitivity summaries, component-gene audits, registration-perturbation tests and nested spatial holdout tests. Raw 10x Genomics Atera WTA and H&E input files are not redistributed by this manuscript package and should be obtained from the original public/vendor example dataset source pages. No IHC or protein-validation data were generated for this study.

## Code availability

The mTM analysis code is available in pyXenium v0.4.6 at [https://github.com/hutaobo/pyXenium/releases/tag/v0.4.6](https://github.com/hutaobo/pyXenium/releases/tag/v0.4.6). The registration/nested-holdout hardened initial-submission manuscript package is archived under the repository release `nbt-initial-submission-20260517-registration-nested-hardening` at [https://github.com/hutaobo/pyXenium/releases/tag/nbt-initial-submission-20260517-registration-nested-hardening](https://github.com/hutaobo/pyXenium/releases/tag/nbt-initial-submission-20260517-registration-nested-hardening). Manuscript-level full replication scripts are provided under `manuscript/mtm_wta_nbt_replication/`. HistoSeg contour-generation software is maintained separately in HistoSeg.

## Large language model assistance

OpenAI Codex was used to assist with manuscript-package formatting, editorial consistency checks and automated QA scripting. The author reviewed the final text, analyses and claims, and takes responsibility for the submitted work.
