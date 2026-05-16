# Foundation-model morphology maps residual programs in spatial-omics contours

Taobo Hu

Science for Life Laboratory, Department of Biochemistry and Biophysics, Stockholm University, Stockholm, Sweden.

Correspondence: taobo.hu@scilifelab.se

## Abstract

Spatial whole-transcriptome (WTA) imaging defines molecular tissue contours, but H&E morphology may retain unresolved molecular information within them. We introduce morphomolecular translation mapping (mTM), which tests whether H&E foundation-model embeddings map residual WTA programs after contour labels and spatial covariates are fixed. In breast and cervical WTA data, mTM nominates endocrine, stress, stromal, immune and boundary-associated programs exceeding compartment-aware spatial nulls.

Spatial biology is moving from sparse assays to high-resolution transcriptome maps that preserve tissue context<sup>1-5</sup>. In parallel, H&E foundation models and histology-spatial-omics systems now support visual-language pathology, whole-slide representation learning and expression-morphology integration<sup>6-16</sup>. Most work asks whether H&E can reconstruct expression, spatial domains or downstream labels. We asked a more conservative question: after a spatial-omics workflow has already defined molecular-spatial tissue contours, do H&E foundation-model embeddings still map WTA variation that those contour labels leave unresolved? This shifts the benchmark from expression prediction to residual biological decoding. It also avoids treating square tiles or whole slides as the primary biological unit when the available spatial-omics data already define irregular, interpretable tissue regions. That distinction matters because an apparent H&E-WTA association can otherwise reflect region identity, slide geography or boundary structure rather than molecular variation within a fixed tissue compartment. We therefore treat contour labels and spatial covariates as structure to be conditioned on, not as labels to be predicted.

We developed morphomolecular translation mapping (mTM), a contour-constrained residual decoding framework (Fig. 1). HistoSeg contours were generated from Atera WTA cell-coordinate and cluster information, not from H&E image segmentation. The same polygons then served as the shared geometry<sup>18,19</sup> for direct-WSI LazySlide PLIP or UNI H&E embeddings<sup>12,14,17</sup>, Atera WTA program summaries and residual statistical testing. This design choice is central to the claim. The contours are therefore not independent histological labels; the design is conservative rather than circular in the claim being tested because the spatial-omics contour label is treated as a discrete covariate, whereas mTM asks whether H&E embeddings map continuous WTA program variation that remains within that label. mTM is not a standalone morphology-only predictor; it is a paired H&E-WTA framework that tests whether morphology contributes residual information after coarse WTA-derived spatial structure, centroid position and boundary-distance covariates are controlled. The output is a ranked set of program-feature pairs, effect sizes and spatial-null checks, rather than a deployed diagnostic classifier. The contour scale is intentionally intermediate: it is larger than a single tile, retains local tissue context and is still small enough to expose within-label molecular heterogeneity. Using contours as units also makes the statistical unit explicit: one row corresponds to one registered polygon with matched image and WTA summaries, so permutation and bootstrap procedures operate on biologically meaningful regions instead of thousands of partially overlapping tiles. This keeps the evidence scale aligned with the biological question.

In breast WTA, the strongest evidence came from S3 contours, which define a molecular-spatial class but are not molecularly homogeneous. mTM places each S3 contour along residual continua of endocrine, stress and metabolic state. Within S3, PLIP-derived morphology was associated with residual luminal estrogen-response variation (partial Spearman's rho -0.639, P = 4.79 × 10<sup>-19</sup>, n = 157), unfolded-protein-response variation (partial Spearman's rho 0.515, P = 7.91 × 10<sup>-12</sup>) and oxidative-phosphorylation variation (partial Spearman's rho 0.531, P = 1.38 × 10<sup>-12</sup>). These values summarize contour-level residual associations and are not per-cell or per-tile measurements. The direction of the image axis is arbitrary, so the biological statement is not that one embedding coordinate has a universal meaning. Rather, contours sharing the same spatial-omics-derived label retained continuous endocrine, stress and metabolic WTA variation that was mapped by H&E foundation-model features. The luminal estrogen-response result is the clearest discovery example because it occurs after restricting to one discrete contour label and after spatial covariate adjustment. Hero patches in Fig. 1b are therefore illustrative examples selected after statistical ranking, not a blinded visual diagnostic claim.

We next tested whether these associations were simply consequences of spatial autocorrelation. Molecular residuals were permuted within strata preserving spatial-omics-derived contour label, centroid-position bins and boundary-distance bins. The breast luminal estrogen-response, unfolded-protein-response and oxidative-phosphorylation associations remained stronger than the compartment-aware null distributions, each with empirical P = 9.999 × 10<sup>-5</sup>. Across breast and cervical analyses, 13 candidate associations passed the 95% spatial-null gate, and the strongest candidates also exceeded the 99% null threshold. This does not remove all fine-scale spatial dependence, but it makes the main claim stricter than a standard contour-level correlation. A cervical Atera WTA dataset served as a cross-cancer stress test rather than a direct replication of the breast luminal phenotype. It was used to ask whether the same residual-decoding procedure could nominate plausible program families in a different epithelial cancer, not whether breast-specific luminal biology would recur. In that second epithelial cancer context, residual decoding nominated stromal-remodeling, immune-ecology and invasion-associated programs, including myofibroblast CAF activation, EMT/invasive-front, TLS-adjacent activation, collagen/ECM organization and immune exclusion; these also exceeded the same spatial-null threshold. PLIP and UNI agreement was evaluated only at the program-family level, because embedding axes can rotate or change sign between models. Thus, the cervical result is consistent with applying the residual-decoding workflow to a different epithelial cancer setting, while cohort-level generality remains a future test.

Finally, ring-level boundary profiles nominated candidate molecularly active zones, defined here as interfaces where H&E and WTA gradients co-varied at selected tissue boundaries. We use this language deliberately. The current data support conservative boundary co-variation, not causality, temporal ordering or a directional mechanism. Similarly, Atera WTA provides an 18,028-gene discovery layer<sup>20</sup>, but transcript evidence does not replace protein validation. The correct translational interpretation is prospective: mTM nominates morphology-associated molecular states and boundary ecologies for validation in larger cohorts and, where clinically relevant, by IHC or orthogonal molecular assays. The study is deliberately validation-limited; its claim is methodological and residual, and the source data expose the reported associations for review and reuse. Its immediate contribution is a rigorous residual test showing that foundation-model morphology can add information within spatial-omics-defined contours rather than merely reproduce those contours. More broadly, mTM reframes H&E-spatial-omics integration as a controlled interrogation of what remains after molecular-spatial labels are fixed. That framing should be useful for future whole-transcriptome imaging studies in which contour labels are available but within-contour functional states remain unresolved.

## References

1. Ståhl, P.L. et al. Science 353, 78-82 (2016).
2. Chen, K.H., Boettiger, A.N., Moffitt, J.R., Wang, S. & Zhuang, X. Science 348, aaa6090 (2015).
3. Rodriques, S.G. et al. Science 363, 1463-1467 (2019).
4. Eng, C.H.L. et al. Nature 568, 235-239 (2019).
5. Rao, A., Barkley, D., França, G.S. & Yanai, I. Nature 596, 211-220 (2021).
6. He, B. et al. Nat. Biomed. Eng. 4, 827-834 (2020).
7. Zhang, D. et al. Nat. Biotechnol. 42, 1372-1377 (2024).
8. Jaume, G. et al. Adv. Neural Inf. Process. Syst. 37 (2024).
9. Huang, T., Liu, T., Babadi, M., Ying, R. & Jin, W. npj Digit. Med. 8, 659 (2025).
10. Liu, T., Huang, T., Ding, T. et al. Nat. Biomed. Eng. https://doi.org/10.1038/s41551-025-01602-6 (2026).
11. Yang, C. et al. Preprint at arXiv https://doi.org/10.48550/arXiv.2507.06418 (2025).
12. Huang, Z., Bianchi, F., Yuksekgonul, M. et al. Nat. Med. 29, 2307-2316 (2023).
13. Lu, M.Y., Chen, B., Williamson, D.F.K. et al. Nat. Med. 30, 863-874 (2024).
14. Chen, R.J., Ding, T., Lu, M.Y. et al. Nat. Med. 30, 850-862 (2024).
15. Xu, H., Usuyama, N., Bagga, J. et al. Nature 630, 181-188 (2024).
16. Vorontsov, E., Bozkurt, A., Casson, A. et al. Nat. Med. 30, 2924-2935 (2024).
17. Zheng, Y., Abila, E., Chrenková, E. et al. Nat. Methods 23, 728-731 (2026).
18. Marconato, L., Palla, G., Yamauchi, K.A. et al. Nat. Methods 22, 58-62 (2025).
19. Palla, G. et al. Nat. Methods 19, 171-178 (2022).
20. 10x Genomics. Atera In Situ preview datasets: FFPE human breast cancer and FFPE human cervical cancer. https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer; https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer (accessed 16 May 2026).

## Figure legend

### Figure 1

Fig. 1 | Contour-constrained residual decoding maps WTA programs from H&E morphology. a, HistoSeg contours are generated from Atera WTA cell-coordinate and cluster information, then used as the shared geometry for H&E LazySlide PLIP/UNI embeddings and Atera WTA program summaries. b, Breast S3 luminal estrogen-response examples show high- and low-program contours under the same spatial-omics-derived contour label; labels report WTA program z-score and oriented H&E embedding z-score. c, Top residual programs exceed compartment-aware spatial-null thresholds after preserving contour label, centroid and boundary-distance bins. d, Candidate molecularly active zone examples show ring-level H&E-WTA boundary co-variation. e, Cross-cancer candidate program-family signatures summarize PLIP and UNI candidates in breast and cervical WTA; dot area reports maximum absolute partial Spearman's rho and text reports the top WTA program.

## Declarations

### Acknowledgements and funding

The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.

### Author contributions

T.H. conceived the study, implemented the analysis workflow, performed the computational analyses, generated the figures and source data, and wrote the manuscript.

### Competing interests

The author declares no competing interests.

### Data availability

Source data for Fig. 1 and Supplementary Figs. 1-3 are included with the submission package. Raw 10x Genomics Atera WTA and H&E input files are not redistributed in this repository and should be obtained from the original public/vendor preview dataset pages for the breast and cervical samples.

### Code availability

The mTM analysis and packaging code is available in pyXenium v0.4.6 at https://github.com/hutaobo/pyXenium/releases/tag/v0.4.6. The exact initial-submission manuscript package is archived under the repository release `nbt-initial-submission-20260516` at https://github.com/hutaobo/pyXenium/releases/tag/nbt-initial-submission-20260516. Manuscript-level replication scripts are provided under `manuscript/mtm_wta_nbt_replication/`. HistoSeg contour-generation software is maintained separately at https://github.com/hutaobo/HistoSeg.

### Ethics and data-use statement

This computational analysis used public/vendor example spatial WTA and H&E data. The study did not involve new human sample collection, intervention or generation of new identifiable participant data.
