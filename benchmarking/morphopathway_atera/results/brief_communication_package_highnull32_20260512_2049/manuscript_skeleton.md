# Manuscript Skeleton

Working title:
H&E-to-WTA morphopathway stress testing in Xenium breast and cervical preview tissues

One-sentence pitch:
`pyXenium.pathway` links PLIP-derived H&E patch embeddings to curated Xenium WTA pathway activity and tests whether pathway-family morphology signals persist across breast discovery and cervical validation tissues.

## Abstract Draft

Spatial transcriptomics creates an opportunity to evaluate whether histological image features carry pathway-scale molecular information, but the statistical framing must separate exploratory morphology associations from direct replication claims. We developed `pyXenium.pathway`, a H&E+WTA morphopathway workflow that scores curated pathway programs from Xenium WTA profiles, aggregates PLIP-derived H&E embeddings into coarse spatial pseudobulk blocks, and evaluates residual morphology-pathway associations with spatial and matched random gene-set controls. Applied to Atera breast and cervical Xenium WTA FFPE preview datasets, three high-null seeds recovered a stable 9-pathway pathway-family stress-test core (9/10 to 10/10 recovered). Axis-masked sensitivity, removing sample-specific candidate generic PLIP axes, remained 9/10 to 10/10. Matched negative controls were imperfect in some breast-side runs, supporting a conservative claim: pathway-family stress-test recovery rather than direct cervical replication.

## Main Text Outline

### Motivation

- Xenium WTA allows morphology-to-pathway benchmarking using measured transcriptome-wide genes rather than targeted panels alone.
- The key methodological challenge is avoiding overclaiming direct cross-cancer replication from exploratory image embedding associations.
- The contribution is a reproducible workflow that combines curated pathway scoring, spatial pseudobulk aggregation, residual association testing, spatial nulls, matched random gene-set controls, and axis-masked sensitivity.

### Methodological Advance

- `pyXenium.pathway` builds curated pathway panels and pathway activity scores from Xenium cell-feature matrices.
- H&E patches are encoded with PLIP and averaged into 12 x 12 spatial blocks with at least 6 cells per retained block.
- Associations are residual partial Spearman correlations adjusted for structure, x/y ranks, boundary distance, and log total counts.
- High-null runs use 32 spatial permutations and 32 expression-matched random gene-set controls.

### Results

- Across 3 high-null seeds, cross-cancer pathway-family recovery ranged from 9/10 to 10/10.
- Axis-masked sensitivity remained 9/10 to 10/10, arguing that candidate generic PLIP axes are not required for the main stress-test signal.
- The stable core contains 9 pathways: unfolded_protein_response, immune_exclusion, luminal_estrogen_response, myofibroblast_caf_activation, oxidative_phosphorylation, basal_squamous_state, collagen_ecm_organization, immune_activation, epithelial_identity.
- `emt_invasive_front` is not part of the stable core and should be described as unstable (emt_invasive_front).
- Spatial null pass95 minima were breast 9/10 and cervical 10/10.
- Matched negative-control pass95 minima were breast 8/10 and cervical 9/10, which should be reported as the main limitation.

### Interpretation

- The evidence supports a pathway-family stress-test claim, not a direct pathway-level cervical replication claim.
- The strongest repeated families are represented by unfolded_protein_response, immune_exclusion, luminal_estrogen_response, myofibroblast_caf_activation among the stable core.
- The analysis should be framed as a method and evidence package for morphopathway benchmarking rather than a clinical biomarker study.

### Limitations

- Public preview datasets are sampled rather than exhaustive full-resolution cohorts.
- Matched negative controls do not pass 95% gates for every pathway/run, especially in breast-side PLIP axes.
- PLIP embeddings may include generic histology axes; these were diagnosed and masked in sensitivity analysis, but independent embedding backends remain future work.
- No patient-level outcome, diagnostic classifier, or causal mechanism is claimed.
