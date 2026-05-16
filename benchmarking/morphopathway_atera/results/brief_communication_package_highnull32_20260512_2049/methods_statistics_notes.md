# Methods And Statistics Notes

Inputs:
- Breast discovery: Atera Xenium WTA FFPE breast preview output.
- Cervical validation: Atera Xenium WTA FFPE cervical preview output.
- H&E backend: `transformers_clip:vinid/plip` for all three high-null runs.

Workflow:
- Sampled 3,000 cells per dataset and encoded H&E patches with PLIP at 64 output dimensions.
- Aggregated cells into a 12 x 12 spatial pseudobulk grid with at least 6 cells per retained block.
- Scored curated WTA pathway panels and fit residual partial Spearman associations after covariate adjustment.
- Used 32 spatial permutations and 32 expression-matched random gene-set controls in the high-null runs.

Evidence gates:
- Cross-cancer pathway/family recovery range: 9/10 to 10/10.
- Axis-masked recovery range: 9/10 to 10/10.
- Runs with candidate generic PLIP axes: 2/3.

Residual risks:
- Matched negative-control pass95 is imperfect in seed17 and seed29, mainly on the breast side.
- The cervical result should remain a pathway-family stress test rather than a direct replication claim.
- This package is based on sampled public preview datasets; full-scale runs and independent cohorts are still needed.
