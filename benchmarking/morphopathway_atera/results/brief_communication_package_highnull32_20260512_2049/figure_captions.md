# Figure Captions

## Figure 1. Breast discovery morphopathway associations from H&E-derived PLIP embeddings and Xenium WTA pathway activity.

Source table: `main_figure_1_source_breast_discovery_highnull32.csv`.

This figure should show the top breast discovery associations between PLIP-derived H&E embedding dimensions and curated WTA pathway activity scores after residual adjustment for spatial and cell-count covariates. Panels should include the pathway family, image feature, absolute partial Spearman rho, spatial-null gate status, and matched negative-control gate status. The caption should state that associations are discovery signals and are not interpreted as direct biomarkers.

## Figure 2. Cross-cancer pathway-family stress-test stability across high-null seeds.

Source table: `main_figure_2_source_cross_cancer_stability_highnull32.csv`.

This figure should summarize pathway-family recovery from breast discovery to cervical validation across three high-null seeds. The main visual should distinguish the stable 9-pathway core from unstable pathways, with primary and axis-masked recovery shown side by side. The caption should report recovery of 9/10 to 10/10, axis-masked recovery of 9/10 to 10/10, and the exclusion of `emt_invasive_front` from the stable core.

## Extended Data Figure 1. Spatial pseudobulk and null-control sensitivity.

Source tables: `supp_table_spatial_block_and_seed_summary.csv` and `supp_table_highnull32_gate_and_axis_masked_summary.csv`.

This figure should document the 12 x 12 spatial pseudobulk setting, retained block counts (breast 133-136; cervical 135-137), spatial null pass95/pass99 counts, and matched negative-control pass95/pass99 counts. The caption should explicitly identify negative-control pass95 as the remaining limitation.

## Extended Data Figure 2. Candidate generic PLIP axes and axis-masked recovery.

Source table: `supp_table_plip_axis_diagnostics_by_run.csv`.

This figure should show sample-specific candidate generic PLIP axes flagged by repeated top-40 usage plus matched-control failure. The caption should state that candidate axes occurred in 2/3 runs, and that removing them left cross-cancer recovery unchanged at 9/10 to 10/10.
