# Reviewer Evidence Audit

Status: **pass**

## Key Numbers

- Cross-cancer recovery: 9/10 to 10/10
- Axis-masked recovery: 9/10 to 10/10
- Stable core: 9 pathways
- Matched negative-control pass95 minima: breast 8/10; cervical 9/10

## Audit Table

| Evidence item | Status | Key numbers | Residual risk | Next action |
| --- | --- | --- | --- | --- |
| Primary claim boundary | pass | stable_core=9 pathways | Claim must remain a pathway-family stress test, not direct cervical replication. | Preserve conservative wording in manuscript title, abstract, and figure captions. |
| Cross-cancer recovery gate | pass | 9/10 to 10/10 | Cervical result supports stress-test recovery rather than independent direct replication. | Report range and per-run source table together. |
| Axis-masked PLIP sensitivity | pass | 9/10 to 10/10; candidate_axis_runs=2 | Two runs contain candidate generic PLIP axes; sensitivity is supportive but smoke-scale. | Keep axis-masked result adjacent to the primary gate in the figure or supplement. |
| Stable 9-pathway core definition | pass | manifest_core=9; figure2_core=9 | The core is a reproducible pathway-family set, not a ranked clinical signature. | Use the exact stable-core list from the manifest. |
| EMT/invasive-front exclusion | pass | emt_invasive_front_stable=False | Tempting biological story, but current evidence is not stable enough. | Mention as non-core or omit from primary claims. |
| Matched negative-control limitation | pass | breast_min=8/10; cervical_min=9/10 | Breast negative-control pass95 minimum is below 10/10; this must be called a limitation. | Surface this in limitations and statistical notes. |
| Figure and supplementary source tables | pass | main_figure_1_source_breast_discovery_highnull32.csv=30; main_figure_2_source_cross_cancer_stability_highnull32.csv=10; source_table_cross_cancer_validation_by_run.csv=60; supp_table_cervical_validation_best_associations.csv=30; supp_table_highnull32_gate_and_axis_masked_summary.csv=3; supp_table_plip_axis_diagnostics_by_run.csv=384; supp_table_spatial_block_and_seed_summary.csv=3 | Large PLIP axis diagnostic table should stay supplementary. | Use source tables directly for figure construction; avoid hand-edited plot data. |
| Methods/statistics traceability | pass | notes=3/3 present | Final manuscript still needs exact software/hardware details. | Copy methods notes into the draft and fill version/hardware blanks. |
| Archive reproducibility | pass | archive_sidecar_match=true | Archive changes when new audit files are intentionally added. | Re-run archive generation after any package content change. |
