# Morphopathway Evidence-to-Claim Map

This handoff map links the conservative Brief Communication wording to the exact evidence files in the new `pyXenium.pathway` morphopathway suite. It is intentionally outside the archived final package so the validated archive hash remains unchanged.

## Evidence Package

- Final package: `benchmarking/morphopathway_atera/results/brief_communication_package_highnull32_20260512_2049`
- Archive: `benchmarking/morphopathway_atera/results/brief_communication_package_highnull32_20260512_2049.zip`
- Archive SHA256: `db731299997fbdbadab9d4eccba981dfe49330e457bdc907a93b6b3d0d1acf54`
- Reviewer audit: `reviewer_evidence_audit.json`, `reviewer_evidence_audit.csv`, `reviewer_evidence_audit.md`
- Package QC: `package_qc_report.json`, `package_qc_report.md`

## Conservative Claim Map

| Manuscript claim | Evidence support | Primary source files | Gate or check | Allowed wording | Do not claim |
| --- | --- | --- | --- | --- | --- |
| PLIP-derived H&E embeddings aggregated to coarse spatial pseudobulk blocks recover a stable pathway-family stress-test core. | `main_figure_2_source_cross_cancer_stability_highnull32.csv` has 10 pathway rows, with 9 pathways marked as the stable core and recovered in 3/3 high-null runs. | `main_figure_2_source_cross_cancer_stability_highnull32.csv`; `source_table_cross_cancer_validation_by_run.csv`; `brief_communication_package_manifest.json` | Cross-cancer recovery minimum `9/10`, maximum `10/10`; stable core count `9`. | "Stable 9-pathway pathway-family stress-test core across three high-null seeds." | "Full pathway replication" or "all pathways replicate." |
| Breast discovery H&E-WTA associations are suitable for a discovery figure. | `main_figure_1_source_breast_discovery_highnull32.csv` contains 30 top breast discovery rows across three high-null runs, including spatial-null and matched random gene-set control columns. | `main_figure_1_source_breast_discovery_highnull32.csv`; `supp_table_highnull32_gate_and_axis_masked_summary.csv`; `supp_table_spatial_block_and_seed_summary.csv` | Breast matched negative-control pass95 minimum is `8/10`; breast spatial-null pass95 is reported per seed. | "Discovery-scale H&E-WTA pathway associations passing predefined gate checks." | "Final inferential p-values", "clinical classifier", or "patient-level biomarker." |
| Cervical data are a cross-cancer pathway-family stress test, not direct replication. | `source_table_cross_cancer_validation_by_run.csv` separates discovery and validation labels and includes both pathway-level and family-level validation columns. `emt_invasive_front` is recovered in only 1/3 runs and is excluded from the stable core. | `source_table_cross_cancer_validation_by_run.csv`; `supp_table_cervical_validation_best_associations.csv`; `claim_wording.md` | Cervical matched negative-control pass95 minimum is `9/10`; stable core remains 9 pathways after excluding unstable EMT/invasive-front behavior. | "Cross-cancer pathway-family stress testing." | "Direct cervical replication of the top breast signal." |
| Axis-masked sensitivity is not dependent on candidate generic PLIP axes. | Axis-masked recovery remains `9/10` to `10/10` after sample-specific candidate generic PLIP axes are removed. | `supp_table_highnull32_gate_and_axis_masked_summary.csv`; `supp_table_plip_axis_diagnostics_by_run.csv`; `main_figure_2_source_cross_cancer_stability_highnull32.csv` | Axis-masked recovery minimum `9/10`, maximum `10/10`; candidate generic PLIP axes recorded by run. | "Axis-masked sensitivity preserved the pathway-family recovery range." | "PLIP axes are biologically specific without qualification." |
| The package is reproducible enough for handoff and review. | Reviewer audit status is pass, package QC status is pass, and archive/file manifests are present. | `reviewer_evidence_audit.*`; `package_qc_report.*`; `package_file_manifest.*`; `brief_communication_package_highnull32_20260512_2049.manifest.json` | Focused tests pass and archive hash is stable at `db731299997fbdbadab9d4eccba981dfe49330e457bdc907a93b6b3d0d1acf54`. | "Reproducible reviewed evidence package." | "Independent external validation" or "production clinical readiness." |

## Stable Core

The stable 9-pathway core is:

- `unfolded_protein_response`
- `immune_exclusion`
- `luminal_estrogen_response`
- `myofibroblast_caf_activation`
- `oxidative_phosphorylation`
- `basal_squamous_state`
- `collagen_ecm_organization`
- `immune_activation`
- `epithelial_identity`

`emt_invasive_front` is intentionally not part of the stable core because it was recovered in only 1/3 high-null runs.

## Residual Risks To Preserve In Wording

- The breast matched negative-control pass95 minimum is `8/10`; present this as a limitation, not a failure.
- The cervical matched negative-control pass95 minimum is `9/10`; this supports stress-test wording but not direct replication.
- Smoke-scale empirical nulls are gate checks. Avoid treating their p-values as final inferential statistics.
- PLIP axis diagnostics identify candidate generic axes in some runs, so axis-masked sensitivity must remain part of the claim support.
- The Atera preview datasets support method/evidence demonstration, not patient-level generalization.

## Reviewer-Ready Summary Sentence

The supported claim is that the new `pyXenium.pathway` morphopathway workflow links PLIP-derived H&E morphology embeddings with WTA pathway activity in Atera preview data, recovering a stable 9-pathway pathway-family stress-test core across three high-null seeds while preserving the result under axis-masked sensitivity; the evidence does not support direct cervical pathway replication, biomarker performance, patient-level generalization, or causal morphology-pathway claims.
