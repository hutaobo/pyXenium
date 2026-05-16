# Handoff Scope Inventory

Generated UTC: 2026-05-12T20:19:28.711769+00:00

## In-Scope Paths
- `src/pyXenium/pathway/_morphopathway.py`
- `src/pyXenium/pathway/__init__.py`
- `src/pyXenium/__init__.py`
- `tests/test_morphopathway.py`
- `benchmarking/morphopathway_atera/`

## Current In-Scope Git Status
- `M` `src/pyXenium/__init__.py`
- `M` `src/pyXenium/pathway/__init__.py`
- `??` `benchmarking/morphopathway_atera/`
- `??` `src/pyXenium/pathway/_morphopathway.py`
- `??` `tests/test_morphopathway.py`

## Unrelated Dirty Paths To Avoid Staging
- `M` `benchmarking/cci_2026_atera/scripts/topolink_cci_validation_framework.py`
- `M` `benchmarking/lazyslide_a100/scripts/boundary_ring_wta_profiles.py`
- `M` `benchmarking/lazyslide_a100/scripts/export_morphomolecular_hero_patches.py`
- `M` `benchmarking/lr_2026_atera/scripts/a100_r_method_rescue.py`
- `??` `benchmarking/cci_2026_atera/scripts/make_completed_results_preview_figures.py`
- `??` `benchmarking/cci_2026_atera/scripts/make_cross_method_comparison_figures.py`
- `??` `benchmarking/cci_2026_atera/scripts/publication_24h_supervisor.py`
- `??` `benchmarking/cci_2026_atera/scripts/run_synthetic_truth_benchmark.py`
- `??` `benchmarking/lazyslide_a100/scripts/autopilot_mtm_wta_24h.py`
- `??` `benchmarking/lazyslide_a100/scripts/autopilot_mtm_wta_defense_24h.py`
- `??` `benchmarking/lazyslide_a100/scripts/start_mtm_wta_autopilot_24h.ps1`
- `??` `benchmarking/lazyslide_a100/scripts/start_mtm_wta_defense_autopilot_24h.ps1`

## Final Package
- Package: `D:\GitHub\pyXenium\benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049`
- Archive manifest: `D:\GitHub\pyXenium\benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049.manifest.json`
- Stable core pathways: unfolded_protein_response, immune_exclusion, luminal_estrogen_response, myofibroblast_caf_activation, oxidative_phosphorylation, basal_squamous_state, collagen_ecm_organization, immune_activation, epithelial_identity

## Verification Commands
- `PYTHONPATH=src pytest tests/test_morphopathway.py tests/test_topology_analysis.py -q`
- `python benchmarking/morphopathway_atera/scripts/validate_brief_communication_package.py <package_dir>`
- `python benchmarking/morphopathway_atera/scripts/archive_brief_communication_package.py <package_dir>`
