# PLIP Morphopathway Stability Notes

Compared runs: 4
Cross-cancer recovery range: 0/10 to 9/10.
Median cross-cancer recovery: 7.0/10.
Stable recovered pathways in all compared runs: none.
Partially recovered pathways (>=2 runs): luminal_estrogen_response, myofibroblast_caf_activation, basal_squamous_state, collagen_ecm_organization, epithelial_identity, unfolded_protein_response, immune_exclusion, oxidative_phosphorylation, immune_activation.

Spatial block settings:
- plip_smoke_3000_d64_seed17_null12_20260512_1908: breast bins=18, blocks=237, median cells/block=11.0; cervical bins=18, blocks=252, median cells/block=10.5.
- plip_smoke_3000_d64_seed29_null12_20260512_1918: breast bins=18, blocks=229, median cells/block=11.0; cervical bins=18, blocks=259, median cells/block=10.0.
- plip_smoke_3000_d64_seed17_bins12_null12_20260512_1934: breast bins=12, blocks=135, median cells/block=20.0; cervical bins=12, blocks=137, median cells/block=21.0.
- plip_smoke_3000_d64_seed29_bins12_null12_20260512_1924: breast bins=12, blocks=133, median cells/block=22.0; cervical bins=12, blocks=137, median cells/block=21.0.

Gate interpretation:
- Strongest evidence: pathways recovered in all compared seeds.
- Supportive evidence: pathways recovered in at least two seeds.
- Do not claim direct cervical replication from one sampled run alone.

Interpretation: this is a stability screen over sampled cells and spatial blocks, not a final exhaustive analysis.
