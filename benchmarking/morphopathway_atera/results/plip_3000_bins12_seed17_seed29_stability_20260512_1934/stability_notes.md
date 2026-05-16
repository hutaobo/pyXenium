# PLIP Morphopathway Stability Notes

Compared runs: 2
Cross-cancer recovery range: 9/10 to 9/10.
Median cross-cancer recovery: 9.0/10.
Stable recovered pathways in all compared runs: immune_exclusion, luminal_estrogen_response, unfolded_protein_response, myofibroblast_caf_activation, oxidative_phosphorylation, collagen_ecm_organization, basal_squamous_state, epithelial_identity, immune_activation.
Partially recovered pathways (>=2 runs): immune_exclusion, luminal_estrogen_response, unfolded_protein_response, myofibroblast_caf_activation, oxidative_phosphorylation, collagen_ecm_organization, basal_squamous_state, epithelial_identity, immune_activation.

Spatial block settings:
- plip_smoke_3000_d64_seed17_bins12_null12_20260512_1934: breast bins=12, blocks=135, median cells/block=20.0; cervical bins=12, blocks=137, median cells/block=21.0.
- plip_smoke_3000_d64_seed29_bins12_null12_20260512_1924: breast bins=12, blocks=133, median cells/block=22.0; cervical bins=12, blocks=137, median cells/block=21.0.

Gate interpretation:
- Strongest evidence: pathways recovered in all compared seeds.
- Supportive evidence: pathways recovered in at least two seeds.
- Do not claim direct cervical replication from one sampled run alone.

Interpretation: this is a stability screen over sampled cells and spatial blocks, not a final exhaustive analysis.
