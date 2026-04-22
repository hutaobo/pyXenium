# Multimodal overview

`pyXenium.multimodal` is the canonical public surface for joint RNA + protein work.

It is organized into four layers:

- `loading`: prepare an `AnnData` with aligned RNA, protein, spatial coordinates, and optional cluster attachments.
- `analysis`: core joint analysis tools such as `rna_protein_cluster_analysis`, `protein_gene_correlation`, and `ProteinMicroEnv`.
- `immune_resistance`: marker/state/pathway discordance and spatial immune-resistance scoring.
- `workflows`: packaged renal validation and pilot workflows with markdown, JSON, CSV, and figure artifacts.

## Canonical entrypoints

- `load_rna_protein_anndata`
- `rna_protein_cluster_analysis`
- `protein_gene_correlation`
- `ProteinMicroEnv`
- `annotate_joint_cell_states`
- `compute_rna_protein_discordance`
- `build_spatial_niches`
- `score_immune_resistance_program`
- `aggregate_multi_sample_study`
- `run_validated_renal_ffpe_smoke`
- `run_renal_immune_resistance_pilot`

## Compatibility note

Older multimodal import paths under `pyXenium.analysis`, `pyXenium.validation`, and
`pyXenium.io.load_xenium_gene_protein(...)` remain available for compatibility only.
New code should import from `pyXenium.multimodal`.
