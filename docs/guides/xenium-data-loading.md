# Xenium data loading

This guide focuses on the split between the two public entry points:

- `pyXenium.io` for raw Xenium artifact access.
- `pyXenium.multimodal` for standardized RNA + Protein AnnData preparation.

## Partial / low-level loading

Use `pyXenium.io.partial_xenium_loader.load_anndata_from_partial(...)` when you need to recover incomplete exports or attach only selected artifacts.

## Canonical RNA + Protein loading

Use `pyXenium.multimodal.load_rna_protein_anndata(...)` when you want the standard joint RNA + Protein AnnData used by downstream multimodal analyses and workflows.
