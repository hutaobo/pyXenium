# Xenium data loading

Use `pyXenium.io` when you want low-level Xenium artifact access and `pyXenium.multimodal`
when you want the canonical RNA + protein `AnnData` used by downstream joint analyses.

## XeniumSData route

```python
from pyXenium.io import read_xenium

sdata = read_xenium(
    "/path/to/xenium_export",
    as_="sdata",
    prefer="zarr",
)
```

This route is the right choice when you want images, shapes, points, and table-level metadata together.

## Canonical multimodal AnnData route

```python
from pyXenium.multimodal import load_rna_protein_anndata

adata = load_rna_protein_anndata(
    base_path="/path/to/xenium_export",
    prefer="auto",
)
```

This route is the right choice when you want the package-standard RNA + protein matrix for
multimodal clustering, protein-gene correlation, or immune-resistance workflows.
