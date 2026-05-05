# Xenium data loading

Use `pyXenium.io` when you want low-level Xenium artifact access and `pyXenium.multimodal`
when you want the canonical RNA + protein `AnnData` used by downstream joint analyses.

## XeniumSlide route

```python
from pyXenium.io import read_xenium

slide = read_xenium(
    "/path/to/xenium_export",
    as_="slide",
    prefer="zarr",
)
```

This route is the right choice when you want images, shapes, points, and table-level metadata together.

`XeniumSlide` is inspired by the data-container ideas documented by
[SpatialData](https://spatialdata.scverse.org/en/stable/), but pyXenium rewrites the
container and on-disk slide store independently and does not require the `spatialdata`
package for core slide I/O. The only bridge back to that ecosystem is the optional
`XeniumSlide.to_spatialdata()` method for users who install `spatialdata` separately.

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
