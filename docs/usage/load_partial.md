# Partial loading

`load_anndata_from_partial(...)` reconstructs an `AnnData` from any subset of Xenium outputs.

It can attach:

- MEX counts
- `analysis.zarr[.zip]` cluster information
- `cells.zarr[.zip]` spatial centroids

## Example

```python
from pyXenium.io import load_anndata_from_partial

adata = load_anndata_from_partial(
    base_url="https://huggingface.co/datasets/hutaobo/pyxenium-gsm9116572/resolve/main",
    analysis_name="analysis.zarr.zip",
    cells_name="cells.zarr.zip",
)
```

If MEX counts are missing, pyXenium can still return an empty-gene `AnnData` with cluster and
spatial metadata attached when those artifacts are available.
