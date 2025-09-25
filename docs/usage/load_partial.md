# Partial Loading: counts + clusters + spatial

`load_anndata_from_partial` reconstructs an `AnnData` using any subset of Xenium outputs:

- **Counts** (required for expression): **MEX triplet** from `cell_feature_matrix/`
- **Clusters** (optional): `analysis.zarr` / `analysis.zarr.zip`
- **Spatial centroids** (optional): `cells.zarr` / `cells.zarr.zip`

## Default behavior

- If `mex_dir` is **not** provided, the loader **automatically** looks for
  `<base>/cell_feature_matrix/{matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz}`.
- If found ⇒ counts are loaded from MEX (fast & robust).
- If not found ⇒ returns an **empty-gene AnnData** but still attaches clusters/spatial if available.

## Examples

### Remote (Hugging Face)

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

BASE = "https://huggingface.co/datasets/hutaobo/pyxenium-gsm9116572/resolve/main"

adata = load_anndata_from_partial(
    base_url=BASE,
    analysis_name="analysis.zarr.zip",
    cells_name="cells.zarr.zip",
)
print(adata)
```

### Local folder

```python
adata = load_anndata_from_partial(
    base_dir="/data/xenium_export",
    analysis_name="analysis.zarr",
    cells_name="cells.zarr",
)
```

### Explicit MEX path (optional)

```python
adata = load_anndata_from_partial(
    base_url=BASE,
    mex_dir=BASE + "/cell_feature_matrix",
)
```

## What gets attached

- **Counts**: MEX → `.X` (CSR) and `.layers["counts"]`
- **Clusters** (if `analysis.zarr*` present): `adata.obs["Cluster"]`
- **Spatial** (if `cells.zarr*` present): `adata.obsm["spatial"]` (or `spatial3d`)

## Notes & FAQ

- We prioritize **10x barcodes** (MEX) as `obs.index`.
- If Zarr stores numeric `cell_id` as `(N,2)` integers, we normalize internally; alignment prefers barcodes.
- Want counts but don’t have MEX? Upload `cell_feature_matrix/` or export MEX from 10x Xenium output.
