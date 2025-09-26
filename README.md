# pyXenium

Utilities for loading and analyzing 10x Genomics Xenium exports in Python.

## Quickstart

Install:

```bash
# 稳定用法（本地开发/CI）
pip install -U "git+https://github.com/hutaobo/pyXenium@main"
# 或
pip install pyXenium
```

Load a dataset hosted online (e.g. Hugging Face). **By default, `load_anndata_from_partial` looks for the 10x MEX triplet under `<base>/cell_feature_matrix/`**:

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

BASE = "https://huggingface.co/datasets/hutaobo/pyxenium-gsm9116572/resolve/main"

adata = load_anndata_from_partial(
    base_url=BASE,
    analysis_name="analysis.zarr.zip",     # optional, attaches clusters if present
    cells_name="cells.zarr.zip",           # optional, attaches spatial centroids if present
    # By default it will read MEX from: BASE + "/cell_feature_matrix/"
)
print(adata)
```

**What gets loaded:**
- **Counts**: from MEX (`cell_feature_matrix/{matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz}`).
- **Clusters** (optional): from `analysis.zarr[.zip]` if provided.
- **Spatial centroids** (optional): from `cells.zarr[.zip]` if provided.

If MEX is missing:
- The function returns an **empty-gene AnnData** (rows=cells if we can infer cell IDs; otherwise empty).
- Clusters/spatial are still attached when possible.
- To get real counts, upload MEX to `<base>/cell_feature_matrix/` or pass `mex_dir=...` explicitly.

### Override the MEX location (optional)
```python
adata = load_anndata_from_partial(
    base_url=BASE,
    mex_dir=BASE + "/cell_feature_matrix",   # explicit
    analysis_name="analysis.zarr.zip",
    cells_name="cells.zarr.zip",
)
```

### Local folder example
```python
adata = load_anndata_from_partial(
    base_dir="/path/to/xenium_export",
    analysis_name="analysis.zarr",
    cells_name="cells.zarr",
    # will look for /path/to/xenium_export/cell_feature_matrix/
)
```

### Protein + RNA correlation (Xenium)

Compute spatial correlation between **mean protein intensity** and **gene transcript density** on a user-defined grid.

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial
from pyXenium.analysis import protein_gene_correlation

# 1) Load data (RNA counts + optional cells/analysis attachments)
BASE = "https://huggingface.co/datasets/<your-dataset>/resolve/main"
adata = load_anndata_from_partial(
    base_url=BASE,
    # or base_dir="/path/to/Xenium_Export"
    # If you also have analysis/cells zarr, pass them here to attach clusters & spatial centroids
)

# 2) Run protein–gene correlation
#    transcripts_zarr_path 可为 .zarr 或 .zarr.zip（本地或远程，fsspec 透明读）
pairs = [("CD3E", "CD3E"), ("E-Cadherin", "CDH1")]   # (protein, gene)
summary = protein_gene_correlation(
    adata=adata,
    transcripts_zarr_path=BASE + "/transcripts.zarr.zip",
    pairs=pairs,
    output_dir="./protein_gene_corr",
    grid_size=(50, 50),     # 可自定义网格
    pixel_size_um=0.2125,   # Xenium 常见像素尺寸
    qv_threshold=20,
    overwrite=False
)
print(summary)
```

### Troubleshooting
- **FileNotFoundError: MEX missing files** → Ensure the three files exist in `cell_feature_matrix/`:
  `matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`.
- **Different obs names** → We honor 10x barcodes (from MEX). If your Zarr stores numeric
  cell IDs, we normalize them to strings internally but prefer the barcodes from MEX.
- **Large downloads** → Remote MEX is downloaded once into a temp dir per session run.
