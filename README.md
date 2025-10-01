# pyXenium

**pyXenium** is a Python library for loading and analyzing 10x Genomics Xenium *in situ* exports.
It supports robust partial loading of incomplete exports and provides utilities for multi‑modal
Xenium runs that include both RNA and protein measurements.

> If you are already familiar with Xenium outputs, jump to:
> - [Partial loading (incomplete exports)](#partial-loading-incomplete-exports)
> - [RNA + Protein loader](#rna--protein-loader)
> - [Gene–protein correlation](#gene–protein-correlation)

## Features

- **Partial loading of incomplete exports** — Load what is available even when the
  `cell_feature_matrix` MEX is missing/partial; optional attachment of clusters and spatial centroids.
- **RNA + Protein support** — Read combined cell-feature matrices, split features by type
  (Gene Expression vs Protein Expression), and return matched cell × gene/protein matrices.
- **Protein–gene correlation** — Compute Pearson/Spearman correlations between gene expression
  and protein intensities across cells.

## Installation

```bash
# From PyPI (if available)
pip install pyXenium

# Or install the latest from GitHub
pip install git+https://github.com/hutaobo/pyXenium.git
```

Python ≥3.9 is recommended.

## Quick start

### Partial loading (incomplete exports)

`pyXenium.io.partial_xenium_loader.load_anndata_from_partial` tries to assemble an `AnnData`
object from a Xenium export directory or HTTP(S) base. It attaches optional results when present
(e.g. `analysis.zarr[.zip]` and `cells.zarr[.zip]`).

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

# Local export directory
adata = load_anndata_from_partial(
    base_dir="/path/to/xenium_export",
    analysis_name="analysis.zarr",   # optional
    cells_name="cells.zarr",         # optional
)

# Or remote base (files hosted under <BASE>/)
# adata = load_anndata_from_partial(
#     base_url="https://example.org/xenium_run",
#     analysis_name="analysis.zarr.zip",
#     cells_name="cells.zarr.zip",
# )
print(adata)  # cells × genes AnnData
```

**What gets loaded** (when available):
- Counts from `cell_feature_matrix/{matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz}`
- Clusters from `analysis.zarr[.zip]`
- Spatial centroids from `cells.zarr[.zip]`

If the MEX triplet is missing, the function still returns a valid `AnnData` (empty genes)
and attaches clusters/spatial if found — useful for inspecting partial/early exports.

### RNA + Protein loader

Use the dedicated loader in `pyXenium.io.xenium_gene_protein_loader` to read a Xenium export
that includes protein measurements. It separates features by type and aligns cells across modalities.

```python
from pyXenium.io.xenium_gene_protein_loader import load_xenium_gene_protein

adata = load_xenium_gene_protein(
    base_path="/mnt/taobo.hu/long/10X_datasets/Xenium/Xenium_Kidney/Xenium_V1_Human_Kidney_FFPE_Protein"
)
```

Notes:
- The loader expects a combined MEX under `cell_feature_matrix/` where the 3rd column in
  `features.tsv.gz` indicates the feature type (e.g., `"Gene Expression"`, `"Protein Expression"`).
- Invalid/control entries (e.g., blank/unassigned codewords) are filtered by default.
- Both matrices share **identical cell order**, enabling 1:1 comparisons across modalities.

### Gene–protein correlation

Compute correlations between gene and protein across cells.

```python
BASE = "/mnt/taobo.hu/long/10X_datasets/Xenium/Xenium_Kidney/Xenium_V1_Human_Kidney_FFPE_Protein"
pairs = [("CD3E", "CD3E"), ("E-Cadherin", "CDH1")]   # (protein, gene)

from pyXenium.analysis.protein_gene_correlation import protein_gene_correlation
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

## Data format expectations

- **Cell-feature matrix (MEX)** under `cell_feature_matrix/`:
  - `matrix.mtx.gz`: sparse counts/intensities
  - `features.tsv.gz`: 3 columns: `id`, `name`, `feature_type`
  - `barcodes.tsv.gz`: cell barcodes (one per row)
- **Optional**: `analysis.zarr[.zip]` (clusters), `cells.zarr[.zip]` (spatial centroids)

## API reference (summary)

- `pyXenium.io.partial_xenium_loader.load_anndata_from_partial(base_dir=None, base_url=None, mex_dir=None, analysis_name=None, cells_name=None)`
- `pyXenium.io.xenium_gene_protein_loader.load_gene_protein(base_dir, mex_dir=None, drop_controls=True)`
- `pyXenium.analysis.protein_gene_correlation.compute(gene_expr, protein_expr, method='pearson')`

## Contributing

Issues and pull requests are welcome. Please include minimal examples and tests where possible.

## License

MIT. See `LICENSE`.
