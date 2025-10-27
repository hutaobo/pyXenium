pyXenium
========

pyXenium is a Python library for loading and analyzing **10x Genomics Xenium** in‑situ outputs.
It supports **robust partial loading** of incomplete exports and provides utilities for **multi‑modal (RNA + Protein)** runs.

Version: 0.1.1

---

Features
--------
- **Partial loading of incomplete exports** — assemble an `AnnData` even when some Xenium artifacts
  are missing; opportunistically attaches clusters (`analysis.zarr[.zip]`) and spatial centroids (`cells.zarr[.zip]`).
- **RNA + Protein support** — read combined cell‑feature matrices from Zarr/HDF5/MEX, split features by type,
  and return matched cell × gene/protein data.
- **Protein–gene spatial correlation** — compute correlations between protein intensity and gene transcript
  density across spatial bins; export plots and CSV summaries.
- **Toy dataset included** — a minimal Xenium‑like dataset (`toy_slide`) to get started quickly.

Installation
------------
The package is organized as a standard `src/` layout. Until a PyPI release is available, install from source or Git:

```bash
# From GitHub (source)
pip install "git+https://github.com/hutaobo/pyXenium.git"
```

Requirements (typical): Python 3.9+; `anndata`, `numpy`, `pandas`, `scipy`, `zarr`, `fsspec`, `matplotlib`, `scikit-learn`, `click`.
(Exact dependencies follow the project configuration and imports.)

Quick Start
-----------

### 1) Partial loading (incomplete exports)

Use `pyXenium.io.partial_xenium_loader.load_anndata_from_partial(...)` to assemble an `AnnData` from any available pieces.

**Local files example:**
```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

adata = load_anndata_from_partial(
    mex_dir="/path/to/xenium_export/cell_feature_matrix",  # MEX triplet folder
    analysis_name="/path/to/xenium_export/analysis.zarr.zip",  # optional
    cells_name="/path/to/xenium_export/cells.zarr.zip",        # optional
    # transcripts_name="/path/to/xenium_export/transcripts.zarr.zip",  # optional
)
print(adata)
```

**Remote base example:**
```python
adata = load_anndata_from_partial(
    base_url="https://example.org/xenium_run",  # artifacts live under <base_url>/
    analysis_name="analysis.zarr.zip",
    cells_name="cells.zarr.zip",
)
```

Behavior:
- If the MEX triplet is unavailable, the function still returns a valid `AnnData` (empty genes) and attaches
  clusters/spatial information when possible.
- Zarr roots are auto‑detected inside `*.zarr.zip` even when the root metadata sits in a subfolder.

**Signature (summary):**
```text
load_anndata_from_partial(
    base_url: str | None = None,
    analysis_name: str | None = None,
    cells_name: str | None = None,
    transcripts_name: str | None = None,
    mex_dir: str | None = None,
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    build_counts_if_missing: bool = True,
) -> anndata.AnnData
```

### 2) RNA + Protein loader

Use `pyXenium.io.xenium_gene_protein_loader.load_xenium_gene_protein(...)` to load Xenium exports with protein measurements.

```python
from pyXenium.io.xenium_gene_protein_loader import load_xenium_gene_protein

adata = load_xenium_gene_protein(
    base_path="/path/to/xenium_export",
    prefer="auto",  # auto | zarr | h5 | mex
)
# adata.X: RNA counts (CSR); adata.layers["rna"] may hold RNA counts explicitly
# adata.obsm["protein"]: DataFrame of protein intensities
# adata.obsm["spatial"]: cell centroids when available
```

Notes:
- Supported matrix formats: Zarr (`cell_feature_matrix.zarr/` or `cell_feature_matrix/`), HDF5 (`cell_feature_matrix.h5`), or MEX (`matrix.mtx.gz` triplet).
- Feature types are split using the 3rd column of `features.tsv.gz` (e.g., "Gene Expression", "Protein Expression").
- Optionally attaches centroids/boundaries into `adata.obsm["spatial"]` and `adata.uns`.
- If present, clustering results at `analysis/clustering/gene_expression_graphclust/clusters.csv` are merged into `adata.obs["cluster"]` by default.

**Signature (summary):**
```text
load_xenium_gene_protein(
    base_path: str,
    *,
    prefer: str = "auto",  # "auto" | "zarr" | "h5" | "mex"
    mex_dirname: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    cells_csv: str = "cells.csv.gz",
    cells_parquet: str | None = None,
    read_morphology: bool = False,
    attach_boundaries: bool = True,
    clusters_relpath: str | None = "analysis/clustering/gene_expression_graphclust/clusters.csv",
    cluster_column_name: str = "cluster",
) -> anndata.AnnData
```

### 3) Protein–gene spatial correlation

`pyXenium.analysis.protein_gene_correlation.protein_gene_correlation(...)` computes Pearson correlations between
**protein average intensity** and **gene transcript density** across spatial bins; it saves per‑pair figures and CSVs,
plus a summary CSV.

```python
from pyXenium.analysis.protein_gene_correlation import protein_gene_correlation

pairs = [("CD3E", "CD3E"), ("E-Cadherin", "CDH1")]  # (protein, gene)
summary = protein_gene_correlation(
    adata=adata,
    transcripts_zarr_path="/path/to/transcripts.zarr.zip",
    pairs=pairs,
    output_dir="./protein_gene_corr",
    grid_size=(50, 50),          # μm per bin (used if grid_counts is None)
    pixel_size_um=0.2125,
    qv_threshold=20,
    overwrite=False,
    auto_detect_cell_units=True,
)
print(summary.head())
```

**Signature (summary):**
```text
protein_gene_correlation(
    adata,
    transcripts_zarr_path,
    pairs,
    output_dir,
    grid_size=(50, 50),
    grid_counts=(50, 50),
    pixel_size_um=0.2125,
    qv_threshold=20,
    overwrite=False,
    auto_detect_cell_units=True,
) -> pandas.DataFrame
```

### 4) RNA/protein joint analysis

Train small classifiers on the RNA latent space to explain within‑cluster protein heterogeneity:

```python
from pyXenium.analysis import rna_protein_cluster_analysis

summary, models = rna_protein_cluster_analysis(
    adata,
    n_clusters=12,
    n_pcs=30,
    min_cells_per_cluster=100,
    min_cells_per_group=30,
    hidden_layer_sizes=(128, 64),
)
print(summary.head())
```

**Signature (summary):**
```text
rna_protein_cluster_analysis(
    adata: anndata.AnnData,
    *,
    n_clusters: int = 12,
    n_pcs: int = 30,
    cluster_key: str = "rna_cluster",
    random_state: int | None = 0,
    target_sum: float = 1e4,
    min_cells_per_cluster: int = 50,
    min_cells_per_group: int = 20,
    protein_split_method: str = "median",
    protein_quantile: float = 0.75,
    test_size: float = 0.2,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 200,
    early_stopping: bool = True,
) -> tuple[pandas.DataFrame, dict]
```

Command‑line
------------

A small CLI is provided via `python -m pyXenium` (requires `click`).

```bash
# Print a quick sanity check on the toy dataset
python -m pyXenium demo

# Fetch a toy dataset to a cache directory
python -m pyXenium datasets --name toy_slide --dest ~/.cache/pyXenium
```

Data layout expectations
------------------------
- **cell_feature_matrix/**
  `matrix.mtx.gz`, `features.tsv.gz` (≥3 columns: id, name, feature_type), `barcodes.tsv.gz`
- Optional: `analysis.zarr[.zip]` (clusters), `cells.zarr[.zip]` (spatial centroids)
- `transcripts.zarr[.zip]` for spatial transcript coordinates used in correlation analyses.

Minimal API reference (index)
-----------------------------
- `pyXenium.io.partial_xenium_loader.load_anndata_from_partial(...)`
- `pyXenium.io.xenium_gene_protein_loader.load_xenium_gene_protein(...)`
- `pyXenium.analysis.protein_gene_correlation.protein_gene_correlation(...)`
- `pyXenium.analysis.rna_protein_cluster_analysis.rna_protein_cluster_analysis(...)`

Example data
------------
The package ships with a tiny Xenium‑like toy dataset. Programmatic access:

```python
from pyXenium.io.io import load_toy
z = load_toy()
cells = z["cells"]          # zarr group
transcripts = z["transcripts"]
analysis = z["analysis"]
```

Citations
---------
If this toolkit helps your work, please cite the project and the 10x Genomics Xenium platform as appropriate.

License
-------
All rights reserved by the author.
