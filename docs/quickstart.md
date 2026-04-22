# Quickstart

## Install

> Until the package is published on PyPI, install from GitHub:

```bash
pip install -U "git+https://github.com/hutaobo/pyXenium@main"
# or for local development:
# git clone https://github.com/hutaobo/pyXenium
# cd pyXenium
# pip install -e ".[docs]"
```

## Canonical modules

- `pyXenium.io` handles Xenium data access and export.
- `pyXenium.multimodal` handles joint RNA + Protein analysis and workflows.
- `pyXenium.contour` handles contour import plus inward/outward density profiling.

## Load a partial Xenium export

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

BASE = "https://huggingface.co/datasets/<your-dataset>/resolve/main"
adata = load_anndata_from_partial(base_url=BASE)

adata
```

## Load a Xenium RNA + Protein AnnData

```python
from pyXenium.multimodal import load_rna_protein_anndata

adata = load_rna_protein_anndata(
    base_path="/path/to/xenium_export",
    prefer="auto",
)
```

## Run a multimodal workflow

```bash
pyxenium multimodal validate-renal-ffpe-protein /path/to/xenium_export
pyxenium multimodal renal-immune-resistance-pilot /path/to/xenium_export
```

The legacy flat commands are still available as deprecated aliases for compatibility.

## Add contours and compute contour-aware density

```python
from pyXenium.contour import (
    add_contours_from_geojson,
    ring_density,
    smooth_density_by_distance,
)
from pyXenium.io import read_xenium

sdata = read_xenium(
    "/path/to/xenium_export",
    as_="sdata",
    include_images=True,
)

add_contours_from_geojson(
    sdata,
    "/path/to/polygon_units.geojson",
    key="protein_cluster_contours",
)

ring_df = ring_density(
    sdata,
    contour_key="protein_cluster_contours",
    target="transcripts",
    contour_query='assigned_structure == "Structure 4"',
    feature_values="VIM",
    inward=100.0,
    outward=100.0,
    ring_width=50.0,
)

smooth_df = smooth_density_by_distance(
    sdata,
    contour_key="protein_cluster_contours",
    target="transcripts",
    contour_query='assigned_structure == "Structure 4"',
    feature_values="VIM",
    inward=100.0,
    outward=100.0,
    bandwidth=25.0,
)
```
