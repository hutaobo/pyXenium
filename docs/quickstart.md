# Installation and quickstart

## Install from PyPI

```bash
pip install pyXenium
```

## Install from source

```bash
git clone https://github.com/hutaobo/pyXenium
cd pyXenium
pip install -e ".[dev]"
```

## Install the docs toolchain

```bash
pip install -e ".[docs]"
```

This installs the Sphinx-based documentation stack used by Read the Docs and local builds.

## First examples

### Xenium I/O

```python
from pyXenium.io import read_xenium

sdata = read_xenium("/path/to/xenium_export", as_="sdata", prefer="zarr")
```

### Canonical multimodal loading

```python
from pyXenium.multimodal import load_rna_protein_anndata

adata = load_rna_protein_anndata(
    base_path="/path/to/xenium_export",
    prefer="auto",
)
```

### Multimodal workflow CLI

```bash
pyxenium multimodal validate-renal-ffpe-protein /path/to/xenium_export
pyxenium multimodal renal-immune-resistance-pilot /path/to/xenium_export
```

## Build the docs locally

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

The docs should build without relying on MkDocs or GitHub Pages.
