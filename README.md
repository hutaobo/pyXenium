<p align="center">
  <img src="https://raw.githubusercontent.com/hutaobo/pyXenium/main/docs/_static/branding/pyxenium-banner.png" alt="pyXenium banner" width="100%">
</p>

<h1 align="center">pyXenium</h1>

<p align="center">
  Xenium I/O, multimodal analysis, topology workflows, and contour-native spatial profiling.
</p>

<p align="center">
  <a href="https://pypi.org/project/pyXenium/"><img src="https://img.shields.io/pypi/v/pyXenium.svg" alt="PyPI version"></a>
  <a href="https://pyxenium.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/pyxenium/badge/?version=latest" alt="Read the Docs"></a>
  <a href="https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml"><img src="https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/pyXenium/"><img src="https://img.shields.io/pypi/pyversions/pyXenium.svg" alt="Python versions"></a>
  <a href="https://github.com/hutaobo/pyXenium/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-non--commercial-d97706.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyXenium/">PyPI</a>
  ·
  <a href="https://pyxenium.readthedocs.io/en/latest/">Read the Docs</a>
  ·
  <a href="https://github.com/hutaobo/pyXenium">GitHub</a>
  ·
  <a href="https://pyxenium.readthedocs.io/en/latest/changelog.html">Changelog</a>
  ·
  <a href="https://github.com/hutaobo/pyXenium/releases">Releases</a>
</p>

pyXenium is a Python toolkit for **10x Genomics Xenium** with five canonical public surfaces:

- `pyXenium.io`: Xenium artifact loading, partial export recovery, SData I/O, and SpatialData-compatible export.
- `pyXenium.multimodal`: canonical RNA + protein loading, joint analysis, immune-resistance scoring, and packaged workflows.
- `pyXenium.ligand_receptor`: topology-native ligand-receptor analysis.
- `pyXenium.pathway`: pathway topology analysis and pathway activity scoring.
- `pyXenium.contour`: contour import, contour expansion, and contour-aware density profiling around polygon annotations.

Legacy compatibility entry points under `pyXenium.analysis`, `pyXenium.validation`, and
`pyXenium.io.load_xenium_gene_protein(...)` remain importable, but new code should target the
canonical namespaces above.

## Release & Build

- Current repository version: `0.2.1`
- Package index: [PyPI](https://pypi.org/project/pyXenium/)
- Documentation site: [pyxenium.readthedocs.io](https://pyxenium.readthedocs.io/en/latest/)
- Canonical build status: [GitHub Actions CI](https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml)
- Supported Python: `>=3.8`
- License: [pyXenium Non-Commercial License](https://github.com/hutaobo/pyXenium/blob/main/LICENSE)

## Install

```bash
pip install pyXenium
```

For local development:

```bash
git clone https://github.com/hutaobo/pyXenium
cd pyXenium
pip install -e ".[dev]"
```

For documentation work:

```bash
pip install -e ".[docs]"
```

## Quick examples

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

### Contour expansion

```python
from pyXenium.contour import expand_contours

expand_contours(
    sdata,
    contour_key="protein_cluster_contours",
    distance=25.0,
    mode="voronoi",
)
```

## Documentation structure

The docs mirror the package surfaces and high-level workflows:

- Installation / Quickstart
- User Guide
- Workflows
- API Reference
- Changelog

Start here: [pyxenium.readthedocs.io](https://pyxenium.readthedocs.io/en/latest/)

## Branding assets

The repository keeps its Read the Docs and GitHub branding assets under:

```text
docs/_static/branding/
```

To update the GitHub social preview manually, upload:

```text
docs/_static/branding/pyxenium-social-preview.png
```

in the repository settings.

## License

Copyright (c) 2025 Taobo Hu. All rights reserved.

This project is source-available, not open source. You may use, modify, and
redistribute it only for non-commercial purposes under the terms of the
[LICENSE](https://github.com/hutaobo/pyXenium/blob/main/LICENSE) file. Commercial use requires
prior written permission from the copyright holder.
