<p align="center">
  <img src="https://raw.githubusercontent.com/hutaobo/pyXenium/main/docs/_static/branding/pyxenium-horizontal-dark.png" alt="pyXenium horizontal logo" width="960">
</p>

<h1 align="center">pyXenium</h1>

<p align="center">
  Xenium I/O, multimodal analysis, topology workflows, contour-native spatial profiling, GMI inference, and mechanostress analysis.
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

pyXenium is a Python toolkit for **10x Genomics Xenium** with seven canonical public surfaces:

- `pyXenium.io`: Xenium artifact loading, partial export recovery, SData I/O, and SpatialData-compatible export.
- `pyXenium.multimodal`: canonical RNA + protein loading, joint analysis, immune-resistance scoring, and packaged workflows.
- `pyXenium.ligand_receptor`: topology-native ligand-receptor analysis.
- `pyXenium.pathway`: pathway topology analysis and pathway activity scoring.
- `pyXenium.contour`: contour import, contour expansion, and contour-aware density profiling around polygon annotations.
- `pyXenium.gmi`: contour-level GMI modeling for sparse main-effect and interaction discovery in spatial transcriptomics.
- `pyXenium.mechanostress`: morphology-derived mechanical stress states, including fibroblast axis strength, tumor-stroma growth patterning, and cell polarity.

Legacy compatibility entry points under `pyXenium.analysis`, `pyXenium.validation`, and
`pyXenium.io.load_xenium_gene_protein(...)` remain importable, but new code should target the
canonical namespaces above.

## Release & Build

- Current repository version: `0.4.1`
- Package index: [PyPI](https://pypi.org/project/pyXenium/)
- Documentation site: [pyxenium.readthedocs.io](https://pyxenium.readthedocs.io/en/latest/)
- Canonical build status: [GitHub Actions CI](https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml)
- Supported Python: `>=3.8`
- License: [pyXenium Non-Commercial License](https://github.com/hutaobo/pyXenium/blob/main/LICENSE)

## Install

```bash
pip install pyXenium==0.4.1
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

### Contour-GMI inference

```python
from pyXenium.gmi import ContourGmiConfig, run_atera_breast_contour_gmi

result = run_atera_breast_contour_gmi(
    dataset_root="/path/to/WTA_Preview_FFPE_Breast_Cancer_outs",
    output_dir="pyxenium_gmi_outputs/atera_s1_s5",
    config=ContourGmiConfig(feature_count=500, spatial_feature_count=100),
)
```

`pyXenium.gmi` is a canonical beta surface: the API is public, while biological interpretation
should be checked with the bundled controls, cross-validation, and PDC Slurm reproducibility workflow.
The Atera S1/S5 validation completed on PDC Dardel in the `v0.4.1` release, supporting an
S5/DCIS RNA program led by `NIBAN1` and `SORL1` under the QC20 primary model.

### Mechanostress analysis

```python
from pyXenium.io import read_xenium
from pyXenium.mechanostress import MechanostressConfig, run_mechanostress_workflow

sdata = read_xenium("/path/to/xenium_export", as_="sdata", include_boundaries=True)
result = run_mechanostress_workflow(
    sdata,
    output_dir="pyxenium_mechanostress_outputs/sample_1",
    config=MechanostressConfig(),
)
```

For cohorts organized as one Xenium sample per directory, use:

```python
from pyXenium.mechanostress import MechanostressConfig, run_mechanostress_cohort

cohort = run_mechanostress_cohort(
    "/path/to/headandneckSCC",
    output_dir="pyxenium_mechanostress_outputs/hnscc",
    config=MechanostressConfig(coupling_genes=("KRT5", "EPCAM")),
)
```

`pyXenium.mechanostress` is a canonical beta surface for extracting mechanical stress
signals from Xenium morphology and spatial cell context.

## Documentation structure

The docs mirror the package surfaces and high-level workflows:

- Installation / Quickstart
- User Guide
- Workflows
- API Reference
- Changelog

Start here: [pyxenium.readthedocs.io](https://pyxenium.readthedocs.io/en/latest/)

## Branding assets

The repository keeps its Read the Docs, GitHub, and PyPI branding assets under:

```text
docs/_static/branding/
```

Use the derived horizontal PNG for README, PyPI, and GitHub social preview surfaces:

```text
docs/_static/branding/pyxenium-horizontal-dark.png
```

The canonical source artwork remains `docs/_static/branding/pyxenium-horizontal-dark.svg`.

## License

Copyright (c) 2025 Taobo Hu. All rights reserved.

This project is source-available, not open source. You may use, modify, and
redistribute it only for non-commercial purposes under the terms of the
[LICENSE](https://github.com/hutaobo/pyXenium/blob/main/LICENSE) file. Commercial use requires
prior written permission from the copyright holder.
