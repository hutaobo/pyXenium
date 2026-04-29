<p align="center">
  <img src="https://raw.githubusercontent.com/hutaobo/pyXenium/main/docs/_static/branding/pyxenium-horizontal-dark.png" alt="pyXenium horizontal logo" width="960">
</p>

<h1 align="center">pyXenium</h1>

<p align="center">
  Nine canonical sections for Xenium I/O, multimodal analysis, CCI, pathway topology, contour geometry, GMI inference, mechanostress analysis, and external workflow bridges.
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

pyXenium is a Python toolkit for **10x Genomics Xenium** organized around nine major sections.

| Section | Canonical entry | Start here |
| --- | --- | --- |
| Xenium I/O | `pyXenium.io` | [Xenium data loading guide](https://pyxenium.readthedocs.io/en/latest/guides/xenium-data-loading.html) |
| Multimodal Analysis | `pyXenium.multimodal` | [Multimodal overview](https://pyxenium.readthedocs.io/en/latest/user-guide/multimodal-overview.html) |
| Cell-Cell Interaction | `pyXenium.cci` | [CCI tutorial hub](https://pyxenium.readthedocs.io/en/latest/tutorials/cci_index.html) |
| Pathway Topology | `pyXenium.pathway` | [Pathway tutorial](https://pyxenium.readthedocs.io/en/latest/tutorials/pathway.html) |
| Contour Geometry | `pyXenium.contour` | [Contour tutorial hub](https://pyxenium.readthedocs.io/en/latest/tutorials/contour_index.html) |
| GMI Inference | `pyXenium.gmi` | [Contour GMI guide](https://pyxenium.readthedocs.io/en/latest/guides/gmi-contour.html) |
| Mechanostress | `pyXenium.mechanostress` | [Mechanostress tutorial](https://pyxenium.readthedocs.io/en/latest/tutorials/mechanostress_atera_pdc.html) |
| AI-Driven Spatial Pathologist | external `spatho` bridge | [RTD bridge guide](https://pyxenium.readthedocs.io/en/latest/tutorials/ai_driven_spatial_pathologist.html) |
| SpatialPerturb Bridge | `pyXenium.perturb` | [SpatialPerturb bridge guide](https://pyxenium.readthedocs.io/en/latest/tutorials/spatialperturb_bridge.html) |

Legacy compatibility entry points under `pyXenium.analysis`, `pyXenium.validation`, and
`pyXenium.io.load_xenium_gene_protein(...)` remain importable, but new code should target the
canonical pyXenium namespaces above. The `spatho` and `SpatialPerturb` workflows are installed
and run separately; pyXenium does not vendor them or add them as core runtime dependencies.

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

For the optional SpatialPerturb Bridge runtime on Python 3.9+:

```bash
pip install -e ".[perturb]"
```

## Representative examples

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

### AI-Driven Spatial Pathologist via spatho

[`AI-Driven Spatial Pathologist`](https://ai-driven-spatial-pathologist.readthedocs.io/en/latest/?badge=latest)
is an external workflow layer for AI-assisted pathology review around Xenium-scale spatial
transcriptomics. Its public Python package and CLI are named `spatho`.

```bash
pip install -U spatho
spatho init-workflow --organ breast --case-name breast_case_01 --dataset-root /path/to/Xenium_outs --output workflow.json
spatho doctor --config workflow.json
spatho run --config workflow.json
```

In pyXenium, this is documented as an optional external workflow bridge rather than a new
`pyXenium.spatho` namespace.
The handoff is possible because `XeniumSData` keeps the cell table, transcript points,
cell/nucleus boundaries, H&E image metadata, and SpatialData-compatible organization together
for downstream tools.

### SpatialPerturb Bridge via SpatialPerturb

[`SpatialPerturb`](https://github.com/hutaobo/SpatialPerturb) is an external workflow package
for combining spatial transcriptomics with Perturb-seq references. pyXenium exposes a lightweight
`pyXenium.perturb` bridge that writes a handoff JSON and stable external CLI commands without
vendoring the SpatialPerturb algorithms.

```python
from pyXenium.perturb import SpatialPerturbBridgeConfig, write_spatialperturb_handoff

spec = write_spatialperturb_handoff(
    SpatialPerturbBridgeConfig(
        xenium_path="/path/to/Xenium_outs",
        output_dir="spatialperturb_reports/breast_case_01",
        cell_group_path="/path/to/cell_groups.csv",
        roi_geojson_path="/path/to/xenium_explorer_annotations.geojson",
        sample_name="breast_case_01",
    ),
    "spatialperturb_bridge.json",
)
print(spec["command_text"]["run_reference_benchmark"])
```

SpatialPerturb Bridge scores mean Perturb-seq-derived program similarity projected onto Xenium
tissue. They do not mean the tissue cell contains the corresponding knockout, guide, or drug
perturbation.

## Documentation entry points

The docs now separate the nine feature sections from the main reading paths:

- [Quickstart](https://pyxenium.readthedocs.io/en/latest/quickstart.html)
- [Tutorials hub](https://pyxenium.readthedocs.io/en/latest/tutorials/index.html)
- [User guide](https://pyxenium.readthedocs.io/en/latest/user-guide/index.html)
- [Workflows](https://pyxenium.readthedocs.io/en/latest/workflows/index.html)
- [API reference](https://pyxenium.readthedocs.io/en/latest/api/index.html)
- [Changelog](https://pyxenium.readthedocs.io/en/latest/changelog.html)

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
