pyXenium
========

pyXenium is a Python toolkit for loading and analyzing **10x Genomics Xenium** data.

The package now has five canonical public surfaces:

- `pyXenium.io`: Xenium artifact loading, partial export recovery, SData I/O, and SpatialData-compatible export.
- `pyXenium.multimodal`: joint RNA + Protein loading, analysis, immune-resistance scoring, and packaged workflows.
- `pyXenium.ligand_receptor`: topology-native ligand-receptor analysis.
- `pyXenium.pathway`: pathway topology analysis and pathway activity scoring.
- `pyXenium.contour`: contour import plus inward/outward density profiling around polygon annotations.

Legacy compatibility entry points under `pyXenium.analysis`, `pyXenium.validation`, and
`pyXenium.io.load_xenium_gene_protein(...)` are still available as compatibility aliases,
but they now emit `DeprecationWarning` and forward to the new parallel namespaces.

Installation
------------

```bash
pip install pyXenium
# or
pip install "git+https://github.com/hutaobo/pyXenium.git"
```

Core capabilities
-----------------

- **Xenium I/O** via `pyXenium.io.read_xenium(...)`, `write_xenium(...)`, `read_sdata(...)`, and `load_anndata_from_partial(...)`
- **Canonical RNA + Protein preparation** via `pyXenium.multimodal.load_rna_protein_anndata(...)`
- **Joint RNA + Protein analyses** including:
  - `rna_protein_cluster_analysis(...)`
  - `protein_gene_correlation(...)`
  - `ProteinMicroEnv`
- **Immune-resistance analysis** including:
  - `annotate_joint_cell_states(...)`
  - `compute_rna_protein_discordance(...)`
  - `build_spatial_niches(...)`
  - `score_immune_resistance_program(...)`
  - `aggregate_multi_sample_study(...)`
- **Packaged multimodal workflows** including:
  - `run_validated_renal_ffpe_smoke(...)`
  - `run_renal_immune_resistance_pilot(...)`
- **Topology workflows** via:
  - `pyXenium.ligand_receptor.ligand_receptor_topology_analysis(...)`
  - `pyXenium.pathway.pathway_topology_analysis(...)`
  - `pyXenium.pathway.compute_pathway_activity_matrix(...)`
  - `pyXenium.validation.run_atera_wta_breast_topology(...)`
- **Contour-aware profiling** via:
  - `pyXenium.contour.add_contours_from_geojson(...)`
  - `pyXenium.contour.ring_density(...)`
  - `pyXenium.contour.smooth_density_by_distance(...)`

Quick start
-----------

### 1) Xenium I/O

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

adata = load_anndata_from_partial(
    mex_dir="/path/to/xenium_export/cell_feature_matrix",
    analysis_name="/path/to/xenium_export/analysis.zarr.zip",
    cells_name="/path/to/xenium_export/cells.zarr.zip",
)
```

### 2) Canonical RNA + Protein loading

```python
from pyXenium.multimodal import load_rna_protein_anndata

adata = load_rna_protein_anndata(
    base_path="/path/to/xenium_export",
    prefer="auto",
)
```

### 3) Joint RNA + Protein analysis

```python
from pyXenium.multimodal import rna_protein_cluster_analysis

summary, models = rna_protein_cluster_analysis(
    adata,
    n_clusters=12,
    n_pcs=30,
    min_cells_per_cluster=100,
    min_cells_per_group=30,
)
```

### 4) Protein-gene spatial correlation

```python
from pyXenium.multimodal import protein_gene_correlation

pairs = [("CD3E", "CD3E"), ("E-Cadherin", "CDH1")]
summary = protein_gene_correlation(
    adata=adata,
    transcripts_zarr_path="/path/to/transcripts.zarr.zip",
    pairs=pairs,
    output_dir="./protein_gene_corr",
)
```

### 5) Contour-aware density profiling

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

CLI
---

The canonical workflow CLI now lives under `pyxenium multimodal ...`.

```bash
pyxenium multimodal validate-renal-ffpe-protein \
  "Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein"

pyxenium multimodal renal-immune-resistance-pilot \
  "Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein" \
  --output-dir ./renal_immune_resistance_outputs
```

The legacy flat commands are still accepted as deprecated aliases:

```bash
pyxenium validate-renal-ffpe-protein ...
pyxenium renal-immune-resistance-pilot ...
```

Validated renal workflow
------------------------

pyXenium has been smoke-tested against the official 10x Genomics dataset
`Xenium In Situ Gene and Protein Expression data for FFPE Human Renal Cell Carcinoma`.

The validated bundle produces:

- `465545` cells
- `405` RNA features
- `27` protein markers
- spatial centroids in `adata.obsm["spatial"]`
- merged default clusters in `adata.obs["cluster"]`

You can run the packaged workflow from Python:

```python
from pyXenium.multimodal import run_validated_renal_ffpe_smoke

payload = run_validated_renal_ffpe_smoke(
    base_path="Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein",
    output_dir="./smoke_test_outputs",
)
```

Or with the bundled example:

```bash
python examples/smoke_test_10x_renal_ffpe_protein.py \
  "Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein"
```

Spatial immune-resistance pilot
-------------------------------

The renal pilot workflow packages discovery-stage joint RNA + Protein analysis:

- joint cell classes and cell states
- marker-, state-, and pathway-level RNA/protein discordance
- spatial niche construction
- multimodal immune-resistance scoring
- ranked ROI patches and report artifacts

Python API:

```python
from pyXenium.multimodal import run_renal_immune_resistance_pilot

study = run_renal_immune_resistance_pilot(
    base_path="Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein",
    output_dir="./renal_immune_resistance_outputs",
)
```

Example script:

```bash
python examples/renal_immune_resistance_pilot.py \
  "Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein" \
  --output-dir ./renal_immune_resistance_outputs
```

Atera breast topology workflow
------------------------------

The Atera WTA FFPE breast reproducibility workflow now sits on top of the dedicated
topology namespaces rather than `pyXenium.analysis`.

It packages:

- `pyXenium.ligand_receptor.ligand_receptor_topology_analysis(...)`
- `pyXenium.pathway.pathway_topology_analysis(...)`
- `pyXenium.pathway.compute_pathway_activity_matrix(...)`
- `run_atera_wta_breast_topology(...)`

CLI:

```bash
pyxenium atera-wta-breast-topology \
  "Y:/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs"
```

Example script:

```bash
python examples/atera_wta_breast_topology.py \
  "Y:/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs"
```

Minimal API index
-----------------

- `pyXenium.io.read_xenium(...)`
- `pyXenium.io.write_xenium(...)`
- `pyXenium.io.read_sdata(...)`
- `pyXenium.io.load_anndata_from_partial(...)`
- `pyXenium.multimodal.load_rna_protein_anndata(...)`
- `pyXenium.multimodal.protein_gene_correlation(...)`
- `pyXenium.multimodal.rna_protein_cluster_analysis(...)`
- `pyXenium.multimodal.run_validated_renal_ffpe_smoke(...)`
- `pyXenium.multimodal.run_renal_immune_resistance_pilot(...)`
- `pyXenium.ligand_receptor.ligand_receptor_topology_analysis(...)`
- `pyXenium.pathway.pathway_topology_analysis(...)`
- `pyXenium.pathway.compute_pathway_activity_matrix(...)`

License
-------

Copyright (c) 2025 Taobo Hu. All rights reserved.

This project is source-available, not open source. You may use, modify, and
redistribute it only for non-commercial purposes under the terms of the
[LICENSE](LICENSE) file. Commercial use requires prior written permission from
the copyright holder.
