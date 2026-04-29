# SpatialPerturb Bridge

## Overview

[SpatialPerturb](https://github.com/hutaobo/SpatialPerturb) is an external
workflow package for combining spatial transcriptomics with Perturb-seq
references. pyXenium treats it as an optional bridge: pyXenium prepares the
Xenium-side handoff, while SpatialPerturb owns reference projection, program
scoring, benchmark reports, and publication-style outputs.

The bridge is useful when the Xenium tissue itself is unperturbed, but the study
needs to ask where Perturb-seq-derived transcriptional programs appear in
spatial context.

## Relationship to Xenium data

SpatialPerturb Bridge starts from the same Xenium data foundation used by the
rest of pyXenium:

- Xenium export directories with `cell_feature_matrix.h5` and cell metadata
- optional Xenium cell-group CSV files for cell-state grouping
- optional Xenium Explorer ROI GeoJSON annotations
- report directories that can sit beside pyXenium contour, GMI, CCI, pathway,
  mechanostress, and pathology artifacts

The bridge does not duplicate SpatialPerturb algorithms inside pyXenium. It
generates a machine-readable handoff JSON plus the external CLI commands needed
to prepare a SpatialPerturb-compatible `.h5ad` file and run reference projection.

## Minimal setup

Install the optional runtime in a Python 3.9+ environment:

```bash
pip install "pyXenium[perturb]"
```

or install the external package directly:

```bash
pip install "SpatialPerturb>=0.3"
```

Build a handoff spec from pyXenium:

```python
from pyXenium.perturb import SpatialPerturbBridgeConfig, write_spatialperturb_handoff

spec = write_spatialperturb_handoff(
    SpatialPerturbBridgeConfig(
        xenium_path="/data/Xenium_outs",
        output_dir="/data/reports/spatialperturb_breast_case_01",
        cache_dir="/data/spatialperturb_cache",
        cell_group_path="/data/Xenium_cell_groups.csv",
        roi_geojson_path="/data/xenium_explorer_annotations.geojson",
        sample_name="breast_case_01",
        reference_datasets=("gse241115_breast_cropseq",),
    ),
    "/data/reports/spatialperturb_bridge.json",
)
```

The returned `spec["command_text"]` entries can be run in the SpatialPerturb
environment.

## Reference projection workflow

The handoff spec contains three command entries:

```bash
python -m pip install "SpatialPerturb>=0.3"
spatialperturb prepare-xenium /data/Xenium_outs /data/reports/spatialperturb_breast_case_01/spatialperturb_xenium.h5ad
spatialperturb run-reference-benchmark /data/reports/spatialperturb_breast_case_01/spatialperturb_xenium.h5ad /data/reports/spatialperturb_breast_case_01
```

When cell groups, ROI GeoJSON, sample name, cache directory, or multiple
reference datasets are provided, the bridge adds the matching SpatialPerturb CLI
options to the generated commands.

## Interpretation caveats

SpatialPerturb Bridge scores mean **Perturb-seq-derived program similarity**
projected onto Xenium tissue. They do not mean the tissue cell contains the
corresponding knockout, guide, or drug perturbation.

Use the projected scores as mechanism-oriented hypotheses that should be checked
against pyXenium spatial context, biological controls, reference quality, and
the SpatialPerturb report outputs.

## pyXenium boundary

pyXenium provides the Xenium data foundation, stable handoff spec, and optional
`pyXenium.perturb` helpers. SpatialPerturb owns the external workflow runtime,
Perturb-seq reference programs, calibration, benchmarks, and report rendering.

This keeps pyXenium lightweight: no vendored SpatialPerturb code, no core
runtime dependency, and no pyXenium CLI proxy are required for the first bridge
surface.
