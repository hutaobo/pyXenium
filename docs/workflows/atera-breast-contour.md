# Atera breast contour workflow

This workflow shows how to go from the raw Atera Xenium export to a
HistoSeg-backed Xenium Explorer annotation export, then into
`pyXenium.contour` for contour import, Voronoi expansion, and shell-based gene
density profiling.

Notebook:

- [RTD tutorial](../tutorials/contour)
- [Source notebook](https://github.com/hutaobo/pyXenium/blob/main/src/pyXenium/notebooks/02_atera_breast_contour_workflow.ipynb)
- [Download notebook](https://raw.githubusercontent.com/hutaobo/pyXenium/main/src/pyXenium/notebooks/02_atera_breast_contour_workflow.ipynb)

This page is the workflow summary. The canonical, same-level
`pyXenium.contour` tutorial is rendered from `docs/tutorials/contour.ipynb`.

## Dataset

The reference notebook is parameterized for:

```text
Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs
```

It expects the standard Xenium export plus these Atera-specific artifacts:

- `analysis/analysis/clustering/gene_expression_graphclust/clusters.csv`
- `WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv`
- `cells.parquet`
- `cell_feature_matrix.h5`
- `transcripts.zarr.zip`

## What the notebook does

1. Reads the Xenium metadata and documents the Atera directory layout.
2. Bridges numeric GraphClust IDs from `clusters.csv` to readable labels and colors from `WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv`.
3. Builds a 7-structure default panel:
   - `11q13 Invasive Tumor Cells`
   - `Basal-like Structured DCIS Cells`
   - `Macrophages`
   - `Plasma Cells`
   - `Endothelial Cells`
   - `Apocrine Cells`
   - `Luminal-like Amorphous DCIS Cells`
4. Calls `pyXenium.contour.generate_xenium_explorer_annotations(...)`, which delegates the actual multi-structure contour generation to HistoSeg.
5. Writes tutorial artifacts under `pyxenium_tutorial_outputs/atera_contour_workflow/` and copies the generated export to `xenium_explorer_annotations.generated.geojson`.
6. Imports the generated GeoJSON into `XeniumSData` with `add_contours_from_geojson(..., id_key="name", pixel_size_um=0.2125)`.
7. Derives structure-level analysis contours from the Explorer polygons, expands them with `expand_contours(..., distance=50.0, mode="voronoi")`, and prepares shell-density inputs.
8. Runs `ring_density` and `smooth_density_by_distance` over the curated marker panel with cache-first helpers so the long-running transcript steps can be resumed.

## Important defaults and caveats

- The legacy `xenium_explorer_annotations.geojson` is treated as a reference artifact and is not overwritten.
- The notebook writes `xenium_explorer_annotations.generated.geojson` beside the legacy file.
- The HistoSeg API keeps `min_cells=500` as its package default. The Atera notebook overrides this to `250` so `Apocrine Cells` are retained in the 7-group export.
- `read_xenium(..., as_="sdata", stream_transcripts=True)` is demonstrated directly on the Atera `transcripts.zarr.zip`, including the real-world case where streamed chunks do not expose `cell_id`.
- Shell profiling is intentionally run on dissolved structure-level analysis contours derived from the Explorer export. This keeps the notebook tractable while preserving the original Explorer polygons for round-tripping and review.

## Outputs

The workflow produces these primary artifacts:

- `pyxenium_tutorial_outputs/atera_contour_workflow/xenium_explorer_annotations.geojson`
- `pyxenium_tutorial_outputs/atera_contour_workflow/xenium_explorer_annotations.csv`
- `pyxenium_tutorial_outputs/atera_contour_workflow/xenium_explorer_annotations_summary.csv`
- `pyxenium_tutorial_outputs/atera_contour_workflow/structure_contour_cell_counts.csv`
- `pyxenium_tutorial_outputs/atera_contour_workflow/structure_contour_metrics.json`
- `xenium_explorer_annotations.generated.geojson`
- Cached ring/profile density tables and derived figures under `pyxenium_tutorial_outputs/atera_contour_workflow/density_cache/`

## Biology readout

The notebook is organized around program-level shell profiles for:

- macrophage markers: `C1QA`, `CD163`, `CSF1R`, `SIGLEC1`, `C3`
- plasma markers: `IGHA1`, `IGHA2`, `JCHAIN`, `IGHM`
- vascular markers: `CDH5`, `MMRN2`, `EPAS1`, `KDR`, `FLT1`
- basal DCIS markers: `SOSTDC1`, `KLK5`, `ITGB6`, `DSC3`, `KRT23`
- apocrine markers: `TAT`, `MYBPC1`
- luminal amorphous markers: `HSPB8`, `CLIC6`, `PIP`, `ESR1`, `PGR`
- invasive or stromal-rim markers: `MMP11`, `COL11A1`, `POSTN`, `COL1A1`, `COL1A2`

The intended summary tables answer three questions:

- Which programs are concentrated inside their assigned structures?
- Which programs peak in the outward 50 µm shell after Voronoi expansion?
- Which sentinel genes show the clearest signed-distance gradients across the contour boundary?
