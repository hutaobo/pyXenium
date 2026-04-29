from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ._histoseg import load_histoseg_module
from pyXenium.io._xenium_defaults import DEFAULT_XENIUM_PIXEL_SIZE_UM

_HISTOSEG_CONTOUR_GENERATION_API = (
    "MultiStructureContourConfig",
    "MultiStructureContourResult",
    "MultiStructureSpec",
    "run_multi_structure_contours",
)


def generate_xenium_explorer_annotations(
    dataset_root: str | Path,
    *,
    structures: Sequence[Mapping[str, Any]],
    output_relpath: str | Path,
    clusters_relpath: str | Path = "analysis/analysis/clustering/gene_expression_graphclust/clusters.csv",
    cells_parquet_relpath: str | Path = "cells.parquet",
    histoseg_root: str | Path | None = None,
    barcode_col: str = "Barcode",
    cluster_col: str = "Cluster",
    bins_x: int = 900,
    bins_y: int = 700,
    gaussian_sigma: float = 2.25,
    density_scale_quantile: float = 0.98,
    support_quantile: float = 0.18,
    tissue_quantile: float = 0.06,
    min_dominance: float = 0.34,
    closing_iterations: int = 2,
    opening_iterations: int = 1,
    fill_holes: bool = True,
    min_cells: int = 500,
    min_component_pixels: int = 180,
    xenium_pixel_size_um: float = DEFAULT_XENIUM_PIXEL_SIZE_UM,
    save_preview_png: bool = True,
) -> dict[str, str]:
    """Generate Xenium Explorer-compatible structure annotations via HistoSeg."""

    dataset_root_path = Path(dataset_root).expanduser().resolve()
    clusters_csv = _resolve_dataset_path(dataset_root_path, clusters_relpath)
    cells_parquet = _resolve_dataset_path(dataset_root_path, cells_parquet_relpath)
    output_dir = _resolve_dataset_path(dataset_root_path, output_relpath)

    module = _load_histoseg_module(histoseg_root=histoseg_root)

    config = module.MultiStructureContourConfig(
        clusters_csv=str(clusters_csv),
        cells_parquet=str(cells_parquet),
        out_dir=str(output_dir),
        structures=list(structures),
        barcode_col=barcode_col,
        cluster_col=cluster_col,
        bins_x=bins_x,
        bins_y=bins_y,
        gaussian_sigma=gaussian_sigma,
        density_scale_quantile=density_scale_quantile,
        support_quantile=support_quantile,
        tissue_quantile=tissue_quantile,
        min_dominance=min_dominance,
        closing_iterations=closing_iterations,
        opening_iterations=opening_iterations,
        fill_holes=fill_holes,
        min_cells=min_cells,
        min_component_pixels=min_component_pixels,
        xenium_pixel_size_um=xenium_pixel_size_um,
        save_preview_png=save_preview_png,
    )
    result = module.run_multi_structure_contours(config)
    return {
        "out_dir": str(result.out_dir),
        "geojson": str(result.geojson),
        "csv": str(result.csv),
        "summary": str(result.summary),
        "preview_png": str(result.preview_png) if result.preview_png is not None else "",
        "partition_table": str(result.partition_table),
        "structure_count_csv": str(result.structure_count_csv),
        "metrics_json": str(result.metrics_json),
    }


def _resolve_dataset_path(dataset_root: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path.expanduser().resolve()
    return (dataset_root / path).resolve()


def _load_histoseg_module(*, histoseg_root: str | Path | None) -> Any:
    return load_histoseg_module(
        required=_HISTOSEG_CONTOUR_GENERATION_API,
        histoseg_root=histoseg_root,
        purpose="generate_xenium_explorer_annotations()",
    )
