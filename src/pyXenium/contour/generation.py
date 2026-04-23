from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


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
    xenium_pixel_size_um: float = 0.2125,
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
    try:
        return _import_histoseg()
    except Exception as exc:
        if histoseg_root is None:
            raise ImportError(
                "HistoSeg is required for `generate_xenium_explorer_annotations()`. "
                "Install `histoseg` or pass `histoseg_root` pointing to a local checkout."
            ) from exc

    histoseg_root_path = Path(histoseg_root).expanduser().resolve()
    src_root = histoseg_root_path / "src"
    if src_root.exists():
        candidate = src_root
    elif histoseg_root_path.name == "src":
        candidate = histoseg_root_path
    else:
        raise ImportError(
            f"Could not locate a `src` directory under histoseg_root={histoseg_root_path!s}."
        )

    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
    try:
        return _import_histoseg()
    except Exception as exc:
        raise ImportError(
            "Unable to import the required HistoSeg multi-structure contour API from "
            f"{candidate_str}."
        ) from exc


def _import_histoseg() -> Any:
    module = importlib.import_module("histoseg")
    required = (
        "MultiStructureContourConfig",
        "MultiStructureContourResult",
        "MultiStructureSpec",
        "run_multi_structure_contours",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "The installed HistoSeg package is missing the multi-structure contour API: "
            f"{', '.join(missing)}"
        )
    return module
