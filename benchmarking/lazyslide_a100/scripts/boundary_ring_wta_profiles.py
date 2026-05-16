from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
from scipy.stats import spearmanr
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

from pyXenium.multimodal.histoseg_lazyslide import _load_wta_gene_program_library


RING_BINS_UM = (-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate candidate MAZ lead-lag effects with cell/tile ring profiles "
            "around HistoSeg contour boundaries."
        )
    )
    parser.add_argument("--run-dir", required=True, help="mTM WTA run directory.")
    parser.add_argument(
        "--spatialdata-zarr",
        required=True,
        help="Xenium SpatialData zarr containing tables/cells and H&E affine metadata.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <run-dir>/maz_ring_validation.")
    parser.add_argument("--top-programs", type=int, default=5)
    parser.add_argument("--wta-program-library", default="breast_tme_wta_v1")
    parser.add_argument("--max-contours-per-structure", type=int, default=80)
    parser.add_argument("--outer-ring-um", type=float, default=100.0)
    parser.add_argument(
        "--ring-bins-um",
        default=",".join(str(value) for value in RING_BINS_UM),
        help="Comma-separated signed distance bin edges in microns.",
    )
    parser.add_argument("--min-cells-per-ring", type=int, default=3)
    parser.add_argument("--min-tiles-per-ring", type=int, default=1)
    return parser.parse_args()


def _parse_ring_bins(value: str) -> tuple[float, ...]:
    bins = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if len(bins) < 4:
        raise ValueError("At least four ring bin edges are required.")
    if any(right <= left for left, right in zip(bins[:-1], bins[1:])):
        raise ValueError("Ring bin edges must be strictly increasing.")
    if min(bins) >= 0 or max(bins) <= 0:
        raise ValueError("Ring bins must include negative and positive distances.")
    return bins


def _read_table(base: Path, stem: str) -> pd.DataFrame:
    parquet = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {base}")


def _load_he_affine(spatialdata_zarr: Path) -> tuple[np.ndarray, float]:
    metadata_path = spatialdata_zarr / "images" / "he" / "zarr.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    attrs = payload.get("attributes", {})
    affine = np.asarray(attrs["image_to_xenium_affine"], dtype=float)
    xenium_pixel_size_um = float(attrs.get("xenium_pixel_size_um", attrs.get("pixel_size_um", 1.0)))
    return affine, xenium_pixel_size_um


def _image_geometry_to_microns(
    geometry: BaseGeometry,
    *,
    affine: np.ndarray,
    xenium_pixel_size_um: float,
) -> BaseGeometry:
    def map_xy(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        out_x = affine[0, 0] * x_arr + affine[0, 1] * y_arr + affine[0, 2]
        out_y = affine[1, 0] * x_arr + affine[1, 1] * y_arr + affine[1, 2]
        return out_x * xenium_pixel_size_um, out_y * xenium_pixel_size_um

    return shapely_transform(map_xy, geometry)


def _point_image_to_microns(
    x: float,
    y: float,
    *,
    affine: np.ndarray,
    xenium_pixel_size_um: float,
) -> Point:
    out_x = affine[0, 0] * float(x) + affine[0, 1] * float(y) + affine[0, 2]
    out_y = affine[1, 0] * float(x) + affine[1, 1] * float(y) + affine[1, 2]
    return Point(out_x * xenium_pixel_size_um, out_y * xenium_pixel_size_um)


def _geometry(value: Any) -> BaseGeometry | None:
    if isinstance(value, BaseGeometry):
        return value
    if pd.isna(value):
        return None
    try:
        return wkt.loads(str(value))
    except Exception:
        return None


def _resolve_gene_indices(adata: ad.AnnData, genes: list[str]) -> list[tuple[str, int]]:
    lookup: dict[str, int] = {}
    for index, value in enumerate(adata.var_names.astype(str)):
        lookup.setdefault(str(value).upper(), int(index))
    for column in ("name", "gene_name", "gene_symbol", "symbol", "id"):
        if column not in adata.var.columns:
            continue
        for index, value in enumerate(adata.var[column].astype(str)):
            lookup.setdefault(str(value).upper(), int(index))
    resolved: list[tuple[str, int]] = []
    seen: set[int] = set()
    for gene in genes:
        index = lookup.get(str(gene).upper())
        if index is None or index in seen:
            continue
        resolved.append((str(gene), int(index)))
        seen.add(int(index))
    return resolved


def _cell_program_scores(
    adata: ad.AnnData,
    programs: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    requested = []
    for genes in programs.values():
        requested.extend(str(gene) for gene in genes)
    requested = list(dict.fromkeys(requested))
    resolved = _resolve_gene_indices(adata, requested)
    if not resolved:
        raise ValueError("No WTA program genes resolved in AnnData table.")
    labels = [label for label, _ in resolved]
    indices = [index for _, index in resolved]
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    if sparse.issparse(matrix):
        csr = matrix.tocsr()
        values = csr[:, indices].toarray()
        library_size = np.asarray(csr.sum(axis=1)).ravel().astype(float)
    else:
        dense = np.asarray(matrix, dtype=float)
        values = dense[:, indices]
        library_size = np.asarray(dense.sum(axis=1)).ravel().astype(float)
    library_size[library_size <= 0] = 1.0
    normalized = np.log1p((np.asarray(values, dtype=float) / library_size[:, None]) * 1e4)
    gene_frame = pd.DataFrame(normalized, columns=labels, index=adata.obs_names.astype(str))
    gene_z = (gene_frame - gene_frame.mean(axis=0)) / gene_frame.std(axis=0, ddof=0).replace(0.0, np.nan)
    score_frame = pd.DataFrame(index=gene_z.index)
    for name, genes in programs.items():
        present = [gene for gene in genes if gene in gene_z.columns]
        if len(present) >= 2:
            score_frame[f"program__wta_{name}"] = gene_z.loc[:, present].mean(axis=1)
    return score_frame


def _query_indices(tree: STRtree, geometry: BaseGeometry, points: list[Point]) -> np.ndarray:
    try:
        result = tree.query(geometry)
    except Exception:
        return np.arange(len(points), dtype=int)
    arr = np.asarray(result)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)
    point_to_index = {id(point): index for index, point in enumerate(points)}
    return np.asarray([point_to_index[id(point)] for point in arr if id(point) in point_to_index], dtype=int)


def _signed_distance_um(geometry: BaseGeometry, point: Point) -> float:
    distance = float(geometry.boundary.distance(point))
    return -distance if geometry.covers(point) else distance


def _ring_label(value: float, bins: tuple[float, ...] = RING_BINS_UM) -> tuple[str, float] | None:
    for left, right in zip(bins[:-1], bins[1:]):
        if left <= value < right or (right == bins[-1] and value <= right):
            return f"{left:g}_to_{right:g}", (left + right) / 2.0
    return None


def _select_program_specs(leaderboard: pd.DataFrame, top_programs: int) -> pd.DataFrame:
    specs = leaderboard.head(int(top_programs)).copy()
    required = {"pathway", "molecular_feature", "best_image_feature", "partial_spearman_rho"}
    missing = required.difference(specs.columns)
    if missing:
        raise ValueError(f"Missing WTA leaderboard columns: {sorted(missing)}")
    return specs


def _select_contours(contours: pd.DataFrame, summary: pd.DataFrame, max_per_structure: int) -> pd.DataFrame:
    tiled = summary.loc[pd.to_numeric(summary.get("n_tiles", 0), errors="coerce").fillna(0).gt(0)].copy()
    tiled = tiled.loc[:, ["contour_id", "n_tiles"]].copy()
    merged = contours.merge(tiled, on="contour_id", how="inner")
    merged = merged.sort_values(["assigned_structure", "n_tiles"], ascending=[True, False], kind="stable")
    return (
        merged.groupby("assigned_structure", sort=True, group_keys=False)
        .head(int(max_per_structure))
        .reset_index(drop=True)
    )


def _build_cell_ring_profiles(
    *,
    contours: pd.DataFrame,
    cell_points: list[Point],
    cell_tree: STRtree,
    cell_scores: pd.DataFrame,
    program_columns: list[str],
    outer_ring_um: float,
    ring_bins_um: tuple[float, ...],
    min_cells_per_ring: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, contour in contours.iterrows():
        geometry = contour["geometry_um"]
        if geometry is None or geometry.is_empty:
            continue
        query_geometry = geometry.buffer(float(outer_ring_um))
        candidate_indices = _query_indices(cell_tree, query_geometry, cell_points)
        if candidate_indices.size == 0:
            continue
        ring_records: list[tuple[int, str, float]] = []
        for index in candidate_indices:
            distance = _signed_distance_um(geometry, cell_points[int(index)])
            label = _ring_label(distance, ring_bins_um)
            if label is None:
                continue
            ring_name, ring_center = label
            ring_records.append((int(index), ring_name, ring_center))
        if not ring_records:
            continue
        ring_frame = pd.DataFrame(ring_records, columns=["cell_index", "ring", "ring_center_um"])
        for program in program_columns:
            values = cell_scores.iloc[ring_frame["cell_index"].to_numpy(dtype=int)][program].to_numpy(dtype=float)
            ring_frame["_value"] = values
            grouped = ring_frame.dropna(subset=["_value"]).groupby(["ring", "ring_center_um"], sort=True)
            for (ring, ring_center), group in grouped:
                if len(group) < int(min_cells_per_ring):
                    continue
                rows.append(
                    {
                        "contour_id": contour["contour_id"],
                        "assigned_structure": contour["assigned_structure"],
                        "feature_type": "molecular",
                        "feature": program,
                        "ring": ring,
                        "ring_center_um": float(ring_center),
                        "mean_value": float(group["_value"].mean()),
                        "n_observations": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def _build_tile_ring_profiles(
    *,
    contours: pd.DataFrame,
    tile_features: pd.DataFrame,
    tile_points: list[Point],
    tile_tree: STRtree,
    image_feature_map: dict[str, str],
    outer_ring_um: float,
    ring_bins_um: tuple[float, ...],
    min_tiles_per_ring: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tile_values = tile_features.loc[:, list(image_feature_map.values())].apply(pd.to_numeric, errors="coerce")
    for _, contour in contours.iterrows():
        geometry = contour["geometry_um"]
        if geometry is None or geometry.is_empty:
            continue
        query_geometry = geometry.buffer(float(outer_ring_um))
        candidate_indices = _query_indices(tile_tree, query_geometry, tile_points)
        if candidate_indices.size == 0:
            continue
        ring_records: list[tuple[int, str, float]] = []
        for index in candidate_indices:
            distance = _signed_distance_um(geometry, tile_points[int(index)])
            label = _ring_label(distance, ring_bins_um)
            if label is None:
                continue
            ring_name, ring_center = label
            ring_records.append((int(index), ring_name, ring_center))
        if not ring_records:
            continue
        ring_frame = pd.DataFrame(ring_records, columns=["tile_index", "ring", "ring_center_um"])
        for feature, source_feature in image_feature_map.items():
            values = tile_values.iloc[ring_frame["tile_index"].to_numpy(dtype=int)][source_feature].to_numpy(dtype=float)
            ring_frame["_value"] = values
            grouped = ring_frame.dropna(subset=["_value"]).groupby(["ring", "ring_center_um"], sort=True)
            for (ring, ring_center), group in grouped:
                if len(group) < int(min_tiles_per_ring):
                    continue
                rows.append(
                    {
                        "contour_id": contour["contour_id"],
                        "assigned_structure": contour["assigned_structure"],
                        "feature_type": "image",
                        "feature": feature,
                        "ring": ring,
                        "ring_center_um": float(ring_center),
                        "mean_value": float(group["_value"].mean()),
                        "n_observations": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return numeric * np.nan
    return (numeric - numeric.mean()) / std


def _gradient_peak_center(profile: pd.DataFrame, value_column: str = "z_mean") -> tuple[float, float]:
    ordered = profile.sort_values("ring_center_um")
    x = ordered["ring_center_um"].to_numpy(dtype=float)
    y = ordered[value_column].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan, np.nan
    gradients = np.diff(y) / np.diff(x)
    if gradients.size == 0 or not np.isfinite(gradients).any():
        return np.nan, np.nan
    idx = int(np.nanargmax(np.abs(gradients)))
    return float((x[idx] + x[idx + 1]) / 2.0), float(gradients[idx])


def _classify_lag(molecular_peak: float, image_peak: float, mol_amp: float, img_amp: float) -> str:
    if not np.isfinite(molecular_peak) or not np.isfinite(image_peak):
        return "insufficient_ring_signal"
    if max(abs(mol_amp), abs(img_amp)) < 0.03:
        return "weak_ring_gradient"
    delta = molecular_peak - image_peak
    if delta > 25.0:
        return "molecular_lead"
    if delta < -25.0:
        return "morphology_lead"
    return "coupled_boundary_zone"


def _summarize_lead_lag(profile: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec_lookup = {row["molecular_feature"]: row for _, row in specs.iterrows()}
    for target, spec in spec_lookup.items():
        image = str(spec["best_image_feature"])
        sign = float(np.sign(spec["partial_spearman_rho"])) or 1.0
        for structure in sorted(profile["assigned_structure"].dropna().astype(str).unique()):
            mol = profile.loc[
                (profile["assigned_structure"].astype(str) == structure)
                & (profile["feature_type"] == "molecular")
                & (profile["feature"] == target)
            ].copy()
            img = profile.loc[
                (profile["assigned_structure"].astype(str) == structure)
                & (profile["feature_type"] == "image")
                & (profile["feature"] == image)
            ].copy()
            if mol.empty or img.empty:
                continue
            mol_summary = (
                mol.groupby("ring_center_um", as_index=False)
                .agg(mean_value=("mean_value", "mean"), n_observations=("n_observations", "sum"))
                .sort_values("ring_center_um")
            )
            img_summary = (
                img.groupby("ring_center_um", as_index=False)
                .agg(mean_value=("mean_value", "mean"), n_observations=("n_observations", "sum"))
                .sort_values("ring_center_um")
            )
            mol_summary["z_mean"] = _zscore(mol_summary["mean_value"])
            img_summary["z_mean"] = sign * _zscore(img_summary["mean_value"])
            mol_peak, mol_grad = _gradient_peak_center(mol_summary)
            img_peak, img_grad = _gradient_peak_center(img_summary)
            joined = mol_summary[["ring_center_um", "z_mean"]].merge(
                img_summary[["ring_center_um", "z_mean"]],
                on="ring_center_um",
                how="inner",
                suffixes=("_molecular", "_image"),
            )
            coupling_r = np.nan
            coupling_p = np.nan
            if len(joined) >= 4:
                result = spearmanr(joined["z_mean_molecular"], joined["z_mean_image"], nan_policy="omit")
                coupling_r = float(result.statistic)
                coupling_p = float(result.pvalue)
            rows.append(
                {
                    "program": str(spec["pathway"]),
                    "molecular_feature": target,
                    "image_feature": image,
                    "assigned_structure": structure,
                    "global_partial_spearman_rho": float(spec["partial_spearman_rho"]),
                    "n_molecular_observations": int(mol_summary["n_observations"].sum()),
                    "n_image_observations": int(img_summary["n_observations"].sum()),
                    "molecular_gradient_peak_center_um": mol_peak,
                    "image_gradient_peak_center_um": img_peak,
                    "molecular_minus_image_peak_um": (
                        float(mol_peak - img_peak)
                        if np.isfinite(mol_peak) and np.isfinite(img_peak)
                        else np.nan
                    ),
                    "molecular_peak_gradient": mol_grad,
                    "image_peak_gradient_oriented": img_grad,
                    "ring_profile_spearman_rho": coupling_r,
                    "ring_profile_spearman_p_value": coupling_p,
                    "lead_lag_class": _classify_lag(mol_peak, img_peak, mol_grad, img_grad),
                    "method": (
                        "ring-level cell WTA program profile and tile embedding profile "
                        "across signed contour-boundary distance bins"
                    ),
                }
            )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            ["lead_lag_class", "n_molecular_observations", "n_image_observations"],
            ascending=[True, False, False],
            kind="stable",
        ).reset_index(drop=True)
    return report


def _resolve_tile_image_feature_map(specs: pd.DataFrame, tile_features: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    available = set(str(column) for column in tile_features.columns)
    for feature in specs["best_image_feature"].astype(str).unique():
        candidates = [feature]
        if feature.endswith("__mean"):
            candidates.append(feature[: -len("__mean")])
        candidates.append(feature.replace("__mean", ""))
        for candidate in candidates:
            if candidate in available:
                mapping[feature] = candidate
                break
    return mapping


def _plot_profiles(profile: pd.DataFrame, report: pd.DataFrame, output_pdf: Path) -> None:
    top = report.head(12).copy()
    with PdfPages(output_pdf) as pdf:
        for _, row in top.iterrows():
            target = row["molecular_feature"]
            image = row["image_feature"]
            structure = row["assigned_structure"]
            sign = float(np.sign(row["global_partial_spearman_rho"])) or 1.0
            mol = profile.loc[
                (profile["assigned_structure"].astype(str) == str(structure))
                & (profile["feature_type"] == "molecular")
                & (profile["feature"] == target)
            ].copy()
            img = profile.loc[
                (profile["assigned_structure"].astype(str) == str(structure))
                & (profile["feature_type"] == "image")
                & (profile["feature"] == image)
            ].copy()
            mol_summary = mol.groupby("ring_center_um", as_index=False)["mean_value"].mean()
            img_summary = img.groupby("ring_center_um", as_index=False)["mean_value"].mean()
            mol_summary["z_mean"] = _zscore(mol_summary["mean_value"])
            img_summary["z_mean"] = sign * _zscore(img_summary["mean_value"])
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            ax.plot(
                mol_summary["ring_center_um"],
                mol_summary["z_mean"],
                marker="o",
                label=str(row["program"]),
            )
            ax.plot(
                img_summary["ring_center_um"],
                img_summary["z_mean"],
                marker="s",
                label=f"{image} (oriented)",
            )
            ax.set_title(f"{structure} {row['program']} ring profile: {row['lead_lag_class']}")
            ax.set_xlabel("Signed distance from contour boundary (um; negative = inside)")
            ax.set_ylabel("Ring mean z-score")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "maz_ring_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    spatialdata_zarr = Path(args.spatialdata_zarr).expanduser().resolve()
    ring_bins_um = _parse_ring_bins(args.ring_bins_um)
    affine, xenium_pixel_size_um = _load_he_affine(spatialdata_zarr)
    contours = _read_table(run_dir, "image_contours")
    summary = _read_table(run_dir, "contour_multimodal_summary")
    tile_features = _read_table(run_dir, "tile_features")
    leaderboard = _read_table(run_dir, "wta_pathway_partial_correlations")
    specs = _select_program_specs(leaderboard, args.top_programs)

    contours = _select_contours(contours, summary, args.max_contours_per_structure)
    contours["geometry"] = contours["geometry_wkt"].map(_geometry)
    contours["geometry_um"] = contours["geometry"].map(
        lambda geom: _image_geometry_to_microns(
            geom,
            affine=affine,
            xenium_pixel_size_um=xenium_pixel_size_um,
        )
        if geom is not None
        else None
    )

    adata = ad.read_zarr(spatialdata_zarr / "tables" / "cells")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    cell_points = [Point(float(x), float(y)) for x, y in coords[:, :2]]
    cell_tree = STRtree(cell_points)
    selected_program_names = [str(value) for value in specs["pathway"].tolist()]
    program_library = _load_wta_gene_program_library(args.wta_program_library)
    programs = {name: program_library[name] for name in selected_program_names if name in program_library}
    cell_scores = _cell_program_scores(adata, programs)
    program_columns = [column for column in specs["molecular_feature"].astype(str) if column in cell_scores.columns]
    image_feature_map = _resolve_tile_image_feature_map(specs, tile_features)

    tile_features = tile_features.copy()
    tile_features["geometry"] = tile_features["geometry_wkt"].map(_geometry)
    tile_features = tile_features.loc[tile_features["geometry"].notna()].reset_index(drop=True)
    tile_points = [
        _point_image_to_microns(
            float(geometry.centroid.x),
            float(geometry.centroid.y),
            affine=affine,
            xenium_pixel_size_um=xenium_pixel_size_um,
        )
        for geometry in tile_features["geometry"]
    ]
    tile_tree = STRtree(tile_points)

    cell_profile = _build_cell_ring_profiles(
        contours=contours,
        cell_points=cell_points,
        cell_tree=cell_tree,
        cell_scores=cell_scores,
        program_columns=program_columns,
        outer_ring_um=float(args.outer_ring_um),
        ring_bins_um=ring_bins_um,
        min_cells_per_ring=int(args.min_cells_per_ring),
    )
    tile_profile = _build_tile_ring_profiles(
        contours=contours,
        tile_features=tile_features,
        tile_points=tile_points,
        tile_tree=tile_tree,
        image_feature_map=image_feature_map,
        outer_ring_um=float(args.outer_ring_um),
        ring_bins_um=ring_bins_um,
        min_tiles_per_ring=int(args.min_tiles_per_ring),
    )
    profile = pd.concat([cell_profile, tile_profile], ignore_index=True)
    profile.to_csv(output_dir / "MAZ_RingLevel_Profile.csv", index=False)
    report = _summarize_lead_lag(profile, specs)
    report.to_csv(output_dir / "MAZ_RingLevel_LeadLag_Report.csv", index=False)
    _plot_profiles(profile, report, output_dir / "MAZ_RingLevel_Profile_Panels.pdf")
    manifest = {
        "status": "completed",
        "run_dir": str(run_dir),
        "spatialdata_zarr": str(spatialdata_zarr),
        "n_contours": int(len(contours)),
        "n_cells": int(len(cell_points)),
        "n_tiles": int(len(tile_points)),
        "programs": selected_program_names,
        "program_columns": program_columns,
        "image_features": list(image_feature_map.keys()),
        "image_source_features": image_feature_map,
        "ring_bins_um": list(ring_bins_um),
        "outputs": {
            "profile": "MAZ_RingLevel_Profile.csv",
            "report": "MAZ_RingLevel_LeadLag_Report.csv",
            "panels": "MAZ_RingLevel_Profile_Panels.pdf",
        },
    }
    (output_dir / "MAZ_RingLevel_Manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'MAZ_RingLevel_Profile.csv'} rows={len(profile)}")
    print(f"Wrote {output_dir / 'MAZ_RingLevel_LeadLag_Report.csv'} rows={len(report)}")


if __name__ == "__main__":
    main()
