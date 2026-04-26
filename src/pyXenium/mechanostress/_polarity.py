from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _shoelace_area(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return np.nan
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _boundary_shape_metrics(
    boundaries: pd.DataFrame,
    *,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    vertex_id_col: str = "vertex_id",
    prefix: str,
) -> pd.DataFrame:
    _require_columns(boundaries, [cell_id_col, x_col, y_col], name=f"{prefix}_boundaries")
    frame = boundaries.copy()
    if vertex_id_col not in frame.columns:
        frame[vertex_id_col] = frame.groupby(cell_id_col, sort=False).cumcount()
    frame = frame[[cell_id_col, vertex_id_col, x_col, y_col]].rename(
        columns={cell_id_col: "cell_id", vertex_id_col: "vertex_id", x_col: "x", y_col: "y"}
    )
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["vertex_id"] = pd.to_numeric(frame["vertex_id"], errors="coerce").fillna(0).astype("int64")
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["cell_id", "x", "y"]).sort_values(["cell_id", "vertex_id"], kind="stable")

    records: list[dict[str, Any]] = []
    for cell_id, group in frame.groupby("cell_id", sort=False):
        points = group[["x", "y"]].to_numpy(dtype=float)
        centroid = points.mean(axis=0)
        area = _shoelace_area(points)
        records.append(
            {
                "cell_id": str(cell_id),
                f"{prefix}_centroid_x": float(centroid[0]),
                f"{prefix}_centroid_y": float(centroid[1]),
                f"{prefix}_area": area,
                f"{prefix}_equivalent_radius_um": float(np.sqrt(area / np.pi)) if np.isfinite(area) and area > 0 else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {column.lower(): column for column in frame.columns.astype(str)}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _metrics_from_cell_table(cell_table: pd.DataFrame, *, cell_id_col: str) -> pd.DataFrame:
    _require_columns(cell_table, [cell_id_col], name="cell_table")
    cell_x = _find_column(cell_table, ["cell_centroid_x", "x_centroid", "x", "centroid_x"])
    cell_y = _find_column(cell_table, ["cell_centroid_y", "y_centroid", "y", "centroid_y"])
    nucleus_x = _find_column(cell_table, ["nucleus_centroid_x", "nucleus_x", "nucleus_x_centroid"])
    nucleus_y = _find_column(cell_table, ["nucleus_centroid_y", "nucleus_y", "nucleus_y_centroid"])
    if cell_x is None or cell_y is None or nucleus_x is None or nucleus_y is None:
        raise ValueError("cell_table must contain cell and nucleus centroid columns.")
    area_col = _find_column(cell_table, ["cell_area", "area"])
    frame = pd.DataFrame(
        {
            "cell_id": cell_table[cell_id_col].astype(str),
            "cell_centroid_x": pd.to_numeric(cell_table[cell_x], errors="coerce"),
            "cell_centroid_y": pd.to_numeric(cell_table[cell_y], errors="coerce"),
            "nucleus_centroid_x": pd.to_numeric(cell_table[nucleus_x], errors="coerce"),
            "nucleus_centroid_y": pd.to_numeric(cell_table[nucleus_y], errors="coerce"),
        }
    )
    if area_col is not None:
        area = pd.to_numeric(cell_table[area_col], errors="coerce")
        frame["cell_area"] = area
        frame["cell_equivalent_radius_um"] = np.sqrt(area / np.pi)
    else:
        frame["cell_area"] = np.nan
        frame["cell_equivalent_radius_um"] = np.nan
    return frame


def compute_cell_polarity(
    *,
    cell_boundaries: pd.DataFrame | None = None,
    nucleus_boundaries: pd.DataFrame | None = None,
    cell_table: pd.DataFrame | None = None,
    cell_id_col: str = "cell_id",
    offset_norm_threshold: float = 0.30,
) -> pd.DataFrame:
    """Compute cell polarity from cell and nucleus centroid offsets."""

    if cell_table is not None:
        metrics = _metrics_from_cell_table(cell_table, cell_id_col=cell_id_col)
    else:
        if cell_boundaries is None or nucleus_boundaries is None:
            raise ValueError("Provide either cell_table or both cell_boundaries and nucleus_boundaries.")
        cell_metrics = _boundary_shape_metrics(cell_boundaries, cell_id_col=cell_id_col, prefix="cell")
        nucleus_metrics = _boundary_shape_metrics(nucleus_boundaries, cell_id_col=cell_id_col, prefix="nucleus")
        metrics = cell_metrics.merge(nucleus_metrics, on="cell_id", how="inner")

    metrics = metrics.dropna(subset=["cell_centroid_x", "cell_centroid_y", "nucleus_centroid_x", "nucleus_centroid_y"]).copy()
    metrics["polarity_dx"] = metrics["nucleus_centroid_x"] - metrics["cell_centroid_x"]
    metrics["polarity_dy"] = metrics["nucleus_centroid_y"] - metrics["cell_centroid_y"]
    metrics["offset_distance_um"] = np.hypot(metrics["polarity_dx"], metrics["polarity_dy"])
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["offset_norm"] = metrics["offset_distance_um"] / metrics["cell_equivalent_radius_um"]
    metrics["polarity_angle_degrees"] = np.mod(np.degrees(np.arctan2(metrics["polarity_dy"], metrics["polarity_dx"])), 360.0)
    metrics["polarized"] = metrics["offset_norm"] >= float(offset_norm_threshold)
    metrics.loc[~np.isfinite(metrics["offset_norm"]), "polarized"] = False
    return metrics.reset_index(drop=True)


def summarize_cell_polarity(
    polarity: pd.DataFrame,
    *,
    groupby: str | Sequence[str] | None = None,
    sample_id: str | None = None,
) -> pd.DataFrame:
    """Summarize polarized cell fractions and offset magnitudes."""

    if polarity.empty:
        columns = ["sample_id", "n_cells", "n_polarized", "polarized_fraction", "offset_norm_median", "offset_distance_median_um"]
        return pd.DataFrame(columns=columns)
    group_columns = [groupby] if isinstance(groupby, str) else list(groupby or [])
    for column in group_columns:
        if column not in polarity.columns:
            raise ValueError(f"polarity is missing requested groupby column: {column!r}")

    grouped = polarity.groupby(group_columns, sort=False, dropna=False) if group_columns else [((), polarity)]
    records: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        record: dict[str, Any] = {"sample_id": sample_id}
        record.update({column: value for column, value in zip(group_columns, key)})
        n_cells = int(len(group))
        n_polarized = int(group["polarized"].fillna(False).astype(bool).sum())
        record.update(
            {
                "n_cells": n_cells,
                "n_polarized": n_polarized,
                "polarized_fraction": float(n_polarized / n_cells) if n_cells else np.nan,
                "offset_norm_median": float(pd.to_numeric(group["offset_norm"], errors="coerce").median()),
                "offset_distance_median_um": float(pd.to_numeric(group["offset_distance_um"], errors="coerce").median()),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)
