from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._types import DEFAULT_MECHANOSTRESS_RADII_UM, normalize_radii

_EPS = np.finfo(float).eps


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_boundary_frame(
    boundaries: pd.DataFrame,
    *,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    vertex_id_col: str = "vertex_id",
) -> pd.DataFrame:
    if not isinstance(boundaries, pd.DataFrame):
        raise TypeError(f"boundaries must be a pandas.DataFrame, got {type(boundaries)!r}.")
    _require_columns(boundaries, [cell_id_col, x_col, y_col], name="boundaries")

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
    return frame.dropna(subset=["cell_id", "x", "y"]).sort_values(["cell_id", "vertex_id"], kind="stable")


def _axis_from_points(points_xy: np.ndarray, *, min_vertices: int) -> dict[str, Any]:
    n_vertices = int(points_xy.shape[0])
    centroid = points_xy.mean(axis=0) if n_vertices else np.array([np.nan, np.nan])
    if n_vertices < int(min_vertices):
        return {
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "n_vertices": n_vertices,
            "axis_angle_degrees": np.nan,
            "axis_angle_radians": np.nan,
            "elongation_ratio": np.nan,
            "major_variance": np.nan,
            "minor_variance": np.nan,
            "valid_axis": False,
        }

    centered = points_xy.astype(float, copy=True) - centroid
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        major_variance = minor_variance = np.nan
        angle_radians = np.nan
        elongation_ratio = np.nan
        valid_axis = False
    else:
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        vectors = vectors[:, order]
        major_variance = float(values[0])
        minor_variance = float(values[1])
        major_vector = vectors[:, 0]
        angle_radians = float(np.mod(np.arctan2(major_vector[1], major_vector[0]), np.pi))
        if major_variance <= _EPS:
            elongation_ratio = np.nan
            valid_axis = False
        elif minor_variance <= _EPS:
            elongation_ratio = np.inf
            valid_axis = True
        else:
            elongation_ratio = float(np.sqrt(major_variance / minor_variance))
            valid_axis = bool(np.isfinite(elongation_ratio))

    return {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "n_vertices": n_vertices,
        "axis_angle_degrees": float(np.degrees(angle_radians)) if np.isfinite(angle_radians) else np.nan,
        "axis_angle_radians": angle_radians,
        "elongation_ratio": elongation_ratio,
        "major_variance": major_variance,
        "minor_variance": minor_variance,
        "valid_axis": valid_axis,
    }


def estimate_cell_axes(
    boundaries: pd.DataFrame,
    *,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    vertex_id_col: str = "vertex_id",
    min_vertices: int = 3,
) -> pd.DataFrame:
    """Estimate per-cell axial orientation and elongation ratio from boundary vertices."""

    frame = _normalize_boundary_frame(
        boundaries,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        vertex_id_col=vertex_id_col,
    )
    records: list[dict[str, Any]] = []
    for cell_id, group in frame.groupby("cell_id", sort=False):
        metrics = _axis_from_points(group[["x", "y"]].to_numpy(dtype=float), min_vertices=min_vertices)
        records.append({"cell_id": str(cell_id), **metrics})
    return pd.DataFrame.from_records(records)


def _valid_axis_frame(
    axes: pd.DataFrame,
    *,
    angle_col: str,
    x_col: str | None = None,
    y_col: str | None = None,
) -> pd.DataFrame:
    _require_columns(axes, [angle_col], name="axes")
    frame = axes.copy()
    mask = np.isfinite(pd.to_numeric(frame[angle_col], errors="coerce").to_numpy(dtype=float))
    if "valid_axis" in frame.columns:
        mask &= frame["valid_axis"].fillna(False).astype(bool).to_numpy()
    if x_col is not None and y_col is not None and x_col in frame.columns and y_col in frame.columns:
        coords = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(coords).all(axis=1)
    return frame.loc[mask].copy()


def _kappa_from_rbar(rbar: float) -> float:
    if not np.isfinite(rbar) or rbar < 0:
        return np.nan
    rbar = min(float(rbar), 0.999999)
    if rbar < 0.53:
        return float(2 * rbar + rbar**3 + 5 * rbar**5 / 6)
    if rbar < 0.85:
        return float(-0.4 + 1.39 * rbar + 0.43 / (1 - rbar))
    return float(1 / (rbar**3 - 4 * rbar**2 + 3 * rbar))


def _rayleigh_p_value(rbar: float, n: int) -> float:
    if n <= 0 or not np.isfinite(rbar):
        return np.nan
    z = float(n) * float(rbar) ** 2
    if n < 50:
        correction = 1 + (2 * z - z**2) / (4 * n) - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
        p_value = np.exp(-z) * correction
    else:
        p_value = np.exp(-z)
    return float(np.clip(p_value, 0.0, 1.0))


def _axial_stats(angles: np.ndarray) -> dict[str, float]:
    if angles.size == 0:
        return {
            "n_axes": 0,
            "axial_rbar": np.nan,
            "mean_axis_degrees": np.nan,
            "kappa": np.nan,
            "rayleigh_p": np.nan,
        }
    doubled = 2.0 * angles
    cbar = float(np.cos(doubled).mean())
    sbar = float(np.sin(doubled).mean())
    rbar = float(np.hypot(cbar, sbar))
    mean_axis = float(np.mod(0.5 * np.arctan2(sbar, cbar), np.pi))
    return {
        "n_axes": int(angles.size),
        "axial_rbar": rbar,
        "mean_axis_degrees": float(np.degrees(mean_axis)),
        "kappa": _kappa_from_rbar(rbar),
        "rayleigh_p": _rayleigh_p_value(rbar, int(angles.size)),
    }


def _local_axial_rbar(
    *,
    coords: np.ndarray,
    angles: np.ndarray,
    local_k: int,
) -> np.ndarray:
    if local_k <= 0 or len(angles) <= 1:
        return np.full(len(angles), np.nan, dtype=float)
    k = min(int(local_k) + 1, len(angles))
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]

    local = np.full(len(angles), np.nan, dtype=float)
    for row_index, neighbors in enumerate(indices):
        neighbors = [int(idx) for idx in np.atleast_1d(neighbors) if int(idx) != row_index]
        if not neighbors:
            continue
        doubled = 2.0 * angles[np.asarray(neighbors, dtype=int)]
        local[row_index] = float(np.hypot(np.cos(doubled).mean(), np.sin(doubled).mean()))
    return local


def summarize_axial_orientation(
    axes: pd.DataFrame,
    *,
    angle_col: str = "axis_angle_radians",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    groupby: str | Sequence[str] | None = None,
    local_k: int = 15,
) -> pd.DataFrame:
    """Summarize global and local axial orientation coherence."""

    group_columns = [groupby] if isinstance(groupby, str) else list(groupby or [])
    for column in group_columns:
        if column not in axes.columns:
            raise ValueError(f"axes is missing requested groupby column: {column!r}")

    frame = _valid_axis_frame(axes, angle_col=angle_col, x_col=x_col, y_col=y_col)
    if group_columns:
        grouped = frame.groupby(group_columns, sort=False, dropna=False)
    else:
        grouped = [((), frame)]

    records: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        angles = pd.to_numeric(group[angle_col], errors="coerce").to_numpy(dtype=float)
        record: dict[str, Any] = {column: value for column, value in zip(group_columns, key)}
        record.update(_axial_stats(angles))
        if x_col in group.columns and y_col in group.columns:
            coords = group[[x_col, y_col]].to_numpy(dtype=float)
            local = _local_axial_rbar(coords=coords, angles=angles, local_k=local_k)
            finite = local[np.isfinite(local)]
            record.update(
                {
                    "local_rbar_median": float(np.median(finite)) if finite.size else np.nan,
                    "local_rbar_q25": float(np.quantile(finite, 0.25)) if finite.size else np.nan,
                    "local_rbar_q75": float(np.quantile(finite, 0.75)) if finite.size else np.nan,
                    "local_rbar_n_cells": int(finite.size),
                }
            )
        else:
            record.update(
                {
                    "local_rbar_median": np.nan,
                    "local_rbar_q25": np.nan,
                    "local_rbar_q75": np.nan,
                    "local_rbar_n_cells": 0,
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _summary_stats(values: np.ndarray) -> tuple[float, float, float, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, 0
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.25)),
        float(np.quantile(finite, 0.75)),
        int(finite.size),
    )


def _compute_ane_group(
    group: pd.DataFrame,
    *,
    radii: tuple[float, ...],
    angle_col: str,
    x_col: str,
    y_col: str,
    group_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coords = group[[x_col, y_col]].to_numpy(dtype=float)
    angles = pd.to_numeric(group[angle_col], errors="coerce").to_numpy(dtype=float)
    cell_ids = group["cell_id"].astype(str).to_numpy() if "cell_id" in group.columns else np.arange(len(group)).astype(str)
    tree = cKDTree(coords) if len(group) else None
    summary_records: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []

    for radius in radii:
        neighbor_count = np.zeros(len(group), dtype=float)
        coherence = np.full(len(group), np.nan, dtype=float)
        if tree is not None and len(group):
            neighborhoods = tree.query_ball_point(coords, r=float(radius))
            for idx, neighbors in enumerate(neighborhoods):
                neighbors = [int(item) for item in neighbors if int(item) != idx]
                neighbor_count[idx] = float(len(neighbors))
                if neighbors:
                    delta = angles[np.asarray(neighbors, dtype=int)] - angles[idx]
                    coherence[idx] = float(np.mean((1.0 + np.cos(2.0 * delta)) / 2.0))

        ane = neighbor_count * coherence
        ane_density = ane / (np.pi * float(radius) ** 2)
        coh_median, coh_q25, coh_q75, coh_n = _summary_stats(coherence)
        neigh_median, neigh_q25, neigh_q75, neigh_n = _summary_stats(neighbor_count)
        ane_median, ane_q25, ane_q75, _ = _summary_stats(ane)
        density_median, density_q25, density_q75, _ = _summary_stats(ane_density)

        summary_records.append(
            {
                **group_context,
                "radius_um": float(radius),
                "coh_median": coh_median,
                "coh_q25": coh_q25,
                "coh_q75": coh_q75,
                "coh_n_cells": coh_n,
                "neigh_median": neigh_median,
                "neigh_q25": neigh_q25,
                "neigh_q75": neigh_q75,
                "neigh_n_cells": neigh_n,
                "ANE_median": ane_median,
                "ANE_q25": ane_q25,
                "ANE_q75": ane_q75,
                "ANE_density_median": density_median,
                "ANE_density_q25": density_q25,
                "ANE_density_q75": density_q75,
            }
        )
        for idx, cell_id in enumerate(cell_ids):
            cell_records.append(
                {
                    **group_context,
                    "cell_id": str(cell_id),
                    "radius_um": float(radius),
                    "neighbor_count": float(neighbor_count[idx]),
                    "coherence": float(coherence[idx]) if np.isfinite(coherence[idx]) else np.nan,
                    "ANE": float(ane[idx]) if np.isfinite(ane[idx]) else np.nan,
                    "ANE_density": float(ane_density[idx]) if np.isfinite(ane_density[idx]) else np.nan,
                }
            )

    return summary_records, cell_records


def compute_ane_density(
    axes: pd.DataFrame,
    *,
    radii_um: Sequence[float] = DEFAULT_MECHANOSTRESS_RADII_UM,
    angle_col: str = "axis_angle_radians",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    groupby: str | Sequence[str] | None = None,
    return_cell_metrics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Aligned Neighbour Energy density across radius-based neighborhoods."""

    radii = normalize_radii(radii_um)
    _require_columns(axes, [angle_col, x_col, y_col], name="axes")
    group_columns = [groupby] if isinstance(groupby, str) else list(groupby or [])
    for column in group_columns:
        if column not in axes.columns:
            raise ValueError(f"axes is missing requested groupby column: {column!r}")

    frame = _valid_axis_frame(axes, angle_col=angle_col, x_col=x_col, y_col=y_col)
    if group_columns:
        grouped = frame.groupby(group_columns, sort=False, dropna=False)
    else:
        grouped = [((), frame)]

    summary_records: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        context = {column: value for column, value in zip(group_columns, key)}
        summary, cell_metrics = _compute_ane_group(
            group,
            radii=radii,
            angle_col=angle_col,
            x_col=x_col,
            y_col=y_col,
            group_context=context,
        )
        summary_records.extend(summary)
        cell_records.extend(cell_metrics)

    summary_df = pd.DataFrame.from_records(summary_records)
    if not return_cell_metrics:
        return summary_df
    return summary_df, pd.DataFrame.from_records(cell_records)
