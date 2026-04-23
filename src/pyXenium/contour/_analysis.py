from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
from shapely import STRtree, contains, distance, intersects, points

from ._geometry import contour_frame_to_geometry_table
from pyXenium.io.sdata_model import XeniumSData

__all__ = ["ring_density", "smooth_density_by_distance"]

_DEFAULT_CONTOUR_METADATA_COLUMNS = (
    "assigned_structure",
    "classification_name",
    "annotation_source",
    "structure_id",
)
_GAUSSIAN_KERNEL = "gaussian"
_KERNEL_TRUNCATION = 4.0
_SMALL_CONTOUR_FASTPATH_THRESHOLD = 32


def ring_density(
    sdata: XeniumSData,
    *,
    contour_key: str,
    target: str = "transcripts",
    contour_query: str | None = None,
    target_query: str | None = None,
    feature_key: str = "gene_name",
    feature_values: str | Sequence[str] | None = None,
    inward: float,
    outward: float,
    ring_width: float,
) -> pd.DataFrame:
    """
    Compute inward/outward ring density around contour annotations in ``XeniumSData``.
    """

    _validate_distance_window(inward=inward, outward=outward)
    ring_width = float(ring_width)
    if ring_width <= 0:
        raise ValueError("`ring_width` must be greater than 0.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    intervals = _build_ring_intervals(
        ring_width=ring_width,
        inward=inward,
        outward=outward,
    )
    interval_edges = np.asarray([intervals[0][0], *[end for _, end in intervals]], dtype=float)
    support_geometries = np.asarray(
        [
            _build_support_geometry(geom, inward=inward, outward=outward, padding=0.0)
            for geom in contour_table["geometry"]
        ],
        dtype=object,
    )

    counts = _count_targets_by_interval(
        sdata=sdata,
        contour_table=contour_table,
        support_geometries=support_geometries,
        interval_edges=interval_edges,
        target=target,
        target_query=target_query,
        feature_key=feature_key,
        feature_values=feature_values,
    )
    areas = np.asarray(
        [
            [_shell_area(geometry, ring_start, ring_end) for ring_start, ring_end in intervals]
            for geometry in contour_table["geometry"]
        ],
        dtype=float,
    )

    feature_key_value, feature_value_payload = _feature_context(
        target=target,
        feature_key=feature_key,
        feature_values=feature_values,
    )
    return _assemble_ring_result(
        contour_key=contour_key,
        contour_table=contour_table,
        target=target,
        feature_key=feature_key_value,
        feature_values=feature_value_payload,
        intervals=intervals,
        counts=counts,
        areas=areas,
    )


def smooth_density_by_distance(
    sdata: XeniumSData,
    *,
    contour_key: str,
    target: str = "transcripts",
    contour_query: str | None = None,
    target_query: str | None = None,
    feature_key: str = "gene_name",
    feature_values: str | Sequence[str] | None = None,
    inward: float,
    outward: float,
    bandwidth: float,
    grid_step: float | None = None,
) -> pd.DataFrame:
    """
    Compute a continuous signed-distance density profile around contour annotations.
    """

    _validate_distance_window(inward=inward, outward=outward)
    bandwidth = float(bandwidth)
    if bandwidth <= 0:
        raise ValueError("`bandwidth` must be greater than 0.")
    if grid_step is None:
        grid_step = bandwidth / 4.0
    grid_step = float(grid_step)
    if grid_step <= 0:
        raise ValueError("`grid_step` must be greater than 0.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    signed_distance_grid = _build_signed_distance_grid(
        inward=inward,
        outward=outward,
        grid_step=grid_step,
    )
    support_geometries = np.asarray(
        [
            _build_support_geometry(
                geometry,
                inward=inward,
                outward=outward,
                padding=_KERNEL_TRUNCATION * bandwidth,
            )
            for geometry in contour_table["geometry"]
        ],
        dtype=object,
    )
    count_density = _count_smoothed_targets(
        sdata=sdata,
        contour_table=contour_table,
        support_geometries=support_geometries,
        signed_distance_grid=signed_distance_grid,
        target=target,
        target_query=target_query,
        feature_key=feature_key,
        feature_values=feature_values,
        inward=inward,
        outward=outward,
        bandwidth=bandwidth,
    )
    geometry_measure = _compute_geometry_measure(
        contour_geometries=np.asarray(contour_table["geometry"], dtype=object),
        signed_distance_grid=signed_distance_grid,
        inward=inward,
        outward=outward,
        grid_step=grid_step,
    )

    feature_key_value, feature_value_payload = _feature_context(
        target=target,
        feature_key=feature_key,
        feature_values=feature_values,
    )
    return _assemble_smooth_result(
        contour_key=contour_key,
        contour_table=contour_table,
        target=target,
        feature_key=feature_key_value,
        feature_values=feature_value_payload,
        signed_distance_grid=signed_distance_grid,
        bandwidth=bandwidth,
        grid_step=grid_step,
        count_density=count_density,
        geometry_measure=geometry_measure,
    )


def _validate_distance_window(*, inward: float, outward: float) -> None:
    inward = float(inward)
    outward = float(outward)
    if inward < 0:
        raise ValueError("`inward` must be non-negative.")
    if outward < 0:
        raise ValueError("`outward` must be non-negative.")
    if inward == 0 and outward == 0:
        raise ValueError("At least one of `inward` or `outward` must be greater than 0.")


def _prepare_contours(
    *,
    sdata: XeniumSData,
    contour_key: str,
    contour_query: str | None,
) -> pd.DataFrame:
    if contour_key not in sdata.shapes:
        raise KeyError(f"Contour key `{contour_key}` not found in `sdata.shapes`.")

    frame = sdata.shapes[contour_key].copy()
    if contour_query is not None:
        frame = frame.query(contour_query, engine="python")
    if frame.empty:
        raise ValueError("No contours remain after applying the requested filters.")
    return contour_frame_to_geometry_table(frame, contour_key=contour_key)


def _feature_context(
    *,
    target: str,
    feature_key: str,
    feature_values: str | Sequence[str] | None,
) -> tuple[str | None, Any]:
    normalized_target = _normalize_target(target)
    if normalized_target == "cells":
        if feature_values is not None:
            raise ValueError("`feature_values` can only be used when `target='transcripts'`.")
        return None, None
    return str(feature_key), _feature_value_payload(feature_values)


def _count_targets_by_interval(
    *,
    sdata: XeniumSData,
    contour_table: pd.DataFrame,
    support_geometries: np.ndarray,
    interval_edges: np.ndarray,
    target: str,
    target_query: str | None,
    feature_key: str,
    feature_values: str | Sequence[str] | None,
) -> np.ndarray:
    if len(contour_table) <= _SMALL_CONTOUR_FASTPATH_THRESHOLD:
        return _count_targets_by_interval_small_n(
            contour_table=contour_table,
            support_geometries=support_geometries,
            interval_edges=interval_edges,
            target_frames=_iter_target_frames(
                sdata=sdata,
                target=target,
                target_query=target_query,
                feature_key=feature_key,
                feature_values=feature_values,
            ),
        )

    contour_geometries = np.asarray(contour_table["geometry"], dtype=object)
    contour_boundaries = np.asarray([geometry.boundary for geometry in contour_geometries], dtype=object)
    tree = STRtree(support_geometries)
    counts = np.zeros((len(contour_table), len(interval_edges) - 1), dtype=np.int64)

    for target_frame in _iter_target_frames(
        sdata=sdata,
        target=target,
        target_query=target_query,
        feature_key=feature_key,
        feature_values=feature_values,
    ):
        if target_frame.empty:
            continue

        target_geometries = np.asarray(
            points(
                target_frame["x"].to_numpy(dtype=float),
                target_frame["y"].to_numpy(dtype=float),
            ),
            dtype=object,
        )
        pair_idx = tree.query(target_geometries, predicate="intersects")
        if pair_idx.size == 0:
            continue

        point_idx = pair_idx[0]
        contour_idx = pair_idx[1]
        signed = _signed_distance(
            point_geometries=target_geometries.take(point_idx),
            contour_geometries=contour_geometries.take(contour_idx),
            contour_boundaries=contour_boundaries.take(contour_idx),
        )
        ring_idx = _assign_intervals(signed_distances=signed, interval_edges=interval_edges)
        valid = ring_idx >= 0
        if np.any(valid):
            np.add.at(counts, (contour_idx[valid], ring_idx[valid]), 1)

    return counts


def _count_targets_by_interval_small_n(
    *,
    contour_table: pd.DataFrame,
    support_geometries: np.ndarray,
    interval_edges: np.ndarray,
    target_frames: Iterator[pd.DataFrame],
) -> np.ndarray:
    contour_geometries = np.asarray(contour_table["geometry"], dtype=object)
    contour_boundaries = np.asarray([geometry.boundary for geometry in contour_geometries], dtype=object)
    support_bounds = np.asarray([geometry.bounds for geometry in support_geometries], dtype=float)
    counts = np.zeros((len(contour_table), len(interval_edges) - 1), dtype=np.int64)

    for target_frame in target_frames:
        if target_frame.empty:
            continue

        x_values = target_frame["x"].to_numpy(dtype=float)
        y_values = target_frame["y"].to_numpy(dtype=float)
        for contour_idx, (support_geometry, support_bbox) in enumerate(
            zip(support_geometries, support_bounds, strict=True)
        ):
            bbox_mask = _points_within_bounds(x_values, y_values, support_bbox)
            if not np.any(bbox_mask):
                continue

            candidate_idx = np.flatnonzero(bbox_mask)
            candidate_points = points(x_values[candidate_idx], y_values[candidate_idx])
            support_mask = np.asarray(intersects(support_geometry, candidate_points), dtype=bool)
            if not np.any(support_mask):
                continue

            contour_points = candidate_points[support_mask]
            signed = _signed_distance_scalar(
                point_geometries=contour_points,
                contour_geometry=contour_geometries[contour_idx],
                contour_boundary=contour_boundaries[contour_idx],
            )
            ring_idx = _assign_intervals(signed_distances=signed, interval_edges=interval_edges)
            valid = ring_idx >= 0
            if np.any(valid):
                np.add.at(counts[contour_idx], ring_idx[valid], 1)

    return counts


def _count_smoothed_targets(
    *,
    sdata: XeniumSData,
    contour_table: pd.DataFrame,
    support_geometries: np.ndarray,
    signed_distance_grid: np.ndarray,
    target: str,
    target_query: str | None,
    feature_key: str,
    feature_values: str | Sequence[str] | None,
    inward: float,
    outward: float,
    bandwidth: float,
) -> np.ndarray:
    if len(contour_table) <= _SMALL_CONTOUR_FASTPATH_THRESHOLD:
        return _count_smoothed_targets_small_n(
            contour_table=contour_table,
            support_geometries=support_geometries,
            signed_distance_grid=signed_distance_grid,
            target_frames=_iter_target_frames(
                sdata=sdata,
                target=target,
                target_query=target_query,
                feature_key=feature_key,
                feature_values=feature_values,
            ),
            inward=inward,
            outward=outward,
            bandwidth=bandwidth,
        )

    contour_geometries = np.asarray(contour_table["geometry"], dtype=object)
    contour_boundaries = np.asarray([geometry.boundary for geometry in contour_geometries], dtype=object)
    tree = STRtree(support_geometries)
    counts = np.zeros((len(contour_table), len(signed_distance_grid)), dtype=float)
    lower = -float(inward)
    upper = float(outward)

    for target_frame in _iter_target_frames(
        sdata=sdata,
        target=target,
        target_query=target_query,
        feature_key=feature_key,
        feature_values=feature_values,
    ):
        if target_frame.empty:
            continue

        target_geometries = np.asarray(
            points(
                target_frame["x"].to_numpy(dtype=float),
                target_frame["y"].to_numpy(dtype=float),
            ),
            dtype=object,
        )
        pair_idx = tree.query(target_geometries, predicate="intersects")
        if pair_idx.size == 0:
            continue

        point_idx = pair_idx[0]
        contour_idx = pair_idx[1]
        signed = _signed_distance(
            point_geometries=target_geometries.take(point_idx),
            contour_geometries=contour_geometries.take(contour_idx),
            contour_boundaries=contour_boundaries.take(contour_idx),
        )
        valid = (signed >= lower) & (signed <= upper)
        if not np.any(valid):
            continue

        valid_contour_idx = contour_idx[valid]
        valid_signed = signed[valid]
        for current_contour_idx in np.unique(valid_contour_idx):
            signed_for_contour = valid_signed[valid_contour_idx == current_contour_idx]
            _accumulate_gaussian_with_reflection(
                counts[current_contour_idx],
                signed_distance_grid=signed_distance_grid,
                sample_locations=signed_for_contour,
                bandwidth=bandwidth,
                lower=lower,
                upper=upper,
            )

    return counts


def _count_smoothed_targets_small_n(
    *,
    contour_table: pd.DataFrame,
    support_geometries: np.ndarray,
    signed_distance_grid: np.ndarray,
    target_frames: Iterator[pd.DataFrame],
    inward: float,
    outward: float,
    bandwidth: float,
) -> np.ndarray:
    contour_geometries = np.asarray(contour_table["geometry"], dtype=object)
    contour_boundaries = np.asarray([geometry.boundary for geometry in contour_geometries], dtype=object)
    support_bounds = np.asarray([geometry.bounds for geometry in support_geometries], dtype=float)
    counts = np.zeros((len(contour_table), len(signed_distance_grid)), dtype=float)
    lower = -float(inward)
    upper = float(outward)

    for target_frame in target_frames:
        if target_frame.empty:
            continue

        x_values = target_frame["x"].to_numpy(dtype=float)
        y_values = target_frame["y"].to_numpy(dtype=float)
        for contour_idx, (support_geometry, support_bbox) in enumerate(
            zip(support_geometries, support_bounds, strict=True)
        ):
            bbox_mask = _points_within_bounds(x_values, y_values, support_bbox)
            if not np.any(bbox_mask):
                continue

            candidate_idx = np.flatnonzero(bbox_mask)
            candidate_points = points(x_values[candidate_idx], y_values[candidate_idx])
            support_mask = np.asarray(intersects(support_geometry, candidate_points), dtype=bool)
            if not np.any(support_mask):
                continue

            signed = _signed_distance_scalar(
                point_geometries=candidate_points[support_mask],
                contour_geometry=contour_geometries[contour_idx],
                contour_boundary=contour_boundaries[contour_idx],
            )
            valid = (signed >= lower) & (signed <= upper)
            if not np.any(valid):
                continue

            _accumulate_gaussian_with_reflection(
                counts[contour_idx],
                signed_distance_grid=signed_distance_grid,
                sample_locations=signed[valid],
                bandwidth=bandwidth,
                lower=lower,
                upper=upper,
            )

    return counts


def _iter_target_frames(
    *,
    sdata: XeniumSData,
    target: str,
    target_query: str | None,
    feature_key: str,
    feature_values: str | Sequence[str] | None,
) -> Iterator[pd.DataFrame]:
    normalized_target = _normalize_target(target)
    if normalized_target == "transcripts":
        normalized_feature_values = _normalize_feature_values(feature_values)
        if "transcripts" in sdata.point_sources:
            iterator = sdata.point_sources["transcripts"].iter_chunks()
        elif "transcripts" in sdata.points:
            iterator = iter([sdata.points["transcripts"]])
        else:
            raise KeyError(
                "Transcript points are not available in `sdata.points` or `sdata.point_sources`."
            )

        for frame in iterator:
            working = frame.copy()
            if target_query is not None:
                working = working.query(target_query, engine="python")
            if normalized_feature_values is not None:
                if feature_key not in working.columns:
                    raise KeyError(
                        f"Feature key `{feature_key}` was not found in the transcript points table."
                    )
                working = working.loc[working[feature_key].isin(normalized_feature_values)]
            if working.empty:
                continue
            missing = {"x", "y"}.difference(working.columns)
            if missing:
                raise ValueError(
                    f"Transcript points are missing required coordinate columns: {sorted(missing)}"
                )
            yield working.loc[:, ["x", "y"]].copy()
        return

    obs = sdata.table.obs.copy()
    if target_query is not None:
        obs = obs.query(target_query, engine="python")
    cell_frame = _cell_centroid_frame(sdata=sdata, obs=obs)
    if not cell_frame.empty:
        yield cell_frame


def _cell_centroid_frame(*, sdata: XeniumSData, obs: pd.DataFrame) -> pd.DataFrame:
    for x_col, y_col in (
        ("x_centroid", "y_centroid"),
        ("cell_centroid_x", "cell_centroid_y"),
        ("x", "y"),
    ):
        if x_col in obs.columns and y_col in obs.columns:
            return pd.DataFrame(
                {
                    "x": obs[x_col].astype(float).to_numpy(),
                    "y": obs[y_col].astype(float).to_numpy(),
                }
            )

    if "spatial" in sdata.table.obsm:
        spatial = np.asarray(sdata.table.obsm["spatial"], dtype=float)
        positions = sdata.table.obs_names.get_indexer(obs.index)
        valid = positions >= 0
        aligned = spatial[positions[valid]]
        return pd.DataFrame({"x": aligned[:, 0], "y": aligned[:, 1]})

    raise ValueError(
        "Unable to infer cell centroid coordinates from `sdata.table.obs`. "
        "Expected `x_centroid/y_centroid`, `cell_centroid_x/cell_centroid_y`, or `x/y`."
    )


def _normalize_target(target: str) -> str:
    normalized = str(target).strip().lower()
    if normalized not in {"transcripts", "cells"}:
        raise ValueError("`target` must be either 'transcripts' or 'cells'.")
    return normalized


def _normalize_feature_values(feature_values: str | Sequence[str] | None) -> set[str] | None:
    if feature_values is None:
        return None
    if isinstance(feature_values, str):
        values = [feature_values]
    else:
        values = [str(value) for value in feature_values]
    values = [value for value in values if value]
    if not values:
        raise ValueError("`feature_values` must contain at least one non-empty value.")
    return set(values)


def _feature_value_payload(feature_values: str | Sequence[str] | None) -> Any:
    if feature_values is None:
        return None
    if isinstance(feature_values, str):
        return feature_values
    values = [str(value) for value in feature_values]
    if len(values) == 1:
        return values[0]
    return tuple(values)


def _build_ring_intervals(*, ring_width: float, inward: float, outward: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    current = -float(inward)
    while current < 0:
        next_edge = min(current + ring_width, 0.0)
        intervals.append((float(current), float(next_edge)))
        current = next_edge

    current = 0.0
    while current < outward:
        next_edge = min(current + ring_width, float(outward))
        intervals.append((float(current), float(next_edge)))
        current = next_edge

    return intervals


def _build_signed_distance_grid(*, inward: float, outward: float, grid_step: float) -> np.ndarray:
    grid = np.arange(-float(inward), float(outward) + grid_step, grid_step, dtype=float)
    if grid.size == 0:
        return np.asarray([-float(inward), float(outward)], dtype=float)
    if np.isclose(grid[-1], outward):
        grid[-1] = float(outward)
    elif grid[-1] < outward:
        grid = np.append(grid, float(outward))
    else:
        grid = np.append(grid[grid < outward], float(outward))
    return np.unique(grid)


def _build_support_geometry(
    geometry: Any,
    *,
    inward: float,
    outward: float,
    padding: float,
) -> Any:
    outer_distance = float(outward) + float(padding)
    outer = geometry.buffer(outer_distance) if outer_distance > 0 else geometry
    inner_distance = float(inward) + float(padding)
    if inner_distance <= 0:
        return outer

    inner = geometry.buffer(-inner_distance)
    if inner.is_empty:
        return outer
    if outward <= 0 and padding <= 0:
        return geometry.difference(inner)
    return outer.difference(inner)


def _compute_geometry_measure(
    *,
    contour_geometries: np.ndarray,
    signed_distance_grid: np.ndarray,
    inward: float,
    outward: float,
    grid_step: float,
) -> np.ndarray:
    geometry_measure = np.zeros((len(contour_geometries), len(signed_distance_grid)), dtype=float)
    lower = -float(inward)
    upper = float(outward)
    half_step = grid_step / 2.0

    for contour_idx, geometry in enumerate(contour_geometries):
        for grid_idx, signed_distance_value in enumerate(signed_distance_grid):
            interval_start = max(lower, float(signed_distance_value) - half_step)
            interval_end = min(upper, float(signed_distance_value) + half_step)
            interval_width = interval_end - interval_start
            if interval_width <= 0:
                continue
            local_area = _shell_area(geometry, interval_start, interval_end)
            geometry_measure[contour_idx, grid_idx] = (
                local_area / interval_width if local_area > 0 else 0.0
            )
    return geometry_measure


def _shell_area(geometry: Any, distance_start: float, distance_end: float) -> float:
    if distance_end <= distance_start:
        return 0.0

    if distance_end <= 0:
        outer = geometry.buffer(-abs(distance_end))
        inner = geometry.buffer(-abs(distance_start))
        return max(float(outer.area) - float(inner.area), 0.0)

    if distance_start >= 0:
        return max(
            float(geometry.buffer(distance_end).area) - float(geometry.buffer(distance_start).area),
            0.0,
        )

    return _shell_area(geometry, distance_start, 0.0) + _shell_area(geometry, 0.0, distance_end)


def _signed_distance(
    *,
    point_geometries: np.ndarray,
    contour_geometries: np.ndarray,
    contour_boundaries: np.ndarray,
) -> np.ndarray:
    dist = np.asarray(distance(point_geometries, contour_boundaries), dtype=float)
    inside = np.asarray(contains(contour_geometries, point_geometries), dtype=bool)
    signed = np.where(dist == 0, 0.0, np.where(inside, -dist, dist))
    return np.asarray(signed, dtype=float)


def _signed_distance_scalar(
    *,
    point_geometries: np.ndarray,
    contour_geometry: Any,
    contour_boundary: Any,
) -> np.ndarray:
    dist = np.asarray(distance(point_geometries, contour_boundary), dtype=float)
    inside = np.asarray(contains(contour_geometry, point_geometries), dtype=bool)
    signed = np.where(dist == 0, 0.0, np.where(inside, -dist, dist))
    return np.asarray(signed, dtype=float)


def _points_within_bounds(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    return (
        (x_values >= minx)
        & (x_values <= maxx)
        & (y_values >= miny)
        & (y_values <= maxy)
    )


def _assign_intervals(*, signed_distances: np.ndarray, interval_edges: np.ndarray) -> np.ndarray:
    ring_idx = np.searchsorted(interval_edges, signed_distances, side="right") - 1
    on_last_edge = np.isclose(signed_distances, interval_edges[-1])
    ring_idx[on_last_edge] = len(interval_edges) - 2
    valid = (ring_idx >= 0) & (ring_idx < len(interval_edges) - 1)
    return np.where(valid, ring_idx, -1)


def _accumulate_gaussian_with_reflection(
    accumulator: np.ndarray,
    *,
    signed_distance_grid: np.ndarray,
    sample_locations: np.ndarray,
    bandwidth: float,
    lower: float,
    upper: float,
    chunk_size: int = 2048,
) -> None:
    if sample_locations.size == 0:
        return

    reflected_left = 2.0 * lower - sample_locations
    reflected_right = 2.0 * upper - sample_locations
    cutoff = _KERNEL_TRUNCATION * bandwidth

    for start in range(0, sample_locations.size, chunk_size):
        slc = slice(start, start + chunk_size)
        accumulator += _gaussian_kernel_sum(
            signed_distance_grid=signed_distance_grid,
            sample_locations=sample_locations[slc],
            bandwidth=bandwidth,
            cutoff=cutoff,
        )
        accumulator += _gaussian_kernel_sum(
            signed_distance_grid=signed_distance_grid,
            sample_locations=reflected_left[slc],
            bandwidth=bandwidth,
            cutoff=cutoff,
        )
        accumulator += _gaussian_kernel_sum(
            signed_distance_grid=signed_distance_grid,
            sample_locations=reflected_right[slc],
            bandwidth=bandwidth,
            cutoff=cutoff,
        )


def _gaussian_kernel_sum(
    *,
    signed_distance_grid: np.ndarray,
    sample_locations: np.ndarray,
    bandwidth: float,
    cutoff: float,
) -> np.ndarray:
    if sample_locations.size == 0:
        return np.zeros(len(signed_distance_grid), dtype=float)

    diffs = signed_distance_grid[:, None] - sample_locations[None, :]
    mask = np.abs(diffs) <= cutoff
    scaled = diffs / bandwidth
    weights = np.exp(-0.5 * scaled**2) / (np.sqrt(2.0 * np.pi) * bandwidth)
    weights[~mask] = 0.0
    return np.asarray(weights.sum(axis=1), dtype=float)


def _assemble_ring_result(
    *,
    contour_key: str,
    contour_table: pd.DataFrame,
    target: str,
    feature_key: str | None,
    feature_values: Any,
    intervals: list[tuple[float, float]],
    counts: np.ndarray,
    areas: np.ndarray,
) -> pd.DataFrame:
    metadata_columns = _resolve_metadata_columns(contour_table)
    records: list[dict[str, Any]] = []

    for contour_idx, contour_row in contour_table.iterrows():
        for ring_idx, (ring_start, ring_end) in enumerate(intervals):
            area = float(areas[contour_idx, ring_idx])
            count = int(counts[contour_idx, ring_idx])
            record = {
                "contour_key": contour_key,
                "contour_id": contour_row["contour_id"],
                "target": target,
                "feature_key": feature_key,
                "feature_values": feature_values,
                "ring_start": float(ring_start),
                "ring_end": float(ring_end),
                "ring_mid": 0.5 * (float(ring_start) + float(ring_end)),
                "count": count,
                "area": area,
                "density": np.nan if area <= 0 else count / area,
            }
            for column in metadata_columns:
                record[column] = contour_row[column]
            records.append(record)
    return pd.DataFrame.from_records(records)


def _assemble_smooth_result(
    *,
    contour_key: str,
    contour_table: pd.DataFrame,
    target: str,
    feature_key: str | None,
    feature_values: Any,
    signed_distance_grid: np.ndarray,
    bandwidth: float,
    grid_step: float,
    count_density: np.ndarray,
    geometry_measure: np.ndarray,
) -> pd.DataFrame:
    metadata_columns = _resolve_metadata_columns(contour_table)
    records: list[dict[str, Any]] = []

    for contour_idx, contour_row in contour_table.iterrows():
        for grid_idx, signed_distance_value in enumerate(signed_distance_grid):
            local_count_density = float(count_density[contour_idx, grid_idx])
            local_geometry_measure = float(geometry_measure[contour_idx, grid_idx])
            record = {
                "contour_key": contour_key,
                "contour_id": contour_row["contour_id"],
                "target": target,
                "feature_key": feature_key,
                "feature_values": feature_values,
                "signed_distance": float(signed_distance_value),
                "bandwidth": float(bandwidth),
                "grid_step": float(grid_step),
                "kernel": _GAUSSIAN_KERNEL,
                "count_density": local_count_density,
                "geometry_measure": local_geometry_measure,
                "density": np.nan
                if local_geometry_measure <= 0
                else local_count_density / local_geometry_measure,
            }
            for column in metadata_columns:
                record[column] = contour_row[column]
            records.append(record)
    return pd.DataFrame.from_records(records)


def _resolve_metadata_columns(contour_table: pd.DataFrame) -> list[str]:
    return [column for column in _DEFAULT_CONTOUR_METADATA_COLUMNS if column in contour_table.columns]
