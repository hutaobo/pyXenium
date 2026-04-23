from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import pandas as pd
from shapely import segmentize, union_all, voronoi_polygons
from shapely.geometry import GeometryCollection, MultiPoint
from shapely.geometry.base import BaseGeometry

from ._geometry import contour_frame_to_geometry_table, geometry_table_to_contour_frame
from pyXenium.io.sdata_model import XeniumSData

__all__ = ["expand_contours"]

_VALID_EXPANSION_MODES = {"overlap", "voronoi"}
_VORONOI_MIN_SAMPLE_STEP = 1.0
_VORONOI_MAX_SAMPLE_STEP = 5.0
_SEED_POINT_ROUND_DECIMALS = 12


def expand_contours(
    sdata: XeniumSData,
    *,
    contour_key: str,
    distance: float,
    mode: str = "overlap",
    output_key: str | None = None,
    copy: bool = False,
    voronoi_sample_step: float | None = None,
) -> XeniumSData | None:
    """
    Expand an existing contour layer into a derived contour layer.

    Parameters
    ----------
    sdata
        Input XeniumSData container.
    contour_key
        Source contour key inside ``sdata.shapes``.
    distance
        Positive outward expansion distance in the contour coordinate units.
    mode
        Either ``"overlap"`` for ordinary buffering or ``"voronoi"`` for
        mutually-exclusive Voronoi partitioning over the buffered support.
    output_key
        Target contour key. Defaults to ``f"{contour_key}_expanded"``.
    copy
        When ``True``, return a copied ``XeniumSData``. Otherwise mutate ``sdata`` in place.
    voronoi_sample_step
        Maximum segment length used to sample boundary seed points in Voronoi mode.
    """

    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance.")

    source_key = str(contour_key)
    if source_key not in sdata.shapes:
        raise KeyError(f"Contour key `{source_key}` not found in `sdata.shapes`.")

    expansion_distance = float(distance)
    if expansion_distance <= 0:
        raise ValueError("`distance` must be greater than 0.")

    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in _VALID_EXPANSION_MODES:
        raise ValueError("`mode` must be one of {'overlap', 'voronoi'}.")

    resolved_output_key = (
        f"{source_key}_expanded" if output_key is None else str(output_key).strip()
    )
    if not resolved_output_key:
        raise ValueError("`output_key` must be a non-empty string when provided.")
    if resolved_output_key in sdata.shapes:
        raise KeyError(f"`sdata.shapes[{resolved_output_key!r}]` already exists.")

    contour_table = contour_frame_to_geometry_table(sdata.shapes[source_key], contour_key=source_key)
    resolved_voronoi_sample_step = (
        _resolve_voronoi_sample_step(expansion_distance, voronoi_sample_step)
        if resolved_mode == "voronoi"
        else None
    )
    expanded_table = _expand_contour_geometry_table(
        contour_table=contour_table,
        distance=expansion_distance,
        mode=resolved_mode,
        voronoi_sample_step=resolved_voronoi_sample_step,
    )
    expanded_frame = geometry_table_to_contour_frame(expanded_table)

    target = _copy_sdata(sdata) if copy else sdata
    if not copy:
        target.metadata = deepcopy(target.metadata)

    target.shapes[resolved_output_key] = expanded_frame
    target.metadata["contours"] = _updated_contour_registry(
        sdata=target,
        source_key=source_key,
        output_key=resolved_output_key,
        distance=expansion_distance,
        mode=resolved_mode,
        voronoi_sample_step=resolved_voronoi_sample_step,
        n_contours=int(expanded_frame["contour_id"].nunique()),
    )
    target._validate()
    return target if copy else None


def _expand_contour_geometry_table(
    *,
    contour_table: pd.DataFrame,
    distance: float,
    mode: str,
    voronoi_sample_step: float | None,
) -> pd.DataFrame:
    expanded_records: list[dict[str, Any]] = []
    metadata_columns = [
        column for column in contour_table.columns if column not in {"contour_id", "geometry"}
    ]
    for _, row in contour_table.iterrows():
        source_geometry = _clean_geometry(row["geometry"])
        expanded_geometry = _buffer_geometry(source_geometry, distance)
        if expanded_geometry.is_empty:
            continue
        record = {"contour_id": row["contour_id"], "geometry": expanded_geometry}
        for column in metadata_columns:
            record[column] = row[column]
        expanded_records.append(record)

    expanded_table = pd.DataFrame.from_records(expanded_records)
    if expanded_table.empty:
        raise ValueError("Contour expansion did not produce any non-empty polygon geometries.")

    expanded_table = expanded_table.reset_index(drop=True)
    if mode == "overlap" or len(expanded_table) == 1:
        return expanded_table

    return _apply_voronoi_partition(
        original_table=contour_table.loc[contour_table["contour_id"].isin(expanded_table["contour_id"])].reset_index(
            drop=True
        ),
        expanded_table=expanded_table,
        sample_step=voronoi_sample_step,
    )


def _buffer_geometry(geometry: BaseGeometry, distance: float) -> BaseGeometry:
    expanded = _clean_geometry(geometry).buffer(distance)
    if expanded.is_empty:
        return GeometryCollection()
    return _clean_geometry(expanded)


def _apply_voronoi_partition(
    *,
    original_table: pd.DataFrame,
    expanded_table: pd.DataFrame,
    sample_step: float | None,
) -> pd.DataFrame:
    expanded_geometries = [_clean_geometry(geometry) for geometry in expanded_table["geometry"]]
    support_union = _clean_geometry(union_all(expanded_geometries))
    if support_union.is_empty:
        raise ValueError("Voronoi contour expansion requires a non-empty buffered support region.")

    sampled_points: list[tuple[float, float]] = []
    sampled_owners: list[int] = []
    occupied_seed_keys: set[tuple[float, float]] = set()

    for contour_index, row in original_table.iterrows():
        representative = row["geometry"].representative_point()
        representative_xy = (float(representative.x), float(representative.y))
        boundary_points = _sample_boundary_points(
            row["geometry"],
            sample_step=sample_step,
            representative_xy=representative_xy,
            occupied_seed_keys=occupied_seed_keys,
        )
        sampled_points.extend(boundary_points)
        sampled_owners.extend([contour_index] * len(boundary_points))

    if len(sampled_points) < 2:
        return expanded_table

    voronoi_cells = voronoi_polygons(
        MultiPoint(sampled_points),
        extend_to=support_union.envelope,
        ordered=True,
    )
    cells = list(voronoi_cells.geoms)
    if len(cells) != len(sampled_points):
        raise ValueError(
            "Voronoi contour expansion could not preserve the sampled boundary-point ordering."
        )

    cells_by_contour: dict[int, list[BaseGeometry]] = {index: [] for index in range(len(expanded_table))}
    for cell, contour_index in zip(cells, sampled_owners, strict=True):
        cells_by_contour[contour_index].append(cell)

    partitioned_records: list[dict[str, Any]] = []
    metadata_columns = [
        column for column in expanded_table.columns if column not in {"contour_id", "geometry"}
    ]
    for contour_index, row in expanded_table.iterrows():
        contour_cells = []
        for cell in cells_by_contour.get(contour_index, []):
            cleaned_cell = _clean_geometry(cell)
            if not cleaned_cell.is_empty:
                contour_cells.append(cleaned_cell)
        if not contour_cells:
            raise ValueError(
                f"Voronoi contour expansion did not produce any cells for contour {row['contour_id']!r}."
            )

        voronoi_region = _clean_geometry(union_all(contour_cells))
        final_geometry = _clean_geometry(
            _clean_geometry(row["geometry"]).intersection(voronoi_region).intersection(support_union)
        )
        if final_geometry.is_empty:
            raise ValueError(
                f"Voronoi contour expansion produced an empty geometry for contour {row['contour_id']!r}."
            )

        record = {"contour_id": row["contour_id"], "geometry": final_geometry}
        for column in metadata_columns:
            record[column] = row[column]
        partitioned_records.append(record)

    return pd.DataFrame.from_records(partitioned_records).reset_index(drop=True)


def _sample_boundary_points(
    geometry: BaseGeometry,
    *,
    sample_step: float | None,
    representative_xy: tuple[float, float],
    occupied_seed_keys: set[tuple[float, float]],
) -> list[tuple[float, float]]:
    segmented_boundary = segmentize(geometry.boundary, max_segment_length=float(sample_step))
    sampled_points: list[tuple[float, float]] = []
    local_seen: set[tuple[float, float]] = set()

    for x, y in _boundary_coordinates(segmented_boundary):
        key = _rounded_seed_key(x, y)
        if key in local_seen:
            continue
        local_seen.add(key)
        sampled_points.append(
            _disambiguate_seed_point(
                x=x,
                y=y,
                representative_xy=representative_xy,
                sample_step=float(sample_step),
                occupied_seed_keys=occupied_seed_keys,
            )
        )

    if sampled_points:
        return sampled_points

    fallback_x, fallback_y = representative_xy
    return [
        _disambiguate_seed_point(
            x=fallback_x,
            y=fallback_y,
            representative_xy=representative_xy,
            sample_step=float(sample_step),
            occupied_seed_keys=occupied_seed_keys,
        )
    ]


def _boundary_coordinates(boundary: BaseGeometry) -> list[tuple[float, float]]:
    try:
        coords_obj = boundary.coords
    except (AttributeError, NotImplementedError):
        coords_obj = None

    if coords_obj is not None:
        coords = [(float(x), float(y)) for x, y in coords_obj]
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        return coords
    if hasattr(boundary, "geoms"):
        coordinates: list[tuple[float, float]] = []
        for part in boundary.geoms:
            coordinates.extend(_boundary_coordinates(part))
        return coordinates
    return []


def _disambiguate_seed_point(
    *,
    x: float,
    y: float,
    representative_xy: tuple[float, float],
    sample_step: float,
    occupied_seed_keys: set[tuple[float, float]],
) -> tuple[float, float]:
    current_x = float(x)
    current_y = float(y)
    current_key = _rounded_seed_key(current_x, current_y)
    if current_key not in occupied_seed_keys:
        occupied_seed_keys.add(current_key)
        return current_x, current_y

    rep_x, rep_y = representative_xy
    direction_x = rep_x - current_x
    direction_y = rep_y - current_y
    direction_norm = math.hypot(direction_x, direction_y)
    if direction_norm == 0:
        direction_x, direction_y = 1.0, 0.0
        direction_norm = 1.0

    epsilon = 1e-6 * max(float(sample_step), 1.0)
    attempt = 1
    while current_key in occupied_seed_keys:
        offset = epsilon * attempt
        current_x = float(x) + (direction_x / direction_norm) * offset
        current_y = float(y) + (direction_y / direction_norm) * offset
        current_key = _rounded_seed_key(current_x, current_y)
        attempt += 1

    occupied_seed_keys.add(current_key)
    return current_x, current_y


def _rounded_seed_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), _SEED_POINT_ROUND_DECIMALS), round(float(y), _SEED_POINT_ROUND_DECIMALS))


def _resolve_voronoi_sample_step(distance: float, voronoi_sample_step: float | None) -> float:
    if voronoi_sample_step is None:
        return min(max(float(distance) / 8.0, _VORONOI_MIN_SAMPLE_STEP), _VORONOI_MAX_SAMPLE_STEP)

    resolved = float(voronoi_sample_step)
    if resolved <= 0:
        raise ValueError("`voronoi_sample_step` must be greater than 0 when provided.")
    return resolved


def _clean_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return GeometryCollection()
    if geometry.is_valid:
        return geometry
    cleaned = geometry.buffer(0)
    if cleaned.is_empty:
        return GeometryCollection()
    return cleaned


def _copy_sdata(sdata: XeniumSData) -> XeniumSData:
    return XeniumSData(
        table=sdata.table.copy(),
        points={name: frame.copy() for name, frame in sdata.points.items()},
        shapes={name: frame.copy() for name, frame in sdata.shapes.items()},
        images=dict(sdata.images),
        contour_images={
            contour_key: dict(images) for contour_key, images in sdata.contour_images.items()
        },
        metadata=deepcopy(sdata.metadata),
        point_sources=dict(sdata.point_sources),
    )


def _updated_contour_registry(
    *,
    sdata: XeniumSData,
    source_key: str,
    output_key: str,
    distance: float,
    mode: str,
    voronoi_sample_step: float | None,
    n_contours: int,
) -> dict[str, Any]:
    existing_registry = sdata.metadata.get("contours", {})
    contour_registry = dict(existing_registry) if isinstance(existing_registry, dict) else {}

    source_metadata = contour_registry.get(source_key, {})
    derived_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    units = derived_metadata.get("units", sdata.metadata.get("units"))
    units_is_um = str(units).strip().lower() in {"micron", "microns", "um", "µm"}

    derived_metadata["derived_from_key"] = source_key
    derived_metadata["generator"] = "expand_contours"
    derived_metadata["expansion_mode"] = mode
    derived_metadata["expansion_distance"] = float(distance)
    if units_is_um:
        derived_metadata["expansion_distance_um"] = float(distance)
    else:
        derived_metadata.pop("expansion_distance_um", None)

    if mode == "voronoi" and voronoi_sample_step is not None:
        derived_metadata["voronoi_sample_step"] = float(voronoi_sample_step)
        if units_is_um:
            derived_metadata["voronoi_sample_step_um"] = float(voronoi_sample_step)
        else:
            derived_metadata.pop("voronoi_sample_step_um", None)
    else:
        derived_metadata.pop("voronoi_sample_step", None)
        derived_metadata.pop("voronoi_sample_step_um", None)

    derived_metadata["n_contours"] = int(n_contours)
    contour_registry[output_key] = derived_metadata
    return contour_registry
