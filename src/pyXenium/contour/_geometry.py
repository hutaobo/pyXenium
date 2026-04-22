from __future__ import annotations

from typing import Any

import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

CONTOUR_STRUCTURE_COLUMNS = (
    "contour_id",
    "part_id",
    "ring_id",
    "is_hole",
    "vertex_id",
    "x",
    "y",
)


def normalize_contour_frame(
    frame: pd.DataFrame,
    *,
    contour_key: str | None = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"`frame` must be a pandas.DataFrame, got {type(frame)!r}.")

    working = frame.copy()
    required = {"contour_id", "vertex_id", "x", "y"}
    missing = required.difference(working.columns)
    if missing:
        message = (
            f"Contour dataframe is missing required contour columns: {sorted(missing)}"
            if contour_key is None
            else (
                f"`sdata.shapes[{contour_key!r}]` is missing required contour columns: "
                f"{sorted(missing)}"
            )
        )
        raise ValueError(message)

    if "part_id" not in working.columns:
        working["part_id"] = 0
    if "ring_id" not in working.columns:
        working["ring_id"] = 0
    if "is_hole" not in working.columns:
        working["is_hole"] = False

    working["contour_id"] = working["contour_id"].astype(str)
    working["part_id"] = working["part_id"].astype("int64")
    working["ring_id"] = working["ring_id"].astype("int64")
    working["is_hole"] = working["is_hole"].astype(bool)
    working["vertex_id"] = working["vertex_id"].astype("int64")
    working["x"] = working["x"].astype(float)
    working["y"] = working["y"].astype(float)
    return working


def contour_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in CONTOUR_STRUCTURE_COLUMNS]


def contour_frame_to_geometry_table(
    frame: pd.DataFrame,
    *,
    contour_key: str | None = None,
) -> pd.DataFrame:
    working = normalize_contour_frame(frame, contour_key=contour_key)
    metadata_columns = contour_metadata_columns(working)
    records: list[dict[str, Any]] = []

    for contour_id, contour_rows in working.groupby("contour_id", sort=False, dropna=False):
        geometry = _frame_to_geometry(contour_rows)
        geometry = _normalize_polygonal_geometry(geometry)
        if geometry.is_empty:
            continue

        record = {"contour_id": str(contour_id), "geometry": geometry}
        first_row = contour_rows.iloc[0]
        for column in metadata_columns:
            record[column] = first_row[column]
        records.append(record)

    contour_table = pd.DataFrame.from_records(records)
    if contour_table.empty:
        raise ValueError("No valid contour geometries could be reconstructed from the contour table.")
    return contour_table.reset_index(drop=True)


def geometry_table_to_contour_frame(geometry_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(geometry_table, pd.DataFrame):
        raise TypeError(
            f"`geometry_table` must be a pandas.DataFrame, got {type(geometry_table)!r}."
        )

    required = {"contour_id", "geometry"}
    missing = required.difference(geometry_table.columns)
    if missing:
        raise ValueError(
            f"`geometry_table` is missing required columns: {sorted(missing)}"
        )

    metadata_columns = [
        column for column in geometry_table.columns if column not in {"contour_id", "geometry"}
    ]
    records: list[dict[str, Any]] = []
    for _, row in geometry_table.iterrows():
        geometry = _normalize_polygonal_geometry(row["geometry"])
        if geometry.is_empty:
            continue

        metadata_values = {column: row[column] for column in metadata_columns}
        records.extend(
            _geometry_to_records(
                geometry=geometry,
                contour_id=str(row["contour_id"]),
                metadata_values=metadata_values,
                metadata_columns=metadata_columns,
            )
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No polygon vertices could be reconstructed from the geometry table.")

    frame = normalize_contour_frame(frame)
    ordered_columns = [*CONTOUR_STRUCTURE_COLUMNS, *metadata_columns]
    return frame.loc[:, ordered_columns]


def _frame_to_geometry(frame: pd.DataFrame) -> BaseGeometry:
    polygons: list[Polygon] = []
    ordered = frame.sort_values(["part_id", "ring_id", "vertex_id"], kind="stable")

    for _, part_rows in ordered.groupby("part_id", sort=False, dropna=False):
        exterior_rings: list[list[tuple[float, float]]] = []
        hole_rings: list[list[tuple[float, float]]] = []
        for _, ring_rows in part_rows.groupby("ring_id", sort=False, dropna=False):
            ring_coords = list(zip(ring_rows["x"].astype(float), ring_rows["y"].astype(float)))
            if ring_coords and ring_coords[0] == ring_coords[-1]:
                ring_coords = ring_coords[:-1]
            if len(ring_coords) < 3:
                continue
            if bool(ring_rows["is_hole"].iloc[0]):
                hole_rings.append(ring_coords)
            else:
                exterior_rings.append(ring_coords)

        exterior_polygons: list[Polygon] = []
        for exterior_ring in exterior_rings:
            exterior_geometry = _normalize_polygonal_geometry(Polygon(exterior_ring))
            if isinstance(exterior_geometry, Polygon):
                exterior_polygons.append(exterior_geometry)
            elif isinstance(exterior_geometry, MultiPolygon):
                exterior_polygons.extend(list(exterior_geometry.geoms))

        if not exterior_polygons:
            continue

        holes_by_exterior: dict[int, list[list[tuple[float, float]]]] = {
            index: [] for index in range(len(exterior_polygons))
        }
        for hole_ring in hole_rings:
            hole_polygon = Polygon(hole_ring)
            if hole_polygon.is_empty:
                continue
            representative = hole_polygon.representative_point()
            candidates = [
                (index, exterior.area)
                for index, exterior in enumerate(exterior_polygons)
                if exterior.covers(representative)
            ]
            if not candidates:
                continue
            assigned_index = min(candidates, key=lambda item: item[1])[0]
            holes_by_exterior[assigned_index].append(hole_ring)

        for index, exterior in enumerate(exterior_polygons):
            polygon = Polygon(exterior.exterior.coords, holes=holes_by_exterior.get(index, []))
            polygon = _normalize_polygonal_geometry(polygon)
            if isinstance(polygon, Polygon):
                polygons.append(polygon)
            elif isinstance(polygon, MultiPolygon):
                polygons.extend(list(polygon.geoms))

    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _geometry_to_records(
    *,
    geometry: BaseGeometry,
    contour_id: str,
    metadata_values: dict[str, Any],
    metadata_columns: list[str],
) -> list[dict[str, Any]]:
    polygons = _polygon_parts(geometry)
    records: list[dict[str, Any]] = []

    for part_id, polygon in enumerate(polygons):
        exterior_coords = _ring_coordinates(polygon.exterior.coords)
        for vertex_id, (x, y) in enumerate(exterior_coords):
            record = {
                "contour_id": contour_id,
                "part_id": part_id,
                "ring_id": 0,
                "is_hole": False,
                "vertex_id": vertex_id,
                "x": float(x),
                "y": float(y),
            }
            for column in metadata_columns:
                record[column] = metadata_values.get(column)
            records.append(record)

        for hole_offset, interior in enumerate(polygon.interiors, start=1):
            hole_coords = _ring_coordinates(interior.coords)
            for vertex_id, (x, y) in enumerate(hole_coords):
                record = {
                    "contour_id": contour_id,
                    "part_id": part_id,
                    "ring_id": hole_offset,
                    "is_hole": True,
                    "vertex_id": vertex_id,
                    "x": float(x),
                    "y": float(y),
                }
                for column in metadata_columns:
                    record[column] = metadata_values.get(column)
                records.append(record)

    return records


def _ring_coordinates(coords: Any) -> list[tuple[float, float]]:
    ring_coords = [(float(x), float(y)) for x, y in coords]
    if ring_coords and ring_coords[0] == ring_coords[-1]:
        ring_coords = ring_coords[:-1]
    return ring_coords


def _normalize_polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return GeometryCollection()

    working = geometry
    if not working.is_valid:
        working = working.buffer(0)
        if working.is_empty:
            return GeometryCollection()

    polygons = _polygon_parts(working)
    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for part in geometry.geoms:
            polygons.extend(_polygon_parts(part))
        return polygons
    return []
