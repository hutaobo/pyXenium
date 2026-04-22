from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, shape

from pyXenium.io.sdata_model import XeniumSData

__all__ = ["add_contours_from_geojson"]

_EXPANDED_METADATA_COLUMNS = (
    "assigned_structure",
    "classification_name",
    "annotation_source",
    "structure_id",
    "segmentation_source",
    "name",
    "object_type",
)


def add_contours_from_geojson(
    sdata: XeniumSData,
    geojson_path: str | Path,
    *,
    key: str,
    id_key: str = "polygon_id",
    coordinate_space: str = "xenium_pixel",
    pixel_size_um: float | None = None,
    copy: bool = False,
) -> XeniumSData | None:
    """
    Import polygon contours from a GeoJSON file into ``XeniumSData.shapes``.

    The imported contour geometry is normalized into pyXenium's dataframe-based shape
    representation, with coordinates expressed in microns.
    """

    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance.")

    shape_key = str(key)
    if shape_key in sdata.shapes:
        raise KeyError(f"`sdata.shapes[{shape_key!r}]` already exists.")

    resolved_path = Path(geojson_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Contour GeoJSON not found: {resolved_path}")

    resolved_coordinate_space = str(coordinate_space).strip().lower()
    if resolved_coordinate_space not in {"xenium_pixel", "xenium_um", "micron"}:
        raise ValueError(
            "`coordinate_space` must be one of {'xenium_pixel', 'xenium_um', 'micron'}."
        )

    resolved_pixel_size_um = _resolve_pixel_size_um(sdata=sdata, pixel_size_um=pixel_size_um)
    contour_frame, contour_metadata = _read_contour_geojson(
        path=resolved_path,
        id_key=id_key,
        coordinate_space=resolved_coordinate_space,
        pixel_size_um=resolved_pixel_size_um,
    )

    if copy:
        target = XeniumSData(
            table=sdata.table.copy(),
            points={name: frame.copy() for name, frame in sdata.points.items()},
            shapes={name: frame.copy() for name, frame in sdata.shapes.items()},
            images=dict(sdata.images),
            metadata=deepcopy(sdata.metadata),
            point_sources=dict(sdata.point_sources),
        )
    else:
        target = sdata
        target.metadata = deepcopy(target.metadata)

    target.shapes[shape_key] = contour_frame
    contour_registry = dict(target.metadata.get("contours", {}))
    contour_registry[shape_key] = contour_metadata
    target.metadata["contours"] = contour_registry
    target._validate()

    if copy:
        return target
    return None


def _resolve_pixel_size_um(sdata: XeniumSData, pixel_size_um: float | None) -> float:
    if pixel_size_um is not None:
        return float(pixel_size_um)

    he_image = sdata.images.get("he")
    if he_image is not None and he_image.pixel_size_um is not None:
        return float(he_image.pixel_size_um)

    image_artifacts = sdata.metadata.get("image_artifacts", {})
    if isinstance(image_artifacts, dict):
        he_artifact = image_artifacts.get("he", {})
        if isinstance(he_artifact, dict) and he_artifact.get("pixel_size_um") is not None:
            return float(he_artifact["pixel_size_um"])

    raise ValueError(
        "Unable to resolve `pixel_size_um` for contour import. "
        "Pass it explicitly or load an H&E image with pixel-size metadata first."
    )


def _read_contour_geojson(
    *,
    path: Path,
    id_key: str,
    coordinate_space: str,
    pixel_size_um: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("Contour GeoJSON must be a FeatureCollection with a `features` list.")
    if not features:
        raise ValueError("Contour GeoJSON does not contain any features.")

    records: list[dict[str, Any]] = []
    properties_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    metadata_columns = list(_EXPANDED_METADATA_COLUMNS)

    for feature_index, feature in enumerate(features):
        contour_id, geometry, metadata_values, raw_properties = _normalize_feature(
            feature=feature,
            feature_index=feature_index,
            id_key=id_key,
            coordinate_space=coordinate_space,
            pixel_size_um=pixel_size_um,
        )
        if contour_id in seen_ids:
            raise ValueError(f"Duplicate contour id {contour_id!r} found in {path.name}.")
        seen_ids.add(contour_id)
        properties_by_id[contour_id] = raw_properties
        records.extend(
            _geometry_to_records(
                geometry=geometry,
                contour_id=contour_id,
                metadata_values=metadata_values,
                metadata_columns=metadata_columns,
            )
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No polygon vertices could be parsed from {path}.")

    frame["contour_id"] = frame["contour_id"].astype(str)
    frame["part_id"] = frame["part_id"].astype("int64")
    frame["ring_id"] = frame["ring_id"].astype("int64")
    frame["vertex_id"] = frame["vertex_id"].astype("int64")
    frame["is_hole"] = frame["is_hole"].astype(bool)
    frame["x"] = frame["x"].astype(float)
    frame["y"] = frame["y"].astype(float)

    metadata = {
        "source_geojson_path": str(path),
        "coordinate_space": coordinate_space,
        "units": "micron",
        "pixel_size_um": float(pixel_size_um),
        "id_key": str(id_key),
        "n_contours": int(frame["contour_id"].nunique()),
        "properties_by_id": properties_by_id,
    }
    return frame, metadata


def _normalize_feature(
    *,
    feature: dict[str, Any],
    feature_index: int,
    id_key: str,
    coordinate_space: str,
    pixel_size_um: float,
) -> tuple[str, Polygon | MultiPolygon, dict[str, Any], dict[str, Any]]:
    if not isinstance(feature, dict):
        raise TypeError(f"GeoJSON feature at index {feature_index} must be a dictionary.")

    properties = feature.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    nested_metadata = properties.get("metadata", {})
    if not isinstance(nested_metadata, dict):
        nested_metadata = {}

    contour_id = properties.get(id_key, nested_metadata.get(id_key))
    if contour_id is None:
        raise KeyError(
            f"Feature index {feature_index} is missing the contour id field `{id_key}`."
        )

    geometry = shape(feature.get("geometry"))
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(
            f"Feature {contour_id!r} must contain Polygon or MultiPolygon geometry, "
            f"got {geometry.geom_type!r}."
        )

    if coordinate_space == "xenium_pixel":
        geometry = affinity.scale(
            geometry,
            xfact=pixel_size_um,
            yfact=pixel_size_um,
            origin=(0.0, 0.0),
        )

    metadata_values = {}
    for column in _EXPANDED_METADATA_COLUMNS:
        value = properties.get(column, nested_metadata.get(column))
        metadata_values[column] = _json_ready(value)

    raw_properties = _json_ready(properties)
    if not isinstance(raw_properties, dict):
        raw_properties = {"properties": raw_properties}

    return str(contour_id), geometry, metadata_values, raw_properties


def _geometry_to_records(
    *,
    geometry: Polygon | MultiPolygon,
    contour_id: str,
    metadata_values: dict[str, Any],
    metadata_columns: list[str],
) -> list[dict[str, Any]]:
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    records: list[dict[str, Any]] = []

    for part_id, polygon in enumerate(polygons):
        if polygon.is_empty:
            continue

        exterior_coords = list(polygon.exterior.coords)
        if exterior_coords and exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]

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
            hole_coords = list(interior.coords)
            if hole_coords and hole_coords[0] == hole_coords[-1]:
                hole_coords = hole_coords[:-1]

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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)
