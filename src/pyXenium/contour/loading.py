from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, shape

from ._geometry import contour_frame_to_geometry_table, geometry_table_to_contour_frame
from pyXenium.io.sdata_model import XeniumImage, XeniumSData

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
_CONTOUR_PATCH_MASK_MODE = "polygon"
_CONTOUR_PATCH_PADDING_PX = 0


def add_contours_from_geojson(
    sdata: XeniumSData,
    geojson_path: str | Path,
    *,
    key: str,
    id_key: str = "polygon_id",
    coordinate_space: str = "xenium_pixel",
    pixel_size_um: float | None = None,
    extract_he_patches: bool = False,
    he_image_key: str = "he",
    copy: bool = False,
) -> XeniumSData | None:
    """
    Import polygon contours from a GeoJSON file into ``XeniumSData.shapes``.

    The imported contour geometry is normalized into pyXenium's dataframe-based shape
    representation, with coordinates expressed in microns.

    When ``extract_he_patches=True``, pyXenium additionally crops one level-0 H&E patch
    per imported contour and stores it under ``XeniumSData.contour_images[key]`` using
    the contour's axis-aligned bounding box plus a polygon mask.
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

    resolved_he_image_key: str | None = None
    he_image: XeniumImage | None = None
    if extract_he_patches:
        resolved_he_image_key, he_image = _resolve_he_image_for_patch_extraction(
            sdata,
            he_image_key=he_image_key,
        )

    if pixel_size_um is not None:
        resolved_pixel_size_um = float(pixel_size_um)
    elif he_image is not None and he_image.pixel_size_um is not None:
        resolved_pixel_size_um = float(he_image.pixel_size_um)
    else:
        resolved_pixel_size_um = _resolve_pixel_size_um(sdata=sdata, pixel_size_um=None)
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
            contour_images={
                contour_key: dict(images) for contour_key, images in sdata.contour_images.items()
            },
            metadata=deepcopy(sdata.metadata),
            point_sources=dict(sdata.point_sources),
        )
    else:
        target = sdata
        target.metadata = deepcopy(target.metadata)

    target.shapes[shape_key] = contour_frame
    contour_registry = dict(target.metadata.get("contours", {}))
    if extract_he_patches:
        if resolved_he_image_key is None or he_image is None:
            raise RuntimeError("Internal error: H&E patch extraction was enabled without a resolved image.")
        contour_patches = _extract_contour_he_patches(
            contour_frame=contour_frame,
            contour_key=shape_key,
            he_image=he_image,
            source_image_key=resolved_he_image_key,
        )
        target.contour_images[shape_key] = contour_patches
        contour_metadata = dict(contour_metadata)
        contour_metadata["he_patches_enabled"] = True
        contour_metadata["he_image_key"] = resolved_he_image_key
        contour_metadata["n_he_patches"] = len(contour_patches)
        contour_metadata["mask_mode"] = _CONTOUR_PATCH_MASK_MODE
        contour_metadata["padding_px"] = _CONTOUR_PATCH_PADDING_PX
        contour_metadata["storage_group"] = f"contour_images/{shape_key}"
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
        record = {
            "contour_id": contour_id,
            "geometry": geometry,
        }
        for column in metadata_columns:
            record[column] = metadata_values.get(column)
        records.append(record)

    geometry_table = pd.DataFrame.from_records(records)
    if geometry_table.empty:
        raise ValueError(f"No polygon vertices could be parsed from {path}.")
    frame = geometry_table_to_contour_frame(geometry_table)

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


def _resolve_he_image_for_patch_extraction(
    sdata: XeniumSData,
    *,
    he_image_key: str,
) -> tuple[str, XeniumImage]:
    resolved_key = str(he_image_key).strip()
    if not resolved_key:
        raise ValueError("`he_image_key` must be a non-empty string.")
    if resolved_key not in sdata.images:
        raise ValueError(
            "H&E patch extraction requires a loaded source image. "
            f"`sdata.images[{resolved_key!r}]` was not found. "
            "Load the sample with `read_xenium(..., as_='sdata', include_images=True)` first."
        )

    he_image = sdata.images[resolved_key]
    if he_image.pixel_size_um is None:
        raise ValueError(
            f"H&E patch extraction requires `sdata.images[{resolved_key!r}].pixel_size_um`."
        )
    if he_image.image_to_xenium_affine is None:
        raise ValueError(
            "H&E patch extraction requires image alignment metadata. "
            f"`sdata.images[{resolved_key!r}].image_to_xenium_affine` is missing."
        )
    return resolved_key, he_image


def _extract_contour_he_patches(
    *,
    contour_frame: pd.DataFrame,
    contour_key: str,
    he_image: XeniumImage,
    source_image_key: str,
) -> dict[str, XeniumImage]:
    contour_table = contour_frame_to_geometry_table(contour_frame, contour_key=contour_key)
    contour_patches: dict[str, XeniumImage] = {}
    level0 = he_image.levels[0]
    reusable_store = None
    reusable_level = None
    if hasattr(level0, "open_zarr_source"):
        reusable_store, reusable_level = level0.open_zarr_source()
    try:
        for _, row in contour_table.iterrows():
            contour_id = str(row["contour_id"])
            contour_patches[contour_id] = _extract_single_contour_he_patch(
                contour_geometry_um=row["geometry"],
                contour_key=contour_key,
                contour_id=contour_id,
                he_image=he_image,
                source_image_key=source_image_key,
                level0_crop_source=reusable_level,
            )
    finally:
        if reusable_store is not None and hasattr(reusable_store, "close"):
            reusable_store.close()
    return contour_patches


def _extract_single_contour_he_patch(
    *,
    contour_geometry_um: Polygon | MultiPolygon,
    contour_key: str,
    contour_id: str,
    he_image: XeniumImage,
    source_image_key: str,
    level0_crop_source: Any | None = None,
) -> XeniumImage:
    level0 = he_image.levels[0]
    image_geometry = _geometry_um_to_image_xy(contour_geometry_um, he_image=he_image)
    bbox = _clipped_image_bbox(
        image_geometry=image_geometry,
        image_shape=tuple(int(value) for value in level0.shape),
        image_axes=he_image.axes,
        contour_key=contour_key,
        contour_id=contour_id,
    )
    crop_level = level0_crop_source if level0_crop_source is not None else level0
    patch_array = _crop_image_level(crop_level, axes=he_image.axes, bbox=bbox)
    mask = _polygon_mask_for_bbox(image_geometry=image_geometry, bbox=bbox)
    masked_patch = _apply_polygon_mask(patch_array, mask=mask, axes=he_image.axes)

    x0, y0, x1, y1 = bbox
    bbox_xenium_um = [float(value) for value in contour_geometry_um.bounds]
    patch_affine = _patch_local_affine(
        he_image.image_to_xenium_affine,
        x_offset=float(x0),
        y_offset=float(y0),
    )
    patch_metadata = {
        "source_image_key": str(source_image_key),
        "contour_key": str(contour_key),
        "contour_id": str(contour_id),
        "mask_mode": _CONTOUR_PATCH_MASK_MODE,
        "padding_px": _CONTOUR_PATCH_PADDING_PX,
        "bbox_image_xy": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_xenium_um": bbox_xenium_um,
        "transform_direction": "patch_local_image_pixel_xy_to_xenium_pixel_xy",
        "transform_input_space": "patch_local_image_pixel_xy",
        "transform_output_space": "xenium_pixel_xy",
        "transform_output_unit": "pixel",
        "xenium_physical_unit": "micron",
    }
    return XeniumImage(
        levels=[masked_patch],
        axes=he_image.axes,
        dtype=np.asarray(masked_patch).dtype.name,
        source_path=he_image.source_path,
        transform_kind="affine",
        image_to_xenium_affine=patch_affine,
        alignment_csv_path=he_image.alignment_csv_path,
        pixel_size_um=he_image.pixel_size_um,
        metadata=patch_metadata,
    )


def _geometry_um_to_image_xy(
    geometry: Polygon | MultiPolygon,
    *,
    he_image: XeniumImage,
) -> Polygon | MultiPolygon:
    if isinstance(geometry, Polygon):
        return _polygon_um_to_image_xy(geometry, he_image=he_image)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon(
            [_polygon_um_to_image_xy(polygon, he_image=he_image) for polygon in geometry.geoms]
        )
    raise TypeError(
        "Contour patch extraction expects Polygon or MultiPolygon geometry, "
        f"got {type(geometry)!r}."
    )


def _polygon_um_to_image_xy(polygon: Polygon, *, he_image: XeniumImage) -> Polygon:
    exterior = _ring_um_to_image_xy(polygon.exterior.coords, he_image=he_image)
    holes = [
        _ring_um_to_image_xy(interior.coords, he_image=he_image)
        for interior in polygon.interiors
    ]
    return Polygon(exterior, holes=holes)


def _ring_um_to_image_xy(coords: Any, *, he_image: XeniumImage) -> list[tuple[float, float]]:
    xy_um = np.asarray([(float(x), float(y)) for x, y in coords], dtype=float)
    if xy_um.size == 0:
        return []
    image_xy = he_image.xenium_um_to_image_xy(xy_um)
    return [(float(x), float(y)) for x, y in image_xy]


def _clipped_image_bbox(
    *,
    image_geometry: Polygon | MultiPolygon,
    image_shape: tuple[int, ...],
    image_axes: str,
    contour_key: str,
    contour_id: str,
) -> tuple[int, int, int, int]:
    x_index = image_axes.index("x")
    y_index = image_axes.index("y")
    image_width = int(image_shape[x_index])
    image_height = int(image_shape[y_index])
    min_x, min_y, max_x, max_y = image_geometry.bounds
    x0 = max(int(np.floor(min_x)), 0)
    y0 = max(int(np.floor(min_y)), 0)
    x1 = min(int(np.ceil(max_x)), image_width)
    y1 = min(int(np.ceil(max_y)), image_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Contour H&E patch bbox does not intersect the source image after clipping: "
            f"contour_key={contour_key!r}, contour_id={contour_id!r}, "
            f"bbox_image_xy={[x0, y0, x1, y1]}."
        )
    return x0, y0, x1, y1


def _crop_image_level(
    level: Any,
    *,
    axes: str,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    slices = [slice(None)] * len(level.shape)
    slices[axes.index("x")] = slice(x0, x1)
    slices[axes.index("y")] = slice(y0, y1)
    try:
        return np.asarray(level[tuple(slices)]).copy()
    except TypeError:
        if hasattr(level, "open_zarr_source"):
            store, source = level.open_zarr_source()
            try:
                return np.asarray(source[tuple(slices)]).copy()
            finally:
                if hasattr(store, "close"):
                    store.close()
        if hasattr(level, "asarray"):
            return np.asarray(level.asarray()[tuple(slices)]).copy()
        raise


def _polygon_mask_for_bbox(
    *,
    image_geometry: Polygon | MultiPolygon,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    width = int(x1 - x0)
    height = int(y1 - y0)
    local_geometry = affinity.translate(image_geometry, xoff=-float(x0), yoff=-float(y0))
    yy, xx = np.mgrid[0:height, 0:width]
    sample_points = np.column_stack(
        [
            xx.reshape(-1).astype(float) + 0.5,
            yy.reshape(-1).astype(float) + 0.5,
        ]
    )

    mask = np.zeros(sample_points.shape[0], dtype=bool)
    polygons = (
        list(local_geometry.geoms)
        if isinstance(local_geometry, MultiPolygon)
        else [local_geometry]
    )
    for polygon in polygons:
        polygon_mask = _polygon_contains_points(polygon, sample_points)
        mask |= polygon_mask
    return mask.reshape(height, width)


def _polygon_contains_points(polygon: Polygon, sample_points: np.ndarray) -> np.ndarray:
    exterior_path = MplPath(np.asarray(polygon.exterior.coords, dtype=float))
    contained = exterior_path.contains_points(sample_points, radius=1e-9)
    for interior in polygon.interiors:
        hole_path = MplPath(np.asarray(interior.coords, dtype=float))
        contained &= ~hole_path.contains_points(sample_points, radius=1e-9)
    return contained


def _apply_polygon_mask(
    patch_array: np.ndarray,
    *,
    mask: np.ndarray,
    axes: str,
) -> np.ndarray:
    y_index = axes.index("y")
    x_index = axes.index("x")
    broadcast_shape = [1] * patch_array.ndim
    broadcast_shape[y_index] = mask.shape[0]
    broadcast_shape[x_index] = mask.shape[1]
    broadcast_mask = mask.reshape(broadcast_shape)
    fill_value = np.zeros((), dtype=patch_array.dtype)
    return np.where(broadcast_mask, patch_array, fill_value)


def _patch_local_affine(
    source_affine: list[list[float]] | None,
    *,
    x_offset: float,
    y_offset: float,
) -> list[list[float]]:
    if source_affine is None:
        raise ValueError("Cannot build a patch-local affine without source image alignment metadata.")
    source = np.asarray(source_affine, dtype=float)
    translation = np.asarray(
        [
            [1.0, 0.0, float(x_offset)],
            [0.0, 1.0, float(y_offset)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return (source @ translation).tolist()


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
        metadata_values[column] = _json_ready(
            _metadata_value_for_column(
                column=column,
                properties=properties,
                nested_metadata=nested_metadata,
            )
        )

    raw_properties = _json_ready(properties)
    if not isinstance(raw_properties, dict):
        raw_properties = {"properties": raw_properties}

    return str(contour_id), geometry, metadata_values, raw_properties


def _metadata_value_for_column(
    *,
    column: str,
    properties: dict[str, Any],
    nested_metadata: dict[str, Any],
) -> Any:
    if column == "classification_name":
        explicit = _first_present(
            properties,
            nested_metadata,
            keys=("classification_name",),
        )
        if explicit is not None:
            return explicit
        classification = properties.get("classification", nested_metadata.get("classification"))
        if isinstance(classification, dict) and classification.get("name") is not None:
            return classification.get("name")
    if column == "object_type":
        return _first_present(
            properties,
            nested_metadata,
            keys=("object_type", "objectType"),
        )
    return _first_present(properties, nested_metadata, keys=(column,))


def _first_present(
    properties: dict[str, Any],
    nested_metadata: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in properties:
            return properties[key]
        if key in nested_metadata:
            return nested_metadata[key]
    return None


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
