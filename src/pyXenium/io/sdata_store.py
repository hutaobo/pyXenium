from __future__ import annotations

import itertools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from .sdata_model import XeniumFrameChunkSource, XeniumImage, XeniumSData

SDATA_FORMAT = "pyxenium.sdata"
SDATA_VERSION = 1


def _ensure_writable_path(path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path).expanduser()
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {target}. Pass overwrite=True to replace it."
            )
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _column_payload(series: pd.Series) -> tuple[np.ndarray, str]:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool), "bool"
    if pd.api.types.is_integer_dtype(series):
        filled = series.fillna(0) if hasattr(series, "fillna") else series
        return np.asarray(filled.to_numpy(), dtype=np.int64), "int64"
    if pd.api.types.is_numeric_dtype(series):
        return np.asarray(series.to_numpy(), dtype=np.float64), "float64"
    return series.astype(str).to_numpy(dtype=np.str_), "string"


def _column_values_for_type(series: pd.Series, dtype_name: str) -> np.ndarray:
    if dtype_name == "bool":
        return series.fillna(False).to_numpy(dtype=bool)
    if dtype_name == "int64":
        filled = series.fillna(0) if hasattr(series, "fillna") else series
        return np.asarray(filled.to_numpy(), dtype=np.int64)
    if dtype_name == "float64":
        return np.asarray(series.to_numpy(), dtype=np.float64)
    if dtype_name == "string":
        return series.astype(str).to_numpy(dtype=np.str_)
    raise ValueError(f"Unsupported dtype_name={dtype_name!r}.")


def _zarr_dtype_for_column(
    dtype_name: str,
    *,
    string_length: int | None = None,
) -> np.dtype:
    if dtype_name == "bool":
        return np.dtype(bool)
    if dtype_name == "int64":
        return np.dtype(np.int64)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    if dtype_name == "string":
        return np.dtype(f"<U{max(int(string_length or 1), 1)}")
    raise ValueError(f"Unsupported dtype_name={dtype_name!r}.")


def _string_length(series: pd.Series) -> int:
    if series.empty:
        return 1
    lengths = series.astype(str).str.len()
    if lengths.empty:
        return 1
    return max(int(lengths.max()), 1)


def _write_frame_group(parent: Any, key: str, frame: pd.DataFrame, *, attrs: dict[str, Any] | None = None) -> None:
    group = parent.require_group(key)
    group.attrs["columns"] = list(frame.columns)
    group.attrs["n_rows"] = int(frame.shape[0])
    group.attrs["column_types"] = {}
    if attrs:
        for attr_key, attr_value in attrs.items():
            group.attrs[attr_key] = attr_value

    for column in frame.columns:
        values, dtype_name = _column_payload(frame[column])
        group.create_array(column, data=values, overwrite=True)
        column_types = dict(group.attrs.get("column_types", {}))
        column_types[column] = dtype_name
        group.attrs["column_types"] = column_types


def _write_chunked_frame_group(
    parent: Any,
    key: str,
    source: XeniumFrameChunkSource,
    *,
    attrs: dict[str, Any] | None = None,
) -> None:
    group = parent.require_group(key)
    columns = list(source.columns)
    column_types = dict(source.column_types)
    group.attrs["columns"] = columns
    group.attrs["column_types"] = column_types
    merged_attrs = dict(source.attrs)
    if attrs:
        merged_attrs.update(attrs)
    for attr_key, attr_value in merged_attrs.items():
        group.attrs[attr_key] = attr_value

    # We scan once to size fixed-width unicode arrays, then stream chunks into place.
    n_rows = 0
    chunk_rows = 0
    string_lengths = {
        column: 1
        for column, dtype_name in column_types.items()
        if dtype_name == "string"
    }
    for frame in source.iter_chunks():
        frame_rows = int(frame.shape[0])
        n_rows += frame_rows
        chunk_rows = max(chunk_rows, frame_rows)
        for column in string_lengths:
            string_lengths[column] = max(string_lengths[column], _string_length(frame[column]))

    group.attrs["n_rows"] = n_rows
    chunk_length = max(1, min(chunk_rows or 65536, n_rows or 1))
    for column in columns:
        group.create_array(
            column,
            shape=(n_rows,),
            chunks=(chunk_length,),
            dtype=_zarr_dtype_for_column(
                column_types[column],
                string_length=string_lengths.get(column),
            ),
            overwrite=True,
        )

    start = 0
    for frame in source.iter_chunks():
        stop = start + int(frame.shape[0])
        for column in columns:
            group[column][start:stop] = _column_values_for_type(frame[column], column_types[column])
        start = stop


def _read_frame_group(group: Any) -> pd.DataFrame:
    columns = list(group.attrs.get("columns", []))
    data: dict[str, Any] = {}
    column_types = dict(group.attrs.get("column_types", {}))
    for column in columns:
        values = np.asarray(group[column][...])
        dtype_name = column_types.get(column)
        if dtype_name == "string":
            data[column] = values.astype(str)
        elif dtype_name == "bool":
            data[column] = values.astype(bool)
        elif dtype_name == "int64":
            data[column] = values.astype(np.int64)
        elif dtype_name == "float64":
            data[column] = values.astype(np.float64)
        else:
            data[column] = values
    return pd.DataFrame(data, columns=columns)


def _default_image_chunks(shape: tuple[int, ...], axes: str) -> tuple[int, ...]:
    chunks = list(int(value) for value in shape)
    if "y" in axes:
        y_index = axes.index("y")
        chunks[y_index] = min(chunks[y_index], 1024)
    if "x" in axes:
        x_index = axes.index("x")
        chunks[x_index] = min(chunks[x_index], 1024)
    return tuple(chunks)


def _normalize_chunks(
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None,
    *,
    axes: str,
) -> tuple[int, ...]:
    if not chunks:
        return _default_image_chunks(shape, axes)
    normalized = []
    for size, chunk in zip(shape, chunks):
        normalized.append(max(1, min(int(chunk), int(size))))
    return tuple(normalized)


def _iter_chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    ranges = [range(0, int(size), int(chunk)) for size, chunk in zip(shape, chunks)]
    for starts in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + chunk, size))
            for start, chunk, size in zip(starts, chunks, shape)
        )


def _write_array_like_dataset(
    parent: Any,
    key: str,
    source: Any,
    *,
    axes: str,
    chunks: tuple[int, ...] | None = None,
) -> None:
    shape = tuple(int(value) for value in source.shape)
    chunk_shape = _normalize_chunks(shape, chunks, axes=axes)
    target = parent.create_array(
        key,
        shape=shape,
        chunks=chunk_shape,
        dtype=np.dtype(source.dtype),
        overwrite=True,
    )
    for slices in _iter_chunk_slices(shape, chunk_shape):
        target[slices] = np.asarray(source[slices])


def _write_image_level(parent: Any, key: str, level: Any, *, axes: str) -> None:
    if hasattr(level, "open_zarr_source"):
        store = None
        try:
            store, source = level.open_zarr_source()
            _write_array_like_dataset(
                parent,
                key,
                source,
                axes=axes,
                chunks=getattr(source, "chunks", None) or getattr(level, "chunks", None),
            )
            return
        except Exception:
            if store is not None and hasattr(store, "close"):
                store.close()
            if hasattr(level, "asarray"):
                array = level.asarray()
                _write_array_like_dataset(parent, key, array, axes=axes)
                return
            raise
        finally:
            if store is not None and hasattr(store, "close"):
                store.close()

    if hasattr(level, "shape") and hasattr(level, "__getitem__") and not isinstance(level, np.ndarray):
        _write_array_like_dataset(
            parent,
            key,
            level,
            axes=axes,
            chunks=getattr(level, "chunks", None),
        )
        return

    if hasattr(level, "asarray") and not isinstance(level, np.ndarray):
        level = level.asarray()

    _write_array_like_dataset(parent, key, np.asarray(level), axes=axes)


def _write_images_group(parent: Any, images: dict[str, XeniumImage]) -> None:
    for key, image in images.items():
        group = parent.require_group(key)
        for level_index, level in enumerate(image.levels):
            _write_image_level(group, str(level_index), level, axes=image.axes)
        transform_metadata = image.transform_metadata()
        group.attrs["axes"] = image.axes
        group.attrs["dtype"] = image.dtype
        group.attrs["multiscale_levels"] = list(range(len(image.levels)))
        group.attrs["level_shapes"] = [list(shape) for shape in image.multiscale_shapes()]
        group.attrs["source_path"] = image.source_path
        group.attrs["transform_kind"] = image.transform_kind
        group.attrs["transform_direction"] = transform_metadata["transform_direction"]
        group.attrs["transform_input_space"] = transform_metadata["transform_input_space"]
        group.attrs["transform_output_space"] = transform_metadata["transform_output_space"]
        group.attrs["transform_output_unit"] = transform_metadata["transform_output_unit"]
        group.attrs["xenium_physical_unit"] = transform_metadata["xenium_physical_unit"]
        group.attrs["image_to_xenium_affine"] = image.image_to_xenium_affine
        group.attrs["alignment_csv_path"] = image.alignment_csv_path
        group.attrs["pixel_size_um"] = image.pixel_size_um
        if "xenium_pixel_size_um" in transform_metadata:
            group.attrs["xenium_pixel_size_um"] = transform_metadata["xenium_pixel_size_um"]
        if image.keypoints_validation is not None:
            group.attrs["keypoints_validation"] = image.keypoints_validation
            for summary_key, summary_value in image.keypoints_validation.items():
                group.attrs[summary_key] = summary_value
        for meta_key, meta_value in image.metadata.items():
            if meta_key in {
                "transform_direction",
                "transform_input_space",
                "transform_output_space",
                "transform_output_unit",
                "xenium_physical_unit",
            }:
                continue
            group.attrs[meta_key] = meta_value


def _read_images_group(parent: Any) -> dict[str, XeniumImage]:
    images: dict[str, XeniumImage] = {}
    for key in parent.keys():
        group = parent[key]
        level_names = sorted(
            (str(level_key) for level_key in group.keys()),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        levels = [group[level_name] for level_name in level_names]
        keypoints_validation = group.attrs.get("keypoints_validation", None)
        metadata: dict[str, Any] = {}
        if "transform_direction" in group.attrs:
            metadata["transform_direction"] = group.attrs["transform_direction"]
        if "transform_input_space" in group.attrs:
            metadata["transform_input_space"] = group.attrs["transform_input_space"]
        if "transform_output_space" in group.attrs:
            metadata["transform_output_space"] = group.attrs["transform_output_space"]
        if "transform_output_unit" in group.attrs:
            metadata["transform_output_unit"] = group.attrs["transform_output_unit"]
        if "xenium_physical_unit" in group.attrs:
            metadata["xenium_physical_unit"] = group.attrs["xenium_physical_unit"]
        if "keypoints_csv_path" in group.attrs:
            metadata["keypoints_csv_path"] = group.attrs["keypoints_csv_path"]
        images[str(key)] = XeniumImage(
            levels=levels,
            axes=str(group.attrs["axes"]),
            dtype=str(group.attrs["dtype"]),
            source_path=str(group.attrs.get("source_path", "")),
            transform_kind=str(group.attrs.get("transform_kind", "affine")),
            image_to_xenium_affine=group.attrs.get("image_to_xenium_affine", None),
            alignment_csv_path=group.attrs.get("alignment_csv_path", None),
            pixel_size_um=group.attrs.get("pixel_size_um", None),
            keypoints_validation=keypoints_validation,
            metadata=metadata,
        )
    return images


def write_xenium_sdata(
    sdata: XeniumSData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = _ensure_writable_path(path, overwrite=overwrite)
    target.mkdir(parents=True, exist_ok=True)
    table_path = target / "tables" / "cells"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="pyxenium_table_"))
    try:
        temp_table_path = temp_root / "cells.zarr"
        sdata.table.write_zarr(str(temp_table_path))
        shutil.copytree(temp_table_path, table_path)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    root = zarr.open_group(str(target), mode="a")
    root.attrs["format"] = SDATA_FORMAT
    root.attrs["version"] = SDATA_VERSION
    root.attrs["created_by"] = "pyXenium"

    tables_root = root.require_group("tables")
    tables_root.attrs["primary_table"] = "cells"

    points_root = root.require_group("points")
    for key, frame in sdata.points.items():
        _write_frame_group(
            points_root,
            key,
            frame,
            attrs={"units": sdata.metadata.get("units", "micron")},
        )
    for key, source in sdata.point_sources.items():
        _write_chunked_frame_group(
            points_root,
            key,
            source,
            attrs={"units": sdata.metadata.get("units", "micron")},
        )

    shapes_root = root.require_group("shapes")
    for key, frame in sdata.shapes.items():
        _write_frame_group(shapes_root, key, frame, attrs={"units": sdata.metadata.get("units", "micron")})

    images_root = root.require_group("images")
    _write_images_group(images_root, sdata.images)

    metadata_root = root.require_group("metadata")
    payload = dict(sdata.metadata)
    payload.setdefault("store_version", SDATA_VERSION)
    payload.setdefault("format", SDATA_FORMAT)
    metadata_root.create_array(
        "json",
        data=np.asarray([json.dumps(payload, ensure_ascii=True, sort_keys=True)], dtype=np.str_),
        overwrite=True,
    )

    return {
        "format": SDATA_FORMAT,
        "version": SDATA_VERSION,
        "output_path": str(target),
        **sdata.component_summary(),
    }


def read_xenium_sdata(path: str | Path) -> XeniumSData:
    target = Path(path).expanduser()
    root = zarr.open_group(str(target), mode="r")
    if root.attrs.get("format") != SDATA_FORMAT:
        raise ValueError(
            f"Unsupported SData format at {target}: {root.attrs.get('format')!r}"
        )

    table = ad.read_zarr(str(target / "tables" / "cells"))

    points: dict[str, pd.DataFrame] = {}
    if "points" in root:
        for key in root["points"].keys():
            points[str(key)] = _read_frame_group(root["points"][key])

    shapes: dict[str, pd.DataFrame] = {}
    if "shapes" in root:
        for key in root["shapes"].keys():
            shapes[str(key)] = _read_frame_group(root["shapes"][key])

    images: dict[str, XeniumImage] = {}
    if "images" in root:
        images = _read_images_group(root["images"])

    metadata: dict[str, Any] = {}
    if "metadata" in root and "json" in root["metadata"]:
        values = np.asarray(root["metadata"]["json"][...]).astype(str)
        if len(values):
            metadata = json.loads(values[0])

    return XeniumSData(table=table, points=points, shapes=shapes, images=images, metadata=metadata)
