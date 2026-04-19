from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from .sdata_model import XeniumSData

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
        group.create_dataset(column, data=values, overwrite=True)
        column_types = dict(group.attrs.get("column_types", {}))
        column_types[column] = dtype_name
        group.attrs["column_types"] = column_types


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


def write_xenium_sdata(
    sdata: XeniumSData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = _ensure_writable_path(path, overwrite=overwrite)
    root = zarr.open_group(str(target), mode="w")
    root.attrs["format"] = SDATA_FORMAT
    root.attrs["version"] = SDATA_VERSION
    root.attrs["created_by"] = "pyXenium"

    tables_root = root.require_group("tables")
    table_path = target / "tables" / "cells"
    sdata.table.write_zarr(str(table_path))
    tables_root.attrs["primary_table"] = "cells"

    points_root = root.require_group("points")
    point_attrs: dict[str, dict[str, Any]] = {}
    for key, frame in sdata.points.items():
        point_attrs[key] = {"units": sdata.metadata.get("units", "micron")}
        _write_frame_group(points_root, key, frame, attrs=point_attrs[key])

    shapes_root = root.require_group("shapes")
    for key, frame in sdata.shapes.items():
        _write_frame_group(shapes_root, key, frame, attrs={"units": sdata.metadata.get("units", "micron")})

    metadata_root = root.require_group("metadata")
    payload = dict(sdata.metadata)
    payload.setdefault("store_version", SDATA_VERSION)
    payload.setdefault("format", SDATA_FORMAT)
    metadata_root.create_dataset(
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

    metadata: dict[str, Any] = {}
    if "metadata" in root and "json" in root["metadata"]:
        values = np.asarray(root["metadata"]["json"][...]).astype(str)
        if len(values):
            metadata = json.loads(values[0])

    return XeniumSData(table=table, points=points, shapes=shapes, metadata=metadata)
