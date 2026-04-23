from __future__ import annotations

from dataclasses import dataclass
import asyncio
import gzip
import io
import json
import os
import re
import shutil
import tempfile
import threading
import warnings
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import fsspec
import numpy as np
import pandas as pd
import requests
from scipy import sparse

from .sdata_model import XeniumImage

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://")
DEFAULT_IMAGE_PYRAMID_MIN_SIZE = 256
BOUNDARY_COLUMN_CANDIDATES = {
    "cell_id": ("cell_id", "cell", "barcode", "CellID"),
    "vertex_id": ("vertex_id", "vertex", "point_id"),
    "x": ("x", "vertex_x", "x_location", "coord_x", "X"),
    "y": ("y", "vertex_y", "y_location", "coord_y", "Y"),
}
SPATIAL_COLUMN_CANDIDATES = (
    ("x_centroid", "y_centroid"),
    ("cell_x_centroid", "cell_y_centroid"),
    ("cell_centroid_x", "cell_centroid_y"),
    ("centroid_x", "centroid_y"),
    ("x", "y"),
)
IMAGE_AXIS_ALIASES = {
    "x": "x",
    "y": "y",
    "c": "c",
    "s": "c",
}


@dataclass
class TiffImageLevel:
    source_path: str
    series_index: int
    level_index: int
    shape: tuple[int, ...]
    dtype: str
    axes: str
    chunks: tuple[int, ...] | None = None

    def asarray(self) -> np.ndarray:
        if tifffile is None:
            raise ImportError("tifffile is required to materialize H&E image levels.")
        return tifffile.imread(
            self.source_path,
            series=self.series_index,
            level=self.level_index,
            maxworkers=1,
        )

    def open_zarr_source(self):
        if tifffile is None or zarr is None:
            raise ImportError(
                "tifffile and zarr are required to stream OME-TIFF pyramid levels into SData."
            )
        store = tifffile.imread(
            self.source_path,
            aszarr=True,
            series=self.series_index,
            level=self.level_index,
        )
        source = zarr.open(store, mode="r")
        return store, _resolve_zarr_array(source)


@dataclass(frozen=True)
class XeniumClusteringArtifact:
    key: str
    path: str
    relpath: str
    analysis_depth: int


@dataclass(frozen=True)
class XeniumProjectionArtifact:
    method: str
    path: str
    relpath: str
    embedding_name: str
    analysis_depth: int
    n_components_hint: int | None


@dataclass
class XeniumAnalysisBundle:
    default_cluster_key: str | None
    default_cluster_column: str | None
    cluster_series: dict[str, pd.Series]
    cluster_columns: dict[str, str]
    projection_frames: dict[str, pd.DataFrame]
    projection_keys: dict[str, str]
    cluster_sources: dict[str, dict[str, Any]]
    projection_sources: dict[str, dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "default_cluster_key": self.default_cluster_key,
            "default_cluster_column": self.default_cluster_column,
            "cluster_columns": dict(self.cluster_columns),
            "cluster_sources": dict(self.cluster_sources),
            "projection_keys": dict(self.projection_keys),
            "projection_sources": dict(self.projection_sources),
        }


def is_remote_path(path_or_url: str) -> bool:
    return str(path_or_url).startswith(REMOTE_PREFIXES)


def join_path(base: str, *names: str) -> str:
    base = str(base).rstrip("/\\")
    current = base
    for name in names:
        if not name:
            continue
        text = str(name)
        if text.startswith(REMOTE_PREFIXES):
            current = text
            continue
        if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("/"):
            current = text
            continue
        relative_text = text.lstrip("/\\")
        current = f"{current}/{relative_text}"
    return current


def exists(path_or_url: str) -> bool:
    try:
        fs, inner = fsspec.core.url_to_fs(path_or_url)
        return bool(fs.exists(inner))
    except Exception:
        try:
            with fsspec.open(path_or_url).open() as _:
                return True
        except Exception:
            return False


def open_text(path_or_url: str):
    handle = fsspec.open(path_or_url, mode="rb").open()
    if str(path_or_url).endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(fileobj=handle), encoding="utf-8")
    return io.TextIOWrapper(handle, encoding="utf-8")


def ensure_local_path(
    path_or_url: str,
    *,
    suffix: str | None = None,
    subdir: str | None = None,
) -> str:
    if not is_remote_path(path_or_url):
        return path_or_url

    root = tempfile.mkdtemp(prefix="pyxenium_")
    target_dir = Path(root)
    if subdir:
        target_dir = target_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

    basename = os.path.basename(path_or_url.rstrip("/")) or f"file{suffix or ''}"
    target = target_dir / basename

    response = requests.get(path_or_url, stream=True, timeout=60)
    response.raise_for_status()
    with target.open("wb") as stream:
        shutil.copyfileobj(response.raw, stream)
    return str(target)


def _decode_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O"}:
        return array.astype(str)
    return array


def _safe_array(group: Any, key: str) -> np.ndarray | None:
    try:
        return np.asarray(group[key][...])
    except Exception:
        return None


def _collect_async_values(async_iterable: Any) -> list[Any]:
    async def _collect() -> list[Any]:
        return [item async for item in async_iterable]

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_collect())

    result: list[Any] = []
    error: Exception | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(_collect())
        except Exception as exc:  # pragma: no cover
            error = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result


def _list_group_members(group: Any) -> list[str]:
    store = getattr(group, "store", None)
    path = getattr(group, "path", None)
    if store is None or path is None or not getattr(store, "supports_listing", False):
        return []
    if not hasattr(store, "list_dir"):
        return []

    try:
        entries = _collect_async_values(store.list_dir(path))
    except Exception:
        return []

    members: list[str] = []
    for entry in entries:
        name = str(entry).strip("/").split("/")[-1]
        if name and not name.startswith("."):
            members.append(name)
    return list(dict.fromkeys(members))


def _attach_extra_chunk_columns(
    frame: pd.DataFrame,
    *,
    chunk: Any,
    chunk_length: int,
    mask: np.ndarray,
    excluded: set[str],
) -> None:
    for key in _list_group_members(chunk):
        if key in excluded:
            continue
        values = _safe_array(chunk, key)
        if values is None:
            continue

        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != chunk_length:
            continue

        decoded = _decode_array(array)
        if decoded.ndim == 1:
            frame[key] = decoded[mask]
            continue
        if decoded.ndim == 2 and decoded.shape[1] == 1:
            frame[key] = decoded.reshape(-1)[mask]
            continue
        if decoded.ndim == 2:
            for idx in range(decoded.shape[1]):
                frame[f"{key}_{idx}"] = decoded[:, idx][mask]


def _find_zarr_root_in_zip(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    prefixes = {""}
    for name in names:
        if "/" in name:
            prefixes.add(name.rsplit("/", 1)[0] + "/")
    ordered = sorted(prefixes, key=lambda value: (value.count("/"), len(value)))

    for prefix in ordered:
        if f"{prefix}zarr.json" in names:
            return prefix.rstrip("/")
    for prefix in ordered:
        if (
            f"{prefix}.zmetadata" in names
            or f"{prefix}.zgroup" in names
            or f"{prefix}.zarray" in names
        ):
            return prefix.rstrip("/")
    raise FileNotFoundError(
        f"Could not locate a Zarr root inside archive: {zip_path}"
    )


def open_zarr_group(path_or_url: str):
    if zarr is None:
        raise ImportError("zarr is required to read Xenium Zarr artifacts.")

    resolved = ensure_local_path(path_or_url, suffix=".zip")
    if resolved.endswith((".zip", ".zarr.zip")):
        root = _find_zarr_root_in_zip(resolved)
        store = zarr.storage.ZipStore(resolved, mode="r")
        if root:
            try:
                return zarr.open_group(store=store, path=root, mode="r")
            except TypeError:  # pragma: no cover
                return zarr.open_group(store, mode="r")[root]
        return zarr.open_group(store=store, mode="r")
    return zarr.open_group(fsspec.get_mapper(resolved), mode="r")


def _resolve_zarr_array(node: Any):
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return node
    if hasattr(node, "keys"):
        keys = sorted(str(key) for key in node.keys())
        if "0" in keys:
            return _resolve_zarr_array(node["0"])
        if len(keys) == 1:
            return _resolve_zarr_array(node[keys[0]])
    raise TypeError(f"Unable to resolve an array-like object from {type(node)!r}.")


def _normalize_image_axes(axes: str) -> str:
    mapped = []
    for axis in str(axes or ""):
        key = axis.lower()
        if key not in IMAGE_AXIS_ALIASES:
            raise ValueError(f"Unsupported OME-TIFF axis {axis!r} in axes={axes!r}.")
        mapped.append(IMAGE_AXIS_ALIASES[key])
    normalized = "".join(mapped)
    if "x" not in normalized or "y" not in normalized:
        raise ValueError(f"Image axes must include X and Y, got {axes!r}.")
    return normalized


def _xy_axis_indices(axes: str) -> tuple[int, int]:
    normalized = _normalize_image_axes(axes)
    return normalized.index("y"), normalized.index("x")


def _infer_tiff_chunks(level: Any, axes: str) -> tuple[int, ...] | None:
    shape = tuple(int(value) for value in getattr(level, "shape", ()))
    if not shape or not getattr(level, "pages", None):
        return None
    page = level.pages[0]
    chunks = list(shape)
    if getattr(page, "is_tiled", False):
        y_index, x_index = _xy_axis_indices(axes)
        tile_y = int(getattr(page, "tilelength", 0) or shape[y_index])
        tile_x = int(getattr(page, "tilewidth", 0) or shape[x_index])
        chunks[y_index] = min(tile_y, shape[y_index])
        chunks[x_index] = min(tile_x, shape[x_index])
        return tuple(int(value) for value in chunks)

    rows_per_strip = int(getattr(page, "rowsperstrip", 0) or 0)
    if rows_per_strip:
        y_index, _ = _xy_axis_indices(axes)
        chunks[y_index] = min(rows_per_strip, shape[y_index])
        return tuple(int(value) for value in chunks)
    return None


def _downsample_image_level(level: np.ndarray, axes: str) -> np.ndarray:
    normalized = _normalize_image_axes(axes)
    slices = [slice(None)] * level.ndim
    slices[normalized.index("y")] = slice(None, None, 2)
    slices[normalized.index("x")] = slice(None, None, 2)
    return np.asarray(level[tuple(slices)])


def _build_image_pyramid(
    base_level: np.ndarray,
    axes: str,
    *,
    min_size: int = DEFAULT_IMAGE_PYRAMID_MIN_SIZE,
) -> list[np.ndarray]:
    normalized = _normalize_image_axes(axes)
    y_index, x_index = normalized.index("y"), normalized.index("x")
    levels = [np.asarray(base_level)]
    while min(levels[-1].shape[y_index], levels[-1].shape[x_index]) > min_size:
        next_level = _downsample_image_level(levels[-1], normalized)
        if next_level.shape == levels[-1].shape:
            break
        levels.append(next_level)
    return levels


def _find_local_matches(base_path: str, patterns: Iterable[str]) -> list[str]:
    if is_remote_path(base_path):
        return []
    root = Path(base_path).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(str(path) for path in sorted(root.glob(pattern)))
    return matches


def _first_local_match(base_path: str, patterns: Iterable[str]) -> str | None:
    matches = _find_local_matches(base_path, patterns)
    return matches[0] if matches else None


def _coerce_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    rename_map = {}
    for source, target in (
        ("feature_id", "id"),
        ("feature_name", "name"),
        ("feature_type", "feature_type"),
        ("gene_name", "name"),
    ):
        if source in frame.columns and target not in frame.columns:
            rename_map[source] = target
    frame = frame.rename(columns=rename_map)

    if "id" not in frame.columns and frame.shape[1] >= 1:
        frame["id"] = frame.iloc[:, 0].astype(str)
    if "name" not in frame.columns and frame.shape[1] >= 2:
        frame["name"] = frame.iloc[:, 1].astype(str)
    if "feature_type" not in frame.columns:
        if frame.shape[1] >= 3:
            frame["feature_type"] = frame.iloc[:, 2].astype(str)
        else:
            frame["feature_type"] = "Gene Expression"

    frame["id"] = frame["id"].astype(str)
    frame["name"] = frame["name"].astype(str)
    frame["feature_type"] = frame["feature_type"].astype(str)
    return frame[["id", "name", "feature_type"]].copy()


def read_mex_triplet(
    mex_dir: str,
    *,
    matrix_name: str = "matrix.mtx.gz",
    features_name: str = "features.tsv.gz",
    barcodes_name: str = "barcodes.tsv.gz",
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    from scipy.io import mmread

    matrix_path = join_path(mex_dir, matrix_name)
    features_path = join_path(mex_dir, features_name)
    barcodes_path = join_path(mex_dir, barcodes_name)
    if not all(exists(path) for path in (matrix_path, features_path, barcodes_path)):
        raise FileNotFoundError(
            f"MEX files missing:\n{matrix_path}\n{features_path}\n{barcodes_path}"
        )

    with fsspec.open(matrix_path).open() as stream:
        matrix = (
            mmread(gzip.GzipFile(fileobj=stream)).tocsr()
            if matrix_path.endswith(".gz")
            else mmread(stream).tocsr()
        )

    with open_text(features_path) as stream:
        features = pd.read_csv(stream, sep="\t", header=None, engine="python")
    with open_text(barcodes_path) as stream:
        barcodes = pd.read_csv(stream, sep="\t", header=None, engine="python")[0].astype(str)

    if matrix.shape[0] == len(features) and matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsr()
    elif not (matrix.shape[0] == len(barcodes) and matrix.shape[1] == len(features)):
        matrix = matrix.T.tocsr()

    features = _coerce_feature_frame(features)
    return matrix.tocsr(), features, pd.Index(barcodes.to_numpy(), name="barcode")


def read_cell_feature_matrix_zarr(base_path: str) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    if zarr is None:
        raise ImportError("zarr is required to read Xenium Zarr matrices.")

    candidates = []
    for name in ("cell_feature_matrix.zarr", "cell_feature_matrix"):
        path = join_path(base_path, name)
        if exists(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No Xenium Zarr cell_feature_matrix found.")

    store = open_zarr_group(candidates[0])
    data = np.asarray(store["X/data"][:])
    indices = np.asarray(store["X/indices"][:])
    indptr = np.asarray(store["X/indptr"][:])
    shape = tuple(np.asarray(store["X/shape"][:]).tolist())

    if len(indptr) == shape[0] + 1:
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape).T.tocsr()
    elif len(indptr) == shape[1] + 1:
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    else:
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)

    features = pd.DataFrame(
        {
            "id": _decode_array(store["features/id"][:]).astype(str),
            "name": _decode_array(store["features/name"][:]).astype(str),
            "feature_type": _decode_array(store["features/feature_type"][:]).astype(str),
        }
    )
    barcodes = pd.Index(_decode_array(store["barcodes"][:]).astype(str), name="barcode")
    return matrix, features, barcodes


def read_cell_feature_matrix_h5(h5_path: str) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    if h5py is None:
        raise ImportError("h5py is required to read Xenium HDF5 matrices.")

    handle = None
    fileobj = None
    managed = False
    try:
        try:
            fileobj = fsspec.open(h5_path).open()
            handle = h5py.File(fileobj, "r")
            managed = True
        except Exception:
            handle = h5py.File(h5_path, "r")

        group = handle.get("X") or handle.get("matrix") or handle.get("cell_feature_matrix")
        if group is None:
            raise KeyError("Unable to find Xenium matrix group in HDF5 file.")

        data = group["data"][()]
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        shape = tuple(group["shape"][()].tolist())

        if len(indptr) == shape[0] + 1:
            matrix = sparse.csr_matrix((data, indices, indptr), shape=shape).T.tocsr()
        elif len(indptr) == shape[1] + 1:
            matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
        else:
            raise ValueError(
                f"Cannot infer sparse layout from shape={shape} and len(indptr)={len(indptr)}."
            )

        def _lookup(node: Any, name: str):
            if name in node:
                return node[name]
            if name in handle:
                return handle[name]
            parent = getattr(node, "parent", None)
            if parent is not None and name in parent:
                return parent[name]
            return None

        feature_group = _lookup(group, "features")
        if feature_group is None:
            raise KeyError("Missing features group in Xenium HDF5 matrix.")

        names = feature_group.get("name") or feature_group.get("gene_names")
        if names is None:
            raise KeyError("Missing features/name dataset in Xenium HDF5 matrix.")

        features = pd.DataFrame(
            {
                "id": _decode_array(feature_group["id"][()]).astype(str),
                "name": _decode_array(names[()]).astype(str),
                "feature_type": _decode_array(feature_group["feature_type"][()]).astype(str),
            }
        )
        barcodes_dataset = _lookup(group, "barcodes")
        if barcodes_dataset is None:
            raise KeyError("Missing barcodes dataset in Xenium HDF5 matrix.")
        barcodes = pd.Index(_decode_array(barcodes_dataset[()]).astype(str), name="barcode")
        return matrix, features, barcodes
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if managed and fileobj is not None:
            try:
                fileobj.close()
            except Exception:
                pass


def read_cell_feature_matrix(
    base_path: str,
    *,
    prefer: str = "auto",
    mex_dirname: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.Index, str]:
    zarr_available = any(
        exists(join_path(base_path, name))
        for name in ("cell_feature_matrix.zarr", "cell_feature_matrix")
    )
    h5_path = join_path(base_path, "cell_feature_matrix.h5")
    mex_dir = join_path(base_path, mex_dirname)
    h5_available = exists(h5_path)
    mex_available = exists(mex_dir)

    order = ["zarr", "h5", "mex"] if prefer == "auto" else [prefer]
    for backend in order:
        if backend == "zarr" and zarr_available:
            matrix, features, barcodes = read_cell_feature_matrix_zarr(base_path)
            return matrix, features, barcodes, "zarr"
        if backend == "h5" and h5_available:
            matrix, features, barcodes = read_cell_feature_matrix_h5(h5_path)
            return matrix, features, barcodes, "h5"
        if backend == "mex" and mex_available:
            matrix, features, barcodes = read_mex_triplet(
                mex_dir,
                matrix_name=mex_matrix_name,
                features_name=mex_features_name,
                barcodes_name=mex_barcodes_name,
            )
            return matrix, features, barcodes, "mex"

    raise FileNotFoundError(
        f"No usable Xenium cell_feature_matrix found under '{base_path}' with prefer='{prefer}'."
    )


def split_rna_and_protein(
    matrix: sparse.csr_matrix,
    features: pd.DataFrame,
    barcodes: pd.Index,
) -> dict[str, Any]:
    feature_frame = _coerce_feature_frame(features)
    lower = feature_frame["feature_type"].str.lower()
    rna_mask = lower.str.contains("gene")
    protein_mask = lower.str.contains("protein")

    rna_indices = np.where(rna_mask.to_numpy())[0]
    protein_indices = np.where(protein_mask.to_numpy())[0]

    rna_matrix = (
        matrix[:, rna_indices].tocsr()
        if len(rna_indices)
        else sparse.csr_matrix((matrix.shape[0], 0), dtype=matrix.dtype)
    )
    rna_var = feature_frame.loc[rna_mask, ["id", "name", "feature_type"]].copy()
    rna_var.index = pd.Index(rna_var["id"].to_numpy(), name=None)

    feature_summary = (
        feature_frame["feature_type"].astype(str).value_counts(dropna=False).to_dict()
    )

    if len(protein_indices):
        protein_var = feature_frame.loc[protein_mask, ["id", "name", "feature_type"]].copy()
        protein_names = protein_var["name"].astype(str).tolist()
        duplicates = pd.Index(protein_names).duplicated(keep=False)
        if duplicates.any():
            counts: dict[str, int] = {}
            normalized = []
            for name in protein_names:
                counts[name] = counts.get(name, 0) + 1
                normalized.append(f"{name}_{counts[name]}" if counts[name] > 1 else name)
            protein_names = normalized
        protein_matrix = matrix[:, protein_indices].astype(np.float32).toarray()
        protein_frame = pd.DataFrame(protein_matrix, index=barcodes, columns=protein_names)
    else:
        protein_frame = pd.DataFrame(index=barcodes)

    return {
        "rna_matrix": rna_matrix,
        "rna_var": rna_var,
        "protein_frame": protein_frame,
        "feature_summary": feature_summary,
    }


def normalize_obs_index(frame: pd.DataFrame, *, barcodes: pd.Index | None = None) -> pd.DataFrame:
    obs = frame.copy()
    if "cell_id" in obs.columns:
        obs = obs.set_index("cell_id")
    elif "barcode" in obs.columns and obs.index.name != "barcode":
        obs = obs.set_index("barcode")
    obs.index = obs.index.astype(str)
    obs.index.name = "barcode"
    if barcodes is not None:
        obs = obs.reindex(barcodes).copy()
    return obs


def read_cells_table(
    base_path: str,
    *,
    cells_csv: str = "cells.csv.gz",
    cells_parquet: str | None = None,
    barcodes: pd.Index | None = None,
) -> pd.DataFrame | None:
    parquet_path = join_path(base_path, cells_parquet) if cells_parquet else None
    csv_path = join_path(base_path, cells_csv)

    if parquet_path and exists(parquet_path):
        obs = pd.read_parquet(parquet_path)
        return normalize_obs_index(obs, barcodes=barcodes)
    if exists(csv_path):
        with open_text(csv_path) as stream:
            obs = pd.read_csv(stream)
        return normalize_obs_index(obs, barcodes=barcodes)
    return None


def extract_spatial_from_obs(obs: pd.DataFrame) -> np.ndarray | None:
    for x_col, y_col in SPATIAL_COLUMN_CANDIDATES:
        if x_col in obs.columns and y_col in obs.columns:
            return obs[[x_col, y_col]].to_numpy()
    return None


def _analysis_relative_path(base_path: str, candidate_path: Path) -> str:
    root = Path(base_path).expanduser().resolve()
    try:
        return candidate_path.resolve().relative_to(root).as_posix()
    except Exception:
        return candidate_path.as_posix()


def _component_count_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)_components?$", str(name))
    if match is None:
        return None
    return int(match.group(1))


def _iter_local_analysis_csvs(base_path: str, filename: str) -> Iterator[Path]:
    if is_remote_path(base_path):
        return iter(())

    root = Path(base_path).expanduser()
    if not root.exists() or not root.is_dir():
        return iter(())
    return root.rglob(filename)


def discover_clustering_artifacts(base_path: str) -> dict[str, XeniumClusteringArtifact]:
    selected: dict[str, XeniumClusteringArtifact] = {}
    for path in _iter_local_analysis_csvs(base_path, "clusters.csv"):
        relpath = _analysis_relative_path(base_path, path)
        rel_parts = Path(relpath).parts
        if "analysis" not in rel_parts or "clustering" not in rel_parts:
            continue
        artifact = XeniumClusteringArtifact(
            key=str(path.parent.name),
            path=str(path),
            relpath=relpath,
            analysis_depth=sum(1 for part in rel_parts if part == "analysis"),
        )
        current = selected.get(artifact.key)
        if current is None or (artifact.analysis_depth, artifact.relpath) < (
            current.analysis_depth,
            current.relpath,
        ):
            selected[artifact.key] = artifact
    return selected


def discover_projection_artifacts(base_path: str) -> dict[str, XeniumProjectionArtifact]:
    selected: dict[str, XeniumProjectionArtifact] = {}
    for path in _iter_local_analysis_csvs(base_path, "projection.csv"):
        relpath = _analysis_relative_path(base_path, path)
        rel_parts = Path(relpath).parts
        if "analysis" not in rel_parts:
            continue

        method = None
        for candidate in ("umap", "pca", "tsne"):
            if candidate in rel_parts:
                method = candidate
                break
        if method is None:
            continue

        artifact = XeniumProjectionArtifact(
            method=method,
            path=str(path),
            relpath=relpath,
            embedding_name=str(path.parent.name),
            analysis_depth=sum(1 for part in rel_parts if part == "analysis"),
            n_components_hint=_component_count_from_name(path.parent.name),
        )
        current = selected.get(method)
        artifact_score = (
            artifact.analysis_depth,
            -(artifact.n_components_hint or 0),
            artifact.relpath,
        )
        if current is None:
            selected[method] = artifact
            continue
        current_score = (
            current.analysis_depth,
            -(current.n_components_hint or 0),
            current.relpath,
        )
        if artifact_score < current_score:
            selected[method] = artifact
    return selected


def _infer_frame_index_column(frame: pd.DataFrame) -> str:
    for candidate in ("Barcode", "barcode", "cell_id", "cell", "cellID", "CellID", "cells"):
        if candidate in frame.columns:
            return str(candidate)
    return str(frame.columns[0])


def _infer_cluster_column(frame: pd.DataFrame, *, index_col: str) -> str:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for key in ("cluster", "clusters", "graphclust", "label", "group"):
        if key in lower_map:
            return lower_map[key]
    if frame.shape[1] == 2:
        return next(str(column) for column in frame.columns if str(column) != index_col)
    return str(frame.columns[-1])


def _cluster_key_from_relpath(relpath: str) -> str:
    path = Path(relpath)
    if path.parent.name:
        return str(path.parent.name)
    return str(path.stem or "cluster")


def read_cluster_series_from_path(
    clusters_path: str,
    *,
    barcodes: pd.Index,
) -> tuple[pd.Series, dict[str, Any]]:
    with open_text(clusters_path) as stream:
        frame = pd.read_csv(stream)
    if frame.empty:
        empty = pd.Series(index=barcodes, dtype="object")
        empty.index.name = "barcode"
        return empty, {
            "index_column": None,
            "cluster_column": None,
            "n_rows": 0,
            "n_obs_assigned": 0,
        }

    index_col = _infer_frame_index_column(frame)
    cluster_col = _infer_cluster_column(frame, index_col=index_col)
    series = (
        frame[[index_col, cluster_col]]
        .dropna(subset=[index_col])
        .assign(**{index_col: lambda table: table[index_col].astype(str)})
        .set_index(index_col)[cluster_col]
        .astype(str)
        .reindex(barcodes)
    )
    series.index.name = "barcode"
    return series, {
        "index_column": index_col,
        "cluster_column": cluster_col,
        "n_rows": int(frame.shape[0]),
        "n_obs_assigned": int(series.notna().sum()),
    }


def _infer_projection_columns(frame: pd.DataFrame, *, index_col: str) -> list[str]:
    component_columns: list[str] = []
    for column in frame.columns:
        if str(column) == index_col:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            component_columns.append(str(column))
    return component_columns


def read_projection_frame_from_path(
    projection_path: str,
    *,
    barcodes: pd.Index,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with open_text(projection_path) as stream:
        frame = pd.read_csv(stream)
    if frame.empty:
        empty = pd.DataFrame(index=barcodes)
        empty.index.name = "barcode"
        return empty, {
            "index_column": None,
            "component_columns": [],
            "n_rows": 0,
            "n_components": 0,
            "n_obs_assigned": 0,
        }

    index_col = _infer_frame_index_column(frame)
    component_columns = _infer_projection_columns(frame, index_col=index_col)
    if not component_columns:
        empty = pd.DataFrame(index=barcodes)
        empty.index.name = "barcode"
        return empty, {
            "index_column": index_col,
            "component_columns": [],
            "n_rows": int(frame.shape[0]),
            "n_components": 0,
            "n_obs_assigned": 0,
        }

    projection = (
        frame[[index_col, *component_columns]]
        .dropna(subset=[index_col])
        .assign(**{index_col: lambda table: table[index_col].astype(str)})
        .set_index(index_col)
    )
    projection = projection.apply(pd.to_numeric, errors="coerce").reindex(barcodes)
    projection.index.name = "barcode"
    return projection, {
        "index_column": index_col,
        "component_columns": component_columns,
        "n_rows": int(frame.shape[0]),
        "n_components": int(len(component_columns)),
        "n_obs_assigned": int(projection.notna().all(axis=1).sum()),
    }


def load_xenium_analysis(
    base_path: str,
    *,
    barcodes: pd.Index,
    clusters_relpath: str | None,
    cluster_column_name: str = "cluster",
) -> XeniumAnalysisBundle:
    cluster_artifacts = discover_clustering_artifacts(base_path)
    projection_artifacts = discover_projection_artifacts(base_path)

    default_cluster_key: str | None = None
    if clusters_relpath:
        explicit_path = join_path(base_path, clusters_relpath)
        if exists(explicit_path):
            explicit_relpath = clusters_relpath.replace("\\", "/")
            explicit_key = _cluster_key_from_relpath(explicit_relpath)
            cluster_artifacts[explicit_key] = XeniumClusteringArtifact(
                key=explicit_key,
                path=explicit_path,
                relpath=explicit_relpath,
                analysis_depth=explicit_relpath.split("/").count("analysis"),
            )
            default_cluster_key = explicit_key

    if default_cluster_key is None and "gene_expression_graphclust" in cluster_artifacts:
        default_cluster_key = "gene_expression_graphclust"
    if default_cluster_key is None and cluster_artifacts:
        default_cluster_key = sorted(cluster_artifacts)[0]

    cluster_series: dict[str, pd.Series] = {}
    cluster_columns: dict[str, str] = {}
    cluster_sources: dict[str, dict[str, Any]] = {}

    for key in sorted(cluster_artifacts):
        artifact = cluster_artifacts[key]
        series, meta = read_cluster_series_from_path(artifact.path, barcodes=barcodes)
        if series.empty and meta["n_rows"] == 0:
            continue
        cluster_series[key] = series
        cluster_columns[key] = (
            cluster_column_name if key == default_cluster_key else f"{cluster_column_name}__{key}"
        )
        cluster_sources[key] = {
            "path": artifact.path,
            "relpath": artifact.relpath,
            "index_column": meta["index_column"],
            "cluster_column": meta["cluster_column"],
            "n_rows": meta["n_rows"],
            "n_obs_assigned": meta["n_obs_assigned"],
        }

    projection_frames: dict[str, pd.DataFrame] = {}
    projection_keys: dict[str, str] = {}
    projection_sources: dict[str, dict[str, Any]] = {}

    for method in sorted(projection_artifacts):
        artifact = projection_artifacts[method]
        frame, meta = read_projection_frame_from_path(artifact.path, barcodes=barcodes)
        if frame.shape[1] == 0:
            continue
        projection_frames[method] = frame
        projection_keys[method] = f"X_{method}"
        projection_sources[method] = {
            "path": artifact.path,
            "relpath": artifact.relpath,
            "embedding_name": artifact.embedding_name,
            "index_column": meta["index_column"],
            "component_columns": meta["component_columns"],
            "n_rows": meta["n_rows"],
            "n_components": meta["n_components"],
            "n_obs_assigned": meta["n_obs_assigned"],
        }

    default_cluster_column = (
        cluster_columns.get(default_cluster_key)
        if default_cluster_key is not None
        else None
    )

    return XeniumAnalysisBundle(
        default_cluster_key=default_cluster_key,
        default_cluster_column=default_cluster_column,
        cluster_series=cluster_series,
        cluster_columns=cluster_columns,
        projection_frames=projection_frames,
        projection_keys=projection_keys,
        cluster_sources=cluster_sources,
        projection_sources=projection_sources,
    )


def read_clusters_series(
    base_path: str,
    *,
    clusters_relpath: str | None,
    barcodes: pd.Index,
) -> pd.Series | None:
    if not clusters_relpath:
        return None

    clusters_path = join_path(base_path, clusters_relpath)
    if not exists(clusters_path):
        return None

    series, _ = read_cluster_series_from_path(clusters_path, barcodes=barcodes)
    return series


def _match_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def normalize_boundary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy()
    mapping: dict[str, str] = {}
    for target, candidates in BOUNDARY_COLUMN_CANDIDATES.items():
        column = _match_column(source, candidates)
        if column is not None:
            mapping[column] = target
    source = source.rename(columns=mapping)

    missing = [name for name in ("cell_id", "x", "y") if name not in source.columns]
    if missing:
        raise ValueError(f"Boundary table is missing required columns: {missing}")

    if "vertex_id" not in source.columns:
        source["vertex_id"] = source.groupby("cell_id").cumcount()

    normalized = source[["cell_id", "vertex_id", "x", "y"]].copy()
    normalized["cell_id"] = normalized["cell_id"].astype(str)
    normalized["vertex_id"] = pd.to_numeric(normalized["vertex_id"], errors="coerce").fillna(0).astype(int)
    normalized["x"] = pd.to_numeric(normalized["x"], errors="coerce")
    normalized["y"] = pd.to_numeric(normalized["y"], errors="coerce")
    return normalized


def read_boundary_tables(
    base_path: str,
    *,
    include_cell: bool = True,
    include_nucleus: bool = True,
) -> dict[str, pd.DataFrame]:
    boundaries: dict[str, pd.DataFrame] = {}
    for filename, key, enabled in (
        ("cell_boundaries.csv.gz", "cell_boundaries", include_cell),
        ("nucleus_boundaries.csv.gz", "nucleus_boundaries", include_nucleus),
    ):
        if not enabled:
            continue
        path = join_path(base_path, filename)
        if not exists(path):
            continue
        with open_text(path) as stream:
            frame = pd.read_csv(stream)
        boundaries[key] = normalize_boundary_frame(frame)
    return boundaries


def _cell_id_ints_to_strings(cell_id_u32: np.ndarray) -> np.ndarray:
    if cell_id_u32.ndim != 2 or cell_id_u32.shape[1] != 2:
        raise ValueError(f"cell_id array must have shape (N, 2), got {cell_id_u32.shape}")
    translation = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    out = []
    for prefix, suffix in cell_id_u32:
        out.append(f"{int(prefix):08x}".translate(translation) + f"-{int(suffix)}")
    return np.asarray(out, dtype=object)


def read_cells_zarr_spatial(cells_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    group = open_zarr_group(cells_path)
    if "cell_id" not in group or "cell_summary" not in group:
        raise KeyError("cells.zarr artifact is missing 'cell_id' or 'cell_summary'.")

    cell_ids = _cell_id_ints_to_strings(np.asarray(group["cell_id"][:]))
    values = np.asarray(group["cell_summary"][:])
    attrs = {}
    try:
        attrs = dict(group["cell_summary"].attrs.items())
    except Exception:
        attrs = {}

    columns = attrs.get("columns") or attrs.get("col_names")
    if not columns:
        columns = [
            "cell_centroid_x",
            "cell_centroid_y",
            "cell_area",
            "nucleus_centroid_x",
            "nucleus_centroid_y",
            "nucleus_area",
            "z_level",
            "nucleus_count",
        ][: values.shape[1]]

    frame = pd.DataFrame(values, columns=columns, index=pd.Index(cell_ids, name="barcode"))
    for old, new in (("cell_centroid_x", "x"), ("cell_centroid_y", "y")):
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(columns={old: new})
    keep = [column for column in ("x", "y") if column in frame.columns]
    return frame[keep].copy(), {
        "spatial_units": "micron",
        "cell_summary_columns": list(columns),
    }


def summarize_analysis_group(group: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        if "cell_groups" in group:
            cell_groups = group["cell_groups"]
            if hasattr(cell_groups, "keys"):
                payload["cell_groups_keys"] = sorted(str(key) for key in cell_groups.keys())
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


def summarize_cells_group(group: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        keys = []
        if hasattr(group, "keys"):
            keys = sorted(str(key) for key in group.keys())
        payload["top_level"] = keys
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


def summarize_zarr_artifact(path_or_url: str, *, kind: str) -> dict[str, Any]:
    group = open_zarr_group(path_or_url)
    if kind == "analysis":
        return summarize_analysis_group(group)
    if kind == "cells":
        return summarize_cells_group(group)
    raise ValueError(f"Unsupported artifact summary kind: {kind}")


def read_analysis_cell_groups(
    analysis_path: str,
    *,
    n_cells: int,
    prefer_group: str = "0",
) -> tuple[pd.Series, dict[str, Any]]:
    group = open_zarr_group(analysis_path)
    if "cell_groups" not in group:
        raise KeyError("analysis.zarr artifact is missing 'cell_groups'.")

    cell_groups = group["cell_groups"]
    if hasattr(cell_groups, "keys"):
        keys = [str(key) for key in cell_groups.keys()]
    else:  # pragma: no cover
        keys = []
    keys = sorted(keys, key=lambda value: int(value) if value.isdigit() else 10**9)
    group_key = prefer_group if prefer_group in keys else keys[0]

    indices = np.asarray(cell_groups[group_key]["indices"][:], dtype=np.int64)
    indptr = np.asarray(cell_groups[group_key]["indptr"][:], dtype=np.int64)
    n_groups = len(indptr) - 1

    matrix = sparse.csr_matrix(
        (np.ones_like(indices, dtype=np.uint8), indices, indptr),
        shape=(n_groups, int(n_cells)),
    )
    assigned = np.asarray(matrix.sum(axis=0)).ravel()
    labels = np.asarray(matrix.argmax(axis=0)).ravel()
    labels[assigned == 0] = -1

    names: list[str] = []
    try:
        attr_names = group["cell_groups"].attrs.get("group_names", None)
        if isinstance(attr_names, (list, tuple)) and len(attr_names) > int(group_key):
            names = [str(value) for value in attr_names[int(group_key)]]
    except Exception:
        names = []

    def _label_name(index: int) -> str:
        if index < 0:
            return "Unassigned"
        if 0 <= index < len(names):
            return names[index]
        return f"Cluster {index + 1}"

    series = pd.Series([_label_name(int(value)) for value in labels], name="cluster")
    meta = {
        "group_key": str(group_key),
        "n_clusters": int(n_groups),
        "grouping_names": group["cell_groups"].attrs.get("grouping_names", None),
    }
    return series, meta


def _resolve_gene_names(root: Any) -> list[str]:
    if "gene_names" not in root.attrs:
        raise KeyError("Transcripts Zarr is missing attrs['gene_names'].")
    raw = root.attrs["gene_names"]
    return [str(item) for item in raw]


def _transcript_level_group(root: Any):
    if "grids" in root and "0" in root["grids"]:
        return root["grids"]["0"]
    return root


def iter_transcript_chunks(
    transcripts_path: str,
    *,
    genes: set[str] | None = None,
) -> Iterator[pd.DataFrame]:
    root = open_zarr_group(transcripts_path)
    gene_names = _resolve_gene_names(root)
    level = _transcript_level_group(root)

    for key in list(level.keys()):
        try:
            chunk = level[key]
        except Exception:
            continue
        if "location" not in chunk or "gene_identity" not in chunk:
            continue

        coords = np.asarray(chunk["location"][:])
        if coords.size == 0:
            continue

        gene_ids = np.asarray(chunk["gene_identity"][:]).reshape(-1)
        if gene_ids.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Transcript chunk '{key}' has mismatched location and gene_identity lengths."
            )

        gene_labels = np.array(
            [
                gene_names[int(gene_id)] if 0 <= int(gene_id) < len(gene_names) else str(gene_id)
                for gene_id in gene_ids
            ],
            dtype=object,
        )
        mask = np.ones(coords.shape[0], dtype=bool)
        if genes:
            mask &= np.isin(gene_labels, list(genes))
        if not np.any(mask):
            continue

        quality = _safe_array(chunk, "quality_score")
        valid = _safe_array(chunk, "valid")
        cell_id = _safe_array(chunk, "cell_id")

        frame = pd.DataFrame(
            {
                "x": coords[mask, 0].astype(float),
                "y": coords[mask, 1].astype(float),
                "gene_identity": gene_ids[mask].astype(int),
                "gene_name": gene_labels[mask].astype(str),
            }
        )
        if coords.shape[1] > 2:
            frame["z"] = coords[mask, 2].astype(float)

        if quality is not None:
            frame["quality_score"] = np.asarray(quality).reshape(-1)[mask]
        else:
            frame["quality_score"] = np.nan

        if valid is not None:
            frame["valid"] = np.asarray(valid).reshape(-1)[mask].astype(bool)
        else:
            frame["valid"] = True

        if cell_id is not None:
            cell_values = np.asarray(cell_id)
            if cell_values.ndim == 2 and cell_values.shape[1] == 2:
                mapped = _cell_id_ints_to_strings(cell_values)
                frame["cell_id"] = mapped[mask].astype(str)
            else:
                frame["cell_id"] = _decode_array(cell_values).reshape(-1)[mask].astype(str)

        _attach_extra_chunk_columns(
            frame,
            chunk=chunk,
            chunk_length=coords.shape[0],
            mask=mask,
            excluded={"location", "gene_identity", "quality_score", "valid", "cell_id"},
        )
        yield frame.reset_index(drop=True)


def read_transcripts_table(
    transcripts_path: str,
    *,
    genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    gene_filter = set(genes) if genes is not None else None
    chunks = list(iter_transcript_chunks(transcripts_path, genes=gene_filter))
    if not chunks:
        columns = ["x", "y", "gene_identity", "gene_name", "quality_score", "valid", "cell_id"]
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def resolve_transcripts_path(base_path: str) -> str | None:
    for candidate in (
        "transcripts.zarr.zip",
        "transcripts.zarr",
        "analysis/transcripts.zarr.zip",
        "analysis/transcripts.zarr",
    ):
        resolved = join_path(base_path, candidate)
        if exists(resolved):
            return resolved
    return None


def read_experiment_metadata(base_path: str) -> dict[str, Any]:
    experiment_path = join_path(base_path, "experiment.xenium")
    if not exists(experiment_path):
        return {}
    with open_text(experiment_path) as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        return payload
    return {}


def read_alignment_affine(path_or_url: str) -> np.ndarray:
    with open_text(path_or_url) as stream:
        frame = pd.read_csv(stream, header=None)
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"H&E alignment CSV must contain a 3x3 matrix, got {matrix.shape}.")
    return matrix


def read_keypoints_validation(
    path_or_url: str,
    *,
    affine: np.ndarray,
) -> dict[str, Any]:
    with open_text(path_or_url) as stream:
        frame = pd.read_csv(stream)
    lower_map = {column.lower(): column for column in frame.columns}
    required = {
        "fixedx": "fixedX",
        "fixedy": "fixedY",
        "alignmentx": "alignmentX",
        "alignmenty": "alignmentY",
    }
    missing = [display for key, display in required.items() if key not in lower_map]
    if missing:
        raise ValueError(
            f"H&E keypoints CSV is missing required columns: {sorted(missing)}"
        )

    fixed = frame[[lower_map["fixedx"], lower_map["fixedy"]]].to_numpy(dtype=float)
    moving = frame[[lower_map["alignmentx"], lower_map["alignmenty"]]].to_numpy(dtype=float)
    transformed = np.c_[moving, np.ones(len(moving), dtype=float)] @ affine.T
    residuals = np.sqrt(np.square(transformed[:, :2] - fixed).sum(axis=1))
    return {
        "n_keypoints": int(len(frame)),
        "mean_residual": float(residuals.mean()) if len(residuals) else 0.0,
        "median_residual": float(np.median(residuals)) if len(residuals) else 0.0,
        "max_residual": float(residuals.max()) if len(residuals) else 0.0,
    }


def discover_he_artifacts(base_path: str) -> dict[str, Any] | None:
    image_path = _first_local_match(
        base_path,
        ("*_he_image.ome.tif", "*_he_image.ome.tiff"),
    )
    if image_path is None:
        return None

    alignment_path = _first_local_match(base_path, ("*_he_alignment.csv",))
    keypoints_path = _first_local_match(base_path, ("*_keypoints.csv",))
    experiment = {}
    pixel_size_um = None
    try:
        experiment = read_experiment_metadata(base_path)
        if "pixel_size" in experiment:
            pixel_size_um = float(experiment["pixel_size"])
    except Exception as exc:
        warnings.warn(f"Failed to read experiment.xenium metadata: {exc}", stacklevel=2)

    affine = None
    if alignment_path is not None:
        affine = read_alignment_affine(alignment_path)

    keypoints_validation = None
    if keypoints_path is not None and affine is not None:
        keypoints_validation = read_keypoints_validation(keypoints_path, affine=affine)

    return {
        "name": "he",
        "path": image_path,
        "source_path": image_path,
        "alignment_csv_path": alignment_path,
        "keypoints_csv_path": keypoints_path,
        "transform_kind": "affine",
        "transform_direction": "image_pixel_xy_to_xenium_pixel_xy",
        "transform_input_space": "image_pixel_xy",
        "transform_output_space": "xenium_pixel_xy",
        "transform_output_unit": "pixel",
        "xenium_physical_unit": "micron",
        "image_to_xenium_affine": affine.tolist() if affine is not None else None,
        "pixel_size_um": pixel_size_um,
        "keypoints_validation": keypoints_validation,
        "experiment": experiment,
    }


def read_he_image(base_path: str) -> XeniumImage | None:
    artifact = discover_he_artifacts(base_path)
    if artifact is None:
        return None
    if tifffile is None:
        raise ImportError("tifffile is required to read Xenium H&E OME-TIFF images.")

    image_path = artifact["source_path"]
    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        axes = _normalize_image_axes(series.axes)
        dtype = np.dtype(series.dtype).name
        if len(series.levels) > 1:
            levels: list[Any] = []
            for level_index, level in enumerate(series.levels):
                levels.append(
                    TiffImageLevel(
                        source_path=image_path,
                        series_index=0,
                        level_index=level_index,
                        shape=tuple(int(value) for value in level.shape),
                        dtype=np.dtype(level.dtype).name,
                        axes=axes,
                        chunks=_infer_tiff_chunks(level, series.axes),
                    )
                )
        else:
            base_level = series.asarray(maxworkers=1)
            levels = _build_image_pyramid(base_level, axes)

    metadata = {
        "transform_direction": artifact["transform_direction"],
        "transform_input_space": artifact["transform_input_space"],
        "transform_output_space": artifact["transform_output_space"],
        "transform_output_unit": artifact["transform_output_unit"],
        "xenium_physical_unit": artifact["xenium_physical_unit"],
    }
    if artifact["keypoints_csv_path"] is not None:
        metadata["keypoints_csv_path"] = artifact["keypoints_csv_path"]

    return XeniumImage(
        levels=levels,
        axes=axes,
        dtype=dtype,
        source_path=image_path,
        transform_kind=artifact["transform_kind"],
        image_to_xenium_affine=artifact["image_to_xenium_affine"],
        alignment_csv_path=artifact["alignment_csv_path"],
        pixel_size_um=artifact["pixel_size_um"],
        keypoints_validation=artifact["keypoints_validation"],
        metadata=metadata,
    )


def discover_image_artifacts(base_path: str) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    he_artifact = discover_he_artifacts(base_path)
    if he_artifact is not None:
        images["he"] = {
            "path": he_artifact["path"],
            "alignment_csv_path": he_artifact["alignment_csv_path"],
            "transform_direction": he_artifact["transform_direction"],
            "transform_input_space": he_artifact["transform_input_space"],
            "transform_output_space": he_artifact["transform_output_space"],
            "transform_output_unit": he_artifact["transform_output_unit"],
            "xenium_physical_unit": he_artifact["xenium_physical_unit"],
            "image_to_xenium_affine": he_artifact["image_to_xenium_affine"],
            "pixel_size_um": he_artifact["pixel_size_um"],
            "keypoints_validation": he_artifact["keypoints_validation"],
        }

    candidates = {
        "morphology_focus": (
            "morphology_focus.ome.tif",
            "morphology_focus.ome.tiff",
            "morphology_focus",
        ),
        "morphology_mip": (
            "morphology_mip.ome.tif",
            "morphology_mip.ome.tiff",
            "morphology_mip",
        ),
        "aligned_images": (
            "aligned_images",
            "aligned_images.ome.tif",
            "aligned_images.ome.tiff",
        ),
    }
    for key, options in candidates.items():
        for option in options:
            path = join_path(base_path, option)
            if exists(path):
                images[key] = {"path": path}
                break
    return images


def build_feature_summary(features: pd.DataFrame) -> dict[str, Any]:
    frame = _coerce_feature_frame(features)
    return {
        "n_features_total": int(frame.shape[0]),
        "feature_types": {
            str(key): int(value)
            for key, value in frame["feature_type"].value_counts(dropna=False).to_dict().items()
        },
    }


def serialize_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)
