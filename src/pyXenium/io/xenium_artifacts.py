from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import fsspec
import numpy as np
import pandas as pd
import requests
from scipy import sparse

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None


REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://")
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
        current = f"{current}/{text.lstrip('/\\')}"
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

    with open_text(clusters_path) as stream:
        frame = pd.read_csv(stream)
    if frame.empty:
        return pd.Series(index=barcodes, dtype="object")

    index_col = None
    for candidate in ("cell_id", "barcode", "cell", "cellID", "CellID", "cells"):
        if candidate in frame.columns:
            index_col = candidate
            break
    if index_col is None:
        index_col = frame.columns[0]

    lower_map = {column.lower(): column for column in frame.columns}
    cluster_col = None
    for key in ("cluster", "clusters", "graphclust", "label", "group"):
        if key in lower_map:
            cluster_col = lower_map[key]
            break
    if cluster_col is None:
        if frame.shape[1] == 2:
            cluster_col = next(column for column in frame.columns if column != index_col)
        else:
            cluster_col = frame.columns[-1]

    series = (
        frame[[index_col, cluster_col]]
        .dropna(subset=[index_col])
        .assign(**{index_col: lambda table: table[index_col].astype(str)})
        .set_index(index_col)[cluster_col]
        .astype(str)
        .reindex(barcodes)
    )
    series.index.name = "barcode"
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


def discover_image_artifacts(base_path: str) -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
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
