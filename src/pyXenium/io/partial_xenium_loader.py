"""
partial_xenium_loader.py — pyXenium I/O helpers

This module adds **robust support for 10x Xenium *.zarr.zip** bundles where the
Zarr root metadata (v2 or v3) is nested inside a subdirectory (e.g. "analysis/"
or "cells.zarr/") rather than at the ZIP root.

Key features
------------
- Auto-detect Zarr root *inside* a ZIP (supports Zarr v2 & v3):
  - v3 detected via `zarr.json`
  - v2 detected via `.zmetadata`, `.zgroup` or `.zarray`
- Works without extracting the archive using `fsspec`'s `zip://` mapper.
- Backwards-compatible call signature for `load_anndata_from_partial(...)`.
- Will gracefully continue if analysis/cells Zarr are missing or partially
  structured — you still get an AnnData built from MEX if available.

Dependencies
------------
- anndata, numpy, pandas, scipy (io.mmread), zarr (>=2.13), fsspec, requests

"""
from __future__ import annotations

import io
import os
import re
import sys
import json
import gzip
import uuid
import time
import math
import shutil
import zipfile
import warnings
import tempfile
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.sparse import csr_matrix

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("anndata is required: pip install anndata") from e

try:
    import zarr
    import fsspec
except Exception as e:  # pragma: no cover
    raise ImportError("zarr and fsspec are required: pip install zarr fsspec") from e

try:
    import requests
except Exception as e:  # pragma: no cover
    raise ImportError("requests is required: pip install requests") from e


# ------------------------------
# Logging helpers
# ------------------------------
_DEF_TS = lambda: time.strftime("%H:%M:%S")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}")


# ------------------------------
# HTTP / Local utils
# ------------------------------

def _is_http(s: str) -> bool:
    return bool(re.match(r"^https?://", str(s)))


def _download(url: str, out_path: str) -> str:
    _info(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return out_path


def _ensure_local(path_or_url: str, suffix: Optional[str] = None, subdir: Optional[str] = None) -> str:
    """Return a local filesystem path for a local path or URL.
    If URL, download to a tmp dir (optionally into a subdir).
    """
    if not _is_http(path_or_url):
        return path_or_url
    tmpdir = tempfile.mkdtemp(prefix="pyxenium_")
    if subdir:
        os.makedirs(os.path.join(tmpdir, subdir), exist_ok=True)
        out_path = os.path.join(tmpdir, subdir, (os.path.basename(path_or_url) or f"file{suffix or ''}"))
    else:
        out_path = os.path.join(tmpdir, os.path.basename(path_or_url) or f"file{suffix or ''}")
    return _download(path_or_url, out_path)


# ------------------------------
# Zarr-in-ZIP robust opener
# ------------------------------

def _find_zarr_root_in_zip(zip_path: str) -> Tuple[str, str]:
    """
    Inspect a ZIP and return (root_prefix, zarr_version) where zarr_version ∈ {"v2","v3"}.
    The returned `root_prefix` is a directory path inside the ZIP (possibly "").
    Detection order: v3 via `zarr.json`, then v2 via `.zmetadata`/`.zgroup`/`.zarray`.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    # Collect candidate directory prefixes (depth 1 and deeper)
    prefixes = {""}
    for n in names:
        if n.endswith('/'):
            # keep full folder paths as-is
            prefixes.add(n)
        elif '/' in n:
            prefixes.add(n.rsplit('/', 1)[0] + '/')

    def has(pfx: str, fname: str) -> bool:
        target = f"{pfx}{fname}" if pfx else fname
        return target in names

    # Prefer shorter prefixes first
    sorted_pfx = sorted(prefixes, key=lambda s: (s.count('/'), len(s)))

    # Try v3
    for pfx in sorted_pfx:
        if has(pfx, "zarr.json"):
            return pfx, "v3"

    # Try v2 consolidated / group / array
    for pfx in sorted_pfx:
        if has(pfx, ".zmetadata") or has(pfx, ".zgroup") or has(pfx, ".zarray"):
            return pfx, "v2"

    raise FileNotFoundError(
        "Could not locate Zarr root in ZIP (no zarr.json/.zmetadata/.zgroup/.zarray)."
    )


def _open_zarr_zip(zip_path: str):
    """Open a zipped Zarr (v2 or v3) **without extraction** and return a zarr Group.
    The Zarr root may be inside a subfolder of the ZIP.
    Uses zarr.storage.ZipStore to avoid fsspec path quirks in certain envs.
    """
    if os.path.isdir(zip_path):
        raise IsADirectoryError(f"Expected a .zip file, got directory: {zip_path}")

    root_prefix, zv = _find_zarr_root_in_zip(zip_path)

    # Prefer ZipStore for local file paths; it's robust and avoids cwd confusion
    try:
        store = zarr.storage.ZipStore(zip_path, mode="r")
    except Exception as e:
        raise RuntimeError(f"Failed to open ZipStore for {zip_path}: {e}") from e

    path = root_prefix.rstrip('/')
    try:
        grp = zarr.open_group(store=store, path=path, mode="r")
    except TypeError:
        # very old zarr fallback
        grp = zarr.open_group(store, mode="r")
        if path:
            grp = grp[path]
    return grp


# ------------------------------
# MEX reader
# ------------------------------

def _read_mex_triplet(matrix_path: str, features_path: str, barcodes_path: str) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    _info(f"Reading MEX from {os.path.dirname(matrix_path)}")
    _info(f"  using {os.path.basename(matrix_path)}, {os.path.basename(features_path)}, {os.path.basename(barcodes_path)}")

    # Read matrix
    if matrix_path.endswith('.gz'):
        with gzip.open(matrix_path, 'rb') as f:
            X = mmread(f).tocsr()
    else:
        X = mmread(matrix_path).tocsr()

    # Read features (TSV: id, name, feature_type)
    # pandas engine='python' to be safe for regex separators in some 10x dumps
    features = pd.read_csv(features_path, sep='\t', header=None, engine='python')
    if features.shape[1] == 3:
        features.columns = ["feature_id", "feature_name", "feature_type"]
    else:
        # be forgiving
        cols = [f"col{i}" for i in range(features.shape[1])]
        features.columns = cols
        if "feature_id" not in cols:
            features["feature_id"] = features.iloc[:, 0].astype(str)
        if "feature_type" not in cols:
            features["feature_type"] = "Gene"

    # Read barcodes
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None, engine='python')
    barcodes.columns = ["barcode"]

    # Ensure matrix orientation is cells x genes (n_obs x n_vars)
    # 10x MEX is typically genes x cells; mmread returns (n_rows x n_cols) as stored
    # We detect whether rows == features and cols == barcodes; if so, transpose.
    need_T = False
    if X.shape[0] == features.shape[0] and X.shape[1] == barcodes.shape[0]:
        need_T = True
    elif X.shape[0] == barcodes.shape[0] and X.shape[1] == features.shape[0]:
        need_T = False
    else:
        # fallback heuristic: larger dimension is cells
        need_T = X.shape[0] == features.shape[0]

    if need_T:
        X = X.T.tocsr()

    return X, features, barcodes


# ------------------------------
# Other helper
# ------------------------------

def _cell_id_ints_to_strings(cell_id_u32: np.ndarray) -> np.ndarray:
    """
    Convert Xenium integer cell_id (uint32 [N,2] -> [prefix, dataset_suffix])
    to string barcodes like 'ffkpbaba-1', per 10x mapping:
      1) prefix -> 8-digit hex
      2) hex digits 0..9,a..f shift to a..p
      3) append '-' + dataset_suffix (decimal)
    """
    if cell_id_u32.ndim != 2 or cell_id_u32.shape[1] != 2:
        raise ValueError(f"cell_id array must be shape (N,2), got {cell_id_u32.shape}")
    trans = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    out = []
    for p, s in cell_id_u32:
        h = f"{int(p):08x}"
        out.append(h.translate(trans) + "-" + str(int(s)))
    return np.asarray(out, dtype=object)


def _extract_cells_xy_and_ids(cells_zip_path: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Return (df_xy, cell_ids_str, meta) from cells.zarr.zip.
    df_xy: index=cell_id_str, columns ['x','y']  (microns)
    meta:  optional extra info (e.g., original column names)
    """
    grp = _open_zarr_zip(cells_zip_path)
    if "cell_id" not in grp or "cell_summary" not in grp:
        raise KeyError("cells.zarr.zip missing required arrays 'cell_id' and/or 'cell_summary'.")

    # map integer cell_id -> string barcodes
    cid = grp["cell_id"][:]
    cell_ids_str = _cell_id_ints_to_strings(np.asarray(cid))

    # read cell_summary table
    arr = np.asarray(grp["cell_summary"][:])   # shape (N, 8) typically
    attrs = {}
    try:
        attrs = dict(grp["cell_summary"].attrs.items())
    except Exception:
        pass

    # column names (fallback to spec order)
    cols = (attrs.get("columns") or attrs.get("col_names")
            or ["cell_centroid_x","cell_centroid_y","cell_area",
                "nucleus_centroid_x","nucleus_centroid_y","nucleus_area",
                "z_level","nucleus_count"][:arr.shape[1]])
    # build and rename to x,y
    df = pd.DataFrame(arr, columns=cols)
    colmap = {}
    if "cell_centroid_x" in df.columns: colmap["cell_centroid_x"] = "x"
    if "cell_centroid_y" in df.columns: colmap["cell_centroid_y"] = "y"
    if colmap:
        df = df.rename(columns=colmap)
    keep = [c for c in ["x","y"] if c in df.columns]
    df_xy = pd.DataFrame(index=cell_ids_str, data=df[keep].to_numpy(), columns=keep)

    meta = {"cell_summary_cols": cols, "spatial_units": "micron"}
    return df_xy, cell_ids_str, meta


def _labels_from_analysis_cell_groups(analysis_zip_path: str, n_cells: int,
                                      prefer_group: str = "0") -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Build per-cell labels from analysis.zarr.zip/cell_groups/[prefer_group].
    Returns (labels_series indexed by 0..n_cells-1, meta)
    """
    grp = _open_zarr_zip(analysis_zip_path)
    if "cell_groups" not in grp:
        raise KeyError("analysis.zarr.zip missing 'cell_groups'.")

    cg = grp["cell_groups"]
    # find group key: prefer '0' (graph-based), otherwise first available
    keys = []
    if hasattr(cg, "group_keys"):
        keys = list(cg.group_keys())
    else:
        keys = list(cg.keys())
    keys = sorted(keys, key=lambda s: int(s) if str(s).isdigit() else 1e9)
    gkey = prefer_group if prefer_group in keys else keys[0]

    idx = np.asarray(cg[gkey]["indices"][:], dtype=np.int64)
    indptr = np.asarray(cg[gkey]["indptr"][:], dtype=np.int64)
    n_rows = len(indptr) - 1
    n_cols = int(n_cells)

    data = np.ones_like(idx, dtype=np.uint8)
    M = csr_matrix((data, idx, indptr), shape=(n_rows, n_cols))  # rows=clusters, cols=cells

    nz_per_cell = np.asarray(M.sum(axis=0)).ravel()
    labels_idx = np.array(M.argmax(axis=0)).ravel()
    labels_idx[nz_per_cell == 0] = -1  # unassigned

    # optional pretty names from attributes
    gnames = []
    try:
        group_names_all = grp["cell_groups"].attrs.get("group_names", None)
        if isinstance(group_names_all, (list, tuple)) and len(group_names_all) > int(gkey):
            gnames = list(group_names_all[int(gkey)])
    except Exception:
        pass

    def name_of(i):
        if i < 0:
            return "Unassigned"
        if 0 <= i < len(gnames):
            return gnames[i]
        return f"Cluster {i+1}"

    labels_str = np.array([name_of(i) for i in labels_idx], dtype=object)
    ser = pd.Series(labels_str, index=np.arange(n_cols), name="cluster")

    meta = {"group_key": str(gkey), "n_clusters": n_rows, "grouping_names": grp["cell_groups"].attrs.get("grouping_names", None)}
    return ser, meta


# ------------------------------
# Parsers for 10x Zarr groups (best-effort, schema-tolerant)
# ------------------------------

def _parse_analysis_group(grp) -> Dict[str, Any]:
    """Best-effort parser for analysis.zarr.zip
    Returns a dict to stash under adata.uns["analysis"] if available.
    """
    out: Dict[str, Any] = {}
    try:
        if "cell_groups" in grp:
            cg = grp["cell_groups"]
            # Many 10x bundles expose tables as subgroups with 1D/2D arrays.
            # We try to summarize keys rather than enforcing a strict schema.
            out["cell_groups_keys"] = sorted(list(cg.array_keys()) if hasattr(cg, "array_keys") else list(cg.keys()))
    except Exception as e:
        _warn(f"Failed to parse analysis cell_groups: {e}")
    return out


def _parse_cells_group(grp) -> Dict[str, Any]:
    """Best-effort parser for cells.zarr.zip
    Returns a dict to stash under adata.uns["cells"] if available.
    Tolerates both Groups and Arrays (some keys like "cell_summary" may be arrays).
    """
    out: Dict[str, Any] = {}

    def _summarize(node):
        try:
            # Array-like: has shape/dtype
            if hasattr(node, "shape") and hasattr(node, "dtype") and not hasattr(node, "keys"):
                return {"type": "array", "shape": tuple(node.shape), "dtype": str(node.dtype)}
            # Group-like
            keys = []
            if hasattr(node, "array_keys"):
                keys = sorted(list(node.array_keys()))
            elif hasattr(node, "keys"):
                keys = sorted(list(node.keys()))
            return {"type": "group", "keys": keys}
        except Exception as e:
            return {"type": "unknown", "error": str(e)}

    try:
        # Common subgroups/arrays observed in Xenium bundles
        for k in ("cell_summary", "masks", "polygon_sets", "polygons", "cells"):
            if k in grp:
                out[k] = _summarize(grp[k])
        # Also provide a shallow directory of top-level entries for debugging
        try:
            top_keys = []
            if hasattr(grp, "array_keys"):
                top_keys = sorted(list(grp.array_keys()))
            elif hasattr(grp, "keys"):
                top_keys = sorted(list(grp.keys()))
            out["__top_level__"] = top_keys
        except Exception:
            pass
    except Exception as e:
        _warn(f"Failed to parse cells.zarr.zip: {e}")
    return out


# ------------------------------
# Public loader
# ------------------------------

def load_anndata_from_partial(
    base_url: Optional[str] = None,
    analysis_name: Optional[str] = None,
    cells_name: Optional[str] = None,
    transcripts_name: Optional[str] = None,
    # MEX triplet (either a local dir or base_url + default folder "cell_feature_matrix/")
    mex_dir: Optional[str] = None,
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    build_counts_if_missing: bool = True,
) -> ad.AnnData:
    """
    Load a partial Xenium dataset into AnnData by opportunistically combining:
    - analysis.zarr.zip (optional)
    - cells.zarr.zip (optional)
    - transcripts.zarr.zip (optional, not yet assembled into counts here)
    - a 10x MEX triplet to build the count matrix (recommended)

    Parameters
    ----------
    base_url : str, optional
        Remote base where artifacts live (e.g. a Hugging Face dataset ".../resolve/main").
        If provided and `mex_dir` is None, we'll assume MEX lives under
        f"{base_url}/cell_feature_matrix/".
    analysis_name, cells_name, transcripts_name : str, optional
        Filenames of the zipped Zarr bundles. If relative and `base_url` is set,
        they will be resolved against `base_url`.
    mex_dir : str, optional
        Directory containing the MEX triplet. If omitted but `base_url` is set,
        defaults to f"{base_url}/cell_feature_matrix".
    build_counts_if_missing : bool
        If True and MEX is not available, we return an empty (0,0) matrix AnnData
        with whatever metadata we could parse from Zarr files.

    Returns
    -------
    AnnData
        with `.X` as CSR counts (if MEX present) and `.layers['counts']` mirrored,
        `.var` with feature metadata, `.obs_names` with cell barcodes, and
        `.uns` containing parsed analysis/cells summaries when available.
    """
    # ---------------- Resolve inputs ----------------
    local_paths: Dict[str, Optional[str]] = {"analysis": None, "cells": None, "transcripts": None}

    def _resolve(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        path = name
        if base_url and _is_http(base_url) and not _is_http(name) and not os.path.isabs(name):
            path = f"{base_url.rstrip('/')}/{name.lstrip('/')}"
        return _ensure_local(path, suffix=".zip")

    local_paths["analysis"] = _resolve(analysis_name)
    local_paths["cells"] = _resolve(cells_name)
    local_paths["transcripts"] = _resolve(transcripts_name)

    # ---------------- Read Zarr bundles (best-effort) ----------------
    uns: Dict[str, Any] = {"io": {}}

    if local_paths["analysis"]:
        try:
            grp = _open_zarr_zip(local_paths["analysis"])  # raises if not a zarr zip
            uns["analysis"] = _parse_analysis_group(grp)
        except Exception as e:
            _warn(f"Failed to parse analysis cell_groups: \"{e}\"")

    if local_paths["cells"]:
        try:
            grp = _open_zarr_zip(local_paths["cells"])  # raises if not a zarr zip
            uns["cells"] = _parse_cells_group(grp)
        except Exception as e:
            _warn(f"Failed to parse cells.zarr.zip: \"{e}\"")

    # ---------------- Read MEX counts if available ----------------
    X = None
    var_df = None
    obs_df = None

    # Decide mex_dir (may be remote or local)
    eff_mex_dir = mex_dir
    if eff_mex_dir is None and base_url:
        eff_mex_dir = f"{base_url.rstrip('/')}/cell_feature_matrix"

    if eff_mex_dir:
        if _is_http(eff_mex_dir):
            m = _ensure_local(f"{eff_mex_dir.rstrip('/')}/{mex_matrix_name}", subdir="mex")
            fts = _ensure_local(f"{eff_mex_dir.rstrip('/')}/{mex_features_name}", subdir="mex")
            bcs = _ensure_local(f"{eff_mex_dir.rstrip('/')}/{mex_barcodes_name}", subdir="mex")
        else:
            m = os.path.join(eff_mex_dir, mex_matrix_name)
            fts = os.path.join(eff_mex_dir, mex_features_name)
            bcs = os.path.join(eff_mex_dir, mex_barcodes_name)
        try:
            X, features, barcodes = _read_mex_triplet(m, fts, bcs)
            # Build var / obs
            var_df = pd.DataFrame(index=features["feature_id"].astype(str).values)
            # Keep some common fields
            for key in ("feature_id", "feature_name", "feature_type"):
                if key in features.columns:
                    var_df[key] = features[key].values
            var_df["feature_types"] = var_df.get("feature_type", pd.Series(["Gene"] * len(var_df), index=var_df.index))

            obs_df = pd.DataFrame(index=barcodes["barcode"].astype(str).values)
        except Exception as e:
            _warn(f"Failed to read MEX triplet: {e}")

    if X is None:
        if build_counts_if_missing:
            _warn("Counts matrix not built (MEX missing). Returning empty AnnData.")
            X = sparse.csr_matrix((0, 0), dtype=np.float32)
            var_df = pd.DataFrame(index=pd.Index([], name=None))
            obs_df = pd.DataFrame(index=pd.Index([], name=None))
        else:
            raise FileNotFoundError("MEX not found and build_counts_if_missing=False")

    # ---------------- Assemble AnnData ----------------
    adata = ad.AnnData(X, obs=obs_df, var=var_df)
    # mirror counts
    adata.layers["counts"] = adata.X.copy()

    # simple QC fields (optional)
    if adata.n_vars > 0 and adata.n_obs > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata.var["total_counts"] = np.asarray(adata.X.sum(axis=0)).ravel()

    # attach I/O trace
    io_meta = {
        "base_url": base_url,
        "analysis_name": analysis_name,
        "cells_name": cells_name,
        "transcripts_name": transcripts_name,
        "mex_dir": eff_mex_dir,
        "mex_matrix_name": mex_matrix_name,
        "mex_features_name": mex_features_name,
        "mex_barcodes_name": mex_barcodes_name,
    }
    uns["io"].update(io_meta)
    adata.uns.update(uns)

    # ---------------- Enrich with spatial coords & cluster labels ----------------
    # Try extract XY + cell_id strings from cells.zarr.zip (if provided)
    cell_ids_str = None
    df_xy = None
    if local_paths.get("cells"):
        try:
            df_xy, cell_ids_str, cells_meta = _extract_cells_xy_and_ids(local_paths["cells"])
            # write obs['x','y'] aligned to adata.obs_names (barcodes)
            xy_aligned = df_xy.reindex(adata.obs_names)
            if "x" in xy_aligned and "y" in xy_aligned:
                adata.obs["x"] = xy_aligned["x"].to_numpy()
                adata.obs["y"] = xy_aligned["y"].to_numpy()
                # standard slot for 2D spatial coordinates (microns)
                adata.obsm["spatial"] = np.c_[adata.obs["x"].to_numpy(), adata.obs["y"].to_numpy()]
                adata.uns.setdefault("spatial", {})["units"] = cells_meta.get("spatial_units", "micron")
            uns.setdefault("cells_meta", {})["cell_summary_cols"] = cells_meta.get("cell_summary_cols")
        except Exception as e:
            _warn(f"Could not extract XY from cells.zarr.zip: {e}")

    # Try compute cluster labels from analysis.zarr.zip (if provided)
    if local_paths.get("analysis"):
        try:
            # Prefer n_cells from df_xy; otherwise fall back to number of obs
            n_cells_for_groups = (len(cell_ids_str) if cell_ids_str is not None else adata.n_obs)
            labels_by_index, groups_meta = _labels_from_analysis_cell_groups(
                local_paths["analysis"], n_cells=n_cells_for_groups, prefer_group="0"
            )
            # Map labels (index=0..N-1) -> barcode strings (via cells.zarr mapping)
            if cell_ids_str is not None:
                label_series = pd.Series(labels_by_index.values, index=pd.Index(cell_ids_str, name=None))
            else:
                # Without cells.zarr we can't robustly map index->barcode;
                # assume adata.obs_names already match 0..N-1 order (best-effort)
                label_series = pd.Series(labels_by_index.values, index=adata.obs_names)

            lab_aligned = label_series.reindex(adata.obs_names).fillna("Unassigned")
            adata.obs["cluster"] = pd.Categorical(lab_aligned.values)

            # keep a light meta trace under uns['analysis']
            adata.uns.setdefault("analysis", {}).update({
                "group_key_used": groups_meta.get("group_key", "0"),
                "grouping_names": groups_meta.get("grouping_names"),
                "n_clusters": groups_meta.get("n_clusters"),
            })
        except Exception as e:
            _warn(f"Could not compute cluster labels from analysis.zarr.zip: {e}")

    return adata
