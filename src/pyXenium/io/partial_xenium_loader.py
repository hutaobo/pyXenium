"""
pyXenium.io.partial_xenium_loader
---------------------------------

Create an AnnData object from partial Xenium artifacts.

What it can use:
- 10x Gene Expression MEX (matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz])
- `analysis.zarr` or `analysis.zarr.zip` (cluster labels / metadata)
- `cells.zarr` or `cells.zarr.zip` (centroids / spatial coords)

Updates (2025-09-24)
--------------------
- Default: load MEX from `<base>/cell_feature_matrix/` (URL or local dir).
- Do NOT build counts from transcripts.zarr anymore. If MEX missing, returns an AnnData with
  empty gene dimension (but still attaches clusters/spatial when available).
- Robust `cell_id` detection: include root-level "cell_id" and support `(N,2)` numeric ids -> "cell_{num}".
- Fix: when downloading MEX from URLs, download all three files into the SAME temporary directory
  to avoid "MEX missing files" errors.
- Keep all previous public APIs and behaviors (backward compatible).

Author: Taobo Hu (pyXenium project)
"""
from __future__ import annotations

import gzip
import io
import logging
import os
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread

try:
    import zarr
    from zarr.storage import ZipStore
except Exception:
    zarr = None  # type: ignore
    ZipStore = None  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------
# Helpers
# -----------------------------

def _is_url(p: Optional[str | os.PathLike]) -> bool:
    if p is None:
        return False
    s = str(p)
    return s.startswith("http://") or s.startswith("https://")


def _fetch_to_temp(src: str, suffix: Optional[str] = None) -> Path:
    """Download a single URL to a temporary file and return its Path."""
    logger.info(f"Downloading: {src}")
    r = requests.get(src, stream=True, timeout=120)
    r.raise_for_status()
    tmpdir = Path(tempfile.mkdtemp(prefix="pyxenium_"))
    name = Path(urllib.parse.urlparse(src).path).name or ("tmp" + (suffix or ""))
    dst = tmpdir / name
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dst


def _download_many_to_same_temp(urls: Sequence[str]) -> Path:
    """Download multiple URLs into the SAME temporary directory.
    Returns the temp directory Path containing all files (base names preserved).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="pyxenium_"))
    for u in urls:
        logger.info(f"Downloading: {u}")
        r = requests.get(u, stream=True, timeout=120)
        r.raise_for_status()
        name = Path(urllib.parse.urlparse(u).path).name or "tmp"
        dst = tmpdir / name
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmpdir


def _p(x: Optional[os.PathLike | str]) -> Optional[Path]:
    return None if x is None else Path(x).expanduser().resolve()


def _open_text_maybe_gz(p: Path) -> io.TextIOBase:
    if str(p).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(p, mode="rb"))
    return open(p, "rt", encoding="utf-8")


@dataclass
class PartialInputs:
    mex_dir: Optional[Path | str] = None
    analysis_zarr: Optional[Path | str] = None
    cells_zarr: Optional[Path | str] = None


# -----------------------------
# MEX loader
# -----------------------------

def _load_mex_triplet(mex_dir: Path, matrix_name: str, features_name: str, barcodes_name: str) -> AnnData:
    logger.info(f"Reading MEX from {mex_dir}")
    candidates = {
        "matrix": [matrix_name, "matrix.mtx", "matrix.mtx.gz"],
        "features": [features_name, "features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"],
        "barcodes": [barcodes_name, "barcodes.tsv", "barcodes.tsv.gz"],
    }

    def _find(names: Sequence[str], base: Path) -> Optional[Path]:
        # check base first
        for name in names:
            p = base / name
            if p.exists():
                return p
        # then scan one-level subfolders (some pipelines write into subdir)
        for child in base.iterdir():
            if child.is_dir():
                for name in names:
                    p = child / name
                    if p.exists():
                        return p
        return None

    mtx_p = _find(candidates["matrix"], mex_dir)
    feat_p = _find(candidates["features"], mex_dir)
    barc_p = _find(candidates["barcodes"], mex_dir)

    if not (mtx_p and feat_p and barc_p):
        missing = [k for k, v in {"matrix": mtx_p, "features": feat_p, "barcodes": barc_p}.items() if v is None]
        raise FileNotFoundError(f"MEX missing files: {', '.join(missing)} under {mex_dir}")

    logger.info(f"  using {mtx_p.name}, {feat_p.name}, {barc_p.name}")
    with (gzip.open(mtx_p, "rb") if str(mtx_p).endswith(".gz") else open(mtx_p, "rb")) as f:
        X = mmread(f).tocsr().astype(np.float32)

    # features
    with _open_text_maybe_gz(feat_p) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    feat_df = pd.DataFrame(rows)
    if feat_df.shape[1] == 1:
        feat_df[[1]] = ""
        feat_df[[2]] = ""
    feat_df.columns = ["feature_id", "gene_name", "feature_type"]
    feat_df.index = feat_df["feature_id"].astype(str)

    # barcodes
    with _open_text_maybe_gz(barc_p) as f:
        barcodes = [line.strip() for line in f]
    obs = pd.DataFrame(index=pd.Index(barcodes, name="cell_id"))

    # orient
    if X.shape[0] == feat_df.shape[0] and X.shape[1] == obs.shape[0]:
        var = feat_df
    elif X.shape[1] == feat_df.shape[0] and X.shape[0] == obs.shape[0]:
        logger.warning("MEX matrix appears transposed; fixing orientation.")
        X = X.T.tocsr()
        var = feat_df
    else:
        raise ValueError(
            f"MEX shapes mismatch: matrix {X.shape}, features {feat_df.shape[0]}, barcodes {obs.shape[0]}"
        )

    # After reading X, var, obs
    n_feat, n_bc = var.shape[0], obs.shape[0]
    r, c = X.shape

    if (r, c) == (n_feat, n_bc):
        X = X.T.tocsr()
    elif (r, c) == (n_bc, n_feat):
        pass
    else:
        raise ValueError(
            f"MEX shapes mismatch: matrix {X.shape}, features {n_feat}, barcodes {n_bc}"
        )

    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    adata.uns.setdefault("io", {})["mex_dir"] = str(mex_dir)
    return adata


# -----------------------------
# Zarr helpers
# -----------------------------

def _open_zarr(path: Path | str):
    if zarr is None:  # pragma: no cover
        raise ImportError("zarr is required to read *.zarr or *.zarr.zip")

    pstr = str(path)
    if _is_url(pstr):
        path = _fetch_to_temp(pstr)
        pstr = str(path)

    # *.zarr.zip
    if pstr.endswith(".zip"):
        store = ZipStore(pstr, mode="r")
        if hasattr(zarr, "open_group"):
            return zarr.open_group(store=store, mode="r")
        if hasattr(zarr, "open"):
            return zarr.open(store, mode="r")
        return zarr.group(store=store, mode="r")

    # directory *.zarr
    if hasattr(zarr, "open_group"):
        return zarr.open_group(pstr, mode="r")
    if hasattr(zarr, "open"):
        return zarr.open(pstr, mode="r")
    return zarr.group(store=pstr, mode="r")


def _first_available(root, candidates: Sequence[str]) -> Optional[np.ndarray]:
    for key in candidates:
        try:
            arr = root[key]
        except Exception:
            continue
        try:
            return np.asarray(arr)
        except Exception:
            continue
    return None


def _normalize_cell_ids(arr: np.ndarray) -> Tuple[pd.Index, Optional[np.ndarray]]:
    """
    Normalize a cell_id array to string index.
    Returns:
      - obs index (strings)
      - numeric_ids (np.ndarray) if from numeric ids, else None
    """
    if arr.dtype.kind in ("i", "u"):
        if arr.ndim == 2:
            nums = arr[:, 0]
        else:
            nums = arr
        obs_names = pd.Index([f"cell_{int(x)}" for x in nums], name="cell_id")
        return obs_names, nums.astype(np.int64)
    else:
        s = pd.Series(arr)
        obs_names = pd.Index(s.astype(str), name="cell_id")
        return obs_names, None


# -----------------------------
# Attach spatial from cells.zarr
# -----------------------------

def _attach_spatial(adata: AnnData, cells_zarr: Path | str) -> None:
    logger.info(f"Attaching spatial from {cells_zarr}")
    root = _open_zarr(cells_zarr)

    x = _first_available(
        root,
        [
            "cells/centroids/x",
            "cell/centroids/x",
            "centroids/x",
            "cells/centroid_x",
            "cells/centre_x",
            "centroid/x",
            "x",
        ],
    )
    y = _first_available(
        root,
        [
            "cells/centroids/y",
            "cell/centroids/y",
            "centroids/y",
            "cells/centroid_y",
            "cells/centre_y",
            "centroid/y",
            "y",
        ],
    )
    z = _first_available(root, ["cells/centroids/z", "centroids/z", "cells/centroid_z", "z"])  # optional

    # include root-level "cell_id"
    cell_ids_raw = _first_available(root, ["cell_id", "cells/cell_id", "cells/ids", "cell_ids", "ids", "barcodes"])

    if x is None or y is None:
        logger.warning("Could not locate centroid x/y in cells.zarr; skipping spatial attach.")
        return

    coords = np.column_stack([x, y]) if z is None else np.column_stack([x, y, z])

    if cell_ids_raw is not None:
        idx, _ = _normalize_cell_ids(cell_ids_raw)
        df = pd.DataFrame(coords, index=idx)
        try:
            coords_df = df.reindex(adata.obs.index)
        except Exception:
            logger.warning("cells.zarr cell_ids cannot be aligned to adata.obs; using positional order where possible.")
            coords_df = pd.DataFrame(coords, index=adata.obs.index[: coords.shape[0]])
    else:
        if coords.shape[0] != adata.n_obs:
            logger.warning(
                f"cells.zarr coords length {coords.shape[0]} != adata.n_obs {adata.n_obs}; attaching partial by position."
            )
        coords_df = pd.DataFrame(coords, index=adata.obs.index[: coords.shape[0]])

    key = "spatial" if coords_df.shape[1] == 2 else "spatial3d"
    adata.obsm[key] = coords_df.to_numpy()
    adata.uns.setdefault("spatial", {})["source"] = str(cells_zarr)


# -----------------------------
# Attach cluster labels from analysis.zarr
# -----------------------------

def _attach_clusters(adata: AnnData, analysis_zarr: Path | str, cluster_key: str = "Cluster") -> None:
    logger.info(f"Attaching clusters from {analysis_zarr}")
    root = _open_zarr(analysis_zarr)

    label_arr = _first_available(
        root,
        [
            "clusters/labels",
            "clustering/labels",
            "labels",
            "cell_labels",
            "cells/labels",
            "annotations/cluster",
        ],
    )
    names = _first_available(root, ["clusters/names", "clustering/names", "names", "cluster_names"])
    ids = _first_available(root, ["clusters/ids", "clustering/ids", "ids", "cluster_ids"])

    cell_ids_raw = _first_available(root, ["cell_id", "cells/cell_id", "cell_ids", "barcodes", "cells/ids"])

    if label_arr is None:
        logger.warning("No cluster labels found in analysis.zarr; skipping.")
        return

    labels = label_arr.astype(str)
    if ids is not None and names is not None and len(ids) == len(names):
        mapping = {str(i): str(n) for i, n in zip(ids, names)}
        labels = np.array([mapping.get(str(x), str(x)) for x in label_arr], dtype=object)

    if cell_ids_raw is not None and len(cell_ids_raw) == len(labels):
        idx, _ = _normalize_cell_ids(cell_ids_raw)
        s = pd.Series(labels, index=idx)
        try:
            adata.obs[cluster_key] = s.reindex(adata.obs.index).astype("category")
        except Exception:
            logger.warning("Failed to align clusters by cell_id; using positional attach.")
            adata.obs[cluster_key] = pd.Categorical(labels[: adata.n_obs])
    else:
        if len(labels) != adata.n_obs:
            logger.warning(
                f"Cluster label length {len(labels)} != adata.n_obs {adata.n_obs}; attaching partial by position."
            )
        adata.obs[cluster_key] = pd.Categorical(labels[: adata.n_obs])

    adata.uns.setdefault("clusters", {})["source"] = str(analysis_zarr)


# -----------------------------
# Public API
# -----------------------------

def load_anndata_from_partial(
    mex_dir: Optional[os.PathLike | str] = None,
    analysis_zarr: Optional[os.PathLike | str] = None,
    cells_zarr: Optional[os.PathLike | str] = None,
    *,
    base_dir: Optional[os.PathLike | str] = None,
    base_url: Optional[str] = None,
    analysis_name: str = "analysis.zarr",
    cells_name: str = "cells.zarr",
    # default MEX file names
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    # where to look (by default) for MEX under base
    mex_default_subdir: str = "cell_feature_matrix",
    cluster_key: str = "Cluster",
    sample: Optional[str] = None,
    keep_unassigned: bool = True,
    build_counts_if_missing: bool = True,  # kept for API compatibility; now only controls empty-matrix behavior
) -> AnnData:
    """Create an AnnData from partial Xenium artifacts.

    Default behavior:
      - If `mex_dir` not given, try `<base>/cell_feature_matrix/` for MEX triplet.
      - If MEX found: load counts from MEX.
      - If MEX not found: DO NOT compute from transcripts; return empty-gene AnnData,
        but still try to attach clusters/spatial from analysis.zarr/cells.zarr.
    """
    # Resolve MEX directory
    mex_dir_p: Optional[Path] = None

    # 1) If explicit mex_dir is provided
    if mex_dir is not None:
        if _is_url(mex_dir):
            # download three files into the SAME temp dir
            base_url_mex = str(mex_dir).rstrip("/")
            tmpdir = _download_many_to_same_temp([
                base_url_mex + "/" + mex_matrix_name,
                base_url_mex + "/" + mex_features_name,
                base_url_mex + "/" + mex_barcodes_name,
            ])
            mex_dir_p = tmpdir
        else:
            mex_dir_p = _p(mex_dir)

    # 2) Otherwise, try `<base>/cell_feature_matrix/`
    if mex_dir_p is None and (base_dir is not None or base_url is not None):
        if base_dir is not None:
            bd = _p(base_dir)
            cand = bd / mex_default_subdir if bd is not None else None
            if cand is not None and cand.exists():
                mex_dir_p = cand

        if mex_dir_p is None and base_url is not None:
            # download three files from base_url/cell_feature_matrix/ into SAME temp dir
            root_url = base_url.rstrip("/") + "/" + mex_default_subdir
            try:
                tmpdir = _download_many_to_same_temp([
                    root_url + "/" + mex_matrix_name,
                    root_url + "/" + mex_features_name,
                    root_url + "/" + mex_barcodes_name,
                ])
                mex_dir_p = tmpdir
            except Exception as e:
                logger.warning(f"Failed to fetch MEX from {root_url}: {e}")

    # Resolve Zarrs (optional)
    def _resolve_zarr(explicit: Optional[os.PathLike | str], name: str) -> Optional[Path | str]:
        if explicit is not None:
            return explicit
        if base_dir is None and base_url is None:
            return None
        if base_dir is not None:
            base = _p(base_dir)
            assert base is not None
            cand = base / name
            if cand.exists():
                return cand
            alt = None
            if name.endswith(".zarr"):
                alt = base / f"{name}.zip"
            elif name.endswith(".zip"):
                alt = base / name[:-4]
            if alt is not None and alt.exists():
                return alt
        if base_url is not None:
            return base_url.rstrip("/") + "/" + name
        return None

    analysis_p = _resolve_zarr(analysis_zarr, analysis_name)
    cells_p = _resolve_zarr(cells_zarr, cells_name)

    # Construct AnnData
    if mex_dir_p is not None:
        adata = _load_mex_triplet(mex_dir_p, mex_matrix_name, mex_features_name, mex_barcodes_name)
    else:
        # No MEX: create empty-gene AnnData but still attempt to obtain obs index from zarrs
        logger.warning(
            "No MEX found under mex_dir or <base>/cell_feature_matrix/. "
            "Returning empty-gene AnnData; counts are unavailable without MEX."
        )
        # try to discover cell ids from any provided zarr to build obs
        cell_ids: Optional[pd.Index] = None
        probe_keys = ["cell_id", "cells/cell_id", "cell_ids", "barcodes", "cells/ids", "cell"]
        for p in (cells_p, analysis_p):
            if p is None:
                continue
            try:
                root = _open_zarr(p)
            except Exception:
                continue
            arr = _first_available(root, probe_keys)
            if arr is not None:
                idx, _ = _normalize_cell_ids(arr)
                cell_ids = idx
                break

        if cell_ids is None:
            # fall back to minimal AnnData
            adata = AnnData(X=sparse.csr_matrix((0, 0)))
        else:
            obs = pd.DataFrame(index=cell_ids)
            var = pd.DataFrame(index=pd.Index([], name="feature_id"))
            adata = AnnData(X=sparse.csr_matrix((len(cell_ids), 0)), obs=obs, var=var)

    # Attach extras
    if analysis_p is not None:
        try:
            _attach_clusters(adata, analysis_p, cluster_key=cluster_key)
            if not keep_unassigned and cluster_key in adata.obs:
                bad = {"-1", "NA", "None", "Unassigned", "unassigned"}
                mask = ~adata.obs[cluster_key].astype(str).isin(bad)
                dropped = int((~mask).sum())
                if dropped:
                    logger.info(f"Dropping {dropped} unassigned cells")
                adata._inplace_subset_obs(mask.values)
        except Exception as e:
            logger.warning(f"Failed attaching clusters: {e}")

    if cells_p is not None:
        try:
            _attach_spatial(adata, cells_p)
        except Exception as e:
            logger.warning(f"Failed attaching spatial coordinates: {e}")

    if sample is not None:
        adata.uns["sample"] = str(sample)

    # provenance
    adata.uns.setdefault("io", {})
    adata.uns["io"]["analysis_zarr"] = str(analysis_p) if analysis_p else None
    adata.uns["io"]["cells_zarr"] = str(cells_p) if cells_p else None
    adata.uns["io"]["mex_dir"] = str(mex_dir_p) if mex_dir_p else None

    return adata


__all__ = ["load_anndata_from_partial", "PartialInputs"]
