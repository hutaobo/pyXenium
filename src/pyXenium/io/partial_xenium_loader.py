"""
partial_xenium_loader.py

A lightweight, robust loader for assembling an AnnData object from partial
10x/Xenium outputs, with automatic MEX orientation correction.

Drop-in replacement for pyXenium.io.partial_xenium_loader.

Requirements:
  - anndata
  - scipy
  - pandas
Optional:
  - requests (for HTTP downloads)
  - zarr (to enrich obs from cells/analysis zarr bundles, best-effort)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import logging
import pathlib
import typing as _t
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.io import mmread
from scipy import sparse

try:
    from anndata import AnnData  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("AnnData is required. Please install `anndata`.") from e

# Optional: zarr support (best-effort)
try:
    import zarr  # type: ignore
except Exception:
    zarr = None  # type: ignore

# Optional: requests for HTTP downloads
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


logger = logging.getLogger("pyXenium.partial_xenium_loader")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


@dataclass
class MEXPaths:
    matrix: pathlib.Path
    features: pathlib.Path
    barcodes: pathlib.Path


def _ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    if requests is None:
        raise RuntimeError("`requests` is required to download from URLs.")
    _ensure_dir(dest.parent)
    logger.info(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return dest


def _resolve_from_base(base: str | None, *parts: str) -> str:
    if base is None:
        raise ValueError("Base is None when resolving a path/URL.")
    if _is_url(base):
        return "/".join([base.rstrip("/")] + [p.strip("/") for p in parts])
    else:
        return str(pathlib.Path(base, *parts))


def _find_mex_triplet(
    mex_dir: str | None = None,
    base_dir: str | None = None,
    base_url: str | None = None,
    mex_default_subdir: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    tmp_root: pathlib.Path | None = None,
) -> MEXPaths:
    """
    Locate (and if necessary, download) the three MEX files and return local paths.

    Priority order:
      1) explicit mex_dir
      2) base_dir/mex_default_subdir
      3) base_url/mex_default_subdir  (download to temp dir)
    """
    if mex_dir is not None:
        mexp = pathlib.Path(mex_dir)
        return MEXPaths(
            mexp / mex_matrix_name, mexp / mex_features_name, mexp / mex_barcodes_name
        )

    # Try local base_dir first
    if base_dir is not None:
        mexp = pathlib.Path(base_dir) / mex_default_subdir
        m = mexp / mex_matrix_name
        f = mexp / mex_features_name
        b = mexp / mex_barcodes_name
        if m.exists() and f.exists() and b.exists():
            return MEXPaths(m, f, b)

    # Try remote base_url
    if base_url is not None:
        # Download to temp dir
        root = tmp_root or pathlib.Path(tempfile.mkdtemp(prefix="pyxenium_"))
        dl_dir = root / "mex"
        _ensure_dir(dl_dir)

        m_url = _resolve_from_base(base_url, mex_default_subdir, mex_matrix_name)
        f_url = _resolve_from_base(base_url, mex_default_subdir, mex_features_name)
        b_url = _resolve_from_base(base_url, mex_default_subdir, mex_barcodes_name)

        m = _download(m_url, dl_dir / mex_matrix_name)
        f = _download(f_url, dl_dir / mex_features_name)
        b = _download(b_url, dl_dir / mex_barcodes_name)

        return MEXPaths(m, f, b)

    raise FileNotFoundError(
        "Could not locate MEX files. Provide `mex_dir`, or `base_dir`/`base_url` "
        f"with default subdir '{mex_default_subdir}' containing the triplet."
    )


def _load_mex_triplet(
    mex_dir: str | pathlib.Path,
    matrix_name: str = "matrix.mtx.gz",
    features_name: str = "features.tsv.gz",
    barcodes_name: str = "barcodes.tsv.gz",
) -> AnnData:
    """
    Read 10x-style MEX triplet and construct an AnnData with X=counts (cells × genes),
    obs=cells (barcodes), var=genes (features). Auto-corrects orientation if needed.
    """
    mex_dir = pathlib.Path(mex_dir)
    matrix_path = mex_dir / matrix_name
    features_path = mex_dir / features_name
    barcodes_path = mex_dir / barcodes_name

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not barcodes_path.exists():
        raise FileNotFoundError(barcodes_path)

    logger.info(f"Reading MEX from {mex_dir}")
    logger.info(f"  using {matrix_name}, {features_name}, {barcodes_name}")

    # Read matrix (COO)
    X = mmread(str(matrix_path))  # COO or coo_array
    if not sparse.isspmatrix(X):
        X = sparse.coo_matrix(X)

    # Read features & barcodes
    feat_df = pd.read_csv(features_path, sep="\\t", header=None)
    bc_df = pd.read_csv(barcodes_path, sep="\\t", header=None)

    # Build var/obs
    # Prefer feature name (2nd column) as index if present
    if feat_df.shape[1] >= 2:
        var_index = feat_df.iloc[:, 1].astype(str).values
    else:
        var_index = feat_df.iloc[:, 0].astype(str).values
    var = pd.DataFrame(index=pd.Index(var_index, name="feature_name"))
    var["feature_id"] = feat_df.iloc[:, 0].astype(str).values
    if feat_df.shape[1] >= 3:
        var["feature_type"] = feat_df.iloc[:, 2].astype(str).values

    obs = pd.DataFrame(index=pd.Index(bc_df.iloc[:, 0].astype(str).values, name="barcode"))

    # Auto-fix orientation (10x MEX is typically features × barcodes)
    n0, n1 = X.shape
    n_feat, n_bar = var.shape[0], obs.shape[0]

    if n0 == n_feat and n1 == n_bar:
        # features × barcodes -> transpose to cells × genes
        X = X.T.tocsr()
    elif n0 == n_bar and n1 == n_feat:
        # already cells × genes
        X = X.tocsr() if not sparse.isspmatrix_csr(X) else X
    else:
        raise ValueError(
            f"MEX shapes mismatch: matrix {X.shape}, features {n_feat}, barcodes {n_bar}. "
            "Check the three files correspond to the same dataset."
        )

    adata = AnnData(X=X, obs=obs, var=var)
    # Keep raw counts in a layer
    adata.layers["counts"] = adata.X.copy()
    adata.uns.setdefault("io", {})["mex_dir"] = str(mex_dir)
    adata.uns["io"]["mex_files"] = {
        "matrix": str(matrix_path.name),
        "features": str(features_path.name),
        "barcodes": str(barcodes_path.name),
    }
    return adata


def _try_load_cells_metadata_from_zarr(
    adata: AnnData,
    cells_zarr: str | None = None,
    base_dir: str | None = None,
    base_url: str | None = None,
    cells_name: str = "cells.zarr.zip",
) -> None:
    """
    Best-effort: if a cells.zarr(.zip) is available, try to enrich adata.obs with
    cell-level information (e.g., centroid coordinates) when keys are recognizable.
    This function won't raise if zarr is unavailable or structure is unknown.
    """
    if zarr is None:
        return

    # Resolve path or download
    path: str | None = None
    tmp_dir: str | None = None

    if cells_zarr is not None:
        path = cells_zarr
    elif base_dir is not None:
        maybe = os.path.join(base_dir, cells_name)
        if os.path.exists(maybe):
            path = maybe
    elif base_url is not None:
        if requests is None:
            return
        tmp_dir = tempfile.mkdtemp(prefix="pyxenium_cells_")
        url = _resolve_from_base(base_url, cells_name)
        logger.info(f"Downloading: {url}")
        local = os.path.join(tmp_dir, cells_name)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        path = local

    if path is None or not os.path.exists(path):
        return

    try:
        store = zarr.DirectoryStore(path) if os.path.isdir(path) else zarr.ZipStore(path, mode="r")
        root = zarr.open(store=store, mode="r")

        # Heuristic keys commonly seen in Xenium/SpaceRanger-like cells.zarr
        obs_keys = {}
        for k in ("cell_id", "barcode"):
            if k in root:
                obs_keys["cell_id"] = np.asarray(root[k])
                break
        if "cell_id" in obs_keys:
            cid = pd.Index(obs_keys["cell_id"].astype(str), name="cell_id")
            if adata.obs.index.equals(cid):
                pass
            else:
                if cid.is_unique and adata.obs.index.is_unique:
                    adata.obs["cell_id"] = cid
                else:
                    adata.uns.setdefault("cells_zarr", {})["cell_id_preview"] = cid[:10].tolist()

        # Common centroid arrays
        for xk, yk in (("cell_centroids/x", "cell_centroids/y"), ("centroids/x", "centroids/y")):
            if xk in root and yk in root:
                x = np.asarray(root[xk]).astype(float)
                y = np.asarray(root[yk]).astype(float)
                if len(x) == adata.n_obs and len(y) == adata.n_obs:
                    adata.obs["x"] = x
                    adata.obs["y"] = y
                break

        try:
            store.close()  # type: ignore[attr-defined]
        except Exception:
            pass

        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Failed to parse cells zarr: {e}")


def _try_load_analysis_from_zarr(
    adata: AnnData,
    analysis_zarr: str | None = None,
    base_dir: str | None = None,
    base_url: str | None = None,
    analysis_name: str = "analysis.zarr.zip",
    cluster_key: str = "Cluster",
    keep_unassigned: bool = True,
) -> None:
    """
    Best-effort: if an analysis.zarr(.zip) is available, try to add clustering labels
    into adata.obs[cluster_key]. This is schema-dependent; we try a few common conventions.
    """
    if zarr is None:
        return

    # Resolve path or download
    path: str | None = None
    tmp_dir: str | None = None

    if analysis_zarr is not None:
        path = analysis_zarr
    elif base_dir is not None:
        maybe = os.path.join(base_dir, analysis_name)
        if os.path.exists(maybe):
            path = maybe
    elif base_url is not None:
        if requests is None:
            return
        tmp_dir = tempfile.mkdtemp(prefix="pyxenium_analysis_")
        url = _resolve_from_base(base_url, analysis_name)
        logger.info(f"Downloading: {url}")
        local = os.path.join(tmp_dir, analysis_name)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        path = local

    if path is None or not os.path.exists(path):
        return

    try:
        store = zarr.DirectoryStore(path) if os.path.isdir(path) else zarr.ZipStore(path, mode="r")
        root = zarr.open(store=store, mode="r")

        candidate_keys = [
            "clusters/labels",
            "clusters/names",
            "cluster_labels",
            "leiden",
            "louvain",
        ]
        for k in candidate_keys:
            if k in root:
                arr = np.asarray(root[k]).astype(str)
                if len(arr) == adata.n_obs:
                    adata.obs[cluster_key] = pd.Categorical(arr)
                    if not keep_unassigned:
                        mask = adata.obs[cluster_key].astype(str).str.lower().isin(
                            ["unassigned", "unknown", "na", "nan", "none", ""]
                        )
                        adata._inplace_subset_obs(~mask.values)
                    break

        try:
            store.close()  # type: ignore[attr-defined]
        except Exception:
            pass

        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Failed to parse analysis zarr: {e}")


def load_anndata_from_partial(
    mex_dir: str | None = None,
    analysis_zarr: str | None = None,
    cells_zarr: str | None = None,
    *,
    base_dir: str | None = None,
    base_url: str | None = None,
    analysis_name: str = "analysis.zarr.zip",
    cells_name: str = "cells.zarr.zip",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    mex_default_subdir: str = "cell_feature_matrix",
    cluster_key: str = "Cluster",
    sample: str | None = None,
    keep_unassigned: bool = True,
    build_counts_if_missing: bool = False,  # kept for API compatibility; not used
) -> AnnData:
    """
    Load an AnnData object when you only have partial Xenium/10x outputs.
    At minimum, provide a MEX triplet (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz)
    either via `mex_dir`, or under `<base>/cell_feature_matrix/` from `base_dir` or `base_url`.

    Parameters
    ----------
    mex_dir
        Directory containing the three MEX files.
    analysis_zarr, cells_zarr
        Optional paths (or URLs when used with base_url) to zarr(.zip) bundles to
        augment obs (e.g., cluster labels, centroid coordinates). Best-effort only.
    base_dir, base_url
        A local base directory or a remote base URL that contains the default layout.
    analysis_name, cells_name
        Filenames for analysis and cells zarr bundles (default: analysis.zarr.zip, cells.zarr.zip).
    mex_* names and mex_default_subdir
        Control the filenames and subdirectory for the MEX triplet discovery.
    cluster_key
        Column name to store cluster labels loaded from analysis.zarr.
    sample
        Optional sample name recorded in `adata.uns["sample"]`.
    keep_unassigned
        Whether to keep rows labeled as unassigned/unknown when cluster labels are loaded.
    build_counts_if_missing
        Kept for backward compatibility; counts are always read from MEX here.

    Returns
    -------
    AnnData
        With `X` as counts (CSR), `layers['counts']` as a copy, `.obs` barcodes,
        `.var` features, and optional metadata from zarr bundles.
    """
    # 1) Locate MEX triplet (may download to tmp if base_url is used)
    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="pyxenium_"))
    try:
        paths = _find_mex_triplet(
            mex_dir=mex_dir,
            base_dir=base_dir,
            base_url=base_url,
            mex_default_subdir=mex_default_subdir,
            mex_matrix_name=mex_matrix_name,
            mex_features_name=mex_features_name,
            mex_barcodes_name=mex_barcodes_name,
            tmp_root=tmp_root,
        )

        # If we downloaded to tmp_root, normalize mex_dir there for _load_mex_triplet
        mex_local_dir = paths.matrix.parent

        adata = _load_mex_triplet(
            mex_local_dir,
            matrix_name=paths.matrix.name,
            features_name=paths.features.name,
            barcodes_name=paths.barcodes.name,
        )

        # 2) Best-effort enrichment from zarr bundles
        _try_load_cells_metadata_from_zarr(
            adata,
            cells_zarr=cells_zarr,
            base_dir=base_dir,
            base_url=base_url,
            cells_name=cells_name,
        )

        _try_load_analysis_from_zarr(
            adata,
            analysis_zarr=analysis_zarr,
            base_dir=base_dir,
            base_url=base_url,
            analysis_name=analysis_name,
            cluster_key=cluster_key,
            keep_unassigned=keep_unassigned,
        )

        # 3) Record provenance
        adata.uns.setdefault("io", {})
        if base_url:
            adata.uns["io"]["base_url"] = base_url
        if base_dir:
            adata.uns["io"]["base_dir"] = base_dir
        if sample:
            adata.uns["sample"] = sample

        return adata
    finally:
        # Clean up temp root if created
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
