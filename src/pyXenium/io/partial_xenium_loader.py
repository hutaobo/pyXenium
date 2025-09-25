"""
partial_xenium_loader.py — patched version

Key changes vs. prior revision
------------------------------
1) Robust MEX orientation handling:
   Automatically detects whether matrix.mtx is (features × barcodes) or
   (barcodes × features) and orients to (cells × genes) before
   constructing AnnData. This fixes the error where obs length equals the
   number of barcodes while X had rows == number of features.

2) Safer IO helpers:
   - _ensure_local(): Accepts local paths or HTTP(S) URLs and returns a
     local Path (downloads to a temporary folder when URL is used).
   - Explicit informative logging for each step.

3) Flexible entry point load_anndata_from_partial():
   - Accepts either explicit mex_dir or (base_dir/base_url + default
     subdir). If nothing is found, returns an empty-gene AnnData but will
     still try to fetch obs indices from optional zarrs.
   - Best-effort, optional enrichment from Xenium zarrs (analysis.zarr,
     cells.zarr) is wrapped in try/except so it never blocks core
     loading.

This module only depends on widely available packages: anndata, pandas,
scipy, numpy. The zarr-related enrichment is optional (guarded by
try/except ImportError).

Author: patched for Taobo Hu / pyXenium users
Date: 2025-09-25
"""
from __future__ import annotations

import io
import os
import sys
import gzip
import json
import math
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse as sp
from anndata import AnnData

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("pyXenium.partial_xenium_loader")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _ensure_local(path_or_url: Optional[str], *, tmpdir: Path, desc: str = "file") -> Optional[Path]:
    """Return a local Path for a path_or_url. If it's an HTTP(S) URL, download it.
    If it's None, return None.
    """
    if path_or_url is None:
        return None

    p = str(path_or_url)
    if p.startswith("http://") or p.startswith("https://"):
        name = Path(p).name
        out = tmpdir / name
        if not out.exists():
            import urllib.request
            logger.info(f"Downloading: {p}")
            with urllib.request.urlopen(p) as r, open(out, "wb") as f:
                shutil.copyfileobj(r, f)
        return out
    else:
        q = Path(p)
        if not q.exists():
            raise FileNotFoundError(f"{desc} not found: {q}")
        return q


def _maybe_join_url(base: Optional[str], *parts: str) -> Optional[str]:
    if base is None:
        return None
    base = base.rstrip("/")
    tail = "/".join(s.strip("/") for s in parts if s)
    return f"{base}/{tail}" if tail else base


# -----------------------------------------------------------------------------
# Core: load MEX triplet and build AnnData
# -----------------------------------------------------------------------------

def _load_mex_triplet(
    mex_dir: Path,
    matrix_name: str = "matrix.mtx.gz",
    features_name: str = "features.tsv.gz",
    barcodes_name: str = "barcodes.tsv.gz",
) -> AnnData:
    """Read a 10x MEX triplet from `mex_dir` and return AnnData with shape
    (cells × genes). Automatically fixes orientation.
    """
    # Read matrix
    X_path = mex_dir / matrix_name
    feats_path = mex_dir / features_name
    bcs_path = mex_dir / barcodes_name

    if not X_path.exists():
        raise FileNotFoundError(f"Missing matrix file: {X_path}")
    if not feats_path.exists():
        raise FileNotFoundError(f"Missing features file: {feats_path}")
    if not bcs_path.exists():
        raise FileNotFoundError(f"Missing barcodes file: {bcs_path}")

    logger.info("Reading MEX from %s", mex_dir)
    logger.info("  using %s, %s, %s", X_path.name, feats_path.name, bcs_path.name)

    X = spio.mmread(str(X_path))
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    # Read features (var)
    feat_df = pd.read_csv(feats_path, sep="\t", header=None, comment="#")
    # 10x convention: col0 = feature_id, col1 = feature_name, col2 = feature_type (optional)
    var_index = feat_df.iloc[:, 0].astype(str).values
    var = pd.DataFrame(index=var_index)
    if feat_df.shape[1] > 1:
        var["gene_name"] = feat_df.iloc[:, 1].astype(str).values
    if feat_df.shape[1] > 2:
        var["feature_type"] = feat_df.iloc[:, 2].astype(str).values

    # Read barcodes (obs)
    bc_df = pd.read_csv(bcs_path, sep="\t", header=None)
    obs_index = bc_df.iloc[:, 0].astype(str).values
    obs = pd.DataFrame(index=obs_index)

    # Orientation check
    n_feat, n_bc = var.shape[0], obs.shape[0]
    r, c = X.shape

    if (r, c) == (n_feat, n_bc):
        # Typical 10x: features × barcodes — transpose to cells × genes
        X = X.T.tocsr()
    elif (r, c) == (n_bc, n_feat):
        # Already cells × genes
        pass
    else:
        raise ValueError(
            f"MEX shapes mismatch: matrix {X.shape}, features {n_feat}, barcodes {n_bc}"
        )

    # Construct AnnData
    adata = AnnData(X=X, obs=obs, var=var)
    # Keep raw counts in a layer named 'counts'
    try:
        adata.layers["counts"] = adata.X.copy()
    except Exception:
        # If memory is tight in CSR, at least keep a reference
        adata.layers["counts"] = adata.X

    # Record basic IO metadata
    adata.uns.setdefault("io", {})["mex_dir"] = str(mex_dir)
    adata.uns["n_cells"] = adata.n_obs
    adata.uns["n_genes"] = adata.n_vars

    return adata


# -----------------------------------------------------------------------------
# Optional: light-touch enrichment from Xenium zarrs
# -----------------------------------------------------------------------------

def _try_enrich_from_zarr(
    adata: AnnData,
    analysis_zarr: Optional[Path] = None,
    cells_zarr: Optional[Path] = None,
    *,
    cluster_key: str = "Cluster",
    keep_unassigned: bool = True,
):
    """Best-effort enrichment using Xenium analysis/cells ZARRs.
    Adds cluster labels to .obs[cluster_key] and spatial centroids to .obsm['spatial']
    when present. All errors are logged as warnings and do not raise.
    """
    # Try zarr import lazily so this module works without zarr installed.
    try:
        import zarr  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.warning("zarr not available; skipping zarr enrichment (%s)", e)
        return

    # Helper to open zip zarr or folder zarr
    def _open_zgroup(p: Path):
        if p.suffix == ".zip":
            # ZipStore via URL-like spec
            store = zarr.storage.ZipStore(str(p), mode="r")
            return zarr.open_group(store, mode="r")
        return zarr.open_group(str(p), mode="r")

    # analysis.zarr: look for cluster labels mapping
    if analysis_zarr is not None and Path(analysis_zarr).exists():
        try:
            g = _open_zgroup(Path(analysis_zarr))
            # Heuristic locations — adjust as needed for your export schema
            # e.g., g['obs']['cluster'] or g['clusters']['labels']
            candidate_paths = [
                ("obs", "cluster"),
                ("clusters", "labels"),
                ("obs", cluster_key),
            ]
            labels = None
            for path in candidate_paths:
                try:
                    node = g
                    for key in path:
                        node = node[key]
                    labels = np.array(node[:])
                    break
                except Exception:
                    continue
            if labels is not None:
                if len(labels) == adata.n_obs:
                    adata.obs[cluster_key] = pd.Categorical(pd.Series(labels, index=adata.obs_names).astype(str))
                    if not keep_unassigned:
                        mask = adata.obs[cluster_key].astype(str).str.lower().isin({"unassigned", "none", "nan"})
                        adata._inplace_subset_obs(~mask)
                else:
                    logger.warning(
                        "Cluster labels length (%d) != n_obs (%d); skipping cluster assignment",
                        len(labels), adata.n_obs,
                    )
        except Exception as e:
            logger.warning("Failed to enrich clusters from analysis.zarr: %s", e)

    # cells.zarr: look for centroid spatial coordinates and optional cell_ids
    if cells_zarr is not None and Path(cells_zarr).exists():
        try:
            g = _open_zgroup(Path(cells_zarr))
            # Common locations for centroids; update for your schema if needed
            candidates = [
                ("cells", "centroids"),
                ("cells", "spatial"),
                ("spatial", "centroids"),
            ]
            coords = None
            for path in candidates:
                try:
                    node = g
                    for key in path:
                        node = node[key]
                    coords = np.asarray(node[:])
                    if coords.ndim == 2 and coords.shape[1] >= 2:
                        break
                    else:
                        coords = None
                except Exception:
                    continue
            if coords is not None:
                # Try align length to n_obs; otherwise, skip with a warning
                if coords.shape[0] == adata.n_obs:
                    # Use first two dims as (x, y)
                    adata.obsm["spatial"] = coords[:, :2]
                else:
                    logger.warning(
                        "Centroid rows (%d) != n_obs (%d); skipping spatial assignment",
                        coords.shape[0], adata.n_obs,
                    )
        except Exception as e:
            logger.warning("Failed to enrich spatial from cells.zarr: %s", e)


# -----------------------------------------------------------------------------
# Public API: load_anndata_from_partial
# -----------------------------------------------------------------------------

def load_anndata_from_partial(
    *,
    # MEX location (either provide mex_dir OR base_dir/base_url + default subdir)
    mex_dir: Optional[str | os.PathLike] = None,
    # Optional Xenium zarr artifacts
    analysis_zarr: Optional[str | os.PathLike] = None,
    cells_zarr: Optional[str | os.PathLike] = None,
    # Convenience: base directory/URL + file names
    base_dir: Optional[str | os.PathLike] = None,
    base_url: Optional[str] = None,
    analysis_name: Optional[str] = None,
    cells_name: Optional[str] = None,
    # MEX triplet names
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    # default subdir used under base_dir/base_url when mex_dir not provided
    mex_default_subdir: str = "cell_feature_matrix",
    # Enrichment options
    cluster_key: str = "Cluster",
    sample: Optional[str] = None,
    keep_unassigned: bool = True,
    # Behavior when no MEX is present
    build_counts_if_missing: bool = False,
) -> AnnData:
    """Load partial Xenium outputs into an AnnData.

    Parameters
    ----------
    mex_dir : path-like or None
        Directory containing the 10x MEX triplet. If None, the loader will
        look under `<base_dir or base_url>/<mex_default_subdir>/`.
    analysis_zarr, cells_zarr : path-like or None
        Optional Xenium ZARR artifacts to enrich clusters and spatial.
        Can be local paths OR HTTP(S) URLs to .zarr.zip.
    base_dir, base_url : str or None
        If provided, used as the base to resolve `analysis_name`, `cells_name`,
        and the MEX default subdir.
    analysis_name, cells_name : str or None
        Filenames of the respective artifacts (e.g., "analysis.zarr.zip").
    mex_*_name : str
        Filenames of the MEX triplet.
    mex_default_subdir : str
        Subdirectory under base_dir/base_url where the MEX triplet lives.
    cluster_key : str
        Column name in .obs to store cluster labels if found in analysis.zarr.
    sample : str or None
        Optional sample name to record in .uns["sample"].
    keep_unassigned : bool
        Whether to keep cells labeled as "unassigned" when cluster labels exist.
    build_counts_if_missing : bool
        If True and no MEX is found, create an AnnData with zero genes (no counts)
        but still attempt to populate obs index and spatial from zarrs.

    Returns
    -------
    AnnData
    """
    with tempfile.TemporaryDirectory(prefix="pyxenium_") as td:
        tmpdir = Path(td)

        # Resolve MEX directory
        if mex_dir is not None:
            mex_dir_p = Path(mex_dir)
            if not mex_dir_p.exists():
                raise FileNotFoundError(f"mex_dir not found: {mex_dir_p}")
        else:
            # Try base_dir/base_url + default subdir
            mex_dir_p = None
            # (A) URL case
            mex_url = _maybe_join_url(base_url, mex_default_subdir)
            if mex_url is not None:
                # Download triplet into a temp folder that mimics a directory
                mex_tmp = tmpdir / "mex"
                mex_tmp.mkdir(parents=True, exist_ok=True)
                for name in (mex_matrix_name, mex_features_name, mex_barcodes_name):
                    u = _maybe_join_url(mex_url, name)
                    assert u is not None
                    _ = _ensure_local(u, tmpdir=mex_tmp, desc=name)
                mex_dir_p = mex_tmp
            # (B) Local base_dir case
            if mex_dir_p is None and base_dir is not None:
                candidate = Path(base_dir) / mex_default_subdir
                if candidate.exists():
                    mex_dir_p = candidate

        # Resolve analysis/cells zarr paths (localize URLs if needed)
        analysis_p = None
        cells_p = None

        if analysis_zarr is not None:
            analysis_p = _ensure_local(str(analysis_zarr), tmpdir=tmpdir, desc="analysis_zarr")
        elif analysis_name is not None and (base_dir or base_url):
            url = _maybe_join_url(base_url, analysis_name)
            pth = os.path.join(str(base_dir), analysis_name) if base_dir else None
            analysis_p = _ensure_local(url or pth, tmpdir=tmpdir, desc="analysis_zarr")

        if cells_zarr is not None:
            cells_p = _ensure_local(str(cells_zarr), tmpdir=tmpdir, desc="cells_zarr")
        elif cells_name is not None and (base_dir or base_url):
            url = _maybe_join_url(base_url, cells_name)
            pth = os.path.join(str(base_dir), cells_name) if base_dir else None
            cells_p = _ensure_local(url or pth, tmpdir=tmpdir, desc="cells_zarr")

        # Construct AnnData from MEX if available
        if mex_dir_p is not None:
            adata = _load_mex_triplet(mex_dir_p, mex_matrix_name, mex_features_name, mex_barcodes_name)
        else:
            if not build_counts_if_missing:
                raise FileNotFoundError(
                    "No MEX found (mex_dir nor <base>/" + mex_default_subdir + ") and build_counts_if_missing=False"
                )
            # Build an empty-gene AnnData; we'll try to get obs index from zarrs
            logger.warning(
                "No MEX found under mex_dir or <base>/%s/. Returning empty-gene AnnData; counts unavailable.",
                mex_default_subdir,
            )
            adata = AnnData(X=sp.csr_matrix((0, 0)))

        # Optional enrichment from zarrs (best-effort)
        _try_enrich_from_zarr(
            adata,
            analysis_zarr=analysis_p,
            cells_zarr=cells_p,
            cluster_key=cluster_key,
            keep_unassigned=keep_unassigned,
        )

        # Tag sample, if provided
        if sample is not None:
            adata.uns.setdefault("sample", sample)

        return adata
