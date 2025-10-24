# -*- coding: utf-8 -*-
"""
xenium_gene_protein_loader.py

Utilities to read 10x Xenium In Situ Gene + Protein outputs (XOA >= v4.0) into
an AnnData object with sensible defaults:

- Load cell_feature_matrix from any of the official formats:
  * Zarr  (cell_feature_matrix.zarr/ or cell_feature_matrix/)
  * HDF5  (cell_feature_matrix.h5)
  * MEX   (matrix.mtx.gz + features.tsv.gz + barcodes.tsv.gz)
- Split features by type:
  * RNA ("Gene Expression") -> adata.X (CSR counts) and adata.layers["rna"]
  * Protein ("Protein Expression") -> adata.obsm["protein"] (float intensities)
- Attach cell table (cells.csv.gz or Parquet) to adata.obs
- Optionally attach boundaries (cell/nucleus) to adata.uns and centroids to adata.obsm["spatial"]
- NEW: Merge clustering results (clusters.csv) into adata.obs['cluster'] (configurable)

This module is dependency-light and relies only on numpy/pandas/scipy/anndata
(and optional zarr/h5py if those formats are used). It supports both local and
remote paths transparently via fsspec.

Notes:
- Protein values in 10x Xenium outputs are typically "scaled mean intensity".
- The code is defensive against minor schema variations across XOA versions.
"""

from __future__ import annotations

import io
import os
import gzip
import json
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# Optional dependencies (import only if available)
try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

# Transparent local/remote file access (e.g., local FS, HTTP(S), GCS, S3)
import fsspec


# --------------------------------------------------------------------------------------
# Small I/O helpers
# --------------------------------------------------------------------------------------

def _open_text(path_or_url: str):
    """
    Open a local/remote file in text mode with UTF-8 encoding.
    Automatically handles gzipped files by content suffix (.gz).
    """
    f = fsspec.open(path_or_url, mode="rb").open()
    if path_or_url.endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(fileobj=f), encoding="utf-8")
    return io.TextIOWrapper(f, encoding="utf-8")


def _exists(path_or_url: str) -> bool:
    """
    Return True if a local/remote file exists and can be opened.
    Failures are swallowed and considered non-existent.
    """
    try:
        with fsspec.open(path_or_url).open() as _:
            return True
    except Exception:
        return False


def _join(base: str, *names: str) -> str:
    """
    Join paths/URLs without making assumptions about the storage backend.
    - Keeps 'base' as-is but strips a trailing slash for consistency.
    - If any child looks like a fully-qualified URL (http/gs://), return that child.
    - Allows absolute child to override the base.
    """
    base = base.rstrip("/")
    if names and any(n and (n.startswith("http") or n.startswith("gs://") or n.startswith("s3://")) for n in names):
        # A child is already a fully-qualified URL -> return that one
        return names[-1]
    for n in names:
        if not n:
            continue
        if n.startswith("/"):
            base = n  # absolute path overrides base
        else:
            base = f"{base}/{n}"
    return base


# --------------------------------------------------------------------------------------
# Read MEX triplet
# --------------------------------------------------------------------------------------

def _read_mex_triplet(
    mex_dir: str,
    matrix_name: str = "matrix.mtx.gz",
    features_name: str = "features.tsv.gz",
    barcodes_name: str = "barcodes.tsv.gz",
) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    """
    Read a 10x MEX (Matrix Exchange) triplet into:
      (X: csr_matrix [cells x features], features_df, barcodes_index)
    MEX is often used for RNA counts; in Gene+Protein runs both appear in the same MEX
    with 'feature_type' disambiguating them.
    """
    from scipy.io import mmread

    mtx_fp = _join(mex_dir, matrix_name)
    feat_fp = _join(mex_dir, features_name)
    bar_fp = _join(mex_dir, barcodes_name)

    if not all(_exists(p) for p in (mtx_fp, feat_fp, bar_fp)):
        raise FileNotFoundError(f"MEX files missing:\n{mtx_fp}\n{feat_fp}\n{bar_fp}")

    # Matrix (COO -> CSR), shape is typically features x cells -> transpose to cells x features
    with fsspec.open(mtx_fp).open() as f:
        if mtx_fp.endswith(".gz"):
            m = mmread(gzip.GzipFile(fileobj=f)).tocsr()
        else:
            m = mmread(f).tocsr()
    X = m.T.tocsr()  # ensure [cells x features]

    # Features
    with _open_text(feat_fp) as f:
        # 10x 'features.tsv.gz' has columns: id, name, feature_type [, genome, ...]
        features = pd.read_csv(f, sep="\t", header=None, comment="#", engine="python")
    cols = ["id", "name", "feature_type"] + [f"col{i}" for i in range(max(0, features.shape[1] - 3))]
    features.columns = cols[:features.shape[1]]

    # Barcodes
    with _open_text(bar_fp) as f:
        barcodes = pd.read_csv(f, sep="\t", header=None, engine="python")[0].astype(str).values

    return X, features, pd.Index(barcodes, name="barcode")


# --------------------------------------------------------------------------------------
# Read Zarr / HDF5 (cell_feature_matrix)
# --------------------------------------------------------------------------------------

def _read_cell_feature_matrix_zarr(zarr_root: str) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    """
    Read the Zarr-based 10x cell_feature_matrix.

    Supports two common layouts under `zarr_root`:
      - <sample>/cell_feature_matrix.zarr/
      - <sample>/cell_feature_matrix/    (a Zarr store directory)

    Returns
    -------
    X : csr_matrix of shape (n_cells, n_features)
    features : DataFrame with at least ['id', 'name', 'feature_type']
    barcodes : Index of cell barcodes (name='barcode')
    """
    if zarr is None:
        raise ImportError("zarr is required to read Zarr-based cell_feature_matrix")

    candidates = []
    for name in ("cell_feature_matrix.zarr", "cell_feature_matrix"):
        p = _join(zarr_root, name)
        if _exists(p):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No Zarr cell_feature_matrix found (*.zarr or directory).")

    store_path = candidates[0]
    store = zarr.open_group(fsspec.get_mapper(store_path), mode="r")

    # Official arrays (X is CSR-like stored components)
    data = store["X/data"][:]
    indices = store["X/indices"][:]
    indptr = store["X/indptr"][:]
    shape = tuple(store["X/shape"][:])

    # Zarr X is typically (features x cells) in many 10x stores; make sure result is [cells x features]
    # We infer CSR by checking length of indptr; if it's features+1, transpose after building CSR.
    if len(indptr) == shape[0] + 1:
        # CSR with rows=features
        mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
        X = mat.T.tocsr()
    elif len(indptr) == shape[1] + 1:
        # CSC with cols=cells
        mat = sparse.csc_matrix((data, indices, indptr), shape=shape)
        X = mat.T.tocsr()
    else:
        # Fallback (assume already cells x features)
        X = sparse.csr_matrix((data, indices, indptr), shape=shape)

    # Features table
    features = pd.DataFrame({
        "id": store["features/id"][:].astype(str),
        "name": store["features/name"][:].astype(str),
        "feature_type": store["features/feature_type"][:].astype(str),
    })

    # Barcodes
    barcodes = pd.Index(store["barcodes"][:].astype(str), name="barcode")
    return X, features, barcodes


def _read_cell_feature_matrix_h5(h5_path: str) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    """
    Read the HDF5-based 10x cell_feature_matrix (Gene+Protein).
    Robust against:
      - Group name variations: 'X', 'matrix', or 'cell_feature_matrix'
      - CSR vs CSC storage
      - Features/barcodes located at group level or root

    Returns
    -------
    X : csr_matrix of shape (n_cells, n_features)
    features : DataFrame with at least ['id', 'name', 'feature_type']
    barcodes : Index of cell barcodes (name='barcode')
    """
    # Open via fsspec when possible (works for remote URLs)
    try:
        fb = fsspec.open(h5_path).open()
        f = h5py.File(fb, "r")
        managed = True
    except Exception:
        f = h5py.File(h5_path, "r")
        managed = False

    def _as_str(arr):
        arr = arr[()]
        if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
            return arr.astype(str)
        return arr

    try:
        # 1) Matrix group
        grp = f.get("X") or f.get("matrix") or f.get("cell_feature_matrix")
        if grp is None:
            raise KeyError("Neither 'X' nor 'matrix' nor 'cell_feature_matrix' exists in the HDF5 file.")

        data = grp["data"][()]
        indices = grp["indices"][()]
        indptr = grp["indptr"][()]
        shape = tuple(grp["shape"][()])  # (n_features, n_barcodes) in most 10x HDF5

        # 2) Infer CSR/CSC and transpose to [cells x features]
        if len(indptr) == shape[0] + 1:
            # CSR rows = features
            mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
            X = mat.T.tocsr()
        elif len(indptr) == shape[1] + 1:
            # CSC cols = barcodes
            mat = sparse.csc_matrix((data, indices, indptr), shape=shape)
            X = mat.T.tocsr()
        else:
            raise ValueError(
                f"Cannot infer matrix format: len(indptr)={len(indptr)}, "
                f"shape={shape} (expect {shape[0]+1} for CSR rows or {shape[1]+1} for CSC cols)."
            )

        # 3) Find features/barcodes in a tolerant way
        def _find(node, name):
            if name in node:
                return node[name]
            if hasattr(node, "parent") and node.parent is not None and name in node.parent:
                return node.parent[name]
            if name in f:
                return f[name]
            return None

        feat_grp = _find(grp, "features")
        if feat_grp is None:
            raise KeyError("Cannot find 'features' group.")

        name_ds = feat_grp.get("name") or feat_grp.get("gene_names")
        if name_ds is None:
            raise KeyError("Cannot find 'features/name' or 'features/gene_names' dataset.")

        features = pd.DataFrame({
            "id": _as_str(feat_grp["id"]),
            "name": _as_str(name_ds),
            "feature_type": _as_str(feat_grp["feature_type"]),
        })

        bc_ds = _find(grp, "barcodes")
        if bc_ds is None:
            raise KeyError("Cannot find 'barcodes' dataset.")
        barcodes = pd.Index(_as_str(bc_ds), name="barcode")

        # 4) Basic sanity checks
        n_cells, n_features = X.shape
        if len(barcodes) != n_cells:
            raise ValueError(f"Barcodes length {len(barcodes)} != X.shape[0] (cells) {n_cells}.")
        if len(features) != n_features:
            raise ValueError(f"Features length {len(features)} != X.shape[1] (features) {n_features}.")

        return X, features, barcodes
    finally:
        try:
            f.close()
        except Exception:
            pass
        if managed:
            try:
                fb.close()
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# Main entry: read Gene + Protein into AnnData
# --------------------------------------------------------------------------------------

def load_xenium_gene_protein(
    base_path: str,
    *,
    prefer: str = "auto",               # "auto" | "zarr" | "h5" | "mex"
    mex_dirname: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    cells_csv: str = "cells.csv.gz",
    cells_parquet: Optional[str] = None,
    read_morphology: bool = False,      # Reserved: hook for morphology_focus/ OME-TIFF
    attach_boundaries: bool = True,     # Attach cell/nucleus boundaries & centroids if available
    # --- NEW: clustering file controls ---
    clusters_relpath: Optional[str] = "analysis/clustering/gene_expression_graphclust/clusters.csv",
    cluster_column_name: str = "cluster",
) -> ad.AnnData:
    """
    Read a Xenium Gene+Protein output folder/URL and produce an AnnData object.

    Parameters
    ----------
    base_path : str
        Local directory or remote URL prefix of a Xenium output (XOA >= v4.0).
    prefer : {"auto","zarr","h5","mex"}, default "auto"
        Format preference. "auto" tries Zarr > HDF5 > MEX.
    mex_dirname, mex_matrix_name, mex_features_name, mex_barcodes_name : str
        File names for MEX components if MEX is selected.
    cells_csv : str, default "cells.csv.gz"
        Cell table; will be used if parquet is not found.
    cells_parquet : str or None
        Optional Parquet for cells table.
    read_morphology : bool, default False
        Placeholder hook (does not load images here; only stores a pointer in .uns).
    attach_boundaries : bool, default True
        If True, attach centroid coordinates (to .obsm["spatial"]) and boundaries CSVs to .uns.
    clusters_relpath : str or None
        If provided and exists, read clustering CSV and merge into adata.obs[cluster_column_name].
        Default path follows 10x GraphClust result under analysis/clustering/.
    cluster_column_name : str, default "cluster"
        Column name used for the merged clustering labels in adata.obs.

    Returns
    -------
    AnnData
        - .X : RNA counts [CSR], RNA features in .var
        - .layers["rna"] : a copy of RNA counts
        - .obsm["protein"] : DataFrame of protein intensities (columns = markers)
        - .obs : cells table (reindexed to barcodes)
        - .uns/.obsm : optional boundaries and spatial (centroids)
        - .obs[cluster_column_name] : merged cluster labels if clusters_relpath is valid
    """
    # ---------- 1) Resolve available cell_feature_matrix sources ----------
    zarr_candidate = None
    for name in ("cell_feature_matrix.zarr", "cell_feature_matrix"):
        p = _join(base_path, name)
        if _exists(p):
            zarr_candidate = p
            break

    h5_path = _join(base_path, "cell_feature_matrix.h5")
    h5_candidate = h5_path if _exists(h5_path) else None

    mex_candidate_dir = _join(base_path, mex_dirname)
    mex_candidate = mex_candidate_dir if _exists(mex_candidate_dir) else None

    def _read_cfm():
        if prefer in ("auto", "zarr") and zarr_candidate:
            return _read_cell_feature_matrix_zarr(base_path)
        if prefer in ("auto", "h5") and h5_candidate:
            return _read_cell_feature_matrix_h5(h5_candidate)
        if prefer in ("auto", "mex") and mex_candidate:
            return _read_mex_triplet(mex_candidate, mex_matrix_name, mex_features_name, mex_barcodes_name)
        raise FileNotFoundError(
            f"No usable cell_feature_matrix found under '{base_path}' "
            f"(tried Zarr/HDF5/MEX according to 'prefer={prefer}')"
        )

    X_all, feat_all, barcodes = _read_cfm()

    # ---------- 2) Split features by type (RNA vs Protein) ----------
    if "feature_type" not in feat_all.columns:
        # Older/edge cases: assume everything is RNA if type is missing.
        feat_all["feature_type"] = "Gene Expression"

    ft_lower = feat_all["feature_type"].astype(str).str.lower()
    mask_rna = ft_lower.str.contains("gene")
    mask_pro = ft_lower.str.contains("protein")

    idx_rna = np.where(mask_rna.values)[0]
    idx_pro = np.where(mask_pro.values)[0]

    # RNA into adata.X (CSR counts)
    X_rna = X_all[:, idx_rna].tocsr() if idx_rna.size else sparse.csr_matrix((X_all.shape[0], 0))
    var_rna = feat_all.loc[mask_rna, ["id", "name", "feature_type"]].copy()
    var_rna.index = var_rna["id"].values

    # Protein into .obsm["protein"] as a DataFrame (float intensities)
    if idx_pro.size:
        X_pro = X_all[:, idx_pro].astype(np.float32)
        var_pro = feat_all.loc[mask_pro, ["id", "name", "feature_type"]].copy()
        var_pro.index = var_pro["id"].values

        pro_names = var_pro["name"].astype(str).values
        # Avoid duplicate column names
        if len(set(pro_names)) != len(pro_names):
            pro_names = [f"{n}_{i}" for i, n in enumerate(pro_names)]
        protein_df = pd.DataFrame(X_pro.toarray(), index=barcodes, columns=pro_names)
    else:
        protein_df = pd.DataFrame(index=barcodes)

    # ---------- 3) Cells table (obs) ----------
    if cells_parquet and _exists(_join(base_path, cells_parquet)):
        obs = pd.read_parquet(_join(base_path, cells_parquet))
    elif _exists(_join(base_path, cells_csv)):
        with _open_text(_join(base_path, cells_csv)) as f:
            obs = pd.read_csv(f)
    else:
        warnings.warn("No cells table found (cells.csv.gz or parquet). Using barcodes as obs index only.")
        obs = pd.DataFrame(index=barcodes)

    # Normalize obs index to barcodes and reindex to match X rows
    if "cell_id" in obs.columns:
        obs = obs.set_index("cell_id")
    if obs.index.name is None or obs.index.name != "barcode":
        obs.index.name = "barcode"
    obs = obs.reindex(barcodes).copy()

    # ---------- 4) NEW: Merge GraphClust clusters (clusters.csv) into obs ----------
    if clusters_relpath:
        clusters_path = _join(base_path, clusters_relpath)
        if _exists(clusters_path):
            try:
                with _open_text(clusters_path) as f:
                    cdf = pd.read_csv(f)
                # Heuristics to find the index (cell ID) column
                idx_col = None
                for cand in ("cell_id", "barcode", "cell", "cellID", "CellID", "cells"):
                    if cand in cdf.columns:
                        idx_col = cand
                        break
                if idx_col is None:
                    idx_col = cdf.columns[0]  # fallback to the first column

                # Heuristics to find the cluster label column
                lower_map = {c.lower(): c for c in cdf.columns}
                clus_col = None
                for key in ("cluster", "clusters", "graphclust", "label", "group"):
                    if key in lower_map:
                        clus_col = lower_map[key]
                        break
                if clus_col is None:
                    # Fallback: if exactly 2 columns, take the non-index one; else use the last column
                    if cdf.shape[1] == 2:
                        clus_col = [c for c in cdf.columns if c != idx_col][0]
                    else:
                        clus_col = cdf.columns[-1]

                cser = (
                    cdf[[idx_col, clus_col]]
                    .dropna(subset=[idx_col])
                    .set_index(idx_col)[clus_col]
                    .astype(str)
                )
                cser.index.name = "barcode"   # In Xenium, cell_id equals barcode
                obs[cluster_column_name] = cser.reindex(barcodes).astype("category")
            except Exception as e:
                warnings.warn(f"Failed to read/merge clusters from '{clusters_path}': {e}")
        else:
            warnings.warn(
                f"Clusters file not found: '{clusters_path}'. "
                f"Set clusters_relpath=None to silence this message."
            )

    # ---------- 5) Assemble AnnData ----------
    adata = ad.AnnData(X=X_rna, obs=obs, var=var_rna)
    adata.layers["rna"] = adata.X.copy()  # explicit naming for downstream code
    adata.obsm["protein"] = protein_df

    # Common metadata
    adata.uns.setdefault("modality", {})
    adata.uns["modality"]["rna"] = {"feature_type": "Gene Expression"}
    if protein_df.shape[1] > 0:
        adata.uns["modality"]["protein"] = {
            "feature_type": "Protein Expression",
            "value": "scaled_mean_intensity"
        }

    # ---------- 6) Optional: attach centroids and boundaries ----------
    if attach_boundaries:
        # Put centroids into .obsm["spatial"] if present in obs
        for cand_x, cand_y in (
            ("x_centroid", "y_centroid"),
            ("cell_x_centroid", "cell_y_centroid"),
            ("centroid_x", "centroid_y"),
        ):
            if cand_x in adata.obs.columns and cand_y in adata.obs.columns:
                adata.obsm["spatial"] = adata.obs[[cand_x, cand_y]].to_numpy()
                break

        # Attach boundary CSVs (kept raw in .uns; advanced users can convert to polygons later)
        for fname, key in (
            ("cell_boundaries.csv.gz", "cell_boundaries"),
            ("nucleus_boundaries.csv.gz", "nucleus_boundaries"),
        ):
            p = _join(base_path, fname)
            if _exists(p):
                try:
                    with _open_text(p) as f:
                        bdf = pd.read_csv(f)
                    adata.uns[key] = bdf
                except Exception as e:
                    warnings.warn(f"Failed to read '{p}': {e}")

    # ---------- 7) Optional: morphology_focus pointer ----------
    if read_morphology:
        adata.uns.setdefault("morphology_focus", {"path": _join(base_path, "morphology_focus")})

    return adata
