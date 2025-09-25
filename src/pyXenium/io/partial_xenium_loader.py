"""
partial_xenium_loader.py

A robust loader for assembling an AnnData object from partial
10x/Xenium outputs, with:
  * Automatic MEX orientation fix (features×barcodes -> cells×genes)
  * Optional enrichment from analysis.zarr(.zip):
      - Parse `cell_groups/<group_id>` (CSR by indices/indptr) -> cluster labels
        (group 0 = graph-based clustering by convention)
  * Optional enrichment from cells.zarr(.zip):
      - Read cell_summary -> centroid (x, y) into adata.obs
      - Heuristic reconstruction of string cell_id from 2-col numeric array
  * Per-gene totals for Gene Expression features: adata.var['total_counts']

Drop-in replacement for pyXenium.io.partial_xenium_loader.

Requirements:
  - anndata
  - scipy
  - pandas
Optional:
  - requests (for HTTP downloads)
  - zarr, fsspec (to enrich from zipped zarr via HTTP/local paths)

"""

from __future__ import annotations

import os
import re
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

# Optional: zarr & fsspec for zipped-remote stores
try:
    import zarr  # type: ignore
except Exception:
    zarr = None  # type: ignore

try:
    import fsspec  # type: ignore
except Exception:
    fsspec = None  # type: ignore

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
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def _hf_to_resolve(url: str) -> str:
    """
    Normalize Hugging Face URLs:
      - .../blob/<branch>/... -> .../resolve/<branch>/...
      - .../tree/<branch>/... -> .../resolve/<branch>/...
    """
    if "huggingface.co" in url:
        url = re.sub(r"/blob/", "/resolve/", url)
        url = re.sub(r"/tree/", "/resolve/", url)
    return url


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    if requests is None:
        raise RuntimeError("`requests` is required to download from URLs.")
    url = _hf_to_resolve(url)
    _ensure_dir(dest.parent)
    logger.info(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return dest


def _resolve_from_base(base: str | None, *parts: str) -> str:
    if base is None:
        raise ValueError("Base is None when resolving a path/URL.")
    if _is_url(base):
        return "/".join([base.rstrip("/") ] + [p.strip("/") for p in parts])
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
    feat_df = pd.read_csv(features_path, sep="\t", header=None)
    bc_df = pd.read_csv(barcodes_path, sep="\t", header=None)

    # Build var/obs
    # Prefer feature name (2nd column) as index if present
    if feat_df.shape[1] >= 2:
        var_index = feat_df.iloc[:, 1].astype(str).values
    else:
        var_index = feat_df.iloc[:, 0].astype(str).values
    var = pd.DataFrame(index=pd.Index(var_index, name="feature_name"))
    var["feature_id"] = feat_df.iloc[:, 0].astype(str).values
    if feat_df.shape[1] >= 3:
        # standard 10x column stores e.g. "Gene Expression"
        var["feature_type"] = feat_df.iloc[:, 2].astype(str).values
        var["feature_types"] = var["feature_type"]  # compatibility alias
    else:
        var["feature_type"] = "Gene Expression"
        var["feature_types"] = "Gene Expression"

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


# ---------- analysis.zarr(.zip) helpers ----------

def _open_zip_store(path_or_url: str) -> tuple[object, str | None]:
    """
    Open a local/remote ZIP path via fsspec+zarr ZipStore (read-only).
    Returns (store, tmp_dir) where tmp_dir should be cleaned by caller if not None.
    """
    if zarr is None or fsspec is None:
        raise RuntimeError("Parsing zipped zarr requires both `zarr` and `fsspec`.")

    tmp_dir: str | None = None
    if _is_url(path_or_url):
        url = _hf_to_resolve(path_or_url)
        # Use fsspec's zip::https://... mapper (no local temp needed)
        mapper = f"zip::{url}"
        store = fsspec.get_mapper(mapper)
        return store, tmp_dir
    else:
        # Local file
        from zarr.storage import ZipStore  # type: ignore
        store = ZipStore(path_or_url, mode="r")
        return store, tmp_dir


def _load_cell_groups_to_clusters(
    adata: AnnData,
    analysis_zip: str | None,
    cluster_key: str = "Cluster",
    keep_unassigned: bool = True,
) -> None:
    """
    From analysis.zarr(.zip), load `cell_groups/<group_id>` CSR matrices and
    assign cluster labels from group 0 (graph-based clustering).
    """
    if analysis_zip is None:
        return
    if zarr is None or fsspec is None:
        logger.warning("zarr/fsspec not available; skip analysis.zarr.zip parsing.")
        return

    try:
        store, tmp = _open_zip_store(analysis_zip)
        root = zarr.open(store, mode="r")
        if "cell_groups" not in root:
            return
        cg = root["cell_groups"]

        # Choose group 0 by convention (graph-based clustering)
        if "0" not in cg.group_keys():
            # pick the smallest numeric key as fallback
            keys = sorted([k for k in cg.group_keys() if k.isdigit()], key=lambda s: int(s))
            if not keys:
                return
            use = keys[0]
        else:
            use = "0"

        grp = cg[use]
        idx = grp["indices"][:].astype(np.int64)
        indptr = grp["indptr"][:].astype(np.int64)
        n_rows = len(indptr) - 1
        # columns = cells
        n_cols = adata.n_obs

        data = np.ones_like(idx, dtype=np.uint8)
        M = sparse.csr_matrix((data, idx, indptr), shape=(n_rows, n_cols))

        # Nonzero per cell -> assigned?
        nz_per_cell = np.asarray(M.sum(axis=0)).ravel()
        labels = np.array(M.argmax(axis=0)).ravel()
        labels[nz_per_cell == 0] = -1  # unassigned
        labels = np.where(labels == -1, -1, labels + 1)  # 1-based cluster ids

        cluster_names = np.where(
            labels == -1, "Unassigned", np.array([f"Cluster {x}" for x in labels], dtype=object)
        )
        adata.obs[cluster_key] = pd.Categorical(cluster_names)

        if not keep_unassigned:
            mask = adata.obs[cluster_key].astype(str).str.lower() == "unassigned"
            adata._inplace_subset_obs(~mask.values)

        # provenance
        adata.uns.setdefault("analysis", {})["cell_groups_from"] = str(analysis_zip)

    except Exception as e:
        logger.warning(f"Failed to parse analysis cell_groups: {e}")


# ---------- cells.zarr(.zip) helpers ----------

_HEX2AP = str.maketrans("0123456789abcdef", "abcdefghijklmnop")

def _numeric_cell_id_to_str(prefix: int, suffix: int) -> str:
    """Rebuild a compact string cell_id from numeric pair (prefix, suffix)."""
    h = f"{int(prefix):08x}"
    return h.translate(_HEX2AP) + "-" + str(int(suffix))


def _load_cells_summary_and_xy(
    adata: AnnData,
    cells_zip: str | None,
) -> None:
    """
    From cells.zarr(.zip), read cell_summary and write centroid (x,y) into adata.obs.
    Also try to align by reconstructed string cell_id if necessary.
    """
    if cells_zip is None:
        return
    if zarr is None or fsspec is None:
        logger.warning("zarr/fsspec not available; skip cells.zarr.zip parsing.")
        return

    try:
        store, tmp = _open_zip_store(cells_zip)
        root = zarr.open(store, mode="r")

        # cell_summary
        if "cell_summary" in root:
            A = np.asarray(root["cell_summary"][:])
            attrs = dict(root["cell_summary"].attrs.items())
            cols = (
                attrs.get("columns")
                or attrs.get("col_names")
                or ["cell_centroid_x", "cell_centroid_y"]
            )
            cols = list(cols)[: A.shape[1]]
            df = pd.DataFrame(A, columns=cols)
        else:
            df = pd.DataFrame()

        # cell_id reconstruction from two-column numeric array
        cell_ids = None
        if "cell_id" in root:
            cid = np.asarray(root["cell_id"][:])
            if cid.ndim == 2 and cid.shape[1] >= 2:
                prefix, suffix = cid[:, 0], cid[:, 1]
                cell_ids = [_numeric_cell_id_to_str(p, s) for p, s in zip(prefix, suffix)]

        if cell_ids is not None:
            if not df.empty:
                df.insert(0, "cell_id", cell_ids)
                xy = (
                    df.set_index("cell_id")[["cell_centroid_x", "cell_centroid_y"]]
                    .rename(columns={"cell_centroid_x": "x", "cell_centroid_y": "y"})
                )
                xy_aligned = xy.reindex(adata.obs_names)
                adata.obs["x"] = xy_aligned["x"].to_numpy()
                adata.obs["y"] = xy_aligned["y"].to_numpy()
            else:
                # no summary; still record reconstructed ids for users
                adata.obs["cell_id_reconstructed"] = cell_ids

        else:
            # fallback: map by row order if shapes match
            if not df.empty and {"cell_centroid_x", "cell_centroid_y"}.issubset(df.columns):
                adata.obs["x"] = df["cell_centroid_x"].to_numpy()[: adata.n_obs]
                adata.obs["y"] = df["cell_centroid_y"].to_numpy()[: adata.n_obs]

        adata.uns.setdefault("cells", {})["cells_from"] = str(cells_zip)

    except Exception as e:
        logger.warning(f"Failed to parse cells.zarr.zip: {e}")


def _populate_total_counts(adata: AnnData) -> None:
    """
    Compute per-gene totals for feature_types == 'Gene Expression' and store in
    `adata.var['total_counts']`. Uses layers['counts'] if present.
    """
    M = adata.layers["counts"] if "counts" in adata.layers else adata.X
    is_sparse = sparse.issparse(M)
    if "feature_types" in adata.var:
        mask = (adata.var["feature_types"] == "Gene Expression").to_numpy()
    elif "feature_type" in adata.var:
        mask = (adata.var["feature_type"] == "Gene Expression").to_numpy()
    else:
        # assume all are gene expression
        mask = np.ones(adata.n_vars, dtype=bool)

    cols = np.flatnonzero(mask)
    if cols.size == 0:
        return

    if is_sparse:
        totals = np.asarray(M[:, cols].sum(axis=0)).ravel()
    else:
        totals = M[:, cols].sum(axis=0)

    out = pd.Series(totals, index=adata.var_names[cols], name="total_counts")
    adata.var.loc[out.index, "total_counts"] = out.values


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
        Optional local paths or HTTP(S) URLs to zipped zarr bundles to
        augment obs (e.g., cluster labels, centroid coordinates). Best-effort only.
        Hugging Face URLs like .../blob/main/*.zip or .../tree/main/* are supported.
    base_dir, base_url
        A local base directory or a remote base URL that contains the default layout.
    analysis_name, cells_name
        Filenames for analysis and cells zarr bundles (default: analysis.zarr.zip, cells.zarr.zip).
    mex_* names and mex_default_subdir
        Control the filenames and subdirectory for the MEX triplet discovery.
    cluster_key
        Column name to store cluster labels loaded from analysis.zarr (group 0).
    sample
        Optional sample name recorded in `adata.uns["sample"]`.
    keep_unassigned
        Whether to keep rows labeled as Unassigned when cluster labels are loaded.
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

        # 2) Enrichment from zipped zarr bundles (best-effort)
        # Normalize HF URLs if provided
        az = _hf_to_resolve(analysis_zarr) if isinstance(analysis_zarr, str) else analysis_zarr
        cz = _hf_to_resolve(cells_zarr) if isinstance(cells_zarr, str) else cells_zarr

        _load_cell_groups_to_clusters(
            adata,
            analysis_zip=az if az is not None else (
                _resolve_from_base(base_url, analysis_name) if base_url else (
                    os.path.join(base_dir, analysis_name) if base_dir else None
                )
            ),
            cluster_key=cluster_key,
            keep_unassigned=keep_unassigned,
        )

        _load_cells_summary_and_xy(
            adata,
            cells_zip=cz if cz is not None else (
                _resolve_from_base(base_url, cells_name) if base_url else (
                    os.path.join(base_dir, cells_name) if base_dir else None
                )
            ),
        )

        # 3) Per-gene totals
        _populate_total_counts(adata)

        # 4) Record provenance
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
