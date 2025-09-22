"""
pyXenium.io.partial_xenium_loader
---------------------------------

Purpose
=======
Load an AnnData object when you *don't* have a full Xenium `out/` folder, by
stitching together any subset of:

- 10x Gene Expression MEX (matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz])
- `analysis.zarr` or `analysis.zarr.zip` (cluster labels / metadata)
- `cells.zarr` or `cells.zarr.zip` (cell centroids / spatial coords)
- `transcripts.zarr` or `transcripts.zarr.zip` (per-gene transcript locations)

This module is conservative and tries multiple common key paths inside the Zarr
stores. When something isn't found, it logs a warning instead of failing.

Public API
==========
- load_anndata_from_partial(...): AnnData

Optional CLI (if Typer installed):
----------------------------------
`python -m pyXenium.io.partial_xenium_loader import-partial --help`

Author: Taobo Hu (pyXenium project) — 2025-09-22
"""
from __future__ import annotations

import gzip
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread

try:
    import zarr
    from zarr.storage import ZipStore
except Exception as e:  # pragma: no cover
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

def _p(x: Optional[os.PathLike | str]) -> Optional[Path]:
    return None if x is None else Path(x).expanduser().resolve()


def _open_text_maybe_gz(p: Path) -> io.TextIOBase:
    if str(p).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(p, mode="rb"))
    return open(p, "rt", encoding="utf-8")


@dataclass
class PartialInputs:
    mex_dir: Optional[Path] = None
    analysis_zarr: Optional[Path] = None
    cells_zarr: Optional[Path] = None
    transcripts_zarr: Optional[Path] = None


# -----------------------------
# MEX loader
# -----------------------------

def _load_mex(mex_dir: Path) -> AnnData:
    """Load counts from a 10x-style MEX directory.

    Expected files (gz or not):
      matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz]
    """
    logger.info(f"Reading MEX from {mex_dir}")
    candidates = {
        "matrix": ["matrix.mtx", "matrix.mtx.gz"],
        "features": ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"],
        "barcodes": ["barcodes.tsv", "barcodes.tsv.gz"],
    }

    def _find(
        names: Sequence[str], base: Path
    ) -> Optional[Path]:  # search either directly or in nested single subdir
        for name in names:
            p = base / name
            if p.exists():
                return p
        # handle case where MEX is in a nested directory (e.g. mex_dir/filtered_feature_bc_matrix)
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
    # matrix
    with (gzip.open(mtx_p, "rb") if str(mtx_p).endswith(".gz") else open(mtx_p, "rb")) as f:
        X = mmread(f).tocsr().astype(np.float32)

    # features
    with _open_text_maybe_gz(feat_p) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    # 10x: feature_id, gene_name, feature_type
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

    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns.setdefault("io", {})["mex_dir"] = str(mex_dir)
    # raw counts
    adata.layers["counts"] = adata.X.copy()
    return adata


# -----------------------------
# Zarr helpers
# -----------------------------

def _open_zarr(path: Path):
    if zarr is None:  # pragma: no cover
        raise ImportError("zarr is required to read *.zarr or *.zarr.zip")
    if str(path).endswith(".zip"):
        store = ZipStore(str(path), mode="r")
        return zarr.group(store=store)
    return zarr.open_group(str(path), mode="r")


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


# -----------------------------
# Attach spatial from cells.zarr
# -----------------------------

def _attach_spatial(adata: AnnData, cells_zarr: Path) -> None:
    logger.info(f"Attaching spatial from {cells_zarr}")
    root = _open_zarr(cells_zarr)

    # Try to find centroids and (optionally) cell_ids inside the store
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
    cell_ids = _first_available(root, ["cells/cell_id", "cells/ids", "cell_ids", "ids", "barcodes"])
    # Normalize cell_ids to str
    if cell_ids is not None:
        cell_ids = cell_ids.astype(str)

    if x is None or y is None:
        logger.warning("Could not locate centroid x/y in cells.zarr; skipping spatial attach.")
        return

    coords = np.column_stack([x, y]) if z is None else np.column_stack([x, y, z])

    # Align to adata.obs by cell_id if provided, else assume same order/length
    if cell_ids is not None:
        df = pd.DataFrame(coords, index=pd.Index(cell_ids, name="cell_id"))
        # reindex to adata.obs order
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

def _attach_clusters(adata: AnnData, analysis_zarr: Path, cluster_key: str = "Cluster") -> None:
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
    cell_ids = _first_available(root, ["cells/cell_id", "cell_ids", "barcodes", "cells/ids"])  # optional
    if cell_ids is not None:
        cell_ids = cell_ids.astype(str)

    if label_arr is None:
        logger.warning("No cluster labels found in analysis.zarr; skipping.")
        return

    # map numerical IDs to names if available
    labels = label_arr.astype(str)
    if ids is not None and names is not None and len(ids) == len(names):
        mapping = {str(i): str(n) for i, n in zip(ids, names)}
        labels = np.array([mapping.get(str(x), str(x)) for x in label_arr], dtype=object)

    if cell_ids is not None and len(cell_ids) == len(labels):
        s = pd.Series(labels, index=pd.Index(cell_ids, name="cell_id"))
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
# Build counts from transcripts if MEX missing
# -----------------------------

def _counts_from_transcripts(transcripts_zarr: Path, cell_id_index: pd.Index) -> Tuple[sparse.csr_matrix, pd.Index]:
    """Build a gene x cell sparse count matrix from transcripts.zarr.

    We try common layouts:
      - transcripts/{gene, cell_id}
      - transcripts/{genes, cells}
      - /gene, /cell
      - gene_names at /genes/name or /gene_names

    Returns (X, gene_index)
    """
    logger.info(f"Aggregating counts from {transcripts_zarr} (this may be slow)")
    root = _open_zarr(transcripts_zarr)

    gene = _first_available(root, ["transcripts/gene", "transcripts/genes", "gene", "genes"])  # usually int or str
    cell = _first_available(root, ["transcripts/cell_id", "transcripts/cells", "cell_id", "cell"])  # usually str or int

    if gene is None or cell is None:
        raise KeyError("Could not locate transcript gene/cell arrays in transcripts.zarr")

    # normalize to string cell ids
    cell = cell.astype(str)

    # map genes to names if there's a dictionary
    gene_names = _first_available(root, ["genes/name", "genes/names", "gene_names", "gene/name"])  # optional
    if gene_names is not None and gene.dtype.kind in ("i", "u"):
        # treat `gene` as indices into gene_names
        gene = gene_names[gene.astype(int)].astype(str)
    else:
        gene = gene.astype(str)

    # Build a crosstab: counts per (gene, cell)
    df = pd.DataFrame({"gene": pd.Categorical(gene), "cell": pd.Categorical(cell)})
    # Align cells to provided index, drop cells not present
    df = df[df["cell"].isin(cell_id_index)]

    # categorical codes give us compressed sparse matrix quickly
    gi = df["gene"].cat.codes.to_numpy()
    ci = df["cell"].cat.codes.to_numpy()
    data = np.ones_like(gi, dtype=np.int32)

    n_genes = int(df["gene"].cat.categories.size)
    n_cells = int(df["cell"].cat.categories.size)
    X = sparse.coo_matrix((data, (gi, ci)), shape=(n_genes, n_cells)).tocsr()

    gene_index = pd.Index(df["gene"].cat.categories.astype(str), name="feature_id")
    # The cell categories are a subset of the provided order; we'll later reindex columns
    cell_cats = pd.Index(df["cell"].cat.categories.astype(str), name="cell_id")

    # Reindex cells to the provided order (column-wise)
    column_reindexer = pd.Series(range(len(cell_cats)), index=cell_cats).reindex(cell_id_index)
    take = column_reindexer.dropna().astype(int).to_numpy()
    X = X[:, take]

    return X, gene_index


# -----------------------------
# Public API
# -----------------------------

def load_anndata_from_partial(
    mex_dir: Optional[os.PathLike | str] = None,
    analysis_zarr: Optional[os.PathLike | str] = None,
    cells_zarr: Optional[os.PathLike | str] = None,
    transcripts_zarr: Optional[os.PathLike | str] = None,
    cluster_key: str = "Cluster",
    sample: Optional[str] = None,
    keep_unassigned: bool = True,
    build_counts_if_missing: bool = True,
) -> AnnData:
    """Create an AnnData from any combination of partial Xenium artifacts.

    Parameters
    ----------
    mex_dir : path-like, optional
        Directory containing 10x MEX (matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz]).
    analysis_zarr : path-like, optional
        Zarr store with cluster labels (e.g., `analysis.zarr` or `analysis.zarr.zip`).
    cells_zarr : path-like, optional
        Zarr store with cell centroid coordinates.
    transcripts_zarr : path-like, optional
        Zarr store with per-transcript gene + cell mapping (and optionally coordinates).
    cluster_key : str
        Column name in `adata.obs` to store cluster labels.
    sample : str, optional
        Stored in `adata.uns['sample']` for provenance.
    keep_unassigned : bool
        If False, drop cells with cluster label in {"-1", "NA", "None", "Unassigned"}.
    build_counts_if_missing : bool
        If True and `mex_dir` is None, try to build a counts matrix from transcripts.

    Returns
    -------
    AnnData
    """
    mex_dir_p = _p(mex_dir)
    analysis_p = _p(analysis_zarr)
    cells_p = _p(cells_zarr)
    transcripts_p = _p(transcripts_zarr)

    if mex_dir_p is not None:
        adata = _load_mex(mex_dir_p)
    else:
        # create empty adata with cell index discovered from any available source
        # Prefer cells.zarr -> cell_ids; else analysis.zarr -> cell_ids; else transcripts.zarr -> cells
        cell_ids: Optional[pd.Index] = None
        for src in [(cells_p, "cells"), (analysis_p, "analysis"), (transcripts_p, "transcripts")]:
            p, tag = src
            if p is None:
                continue
            try:
                root = _open_zarr(p)
            except Exception:
                continue
            arr = _first_available(root, ["cells/cell_id", "cell_ids", "barcodes", "cells/ids", "cell"])
            if arr is not None:
                cell_ids = pd.Index(pd.Series(arr).astype(str), name="cell_id")
                logger.info(f"Discovered {len(cell_ids)} cells from {tag}.zarr")
                break
        if cell_ids is None and transcripts_p is not None:
            # As last resort, take unique from transcripts
            root = _open_zarr(transcripts_p)
            c = _first_available(root, ["transcripts/cell_id", "transcripts/cells", "cell_id", "cell"])  # type: ignore
            if c is not None:
                cell_ids = pd.Index(pd.unique(pd.Series(c).astype(str)), name="cell_id")
                logger.info(f"Discovered {len(cell_ids)} cells from transcripts.zarr")

        if cell_ids is None:
            raise ValueError("Could not determine cell IDs without MEX; please provide cells.zarr/analysis.zarr/transcripts.zarr")

        # X from transcripts if requested
        if build_counts_if_missing and transcripts_p is not None:
            X, gene_index = _counts_from_transcripts(transcripts_p, cell_ids)
            obs = pd.DataFrame(index=cell_ids)
            var = pd.DataFrame(index=gene_index)
            var["gene_name"] = var.index  # best-effort
            adata = AnnData(X=X, obs=obs, var=var)
            adata.layers["counts"] = adata.X.copy()
        else:
            obs = pd.DataFrame(index=cell_ids)
            var = pd.DataFrame(index=pd.Index([], name="feature_id"))
            adata = AnnData(X=sparse.csr_matrix((len(cell_ids), 0)), obs=obs, var=var)

    # Attach extras if available
    if analysis_p is not None:
        try:
            _attach_clusters(adata, analysis_p, cluster_key=cluster_key)
            if not keep_unassigned:
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

    # Provenance
    adata.uns.setdefault("io", {})["analysis_zarr"] = str(analysis_p) if analysis_p else None
    adata.uns.setdefault("io", {})["cells_zarr"] = str(cells_p) if cells_p else None
    adata.uns.setdefault("io", {})["transcripts_zarr"] = str(transcripts_p) if transcripts_p else None
    if sample is not None:
        adata.uns["sample"] = str(sample)

    return adata


# -----------------------------
# Optional CLI via Typer
# -----------------------------
try:  # pragma: no cover
    import typer

    app = typer.Typer(help="Import AnnData from partial Xenium artifacts")

    @app.command("import-partial")
    def cli_import_partial(
        mex_dir: Optional[str] = typer.Option(None, help="Path to 10x MEX directory"),
        analysis_zarr: Optional[str] = typer.Option(None, help="Path to analysis.zarr or .zarr.zip"),
        cells_zarr: Optional[str] = typer.Option(None, help="Path to cells.zarr or .zarr.zip"),
        transcripts_zarr: Optional[str] = typer.Option(None, help="Path to transcripts.zarr or .zarr.zip"),
        cluster_key: str = typer.Option("Cluster", help="Obs column name for cluster labels"),
        sample: Optional[str] = typer.Option(None, help="Sample name for provenance"),
        keep_unassigned: bool = typer.Option(True, help="Keep cells with unassigned cluster"),
        build_counts_if_missing: bool = typer.Option(True, help="If no MEX, build counts from transcripts"),
        output_h5ad: str = typer.Option("partial.h5ad", help="Output .h5ad path"),
    ):
        adata = load_anndata_from_partial(
            mex_dir=mex_dir,
            analysis_zarr=analysis_zarr,
            cells_zarr=cells_zarr,
            transcripts_zarr=transcripts_zarr,
            cluster_key=cluster_key,
            sample=sample,
            keep_unassigned=keep_unassigned,
            build_counts_if_missing=build_counts_if_missing,
        )
        adata.write_h5ad(output_h5ad)
        typer.echo(f"Wrote {output_h5ad} (n_cells={adata.n_obs}, n_genes={adata.n_vars})")

except Exception:  # Typer not installed
    app = None  # type: ignore


__all__ = ["load_anndata_from_partial", "PartialInputs", "app"]
