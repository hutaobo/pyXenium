"""
pyXenium.io.partial_xenium_loader_v2
------------------------------------

Purpose
=======
Load an AnnData object when you *don't* have a full Xenium `out/` folder, by
stitching together any subset of:

- 10x Gene Expression MEX (matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz])
- `analysis.zarr` or `analysis.zarr.zip` (cluster labels / metadata)
- `cells.zarr` or `cells.zarr.zip` (cell centroids / spatial coords)
- `transcripts.zarr` or `transcripts.zarr.zip` (per-gene transcript locations)

New capabilities (2025-09-22)
-----------------------------
- **Common base path** support: pass `base_dir` (local folder) or `base_url` (HTTP/HTTPS)
  plus filename knobs: `analysis_name`, `cells_name`, `transcripts_name`.
- **Remote loading**: if a URL is provided for zarr (or via `base_url+name`), the file
  is downloaded to a temporary location transparently before reading (so `ZipStore`
  can seek). For MEX, you can pass a `mex_dir` URL (base) together with
  `mex_matrix_name/mex_features_name/mex_barcodes_name`.

Author: Taobo Hu (pyXenium project) — 2025-09-22
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

def _is_url(p: Optional[str | os.PathLike]) -> bool:
    if p is None:
        return False
    s = str(p)
    return s.startswith("http://") or s.startswith("https://")


def _fetch_to_temp(src: str, suffix: Optional[str] = None) -> Path:
    """Download a URL to a temporary file and return its Path.
    Ensures a local, seekable file for ZipStore and mmread.
    """
    logger.info(f"Downloading: {src}")
    r = requests.get(src, stream=True, timeout=60)
    r.raise_for_status()
    tmpdir = Path(tempfile.mkdtemp(prefix="pyxenium_"))
    name = Path(urllib.parse.urlparse(src).path).name or ("tmp" + (suffix or ""))
    dst = tmpdir / name
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dst


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
    transcripts_zarr: Optional[Path | str] = None


# -----------------------------
# MEX loader
# -----------------------------

def _load_mex(mex_dir: Path | str) -> AnnData:
    logger.info(f"Reading MEX from {mex_dir}")
    base = Path(mex_dir) if not _is_url(mex_dir) else Path(mex_dir)  # URL case is pre-downloaded

    candidates = {
        "matrix": ["matrix.mtx", "matrix.mtx.gz"],
        "features": ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"],
        "barcodes": ["barcodes.tsv", "barcodes.tsv.gz"],
    }

    def _find(names: Sequence[str], base: Path) -> Optional[Path]:
        for name in names:
            p = base / name
            if p.exists():
                return p
        for child in base.iterdir():
            if child.is_dir():
                for name in names:
                    p = child / name
                    if p.exists():
                        return p
        return None

    mtx_p = _find(candidates["matrix"], base)
    feat_p = _find(candidates["features"], base)
    barc_p = _find(candidates["barcodes"], base)

    if not (mtx_p and feat_p and barc_p):
        missing = [k for k, v in {"matrix": mtx_p, "features": feat_p, "barcodes": barc_p}.items() if v is None]
        raise FileNotFoundError(f"MEX missing files: {', '.join(missing)} under {base}")

    logger.info(f"  using {mtx_p.name}, {feat_p.name}, {barc_p.name}")
    with (gzip.open(mtx_p, "rb") if str(mtx_p).endswith(".gz") else open(mtx_p, "rb")) as f:
        X = mmread(f).tocsr().astype(np.float32)

    with _open_text_maybe_gz(feat_p) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    feat_df = pd.DataFrame(rows)
    if feat_df.shape[1] == 1:
        feat_df[[1]] = ""
        feat_df[[2]] = ""
    feat_df.columns = ["feature_id", "gene_name", "feature_type"]
    feat_df.index = feat_df["feature_id"].astype(str)

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
    adata.uns.setdefault("io", {})["mex_dir"] = str(base)
    adata.layers["counts"] = adata.X.copy()
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
    if pstr.endswith(".zip"):
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
    cell_ids = _first_available(root, ["cells/cell_id", "cells/ids", "cell_ids", "ids", "barcodes"])  # optional
    if cell_ids is not None:
        cell_ids = cell_ids.astype(str)

    if x is None or y is None:
        logger.warning("Could not locate centroid x/y in cells.zarr; skipping spatial attach.")
        return

    coords = np.column_stack([x, y]) if z is None else np.column_stack([x, y, z])

    if cell_ids is not None:
        df = pd.DataFrame(coords, index=pd.Index(cell_ids, name="cell_id"))
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
    cell_ids = _first_available(root, ["cells/cell_id", "cell_ids", "barcodes", "cells/ids"])  # optional
    if cell_ids is not None:
        cell_ids = cell_ids.astype(str)

    if label_arr is None:
        logger.warning("No cluster labels found in analysis.zarr; skipping.")
        return

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

def _counts_from_transcripts(transcripts_zarr: Path | str, cell_id_index: pd.Index) -> Tuple[sparse.csr_matrix, pd.Index]:
    logger.info(f"Aggregating counts from {transcripts_zarr} (this may be slow)")
    root = _open_zarr(transcripts_zarr)

    gene = _first_available(root, ["transcripts/gene", "transcripts/genes", "gene", "genes"])  # num or str
    cell = _first_available(root, ["transcripts/cell_id", "transcripts/cells", "cell_id", "cell"])  # str or num

    if gene is None or cell is None:
        raise KeyError("Could not locate transcript gene/cell arrays in transcripts.zarr")

    cell = cell.astype(str)

    gene_names = _first_available(root, ["genes/name", "genes/names", "gene_names", "gene/name"])  # optional
    if gene_names is not None and gene.dtype.kind in ("i", "u"):
        gene = gene_names[gene.astype(int)].astype(str)
    else:
        gene = gene.astype(str)

    df = pd.DataFrame({"gene": pd.Categorical(gene), "cell": pd.Categorical(cell)})
    df = df[df["cell"].isin(cell_id_index)]

    gi = df["gene"].cat.codes.to_numpy()
    ci = df["cell"].cat.codes.to_numpy()
    data = np.ones_like(gi, dtype=np.int32)

    n_genes = int(df["gene"].cat.categories.size)
    X = sparse.coo_matrix((data, (gi, ci)), shape=(n_genes, int(df["cell"].cat.categories.size))).tocsr()

    gene_index = pd.Index(df["gene"].cat.categories.astype(str), name="feature_id")
    cell_cats = pd.Index(df["cell"].cat.categories.astype(str), name="cell_id")

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
    *,
    base_dir: Optional[os.PathLike | str] = None,
    base_url: Optional[str] = None,
    analysis_name: str = "analysis.zarr",
    cells_name: str = "cells.zarr",
    transcripts_name: str = "transcripts.zarr",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    cluster_key: str = "Cluster",
    sample: Optional[str] = None,
    keep_unassigned: bool = True,
    build_counts_if_missing: bool = True,
) -> AnnData:
    """Create an AnnData from any combination of partial Xenium artifacts.

    You can pass explicit paths, or a common ``base_dir`` / ``base_url`` plus the
    default filenames. Explicit paths take precedence over base resolution.
    """
    # Resolve MEX
    if mex_dir is not None and _is_url(mex_dir):
        # Download the triplet next to each other and point to that temp dir
        url_base = str(mex_dir).rstrip("/")
        m_p = _fetch_to_temp(url_base + "/" + mex_matrix_name)
        _ = _fetch_to_temp(url_base + "/" + mex_features_name)
        _ = _fetch_to_temp(url_base + "/" + mex_barcodes_name)
        mex_dir_p: Optional[Path] = m_p.parent
    else:
        mex_dir_p = _p(mex_dir)

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
    transcripts_p = _resolve_zarr(transcripts_zarr, transcripts_name)

    # Start with counts
    if mex_dir_p is not None:
        adata = _load_mex(mex_dir_p)
    else:
        # create obs index from any available source
        cell_ids: Optional[pd.Index] = None
        for p in (cells_p, analysis_p, transcripts_p):
            if p is None:
                continue
            try:
                root = _open_zarr(p)
            except Exception:
                continue
            arr = _first_available(root, ["cells/cell_id", "cell_ids", "barcodes", "cells/ids", "cell"])  # type: ignore
            if arr is not None:
                cell_ids = pd.Index(pd.Series(arr).astype(str), name="cell_id")
                break
        if cell_ids is None and transcripts_p is not None:
            root = _open_zarr(transcripts_p)
            c = _first_available(root, ["transcripts/cell_id", "transcripts/cells", "cell_id", "cell"])  # type: ignore
            if c is not None:
                cell_ids = pd.Index(pd.unique(pd.Series(c).astype(str)), name="cell_id")
        if cell_ids is None:
            raise ValueError("Could not determine cell IDs; provide MEX or any zarr with cell ids.")

        if build_counts_if_missing and transcripts_p is not None:
            X, gene_index = _counts_from_transcripts(transcripts_p, cell_ids)
            obs = pd.DataFrame(index=cell_ids)
            var = pd.DataFrame(index=gene_index)
            var["gene_name"] = var.index
            adata = AnnData(X=X, obs=obs, var=var)
            adata.layers["counts"] = adata.X.copy()
        else:
            obs = pd.DataFrame(index=cell_ids)
            var = pd.DataFrame(index=pd.Index([], name="feature_id"))
            adata = AnnData(X=sparse.csr_matrix((len(cell_ids), 0)), obs=obs, var=var)

    # Attach extras
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

    app = typer.Typer(help="Import AnnData from partial Xenium artifacts (local or remote)")

    @app.command("import-partial")
    def cli_import_partial(
        mex_dir: Optional[str] = typer.Option(None, help="Path or URL to 10x MEX directory (URL = base)"),
        analysis_zarr: Optional[str] = typer.Option(None, help="Path/URL to analysis.zarr or .zarr.zip"),
        cells_zarr: Optional[str] = typer.Option(None, help="Path/URL to cells.zarr or .zarr.zip"),
        transcripts_zarr: Optional[str] = typer.Option(None, help="Path/URL to transcripts.zarr or .zarr.zip"),
        base_dir: Optional[str] = typer.Option(None, help="Common local folder for default names"),
        base_url: Optional[str] = typer.Option(None, help="Common HTTP(S) base for default names"),
        analysis_name: str = typer.Option("analysis.zarr", help="Filename for analysis store"),
        cells_name: str = typer.Option("cells.zarr", help="Filename for cells store"),
        transcripts_name: str = typer.Option("transcripts.zarr", help="Filename for transcripts store"),
        mex_matrix_name: str = typer.Option("matrix.mtx.gz", help="MEX matrix filename under mex_dir"),
        mex_features_name: str = typer.Option("features.tsv.gz", help="MEX features filename under mex_dir"),
        mex_barcodes_name: str = typer.Option("barcodes.tsv.gz", help="MEX barcodes filename under mex_dir"),
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
            base_dir=base_dir,
            base_url=base_url,
            analysis_name=analysis_name,
            cells_name=cells_name,
            transcripts_name=transcripts_name,
            mex_matrix_name=mex_matrix_name,
            mex_features_name=mex_features_name,
            mex_barcodes_name=mex_barcodes_name,
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
