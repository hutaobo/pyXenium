from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.neighbors import NearestNeighbors

from pyXenium.validation.atera_wta_breast_topology import (
    DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
    DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR,
)

from .lr_atera import (
    ATERA_BENCHMARK_RELATIVE_ROOT,
    STANDARDIZED_RESULT_COLUMNS,
    load_method_registry,
    resolve_layout,
    run_pyxenium_smoke,
    standardize_result_table,
)


SUPPORTED_REAL_ADAPTERS = (
    "pyxenium",
    "squidpy",
    "liana",
    "commot",
    "cellchat",
    "spatialdm",
    "stlearn",
    "cellphonedb",
    "laris",
    "giotto",
    "spatalk",
    "niches",
    "cellnest",
    "cellagentchat",
    "scild",
)

EXTERNAL_ATTEMPT_METHODS = {"giotto", "spatalk", "niches", "cellnest", "cellagentchat", "scild"}


@dataclass(frozen=True)
class MethodRunSpec:
    method: str
    input_manifest: Path
    output_dir: Path
    database_mode: str = "common-db"
    phase: str = "smoke"
    max_lr_pairs: int | None = None
    n_perms: int = 100
    benchmark_root: Path | None = None
    tbc_results: Path | None = None
    export_figures: bool = False
    rscript: str | None = None
    chunk_id: int | None = None
    num_chunks: int | None = None
    bounded_mode: str | None = None
    gpu_id: str | None = None
    job_id: str | None = None
    gzip_standardized: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _write_tsv(path: str | Path, table: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)
    return path


def _write_method_card(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    method = str(payload.get("method", "unknown"))
    lines = [
        f"# Method Card: {method}",
        "",
        f"- Status: `{payload.get('status', 'unknown')}`",
        f"- Phase: `{payload.get('phase', 'unknown')}`",
        f"- Database mode: `{payload.get('database_mode', 'unknown')}`",
        f"- Reason: `{payload.get('reason', payload.get('error', payload.get('returncode', 'not recorded')))}`",
        f"- Reproduce: `{payload.get('reproduce', 'See run_summary.json and logs.')}`",
        "",
    ]
    if payload.get("package_candidates"):
        lines.append(f"- Package candidates checked: `{payload.get('package_candidates')}`")
    path = output_dir / "method_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _normalize_database_mode(database_mode: str) -> str:
    aliases = {
        "common": "common-db",
        "common-db": "common-db",
        "native": "native-db",
        "native-db": "native-db",
        "smoke": "smoke-panel",
        "smoke-panel": "smoke-panel",
    }
    key = str(database_mode).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported database mode {database_mode!r}. Expected common-db, native-db, or smoke-panel.")
    return aliases[key]


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def load_input_manifest(input_manifest: str | Path) -> dict[str, Any]:
    path = Path(input_manifest)
    if not path.exists():
        raise FileNotFoundError(f"Input manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_input_manifest(manifest: Mapping[str, Any], *, require_full: bool = False) -> dict[str, Any]:
    issues: list[str] = []
    required_bundle_keys = ["counts_symbol_mtx", "barcodes_tsv", "genes_tsv", "meta_tsv", "coords_tsv"]

    def check_bundle(bundle: Mapping[str, Any], label: str) -> None:
        missing = [key for key in required_bundle_keys if not bundle.get(key)]
        if missing:
            issues.append(f"{label} bundle missing paths: {missing}")
            return
        for key in required_bundle_keys:
            if not Path(str(bundle[key])).exists():
                issues.append(f"{label} bundle path for {key} does not exist: {bundle[key]}")

    if not manifest.get("lr_db_common_tsv"):
        issues.append("missing lr_db_common_tsv")
    elif not Path(str(manifest["lr_db_common_tsv"])).exists():
        issues.append(f"lr_db_common_tsv does not exist: {manifest['lr_db_common_tsv']}")
    if manifest.get("smoke_h5ad"):
        if not Path(str(manifest["smoke_h5ad"])).exists():
            issues.append(f"smoke_h5ad does not exist: {manifest['smoke_h5ad']}")
    elif manifest.get("smoke_bundle"):
        check_bundle(dict(manifest["smoke_bundle"]), "smoke")
    else:
        issues.append("missing smoke_h5ad or smoke_bundle")
    if require_full:
        if manifest.get("full_h5ad"):
            if not Path(str(manifest["full_h5ad"])).exists():
                issues.append(f"full_h5ad does not exist: {manifest['full_h5ad']}")
        elif manifest.get("full_bundle"):
            check_bundle(dict(manifest["full_bundle"]), "full")
        else:
            issues.append("missing full_h5ad or full_bundle")
    return {"valid": not issues, "issues": issues}


def _select_input_h5ad(manifest: Mapping[str, Any], phase: str) -> Path | None:
    phase = str(phase).lower()
    if phase == "full":
        if manifest.get("full_h5ad"):
            return Path(str(manifest["full_h5ad"]))
        return None
    if manifest.get("smoke_h5ad"):
        return Path(str(manifest["smoke_h5ad"]))
    return None


def _select_sparse_bundle(manifest: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    if str(phase).lower() == "full":
        if manifest.get("full_bundle"):
            return dict(manifest["full_bundle"])
        raise ValueError("Phase 'full' was requested, but the input manifest does not contain a full sparse bundle.")
    if manifest.get("smoke_bundle"):
        return dict(manifest["smoke_bundle"])
    raise ValueError("Input manifest does not contain a usable sparse bundle for this phase.")


def select_input_source(manifest: Mapping[str, Any], phase: str) -> dict[str, Any]:
    h5ad = _select_input_h5ad(manifest, phase)
    if h5ad is not None and h5ad.exists():
        return {"kind": "h5ad", "path": str(h5ad), "bundle": None}
    bundle = _select_sparse_bundle(manifest, phase)
    return {"kind": "sparse_bundle", "path": str(bundle.get("counts_symbol_mtx", "")), "bundle": dict(bundle)}


def read_sparse_bundle_as_adata(bundle: Mapping[str, Any]) -> ad.AnnData:
    required = ["counts_symbol_mtx", "barcodes_tsv", "genes_tsv", "meta_tsv", "coords_tsv"]
    missing = [key for key in required if not bundle.get(key)]
    if missing:
        raise ValueError(f"Sparse bundle is missing required paths: {missing}")
    for key in required:
        path = Path(str(bundle[key]))
        if not path.exists():
            raise FileNotFoundError(f"Sparse bundle path for {key!r} does not exist: {path}")

    matrix = mmread(str(bundle["counts_symbol_mtx"])).tocsr()
    barcodes = pd.read_csv(str(bundle["barcodes_tsv"]), header=None, sep="\t").iloc[:, 0].astype(str).tolist()
    genes = pd.read_csv(str(bundle["genes_tsv"]), sep="\t")
    meta = pd.read_csv(str(bundle["meta_tsv"]), sep="\t")
    coords = pd.read_csv(str(bundle["coords_tsv"]), sep="\t")

    gene_symbols = genes["gene_symbol"].astype(str) if "gene_symbol" in genes.columns else genes.iloc[:, 0].astype(str)
    if matrix.shape == (len(gene_symbols), len(barcodes)):
        x = matrix.T.tocsr()
    elif matrix.shape == (len(barcodes), len(gene_symbols)):
        x = matrix.tocsr()
    else:
        raise ValueError(
            "Sparse bundle matrix shape does not match genes/barcodes: "
            f"matrix={matrix.shape}, genes={len(gene_symbols)}, barcodes={len(barcodes)}"
        )

    if "cell_id" not in meta.columns:
        raise ValueError("Sparse bundle meta.tsv must contain a 'cell_id' column.")
    meta = meta.copy()
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta = meta.set_index("cell_id").reindex(barcodes)
    if meta.index.hasnans or meta.isna().all(axis=1).any():
        raise ValueError("Sparse bundle meta.tsv does not cover all barcodes.")
    if "cell_type" not in meta.columns:
        raise ValueError("Sparse bundle meta.tsv must contain a 'cell_type' column.")

    if {"x", "y"}.issubset(meta.columns):
        spatial = meta.loc[:, ["x", "y"]].to_numpy(dtype=float)
    else:
        if "cell_id" not in coords.columns or not {"x", "y"}.issubset(coords.columns):
            raise ValueError("Sparse bundle coords.tsv must contain cell_id, x, and y columns.")
        coords = coords.copy()
        coords["cell_id"] = coords["cell_id"].astype(str)
        coords = coords.set_index("cell_id").reindex(barcodes)
        spatial = coords.loc[:, ["x", "y"]].to_numpy(dtype=float)
        meta["x"] = spatial[:, 0]
        meta["y"] = spatial[:, 1]

    var = genes.copy()
    var.index = pd.Index(gene_symbols, name="gene_symbol")
    var.index = pd.Index(var.index.astype(str))
    obs = meta.copy()
    obs["cell_id"] = barcodes
    obs["cell_type"] = obs["cell_type"].astype(str)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obsm["spatial"] = spatial
    return adata


def load_adata_from_manifest(manifest: Mapping[str, Any], phase: str) -> tuple[ad.AnnData, dict[str, Any]]:
    source = select_input_source(manifest, phase)
    if source["kind"] == "h5ad":
        return ad.read_h5ad(source["path"]), source
    return read_sparse_bundle_as_adata(source["bundle"]), source


def _file_sha256(path: str | Path, *, max_bytes: int | None = None) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        while True:
            chunk_size = 1024 * 1024
            if max_bytes is not None:
                remaining = max_bytes - seen
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            seen += len(chunk)
    suffix = "" if max_bytes is None else f":first{seen}"
    return f"sha256{suffix}:{digest.hexdigest()}"


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256_head": _file_sha256(path, max_bytes=8 * 1024 * 1024),
    }


def _package_versions(packages: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def read_lr_resource(
    manifest: Mapping[str, Any],
    *,
    database_mode: str,
    max_lr_pairs: int | None = None,
) -> pd.DataFrame | None:
    mode = _normalize_database_mode(database_mode)
    if mode == "native-db":
        return None

    if mode == "smoke-panel":
        path_key = "atera_smoke_panel_tsv"
    else:
        path_key = "lr_db_common_tsv"
    if not manifest.get(path_key):
        raise ValueError(f"Input manifest is missing {path_key!r} for database mode {mode!r}.")

    resource = pd.read_csv(str(manifest[path_key]), sep="\t").copy()
    rename = {"source": "ligand", "target": "receptor"}
    resource = resource.rename(columns={old: new for old, new in rename.items() if old in resource.columns})
    missing = [col for col in ("ligand", "receptor") if col not in resource.columns]
    if missing:
        raise ValueError(f"LR resource {manifest[path_key]!r} is missing columns: {missing}")

    resource["ligand"] = resource["ligand"].astype(str)
    resource["receptor"] = resource["receptor"].astype(str)
    if "pathway" not in resource.columns:
        resource["pathway"] = "custom"
    if "evidence_weight" not in resource.columns:
        resource["evidence_weight"] = 1.0
    resource = resource.drop_duplicates(["ligand", "receptor"]).reset_index(drop=True)
    if max_lr_pairs is not None and max_lr_pairs > 0:
        resource = resource.head(int(max_lr_pairs)).copy()
    return resource


def _select_lr_chunk(
    resource: pd.DataFrame,
    *,
    chunk_id: int | None = None,
    num_chunks: int | None = None,
    max_lr_pairs: int | None = None,
) -> pd.DataFrame:
    selected = resource.drop_duplicates(["ligand", "receptor"]).reset_index(drop=True).copy()
    if num_chunks is not None or chunk_id is not None:
        if num_chunks is None or chunk_id is None:
            raise ValueError("Both chunk_id and num_chunks must be provided for LR chunking.")
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive.")
        if chunk_id < 0 or chunk_id >= num_chunks:
            raise ValueError(f"chunk_id must be in [0, {num_chunks - 1}], got {chunk_id}.")
        indices = np.array_split(np.arange(len(selected)), int(num_chunks))[int(chunk_id)]
        selected = selected.iloc[indices].reset_index(drop=True)
    if max_lr_pairs is not None and max_lr_pairs > 0:
        selected = selected.head(int(max_lr_pairs)).copy()
    return selected


def _dense_gene_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(gene)
    values = adata[:, [gene]].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=float).reshape(-1)


def _dense_gene_matrix(adata: ad.AnnData, genes: Sequence[str]) -> np.ndarray:
    values = adata[:, list(genes)].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=float)


def aggregate_lr_by_spatial_neighbors(
    adata: ad.AnnData,
    lr_resource: pd.DataFrame,
    *,
    connectivity_key: str = "spatial_connectivities",
    global_scores: pd.DataFrame | None = None,
    global_score_col: str = "LR_score",
    pvalue_col: str | None = "pvalue",
) -> pd.DataFrame:
    if connectivity_key not in adata.obsp:
        ensure_spatial_connectivities(adata, key=connectivity_key)
    labels, _, one_hot = _celltype_one_hot(adata.obs["cell_type"].astype(str).to_numpy())
    conn = adata.obsp[connectivity_key].tocsr()
    edge_counts = one_hot.T @ (conn > 0).astype(float) @ one_hot
    edge_arr = np.asarray(edge_counts.toarray() if sparse.issparse(edge_counts) else edge_counts, dtype=float)
    score_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    if global_scores is not None and not global_scores.empty:
        for _, row in global_scores.iterrows():
            score_lookup[(str(row["ligand"]), str(row["receptor"]))] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for _, lr in lr_resource.iterrows():
        ligand = str(lr["ligand"])
        receptor = str(lr["receptor"])
        if ligand not in adata.var_names or receptor not in adata.var_names:
            continue
        ligand_expr = _dense_gene_vector(adata, ligand)
        receptor_expr = _dense_gene_vector(adata, receptor)
        weighted = conn.multiply(ligand_expr[:, None]).multiply(receptor_expr[None, :]).tocsr()
        aggregate = one_hot.T @ weighted @ one_hot
        aggregate_arr = np.asarray(aggregate.toarray() if sparse.issparse(aggregate) else aggregate, dtype=float)
        global_row = score_lookup.get((ligand, receptor), {})
        global_score = float(global_row.get(global_score_col, 1.0))
        if not np.isfinite(global_score):
            global_score = 1.0
        positive_weight = max(global_score, 0.0)
        aggregate_max = float(np.max(aggregate_arr)) if aggregate_arr.size else 0.0
        if positive_weight == 0.0 and aggregate_max > 0:
            positive_weight = 1e-12
        for sender_idx, sender in enumerate(labels):
            for receiver_idx, receiver in enumerate(labels):
                local_score = float(aggregate_arr[sender_idx, receiver_idx])
                if not np.isfinite(local_score) or local_score <= 0:
                    continue
                edges = float(edge_arr[sender_idx, receiver_idx])
                row = {
                    "ligand": ligand,
                    "receptor": receptor,
                    "sender_celltype": sender,
                    "receiver_celltype": receiver,
                    "LR_score": local_score * positive_weight,
                    "local_lr_score": local_score,
                    "edge_count": edges,
                    "spatial_coherence": local_score / edges if edges > 0 else np.nan,
                }
                if global_row:
                    for key, value in global_row.items():
                        if key in {"ligand", "receptor", "sender_celltype", "receiver_celltype"}:
                            continue
                        row[key] = value
                    if pvalue_col and pvalue_col in global_row:
                        row[pvalue_col] = global_row[pvalue_col]
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_lr_by_celltype_means(adata: ad.AnnData, lr_resource: pd.DataFrame) -> pd.DataFrame:
    labels, _, one_hot = _celltype_one_hot(adata.obs["cell_type"].astype(str).to_numpy())
    counts = np.asarray(one_hot.sum(axis=0)).reshape(-1)
    counts = np.where(counts > 0, counts, 1.0)
    rows: list[dict[str, Any]] = []
    for _, lr in lr_resource.iterrows():
        ligand = str(lr["ligand"])
        receptor = str(lr["receptor"])
        if ligand not in adata.var_names or receptor not in adata.var_names:
            continue
        ligand_expr = _dense_gene_vector(adata, ligand)
        receptor_expr = _dense_gene_vector(adata, receptor)
        ligand_mean = np.asarray(one_hot.T @ ligand_expr).reshape(-1) / counts
        receptor_mean = np.asarray(one_hot.T @ receptor_expr).reshape(-1) / counts
        scores = ligand_mean[:, None] * receptor_mean[None, :]
        for sender_idx, sender in enumerate(labels):
            for receiver_idx, receiver in enumerate(labels):
                score = float(scores[sender_idx, receiver_idx])
                if not np.isfinite(score) or score <= 0:
                    continue
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_celltype": sender,
                        "receiver_celltype": receiver,
                        "LR_score": score,
                        "sender_mean_expr": float(ligand_mean[sender_idx]),
                        "receiver_mean_expr": float(receptor_mean[receiver_idx]),
                    }
                )
    return pd.DataFrame(rows)


def _prepare_expression_adata(adata: ad.AnnData, *, normalize: bool = True) -> ad.AnnData:
    prepared = adata.copy()
    if "cell_type" not in prepared.obs.columns:
        raise ValueError("AnnData.obs must contain 'cell_type' for LR benchmark adapters.")
    if "spatial" not in prepared.obsm:
        spatial_cols = [col for col in ("x", "y") if col in prepared.obs.columns]
        if len(spatial_cols) == 2:
            prepared.obsm["spatial"] = prepared.obs.loc[:, ["x", "y"]].to_numpy(dtype=float)
        else:
            raise ValueError("AnnData must contain obsm['spatial'] or obs columns x/y.")

    if normalize:
        import scanpy as sc

        prepared.layers["counts"] = prepared.X.copy()
        sc.pp.normalize_total(prepared, target_sum=1e4)
        sc.pp.log1p(prepared)
    return prepared


def ensure_categorical_obs(adata: ad.AnnData, column: str) -> ad.AnnData:
    if column not in adata.obs.columns:
        raise ValueError(f"AnnData.obs is missing required column {column!r}.")
    if not isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
        adata = adata.copy()
        adata.obs[column] = pd.Categorical(adata.obs[column].astype(str))
    return adata


def ensure_spatial_connectivities(
    adata: ad.AnnData,
    *,
    key: str = "spatial_connectivities",
    n_neighbors: int = 6,
) -> sparse.csr_matrix:
    if key in adata.obsp:
        conn = adata.obsp[key].tocsr()
    else:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        n_neighbors = max(1, min(int(n_neighbors), coords.shape[0] - 1))
        model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
        model.fit(coords)
        distances, indices = model.kneighbors(coords)
        rows = np.repeat(np.arange(coords.shape[0]), n_neighbors)
        cols = indices[:, 1:].reshape(-1)
        dist = distances[:, 1:].reshape(-1)
        weights = 1.0 / (1.0 + dist)
        conn = sparse.csr_matrix((weights, (rows, cols)), shape=(coords.shape[0], coords.shape[0]))
        conn = conn.maximum(conn.T).tocsr()
        row_sum = np.asarray(conn.sum(axis=1)).reshape(-1)
        row_scale = np.divide(1.0, row_sum, out=np.zeros_like(row_sum, dtype=float), where=row_sum > 0)
        conn = sparse.diags(row_scale) @ conn
        conn = conn.tocsr()
        adata.obsp[key] = conn
    return conn


def estimate_commot_distance_threshold(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    n_neighbors: int = 6,
    quantile: float = 0.95,
) -> float:
    if spatial_key not in adata.obsm:
        raise ValueError(f"AnnData is missing adata.obsm[{spatial_key!r}] required to estimate a COMMOT distance threshold.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        raise ValueError("COMMOT distance threshold estimation requires at least two spatial coordinates.")
    k = int(max(1, min(n_neighbors, coords.shape[0] - 1)))
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(coords)
    distances, _ = model.kneighbors(coords)
    radius = distances[:, k]
    threshold = float(np.quantile(radius, quantile))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Estimated COMMOT distance threshold was non-finite or non-positive.")
    return threshold


def _gene_symbol_key(adata: ad.AnnData) -> str | None:
    for key in ("name", "gene_symbol"):
        if key in adata.var.columns:
            values = adata.var[key].astype(str)
            if values.notna().all() and values.nunique() == len(values):
                return key
    return None


def _split_lr_token(value: Any) -> tuple[str, str]:
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[0]), str(value[1])
    token = str(value)
    for sep in ("^", "|", "->", "~", "::"):
        if sep in token:
            left, right = token.split(sep, 1)
            return left, right
    if "_" in token:
        left, right = token.split("_", 1)
        return left, right
    return token, token


def _split_sender_receiver(value: Any) -> tuple[str, str]:
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[0]), str(value[1])
    token = str(value)
    for sep in ("|", "^", "->", "~", "::"):
        if sep in token:
            left, right = token.split(sep, 1)
            return left, right
    return token, token


def flatten_squidpy_ligrec_result(result: Mapping[str, Any]) -> pd.DataFrame:
    means = result.get("means")
    if not isinstance(means, pd.DataFrame):
        raise ValueError("Squidpy result must contain a 'means' DataFrame.")
    pvalues = result.get("pvalues")
    if pvalues is not None and not isinstance(pvalues, pd.DataFrame):
        raise ValueError("Squidpy result 'pvalues' must be a DataFrame when present.")
    metadata = result.get("metadata")
    metadata_df = metadata if isinstance(metadata, pd.DataFrame) else pd.DataFrame(index=means.index)

    rows: list[dict[str, Any]] = []
    for row_pos, idx in enumerate(means.index):
        if row_pos < len(metadata_df):
            meta = metadata_df.iloc[row_pos]
            ligand = meta.get("source", meta.get("ligand", None))
            receptor = meta.get("target", meta.get("receptor", None))
            if pd.isna(ligand) or pd.isna(receptor):
                ligand, receptor = _split_lr_token(idx)
        else:
            ligand, receptor = _split_lr_token(idx)
        for col in means.columns:
            sender, receiver = _split_sender_receiver(col)
            score = means.loc[idx, col]
            pvalue = pvalues.loc[idx, col] if pvalues is not None and idx in pvalues.index and col in pvalues.columns else np.nan
            rows.append(
                {
                    "ligand": str(ligand),
                    "receptor": str(receptor),
                    "sender_celltype": str(sender),
                    "receiver_celltype": str(receiver),
                    "LR_score": score,
                    "pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def _celltype_one_hot(celltypes: Sequence[str]) -> tuple[list[str], np.ndarray, sparse.csr_matrix]:
    labels = sorted({str(value) for value in celltypes})
    label_to_code = {label: idx for idx, label in enumerate(labels)}
    codes = np.asarray([label_to_code[str(value)] for value in celltypes], dtype=int)
    one_hot = sparse.csr_matrix(
        (np.ones(len(codes), dtype=float), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(labels)),
    )
    return labels, codes, one_hot


def _liana_lr_pairs(local_scores: ad.AnnData) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    var = local_scores.var
    for idx, name in enumerate(local_scores.var_names.astype(str)):
        ligand = var.iloc[idx].get("ligand", var.iloc[idx].get("source", None))
        receptor = var.iloc[idx].get("receptor", var.iloc[idx].get("target", None))
        if pd.isna(ligand) or pd.isna(receptor):
            ligand, receptor = _split_lr_token(name)
        pairs.append((str(ligand), str(receptor)))
    return pairs


def aggregate_liana_bivariate_result(
    local_scores: ad.AnnData,
    source_adata: ad.AnnData,
    *,
    connectivity_key: str = "spatial_connectivities",
    batch_size: int = 128,
) -> pd.DataFrame:
    if local_scores.n_obs != source_adata.n_obs:
        raise ValueError("LIANA local score AnnData must have the same number of observations as the source AnnData.")
    if connectivity_key not in source_adata.obsp:
        raise ValueError(f"Source AnnData is missing obsp[{connectivity_key!r}].")

    labels, receiver_codes, receiver_one_hot = _celltype_one_hot(source_adata.obs["cell_type"].astype(str).to_numpy())
    conn = source_adata.obsp[connectivity_key].tocsr()
    sender_mix = np.asarray(conn @ receiver_one_hot.toarray(), dtype=float)
    lr_pairs = _liana_lr_pairs(local_scores)
    score_matrix = local_scores.X
    rows: list[dict[str, Any]] = []

    for start in range(0, local_scores.n_vars, batch_size):
        end = min(start + batch_size, local_scores.n_vars)
        block = score_matrix[:, start:end]
        block_dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
        block_dense = np.nan_to_num(block_dense.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        for receiver_idx, receiver in enumerate(labels):
            mask = receiver_codes == receiver_idx
            if not np.any(mask):
                continue
            mix = sender_mix[mask, :]
            denom = mix.sum(axis=0)
            if np.allclose(denom, 0.0):
                continue
            aggregate = block_dense[mask, :].T @ mix
            mean_scores = np.divide(aggregate, denom.reshape(1, -1), out=np.zeros_like(aggregate), where=denom.reshape(1, -1) > 0)
            for local_col, (ligand, receptor) in enumerate(lr_pairs[start:end]):
                for sender_idx, sender in enumerate(labels):
                    score = float(mean_scores[local_col, sender_idx])
                    if not np.isfinite(score) or score <= 0:
                        continue
                    rows.append(
                        {
                            "ligand": ligand,
                            "receptor": receptor,
                            "sender_celltype": sender,
                            "receiver_celltype": receiver,
                            "LR_score": score,
                            "spatial_coherence": score,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_commot_obsp_result(
    adata: ad.AnnData,
    lr_resource: pd.DataFrame,
    *,
    database_name: str,
) -> pd.DataFrame:
    labels, _, one_hot = _celltype_one_hot(adata.obs["cell_type"].astype(str).to_numpy())
    rows: list[dict[str, Any]] = []
    for _, lr in lr_resource.iterrows():
        ligand = str(lr["ligand"])
        receptor = str(lr["receptor"])
        candidates = [
            f"commot-{database_name}-{ligand}-{receptor}",
            f"commot-{database_name}-{ligand}-{receptor}".replace("/", "_"),
        ]
        key = next((candidate for candidate in candidates if candidate in adata.obsp), None)
        if key is None:
            prefix = f"commot-{database_name}-{ligand}-"
            key = next((candidate for candidate in adata.obsp.keys() if candidate.startswith(prefix) and candidate.endswith(receptor)), None)
        if key is None:
            continue
        matrix = adata.obsp[key].tocsr()
        aggregate = one_hot.T @ matrix @ one_hot
        edge_counts = one_hot.T @ matrix.astype(bool).astype(float) @ one_hot
        aggregate_arr = np.asarray(aggregate.toarray() if sparse.issparse(aggregate) else aggregate, dtype=float)
        edge_arr = np.asarray(edge_counts.toarray() if sparse.issparse(edge_counts) else edge_counts, dtype=float)
        for sender_idx, sender in enumerate(labels):
            for receiver_idx, receiver in enumerate(labels):
                score = float(aggregate_arr[sender_idx, receiver_idx])
                if not np.isfinite(score) or score <= 0:
                    continue
                edges = float(edge_arr[sender_idx, receiver_idx])
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_celltype": sender,
                        "receiver_celltype": receiver,
                        "LR_score": score,
                        "communication_mass": score,
                        "edge_count": edges,
                        "spatial_coherence": score / edges if edges > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _write_standardized(
    raw: pd.DataFrame,
    *,
    output_dir: Path,
    method: str,
    database_mode: str,
    resolution: str,
    spatial_support_type: str,
    pvalue_col: str | None = None,
    extra_numeric_cols: Sequence[str] = (),
    gzip_output: bool = False,
) -> tuple[Path, pd.DataFrame]:
    standardized = standardize_result_table(
        raw,
        method=method,
        database_mode=database_mode,
        score_col="LR_score",
        sender_col="sender_celltype",
        receiver_col="receiver_celltype",
        pvalue_col=pvalue_col,
        resolution=resolution,
        spatial_support_type=spatial_support_type,
        artifact_path=output_dir / "raw",
        extra_numeric_cols=extra_numeric_cols,
    )
    suffix = ".tsv.gz" if gzip_output else ".tsv"
    path = output_dir / f"{method}_standardized{suffix}"
    _write_tsv(path, standardized)
    return path, standardized


def run_squidpy_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    try:
        import squidpy as sq  # type: ignore
    except ImportError as exc:
        if "FSStore" in str(exc) or "zarr.storage" in str(exc):
            raise ImportError("Squidpy import failed because the active environment has an incompatible zarr/ome-zarr stack. Rebuild pyx-lr-squidpy with the benchmark env file, which pins zarr<3.") from exc
        raise

    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=spec.max_lr_pairs)
    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = _prepare_expression_adata(raw_adata, normalize=True)
    adata = ensure_categorical_obs(adata, "cell_type")
    interactions = None
    if lr_resource is not None:
        interactions = lr_resource.rename(columns={"ligand": "source", "receptor": "target"}).loc[:, ["source", "target"]]

    gene_symbols = _gene_symbol_key(adata)
    result = sq.gr.ligrec(
        adata,
        cluster_key="cell_type",
        interactions=interactions,
        use_raw=False,
        gene_symbols=gene_symbols,
        copy=True,
        n_perms=spec.n_perms,
        seed=0,
        corr_method="fdr_bh",
        corr_axis="clusters",
    )

    raw_dir = spec.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            _write_tsv(raw_dir / f"squidpy_{key}.tsv", value.reset_index())
    raw = flatten_squidpy_ligrec_result(result)
    raw_path = _write_tsv(raw_dir / "squidpy_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="squidpy",
        database_mode=spec.database_mode,
        resolution="celltype_pair",
        spatial_support_type="cluster_permutation",
        pvalue_col="pvalue",
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "squidpy",
        "status": "success",
        "raw_tsv": str(raw_path),
        "standardized_tsv": str(standardized_path),
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
    }


def _call_liana_bivariate(adata: ad.AnnData, lr_resource: pd.DataFrame | None, *, n_perms: int | None = None) -> ad.AnnData:
    import liana as li  # type: ignore

    method_namespace = getattr(li, "mt", None) or getattr(li, "method", None)
    bivariate = getattr(method_namespace, "bivariate", None) if method_namespace is not None else None
    if bivariate is None:
        bivariate = getattr(getattr(li, "method", None), "bivariate", None)
    if bivariate is None:
        raise AttributeError("Could not locate liana bivariate method on li.mt or li.method.")

    adata = adata.copy()
    # LIANA's AnnData path expects reset_index() to create an `index` column; named var indices
    # (e.g. `gene_symbol`) otherwise propagate into incompatible merge keys such as `ligand_gene_symbol`.
    adata.var_names = pd.Index(adata.var_names.astype(str), name=None)
    adata.var.index.name = None

    kwargs: dict[str, Any] = {
        "local_name": "cosine",
        "connectivity_key": "spatial_connectivities",
        "n_perms": n_perms,
        "seed": 0,
        "nz_prop": 0.01,
        "remove_self_interactions": True,
        "use_raw": False,
        "verbose": False,
    }
    if lr_resource is not None:
        kwargs["resource"] = lr_resource.loc[:, ["ligand", "receptor"]].copy()
        kwargs["x_name"] = "ligand"
        kwargs["y_name"] = "receptor"
    else:
        kwargs["resource_name"] = "consensus"

    output = bivariate(adata, **kwargs)
    if isinstance(output, ad.AnnData):
        return output
    if isinstance(output, pd.DataFrame):
        matrix = sparse.csr_matrix(output.to_numpy(dtype=float))
        return ad.AnnData(X=matrix, obs=adata.obs.copy(), var=pd.DataFrame(index=output.columns.astype(str)))
    if output is None and "liana_bivariate" in adata.uns:
        value = adata.uns["liana_bivariate"]
        if isinstance(value, ad.AnnData):
            return value
    raise ValueError("LIANA bivariate did not return an AnnData or recognizable result.")


def run_liana_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=spec.max_lr_pairs)
    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = _prepare_expression_adata(raw_adata, normalize=True)
    ensure_spatial_connectivities(adata, key="spatial_connectivities")

    local_scores = _call_liana_bivariate(adata, lr_resource, n_perms=spec.n_perms if spec.n_perms > 0 else None)
    raw_dir = spec.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_scores.write_h5ad(raw_dir / "liana_bivariate_local_scores.h5ad")

    raw = aggregate_liana_bivariate_result(local_scores, adata, connectivity_key="spatial_connectivities")
    raw_path = _write_tsv(raw_dir / "liana_bivariate_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="liana",
        database_mode=spec.database_mode,
        resolution="local_lr",
        spatial_support_type="spatial_bivariate_neighbor",
        extra_numeric_cols=("spatial_coherence",),
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "liana",
        "status": "success",
        "raw_tsv": str(raw_path),
        "standardized_tsv": str(standardized_path),
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
    }


def _commot_native_resource() -> pd.DataFrame:
    import commot as ct  # type: ignore

    if not hasattr(ct, "pp") or not hasattr(ct.pp, "ligand_receptor_database"):
        raise NotImplementedError("COMMOT native-db mode requires ct.pp.ligand_receptor_database, which is unavailable in this installation.")
    for kwargs in (
        {"database": "CellChat", "species": "human"},
        {"database": "CellChat", "species": "Human"},
        {"database": "CellPhoneDB", "species": "human"},
    ):
        try:
            resource = ct.pp.ligand_receptor_database(**kwargs)
            resource = pd.DataFrame(resource).rename(columns={0: "ligand", 1: "receptor", 2: "pathway"})
            if {"ligand", "receptor"}.issubset(resource.columns):
                if "pathway" not in resource.columns:
                    resource["pathway"] = "native"
                return resource.loc[:, ["ligand", "receptor", "pathway"]].drop_duplicates().reset_index(drop=True)
        except Exception:
            continue
    raise RuntimeError("Could not load a COMMOT native ligand-receptor database.")


def run_commot_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    import commot as ct  # type: ignore

    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=None)
    if lr_resource is None:
        lr_resource = _commot_native_resource()
    lr_resource = _select_lr_chunk(
        lr_resource,
        chunk_id=spec.chunk_id,
        num_chunks=spec.num_chunks,
        max_lr_pairs=spec.max_lr_pairs,
    )
    lr_resource = lr_resource.loc[:, ["ligand", "receptor", "pathway"]].copy()

    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = _prepare_expression_adata(raw_adata, normalize=True)
    database_name = "pyx_common" if _normalize_database_mode(spec.database_mode) != "native-db" else "commot_native"
    distance_threshold = estimate_commot_distance_threshold(adata)
    ct.tl.spatial_communication(
        adata,
        database_name=database_name,
        df_ligrec=lr_resource,
        pathway_sum=True,
        heteromeric=True,
        heteromeric_delimiter="_",
        dis_thr=distance_threshold,
        copy=False,
    )

    raw_dir = spec.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(raw_dir / "commot_result.h5ad")
    raw = aggregate_commot_obsp_result(adata, lr_resource, database_name=database_name)
    raw_path = _write_tsv(raw_dir / "commot_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="commot",
        database_mode=spec.database_mode,
        resolution="cell_pair",
        spatial_support_type="collective_optimal_transport",
        extra_numeric_cols=("communication_mass", "edge_count", "spatial_coherence"),
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "commot",
        "status": "success",
        "chunk_id": spec.chunk_id,
        "num_chunks": spec.num_chunks,
        "raw_tsv": str(raw_path),
        "standardized_tsv": str(standardized_path),
        "standardized_tsv_gz": str(standardized_path) if str(standardized_path).endswith(".gz") else None,
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
        "distance_threshold": distance_threshold,
    }


def run_spatialdm_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    import spatialdm as sdm  # type: ignore
    from spatialdm.stats import Moran_R, rbfweight  # type: ignore

    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=None)
    if lr_resource is None:
        raise NotImplementedError("SpatialDM native-db mode is not implemented in this benchmark adapter; use common-db.")
    lr_resource = _select_lr_chunk(
        lr_resource,
        chunk_id=spec.chunk_id,
        num_chunks=spec.num_chunks,
        max_lr_pairs=spec.max_lr_pairs,
    )
    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = _prepare_expression_adata(raw_adata, normalize=True)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    spatial_scale = estimate_commot_distance_threshold(adata, n_neighbors=6, quantile=0.5)
    spatial_w, knn_connect = rbfweight(coords, l=spatial_scale, n_neighbors=6, single_cell=True)
    adata.obsp["spatial_connectivities"] = knn_connect.tocsr() if sparse.issparse(knn_connect) else sparse.csr_matrix(knn_connect)

    available = lr_resource[lr_resource["ligand"].isin(adata.var_names) & lr_resource["receptor"].isin(adata.var_names)].reset_index(drop=True)
    global_rows: list[pd.DataFrame] = []
    batch_size = int(spec.extra.get("batch_size", 64)) if spec.extra else 64
    for start in range(0, len(available), batch_size):
        batch = available.iloc[start : start + batch_size].copy()
        ligands = batch["ligand"].astype(str).tolist()
        receptors = batch["receptor"].astype(str).tolist()
        ligand_expr = _dense_gene_matrix(adata, ligands)
        receptor_expr = _dense_gene_matrix(adata, receptors)
        moran_r, zscore, pvalue = Moran_R(ligand_expr, receptor_expr, spatial_w, nproc=1)
        moran_r = np.asarray(moran_r, dtype=float).reshape(-1)
        zscore = np.asarray(zscore, dtype=float).reshape(-1)
        pvalue = np.asarray(pvalue, dtype=float).reshape(-1)
        score = np.maximum(zscore, 0.0) * -np.log10(np.clip(pvalue, 1e-300, 1.0))
        global_rows.append(
            pd.DataFrame(
                {
                    "ligand": ligands,
                    "receptor": receptors,
                    "LR_score": score,
                    "spatialdm_moran_r": moran_r,
                    "spatialdm_zscore": zscore,
                    "pvalue": pvalue,
                }
            )
        )
    global_scores = pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame(columns=["ligand", "receptor", "LR_score"])

    raw_dir = spec.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    global_path = _write_tsv(raw_dir / "spatialdm_global.tsv", global_scores)
    raw = aggregate_lr_by_spatial_neighbors(adata, available, global_scores=global_scores, pvalue_col="pvalue")
    raw_path = _write_tsv(raw_dir / "spatialdm_celltype_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="spatialdm",
        database_mode=spec.database_mode,
        resolution="local_lr",
        spatial_support_type="bivariate_moran_neighbor",
        pvalue_col="pvalue",
        extra_numeric_cols=("local_lr_score", "edge_count", "spatial_coherence", "spatialdm_moran_r", "spatialdm_zscore"),
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "spatialdm",
        "status": "success",
        "chunk_id": spec.chunk_id,
        "num_chunks": spec.num_chunks,
        "raw_tsv": str(raw_path),
        "global_tsv": str(global_path),
        "standardized_tsv": str(standardized_path),
        "standardized_tsv_gz": str(standardized_path) if str(standardized_path).endswith(".gz") else None,
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
        "package_versions": {"spatialdm": getattr(sdm, "__version__", "unknown")},
        "spatial_scale": spatial_scale,
    }


def run_stlearn_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    import scanpy as sc
    import stlearn as st  # type: ignore

    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=None)
    if lr_resource is None:
        try:
            native_lrs = st.tl.cci.load_lrs(["connectomeDB2020_lit"], species="human")
        except Exception as exc:
            raise NotImplementedError("stLearn native-db mode failed to load connectomeDB2020_lit.") from exc
        lr_resource = pd.DataFrame([_split_lr_token(value) for value in native_lrs], columns=["ligand", "receptor"])
        lr_resource["pathway"] = "stlearn_native"
    lr_resource = _select_lr_chunk(
        lr_resource,
        chunk_id=spec.chunk_id,
        num_chunks=spec.num_chunks,
        max_lr_pairs=spec.max_lr_pairs,
    )
    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = raw_adata.copy()
    if "cell_type" not in adata.obs.columns:
        raise ValueError("stLearn adapter requires obs['cell_type'].")
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs.loc[:, ["x", "y"]].to_numpy(dtype=float)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    adata.obs["imagecol"] = coords[:, 0]
    adata.obs["imagerow"] = coords[:, 1]
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    distance_threshold = estimate_commot_distance_threshold(adata, n_neighbors=6, quantile=0.5)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)

    lrs = np.asarray([f"{row.ligand}_{row.receptor}" for row in lr_resource.itertuples(index=False)], dtype=str)
    n_pairs = max(1, int(spec.n_perms))
    st.tl.cci.run(
        adata,
        lrs,
        min_spots=20 if adata.n_obs >= 1000 else 1,
        distance=distance_threshold,
        n_pairs=n_pairs,
        n_cpus=int(spec.extra.get("n_cpus", 8)) if spec.extra else 8,
    )
    summary = pd.DataFrame(adata.uns.get("lr_summary", pd.DataFrame())).copy()
    if summary.empty:
        global_scores = pd.DataFrame(columns=["ligand", "receptor", "LR_score"])
    else:
        summary = summary.reset_index(names="lr_pair")
        pairs = summary["lr_pair"].map(_split_lr_token)
        summary["ligand"] = [p[0] for p in pairs]
        summary["receptor"] = [p[1] for p in pairs]
        score_col = "n_spots_sig" if "n_spots_sig" in summary.columns else "n_spots"
        summary["LR_score"] = pd.to_numeric(summary.get(score_col, 0), errors="coerce").fillna(0.0) + 1e-12
        global_scores = summary.loc[:, [col for col in ["ligand", "receptor", "LR_score", "n_spots", "n_spots_sig", "n_spots_sig_pval"] if col in summary.columns]]
    ensure_spatial_connectivities(adata, key="spatial_connectivities")

    raw_dir = spec.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_tsv(raw_dir / "stlearn_lr_summary.tsv", global_scores)
    raw = aggregate_lr_by_spatial_neighbors(adata, lr_resource, global_scores=global_scores, pvalue_col=None)
    raw_path = _write_tsv(raw_dir / "stlearn_celltype_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="stlearn",
        database_mode=spec.database_mode,
        resolution="local_lr",
        spatial_support_type="significant_lr_hotspot_neighbor",
        extra_numeric_cols=("local_lr_score", "edge_count", "spatial_coherence", "n_spots", "n_spots_sig", "n_spots_sig_pval"),
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "stlearn",
        "status": "success",
        "chunk_id": spec.chunk_id,
        "num_chunks": spec.num_chunks,
        "raw_tsv": str(raw_path),
        "summary_tsv": str(summary_path),
        "standardized_tsv": str(standardized_path),
        "standardized_tsv_gz": str(standardized_path) if str(standardized_path).endswith(".gz") else None,
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
        "distance_threshold": distance_threshold,
    }


def run_cellphonedb_baseline_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=None)
    if lr_resource is None:
        raise NotImplementedError("CellPhoneDB native-db mode requires the external CellPhoneDB database export; use common-db for this baseline.")
    lr_resource = _select_lr_chunk(
        lr_resource,
        chunk_id=spec.chunk_id,
        num_chunks=spec.num_chunks,
        max_lr_pairs=spec.max_lr_pairs,
    )
    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = _prepare_expression_adata(raw_adata, normalize=True)
    raw = aggregate_lr_by_celltype_means(adata, lr_resource)
    raw_dir = spec.output_dir / "raw"
    raw_path = _write_tsv(raw_dir / "cellphonedb_mean_product_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="cellphonedb",
        database_mode=spec.database_mode,
        resolution="celltype_pair",
        spatial_support_type="nonspatial_expression_baseline",
        extra_numeric_cols=("sender_mean_expr", "receiver_mean_expr"),
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "cellphonedb",
        "status": "success",
        "chunk_id": spec.chunk_id,
        "num_chunks": spec.num_chunks,
        "raw_tsv": str(raw_path),
        "standardized_tsv": str(standardized_path),
        "standardized_tsv_gz": str(standardized_path) if str(standardized_path).endswith(".gz") else None,
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
        "note": "CellPhoneDB-compatible mean-expression product baseline; spatial support intentionally disabled.",
    }


def run_laris_adapter(spec: MethodRunSpec) -> dict[str, Any]:
    manifest = load_input_manifest(spec.input_manifest)
    lr_resource = read_lr_resource(manifest, database_mode=spec.database_mode, max_lr_pairs=None)
    if lr_resource is None:
        raise NotImplementedError("LARIS native-db mode is not implemented in this benchmark adapter; use common-db.")
    lr_resource = _select_lr_chunk(
        lr_resource,
        chunk_id=spec.chunk_id,
        num_chunks=spec.num_chunks,
        max_lr_pairs=spec.max_lr_pairs,
    )
    raw_adata, _ = load_adata_from_manifest(manifest, spec.phase)
    adata = _prepare_expression_adata(raw_adata, normalize=True)
    conn = ensure_spatial_connectivities(adata, key="spatial_connectivities", n_neighbors=8)
    alpha = float(spec.extra.get("diffusion_alpha", 0.5)) if spec.extra else 0.5
    rows: list[dict[str, Any]] = []
    labels, _, one_hot = _celltype_one_hot(adata.obs["cell_type"].astype(str).to_numpy())
    counts = np.asarray(one_hot.sum(axis=0)).reshape(-1)
    counts = np.where(counts > 0, counts, 1.0)
    for _, lr in lr_resource.iterrows():
        ligand = str(lr["ligand"])
        receptor = str(lr["receptor"])
        if ligand not in adata.var_names or receptor not in adata.var_names:
            continue
        ligand_expr = _dense_gene_vector(adata, ligand)
        receptor_expr = _dense_gene_vector(adata, receptor)
        ligand_diffused = (1.0 - alpha) * ligand_expr + alpha * np.asarray(conn @ ligand_expr).reshape(-1)
        receptor_diffused = (1.0 - alpha) * receptor_expr + alpha * np.asarray(conn @ receptor_expr).reshape(-1)
        ligand_mean = np.asarray(one_hot.T @ ligand_diffused).reshape(-1) / counts
        receptor_mean = np.asarray(one_hot.T @ receptor_diffused).reshape(-1) / counts
        scores = ligand_mean[:, None] * receptor_mean[None, :]
        for sender_idx, sender in enumerate(labels):
            for receiver_idx, receiver in enumerate(labels):
                score = float(scores[sender_idx, receiver_idx])
                if not np.isfinite(score) or score <= 0:
                    continue
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_celltype": sender,
                        "receiver_celltype": receiver,
                        "LR_score": score,
                        "sender_diffused_mean": float(ligand_mean[sender_idx]),
                        "receiver_diffused_mean": float(receptor_mean[receiver_idx]),
                        "diffusion_alpha": alpha,
                    }
                )
    raw = pd.DataFrame(rows)
    raw_dir = spec.output_dir / "raw"
    raw_path = _write_tsv(raw_dir / "laris_knn_diffusion_flat.tsv", raw)
    standardized_path, standardized = _write_standardized(
        raw,
        output_dir=spec.output_dir,
        method="laris",
        database_mode=spec.database_mode,
        resolution="celltype_pair",
        spatial_support_type="knn_diffusion_lr",
        extra_numeric_cols=("sender_diffused_mean", "receiver_diffused_mean", "diffusion_alpha"),
        gzip_output=spec.gzip_standardized,
    )
    return {
        "method": "laris",
        "status": "success",
        "chunk_id": spec.chunk_id,
        "num_chunks": spec.num_chunks,
        "raw_tsv": str(raw_path),
        "standardized_tsv": str(standardized_path),
        "standardized_tsv_gz": str(standardized_path) if str(standardized_path).endswith(".gz") else None,
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
        "note": "LARIS-style kNN diffusion implementation for benchmark comparability.",
    }


def run_external_attempt_adapter(spec: MethodRunSpec, *, package_candidates: Sequence[str]) -> dict[str, Any]:
    if spec.method in {"giotto", "spatalk", "niches"}:
        rscript = spec.rscript or os.environ.get("RSCRIPT", "Rscript")
        if shutil.which(rscript) is None:
            payload = {
                "method": spec.method,
                "status": "failed",
                "phase": spec.phase,
                "database_mode": spec.database_mode,
                "reason": f"Rscript executable was not found: {rscript}",
                "package_candidates": list(package_candidates),
                "reproduce": f"conda run --name <method-env> Rscript -e \"requireNamespace('{package_candidates[0]}')\"",
            }
            _write_json(spec.output_dir / "run_summary.json", payload)
            _write_method_card(spec.output_dir, payload)
            return payload
        checks = " || ".join([f"requireNamespace('{pkg}', quietly=TRUE)" for pkg in package_candidates])
        expr = f"ok <- {checks}; if (!ok) stop('none of the candidate R packages is installed'); cat('R package check passed\\n')"
        completed = subprocess.run([rscript, "-e", expr], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        (spec.output_dir / "external_r_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (spec.output_dir / "external_r_stderr.log").write_text(completed.stderr, encoding="utf-8")
        reason = (
            "Package is installed, but a stable benchmark adapter for its public API has not been finalized yet."
            if completed.returncode == 0
            else "No supported R package was importable in the method-specific environment."
        )
        payload = {
            "method": spec.method,
            "status": "failed",
            "phase": spec.phase,
            "database_mode": spec.database_mode,
            "reason": reason,
            "returncode": completed.returncode,
            "package_candidates": list(package_candidates),
            "stdout_log": str(spec.output_dir / "external_r_stdout.log"),
            "stderr_log": str(spec.output_dir / "external_r_stderr.log"),
            "reproduce": f"conda run --name <method-env> Rscript -e \"{expr}\"",
        }
        _write_json(spec.output_dir / "run_summary.json", payload)
        _write_method_card(spec.output_dir, payload)
        return payload

    found: dict[str, str] = {}
    missing: list[str] = []
    for package in package_candidates:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if not found:
        payload = {
            "method": spec.method,
            "status": "failed",
            "phase": spec.phase,
            "database_mode": spec.database_mode,
            "reason": "No supported package distribution was importable in the method-specific environment.",
            "package_candidates": list(package_candidates),
            "reproduce": f"conda run --name <method-env> python -m pyXenium benchmark atera-lr run-method --method {spec.method}",
        }
        _write_json(spec.output_dir / "run_summary.json", payload)
        _write_method_card(spec.output_dir, payload)
        return payload
    payload = {
        "method": spec.method,
        "status": "failed",
        "phase": spec.phase,
        "database_mode": spec.database_mode,
        "reason": "Package is installed, but a stable benchmark adapter for its public API has not been finalized yet.",
        "package_versions": found,
        "package_candidates": list(package_candidates),
        "reproduce": f"conda run --name <method-env> python -m pyXenium benchmark atera-lr run-method --method {spec.method}",
    }
    _write_json(spec.output_dir / "run_summary.json", payload)
    _write_method_card(spec.output_dir, payload)
    return payload


def run_cellchat_subprocess(spec: MethodRunSpec) -> dict[str, Any]:
    layout = resolve_layout(relative_root=spec.benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    script = layout.runners_dir / "r" / "run_cellchat.R"
    if not script.exists():
        raise FileNotFoundError(f"CellChat R runner does not exist: {script}")
    rscript = spec.rscript or os.environ.get("RSCRIPT", "Rscript")
    if shutil.which(rscript) is None:
        payload = {
            "method": "cellchat",
            "status": "failed",
            "reason": f"Rscript executable was not found: {rscript}",
            "reproduce": f"conda run --name r-lr-cellchat Rscript {script} --method cellchat --input-manifest {spec.input_manifest} --output-dir {spec.output_dir}",
        }
        _write_json(spec.output_dir / "run_summary.json", payload)
        _write_method_card(spec.output_dir, payload)
        raise RuntimeError(payload["reason"])

    command = [
        rscript,
        str(script),
        "--method",
        "cellchat",
        "--input-manifest",
        str(spec.input_manifest),
        "--output-dir",
        str(spec.output_dir),
        "--database-mode",
        spec.database_mode,
        "--phase",
        spec.phase,
    ]
    if spec.max_lr_pairs is not None:
        command.extend(["--max-lr-pairs", str(spec.max_lr_pairs)])
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (spec.output_dir / "cellchat_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (spec.output_dir / "cellchat_stderr.log").write_text(completed.stderr, encoding="utf-8")
    summary_path = spec.output_dir / "run_summary.json"
    if completed.returncode != 0:
        payload = {
            "method": "cellchat",
            "status": "failed",
            "returncode": completed.returncode,
            "stdout_log": str(spec.output_dir / "cellchat_stdout.log"),
            "stderr_log": str(spec.output_dir / "cellchat_stderr.log"),
        }
        _write_json(summary_path, payload)
        _write_method_card(spec.output_dir, payload)
        raise RuntimeError(f"CellChat runner failed with return code {completed.returncode}.")
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"method": "cellchat", "status": "success", "stdout": completed.stdout}


def build_method_run_plan(
    *,
    method: str,
    input_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
    benchmark_root: str | Path | None = None,
    database_mode: str = "common-db",
    phase: str = "smoke",
    max_lr_pairs: int | None = None,
    n_perms: int = 100,
    chunk_id: int | None = None,
    num_chunks: int | None = None,
    bounded_mode: str | None = None,
    gpu_id: str | None = None,
    job_id: str | None = None,
    gzip_standardized: bool = False,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    manifest_path = Path(input_manifest) if input_manifest else layout.data_dir / "input_manifest.json"
    manifest = load_input_manifest(manifest_path)
    mode = _normalize_database_mode(database_mode)
    method_key = str(method).lower()
    registry = load_method_registry(layout.config_dir / "methods.yaml")
    method_info = next((item for item in registry if item.get("slug") == method_key), None)
    if method_info is None:
        raise ValueError(f"Method {method!r} is not registered in {layout.config_dir / 'methods.yaml'}.")
    resolved_output_dir = Path(output_dir) if output_dir else layout.runs_dir / f"{method_key}_{phase}_{_safe_slug(mode)}"
    input_source = select_input_source(manifest, phase)
    lr_resource = read_lr_resource(manifest, database_mode=mode, max_lr_pairs=max_lr_pairs)
    runner = layout.root / str(method_info.get("runner", ""))
    return {
        "method": method_key,
        "display_name": method_info.get("display_name", method_key),
        "language": method_info.get("language"),
        "env_name": method_info.get("env_name"),
        "runner": str(runner),
        "input_manifest": str(manifest_path),
        "input_kind": input_source["kind"],
        "input_h5ad": input_source["path"] if input_source["kind"] == "h5ad" else None,
        "input_bundle_mtx": input_source["path"] if input_source["kind"] == "sparse_bundle" else None,
        "output_dir": str(resolved_output_dir),
        "database_mode": mode,
        "phase": phase,
        "max_lr_pairs": max_lr_pairs,
        "n_perms": n_perms,
        "chunk_id": chunk_id,
        "num_chunks": num_chunks,
        "bounded_mode": bounded_mode,
        "gpu_id": gpu_id,
        "job_id": job_id,
        "gzip_standardized": gzip_standardized,
        "lr_pairs": None if lr_resource is None else int(len(lr_resource)),
        "will_execute": method_key in SUPPORTED_REAL_ADAPTERS,
    }


def run_registered_method(
    *,
    method: str,
    input_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
    benchmark_root: str | Path | None = None,
    database_mode: str = "common-db",
    phase: str = "smoke",
    max_lr_pairs: int | None = None,
    n_perms: int = 100,
    tbc_results: str | Path | None = None,
    export_figures: bool = False,
    rscript: str | None = None,
    chunk_id: int | None = None,
    num_chunks: int | None = None,
    bounded_mode: str | None = None,
    gpu_id: str | None = None,
    job_id: str | None = None,
    gzip_standardized: bool = False,
) -> dict[str, Any]:
    plan = build_method_run_plan(
        method=method,
        input_manifest=input_manifest,
        output_dir=output_dir,
        benchmark_root=benchmark_root,
        database_mode=database_mode,
        phase=phase,
        max_lr_pairs=max_lr_pairs,
        n_perms=n_perms,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        bounded_mode=bounded_mode,
        gpu_id=gpu_id,
        job_id=job_id,
        gzip_standardized=gzip_standardized,
    )
    method_key = plan["method"]
    if not plan["will_execute"]:
        raise NotImplementedError(f"Method {method_key!r} does not have a real adapter yet.")

    resolved_output_dir = Path(plan["output_dir"])
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(plan["input_manifest"])
    manifest = load_input_manifest(manifest_path)
    input_source = select_input_source(manifest, phase)
    lr_resource = read_lr_resource(manifest, database_mode=plan["database_mode"], max_lr_pairs=max_lr_pairs)
    input_fingerprint_path = input_source["path"] if input_source.get("path") else manifest_path
    params = {
        **plan,
        "input_fingerprint": _file_fingerprint(input_fingerprint_path),
        "input_manifest_sha256": _file_sha256(manifest_path),
        "lr_resource_sha256": None if lr_resource is None else _file_sha256(str(manifest["lr_db_common_tsv"])) if plan["database_mode"] == "common-db" else None,
    }
    _write_json(resolved_output_dir / "params.json", params)

    spec = MethodRunSpec(
        method=method_key,
        input_manifest=manifest_path,
        output_dir=resolved_output_dir,
        database_mode=plan["database_mode"],
        phase=phase,
        max_lr_pairs=max_lr_pairs,
        n_perms=n_perms,
        benchmark_root=Path(benchmark_root) if benchmark_root else None,
        tbc_results=Path(tbc_results) if tbc_results else None,
        export_figures=export_figures,
        rscript=rscript,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        bounded_mode=bounded_mode,
        gpu_id=gpu_id,
        job_id=job_id,
        gzip_standardized=gzip_standardized,
    )
    started = time.perf_counter()
    _write_json(
        resolved_output_dir / "run_summary.json",
        {
            "method": method_key,
            "status": "running",
            "elapsed_seconds": 0.0,
            "params_json": str(resolved_output_dir / "params.json"),
        },
    )
    try:
        if method_key == "pyxenium":
            if plan["database_mode"] == "native-db":
                raise ValueError("pyXenium topology adapter does not expose a native database mode; use common-db or smoke-panel.")
            lr_path = manifest.get("lr_db_common_tsv") if plan["database_mode"] == "common-db" else manifest.get("atera_smoke_panel_tsv")
            if max_lr_pairs is not None and lr_path:
                limited = read_lr_resource(manifest, database_mode=plan["database_mode"], max_lr_pairs=max_lr_pairs)
                lr_path = resolved_output_dir / "raw" / "limited_lr_panel.tsv"
                _write_tsv(lr_path, limited if limited is not None else pd.DataFrame())
            if input_source["kind"] == "h5ad":
                selected_h5ad = Path(str(input_source["path"]))
            else:
                adata_from_bundle, _ = load_adata_from_manifest(manifest, phase)
                cache_dir = resolved_output_dir / "input_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                selected_h5ad = cache_dir / f"adata_{phase}_from_sparse_bundle.h5ad"
                adata_from_bundle.write_h5ad(selected_h5ad)
            payload = run_pyxenium_smoke(
                input_h5ad=selected_h5ad,
                output_dir=resolved_output_dir,
                tbc_results=tbc_results or manifest.get("tbc_results") or Path(DEFAULT_ATERA_WTA_BREAST_DATASET_PATH) / DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR,
                lr_panel_path=lr_path,
                database_mode=plan["database_mode"],
                export_figures=export_figures,
            )
        elif method_key == "squidpy":
            payload = run_squidpy_adapter(spec)
        elif method_key == "liana":
            payload = run_liana_adapter(spec)
        elif method_key == "commot":
            payload = run_commot_adapter(spec)
        elif method_key == "cellchat":
            payload = run_cellchat_subprocess(spec)
        elif method_key == "spatialdm":
            payload = run_spatialdm_adapter(spec)
        elif method_key == "stlearn":
            payload = run_stlearn_adapter(spec)
        elif method_key == "cellphonedb":
            payload = run_cellphonedb_baseline_adapter(spec)
        elif method_key == "laris":
            payload = run_laris_adapter(spec)
        elif method_key == "giotto":
            payload = run_external_attempt_adapter(spec, package_candidates=("Giotto", "GiottoClass", "GiottoUtils"))
        elif method_key == "spatalk":
            payload = run_external_attempt_adapter(spec, package_candidates=("SpaTalk",))
        elif method_key == "niches":
            payload = run_external_attempt_adapter(spec, package_candidates=("NICHES",))
        elif method_key == "cellnest":
            payload = run_external_attempt_adapter(spec, package_candidates=("CellNEST", "cellnest"))
        elif method_key == "cellagentchat":
            payload = run_external_attempt_adapter(spec, package_candidates=("CellAgentChat", "cellagentchat"))
        elif method_key == "scild":
            payload = run_external_attempt_adapter(spec, package_candidates=("SCILD", "scild"))
        else:
            raise NotImplementedError(f"Method {method_key!r} does not have a real adapter yet.")
        payload = {
            **payload,
            "elapsed_seconds": float(time.perf_counter() - started),
            "package_versions": _package_versions(["pyXenium", method_key, "anndata", "scanpy", "pandas", "numpy"]),
            "params_json": str(resolved_output_dir / "params.json"),
        }
        _write_json(resolved_output_dir / "run_summary.json", payload)
        return payload
    except Exception as exc:
        payload = {
            "method": method_key,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": float(time.perf_counter() - started),
            "params_json": str(resolved_output_dir / "params.json"),
        }
        _write_json(resolved_output_dir / "run_summary.json", payload)
        _write_method_card(resolved_output_dir, payload)
        raise


def run_smoke_core(
    *,
    methods: Sequence[str] = SUPPORTED_REAL_ADAPTERS,
    input_manifest: str | Path | None = None,
    benchmark_root: str | Path | None = None,
    database_mode: str = "common-db",
    max_lr_pairs: int | None = None,
    n_perms: int = 100,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    manifest_path = Path(input_manifest) if input_manifest else layout.data_dir / "input_manifest.json"
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_key = str(method).strip().lower()
        if not method_key:
            continue
        output_dir = layout.runs_dir / f"{method_key}_smoke_{_safe_slug(_normalize_database_mode(database_mode))}"
        try:
            if dry_run:
                payload = build_method_run_plan(
                    method=method_key,
                    input_manifest=manifest_path,
                    output_dir=output_dir,
                    benchmark_root=benchmark_root,
                    database_mode=database_mode,
                    phase="smoke",
                    max_lr_pairs=max_lr_pairs,
                    n_perms=n_perms,
                )
                payload["status"] = "dry-run"
            else:
                payload = run_registered_method(
                    method=method_key,
                    input_manifest=manifest_path,
                    output_dir=output_dir,
                    benchmark_root=benchmark_root,
                    database_mode=database_mode,
                    phase="smoke",
                    max_lr_pairs=max_lr_pairs,
                    n_perms=n_perms,
                )
            rows.append(payload)
        except Exception as exc:
            failure = {"method": method_key, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            rows.append(failure)
            if not continue_on_error:
                raise

    payload = {
        "status": "completed" if all(row.get("status") != "failed" for row in rows) else "completed_with_failures",
        "database_mode": _normalize_database_mode(database_mode),
        "dry_run": dry_run,
        "methods": rows,
    }
    if not dry_run:
        _write_json(layout.runs_dir / "smoke_core_summary.json", payload)
    return payload
