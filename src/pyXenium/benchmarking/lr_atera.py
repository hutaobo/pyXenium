from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.io import mmwrite

from pyXenium.ligand_receptor import ligand_receptor_topology_analysis
from pyXenium.validation.atera_wta_breast_topology import (
    DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS,
    DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
    DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR,
    DEFAULT_LR_SMOKE_PANEL,
)

ATERA_BENCHMARK_RELATIVE_ROOT = Path("benchmarking") / "lr_2026_atera"
STANDARDIZED_RESULT_COLUMNS = [
    "method",
    "database_mode",
    "ligand",
    "receptor",
    "sender",
    "receiver",
    "score_raw",
    "score_std",
    "rank_within_method",
    "rank_fraction",
    "fdr_or_pvalue",
    "resolution",
    "spatial_support_type",
    "artifact_path",
]


@dataclass(frozen=True)
class BenchmarkLayout:
    root: Path
    config_dir: Path
    env_dir: Path
    scripts_dir: Path
    runners_dir: Path
    data_dir: Path
    logs_dir: Path
    results_dir: Path
    reports_dir: Path
    runs_dir: Path
    templates_dir: Path


def _portable_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def _portable_join(base: str | Path, relative: str | Path) -> Path:
    parts = [part for part in str(relative).replace("\\", "/").split("/") if part]
    return Path(base).joinpath(*parts)


def _find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root from the provided start path.")


def resolve_layout(repo_root: str | Path | None = None, relative_root: str | Path = ATERA_BENCHMARK_RELATIVE_ROOT) -> BenchmarkLayout:
    root = _find_repo_root(repo_root) / Path(relative_root)
    return BenchmarkLayout(
        root=root,
        config_dir=root / "configs",
        env_dir=root / "envs",
        scripts_dir=root / "scripts",
        runners_dir=root / "runners",
        data_dir=root / "data",
        logs_dir=root / "logs",
        results_dir=root / "results",
        reports_dir=root / "reports",
        runs_dir=root / "runs",
        templates_dir=root / "templates",
    )


def ensure_layout(layout: BenchmarkLayout) -> BenchmarkLayout:
    for path in (
        layout.root,
        layout.config_dir,
        layout.env_dir,
        layout.scripts_dir,
        layout.runners_dir,
        layout.data_dir,
        layout.logs_dir,
        layout.results_dir,
        layout.reports_dir,
        layout.runs_dir,
        layout.templates_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_method_registry(methods_path: str | Path) -> list[dict[str, Any]]:
    payload = _load_yaml(methods_path)
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise ValueError("methods.yaml must contain a top-level 'methods' list.")
    return methods


def _as_sparse_matrix(matrix: Any) -> sparse.spmatrix:
    if sparse.issparse(matrix):
        return matrix.tocsc()
    return sparse.csc_matrix(np.asarray(matrix))


def harmonize_adata_for_benchmark(adata: ad.AnnData, *, cell_type_col: str = "cluster", copy: bool = True) -> ad.AnnData:
    bench = adata.copy() if copy else adata
    if "spatial" not in bench.obsm:
        raise ValueError("AnnData must contain .obsm['spatial'] to participate in the LR benchmark.")

    bench.obs = bench.obs.copy()
    bench.var = bench.var.copy()

    bench.obs["cell_id"] = bench.obs.get("cell_id", bench.obs_names.astype(str))
    bench.obs["cell_id"] = bench.obs["cell_id"].astype(str)

    if cell_type_col not in bench.obs.columns:
        raise ValueError(f"AnnData.obs is missing the requested cell type column {cell_type_col!r}.")
    bench.obs["cell_type"] = bench.obs[cell_type_col].astype(str)

    spatial = np.asarray(bench.obsm["spatial"])
    if spatial.shape[1] < 2:
        raise ValueError("AnnData.obsm['spatial'] must have at least two columns.")
    bench.obs["x"] = spatial[:, 0].astype(float)
    bench.obs["y"] = spatial[:, 1].astype(float)

    original_var_names = pd.Index(bench.var_names.astype(str), name="ensembl_id")
    bench.var["ensembl_id"] = original_var_names.to_numpy()
    if "name" in bench.var.columns:
        gene_symbol = bench.var["name"].astype(str)
    else:
        gene_symbol = pd.Series(original_var_names.astype(str), index=bench.var_names, dtype=str)
    bench.var["gene_symbol"] = gene_symbol.to_numpy()
    bench.var_names = pd.Index(gene_symbol.astype(str), name="gene_symbol")
    bench.var_names_make_unique()

    return bench


def _stratified_indices(labels: pd.Series, n_cells: int, seed: int) -> np.ndarray:
    if n_cells >= len(labels):
        return np.arange(len(labels), dtype=int)
    counts = labels.value_counts().sort_index()
    weights = counts / counts.sum()
    raw = weights * n_cells
    alloc = pd.Series(np.floor(raw.to_numpy()), index=raw.index, dtype=int)
    alloc = alloc.clip(lower=1)
    excess = int(alloc.sum() - n_cells)
    deficit = int(n_cells - alloc.sum())

    if excess > 0:
        fractional = pd.Series(raw.to_numpy() - np.floor(raw.to_numpy()), index=raw.index).sort_values()
        for label in fractional.index:
            while excess > 0 and alloc[label] > 1:
                alloc[label] -= 1
                excess -= 1
            if excess == 0:
                break
    elif deficit > 0:
        fractional = pd.Series(raw.to_numpy() - np.floor(raw.to_numpy()), index=raw.index).sort_values(ascending=False)
        for label in fractional.index:
            alloc[label] += 1
            deficit -= 1
            if deficit == 0:
                break

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label, want in alloc.items():
        group_idx = np.flatnonzero(labels.to_numpy() == label)
        take = min(int(want), len(group_idx))
        chosen.extend(rng.choice(group_idx, size=take, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=int)


def stratified_subset(adata: ad.AnnData, *, n_cells: int, stratify_key: str = "cell_type", seed: int = 0) -> ad.AnnData:
    if stratify_key not in adata.obs:
        raise ValueError(f"AnnData.obs is missing the requested stratification column {stratify_key!r}.")
    idx = _stratified_indices(adata.obs[stratify_key].astype(str), n_cells=n_cells, seed=seed)
    return adata[idx].copy()


def _write_sparse_bundle(adata: ad.AnnData, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = _as_sparse_matrix(adata.X).T.tocoo()

    matrix_path = out_dir / "counts_symbol.mtx"
    barcodes_path = out_dir / "barcodes.tsv"
    features_path = out_dir / "genes.tsv"
    meta_path = out_dir / "meta.tsv"
    coords_path = out_dir / "coords.tsv"

    mmwrite(str(matrix_path), matrix)
    pd.Series(adata.obs["cell_id"].astype(str)).to_csv(barcodes_path, index=False, header=False)
    pd.DataFrame(
        {
            "gene_symbol": adata.var_names.astype(str),
            "ensembl_id": adata.var.get("ensembl_id", adata.var_names).astype(str),
            "feature_type": adata.var.get("feature_type", pd.Series("Gene Expression", index=adata.var_names)).astype(str),
        }
    ).to_csv(features_path, sep="\t", index=False)
    adata.obs.loc[:, ["cell_id", "cell_type", "x", "y"]].to_csv(meta_path, sep="\t", index=False)
    adata.obs.loc[:, ["cell_id", "x", "y"]].to_csv(coords_path, sep="\t", index=False)

    return {
        "counts_symbol_mtx": str(matrix_path),
        "barcodes_tsv": str(barcodes_path),
        "genes_tsv": str(features_path),
        "meta_tsv": str(meta_path),
        "coords_tsv": str(coords_path),
    }


def _file_fingerprint(path: str | Path, *, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        while seen < max_bytes:
            chunk = handle.read(min(1024 * 1024, max_bytes - seen))
            if not chunk:
                break
            digest.update(chunk)
            seen += len(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256_head": digest.hexdigest(),
        "sha256_head_bytes": seen,
    }


def _bundle_fingerprints(bundle: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    return {key: _file_fingerprint(path) for key, path in bundle.items()}


def _build_celltype_pairs(celltypes: Sequence[str]) -> pd.DataFrame:
    values = sorted({str(item) for item in celltypes})
    rows = [{"sender": sender, "receiver": receiver} for sender in values for receiver in values]
    return pd.DataFrame(rows)


def _default_lr_common_db(adata: ad.AnnData) -> tuple[pd.DataFrame, dict[str, Any]]:
    genes = set(adata.var_names.astype(str))
    resource: pd.DataFrame | None = None
    source = "liana_consensus"
    try:
        import liana as li  # type: ignore

        resource = li.resource.select_resource("consensus")[["ligand", "receptor"]].drop_duplicates()
    except Exception:
        source = "atera_smoke_panel_fallback"
        resource = DEFAULT_LR_SMOKE_PANEL.loc[:, ["ligand", "receptor"]].drop_duplicates()

    resource = resource.copy()
    resource["ligand"] = resource["ligand"].astype(str)
    resource["receptor"] = resource["receptor"].astype(str)
    resource = resource.loc[resource["ligand"].isin(genes) & resource["receptor"].isin(genes)].drop_duplicates()

    if resource.empty:
        source = "smoke_panel_only"
        resource = DEFAULT_LR_SMOKE_PANEL.loc[:, ["ligand", "receptor"]].drop_duplicates().copy()

    resource = resource.sort_values(["ligand", "receptor"]).reset_index(drop=True)
    return resource, {"source": source, "n_pairs": int(len(resource))}


def prepare_atera_lr_benchmark(
    *,
    dataset_root: str | Path | None = DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
    benchmark_root: str | Path | None = None,
    tbc_results: str | Path | None = None,
    smoke_n_cells: int = 20_000,
    seed: int = 0,
    prefer: str = "h5",
    export_full_bundle: bool = True,
    write_full_h5ad: bool = True,
) -> dict[str, Any]:
    from pyXenium.io import read_xenium

    repo_root = _find_repo_root()
    layout = ensure_layout(resolve_layout(repo_root=repo_root, relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT))
    dataset_root = Path(dataset_root or DEFAULT_ATERA_WTA_BREAST_DATASET_PATH)
    tbc_path = _portable_path(tbc_results) if tbc_results else _portable_join(dataset_root, DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR)

    raw = read_xenium(
        str(dataset_root),
        as_="anndata",
        prefer=prefer,
        include_transcripts=False,
        include_boundaries=False,
        include_images=False,
        clusters_relpath=DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS,
        cluster_column_name="cluster",
        cells_parquet="cells.parquet",
    )
    adata = harmonize_adata_for_benchmark(raw, cell_type_col="cluster", copy=True)
    smoke = stratified_subset(adata, n_cells=smoke_n_cells, stratify_key="cell_type", seed=seed)

    full_dir = layout.data_dir / "full"
    smoke_dir = layout.data_dir / "smoke"
    full_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    full_h5ad = full_dir / "adata_full.h5ad"
    smoke_h5ad = smoke_dir / "adata_smoke.h5ad"
    if export_full_bundle and write_full_h5ad:
        adata.write_h5ad(full_h5ad)
    smoke.write_h5ad(smoke_h5ad)

    full_bundle = _write_sparse_bundle(adata, full_dir) if export_full_bundle else {}
    smoke_bundle = _write_sparse_bundle(smoke, smoke_dir)

    common_db, common_db_meta = _default_lr_common_db(adata)
    common_db["evidence_weight"] = 1.0
    common_db_path = layout.data_dir / "lr_db_common.tsv"
    common_db.to_csv(common_db_path, sep="\t", index=False)

    smoke_panel_path = layout.data_dir / "atera_smoke_panel.tsv"
    smoke_panel = DEFAULT_LR_SMOKE_PANEL.copy()
    smoke_panel.to_csv(smoke_panel_path, sep="\t", index=False)

    celltype_pairs = _build_celltype_pairs(adata.obs["cell_type"].astype(str))
    celltype_pairs_path = layout.data_dir / "celltype_pairs.tsv"
    celltype_pairs.to_csv(celltype_pairs_path, sep="\t", index=False)

    payload = {
        "xenium_root": str(dataset_root),
        "dataset_root": str(dataset_root),
        "readonly_xenium_root": str(dataset_root),
        "writable_benchmark_root": str(layout.root),
        "tbc_results": str(tbc_path),
        "benchmark_root": str(layout.root),
        "full_h5ad": str(full_h5ad) if export_full_bundle and write_full_h5ad else None,
        "smoke_h5ad": str(smoke_h5ad),
        "lr_db_common_tsv": str(common_db_path),
        "atera_smoke_panel_tsv": str(smoke_panel_path),
        "celltype_pairs_tsv": str(celltype_pairs_path),
        "full_bundle": full_bundle,
        "smoke_bundle": smoke_bundle,
        "full_bundle_fingerprints": _bundle_fingerprints(full_bundle),
        "smoke_bundle_fingerprints": _bundle_fingerprints(smoke_bundle),
        "full_h5ad_fingerprint": _file_fingerprint(full_h5ad) if export_full_bundle and write_full_h5ad else None,
        "smoke_h5ad_fingerprint": _file_fingerprint(smoke_h5ad),
        "full_n_cells": int(adata.n_obs),
        "full_n_genes": int(adata.n_vars),
        "smoke_n_cells": int(smoke.n_obs),
        "smoke_n_genes": int(smoke.n_vars),
        "n_celltypes": int(adata.obs["cell_type"].nunique()),
        "common_db": common_db_meta,
        "export_full_bundle": export_full_bundle,
        "write_full_h5ad": write_full_h5ad,
        "matrix_note": "Dense counts_symbol.tsv is intentionally omitted for the full Xenium matrix; sparse counts_symbol.mtx is exported instead.",
    }
    manifest_path = layout.data_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["input_manifest_json"] = str(manifest_path)
    return payload


def _resolve_result_rank(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    if values.empty:
        empty = pd.Series(dtype=float, index=values.index)
        return empty, empty
    rank = values.rank(method="min", ascending=False)
    denom = float(len(values))
    rank_fraction = 1.0 - ((rank - 1.0) / denom)
    return rank, rank_fraction


def _resolve_ordered_rank(scores: pd.Series, pvalues: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    if scores.empty:
        empty = pd.Series(dtype=float, index=scores.index)
        return empty, empty

    sortable = pd.DataFrame({"score": pd.to_numeric(scores, errors="coerce")}, index=scores.index)
    if sortable["score"].notna().any():
        fill_value = float(sortable["score"].min(skipna=True)) - 1.0
    else:
        fill_value = 0.0
    sortable["score"] = sortable["score"].fillna(fill_value)
    if pvalues is not None:
        sortable["pvalue"] = pd.to_numeric(pvalues, errors="coerce").fillna(np.inf)
    else:
        sortable["pvalue"] = np.inf

    ordered_index = sortable.sort_values(["score", "pvalue"], ascending=[False, True], kind="mergesort").index
    rank = pd.Series(np.arange(1, len(ordered_index) + 1, dtype=float), index=ordered_index).reindex(scores.index)
    rank_fraction = 1.0 - ((rank - 1.0) / float(len(ordered_index)))
    return rank.astype(float), rank_fraction.astype(float)


def _zscore(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float, index=values.index)
    std = float(values.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series(0.0, index=values.index, dtype=float)
    mean = float(values.mean())
    return (values - mean) / std


def standardize_result_table(
    df: pd.DataFrame,
    *,
    method: str,
    database_mode: str,
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    sender_col: str = "sender_celltype",
    receiver_col: str = "receiver_celltype",
    score_col: str = "LR_score",
    pvalue_col: str | None = None,
    resolution: str = "celltype_pair",
    spatial_support_type: str = "local_contact",
    artifact_path: str | Path | None = None,
    extra_numeric_cols: Sequence[str] = (),
) -> pd.DataFrame:
    required = [ligand_col, receptor_col, sender_col, receiver_col, score_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot standardize result table because columns are missing: {missing}")

    result = pd.DataFrame(
        {
            "method": method,
            "database_mode": database_mode,
            "ligand": df[ligand_col].astype(str),
            "receptor": df[receptor_col].astype(str),
            "sender": df[sender_col].astype(str),
            "receiver": df[receiver_col].astype(str),
            "score_raw": pd.to_numeric(df[score_col], errors="coerce").astype(float),
            "fdr_or_pvalue": pd.to_numeric(df[pvalue_col], errors="coerce").astype(float) if pvalue_col and pvalue_col in df.columns else np.nan,
            "resolution": resolution,
            "spatial_support_type": spatial_support_type,
            "artifact_path": str(artifact_path) if artifact_path is not None else "",
        }
    )
    for column in extra_numeric_cols:
        if column in df.columns:
            result[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    score_raw = result["score_raw"].fillna(result["score_raw"].min(skipna=True) if result["score_raw"].notna().any() else 0.0)
    rank, rank_fraction = _resolve_ordered_rank(score_raw, result["fdr_or_pvalue"])
    result["rank_within_method"] = rank.astype(float)
    result["rank_fraction"] = rank_fraction.astype(float)
    result["score_std"] = rank_fraction.astype(float)
    extra_columns = [col for col in result.columns if col not in STANDARDIZED_RESULT_COLUMNS]
    return result.loc[:, STANDARDIZED_RESULT_COLUMNS + extra_columns].sort_values("rank_within_method").reset_index(drop=True)


def run_pyxenium_smoke(
    *,
    input_h5ad: str | Path,
    output_dir: str | Path,
    tbc_results: str | Path,
    lr_panel_path: str | Path | None = None,
    database_mode: str = "common-db",
    export_figures: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tbc_results = _portable_path(tbc_results)

    adata = ad.read_h5ad(input_h5ad)
    lr_pairs = pd.read_csv(lr_panel_path, sep="\t") if lr_panel_path else DEFAULT_LR_SMOKE_PANEL.copy()
    if "evidence_weight" not in lr_pairs.columns:
        lr_pairs["evidence_weight"] = 1.0

    result = ligand_receptor_topology_analysis(
        adata=adata,
        lr_pairs=lr_pairs,
        output_dir=artifact_dir,
        tbc_results=tbc_results,
        cluster_col="cell_type",
        cell_id_col="cell_id",
        x_col="x",
        y_col="y",
        anchor_mode="precomputed",
        top_n_pairs=min(len(lr_pairs), 50),
        min_cross_edges=50,
        export_figures=export_figures,
        use_raw=False,
    )
    scores = result["scores"].copy()
    standardized = standardize_result_table(
        scores,
        method="pyxenium",
        database_mode=database_mode,
        score_col="LR_score",
        sender_col="sender_celltype",
        receiver_col="receiver_celltype",
        spatial_support_type="topology_local_contact",
        artifact_path=artifact_dir,
        extra_numeric_cols=("local_contact", "contact_strength_normalized", "contact_coverage", "cross_edge_count"),
    )

    standardized_path = output_dir / "pyxenium_standardized.tsv"
    scores_path = output_dir / "pyxenium_scores.tsv"
    scores.to_csv(scores_path, sep="\t", index=False)
    standardized.to_csv(standardized_path, sep="\t", index=False)

    payload = {
        "method": "pyxenium",
        "database_mode": database_mode,
        "input_h5ad": str(input_h5ad),
        "lr_panel_path": str(lr_panel_path) if lr_panel_path else None,
        "scores_tsv": str(scores_path),
        "standardized_tsv": str(standardized_path),
        "artifact_dir": str(artifact_dir),
        "n_rows": int(len(standardized)),
        "top_hit": standardized.head(1).to_dict(orient="records"),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["run_summary_json"] = str(summary_path)
    return payload


def aggregate_standardized_results(result_paths: Sequence[str | Path], *, output_path: str | Path | None = None) -> pd.DataFrame:
    frames = []
    extra_columns: list[str] = []
    for path in result_paths:
        table = pd.read_csv(path, sep="\t", compression="infer")
        missing = [col for col in STANDARDIZED_RESULT_COLUMNS if col not in table.columns]
        if missing:
            raise ValueError(f"Standardized result table {path} is missing columns: {missing}")
        for column in table.columns:
            if column not in STANDARDIZED_RESULT_COLUMNS and column not in extra_columns:
                extra_columns.append(column)
        frames.append(table.copy())
    ordered_columns = STANDARDIZED_RESULT_COLUMNS + extra_columns
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ordered_columns)
    combined = combined.reindex(columns=ordered_columns)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, sep="\t", index=False)
    return combined


def _load_axes(canonical_axes: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(canonical_axes, (str, Path)):
        payload = _load_yaml(canonical_axes)
        axes = payload.get("canonical_axes", [])
        return [dict(item) for item in axes]
    return [dict(item) for item in canonical_axes]


def compute_canonical_recovery(
    results: pd.DataFrame,
    *,
    canonical_axes: str | Path | Sequence[Mapping[str, Any]],
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = _load_axes(canonical_axes)
    detail_rows: list[dict[str, Any]] = []
    for (method, database_mode), group in results.groupby(["method", "database_mode"], dropna=False):
        for axis in axes:
            ligand = str(axis["ligand"])
            receptor = str(axis["receptor"])
            sender = str(axis["sender"])
            receiver = str(axis["receiver"])
            matches = group.loc[
                (group["ligand"] == ligand)
                & (group["receptor"] == receptor)
                & (group["sender"] == sender)
                & (group["receiver"] == receiver)
            ].sort_values("rank_within_method")
            best_rank = float(matches.iloc[0]["rank_within_method"]) if not matches.empty else np.nan
            detail_rows.append(
                {
                    "method": method,
                    "database_mode": database_mode,
                    "axis_name": axis.get("name", f"{ligand}-{receptor}"),
                    "ligand": ligand,
                    "receptor": receptor,
                    "sender": sender,
                    "receiver": receiver,
                    "matched": bool(not matches.empty),
                    "best_rank": best_rank,
                    "top_k_hit": bool(not matches.empty and best_rank <= float(top_k)),
                    "weight": float(axis.get("weight", 1.0)),
                }
            )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        summary = pd.DataFrame(columns=["method", "database_mode", "canonical_recovery_score"])
        return detail, summary

    detail["axis_score"] = np.where(
        detail["matched"],
        1.0 / np.log2(detail["best_rank"].fillna(float(top_k) + 1.0) + 1.0),
        0.0,
    )
    summary = (
        detail.assign(weighted_score=detail["axis_score"] * detail["weight"])
        .groupby(["method", "database_mode"], as_index=False)
        .agg(
            canonical_recovery_score=("weighted_score", "sum"),
            canonical_recovery_hits=("matched", "sum"),
            canonical_topk_hits=("top_k_hit", "sum"),
            canonical_axes_tested=("axis_name", "count"),
        )
    )
    max_score = float(summary["canonical_recovery_score"].max()) if not summary.empty else 0.0
    if max_score > 0:
        summary["canonical_recovery_score"] = summary["canonical_recovery_score"] / max_score
    return detail, summary


def _load_pathways(pathway_config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(pathway_config, (str, Path)):
        payload = _load_yaml(pathway_config)
        return dict(payload.get("pathways", {}))
    return dict(pathway_config)


def compute_pathway_relevance(
    results: pd.DataFrame,
    *,
    pathway_config: str | Path | Mapping[str, Any],
    top_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pathways = _load_pathways(pathway_config)
    rows: list[dict[str, Any]] = []
    for (method, database_mode), group in results.groupby(["method", "database_mode"], dropna=False):
        top = group.sort_values("rank_within_method").head(top_n).copy()
        pair_tokens = set((top["ligand"] + "^" + top["receptor"]).astype(str))
        gene_tokens = set(top["ligand"].astype(str)) | set(top["receptor"].astype(str))
        for pathway_name, spec in pathways.items():
            expected_pairs = {str(item) for item in spec.get("pairs", [])}
            expected_genes = {str(item) for item in spec.get("genes", [])}
            pair_hits = len(pair_tokens & expected_pairs)
            gene_hits = len(gene_tokens & expected_genes)
            rows.append(
                {
                    "method": method,
                    "database_mode": database_mode,
                    "pathway": pathway_name,
                    "pair_hits": pair_hits,
                    "gene_hits": gene_hits,
                    "score": pair_hits + 0.25 * gene_hits,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame(columns=["method", "database_mode", "pathway_relevance_score"])
        return detail, summary
    summary = detail.groupby(["method", "database_mode"], as_index=False).agg(
        pathway_relevance_score=("score", "sum"),
        pathway_hits=("pair_hits", "sum"),
    )
    max_score = float(summary["pathway_relevance_score"].max()) if not summary.empty else 0.0
    if max_score > 0:
        summary["pathway_relevance_score"] = summary["pathway_relevance_score"] / max_score
    return detail, summary


def compute_robustness(
    results: pd.DataFrame,
    *,
    repeat_col: str = "repeat_id",
    top_n: int = 50,
) -> pd.DataFrame:
    if repeat_col not in results.columns:
        return pd.DataFrame(columns=["method", "database_mode", "robustness_score", "repeat_pairs"])

    rows: list[dict[str, Any]] = []
    for (method, database_mode), group in results.groupby(["method", "database_mode"], dropna=False):
        repeat_sets: list[set[str]] = []
        for _, repeat_group in group.groupby(repeat_col):
            top = repeat_group.sort_values("rank_within_method").head(top_n)
            repeat_sets.append(set((top["ligand"] + "^" + top["receptor"] + "|" + top["sender"] + "|" + top["receiver"]).astype(str)))
        if len(repeat_sets) < 2:
            score = np.nan
        else:
            pair_scores = []
            for i in range(len(repeat_sets)):
                for j in range(i + 1, len(repeat_sets)):
                    union = repeat_sets[i] | repeat_sets[j]
                    pair_scores.append((len(repeat_sets[i] & repeat_sets[j]) / len(union)) if union else np.nan)
            score = float(np.nanmean(pair_scores)) if pair_scores else np.nan
        rows.append(
            {
                "method": method,
                "database_mode": database_mode,
                "robustness_score": score,
                "repeat_pairs": len(repeat_sets),
            }
        )
    return pd.DataFrame(rows)


def compute_spatial_coherence(results: pd.DataFrame, *, top_n: int = 100) -> pd.DataFrame:
    preferred_columns = [
        "local_contact",
        "contact_strength_normalized",
        "contact_coverage",
        "spatial_coherence",
    ]
    available = [column for column in preferred_columns if column in results.columns]
    if not available:
        return pd.DataFrame(columns=["method", "database_mode", "spatial_coherence_score", "spatial_metric_count"])

    rows: list[dict[str, Any]] = []
    for (method, database_mode), group in results.groupby(["method", "database_mode"], dropna=False):
        top = group.sort_values("rank_within_method").head(top_n)
        values = []
        for column in available:
            series = pd.to_numeric(top[column], errors="coerce")
            if series.notna().any():
                values.append(float(series.fillna(0.0).mean()))
        rows.append(
            {
                "method": method,
                "database_mode": database_mode,
                "spatial_coherence_score": float(np.mean(values)) if values else np.nan,
                "spatial_metric_count": len(values),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty and summary["spatial_coherence_score"].notna().any():
        max_score = float(summary["spatial_coherence_score"].max())
        if max_score > 0:
            summary["spatial_coherence_score"] = summary["spatial_coherence_score"] / max_score
    return summary


def compute_novelty_support(
    results: pd.DataFrame,
    *,
    top_n: int = 50,
    support_columns: Sequence[str] = ("spatial_supported", "literature_supported", "downstream_supported", "weak_consensus_supported"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top = results.sort_values("rank_within_method").groupby(["method", "database_mode"], group_keys=False).head(top_n).copy()
    if top.empty:
        empty = pd.DataFrame(columns=["method", "database_mode", "novelty_support_score"])
        return top, empty

    top["interaction_key"] = (
        top["ligand"].astype(str)
        + "^"
        + top["receptor"].astype(str)
        + "|"
        + top["sender"].astype(str)
        + "|"
        + top["receiver"].astype(str)
    )
    counts = top.groupby("interaction_key")["method"].nunique().rename("method_count")
    top = top.merge(counts, on="interaction_key", how="left")
    top["is_method_specific"] = top["method_count"] == 1
    available_support = [col for col in support_columns if col in top.columns]
    if available_support:
        top["support_count"] = top.loc[:, available_support].fillna(False).astype(bool).sum(axis=1)
    else:
        top["support_count"] = 0
    top["novelty_support_contribution"] = np.where(top["is_method_specific"], top["support_count"], 0.0)

    summary = top.groupby(["method", "database_mode"], as_index=False).agg(
        novelty_support_score=("novelty_support_contribution", "sum"),
        method_specific_hits=("is_method_specific", "sum"),
    )
    max_score = float(summary["novelty_support_score"].max()) if not summary.empty else 0.0
    if max_score > 0:
        summary["novelty_support_score"] = summary["novelty_support_score"] / max_score
    return top, summary


def score_biological_performance(
    *,
    canonical_summary: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    spatial_summary: pd.DataFrame | None = None,
    robustness_summary: pd.DataFrame | None = None,
    novelty_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pieces = [canonical_summary, pathway_summary]
    if spatial_summary is not None and not spatial_summary.empty:
        pieces.append(spatial_summary)
    if robustness_summary is not None and not robustness_summary.empty:
        pieces.append(robustness_summary)
    if novelty_summary is not None and not novelty_summary.empty:
        pieces.append(novelty_summary)

    if not pieces:
        return pd.DataFrame(columns=["method", "database_mode", "biology_score"])

    merged = pieces[0].copy()
    for item in pieces[1:]:
        merged = merged.merge(item, on=["method", "database_mode"], how="outer")

    for column in (
        "canonical_recovery_score",
        "pathway_relevance_score",
        "spatial_coherence_score",
        "robustness_score",
        "novelty_support_score",
    ):
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged["biology_score"] = (
        0.30 * merged["canonical_recovery_score"]
        + 0.25 * merged["spatial_coherence_score"]
        + 0.20 * merged["pathway_relevance_score"]
        + 0.15 * merged["robustness_score"]
        + 0.10 * merged["novelty_support_score"]
    )
    return merged.sort_values("biology_score", ascending=False).reset_index(drop=True)


def build_canonical_rank_matrix(canonical_detail: pd.DataFrame) -> pd.DataFrame:
    if canonical_detail.empty:
        return pd.DataFrame()
    required = {"method", "database_mode", "axis_name", "best_rank"}
    if not required.issubset(canonical_detail.columns):
        return pd.DataFrame()
    matrix = canonical_detail.copy()
    matrix["method_db"] = matrix["method"].astype(str) + " / " + matrix["database_mode"].astype(str)
    pivot = matrix.pivot_table(index="axis_name", columns="method_db", values="best_rank", aggfunc="min")
    return pivot.reset_index()


def render_atera_lr_benchmark_report(
    *,
    combined_results: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    biology_summary: pd.DataFrame,
    benchmark_root: str | Path,
    run_status: pd.DataFrame | None = None,
    engineering_summary: pd.DataFrame | None = None,
    canonical_detail: pd.DataFrame | None = None,
    a100_resource_summary: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Atera Xenium LR Benchmark",
        "",
        f"- Benchmark root: `{Path(benchmark_root)}`",
        f"- Methods with standardized results: `{combined_results['method'].nunique() if not combined_results.empty else 0}`",
        f"- Total standardized rows: `{len(combined_results)}`",
        "",
        "## Biology Scoreboard",
    ]

    if biology_summary.empty:
        lines.append("")
        lines.append("No biology score summary is available yet.")
    else:
        lines.extend(["", biology_summary.to_markdown(index=False)])

    if run_status is not None and not run_status.empty:
        lines.append("")
        lines.append("## Run Status")
        status_cols = [
            col
            for col in [
                "method",
                "phase",
                "database_mode",
                "status",
                "n_rows",
                "elapsed_seconds",
                "peak_memory_gb",
                "error",
            ]
            if col in run_status.columns
        ]
        lines.extend(["", run_status.loc[:, status_cols].to_markdown(index=False)])

    if engineering_summary is not None and not engineering_summary.empty:
        lines.append("")
        lines.append("## Engineering Reproducibility")
        lines.extend(["", engineering_summary.to_markdown(index=False)])

    lines.append("")
    lines.append("## Canonical Recovery")
    if canonical_summary.empty:
        lines.extend(["", "No canonical recovery summary is available yet."])
    else:
        lines.extend(["", canonical_summary.to_markdown(index=False)])

    if canonical_detail is not None and not canonical_detail.empty:
        rank_matrix = build_canonical_rank_matrix(canonical_detail)
        if not rank_matrix.empty:
            lines.append("")
            lines.append("## Canonical Pair Rank Matrix")
            lines.extend(["", rank_matrix.to_markdown(index=False)])

    lines.append("")
    lines.append("## Pathway Relevance")
    if pathway_summary.empty:
        lines.extend(["", "No pathway relevance summary is available yet."])
    else:
        lines.extend(["", pathway_summary.to_markdown(index=False)])

    if "spatial_coherence_score" in biology_summary.columns:
        spatial_view = biology_summary.loc[:, [col for col in ["method", "database_mode", "spatial_coherence_score"] if col in biology_summary.columns]]
        lines.append("")
        lines.append("## Spatial Coherence")
        lines.extend(["", spatial_view.to_markdown(index=False)])

    if not combined_results.empty:
        top_hits = (
            combined_results.sort_values("rank_within_method")
            .groupby(["method", "database_mode"], as_index=False)
            .first()[["method", "database_mode", "ligand", "receptor", "sender", "receiver", "score_raw"]]
        )
        lines.append("")
        lines.append("## Top Discovery per Method")
        lines.extend(["", top_hits.to_markdown(index=False)])

    if a100_resource_summary is not None and not a100_resource_summary.empty:
        lines.append("")
        lines.append("## A100 Resource Summary")
        lines.extend(["", a100_resource_summary.to_markdown(index=False)])

    return "\n".join(lines) + "\n"


def build_stage_manifest(
    *,
    local_root: str | Path,
    remote_root: str | Path,
    host: str,
    user: str,
    include_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    local_root = Path(local_root).resolve()
    staged = [str(Path(path).resolve()) for path in (include_paths or [local_root / "data", local_root / "configs", local_root / "envs", local_root / "scripts", local_root / "runners"])]
    remote = Path(remote_root)
    mkdir_command = f"ssh {user}@{host} \"mkdir -p {remote.as_posix()}/data {remote.as_posix()}/envs {remote.as_posix()}/runs {remote.as_posix()}/results {remote.as_posix()}/logs\""
    copy_commands = [
        f"scp -r \"{path}\" {user}@{host}:\"{remote.as_posix()}/\""
        for path in staged
    ]
    return {
        "local_root": str(local_root),
        "remote_root": str(remote),
        "host": host,
        "user": user,
        "include_paths": staged,
        "mkdir_command": mkdir_command,
        "copy_commands": copy_commands,
    }


def write_stage_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def run_subprocess(command: Sequence[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def python_executable() -> str:
    return sys.executable
