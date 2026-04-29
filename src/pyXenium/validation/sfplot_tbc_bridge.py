from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from pyXenium._topology_core import (
    compute_cophenetic_from_distance_matrix,
    compute_weighted_searcher_findee_distance_matrix_from_df,
)
from pyXenium.io.xenium_artifacts import iter_transcript_chunks, resolve_transcripts_path

__all__ = ["run_sfplot_tbc_table_bundle"]

_WORKER_CELL_GROUP_MEAN: pd.DataFrame | None = None
_WORKER_CELL_GROUP_MATRIX: np.ndarray | None = None
_WORKER_CLUSTER_LABELS: list[str] | None = None
_WORKER_METHOD: str = "average"


def _clear_sfplot_modules() -> None:
    for name in list(sys.modules):
        if name == "sfplot" or name.startswith("sfplot."):
            del sys.modules[name]


def _candidate_sfplot_src_roots(sfplot_root: str | os.PathLike[str] | None) -> list[Path]:
    candidates: list[Path] = []
    explicit = Path(sfplot_root).expanduser().resolve() if sfplot_root is not None else None
    env_root = os.environ.get("SFPLOT_ROOT")
    repo_root = Path(__file__).resolve().parents[3]

    for root in (
        explicit,
        Path(env_root).expanduser().resolve() if env_root else None,
        repo_root.parent / "sfplot",
    ):
        if root is None:
            continue
        if root.name == "src" and (root / "sfplot").exists():
            candidates.append(root)
            continue
        src_root = root / "src"
        if src_root.exists() and (src_root / "sfplot").exists():
            candidates.append(src_root)

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _load_sfplot_public_api(
    *, sfplot_root: str | os.PathLike[str] | None = None
) -> dict[str, Callable[..., Any]]:
    last_error: Exception | None = None

    for candidate in [None, *_candidate_sfplot_src_roots(sfplot_root)]:
        try:
            _clear_sfplot_modules()
            if candidate is not None:
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)

            import sfplot
            from sfplot.Searcher_Findee_Score import (
                compute_searcher_findee_distance_matrix_from_df,
                plot_cophenetic_heatmap,
            )

            return {
                "load_xenium_table_bundle": sfplot.load_xenium_table_bundle,
                "compute_searcher_findee_distance_matrix_from_df": compute_searcher_findee_distance_matrix_from_df,
                "plot_cophenetic_heatmap": plot_cophenetic_heatmap,
            }
        except Exception as exc:  # pragma: no cover - exercised via fallback path
            last_error = exc

    raise ImportError(
        "Unable to import a usable sfplot installation. Pass `sfplot_root` or set `SFPLOT_ROOT` "
        "to a local checkout that contains `src/sfplot`."
    ) from last_error


def _prepare_obs_df(adata, group_df: pd.DataFrame | None) -> pd.DataFrame:
    obs = adata.obs.copy().reset_index(drop=True)
    obs["cell_id"] = obs["cell_id"].astype(str) if "cell_id" in obs.columns else adata.obs_names.astype(str)
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    obs["x"] = spatial[:, 0]
    obs["y"] = spatial[:, 1]

    if group_df is not None:
        required = {"cell_id", "group"}
        missing = required.difference(group_df.columns)
        if missing:
            raise ValueError(f"`df` is missing required columns: {sorted(missing)}")
        labels = group_df.loc[:, ["cell_id", "group"]].copy()
        labels["cell_id"] = labels["cell_id"].astype(str)
        obs = obs.merge(labels, on="cell_id", how="left")
        celltype_col = "group"
    elif "Cluster" in obs.columns:
        celltype_col = "Cluster"
    elif "cluster" in obs.columns:
        celltype_col = "cluster"
    else:
        raise ValueError("Unable to resolve cluster labels from `adata.obs`.")

    prepared = (
        obs.loc[:, ["cell_id", "x", "y", celltype_col]]
        .rename(columns={celltype_col: "celltype"})
        .dropna(subset=["celltype"])
        .copy()
    )
    prepared["celltype"] = prepared["celltype"].astype(str)
    return prepared.reset_index(drop=True)


def _build_cluster_models(obs_df: pd.DataFrame, cluster_labels: list[str]) -> list[cKDTree]:
    models: list[cKDTree] = []
    for label in cluster_labels:
        coords = obs_df.loc[obs_df["celltype"] == label, ["x", "y"]].to_numpy(dtype=float)
        if coords.shape[0] == 0:
            raise ValueError(f"No cells were found for cluster {label!r}.")
        models.append(cKDTree(coords))
    return models


def _resolve_gene_names(adata) -> list[str]:
    return [str(gene) for gene in adata.var_names.astype(str).tolist()]


def _accumulate_gene_to_cluster_means(
    *,
    transcripts_path: str,
    gene_names: list[str],
    cluster_models: list[cKDTree],
) -> tuple[np.ndarray, np.ndarray]:
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}
    gene_filter = set(gene_names)
    sums = np.zeros((len(gene_names), len(cluster_models)), dtype=np.float64)
    counts = np.zeros(len(gene_names), dtype=np.int64)

    for chunk in iter_transcript_chunks(transcripts_path, genes=gene_filter):
        working = chunk.loc[:, ["x", "y", "gene_name"]].copy()
        working["gene_name"] = working["gene_name"].astype(str)
        working = working.loc[
            ~working["gene_name"].str.contains("NegControl|Unassigned", na=False)
        ].reset_index(drop=True)
        if working.empty:
            continue

        gene_indices = np.fromiter(
            (gene_to_index.get(gene_name, -1) for gene_name in working["gene_name"]),
            dtype=np.int64,
            count=len(working),
        )
        valid = gene_indices >= 0
        if not np.any(valid):
            continue

        coords = working.loc[valid, ["x", "y"]].to_numpy(dtype=float)
        gene_indices = gene_indices[valid]
        np.add.at(counts, gene_indices, 1)

        for cluster_idx, model in enumerate(cluster_models):
            distances, _ = model.query(coords, k=1, workers=-1)
            np.add.at(sums[:, cluster_idx], gene_indices, distances)

    return sums, counts


def _fallback_tbc_row(
    gene: str,
    *,
    cluster_labels: list[str],
    row_cophenetic: pd.DataFrame,
) -> np.ndarray:
    if gene in row_cophenetic.index:
        return row_cophenetic.loc[gene].reindex(cluster_labels).to_numpy(dtype=float)
    return np.full(len(cluster_labels), np.nan, dtype=float)


def _compute_tbc_row_from_means(
    gene: str,
    mean_vector: np.ndarray,
    *,
    cell_group_mean: pd.DataFrame | None = None,
    cell_group_matrix: np.ndarray | None = None,
    cluster_labels: list[str] | None = None,
    method: str,
) -> np.ndarray:
    del gene
    matrix = (
        np.asarray(cell_group_matrix, dtype=float)
        if cell_group_matrix is not None
        else np.asarray(cell_group_mean.loc[:, cluster_labels], dtype=float)
    )
    gene_row = np.asarray(mean_vector, dtype=float).reshape(1, -1)
    augmented = np.vstack([matrix, gene_row])
    if augmented.shape[0] == 1:
        return np.zeros(1, dtype=float)

    row_linkage = linkage(augmented, method=method)
    _, row_condensed = cophenet(row_linkage, pdist(augmented))
    row_square = squareform(row_condensed)
    max_value = float(np.nanmax(row_square)) if row_square.size else 0.0
    if not np.isfinite(max_value) or max_value <= 0.0:
        normalized = np.zeros_like(row_square, dtype=float)
    else:
        normalized = row_square / max_value
    return normalized[-1, :-1].astype(float, copy=False)


def _init_tbc_worker(cell_group_mean: pd.DataFrame, cluster_labels: list[str], method: str) -> None:
    global _WORKER_CELL_GROUP_MEAN, _WORKER_CELL_GROUP_MATRIX, _WORKER_CLUSTER_LABELS, _WORKER_METHOD
    _WORKER_CELL_GROUP_MEAN = cell_group_mean
    _WORKER_CELL_GROUP_MATRIX = np.asarray(cell_group_mean.loc[:, cluster_labels], dtype=float)
    _WORKER_CLUSTER_LABELS = list(cluster_labels)
    _WORKER_METHOD = method


def _compute_tbc_row_worker(task: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    gene, mean_vector = task
    if _WORKER_CELL_GROUP_MATRIX is None or _WORKER_CLUSTER_LABELS is None:  # pragma: no cover
        raise RuntimeError("t_and_c worker was not initialized.")
    row = _compute_tbc_row_from_means(
        gene,
        mean_vector,
        cell_group_matrix=_WORKER_CELL_GROUP_MATRIX,
        cluster_labels=_WORKER_CLUSTER_LABELS,
        method=_WORKER_METHOD,
    )
    return gene, row


def run_sfplot_tbc_table_bundle(
    folder: str | os.PathLike[str],
    *,
    sample_name: str | None = None,
    output_folder: str | os.PathLike[str] | None = None,
    coph_method: str = "average",
    n_jobs: int = 8,
    maxtasks: int = 50,
    df: pd.DataFrame | None = None,
    gene_batch_size: int = 128,
    sfplot_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """
    Stable Xenium `t_and_c` generation that preserves the breast workflow output contract.

    The helper prefers sfplot's table-bundle loader while avoiding the legacy
    `spatialdata_io` dependency chain that breaks on some reviewer environments.
    """

    if gene_batch_size <= 0:
        raise ValueError("`gene_batch_size` must be greater than 0.")

    sfplot_api = _load_sfplot_public_api(sfplot_root=sfplot_root)
    dataset_dir = Path(folder).expanduser().resolve()
    sample = sample_name or dataset_dir.name
    out_dir = Path(output_folder).expanduser().resolve() if output_folder is not None else dataset_dir / "sfplot_tbc_formal_wta" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sfplot_api["load_xenium_table_bundle"](str(dataset_dir), normalize=False)
    obs_df = _prepare_obs_df(adata, df)

    cell_group_mean = sfplot_api["compute_searcher_findee_distance_matrix_from_df"](
        obs_df,
        x_col="x",
        y_col="y",
        celltype_col="celltype",
    )
    row_cophenetic, _ = compute_cophenetic_from_distance_matrix(
        cell_group_mean,
        method=coph_method,
        show_corr=False,
    )
    cluster_labels = row_cophenetic.columns.astype(str).tolist()

    structure_map_pdf = out_dir / f"StructureMap_of_{sample}.pdf"
    structure_map_csv = out_dir / f"StructureMap_table_{sample}.csv"
    t_and_c_csv = out_dir / f"t_and_c_result_{sample}.csv"

    sfplot_api["plot_cophenetic_heatmap"](
        row_cophenetic,
        matrix_name="row_coph",
        output_dir=str(out_dir),
        output_filename=structure_map_pdf.name,
        sample=sample,
    )
    row_cophenetic.to_csv(structure_map_csv)

    transcripts_path = resolve_transcripts_path(str(dataset_dir))
    if transcripts_path is None:
        raise FileNotFoundError(f"Could not resolve a Xenium transcripts Zarr path under {dataset_dir}")

    gene_names = _resolve_gene_names(adata)
    cluster_models = _build_cluster_models(obs_df, cluster_labels)
    distance_sums, transcript_counts = _accumulate_gene_to_cluster_means(
        transcripts_path=transcripts_path,
        gene_names=gene_names,
        cluster_models=cluster_models,
    )

    row_lookup: dict[str, np.ndarray] = {}
    executor: ThreadPoolExecutor | None = None
    try:
        if int(n_jobs) > 1:
            _init_tbc_worker(cell_group_mean.loc[:, cluster_labels], cluster_labels, coph_method)
            executor = ThreadPoolExecutor(max_workers=int(n_jobs))

        with t_and_c_csv.open("w", newline="", encoding="utf-8") as handle:
            header_written = False
            for start in range(0, len(gene_names), int(gene_batch_size)):
                batch_genes = gene_names[start : start + int(gene_batch_size)]
                worker_tasks: list[tuple[str, np.ndarray]] = []

                for gene_index, gene in enumerate(batch_genes, start=start):
                    if transcript_counts[gene_index] <= 0:
                        row_lookup[gene] = _fallback_tbc_row(
                            gene,
                            cluster_labels=cluster_labels,
                            row_cophenetic=row_cophenetic,
                        )
                        continue
                    mean_vector = distance_sums[gene_index] / float(transcript_counts[gene_index])
                    worker_tasks.append((gene, mean_vector.astype(float, copy=False)))

                if worker_tasks:
                    if executor is None:
                        for gene, mean_vector in worker_tasks:
                            row_lookup[gene] = _compute_tbc_row_from_means(
                                gene,
                                mean_vector,
                                cell_group_mean=cell_group_mean.loc[:, cluster_labels],
                                cluster_labels=cluster_labels,
                                method=coph_method,
                            )
                    else:
                        for gene, row in executor.map(_compute_tbc_row_worker, worker_tasks):
                            row_lookup[gene] = row

                batch_rows = [
                    pd.Series(row_lookup[gene], index=cluster_labels, name=gene, dtype=float)
                    for gene in batch_genes
                ]
                batch_df = pd.DataFrame(batch_rows, columns=cluster_labels)
                batch_df.to_csv(handle, header=not header_written, index=True)
                header_written = True
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return {
        "sample_name": sample,
        "output_dir": str(out_dir),
        "structure_map_pdf": str(structure_map_pdf),
        "structure_map_table": str(structure_map_csv),
        "t_and_c_result": str(t_and_c_csv),
        "transcripts_source": str(transcripts_path),
        "n_celltypes": int(len(cluster_labels)),
        "n_genes": int(len(gene_names)),
    }
