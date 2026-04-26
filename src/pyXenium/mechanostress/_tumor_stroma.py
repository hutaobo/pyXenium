from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.stats import spearmanr


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_label(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def _build_delaunay_neighbors(coords: np.ndarray) -> list[set[int]]:
    n_points = int(coords.shape[0])
    neighbors = [set() for _ in range(n_points)]
    if n_points < 2:
        return neighbors
    if n_points == 2:
        neighbors[0].add(1)
        neighbors[1].add(0)
        return neighbors

    try:
        triangulation = Delaunay(coords)
    except QhullError:
        # Degenerate collinear examples still need deterministic behavior.
        order = np.argsort(coords[:, 0] + coords[:, 1] * 1e-9)
        for left, right in zip(order[:-1], order[1:]):
            neighbors[int(left)].add(int(right))
            neighbors[int(right)].add(int(left))
        return neighbors

    for simplex in triangulation.simplices:
        for a in range(len(simplex)):
            for b in range(a + 1, len(simplex)):
                i = int(simplex[a])
                j = int(simplex[b])
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def _nearest_distance_to_group(coords: np.ndarray, query_indices: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    distances = np.full(len(query_indices), np.nan, dtype=float)
    if len(query_indices) == 0 or len(target_indices) == 0:
        return distances
    tree = cKDTree(coords[target_indices])
    distances, _ = tree.query(coords[query_indices], k=1)
    return distances.astype(float)


def _nearest_tumor_distance(coords: np.ndarray, tumor_indices: np.ndarray) -> np.ndarray:
    distances = np.full(len(tumor_indices), np.nan, dtype=float)
    if len(tumor_indices) <= 1:
        return distances
    tree = cKDTree(coords[tumor_indices])
    query_distances, _ = tree.query(coords[tumor_indices], k=min(2, len(tumor_indices)))
    if query_distances.ndim == 1:
        return distances
    return query_distances[:, 1].astype(float)


def _classify_delaunay_hop(
    frame: pd.DataFrame,
    *,
    annotation_col: str,
    tumor_label: str,
    stroma_label: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    normalized = _normalize_label(frame[annotation_col])
    tumor_key = str(tumor_label).strip().lower()
    stroma_key = str(stroma_label).strip().lower()
    valid_mask = normalized.isin([tumor_key, stroma_key]).to_numpy()
    valid = frame.loc[valid_mask].copy()
    if valid.empty:
        out = frame.copy()
        out["stroma_hop"] = np.nan
        out["tumor_growth_pattern"] = pd.Series([np.nan] * len(out), index=out.index, dtype="object")
        out["dist_to_nearest_stromal"] = np.nan
        out["dist_to_nearest_tumor"] = np.nan
        return out

    coords = valid[[x_col, y_col]].to_numpy(dtype=float)
    labels = _normalize_label(valid[annotation_col])
    tumor_nodes = np.where(labels.to_numpy() == tumor_key)[0]
    stromal_nodes = np.where(labels.to_numpy() == stroma_key)[0]
    neighbors = _build_delaunay_neighbors(coords)

    inf = 10**9
    hop = np.full(len(valid), inf, dtype=int)
    queue: deque[int] = deque()
    for node in stromal_nodes:
        hop[int(node)] = 0
        queue.append(int(node))
    while queue:
        node = queue.popleft()
        for neighbor in neighbors[node]:
            if hop[neighbor] == inf:
                hop[neighbor] = hop[node] + 1
                queue.append(neighbor)

    pattern = np.full(len(valid), "", dtype=object)
    pattern[stromal_nodes] = "stromal"
    for node in tumor_nodes:
        if hop[node] == 1:
            pattern[node] = "infiltrative"
        elif 1 < hop[node] < inf:
            pattern[node] = "expanding"
        else:
            pattern[node] = "no_stromal_neighbor"

    stromal_distance = _nearest_distance_to_group(coords, tumor_nodes, stromal_nodes)
    tumor_distance = _nearest_tumor_distance(coords, tumor_nodes)
    dist_to_stromal_valid = np.full(len(valid), np.nan, dtype=float)
    dist_to_tumor_valid = np.full(len(valid), np.nan, dtype=float)
    dist_to_stromal_valid[tumor_nodes] = stromal_distance
    dist_to_tumor_valid[tumor_nodes] = tumor_distance

    out = frame.copy()
    out["stroma_hop"] = np.nan
    out["tumor_growth_pattern"] = pd.Series([np.nan] * len(out), index=out.index, dtype="object")
    out["dist_to_nearest_stromal"] = np.nan
    out["dist_to_nearest_tumor"] = np.nan
    original_index = valid.index
    out.loc[original_index, "stroma_hop"] = np.where(hop == inf, np.nan, hop).astype(float)
    out.loc[original_index, "tumor_growth_pattern"] = pattern
    out.loc[original_index, "dist_to_nearest_stromal"] = dist_to_stromal_valid
    out.loc[original_index, "dist_to_nearest_tumor"] = dist_to_tumor_valid
    return out


def _classify_nearest_distance_ratio(
    frame: pd.DataFrame,
    *,
    annotation_col: str,
    tumor_label: str,
    stroma_label: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    normalized = _normalize_label(frame[annotation_col])
    tumor_key = str(tumor_label).strip().lower()
    stroma_key = str(stroma_label).strip().lower()
    valid_mask = normalized.isin([tumor_key, stroma_key]).to_numpy()
    valid = frame.loc[valid_mask].copy()
    coords = valid[[x_col, y_col]].to_numpy(dtype=float)
    labels = _normalize_label(valid[annotation_col]).to_numpy()
    tumor_nodes = np.where(labels == tumor_key)[0]
    stromal_nodes = np.where(labels == stroma_key)[0]

    dist_to_stromal = np.full(len(valid), np.nan, dtype=float)
    dist_to_tumor = np.full(len(valid), np.nan, dtype=float)
    dist_to_stromal[tumor_nodes] = _nearest_distance_to_group(coords, tumor_nodes, stromal_nodes)
    dist_to_tumor[tumor_nodes] = _nearest_tumor_distance(coords, tumor_nodes)

    pattern = np.full(len(valid), "", dtype=object)
    pattern[stromal_nodes] = "stromal"
    for position, node in enumerate(tumor_nodes):
        stromal_distance = dist_to_stromal[node]
        tumor_distance = dist_to_tumor[node]
        if not np.isfinite(stromal_distance):
            pattern[node] = "no_stromal_neighbor"
        elif not np.isfinite(tumor_distance) or stromal_distance <= tumor_distance:
            pattern[node] = "infiltrative"
        else:
            pattern[node] = "expanding"

    out = frame.copy()
    out["stroma_hop"] = np.nan
    out["tumor_growth_pattern"] = pd.Series([np.nan] * len(out), index=out.index, dtype="object")
    out["dist_to_nearest_stromal"] = np.nan
    out["dist_to_nearest_tumor"] = np.nan
    out["tumor_stroma_distance_ratio"] = np.nan
    original_index = valid.index
    out.loc[original_index, "tumor_growth_pattern"] = pattern
    out.loc[original_index, "dist_to_nearest_stromal"] = dist_to_stromal
    out.loc[original_index, "dist_to_nearest_tumor"] = dist_to_tumor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dist_to_stromal / dist_to_tumor
    out.loc[original_index, "tumor_stroma_distance_ratio"] = ratio
    return out


def classify_tumor_stroma_growth(
    cell_table: pd.DataFrame,
    *,
    annotation_col: str = "Annotation",
    tumor_label: str = "Tumor",
    stroma_label: str = "Stromal",
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    method: str = "delaunay_hop",
) -> pd.DataFrame:
    """Classify tumor cells as infiltrative or expanding from tumor-stroma geometry."""

    _require_columns(cell_table, [annotation_col, x_col, y_col], name="cell_table")
    frame = cell_table.copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame.dropna(subset=[x_col, y_col]).copy()

    if method == "delaunay_hop":
        return _classify_delaunay_hop(
            frame,
            annotation_col=annotation_col,
            tumor_label=tumor_label,
            stroma_label=stroma_label,
            x_col=x_col,
            y_col=y_col,
        )
    if method == "nearest_distance_ratio":
        return _classify_nearest_distance_ratio(
            frame,
            annotation_col=annotation_col,
            tumor_label=tumor_label,
            stroma_label=stroma_label,
            x_col=x_col,
            y_col=y_col,
        )
    raise ValueError("method must be either 'delaunay_hop' or 'nearest_distance_ratio'.")


def summarize_tumor_growth(
    growth_table: pd.DataFrame,
    *,
    annotation_col: str = "Annotation",
    tumor_label: str = "Tumor",
    sample_id: str | None = None,
) -> pd.DataFrame:
    """Summarize infiltrative and expanding tumor fractions."""

    _require_columns(growth_table, [annotation_col, "tumor_growth_pattern"], name="growth_table")
    tumor_mask = _normalize_label(growth_table[annotation_col]) == str(tumor_label).strip().lower()
    tumor = growth_table.loc[tumor_mask].copy()
    valid = tumor.loc[tumor["tumor_growth_pattern"].isin(["infiltrative", "expanding"])]
    n_infiltrative = int((valid["tumor_growth_pattern"] == "infiltrative").sum())
    n_expanding = int((valid["tumor_growth_pattern"] == "expanding").sum())
    denominator = n_infiltrative + n_expanding
    record = {
        "sample_id": sample_id,
        "n_tumor": int(len(tumor)),
        "n_infiltrative": n_infiltrative,
        "n_expanding": n_expanding,
        "n_tumor_valid": int(denominator),
        "infiltrative_proportion": float(n_infiltrative / denominator) if denominator else np.nan,
        "mean_infil_dist_to_stromal": float(
            valid.loc[valid["tumor_growth_pattern"] == "infiltrative", "dist_to_nearest_stromal"].mean()
        )
        if "dist_to_nearest_stromal" in valid.columns
        else np.nan,
        "mean_expand_dist_to_stromal": float(
            valid.loc[valid["tumor_growth_pattern"] == "expanding", "dist_to_nearest_stromal"].mean()
        )
        if "dist_to_nearest_stromal" in valid.columns
        else np.nan,
    }
    return pd.DataFrame([record])


def _expression_from_adata(adata: Any, *, cell_ids: Sequence[str], genes: Sequence[str]) -> pd.DataFrame:
    present_genes = [gene for gene in genes if gene in adata.var_names]
    if not present_genes:
        return pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs_names = set(adata.obs_names.astype(str))
    available_ids = [str(cell_id) for cell_id in cell_ids if str(cell_id) in obs_names]
    if not available_ids:
        return pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"), columns=present_genes)
    subset = adata[available_ids, present_genes]
    matrix = subset.X
    if hasattr(matrix, "toarray"):
        values = matrix.toarray()
    else:
        values = np.asarray(matrix)
    return pd.DataFrame(values, index=subset.obs_names.astype(str), columns=present_genes)


def compute_distance_expression_coupling(
    *,
    growth_table: pd.DataFrame,
    expression_df: pd.DataFrame | None = None,
    adata: Any = None,
    genes: Iterable[str] | None = None,
    tumor_enriched_genes: Iterable[str] | None = None,
    annotation_col: str = "Annotation",
    tumor_label: str = "Tumor",
    cell_id_col: str = "cell_id",
    distance_col: str = "dist_to_nearest_stromal",
    min_cells: int = 3,
    min_nonzero_cells: int = 3,
) -> pd.DataFrame:
    """Compute gene-wise Spearman coupling to tumor-stromal distance."""

    _require_columns(growth_table, [annotation_col, cell_id_col, distance_col], name="growth_table")
    if expression_df is None and adata is None:
        raise ValueError("Either expression_df or adata must be provided.")

    tumor_mask = _normalize_label(growth_table[annotation_col]) == str(tumor_label).strip().lower()
    tumor = growth_table.loc[tumor_mask].copy()
    tumor[cell_id_col] = tumor[cell_id_col].astype(str)
    tumor[distance_col] = pd.to_numeric(tumor[distance_col], errors="coerce")
    tumor = tumor.dropna(subset=[distance_col])
    requested_genes = genes if genes is not None else tumor_enriched_genes
    if requested_genes is None and expression_df is not None:
        gene_list = expression_df.columns.astype(str).tolist()
    elif requested_genes is None and adata is not None:
        gene_list = adata.var_names.astype(str).tolist()
    else:
        gene_list = [str(gene) for gene in (requested_genes or [])]

    if expression_df is None:
        expression = _expression_from_adata(adata, cell_ids=tumor[cell_id_col].tolist(), genes=gene_list)
    else:
        expression = expression_df.copy()
        expression.index = expression.index.astype(str)
        if gene_list:
            gene_list = [gene for gene in gene_list if gene in expression.columns]
            expression = expression.loc[:, gene_list]
        expression = expression.reindex(tumor[cell_id_col].astype(str))

    enriched_set = {str(gene) for gene in (tumor_enriched_genes or [])}
    records: list[dict[str, Any]] = []
    distances = tumor.set_index(cell_id_col).reindex(expression.index)[distance_col]
    for gene in expression.columns.astype(str):
        values = pd.to_numeric(expression[gene], errors="coerce")
        valid = values.notna() & distances.notna()
        valid &= np.isfinite(values.to_numpy(dtype=float)) & np.isfinite(distances.to_numpy(dtype=float))
        n_cells = int(valid.sum())
        n_nonzero = int((values.loc[valid] > 0).sum()) if n_cells else 0
        if n_cells < int(min_cells) or n_nonzero < int(min_nonzero_cells):
            rho = p_value = np.nan
        else:
            rho, p_value = spearmanr(values.loc[valid].to_numpy(dtype=float), distances.loc[valid].to_numpy(dtype=float))
        records.append(
            {
                "gene": str(gene),
                "n_cells": n_cells,
                "n_nonzero_cells": n_nonzero,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "mean_expression": float(values.loc[valid].mean()) if n_cells else np.nan,
                "tumor_enriched": bool(str(gene) in enriched_set) if enriched_set else False,
            }
        )
    return pd.DataFrame.from_records(records)
