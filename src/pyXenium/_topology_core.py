from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors


def ensure_output_dir(output_dir: Optional[str | Path]) -> Optional[Path]:
    if output_dir is None:
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_figures_dir(output_dir: Optional[str | Path]) -> Optional[Path]:
    out = ensure_output_dir(output_dir)
    if out is None:
        return None
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _coerce_nonnegative(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    min_value = float(numeric.min()) if len(numeric) else 0.0
    if min_value < 0.0:
        numeric = numeric - min_value
    return numeric.clip(lower=0.0)


def normalize_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(float)
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        fill_value = 0.0 if math.isclose(max_value, 0.0) else 1.0
        return pd.Series(fill_value, index=numeric.index, dtype=float)
    return (numeric - min_value) / (max_value - min_value)


def normalize_frame_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    row_min = numeric.min(axis=1)
    row_max = numeric.max(axis=1)
    span = (row_max - row_min).replace(0.0, np.nan)
    normalized = numeric.sub(row_min, axis=0).div(span, axis=0).fillna(0.0)
    constant_nonzero = span.isna() & (row_max > 0.0)
    if constant_nonzero.any():
        normalized.loc[constant_nonzero] = 1.0
    return normalized


def normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    col_min = numeric.min(axis=0)
    col_max = numeric.max(axis=0)
    span = (col_max - col_min).replace(0.0, np.nan)
    normalized = numeric.sub(col_min, axis=1).div(span, axis=1).fillna(0.0)
    constant_nonzero = span.isna() & (col_max > 0.0)
    for column in numeric.columns[constant_nonzero]:
        normalized[column] = 1.0
    return normalized


def winsorized_minmax(values: np.ndarray, upper_quantile: float = 0.99) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(array, array.flat[0]):
        return np.zeros_like(array) if math.isclose(float(array.flat[0]), 0.0) else np.ones_like(array)
    clipped = np.clip(array, a_min=float(np.min(array)), a_max=float(np.quantile(array, upper_quantile)))
    min_value = float(np.min(clipped))
    max_value = float(np.max(clipped))
    if math.isclose(min_value, max_value):
        return np.zeros_like(clipped) if math.isclose(max_value, 0.0) else np.ones_like(clipped)
    return (clipped - min_value) / (max_value - min_value)


def winsorized_normalize_series(values: pd.Series, upper_quantile: float = 0.99) -> pd.Series:
    if values.empty:
        return values.astype(float)
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(
        winsorized_minmax(numeric.to_numpy(dtype=float), upper_quantile=upper_quantile),
        index=numeric.index,
        dtype=float,
    )


def winsorized_normalize_frame(frame: pd.DataFrame, upper_quantile: float = 0.99) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalized = frame.copy().astype(float)
    for column in normalized.columns:
        normalized[column] = winsorized_normalize_series(normalized[column], upper_quantile=upper_quantile)
    return normalized


def robust_scale_columns(
    frame: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    scaled = frame.copy().astype(float)
    for column in scaled.columns:
        values = scaled[column].to_numpy(dtype=float)
        if values.size == 0:
            continue
        lower = float(np.quantile(values, lower_quantile))
        upper = float(np.quantile(values, upper_quantile))
        if math.isclose(lower, upper):
            scaled[column] = 0.0 if math.isclose(upper, 0.0) else 1.0
            continue
        clipped = np.clip(values, lower, upper)
        scaled[column] = (clipped - lower) / (upper - lower)
    return scaled.fillna(0.0)


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any():
        return float("nan")
    values_valid = np.asarray(values[valid], dtype=float)
    weights_valid = np.clip(np.asarray(weights[valid], dtype=float), 0.0, None)
    if np.allclose(weights_valid.sum(), 0.0):
        return float(np.mean(values_valid))
    return float(np.average(values_valid, weights=weights_valid))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    numeric_values = np.asarray(values, dtype=float)
    numeric_weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(numeric_values) & np.isfinite(numeric_weights)
    if not valid.any():
        return float("nan")
    numeric_values = numeric_values[valid]
    numeric_weights = np.clip(numeric_weights[valid], 0.0, None)
    if np.allclose(numeric_weights.sum(), 0.0):
        return float(np.quantile(numeric_values, quantile))
    order = np.argsort(numeric_values)
    sorted_values = numeric_values[order]
    sorted_weights = numeric_weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def aggregate_weighted_values(values: np.ndarray, weights: np.ndarray, method: str = "weighted_median") -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any():
        return float("nan")
    numeric_values = np.asarray(values[valid], dtype=float)
    numeric_weights = np.clip(np.asarray(weights[valid], dtype=float), 0.0, None)
    if method == "weighted_median":
        return weighted_quantile(numeric_values, numeric_weights, 0.5)
    if method == "weighted_trimmed_mean":
        if np.allclose(numeric_weights.sum(), 0.0):
            return float(np.mean(numeric_values))
        lower = weighted_quantile(numeric_values, numeric_weights, 0.1)
        upper = weighted_quantile(numeric_values, numeric_weights, 0.9)
        keep = (numeric_values >= lower) & (numeric_values <= upper)
        if not keep.any():
            keep = np.ones(len(numeric_values), dtype=bool)
        return weighted_average(numeric_values[keep], numeric_weights[keep])
    if method == "mean":
        return weighted_average(numeric_values, numeric_weights)
    raise ValueError("method must be one of: weighted_median, weighted_trimmed_mean, mean")


def normalize_distance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    min_value = float(np.nanmin(numeric.to_numpy(dtype=float)))
    max_value = float(np.nanmax(numeric.to_numpy(dtype=float)))
    if math.isclose(min_value, max_value):
        fill_value = 0.0 if math.isclose(max_value, 0.0) else 1.0
        return pd.DataFrame(fill_value, index=numeric.index, columns=numeric.columns, dtype=float)
    return (numeric - min_value) / (max_value - min_value)


def _square_cophenetic(labels: pd.Index, condensed: np.ndarray) -> pd.DataFrame:
    square = squareform(condensed)
    return pd.DataFrame(square, index=labels, columns=labels)


def compute_cophenetic_from_distance_matrix(
    distance_matrix: pd.DataFrame,
    *,
    method: str = "average",
    show_corr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if distance_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    numeric = distance_matrix.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        finite_values = numeric.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        fill_value = float(np.max(finite_values)) if finite_values.size else 1.0
        numeric = numeric.fillna(fill_value)

    if numeric.shape[0] == 1:
        row_cophenetic = pd.DataFrame([[0.0]], index=numeric.index, columns=numeric.index, dtype=float)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_linkage = linkage(numeric.values, method=method)
            row_corr, row_condensed = cophenet(row_linkage, pdist(numeric.values))
        if show_corr:
            print(f"Row cophenetic correlation coefficient: {row_corr:.4f}")
        row_cophenetic = _square_cophenetic(numeric.index, row_condensed)

    if numeric.shape[1] == 1:
        col_cophenetic = pd.DataFrame([[0.0]], index=numeric.columns, columns=numeric.columns, dtype=float)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_linkage = linkage(numeric.T.values, method=method)
            col_corr, col_condensed = cophenet(col_linkage, pdist(numeric.T.values))
        if show_corr:
            print(f"Column cophenetic correlation coefficient: {col_corr:.4f}")
        col_cophenetic = _square_cophenetic(numeric.columns, col_condensed)

    row_normalized = normalize_distance_frame(row_cophenetic)
    col_normalized = normalize_distance_frame(col_cophenetic)
    for label in row_normalized.index.intersection(row_normalized.columns):
        row_normalized.loc[label, label] = 0.0
    for label in col_normalized.index.intersection(col_normalized.columns):
        col_normalized.loc[label, label] = 0.0
    return row_normalized, col_normalized


def safe_row_cophenetic(distance_matrix: pd.DataFrame, *, method: str = "average") -> pd.DataFrame:
    if distance_matrix.empty:
        return pd.DataFrame()
    if distance_matrix.shape[0] == 1:
        idx = distance_matrix.index.astype(str)
        return pd.DataFrame([[0.0]], index=idx, columns=idx, dtype=float)
    row_cophenetic, _ = compute_cophenetic_from_distance_matrix(distance_matrix, method=method, show_corr=False)
    return row_cophenetic


def compute_weighted_searcher_findee_distance_matrix_from_df(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    group_col: str = "celltype",
    weight_col: Optional[str] = "weight",
    min_weight: float = 0.0,
) -> pd.DataFrame:
    required = {x_col, y_col, group_col}
    if z_col is not None:
        required.add(z_col)
    if weight_col is not None:
        required.add(weight_col)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")

    work = df.copy()
    if weight_col is None:
        work["__weight"] = 1.0
        weight_col = "__weight"

    work[group_col] = work[group_col].astype("category").cat.remove_unused_categories()
    work[weight_col] = _coerce_nonnegative(work[weight_col])
    work = work.loc[work[weight_col] > float(min_weight)].copy()
    if work.empty:
        raise ValueError("No weighted points remain after filtering; cannot compute weighted distances.")

    coord_cols = [x_col, y_col] + ([z_col] if z_col is not None else [])
    coords = work[coord_cols].to_numpy(dtype=float)
    groups = work[group_col].astype("category").cat.remove_unused_categories()
    unique_groups = list(groups.cat.categories)

    nearest = pd.DataFrame(index=work.index, columns=unique_groups, dtype=float)
    for target in unique_groups:
        target_mask = (groups == target).to_numpy()
        coords_target = coords[target_mask]
        if coords_target.shape[0] == 0:
            nearest.loc[:, target] = np.nan
            continue
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(coords_target)
        distances, _ = nbrs.kneighbors(coords)
        nearest[target] = distances[:, 0]

    weights = work[weight_col].to_numpy(dtype=float)
    rows: list[pd.Series] = []
    for group in unique_groups:
        source_mask = (groups == group).to_numpy()
        source_values = nearest.loc[source_mask, unique_groups]
        source_weights = weights[source_mask]
        rows.append(
            pd.Series(
                {
                    target: weighted_average(source_values[target].to_numpy(dtype=float), source_weights)
                    for target in unique_groups
                },
                name=str(group),
            )
        )
    return pd.DataFrame(rows, index=unique_groups, columns=unique_groups)


def compute_weighted_cophenetic_distances_from_df(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    group_col: str = "celltype",
    weight_col: Optional[str] = "weight",
    min_weight: float = 0.0,
    method: str = "average",
    show_corr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_mean = compute_weighted_searcher_findee_distance_matrix_from_df(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        group_col=group_col,
        weight_col=weight_col,
        min_weight=min_weight,
    )
    return compute_cophenetic_from_distance_matrix(group_mean, method=method, show_corr=show_corr)


def build_entity_points_from_expression(
    reference_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    *,
    entities: Optional[Iterable[str]] = None,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    min_weight: float = 0.0,
    entity_col: str = "entity",
    weight_col: str = "weight",
) -> pd.DataFrame:
    selected_entities = list(entities) if entities is not None else list(expression_df.columns)
    aligned_expression = expression_df.reindex(reference_df[cell_id_col]).fillna(0.0)
    aligned_expression.index = aligned_expression.index.astype(str)
    records: list[pd.DataFrame] = []

    for entity in selected_entities:
        if entity not in aligned_expression.columns:
            continue
        weights = _coerce_nonnegative(aligned_expression[entity])
        keep = weights > float(min_weight)
        if not keep.any():
            continue
        entity_points = reference_df.loc[keep.to_numpy(), [cell_id_col, x_col, y_col, "celltype"]].copy()
        entity_points[entity_col] = str(entity)
        entity_points[weight_col] = weights.loc[keep].to_numpy(dtype=float)
        records.append(entity_points)

    if not records:
        return pd.DataFrame(columns=[cell_id_col, x_col, y_col, "celltype", entity_col, weight_col])
    return pd.concat(records, ignore_index=True)


def compute_entity_to_cell_topology(
    reference_df: pd.DataFrame,
    entity_points_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    celltype_col: str = "celltype",
    entity_col: str = "entity",
    weight_col: str = "weight",
    min_weight: float = 0.0,
    method: str = "average",
) -> pd.DataFrame:
    reference = reference_df.copy()
    if celltype_col not in reference.columns:
        raise ValueError(f"reference_df must contain {celltype_col!r}.")
    required = {x_col, y_col, entity_col, weight_col}
    if z_col is not None:
        required.add(z_col)
    missing = required.difference(entity_points_df.columns)
    if missing:
        raise ValueError(f"entity_points_df is missing required columns: {sorted(missing)}")

    reference = reference.rename(columns={celltype_col: "__group"})
    reference["__weight"] = 1.0
    reference_cols = [x_col, y_col] + ([z_col] if z_col is not None else []) + ["__group", "__weight"]
    reference = reference.loc[:, reference_cols]
    unique_celltypes = list(dict.fromkeys(reference["__group"].astype(str).tolist()))

    entity_points = entity_points_df.copy()
    entity_points[entity_col] = entity_points[entity_col].astype(str)
    entity_points[weight_col] = _coerce_nonnegative(entity_points[weight_col])
    unique_entities = list(dict.fromkeys(entity_points[entity_col].tolist()))

    rows: list[pd.Series] = []
    for entity in unique_entities:
        sub = entity_points.loc[
            (entity_points[entity_col] == entity) & (entity_points[weight_col] > float(min_weight))
        ].copy()
        if sub.empty:
            rows.append(pd.Series(np.nan, index=unique_celltypes, name=entity))
            continue
        sub = sub.rename(columns={entity_col: "__group", weight_col: "__weight"})
        sub["__group"] = str(entity)
        sub = sub.loc[:, reference_cols]
        combined = pd.concat([reference, sub], ignore_index=True)
        row_cophenetic, _ = compute_weighted_cophenetic_distances_from_df(
            combined,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            group_col="__group",
            weight_col="__weight",
            min_weight=min_weight,
            method=method,
        )
        rows.append(pd.Series(row_cophenetic.loc[str(entity)].reindex(unique_celltypes), name=str(entity)))

    if not rows:
        return pd.DataFrame(columns=unique_celltypes)
    out = pd.DataFrame(rows)
    out.index.name = entity_col
    out.columns.name = celltype_col
    return out


def compute_entity_structuremap(
    entity_points_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    entity_col: str = "entity",
    weight_col: str = "weight",
    min_weight: float = 0.0,
    method: str = "average",
) -> pd.DataFrame:
    if entity_points_df.empty:
        return pd.DataFrame()
    work = entity_points_df.copy()
    work[entity_col] = work[entity_col].astype(str)
    work[weight_col] = _coerce_nonnegative(work[weight_col])
    row_cophenetic, _ = compute_weighted_cophenetic_distances_from_df(
        work,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        group_col=entity_col,
        weight_col=weight_col,
        min_weight=min_weight,
        method=method,
    )
    row_cophenetic.index.name = entity_col
    row_cophenetic.columns.name = entity_col
    return row_cophenetic


def build_neighbor_index(
    reference_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    k_neighbors: int = 8,
    radius: Optional[float] = None,
) -> list[np.ndarray]:
    coords = reference_df[[x_col, y_col]].to_numpy(dtype=float)
    if len(coords) == 0:
        return []
    if radius is not None:
        model = NearestNeighbors(radius=radius, algorithm="auto")
        model.fit(coords)
        neighbors = model.radius_neighbors(coords, return_distance=False)
        return [arr[arr != idx] for idx, arr in enumerate(neighbors)]
    n_neighbors = min(len(coords), max(2, int(k_neighbors) + 1))
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    model.fit(coords)
    indices = model.kneighbors(coords, return_distance=False)
    return [arr[arr != idx] for idx, arr in enumerate(indices)]


def smooth_matrix_by_neighbors(
    matrix: pd.DataFrame,
    neighbor_index: list[np.ndarray],
    *,
    include_self: bool = True,
) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()
    values = matrix.to_numpy(dtype=float)
    smoothed = np.zeros_like(values)
    for idx, neighbors in enumerate(neighbor_index):
        neighborhood = np.unique(np.append(neighbors, idx)) if include_self else np.asarray(neighbors, dtype=int)
        if len(neighborhood) == 0:
            smoothed[idx] = values[idx]
        else:
            smoothed[idx] = values[neighborhood].mean(axis=0)
    return pd.DataFrame(smoothed, index=matrix.index, columns=matrix.columns)


def summarize_expression_by_celltype(
    expression_df: pd.DataFrame,
    celltypes: pd.Series,
    *,
    detection_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = expression_df.copy()
    expression.index = expression.index.astype(str)
    typed = celltypes.astype(str)
    pseudobulk = expression.groupby(typed).sum()
    pseudobulk_share = pseudobulk.div(pseudobulk.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
    detection_fraction = (expression > float(detection_threshold)).groupby(typed).mean()
    combined = pseudobulk_share.mul(np.sqrt(detection_fraction))
    raw = combined.T.astype(float)
    raw.index.name = "gene"
    raw.columns.name = "celltype"
    normalized = normalize_frame_rows(raw)
    normalized.index.name = "gene"
    normalized.columns.name = "celltype"
    return raw, normalized


def normalize_lr_prior(lr_pairs: pd.DataFrame, prior_col: Optional[str]) -> pd.Series:
    if prior_col is None or prior_col not in lr_pairs.columns:
        return pd.Series(np.ones(len(lr_pairs)), index=lr_pairs.index, name="prior_confidence")
    prior = pd.to_numeric(lr_pairs[prior_col], errors="coerce").fillna(0.0).astype(float)
    return normalize_series(prior).rename("prior_confidence")


def compute_local_contact_matrix(
    reference_df: pd.DataFrame,
    ligand_values: pd.Series,
    receptor_values: pd.Series,
    neighbor_index: list[np.ndarray],
    *,
    celltype_col: str = "celltype",
    min_cross_edges: int = 50,
    contact_expr_threshold: str | float = "q75_nonzero",
    winsor_quantile: float = 0.99,
) -> dict[str, pd.DataFrame]:
    celltypes = reference_df[celltype_col].astype(str).to_numpy()
    unique_celltypes = list(dict.fromkeys(celltypes.tolist()))
    code_map = {celltype: idx for idx, celltype in enumerate(unique_celltypes)}
    n_celltypes = len(unique_celltypes)

    ligand_raw = pd.to_numeric(ligand_values.reindex(reference_df.index).fillna(0.0), errors="coerce").fillna(0.0).astype(float)
    receptor_raw = pd.to_numeric(receptor_values.reindex(reference_df.index).fillna(0.0), errors="coerce").fillna(0.0).astype(float)
    ligand_norm = winsorized_normalize_series(ligand_raw, upper_quantile=winsor_quantile)
    receptor_norm = winsorized_normalize_series(receptor_raw, upper_quantile=winsor_quantile)

    def _resolve_threshold(series: pd.Series) -> float:
        if isinstance(contact_expr_threshold, (int, float)):
            return float(contact_expr_threshold)
        if contact_expr_threshold == "q75_nonzero":
            positive = series.loc[series > 0]
            return float(positive.quantile(0.75)) if not positive.empty else float("inf")
        raise ValueError("contact_expr_threshold must be a float or 'q75_nonzero'")

    ligand_threshold = _resolve_threshold(ligand_raw)
    receptor_threshold = _resolve_threshold(receptor_raw)

    sender_codes: list[int] = []
    receiver_codes: list[int] = []
    edge_strengths: list[float] = []
    active_edges: list[int] = []

    ligand_raw_values = ligand_raw.to_numpy(dtype=float)
    receptor_raw_values = receptor_raw.to_numpy(dtype=float)
    ligand_norm_values = ligand_norm.to_numpy(dtype=float)
    receptor_norm_values = receptor_norm.to_numpy(dtype=float)

    for idx, neighbors in enumerate(neighbor_index):
        if len(neighbors) == 0:
            continue
        sender_code = code_map[str(celltypes[idx])]
        ligand_raw_value = ligand_raw_values[idx]
        ligand_norm_value = ligand_norm_values[idx]
        for neighbor in neighbors:
            receiver_code = code_map[str(celltypes[neighbor])]
            sender_codes.append(sender_code)
            receiver_codes.append(receiver_code)
            receptor_norm_value = receptor_norm_values[neighbor]
            receptor_raw_value = receptor_raw_values[neighbor]
            edge_strengths.append(float(ligand_norm_value * receptor_norm_value))
            active_edges.append(int(ligand_raw_value > ligand_threshold and receptor_raw_value > receptor_threshold))

    strength_sum = np.zeros((n_celltypes, n_celltypes), dtype=float)
    edge_count = np.zeros((n_celltypes, n_celltypes), dtype=int)
    active_count = np.zeros((n_celltypes, n_celltypes), dtype=int)

    if sender_codes:
        sender_codes_arr = np.asarray(sender_codes, dtype=int)
        receiver_codes_arr = np.asarray(receiver_codes, dtype=int)
        edge_strengths_arr = np.asarray(edge_strengths, dtype=float)
        active_edges_arr = np.asarray(active_edges, dtype=int)
        np.add.at(strength_sum, (sender_codes_arr, receiver_codes_arr), edge_strengths_arr)
        np.add.at(edge_count, (sender_codes_arr, receiver_codes_arr), 1)
        np.add.at(active_count, (sender_codes_arr, receiver_codes_arr), active_edges_arr)

    with np.errstate(divide="ignore", invalid="ignore"):
        strength = np.divide(
            strength_sum,
            edge_count,
            out=np.zeros_like(strength_sum, dtype=float),
            where=edge_count > 0,
        )
        coverage = np.divide(
            active_count,
            edge_count,
            out=np.zeros_like(strength_sum, dtype=float),
            where=edge_count > 0,
        )

    strength_df = pd.DataFrame(strength, index=unique_celltypes, columns=unique_celltypes, dtype=float)
    coverage_df = pd.DataFrame(coverage, index=unique_celltypes, columns=unique_celltypes, dtype=float)
    edge_count_df = pd.DataFrame(edge_count, index=unique_celltypes, columns=unique_celltypes, dtype=int)

    strength_norm_df = winsorized_normalize_frame(strength_df, upper_quantile=winsor_quantile)
    off_diagonal_mask = ~np.eye(n_celltypes, dtype=bool)
    off_diagonal_edges = edge_count[off_diagonal_mask]
    off_diagonal_edges = off_diagonal_edges[off_diagonal_edges > 0]
    edge_scale = float(np.quantile(off_diagonal_edges, 0.99)) if off_diagonal_edges.size else 1.0
    if math.isclose(edge_scale, 0.0):
        edge_scale = 1.0
    edge_support_df = (edge_count_df.astype(float) / edge_scale).clip(lower=0.0, upper=1.0)

    enough_edges = edge_count_df >= int(min_cross_edges)
    local_contact_df = (np.sqrt(strength_norm_df.mul(coverage_df)) * edge_support_df).where(enough_edges, other=0.0)

    threshold_template = pd.DataFrame(
        np.zeros((n_celltypes, n_celltypes), dtype=float),
        index=unique_celltypes,
        columns=unique_celltypes,
    )
    ligand_threshold_df = threshold_template + float(ligand_threshold)
    receptor_threshold_df = threshold_template + float(receptor_threshold)
    return {
        "local_contact": local_contact_df.astype(float),
        "contact_strength_raw": strength_df.astype(float),
        "contact_strength_normalized": strength_norm_df.astype(float),
        "contact_coverage": coverage_df.astype(float),
        "cross_edge_count": edge_count_df.astype(int),
        "edge_support": edge_support_df.astype(float),
        "ligand_threshold": ligand_threshold_df,
        "receptor_threshold": receptor_threshold_df,
    }


def geometric_mean(values: Iterable[float], eps: float = 1e-8) -> float:
    array = np.asarray(list(values), dtype=float)
    array = np.clip(array, 0.0, None)
    return float(np.exp(np.mean(np.log(array + eps))))


def _feature_name_lookup(adata: Any, *, use_raw: bool = False) -> tuple[Any, dict[str, int]]:
    if use_raw and getattr(adata, "raw", None) is not None:
        matrix_holder = adata.raw
        var = adata.raw.var.copy()
        var_names = pd.Index(adata.raw.var_names.astype(str))
    else:
        matrix_holder = adata
        var = adata.var.copy()
        var_names = pd.Index(adata.var_names.astype(str))

    lookup: dict[str, int] = {}
    for idx, value in enumerate(var_names):
        lookup.setdefault(str(value).casefold(), idx)
    for alias_col in ("name", "feature_name", "gene_name"):
        if alias_col in var.columns:
            for idx, value in enumerate(var[alias_col].astype(str).tolist()):
                lookup.setdefault(str(value).casefold(), idx)
    return matrix_holder, lookup


def _default_cell_ids(adata: Any, cell_id_col: str) -> pd.Series:
    if cell_id_col in adata.obs.columns:
        return adata.obs[cell_id_col].astype(str)
    return pd.Series(adata.obs_names.astype(str), index=adata.obs_names, name=cell_id_col)


def reference_from_adata(
    adata: Any,
    *,
    cluster_col: str = "cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required to derive the reference table.")
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"{cluster_col!r} is missing from adata.obs.")

    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must contain at least two columns.")

    cell_ids = _default_cell_ids(adata, cell_id_col)
    return pd.DataFrame(
        {
            cell_id_col: cell_ids.to_numpy(),
            x_col: spatial[:, 0],
            y_col: spatial[:, 1],
            "celltype": adata.obs[cluster_col].astype(str).to_numpy(),
        },
        index=adata.obs_names.astype(str),
    )


def expression_from_adata(
    adata: Any,
    genes: Iterable[str],
    *,
    cell_id_col: str = "cell_id",
    use_raw: bool = False,
) -> pd.DataFrame:
    requested_genes = list(dict.fromkeys(str(gene) for gene in genes))
    matrix_holder, lookup = _feature_name_lookup(adata, use_raw=use_raw)
    positions: list[int] = []
    column_names: list[str] = []
    for gene in requested_genes:
        position = lookup.get(gene.casefold())
        if position is None:
            continue
        positions.append(position)
        column_names.append(gene)

    cell_ids = _default_cell_ids(adata, cell_id_col).astype(str).to_numpy()
    if not positions:
        return pd.DataFrame(index=cell_ids)

    matrix = matrix_holder[:, positions].X
    if issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
    return pd.DataFrame(matrix, index=cell_ids, columns=column_names)


def coerce_reference_df(
    reference_df: Optional[pd.DataFrame] = None,
    *,
    adata: Any = None,
    cluster_col: str = "cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
) -> pd.DataFrame:
    if reference_df is None:
        if adata is None:
            raise ValueError("Either reference_df or adata must be provided.")
        reference = reference_from_adata(
            adata,
            cluster_col=cluster_col,
            cell_id_col=cell_id_col,
            x_col=x_col,
            y_col=y_col,
        )
    else:
        reference = reference_df.copy()
        if cell_id_col not in reference.columns:
            if reference.index.is_unique:
                reference[cell_id_col] = reference.index.astype(str)
            else:
                reference[cell_id_col] = [f"cell_{idx}" for idx in range(len(reference))]
        if celltype_col not in reference.columns:
            raise ValueError(f"reference_df must contain {celltype_col!r}.")
        rename_map = {}
        if celltype_col != "celltype":
            rename_map[celltype_col] = "celltype"
        if rename_map:
            reference = reference.rename(columns=rename_map)

    required = {cell_id_col, x_col, y_col, "celltype"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"reference_df is missing required columns: {sorted(missing)}")
    reference[cell_id_col] = reference[cell_id_col].astype(str)
    reference["celltype"] = reference["celltype"].astype(str)
    return reference[[cell_id_col, x_col, y_col, "celltype"]].copy()


def coerce_expression_df(
    reference_df: pd.DataFrame,
    expression_df: Optional[pd.DataFrame] = None,
    *,
    adata: Any = None,
    genes: Optional[Iterable[str]] = None,
    cell_id_col: str = "cell_id",
    use_raw: bool = False,
) -> pd.DataFrame:
    if expression_df is None:
        if adata is None:
            raise ValueError("Either expression_df or adata must be provided.")
        expression = expression_from_adata(adata, genes or [], cell_id_col=cell_id_col, use_raw=use_raw)
    else:
        expression = expression_df.copy()
        if cell_id_col in expression.columns:
            expression[cell_id_col] = expression[cell_id_col].astype(str)
            expression = expression.set_index(cell_id_col)
        expression.index = expression.index.astype(str)
        if genes is not None:
            keep = [gene for gene in genes if gene in expression.columns]
            expression = expression.loc[:, keep]

    aligned = expression.reindex(reference_df[cell_id_col].astype(str)).fillna(0.0)
    aligned.index = reference_df[cell_id_col].astype(str)
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _pick_matching_file(base: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_matrix_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.apply(pd.to_numeric, errors="coerce")


def resolve_precomputed_tables(
    *,
    tbc_results: Optional[str | Path] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    topology_df = t_and_c_df.copy() if t_and_c_df is not None else None
    structure_df = structure_map_df.copy() if structure_map_df is not None else None
    if structure_df is None and structure_map is not None and isinstance(structure_map, pd.DataFrame):
        structure_df = structure_map.copy()

    if tbc_results is not None:
        base = Path(tbc_results)
        if base.is_file():
            if topology_df is None:
                topology_df = load_matrix_csv(base)
            if structure_df is None:
                sibling = _pick_matching_file(base.parent, ["StructureMap_table*.csv", "*StructureMap*.csv"])
                if sibling is not None:
                    structure_df = load_matrix_csv(sibling)
        elif base.is_dir():
            if topology_df is None:
                topology_path = _pick_matching_file(base, ["t_and_c_result*.csv", "*t_and_c*.csv"])
                if topology_path is not None:
                    topology_df = load_matrix_csv(topology_path)
            if structure_df is None:
                structure_path = _pick_matching_file(base, ["StructureMap_table*.csv", "*StructureMap*.csv"])
                if structure_path is not None:
                    structure_df = load_matrix_csv(structure_path)

    if topology_df is not None:
        topology_df.index = topology_df.index.astype(str)
        topology_df.columns = topology_df.columns.astype(str)
        topology_df = topology_df.apply(pd.to_numeric, errors="coerce")
    if structure_df is not None:
        structure_df.index = structure_df.index.astype(str)
        structure_df.columns = structure_df.columns.astype(str)
        structure_df = structure_df.apply(pd.to_numeric, errors="coerce")
    return topology_df, structure_df


def recompute_gene_topology(
    reference_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    genes: Sequence[str],
    *,
    entity_points_df: Optional[pd.DataFrame] = None,
    cell_id_col: str,
    x_col: str,
    y_col: str,
    entity_min_weight: float,
    topology_method: str,
) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame(columns=reference_df["celltype"].astype(str).drop_duplicates().tolist())
    present = [gene for gene in genes if gene in expression_df.columns]
    if entity_points_df is not None:
        entity_points = entity_points_df.loc[entity_points_df["entity"].astype(str).isin(genes)].copy()
    else:
        if not present:
            return pd.DataFrame(columns=reference_df["celltype"].astype(str).drop_duplicates().tolist())
        entity_points = build_entity_points_from_expression(
            reference_df,
            expression_df,
            entities=present,
            cell_id_col=cell_id_col,
            x_col=x_col,
            y_col=y_col,
            min_weight=entity_min_weight,
        )
    if entity_points.empty:
        return pd.DataFrame(index=present, columns=reference_df["celltype"].astype(str).drop_duplicates().tolist())
    return compute_entity_to_cell_topology(
        reference_df,
        entity_points,
        x_col=x_col,
        y_col=y_col,
        celltype_col="celltype",
        entity_col="entity",
        weight_col="weight",
        min_weight=entity_min_weight,
        method=topology_method,
    )


def resolve_gene_topology_anchors(
    reference_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    genes: Sequence[str],
    *,
    tbc_results: Optional[str | Path] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
    anchor_mode: str = "precomputed",
    entity_points_df: Optional[pd.DataFrame] = None,
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    entity_min_weight: float = 0.0,
    topology_method: str = "average",
) -> tuple[pd.DataFrame, dict[str, str], Optional[pd.DataFrame], str]:
    if anchor_mode not in {"precomputed", "recompute", "hybrid"}:
        raise ValueError("anchor_mode must be one of: precomputed, recompute, hybrid")

    celltypes = list(dict.fromkeys(reference_df["celltype"].astype(str).tolist()))
    unique_genes = list(dict.fromkeys(str(gene) for gene in genes))
    precomputed_topology, precomputed_structure = resolve_precomputed_tables(
        tbc_results=tbc_results,
        t_and_c_df=t_and_c_df,
        structure_map=structure_map,
        structure_map_df=structure_map_df,
    )

    topology_parts: list[pd.DataFrame] = []
    source_by_gene: dict[str, str] = {}

    use_precomputed = anchor_mode in {"precomputed", "hybrid"} and precomputed_topology is not None
    if use_precomputed:
        available = [gene for gene in unique_genes if gene in precomputed_topology.index]
        if available:
            topology_parts.append(precomputed_topology.reindex(available).reindex(columns=celltypes))
            source_by_gene.update({gene: "precomputed" for gene in available})

    if anchor_mode == "recompute" or precomputed_topology is None:
        needs_recompute = unique_genes
    else:
        needs_recompute = [gene for gene in unique_genes if gene not in source_by_gene]

    if needs_recompute:
        recomputed = recompute_gene_topology(
            reference_df,
            expression_df,
            needs_recompute,
            entity_points_df=entity_points_df,
            cell_id_col=cell_id_col,
            x_col=x_col,
            y_col=y_col,
            entity_min_weight=entity_min_weight,
            topology_method=topology_method,
        )
        if not recomputed.empty:
            topology_parts.append(recomputed.reindex(columns=celltypes))
        source_by_gene.update({gene: "recompute" for gene in needs_recompute})

    if topology_parts:
        topology = pd.concat(topology_parts, axis=0)
        topology = topology[~topology.index.duplicated(keep="first")]
        topology = topology.reindex(unique_genes).reindex(columns=celltypes)
    else:
        topology = pd.DataFrame(index=unique_genes, columns=celltypes, dtype=float)

    if precomputed_structure is not None and anchor_mode in {"precomputed", "hybrid"}:
        structure_source = "precomputed"
        resolved_structure = precomputed_structure.reindex(index=celltypes, columns=celltypes)
    elif structure_map is not None and isinstance(structure_map, pd.DataFrame):
        structure_source = "provided"
        resolved_structure = structure_map.copy().reindex(index=celltypes, columns=celltypes)
    else:
        structure_source = "recompute"
        resolved_structure, _ = compute_cophenetic_from_distance_matrix(
            compute_weighted_searcher_findee_distance_matrix_from_df(
                reference_df.assign(weight=1.0),
                x_col=x_col,
                y_col=y_col,
                group_col="celltype",
                weight_col="weight",
                min_weight=0.0,
            ),
            method=topology_method,
            show_corr=False,
        )
        resolved_structure = resolved_structure.reindex(index=celltypes, columns=celltypes)
    resolved_structure = resolved_structure.apply(pd.to_numeric, errors="coerce").fillna(1.0)
    for celltype in celltypes:
        if celltype in resolved_structure.index and celltype in resolved_structure.columns:
            resolved_structure.loc[celltype, celltype] = 0.0

    topology.index.name = "gene"
    topology.columns.name = "celltype"
    return topology, source_by_gene, resolved_structure, structure_source


def prepare_hotspot_table(
    reference_df: pd.DataFrame,
    *,
    sender_mask: pd.Series,
    receiver_mask: pd.Series,
    sender_score: pd.Series,
    receiver_score: pd.Series,
    ligand: str,
    receptor: str,
    sender_celltype: str,
    receiver_celltype: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    sender_table = reference_df.loc[sender_mask, ["cell_id", x_col, y_col, "celltype"]].copy()
    sender_table["role"] = "sender"
    sender_table["feature"] = ligand
    sender_table["score"] = sender_score.loc[sender_mask].to_numpy(dtype=float)
    sender_table["sender_celltype"] = sender_celltype
    sender_table["receiver_celltype"] = receiver_celltype
    sender_table["ligand"] = ligand
    sender_table["receptor"] = receptor

    receiver_table = reference_df.loc[receiver_mask, ["cell_id", x_col, y_col, "celltype"]].copy()
    receiver_table["role"] = "receiver"
    receiver_table["feature"] = receptor
    receiver_table["score"] = receiver_score.loc[receiver_mask].to_numpy(dtype=float)
    receiver_table["sender_celltype"] = sender_celltype
    receiver_table["receiver_celltype"] = receiver_celltype
    receiver_table["ligand"] = ligand
    receiver_table["receptor"] = receptor
    return pd.concat([sender_table, receiver_table], ignore_index=True)


def safe_to_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def save_matrix_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    output_prefix: Path,
    cmap: str = "viridis",
) -> list[str]:
    if matrix.empty:
        return []
    fig_width = max(6.0, 0.45 * max(1, matrix.shape[1]))
    fig_height = max(4.0, 0.35 * max(1, matrix.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=90, fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str), fontsize=8)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    outputs: list[str] = []
    for ext in ("png", "pdf"):
        path = output_prefix.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def save_hotspot_overlay(
    reference_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    sender_mask: pd.Series,
    receiver_mask: pd.Series,
    sender_score: pd.Series,
    receiver_score: pd.Series,
    title: str,
    output_prefix: Path,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(
        reference_df[x_col],
        reference_df[y_col],
        s=8,
        c="#D0D0D0",
        alpha=0.45,
        linewidths=0.0,
        label="Background cells",
    )

    if sender_mask.any():
        sender_cells = reference_df.loc[sender_mask]
        ax.scatter(
            sender_cells[x_col],
            sender_cells[y_col],
            s=24 + 36 * sender_score.loc[sender_mask].to_numpy(dtype=float),
            c=sender_score.loc[sender_mask].to_numpy(dtype=float),
            cmap="Reds",
            alpha=0.85,
            linewidths=0.0,
            label="Sender hotspot",
        )

    if receiver_mask.any():
        receiver_cells = reference_df.loc[receiver_mask]
        ax.scatter(
            receiver_cells[x_col],
            receiver_cells[y_col],
            s=24 + 36 * receiver_score.loc[receiver_mask].to_numpy(dtype=float),
            c=receiver_score.loc[receiver_mask].to_numpy(dtype=float),
            cmap="Blues",
            alpha=0.85,
            linewidths=0.0,
            label="Receiver hotspot",
            marker="s",
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    outputs: list[str] = []
    for ext in ("png", "pdf"):
        path = output_prefix.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs
