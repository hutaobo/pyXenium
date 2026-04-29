from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from topolink_cci_validation_framework import (
    COMPONENTS,
    TARGET_AXES,
    TargetAxis,
    build_controls_table,
    component_ablation,
    cross_method_support,
    expression_specificity,
    find_target_rows,
    geometric_mean,
    load_bundle_expression,
    load_manifest,
    markdown_table,
    matched_gene_controls,
    qnorm_sf,
    read_scores,
    receiver_panel_support,
    spatial_abundance_null,
    summarize_expression,
)


DEFAULT_ROOT = Path("/data/taobo.hu/pyxenium_cci_benchmark_2026-04")
RBC_PLATELET_MARKERS = ("PPBP", "PF4", "ITGA2B", "GP1BA", "HBB", "HBA1", "HBA2")
ENDOTHELIAL_MARKERS = ("PECAM1", "EMCN", "CDH5", "KDR", "FLT1", "MMRN2", "CLEC14A", "VWF")
HIGH_SCORE_NONCLASSIC_CONTROLS = (
    ("GNAS", "ADCY1", "11q13 Invasive Tumor Cells", "11q13 Invasive Tumor Cells"),
    ("CDH1", "IGF1R", "Basal-like DCIS Cells", "Basal-like DCIS Cells"),
    ("DSC3", "DSG3", "Basal-like DCIS Cells", "Basal-like DCIS Cells"),
)


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    arr = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    valid = np.where(np.isfinite(arr))[0]
    if len(valid) == 0:
        return pd.Series(out, index=pvalues.index)
    order = valid[np.argsort(arr[valid])]
    ranked = arr[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return pd.Series(out, index=pvalues.index)


def cell_type_codes(meta: pd.DataFrame) -> tuple[pd.Categorical, dict[str, np.ndarray]]:
    cats = pd.Categorical(meta["cell_type"].astype(str))
    indices = {str(cat): np.where(cats.codes == i)[0] for i, cat in enumerate(cats.categories)}
    return cats, indices


def dense_gene_cache(matrix, gene_to_index: dict[str, int], genes: list[str]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for gene in sorted({g.upper() for g in genes if g and g.upper() in gene_to_index}):
        cache[gene] = np.asarray(matrix.getrow(gene_to_index[gene]).toarray()).ravel().astype(np.float32)
    return cache


def row_for_gene(cache: dict[str, np.ndarray], gene: str) -> np.ndarray | None:
    return cache.get(str(gene).upper())


def empirical_p_greater(observed: float, null: np.ndarray) -> float:
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))


def empirical_p_less(observed: float, null: np.ndarray) -> float:
    if len(null) == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.count_nonzero(null <= observed) + 1) / (len(null) + 1))


def build_spatial_edges(
    coords: np.ndarray,
    sender_indices: np.ndarray,
    receiver_indices: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(sender_indices) == 0 or len(receiver_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    same_pool = np.array_equal(sender_indices, receiver_indices)
    query_k = min(len(receiver_indices), k + 1 if same_pool else k)
    tree = cKDTree(coords[receiver_indices])
    distances, neighbor_pos = tree.query(coords[sender_indices], k=query_k)
    distances = np.atleast_2d(distances)
    neighbor_pos = np.atleast_2d(neighbor_pos)
    if distances.shape[0] != len(sender_indices):
        distances = distances.T
        neighbor_pos = neighbor_pos.T
    edge_s: list[np.ndarray] = []
    edge_r: list[np.ndarray] = []
    edge_d: list[np.ndarray] = []
    for i, sender in enumerate(sender_indices):
        rec = receiver_indices[neighbor_pos[i]]
        dist = distances[i]
        mask = np.isfinite(dist)
        if same_pool:
            mask &= rec != sender
        rec = rec[mask][:k]
        dist = dist[mask][:k]
        if len(rec):
            edge_s.append(np.repeat(sender, len(rec)))
            edge_r.append(rec)
            edge_d.append(dist)
    if not edge_s:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(edge_s), np.concatenate(edge_r), np.concatenate(edge_d)


def nearest_pool(
    gene_idx: int,
    global_mean: np.ndarray,
    global_det: np.ndarray,
    protected: set[int],
    *,
    pool_size: int = 400,
) -> np.ndarray:
    all_indices = np.arange(len(global_mean))
    distance = np.abs(np.log1p(global_mean) - np.log1p(global_mean[gene_idx]))
    distance += np.abs(global_det - global_det[gene_idx])
    if protected:
        distance[list(protected)] = np.inf
    distance[gene_idx] = np.inf
    valid = all_indices[np.isfinite(distance)]
    if len(valid) == 0:
        return np.array([], dtype=int)
    ordered = valid[np.argsort(distance[valid])]
    return ordered[: min(pool_size, len(ordered))]


def global_gene_summaries(mean_by_celltype: np.ndarray, det_by_celltype: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cats = pd.Categorical(meta["cell_type"].astype(str))
    sizes = np.asarray(pd.Series(cats).value_counts(sort=False), dtype=float)
    weights = sizes / max(float(sizes.sum()), 1.0)
    return mean_by_celltype @ weights, det_by_celltype @ weights


def cell_label_permutation(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    gene_cache: dict[str, np.ndarray],
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _, indices = cell_type_codes(meta)
    n_cells = len(meta)
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        lig = row_for_gene(gene_cache, row.ligand)
        rec = row_for_gene(gene_cache, row.receptor)
        sender_idx = indices.get(row.sender_celltype, np.array([], dtype=int))
        receiver_idx = indices.get(row.receiver_celltype, np.array([], dtype=int))
        if lig is None or rec is None or len(sender_idx) == 0 or len(receiver_idx) == 0:
            rows.append({"axis_id": row.axis_id, "cell_label_perm_status": "missing"})
            continue
        observed = float(np.mean(lig[sender_idx]) * np.mean(rec[receiver_idx]))
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            s = rng.choice(n_cells, size=len(sender_idx), replace=False)
            r = rng.choice(n_cells, size=len(receiver_idx), replace=False)
            null[i] = float(np.mean(lig[s]) * np.mean(rec[r]))
        rows.append(
            {
                "axis_id": row.axis_id,
                "cell_label_perm_status": "success",
                "cell_label_comm_prob": observed,
                "cell_label_perm_mean": float(np.mean(null)),
                "cell_label_perm_sd": float(np.std(null, ddof=1)),
                "cell_label_perm_z": float((observed - np.mean(null)) / max(float(np.std(null, ddof=1)), 1e-12)),
                "cell_label_perm_p": empirical_p_greater(observed, null),
                "cell_label_perm_n": n_permutations,
            }
        )
    out = pd.DataFrame(rows)
    out["cell_label_perm_fdr"] = bh_fdr(out.get("cell_label_perm_p", pd.Series(dtype=float)))
    return out


def spatial_neighborhood_controls(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    matrix,
    mean_by_celltype: np.ndarray,
    det_by_celltype: np.ndarray,
    gene_to_index: dict[str, int],
    gene_cache: dict[str, np.ndarray],
    *,
    k_neighbors: int,
    n_permutations: int,
    n_matched_pairs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    _, indices = cell_type_codes(meta)
    coords = meta[["x", "y"]].to_numpy(dtype=float)
    global_mean, global_det = global_gene_summaries(mean_by_celltype, det_by_celltype, meta)
    protected = {gene_to_index[g.upper()] for axis in TARGET_AXES for g in (axis.ligand, axis.receptor) if g.upper() in gene_to_index}
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    local_cache = dict(gene_cache)

    def get_gene_row_by_idx(idx: int) -> np.ndarray:
        symbol = f"__idx_{idx}"
        if symbol not in local_cache:
            local_cache[symbol] = np.asarray(matrix.getrow(idx).toarray()).ravel().astype(np.float32)
        return local_cache[symbol]

    for row in targets.itertuples(index=False):
        lig = row_for_gene(local_cache, row.ligand)
        rec = row_for_gene(local_cache, row.receptor)
        sender_idx = indices.get(row.sender_celltype, np.array([], dtype=int))
        receiver_idx = indices.get(row.receiver_celltype, np.array([], dtype=int))
        if lig is None or rec is None or len(sender_idx) == 0 or len(receiver_idx) == 0:
            rows.append({"axis_id": row.axis_id, "spatial_neighborhood_status": "missing"})
            continue
        edge_s, edge_r, edge_d = build_spatial_edges(coords, sender_idx, receiver_idx, k=k_neighbors)
        if len(edge_s) == 0:
            rows.append({"axis_id": row.axis_id, "spatial_neighborhood_status": "no_edges"})
            continue
        edge_product = lig[edge_s] * rec[edge_r]
        observed = float(np.mean(edge_product))
        active_fraction = float(np.mean((lig[edge_s] > 0) & (rec[edge_r] > 0)))

        perm_null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            shuffled_r = rng.permutation(edge_r)
            perm_null[i] = float(np.mean(lig[edge_s] * rec[shuffled_r]))

        lig_idx = gene_to_index.get(str(row.ligand).upper())
        rec_idx = gene_to_index.get(str(row.receptor).upper())
        matched_null: list[float] = []
        if lig_idx is not None and rec_idx is not None:
            lig_pool = nearest_pool(lig_idx, global_mean, global_det, protected)
            rec_pool = nearest_pool(rec_idx, global_mean, global_det, protected)
            if len(lig_pool) and len(rec_pool):
                lig_sample = rng.choice(lig_pool, size=n_matched_pairs, replace=len(lig_pool) < n_matched_pairs)
                rec_sample = rng.choice(rec_pool, size=n_matched_pairs, replace=len(rec_pool) < n_matched_pairs)
                for lidx, ridx in zip(lig_sample, rec_sample):
                    lrow = get_gene_row_by_idx(int(lidx))
                    rrow = get_gene_row_by_idx(int(ridx))
                    matched_null.append(float(np.mean(lrow[edge_s] * rrow[edge_r])))
        matched_arr = np.asarray(matched_null, dtype=float)
        rows.append(
            {
                "axis_id": row.axis_id,
                "spatial_neighborhood_status": "success",
                "spatial_edge_count": int(len(edge_s)),
                "spatial_edge_mean_distance": float(np.mean(edge_d)),
                "spatial_lr_edge_score": observed,
                "spatial_lr_active_edge_fraction": active_fraction,
                "spatial_perm_mean": float(np.mean(perm_null)),
                "spatial_perm_sd": float(np.std(perm_null, ddof=1)),
                "spatial_perm_z": float((observed - np.mean(perm_null)) / max(float(np.std(perm_null, ddof=1)), 1e-12)),
                "spatial_perm_p": empirical_p_greater(observed, perm_null),
                "spatial_matched_gene_mean": float(np.mean(matched_arr)) if len(matched_arr) else np.nan,
                "spatial_matched_gene_sd": float(np.std(matched_arr, ddof=1)) if len(matched_arr) > 1 else np.nan,
                "spatial_matched_gene_z": float((observed - np.mean(matched_arr)) / max(float(np.std(matched_arr, ddof=1)), 1e-12)) if len(matched_arr) > 1 else np.nan,
                "spatial_matched_gene_p": empirical_p_greater(observed, matched_arr) if len(matched_arr) else np.nan,
            }
        )
        edge_rows.append(
            {
                "axis_id": row.axis_id,
                "edge_count": int(len(edge_s)),
                "mean_distance": float(np.mean(edge_d)),
                "median_distance": float(np.median(edge_d)),
                "p90_distance": float(np.quantile(edge_d, 0.9)),
            }
        )
    out = pd.DataFrame(rows)
    out["spatial_null_fdr"] = bh_fdr(out.get("spatial_perm_p", pd.Series(dtype=float)))
    out["spatial_matched_gene_fdr"] = bh_fdr(out.get("spatial_matched_gene_p", pd.Series(dtype=float)))
    return out, pd.DataFrame(edge_rows)


def downstream_target_support(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    matrix,
    mean_by_celltype: np.ndarray,
    det_by_celltype: np.ndarray,
    gene_to_index: dict[str, int],
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _, indices = cell_type_codes(meta)
    global_mean, global_det = global_gene_summaries(mean_by_celltype, det_by_celltype, meta)
    protected = {idx for idx in gene_to_index.values() if idx < len(global_mean)}
    axis_to_config = {axis.axis_id: axis for axis in TARGET_AXES}
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        axis = axis_to_config[row.axis_id]
        receiver_idx = indices.get(row.receiver_celltype, np.array([], dtype=int))
        present = [gene_to_index[g.upper()] for g in axis.target_panel if g.upper() in gene_to_index]
        if len(receiver_idx) == 0 or not present:
            rows.append({"axis_id": row.axis_id, "downstream_status": "missing"})
            continue
        gene_scores = []
        null_scores: list[float] = []
        for gene_idx in present:
            expr = np.asarray(matrix.getrow(gene_idx).toarray()).ravel().astype(np.float32)
            observed = float(np.mean(expr[receiver_idx]) / max(float(np.mean(expr) + 1e-12), 1e-12))
            gene_scores.append(observed)
            pool = nearest_pool(gene_idx, global_mean, global_det, protected=set(), pool_size=800)
            if len(pool):
                sample = rng.choice(pool, size=n_permutations, replace=len(pool) < n_permutations)
                for idx in sample:
                    ctrl = np.asarray(matrix.getrow(int(idx)).toarray()).ravel().astype(np.float32)
                    null_scores.append(float(np.mean(ctrl[receiver_idx]) / max(float(np.mean(ctrl) + 1e-12), 1e-12)))
        observed_panel = float(np.mean(gene_scores))
        null_arr = np.asarray(null_scores, dtype=float)
        rows.append(
            {
                "axis_id": row.axis_id,
                "downstream_status": "success",
                "downstream_target_genes_present": len(present),
                "downstream_target_score": observed_panel,
                "downstream_target_null_mean": float(np.mean(null_arr)) if len(null_arr) else np.nan,
                "downstream_target_null_sd": float(np.std(null_arr, ddof=1)) if len(null_arr) > 1 else np.nan,
                "downstream_target_z": float((observed_panel - np.mean(null_arr)) / max(float(np.std(null_arr, ddof=1)), 1e-12)) if len(null_arr) > 1 else np.nan,
                "downstream_target_p": empirical_p_greater(observed_panel, null_arr) if len(null_arr) else np.nan,
                "downstream_target_gene_panel": ",".join([g for g in axis.target_panel if g.upper() in gene_to_index]),
            }
        )
    out = pd.DataFrame(rows)
    out["downstream_target_fdr"] = bh_fdr(out.get("downstream_target_p", pd.Series(dtype=float)))
    return out


def functional_received_signal_support(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    matrix,
    gene_to_index: dict[str, int],
    gene_cache: dict[str, np.ndarray],
    *,
    k_neighbors: int,
) -> pd.DataFrame:
    _, indices = cell_type_codes(meta)
    coords = meta[["x", "y"]].to_numpy(dtype=float)
    axis_to_config = {axis.axis_id: axis for axis in TARGET_AXES}
    rows: list[dict[str, Any]] = []
    panel_cache: dict[int, np.ndarray] = {}

    def gene_row(idx: int) -> np.ndarray:
        if idx not in panel_cache:
            panel_cache[idx] = np.asarray(matrix.getrow(idx).toarray()).ravel().astype(np.float32)
        return panel_cache[idx]

    for row in targets.itertuples(index=False):
        axis = axis_to_config[row.axis_id]
        lig = row_for_gene(gene_cache, row.ligand)
        rec = row_for_gene(gene_cache, row.receptor)
        sender_idx = indices.get(row.sender_celltype, np.array([], dtype=int))
        receiver_idx = indices.get(row.receiver_celltype, np.array([], dtype=int))
        present = [gene_to_index[g.upper()] for g in axis.target_panel if g.upper() in gene_to_index]
        if lig is None or rec is None or len(sender_idx) == 0 or len(receiver_idx) == 0 or not present:
            rows.append({"axis_id": row.axis_id, "functional_signal_status": "missing"})
            continue
        edge_s, edge_r, _ = build_spatial_edges(coords, receiver_idx, sender_idx, k=k_neighbors)
        if len(edge_s) == 0:
            rows.append({"axis_id": row.axis_id, "functional_signal_status": "no_edges"})
            continue
        # build received signal on receiver cells: receptor level multiplied by local sender ligand average
        receiver_to_values: dict[int, list[float]] = {}
        for rec_cell, sender_cell in zip(edge_s, edge_r):
            receiver_to_values.setdefault(int(rec_cell), []).append(float(lig[sender_cell]))
        receiver_cells = np.array(sorted(receiver_to_values), dtype=int)
        sender_lig_mean = np.array([np.mean(receiver_to_values[int(cell)]) for cell in receiver_cells], dtype=float)
        signal = rec[receiver_cells] * sender_lig_mean
        panel_expr = np.vstack([gene_row(idx)[receiver_cells] for idx in present])
        panel_score = np.mean(np.log1p(panel_expr), axis=0)
        if np.std(signal) == 0 or np.std(panel_score) == 0:
            rho = np.nan
            p = np.nan
        else:
            rho, p = spearmanr(signal, panel_score)
        top = signal >= np.quantile(signal, 0.8)
        bottom = signal <= np.quantile(signal, 0.2)
        rows.append(
            {
                "axis_id": row.axis_id,
                "functional_signal_status": "success",
                "received_signal_n_cells": int(len(receiver_cells)),
                "received_signal_target_spearman": float(rho) if np.isfinite(rho) else np.nan,
                "received_signal_target_p": float(p) if np.isfinite(p) else np.nan,
                "received_signal_top_vs_bottom_target_delta": float(np.mean(panel_score[top]) - np.mean(panel_score[bottom])) if np.any(top) and np.any(bottom) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["received_signal_target_fdr"] = bh_fdr(out.get("received_signal_target_p", pd.Series(dtype=float)))
    return out


def contamination_controls(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    matrix,
    gene_to_index: dict[str, int],
) -> pd.DataFrame:
    _, indices = cell_type_codes(meta)
    rows: list[dict[str, Any]] = []

    def marker_score(markers: tuple[str, ...], cells: np.ndarray) -> tuple[float, float, int]:
        present = [gene_to_index[m.upper()] for m in markers if m.upper() in gene_to_index]
        if not present or len(cells) == 0:
            return np.nan, np.nan, 0
        means = []
        dets = []
        for idx in present:
            expr = np.asarray(matrix.getrow(idx).toarray()).ravel().astype(np.float32)
            means.append(float(np.mean(expr[cells])))
            dets.append(float(np.mean(expr[cells] > 0)))
        return float(np.mean(means)), float(np.mean(dets)), len(present)

    for row in targets.itertuples(index=False):
        cells = np.unique(
            np.concatenate(
                [
                    indices.get(row.sender_celltype, np.array([], dtype=int)),
                    indices.get(row.receiver_celltype, np.array([], dtype=int)),
                ]
            )
        )
        cont_mean, cont_det, cont_n = marker_score(RBC_PLATELET_MARKERS, cells)
        endo_mean, endo_det, endo_n = marker_score(ENDOTHELIAL_MARKERS, cells)
        ratio = cont_mean / max(endo_mean, 1e-12) if np.isfinite(cont_mean) and np.isfinite(endo_mean) else np.nan
        flag = bool(np.isfinite(ratio) and ratio > 0.35 and np.isfinite(cont_det) and cont_det > 0.10)
        rows.append(
            {
                "axis_id": row.axis_id,
                "contamination_marker_mean": cont_mean,
                "contamination_marker_detection": cont_det,
                "contamination_marker_n": cont_n,
                "endothelial_marker_mean": endo_mean,
                "endothelial_marker_detection": endo_det,
                "endothelial_marker_n": endo_n,
                "contamination_to_endothelial_ratio": ratio,
                "contamination_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_stability(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    gene_cache: dict[str, np.ndarray],
    *,
    n_bootstraps: int,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cats, indices = cell_type_codes(meta)
    repeats: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for repeat in range(1, n_bootstraps + 1):
        sampled_by_type: dict[str, np.ndarray] = {}
        for cell_type, idx in indices.items():
            n = max(1, int(round(len(idx) * fraction)))
            sampled_by_type[cell_type] = rng.choice(idx, size=n, replace=False)
        scores = []
        for row in targets.itertuples(index=False):
            lig = row_for_gene(gene_cache, row.ligand)
            rec = row_for_gene(gene_cache, row.receptor)
            sidx = sampled_by_type.get(row.sender_celltype, np.array([], dtype=int))
            ridx = sampled_by_type.get(row.receiver_celltype, np.array([], dtype=int))
            if lig is None or rec is None or len(sidx) == 0 or len(ridx) == 0:
                score = np.nan
            else:
                lig_sender = float(np.mean(lig[sidx]) / max(float(np.mean(lig) + 1e-12), 1e-12))
                rec_receiver = float(np.mean(rec[ridx]) / max(float(np.mean(rec) + 1e-12), 1e-12))
                lig_det = float(np.mean(lig[sidx] > 0))
                rec_det = float(np.mean(rec[ridx] > 0))
                score = float(geometric_mean(np.array([lig_sender, rec_receiver, lig_det, rec_det])))
            scores.append((row.axis_id, score))
        score_values = np.array([s for _, s in scores], dtype=float)
        order = np.argsort(-np.nan_to_num(score_values, nan=-np.inf))
        rank_map = {scores[idx][0]: rank + 1 for rank, idx in enumerate(order)}
        for axis_id, score in scores:
            repeats.append({"axis_id": axis_id, "bootstrap_id": repeat, "bootstrap_validation_score": score, "bootstrap_rank": rank_map[axis_id]})
    repeat_df = pd.DataFrame(repeats)
    for axis_id, grp in repeat_df.groupby("axis_id"):
        summary.append(
            {
                "axis_id": axis_id,
                "bootstrap_rank_median": float(np.median(grp["bootstrap_rank"])),
                "bootstrap_rank_iqr": float(np.quantile(grp["bootstrap_rank"], 0.75) - np.quantile(grp["bootstrap_rank"], 0.25)),
                "bootstrap_score_mean": float(np.mean(grp["bootstrap_validation_score"])),
                "bootstrap_score_sd": float(np.std(grp["bootstrap_validation_score"], ddof=1)) if len(grp) > 1 else 0.0,
                "bootstrap_n": int(len(grp)),
            }
        )
    return pd.DataFrame(summary), repeat_df


def classify_v2(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in evidence.itertuples(index=False):
        expression_ok = bool(getattr(row, "ligand_sender_specificity_ratio", 0) >= 0.75 and getattr(row, "receptor_receiver_specificity_ratio", 0) >= 0.50)
        label_perm_ok = bool(pd.notna(getattr(row, "cell_label_perm_fdr", np.nan)) and getattr(row, "cell_label_perm_fdr") <= 0.05)
        spatial_ok = bool(pd.notna(getattr(row, "spatial_null_fdr", np.nan)) and getattr(row, "spatial_null_fdr") <= 0.10)
        matched_ok = bool(pd.notna(getattr(row, "matched_gene_percentile", np.nan)) and getattr(row, "matched_gene_percentile") >= 0.90)
        downstream_ok = bool(pd.notna(getattr(row, "downstream_target_fdr", np.nan)) and getattr(row, "downstream_target_fdr") <= 0.10)
        functional_ok = bool(
            (
                pd.notna(getattr(row, "received_signal_target_fdr", np.nan))
                and getattr(row, "received_signal_target_fdr") <= 0.10
                and getattr(row, "received_signal_target_spearman", 0) > 0
            )
            or (pd.notna(getattr(row, "received_signal_top_vs_bottom_target_delta", np.nan)) and getattr(row, "received_signal_top_vs_bottom_target_delta") > 0)
        )
        cross_ok = bool(getattr(row, "cross_method_exact_count", 0) >= 1 or getattr(row, "cross_method_same_lr_count", 0) >= 2)
        ablation_ok = bool(pd.notna(getattr(row, "max_rank_after_component_removal", np.nan)) and getattr(row, "max_rank_after_component_removal") <= 750)
        bootstrap_ok = bool(pd.notna(getattr(row, "bootstrap_rank_iqr", np.nan)) and getattr(row, "bootstrap_rank_iqr") <= 2.0)
        contamination_flag = bool(getattr(row, "contamination_flag", False))
        support_flags = {
            "expression_specificity_support": expression_ok,
            "cell_label_permutation_support": label_perm_ok,
            "spatial_null_support": spatial_ok,
            "matched_gene_control_support": matched_ok,
            "downstream_target_support": downstream_ok,
            "functional_received_signal_support": functional_ok,
            "cross_method_support": cross_ok,
            "component_ablation_support": ablation_ok,
            "bootstrap_stability_support": bootstrap_ok,
        }
        support_count = int(sum(support_flags.values()))
        if contamination_flag:
            evidence_class = "artifact_risk"
        elif support_count >= 5:
            evidence_class = "strong"
        elif support_count >= 3:
            evidence_class = "moderate"
        else:
            evidence_class = "hypothesis_only"
        interpretation = (
            f"{evidence_class}: {support_count}/9 computational evidence layers support the axis; "
            f"contamination_flag={contamination_flag}."
        )
        rows.append({"axis_id": row.axis_id, **support_flags, "support_count": support_count, "evidence_class": evidence_class, "interpretation_note": interpretation})
    return pd.DataFrame(rows)


def make_figures(evidence: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    support_cols = [
        "expression_specificity_support",
        "cell_label_permutation_support",
        "spatial_null_support",
        "matched_gene_control_support",
        "downstream_target_support",
        "functional_received_signal_support",
        "cross_method_support",
        "component_ablation_support",
        "bootstrap_stability_support",
    ]
    label_df = evidence.set_index("axis_label")[support_cols].astype(float)
    fig, ax = plt.subplots(figsize=(13.5, max(4.5, 0.55 * len(label_df) + 1.8)))
    im = ax.imshow(label_df.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(support_cols)))
    ax.set_xticklabels(
        [
            "Expression",
            "Label\nperm",
            "Spatial\nnull",
            "Matched\ngenes",
            "Downstream",
            "Received\nsignal",
            "Cross\nmethod",
            "Ablation",
            "Bootstrap",
        ],
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(label_df)))
    ax.set_yticklabels(label_df.index, fontsize=8)
    ax.set_title("TopoLink-CCI validation v2: multi-layer false-positive controls", fontsize=13, weight="bold")
    for i in range(label_df.shape[0]):
        for j in range(label_df.shape[1]):
            if label_df.iat[i, j] > 0:
                ax.text(j, i, "✓", ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(figures / "topolink_cci_validation_v2_evidence_matrix.png", dpi=300)
    fig.savefig(figures / "topolink_cci_validation_v2_evidence_matrix.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    plot = evidence.sort_values("CCI_score", ascending=True)
    colors = plot["evidence_class"].map({"strong": "#0f766e", "moderate": "#f59e0b", "hypothesis_only": "#64748b", "artifact_risk": "#dc2626"}).fillna("#64748b")
    ax.barh(plot["ligand"] + "-" + plot["receptor"], plot["support_count"], color=colors)
    ax.set_xlabel("Supported evidence layers (max 9)")
    ax.set_xlim(0, 9)
    ax.set_title("Validation support count by LR axis")
    fig.tight_layout()
    fig.savefig(figures / "topolink_cci_validation_v2_support_counts.png", dpi=300)
    fig.savefig(figures / "topolink_cci_validation_v2_support_counts.pdf")
    plt.close(fig)


def write_validation_cards(evidence: pd.DataFrame, output_dir: Path) -> None:
    cards = output_dir / "reports" / "validation_cards"
    cards.mkdir(parents=True, exist_ok=True)
    for row in evidence.itertuples(index=False):
        path = cards / f"{row.ligand}_{row.receptor}_{row.sender}_{row.receiver}.md".replace("/", "-").replace(" ", "_").replace(",", "")
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"# {row.ligand}-{row.receptor}: {row.sender} -> {row.receiver}\n\n")
            handle.write(f"- Biology label: {row.biology_label}\n")
            handle.write(f"- pyXenium CCI_score: {row.CCI_score:.6g}\n")
            handle.write(f"- pyXenium rank: {int(row.pyxenium_rank)}\n")
            handle.write(f"- Evidence class: {row.evidence_class}\n")
            handle.write(f"- Supported layers: {int(row.support_count)}/9\n")
            handle.write(f"- Interpretation: {row.interpretation_note}\n\n")
            handle.write("## Key controls\n\n")
            cols = [
                "cell_label_perm_fdr",
                "spatial_null_fdr",
                "matched_gene_z",
                "matched_gene_percentile",
                "downstream_target_score",
                "downstream_target_fdr",
                "received_signal_target_spearman",
                "cross_method_exact_count",
                "cross_method_same_lr_count",
                "bootstrap_rank_median",
                "bootstrap_rank_iqr",
                "max_rank_after_component_removal",
                "contamination_flag",
            ]
            for col in cols:
                if hasattr(row, col):
                    handle.write(f"- `{col}`: {getattr(row, col)}\n")
            handle.write("\nThis is computational support only; it does not prove protein-level binding or functional signaling.\n")


def write_report(evidence: pd.DataFrame, output_dir: Path) -> None:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / "topolink_cci_validation_v2_report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# TopoLink-CCI validation v2\n\n")
        handle.write(
            "This report applies the main computational false-positive controls used by classic LR/CCC papers to TopoLink-CCI axes. "
            "The result should be read as computational validation, not wet-lab proof.\n\n"
        )
        handle.write("## Summary\n\n")
        counts = evidence["evidence_class"].value_counts().to_dict()
        for cls in ["strong", "moderate", "hypothesis_only", "artifact_risk"]:
            handle.write(f"- {cls}: {counts.get(cls, 0)}\n")
        handle.write("\n")
        cols = [
            "ligand",
            "receptor",
            "sender",
            "receiver",
            "CCI_score",
            "pyxenium_rank",
            "evidence_class",
            "support_count",
            "cell_label_perm_fdr",
            "spatial_null_fdr",
            "matched_gene_z",
            "downstream_target_fdr",
            "cross_method_same_lr_count",
            "bootstrap_rank_median",
            "contamination_flag",
        ]
        handle.write(markdown_table(evidence[cols], floatfmt=".3g"))
        handle.write("\n\n## Evidence layers implemented\n\n")
        handle.write("- CellPhoneDB/Squidpy-style cell-label permutation of CCI communication probability.\n")
        handle.write("- CellChat-style sender/receiver group specificity and permutation significance.\n")
        handle.write("- stLearn-style spatial-neighborhood LR co-expression plus matched-expression random gene pairs.\n")
        handle.write("- SpatialDM-style spatial expression null based on CCI neighborhood coupling.\n")
        handle.write("- NicheNet-style receiver target/pathway support using predefined biology panels.\n")
        handle.write("- COMMOT/SpaTalk-style received-signal association with receiver target programs.\n")
        handle.write("- LIANA-style cross-method consensus across completed benchmark methods.\n")
        handle.write("- pyXenium component ablation and stratified bootstrap stability.\n")
        handle.write("\n## Caveat\n\n")
        handle.write("The framework reduces false-positive risk, but protein-level receptor binding, secretion, and functional causality require orthogonal experimental validation.\n")


def make_control_rows(
    cell_perm: pd.DataFrame,
    spatial: pd.DataFrame,
    downstream: pd.DataFrame,
    functional: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for df, control_name, value_cols in [
        (cell_perm, "cell_label_permutation", ["cell_label_comm_prob", "cell_label_perm_mean", "cell_label_perm_z", "cell_label_perm_p", "cell_label_perm_fdr"]),
        (spatial, "spatial_neighborhood_permutation", ["spatial_lr_edge_score", "spatial_perm_mean", "spatial_perm_z", "spatial_perm_p", "spatial_null_fdr"]),
        (spatial, "spatial_matched_gene_pairs", ["spatial_lr_edge_score", "spatial_matched_gene_mean", "spatial_matched_gene_z", "spatial_matched_gene_p", "spatial_matched_gene_fdr"]),
        (downstream, "downstream_target_enrichment", ["downstream_target_score", "downstream_target_null_mean", "downstream_target_z", "downstream_target_p", "downstream_target_fdr"]),
        (functional, "received_signal_target_correlation", ["received_signal_target_spearman", "received_signal_target_p", "received_signal_target_fdr", "received_signal_top_vs_bottom_target_delta"]),
        (bootstrap, "bootstrap_stability", ["bootstrap_rank_median", "bootstrap_rank_iqr", "bootstrap_score_mean", "bootstrap_score_sd"]),
    ]:
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            item = {"axis_id": row.axis_id, "control_type": control_name}
            for col in value_cols:
                if hasattr(row, col):
                    item[col] = getattr(row, col)
            rows.append(item)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "runs" / "validation" / "topolink_cci_validation_v2"
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    reports = output_dir / "reports"
    for folder in (output_dir, tables, figures, reports):
        folder.mkdir(parents=True, exist_ok=True)

    with (output_dir / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    manifest = load_manifest(root)
    scores = read_scores(root)
    if len(scores) != args.expected_score_rows:
        raise RuntimeError(f"Expected {args.expected_score_rows} pyXenium rows, observed {len(scores)}")
    targets = find_target_rows(scores)
    targets = targets.rename(columns={"global_rank": "pyxenium_rank", "sender_celltype": "sender", "receiver_celltype": "receiver"})
    if not ((targets["ligand"] == "VWF") & (targets["receptor"] == "SELP") & (targets["pyxenium_rank"] == 1)).any():
        raise RuntimeError("VWF-SELP was not detected as pyXenium rank 1 in the selected target table.")

    # Rebuild the column names expected by the v1 helper functions.
    helper_targets = targets.rename(columns={"pyxenium_rank": "global_rank", "sender": "sender_celltype", "receiver": "receiver_celltype"})

    ablation = component_ablation(scores, helper_targets)
    genes, meta, matrix = load_bundle_expression(manifest)
    expr_summary, mean_by_celltype, det_by_celltype, gene_to_index = summarize_expression(genes, meta, matrix)

    all_needed_genes = sorted(
        {
            g
            for axis in TARGET_AXES
            for g in (axis.ligand, axis.receptor, *axis.target_panel, *RBC_PLATELET_MARKERS, *ENDOTHELIAL_MARKERS)
        }
    )
    gene_cache = dense_gene_cache(matrix, gene_to_index, all_needed_genes)

    expr = expression_specificity(helper_targets, meta, mean_by_celltype, det_by_celltype, gene_to_index)
    matched = matched_gene_controls(helper_targets, meta, mean_by_celltype, det_by_celltype, gene_to_index, n_controls=args.n_matched_controls, seed=args.seed)
    abundance_null = spatial_abundance_null(scores, helper_targets, meta)
    cross_summary, cross_detail = cross_method_support(root, helper_targets)
    receiver_context = receiver_panel_support(helper_targets, meta, mean_by_celltype, gene_to_index)
    cell_perm = cell_label_permutation(helper_targets, meta, gene_cache, n_permutations=args.n_label_permutations, seed=args.seed + 1)
    spatial, spatial_edges = spatial_neighborhood_controls(
        helper_targets,
        meta,
        matrix,
        mean_by_celltype,
        det_by_celltype,
        gene_to_index,
        gene_cache,
        k_neighbors=args.k_neighbors,
        n_permutations=args.n_spatial_permutations,
        n_matched_pairs=args.n_spatial_matched_pairs,
        seed=args.seed + 2,
    )
    downstream = downstream_target_support(
        helper_targets,
        meta,
        matrix,
        mean_by_celltype,
        det_by_celltype,
        gene_to_index,
        n_permutations=args.n_downstream_permutations,
        seed=args.seed + 3,
    )
    functional = functional_received_signal_support(helper_targets, meta, matrix, gene_to_index, gene_cache, k_neighbors=args.k_neighbors)
    contamination = contamination_controls(helper_targets, meta, matrix, gene_to_index)
    bootstrap, bootstrap_repeats = bootstrap_stability(
        helper_targets,
        meta,
        gene_cache,
        n_bootstraps=args.n_bootstraps,
        fraction=args.bootstrap_fraction,
        seed=args.seed + 4,
    )

    ablation_summary = (
        ablation.groupby("axis_id")["rank_without_component"]
        .agg(max_rank_after_ablation="max", min_rank_after_ablation="min")
        .reset_index()
    )
    evidence = (
        targets[
            [
                "axis_id",
                "biology_label",
                "ligand",
                "receptor",
                "sender",
                "receiver",
                "CCI_score",
                "pyxenium_rank",
                *COMPONENTS,
                "cross_edge_count",
            ]
        ]
        .merge(expr.rename(columns={"sender": "sender_expr_label", "receiver": "receiver_expr_label"}), on=["axis_id", "ligand", "receptor"], how="left")
        .merge(abundance_null, on="axis_id", how="left")
        .merge(matched, on="axis_id", how="left")
        .merge(ablation_summary, on="axis_id", how="left")
        .merge(cross_summary, on="axis_id", how="left")
        .merge(receiver_context, on="axis_id", how="left")
        .merge(cell_perm, on="axis_id", how="left")
        .merge(spatial, on="axis_id", how="left")
        .merge(downstream, on="axis_id", how="left")
        .merge(functional, on="axis_id", how="left")
        .merge(contamination, on="axis_id", how="left")
        .merge(bootstrap, on="axis_id", how="left")
    )
    classes = classify_v2(evidence)
    evidence = evidence.merge(classes, on="axis_id", how="left")
    evidence["axis_label"] = evidence["ligand"] + "-" + evidence["receptor"] + "\n" + evidence["sender"] + " -> " + evidence["receiver"]
    evidence = evidence.sort_values("CCI_score", ascending=False, kind="mergesort")

    classic_controls = build_controls_table(abundance_null, matched, ablation)
    v2_controls = make_control_rows(cell_perm, spatial, downstream, functional, bootstrap)
    control_table = pd.concat([classic_controls, v2_controls], ignore_index=True, sort=False)

    evidence.to_csv(tables / "topolink_cci_validation_v2_evidence.tsv", sep="\t", index=False)
    control_table.to_csv(tables / "topolink_cci_validation_v2_false_positive_controls.tsv", sep="\t", index=False)
    cell_perm.to_csv(tables / "cell_label_permutation.tsv", sep="\t", index=False)
    spatial.to_csv(tables / "spatial_neighborhood_controls.tsv", sep="\t", index=False)
    spatial_edges.to_csv(tables / "spatial_edge_summary.tsv", sep="\t", index=False)
    downstream.to_csv(tables / "downstream_target_support.tsv", sep="\t", index=False)
    functional.to_csv(tables / "functional_received_signal_support.tsv", sep="\t", index=False)
    contamination.to_csv(tables / "contamination_controls.tsv", sep="\t", index=False)
    bootstrap.to_csv(tables / "bootstrap_stability_summary.tsv", sep="\t", index=False)
    bootstrap_repeats.to_csv(tables / "bootstrap_stability_repeats.tsv", sep="\t", index=False)
    cross_detail.to_csv(tables / "cross_method_support_detail.tsv", sep="\t", index=False)
    expr_summary.to_csv(tables / "expression_gene_specificity.tsv", sep="\t", index=False)
    ablation.to_csv(tables / "component_ablation.tsv", sep="\t", index=False)

    make_figures(evidence, output_dir)
    write_report(evidence, output_dir)
    write_validation_cards(evidence, output_dir)

    summary = {
        "status": "success",
        "runtime_seconds": round(time.time() - started, 3),
        "n_scores": int(len(scores)),
        "n_target_axes": int(len(evidence)),
        "evidence_class_counts": evidence["evidence_class"].value_counts().to_dict(),
        "outputs": {
            "evidence": str(tables / "topolink_cci_validation_v2_evidence.tsv"),
            "controls": str(tables / "topolink_cci_validation_v2_false_positive_controls.tsv"),
            "report": str(reports / "topolink_cci_validation_v2_report.md"),
            "figure": str(figures / "topolink_cci_validation_v2_evidence_matrix.png"),
        },
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full TopoLink-CCI validation v2 false-positive controls.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-score-rows", type=int, default=1_319_600)
    parser.add_argument("--n-label-permutations", type=int, default=500)
    parser.add_argument("--n-spatial-permutations", type=int, default=300)
    parser.add_argument("--n-spatial-matched-pairs", type=int, default=120)
    parser.add_argument("--n-matched-controls", type=int, default=250)
    parser.add_argument("--n-downstream-permutations", type=int, default=120)
    parser.add_argument("--n-bootstraps", type=int, default=5)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260428)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
