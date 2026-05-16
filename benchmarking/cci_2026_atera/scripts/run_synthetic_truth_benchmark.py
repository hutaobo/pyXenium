from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_synthetic_cci_truth import generate_synthetic_cci_truth
from pyXenium.cci import cci_topology_analysis


POSITIVE_AXES = {
    "PLANTED_L|PLANTED_R|SenderA|ReceiverB",
    "CONTACT_L|CONTACT_R|SenderA|ReceiverB",
}


def _axis_id(row: pd.Series) -> str:
    return f"{row['ligand']}|{row['receptor']}|{row['sender_celltype']}|{row['receiver_celltype']}"


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = float(len(pos) * len(neg))
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / total


def _auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(1, int(y_true.sum()))
    precision = tp / np.maximum(1, tp + fp)
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def _f1_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float, float]:
    order = np.argsort(-scores)[:k]
    tp = int(y_true[order].sum())
    precision = tp / max(1, k)
    recall = tp / max(1, int(y_true.sum()))
    denom = precision + recall
    f1 = 0.0 if denom == 0 else 2.0 * precision * recall / denom
    return precision, recall, f1


def _score_columns(scores: pd.DataFrame) -> dict[str, pd.Series]:
    cols: dict[str, pd.Series] = {"TopoLink-CCI": pd.to_numeric(scores["CCI_score"], errors="coerce")}
    if {"sender_expr", "receiver_expr"}.issubset(scores.columns):
        cols["expression_only"] = (
            pd.to_numeric(scores["sender_expr"], errors="coerce").fillna(0.0)
            * pd.to_numeric(scores["receiver_expr"], errors="coerce").fillna(0.0)
        )
    if "local_contact" in scores.columns:
        cols["contact_only"] = pd.to_numeric(scores["local_contact"], errors="coerce").fillna(0.0)
    anchor_cols = [c for c in ("sender_anchor", "receiver_anchor", "structure_bridge") if c in scores.columns]
    if anchor_cols:
        values = scores[anchor_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        cols["topology_anchor_only"] = values.prod(axis=1) ** (1.0 / len(anchor_cols))
    return cols


def run_synthetic_truth_benchmark(*, output_dir: str | Path, cells_per_type: int, seed: int) -> dict[str, object]:
    start = time.time()
    output = Path(output_dir)
    data_dir = output / "data"
    tables = output / "tables"
    figures = output / "figures"
    for folder in (output, data_dir, tables, figures):
        folder.mkdir(parents=True, exist_ok=True)

    synthetic = generate_synthetic_cci_truth(output_dir=data_dir, cells_per_type=cells_per_type, seed=seed)
    reference = pd.read_csv(synthetic["reference_tsv"], sep="\t")
    expression = pd.read_csv(synthetic["expression_tsv"], sep="\t", index_col=0)
    pairs = pd.read_csv(synthetic["interaction_pairs_tsv"], sep="\t")
    t_and_c = pd.read_csv(synthetic["t_and_c_tsv"], sep="\t", index_col=0)
    structure_map = pd.read_csv(synthetic["structure_map_tsv"], sep="\t", index_col=0)

    result = cci_topology_analysis(
        reference_df=reference,
        expression_df=expression,
        interaction_pairs=pairs,
        t_and_c_df=t_and_c,
        structure_map_df=structure_map,
        anchor_mode="precomputed",
        min_cross_edges=1,
        top_n_pairs=8,
        export_figures=False,
        return_hotspots=False,
        random_state=seed,
    )
    scores = result["scores"].copy()
    scores["axis_id"] = scores.apply(_axis_id, axis=1)
    scores["truth_label"] = np.where(scores["axis_id"].isin(POSITIVE_AXES), 1, 0)
    scores.to_csv(tables / "synthetic_truth_scores.tsv", sep="\t", index=False)

    y_true = scores["truth_label"].to_numpy(dtype=int)
    metric_rows = []
    for method, values in _score_columns(scores).items():
        score_values = values.fillna(float("-inf")).to_numpy(dtype=float)
        row = {
            "method": method,
            "auroc": _auroc(y_true, score_values),
            "auprc": _auprc(y_true, score_values),
            "n_positive_axes": int(y_true.sum()),
            "n_candidate_axes": int(len(y_true)),
        }
        for k in (2, 5, 10):
            precision, recall, f1 = _f1_at_k(y_true, score_values, k)
            row[f"precision_at_{k}"] = precision
            row[f"recall_at_{k}"] = recall
            row[f"f1_at_{k}"] = f1
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values("auprc", ascending=False)
    metrics.to_csv(tables / "synthetic_truth_metrics.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    y = np.arange(len(metrics))
    ax.barh(y - 0.18, metrics["auroc"], height=0.34, label="AUROC", color="#4e79a7")
    ax.barh(y + 0.18, metrics["auprc"], height=0.34, label="AUPRC", color="#1f9d8a")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics["method"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("synthetic truth metric")
    ax.set_title("Topology-preserving synthetic CCI truth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "synthetic_truth_auroc_auprc.png", dpi=300)
    fig.savefig(figures / "synthetic_truth_auroc_auprc.pdf")
    plt.close(fig)

    summary = {
        "status": "success",
        "elapsed_seconds": time.time() - start,
        "cells_per_type": int(cells_per_type),
        "seed": int(seed),
        "n_cells": synthetic["n_cells"],
        "n_pairs": synthetic["n_pairs"],
        "outputs": {
            "scores": str(tables / "synthetic_truth_scores.tsv"),
            "metrics": str(tables / "synthetic_truth_metrics.tsv"),
            "figure_png": str(figures / "synthetic_truth_auroc_auprc.png"),
            "figure_pdf": str(figures / "synthetic_truth_auroc_auprc.pdf"),
        },
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "params.json").write_text(
        json.dumps({"cells_per_type": cells_per_type, "seed": seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic topology-preserving CCI truth benchmark.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cells-per-type", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()
    print(json.dumps(run_synthetic_truth_benchmark(output_dir=args.output_dir, cells_per_type=args.cells_per_type, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
