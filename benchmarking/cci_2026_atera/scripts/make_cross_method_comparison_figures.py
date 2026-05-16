from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_method_comparison_20260511"
FIG_DIR = OUT / "figures"
SRC_DIR = OUT / "source_data"
REMOTE_DIR = OUT / "raw_remote"

TOP_N_READ = 5000
TOP_K_CONSISTENCY = 100
F1_KS = (50, 500)


METHODS = {
    "TopoLink-CCI": {
        "kind": "full",
        "standardized": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/pyxenium/pyxenium_standardized.tsv",
        "summary": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/pyxenium/run_summary.json",
    },
    "CellPhoneDB": {
        "kind": "full",
        "standardized": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/cellphonedb/cellphonedb_standardized.tsv.gz",
        "summary": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/cellphonedb/run_summary.json",
    },
    "LARIS": {
        "kind": "full",
        "standardized": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/laris/laris_standardized.tsv.gz",
        "summary": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/laris/run_summary.json",
    },
    "LIANA+": {
        "kind": "full",
        "standardized": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/liana/liana_standardized.tsv.gz",
        "summary": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/liana/run_summary.json",
    },
    "SpatialDM": {
        "kind": "full",
        "standardized": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/spatialdm/spatialdm_standardized.tsv.gz",
        "summary": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/spatialdm/run_summary.json",
    },
    "stLearn": {
        "kind": "full",
        "standardized": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/stlearn/stlearn_standardized.tsv.gz",
        "summary": ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/stlearn/run_summary.json",
    },
    "Squidpy": {
        "kind": "full",
        "standardized": ROOT / "runs/a100_collected/full_common/squidpy/squidpy_standardized.tsv",
        "summary": ROOT / "runs/a100_collected/full_common/squidpy/run_summary.json",
    },
    "CellChat": {
        "kind": "full_lr_only",
        "standardized": REMOTE_DIR / "cellchat_standardized.tsv.gz",
        "summary": REMOTE_DIR / "cellchat_run_summary.json",
    },
    "COMMOT": {
        "kind": "chunked_full",
        "standardized": REMOTE_DIR / "commot_standardized.tsv.gz",
        "summary": None,
    },
    "Giotto": {
        "kind": "bounded_50k",
        "standardized": REMOTE_DIR / "giotto_standardized.tsv.gz",
        "summary": REMOTE_DIR / "giotto_run_summary.json",
    },
    "CellAgentChat": {
        "kind": "bounded_50k",
        "standardized": REMOTE_DIR / "cellagentchat_standardized.tsv.gz",
        "summary": REMOTE_DIR / "cellagentchat_run_summary.json",
    },
    "CellNEST": {
        "kind": "bounded_50k",
        "standardized": REMOTE_DIR / "cellnest_standardized.tsv.gz",
        "summary": REMOTE_DIR / "cellnest_run_summary.json",
    },
}


CANONICAL_PAIRS = {
    "VWF|SELP",
    "VWF|LRP1",
    "MMRN2|CD93",
    "DLL4|NOTCH3",
    "CXCL12|CXCR4",
    "CD48|CD2",
    "JAG1|NOTCH2",
    "VEGFC|FLT1",
    "COL4A2|CD93",
    "HSPG2|LRP1",
}
CANONICAL_AXES = {
    "VWF|SELP|Endothelial Cells|Endothelial Cells",
    "VWF|LRP1|Endothelial Cells|CAFs, DCIS Associated",
    "MMRN2|CD93|Endothelial Cells|Endothelial Cells",
    "DLL4|NOTCH3|Endothelial Cells|Pericytes",
    "CXCL12|CXCR4|CAFs, DCIS Associated|T Lymphocytes",
    "CD48|CD2|T Lymphocytes|T Lymphocytes",
    "JAG1|NOTCH2|11q13 Invasive Tumor Cells|11q13 Invasive Tumor Cells",
    "VEGFC|FLT1|Endothelial Cells|Endothelial Cells",
    "COL4A2|CD93|Endothelial Cells|Endothelial Cells",
    "HSPG2|LRP1|Endothelial Cells|CAFs, DCIS Associated",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, name: str, source: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(SRC_DIR / f"{name}.tsv", sep="\t", index=False)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def canonical_method_name(name: str) -> str:
    return name.strip()


def read_top_table(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, sep="\t", nrows=TOP_N_READ, compression="infer", low_memory=False)
    required = {"ligand", "receptor", "sender", "receiver"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "rank_within_method" in data.columns:
        data = data.sort_values("rank_within_method", ascending=True)
    elif "score_std" in data.columns:
        data = data.sort_values("score_std", ascending=False)
    elif "score_raw" in data.columns:
        data = data.sort_values("score_raw", ascending=False)
    data = data.head(TOP_N_READ).copy()
    for col in ["ligand", "receptor", "sender", "receiver"]:
        data[col] = data[col].astype(str)
    data["method_label"] = method
    data["pair_id"] = data["ligand"] + "|" + data["receptor"]
    data["axis_id"] = data["pair_id"] + "|" + data["sender"] + "|" + data["receiver"]
    data["sender_receiver_id"] = data["sender"] + "|" + data["receiver"]
    data["rank_for_eval"] = np.arange(1, len(data) + 1)
    return data


def runtime_table() -> pd.DataFrame:
    rows = []
    completion = pd.read_csv(ROOT / "results/method_completion_matrix.tsv", sep="\t")
    for method, cfg in METHODS.items():
        summary = read_json(cfg.get("summary"))
        row = {
            "method": method,
            "kind": cfg["kind"],
            "status": "available" if Path(cfg["standardized"]).exists() else "missing",
            "n_rows": summary.get("n_rows"),
            "elapsed_seconds": summary.get("elapsed_seconds", summary.get("runtime_seconds")),
            "standardized_path": str(cfg["standardized"]),
        }
        comp_match = completion[completion["method"].str.lower().str.replace("+", "", regex=False) == method.lower().replace("+", "")]
        if pd.isna(row["n_rows"]) or row["n_rows"] is None:
            if not comp_match.empty:
                row["n_rows"] = pd.to_numeric(comp_match.iloc[0].get("n_rows"), errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows)


def f1_for_set(pred: set[str], truth: set[str]) -> tuple[float, float, float, int]:
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp


def comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def binary_ari(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    contingency = np.zeros((2, 2), dtype=int)
    for x, y in zip(a, b):
        contingency[int(x), int(y)] += 1
    sum_comb = sum(comb2(int(v)) for v in contingency.ravel())
    row_comb = sum(comb2(int(v)) for v in contingency.sum(axis=1))
    col_comb = sum(comb2(int(v)) for v in contingency.sum(axis=0))
    total = comb2(n)
    if total == 0:
        return float("nan")
    expected = row_comb * col_comb / total
    maximum = 0.5 * (row_comb + col_comb)
    denom = maximum - expected
    return (sum_comb - expected) / denom if denom else 1.0


def collect_top_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    errors = []
    for method, cfg in METHODS.items():
        path = Path(cfg["standardized"])
        if not path.exists():
            errors.append({"method": method, "path": str(path), "error": "missing"})
            continue
        try:
            tables[method] = read_top_table(path, method)
        except Exception as exc:
            errors.append({"method": method, "path": str(path), "error": str(exc)})
    pd.DataFrame(errors).to_csv(SRC_DIR / "load_errors.tsv", sep="\t", index=False)
    return tables


def compute_recovery(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, data in tables.items():
        for k in F1_KS:
            top = data.head(k)
            pair_precision, pair_recall, pair_f1, pair_tp = f1_for_set(set(top["pair_id"]), CANONICAL_PAIRS)
            axis_precision, axis_recall, axis_f1, axis_tp = f1_for_set(set(top["axis_id"]), CANONICAL_AXES)
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "pair_precision": pair_precision,
                    "pair_recall": pair_recall,
                    "pair_f1": pair_f1,
                    "pair_tp": pair_tp,
                    "axis_precision": axis_precision,
                    "axis_recall": axis_recall,
                    "axis_f1": axis_f1,
                    "axis_tp": axis_tp,
                    "truth_n_pairs": len(CANONICAL_PAIRS),
                    "truth_n_axes": len(CANONICAL_AXES),
                }
            )
    return pd.DataFrame(rows)


def pairwise_metrics(tables: dict[str, pd.DataFrame], k: int = TOP_K_CONSISTENCY) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = list(tables)
    top_sets = {m: set(tables[m].head(k)["pair_id"]) for m in methods}
    universe = sorted(set().union(*top_sets.values()))
    rows = []
    for a, b in combinations(methods, 2):
        set_a = top_sets[a]
        set_b = top_sets[b]
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else float("nan")
        vec_a = np.array([item in set_a for item in universe], dtype=int)
        vec_b = np.array([item in set_b for item in universe], dtype=int)
        rows.append({"method_a": a, "method_b": b, "top_k": k, "jaccard": jaccard, "ari_binary_selection": binary_ari(vec_a, vec_b), "overlap_n": len(set_a & set_b)})
    pairwise = pd.DataFrame(rows)
    matrix_rows = []
    for metric in ["jaccard", "ari_binary_selection"]:
        for a in methods:
            for b in methods:
                if a == b:
                    value = 1.0
                else:
                    match = pairwise[((pairwise.method_a == a) & (pairwise.method_b == b)) | ((pairwise.method_a == b) & (pairwise.method_b == a))]
                    value = float(match.iloc[0][metric]) if not match.empty else float("nan")
                matrix_rows.append({"metric": metric, "method_a": a, "method_b": b, "value": value})
    return pairwise, pd.DataFrame(matrix_rows)


def topolink_overlap(tables: dict[str, pd.DataFrame], k: int = TOP_K_CONSISTENCY) -> pd.DataFrame:
    if "TopoLink-CCI" not in tables:
        return pd.DataFrame()
    ref_pairs = set(tables["TopoLink-CCI"].head(k)["pair_id"])
    ref_axes = set(tables["TopoLink-CCI"].head(k)["axis_id"])
    rows = []
    for method, data in tables.items():
        pairs = set(data.head(k)["pair_id"])
        axes = set(data.head(k)["axis_id"])
        rows.append(
            {
                "method": method,
                "top_k": k,
                "pair_overlap_n": len(ref_pairs & pairs),
                "pair_jaccard_to_topolink": len(ref_pairs & pairs) / len(ref_pairs | pairs) if ref_pairs | pairs else float("nan"),
                "axis_overlap_n": len(ref_axes & axes),
                "axis_jaccard_to_topolink": len(ref_axes & axes) / len(ref_axes | axes) if ref_axes | axes else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def canonical_rank_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, data in tables.items():
        pair_rank = data.drop_duplicates("pair_id").set_index("pair_id")["rank_for_eval"].to_dict()
        axis_rank = data.drop_duplicates("axis_id").set_index("axis_id")["rank_for_eval"].to_dict()
        for pair in sorted(CANONICAL_PAIRS):
            rows.append({"method": method, "canonical_id": pair, "level": "pair", "rank": pair_rank.get(pair, math.nan), "found_top_n": pair in pair_rank})
        for axis in sorted(CANONICAL_AXES):
            rows.append({"method": method, "canonical_id": axis, "level": "axis", "rank": axis_rank.get(axis, math.nan), "found_top_n": axis in axis_rank})
    return pd.DataFrame(rows)


def plot_runtime(runtime: pd.DataFrame) -> None:
    data = runtime.copy()
    data["hours"] = pd.to_numeric(data["elapsed_seconds"], errors="coerce") / 3600.0
    data["n_rows"] = pd.to_numeric(data["n_rows"], errors="coerce")
    data = data.sort_values("hours", na_position="last")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = data["kind"].map({"full": "#1f9d8a", "full_lr_only": "#8fb339", "chunked_full": "#4e79a7", "bounded_50k": "#f2a541"}).fillna("#999999")
    ax.barh(data["method"], data["hours"], color=colors)
    ax.set_xlabel("runtime (hours; missing values omitted)")
    ax.set_title("Runtime of completed benchmark outputs")
    for y, row in enumerate(data.itertuples(index=False)):
        if pd.notna(row.hours):
            ax.text(row.hours + 0.15, y, f"{row.hours:.1f} h", va="center", fontsize=6)
        else:
            ax.text(0.1, y, "runtime n/a", va="center", fontsize=6, color="#777777")
    save(fig, "fig1_runtime_hours", data)


def plot_recovery(recovery: pd.DataFrame) -> None:
    data = recovery[recovery["k"] == 500].sort_values("pair_f1")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    y = np.arange(len(data))
    ax.barh(y - 0.18, data["pair_f1"], height=0.34, color="#1f9d8a", label="pair-level F1@500")
    ax.barh(y + 0.18, data["axis_f1"], height=0.34, color="#4e79a7", label="sender-receiver exact F1@500")
    ax.set_yticks(y)
    ax.set_yticklabels(data["method"])
    ax.set_xlim(0, max(0.35, float(data[["pair_f1", "axis_f1"]].max().max()) * 1.2))
    ax.set_xlabel("canonical recovery F1")
    ax.set_title("Canonical CCI recovery")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig2_canonical_recovery_f1", data)


def plot_recovery_multik(recovery: pd.DataFrame) -> None:
    data = recovery[recovery["k"].isin([50, 500])].copy()
    order = (
        data[data["k"] == 50]
        .sort_values(["pair_f1", "axis_f1"], ascending=False)["method"]
        .tolist()
    )
    data["method"] = pd.Categorical(data["method"], categories=order, ordered=True)
    data = data.sort_values(["method", "k"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    for ax, metric, title in [
        (axes[0], "pair_f1", "Pair-level canonical recovery"),
        (axes[1], "axis_f1", "Exact sender-receiver recovery"),
    ]:
        pivot = data.pivot(index="method", columns="k", values=metric).loc[order]
        y = np.arange(len(pivot))
        ax.barh(y - 0.18, pivot[50], height=0.34, color="#1f9d8a", label="F1@50")
        ax.barh(y + 0.18, pivot[500], height=0.34, color="#4e79a7", label="F1@500")
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.36, float(pivot.max().max()) * 1.2))
        ax.set_xlabel("F1")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Canonical CCI recovery at top-50 and top-500")
    fig.tight_layout()
    save(fig, "fig2b_canonical_recovery_f1_top50_top500", data)


def plot_heatmap(matrix: pd.DataFrame, metric: str, name: str, title: str, cmap: str = "viridis") -> None:
    data = matrix[matrix["metric"] == metric].copy()
    pivot = data.pivot(index="method_a", columns="method_b", values="value")
    methods = list(pivot.index)
    fig, ax = plt.subplots(figsize=(7.8, 6.7))
    im = ax.imshow(pivot.values, vmin=-0.05 if "ari" in metric else 0, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color="white" if val > 0.45 else "black")
    save(fig, name, data)


def plot_topolink_overlap(overlap: pd.DataFrame) -> None:
    data = overlap.sort_values("pair_jaccard_to_topolink")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(data["method"], data["pair_jaccard_to_topolink"], color="#1f9d8a")
    ax.set_xlabel(f"Jaccard overlap with TopoLink-CCI top {TOP_K_CONSISTENCY} pairs")
    ax.set_title("Agreement with TopoLink-CCI top-ranked pairs")
    for y, row in enumerate(data.itertuples(index=False)):
        ax.text(row.pair_jaccard_to_topolink + 0.01, y, f"n={row.pair_overlap_n}", va="center", fontsize=6)
    save(fig, "fig5_overlap_with_topolink", data)


def plot_canonical_rank(rank_table: pd.DataFrame) -> None:
    data = rank_table[rank_table["level"] == "pair"].copy()
    data["rank_capped"] = pd.to_numeric(data["rank"], errors="coerce").fillna(TOP_N_READ + 1)
    data["score"] = -np.log10(data["rank_capped"])
    pivot = data.pivot(index="canonical_id", columns="method", values="score")
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(pivot.values, cmap="magma", vmin=0, vmax=np.nanmax(pivot.values))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels([x.replace("|", "-") for x in pivot.index])
    ax.set_title("Canonical pair rank visibility (-log10 rank; top 5000 scan)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "fig6_canonical_pair_rank_heatmap", data)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    tables = collect_top_tables()
    runtime = runtime_table()
    recovery = compute_recovery(tables)
    pairwise, matrix = pairwise_metrics(tables)
    overlap = topolink_overlap(tables)
    ranks = canonical_rank_matrix(tables)

    runtime.to_csv(SRC_DIR / "runtime_table.tsv", sep="\t", index=False)
    recovery.to_csv(SRC_DIR / "canonical_recovery_metrics.tsv", sep="\t", index=False)
    pairwise.to_csv(SRC_DIR / "pairwise_top100_consistency.tsv", sep="\t", index=False)
    matrix.to_csv(SRC_DIR / "pairwise_top100_matrix.tsv", sep="\t", index=False)
    overlap.to_csv(SRC_DIR / "topolink_top100_overlap.tsv", sep="\t", index=False)
    ranks.to_csv(SRC_DIR / "canonical_rank_matrix.tsv", sep="\t", index=False)

    plot_runtime(runtime)
    plot_recovery(recovery)
    plot_recovery_multik(recovery)
    plot_heatmap(matrix, "jaccard", "fig3_top100_pair_jaccard_heatmap", "Top-100 CCI pair Jaccard consistency")
    plot_heatmap(matrix, "ari_binary_selection", "fig4_top100_binary_selection_ari_heatmap", "Top-100 pair selection ARI (diagnostic)")
    plot_topolink_overlap(overlap)
    plot_canonical_rank(ranks)

    manifest = pd.DataFrame(
        {
            "figure": sorted(p.name for p in FIG_DIR.glob("*.png")),
            "path": [str(p) for p in sorted(FIG_DIR.glob("*.png"))],
        }
    )
    manifest.to_csv(OUT / "cross_method_figure_manifest.tsv", sep="\t", index=False)
    print(f"Wrote cross-method comparison figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
