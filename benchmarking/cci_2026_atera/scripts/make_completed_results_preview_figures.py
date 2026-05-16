from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "preview_completed_20260511"
FIG_DIR = OUT / "figures"
SRC_DIR = OUT / "source_data"


STATUS_ORDER = {
    "full_result": 0,
    "bounded_subset_result": 1,
    "reproducible_failure_card": 2,
    "running_rescue": 3,
    "pending_subset": 4,
}
STATUS_COLORS = {
    "full_result": "#1f9d8a",
    "bounded_subset_result": "#4e79a7",
    "reproducible_failure_card": "#d65f5f",
    "running_rescue": "#f2a541",
    "pending_subset": "#a0a0a0",
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


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summary_candidates() -> dict[str, list[Path]]:
    return {
        "CellPhoneDB": [ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/cellphonedb/run_summary.json"],
        "LARIS": [ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/laris/run_summary.json"],
        "LIANA+": [ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/liana/run_summary.json"],
        "SpatialDM": [ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/spatialdm/run_summary.json"],
        "stLearn": [ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/stlearn/run_summary.json"],
        "Squidpy": [ROOT / "runs/a100_collected/full_common/squidpy/run_summary.json"],
        "CellChat": [ROOT / "runs/a100_collected/full_common/cellchat/run_summary.json"],
        "TopoLink-CCI": [ROOT / "pdc_collected/pdc_20260426_1327/runs/full_common/pyxenium/run_summary.json"],
    }


def fill_completion_rows(completion: pd.DataFrame) -> pd.DataFrame:
    completion = completion.copy()
    for method, paths in summary_candidates().items():
        mask = (completion["dataset"] == "atera_breast_wta") & (completion["method"] == method)
        if not mask.any():
            continue
        for path in paths:
            if not path.exists():
                continue
            payload = read_json(path)
            if payload.get("n_rows") and completion.loc[mask, "n_rows"].isna().all():
                completion.loc[mask, "n_rows"] = int(payload["n_rows"])
            if payload.get("top_hit"):
                hit = payload["top_hit"][0]
                completion.loc[mask, "top_axis"] = f"{hit.get('ligand')}-{hit.get('receptor')}"
                completion.loc[mask, "top_sender_receiver"] = f"{hit.get('sender')} -> {hit.get('receiver')}"
                completion.loc[mask, "top_score"] = hit.get("score_raw")
            break
    cervical_summary = SRC_DIR / "cervical_full_pyxenium_run_summary.json"
    if cervical_summary.exists():
        payload = read_json(cervical_summary)
        mask = (completion["dataset"] == "atera_cervical_wta") & (completion["method"] == "TopoLink-CCI")
        if payload.get("top_hit") and mask.any():
            hit = payload["top_hit"][0]
            completion.loc[mask, "top_axis"] = f"{hit.get('ligand')}-{hit.get('receptor')}"
            completion.loc[mask, "top_sender_receiver"] = f"{hit.get('sender')} -> {hit.get('receiver')}"
            completion.loc[mask, "top_score"] = hit.get("score_raw")
    return completion


def plot_completion(completion: pd.DataFrame) -> None:
    data = completion.copy()
    data["status_rank"] = data["status"].map(STATUS_ORDER).fillna(99)
    data = data.sort_values(["dataset", "status_rank", "method"])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(data))
    colors = data["status"].map(STATUS_COLORS).fillna("#cccccc")
    ax.barh(y, np.ones(len(data)), color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["method"] + "  [" + data["dataset"].str.replace("atera_", "").str.replace("_wta", "") + "]")
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Benchmark completion status")
    for idx, row in enumerate(data.itertuples(index=False)):
        label = str(row.status).replace("_", " ")
        ax.text(0.02, idx, label, va="center", ha="left", color="white" if row.status in {"full_result", "bounded_subset_result"} else "black", fontsize=7)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=c, label=s.replace("_", " ")) for s, c in STATUS_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save(fig, "fig1_completion_status", data)


def plot_rows(completion: pd.DataFrame) -> None:
    data = completion[pd.to_numeric(completion["n_rows"], errors="coerce").notna()].copy()
    data["n_rows"] = pd.to_numeric(data["n_rows"], errors="coerce").astype(int)
    data["label"] = data["method"] + " (" + data["dataset"].str.replace("atera_", "").str.replace("_wta", "") + ")"
    data = data.sort_values("n_rows")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = data["status"].map(STATUS_COLORS).fillna("#999999")
    ax.barh(data["label"], data["n_rows"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("standardized rows (log scale)")
    ax.set_title("Completed result scale")
    for y, value in enumerate(data["n_rows"]):
        ax.text(value * 1.04, y, f"{value:,}", va="center", fontsize=6)
    save(fig, "fig2_result_row_counts", data)


def topolink_top_axes(path: Path, n: int, dataset_label: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", nrows=max(n, 30))
    score_col = "CCI_score" if "CCI_score" in data.columns else "LR_score"
    sender_col = "sender_celltype" if "sender_celltype" in data.columns else "sender"
    receiver_col = "receiver_celltype" if "receiver_celltype" in data.columns else "receiver"
    out = data.nlargest(n, score_col).copy()
    out["axis"] = out["ligand"].astype(str) + "-" + out["receptor"].astype(str)
    out["sender_receiver"] = out[sender_col].astype(str) + " -> " + out[receiver_col].astype(str)
    out["dataset"] = dataset_label
    out = out[["dataset", "axis", "sender_receiver", score_col, "local_contact", "cross_edge_count"]].rename(columns={score_col: "CCI_score"})
    return out


def plot_top_axes(df: pd.DataFrame, name: str, title: str) -> None:
    data = df.sort_values("CCI_score")
    labels = data["axis"] + "\n" + data["sender_receiver"].str.replace(" -> ", "→")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.barh(labels, data["CCI_score"], color="#1f9d8a")
    ax.set_xlabel("TopoLink-CCI score")
    ax.set_title(title)
    ax.set_xlim(0, max(0.85, data["CCI_score"].max() * 1.08))
    for y, row in enumerate(data.itertuples(index=False)):
        ax.text(row.CCI_score + 0.01, y, f"{row.CCI_score:.3f}", va="center", fontsize=6)
    save(fig, name, data)


def plot_top_hit_table(completion: pd.DataFrame) -> None:
    data = completion[completion["top_axis"].notna()].copy()
    data = data[["dataset", "method", "status", "top_axis", "top_sender_receiver", "top_score"]].sort_values(["dataset", "method"])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axis("off")
    display = data.copy()
    display["top_score"] = pd.to_numeric(display["top_score"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_facecolor("#e8f1f0")
            cell.set_text_props(weight="bold")
    ax.set_title("Top hit per completed full method", pad=10)
    save(fig, "fig5_top_hit_table", data)


def plot_breast_vs_cervical_topolink(breast: pd.DataFrame, cervical: pd.DataFrame) -> None:
    data = pd.concat([breast.head(10), cervical.head(10)], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True)
    for ax, dataset, title in zip(axes, ["breast", "cervical"], ["Breast WTA", "Cervical WTA"]):
        sub = data[data["dataset"] == dataset].sort_values("CCI_score")
        labels = sub["axis"] + "\n" + sub["sender_receiver"].str.replace(" -> ", "→")
        ax.barh(labels, sub["CCI_score"], color="#1f9d8a" if dataset == "breast" else "#d8843a")
        ax.set_title(title)
        ax.set_xlabel("CCI score")
        ax.set_xlim(0, 0.86)
    fig.suptitle("TopoLink-CCI top axes across Xenium WTA datasets", y=1.02)
    save(fig, "fig4_breast_cervical_topolink_comparison", data)


def main() -> None:
    setup_style()
    completion_path = ROOT / "results" / "method_completion_matrix.tsv"
    completion = pd.read_csv(completion_path, sep="\t")
    completion["n_rows"] = pd.to_numeric(completion.get("n_rows"), errors="coerce")
    completion = fill_completion_rows(completion)

    plot_completion(completion)
    plot_rows(completion)

    breast_scores = ROOT / "pdc_collected" / "pdc_20260426_1327" / "runs" / "full_common" / "pyxenium" / "pyxenium_scores.tsv"
    cervical_scores = SRC_DIR / "cervical_topolink_top30.tsv"
    breast_top = topolink_top_axes(breast_scores, 15, "breast")
    cervical_top = topolink_top_axes(cervical_scores, 15, "cervical")
    plot_top_axes(breast_top, "fig3_breast_topolink_top_axes", "Breast TopoLink-CCI top axes")
    plot_top_axes(cervical_top, "fig3b_cervical_topolink_top_axes", "Cervical TopoLink-CCI top axes")
    plot_breast_vs_cervical_topolink(breast_top, cervical_top)
    plot_top_hit_table(completion)

    manifest = pd.DataFrame(
        {
            "figure": sorted(p.name for p in FIG_DIR.glob("*.png")),
            "path": [str(p) for p in sorted(FIG_DIR.glob("*.png"))],
        }
    )
    manifest.to_csv(OUT / "preview_figure_manifest.tsv", sep="\t", index=False)
    print(f"Wrote preview figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
