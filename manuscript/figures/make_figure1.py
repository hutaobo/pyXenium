from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "manuscript" / "evidence"
OUTDIR = ROOT / "manuscript" / "figures"
PNG_PATH = OUTDIR / "figure1_pyxenium_validation.png"
PDF_PATH = OUTDIR / "figure1_pyxenium_validation.pdf"


COLORS = {
    "navy": "#17324D",
    "teal": "#2A7F86",
    "blue": "#5D89B3",
    "green": "#5F9858",
    "gold": "#D9A648",
    "coral": "#C96B53",
    "slate": "#5A6673",
    "ink": "#1D242C",
    "muted": "#66717C",
    "border": "#CFD7DF",
    "grid": "#E8EDF2",
    "bg": "#F7F9FB",
    "card": "#FFFFFF",
    "light_teal": "#E9F5F5",
    "light_blue": "#ECF2F8",
    "light_green": "#EEF7ED",
    "light_gold": "#FCF5E6",
    "light_coral": "#FAEEEA",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(ax):
    ax.set_facecolor(COLORS["card"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["ink"], labelsize=9)


def panel_label(ax, label: str, title: str):
    ax.text(-0.06, 1.01, label, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=COLORS["navy"], ha="left", va="bottom")
    ax.text(0.00, 1.01, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=COLORS["ink"], ha="left", va="bottom")


def rounded_note(ax, xy, width, height, text, *, fc, ec, fontsize=9, weight="normal", color=None):
    color = color or COLORS["ink"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.25,
    )


def draw_panel_a(ax, smoke_auto: dict, smoke_h5: dict):
    ax.set_axis_off()
    panel_label(ax, "A", "Smoke-test summary")

    summary_a = smoke_auto["summary"]
    summary_h = smoke_h5["summary"]
    dataset_title = summary_a["dataset_title"]

    ax.text(0.00, 0.93, dataset_title, transform=ax.transAxes, fontsize=8.8, color=COLORS["muted"], ha="left", va="top")
    rounded_note(ax, (0.00, 0.80), 0.22, 0.09, "public 10x dataset", fc=COLORS["light_blue"], ec=COLORS["blue"], fontsize=9.3, weight="bold")
    rounded_note(ax, (0.25, 0.80), 0.22, 0.09, "auto and h5 match", fc=COLORS["light_green"], ec=COLORS["green"], fontsize=9.3, weight="bold")
    rounded_note(ax, (0.50, 0.80), 0.23, 0.09, "issues = []", fc=COLORS["light_coral"], ec=COLORS["coral"], fontsize=9.3, weight="bold")
    rounded_note(ax, (0.76, 0.80), 0.20, 0.09, "spatial + cluster", fc=COLORS["light_teal"], ec=COLORS["teal"], fontsize=9.3, weight="bold")

    rows = [
        ["Cells", f"{summary_a['n_cells']:,}", f"{summary_h['n_cells']:,}"],
        ["RNA features", str(summary_a["n_rna_features"]), str(summary_h["n_rna_features"])],
        ["Protein markers", str(summary_a["n_protein_markers"]), str(summary_h["n_protein_markers"])],
        ["RNA matrix nnz", f"{summary_a['x_nnz']:,}", f"{summary_h['x_nnz']:,}"],
        ["metrics_summary.csv", f"{summary_a['metrics_summary_num_cells_detected']:,}", f"{summary_h['metrics_summary_num_cells_detected']:,}"],
        ["Spatial", str(summary_a["has_spatial"]).lower(), str(summary_h["has_spatial"]).lower()],
        ["Cluster", str(summary_a["has_cluster"]).lower(), str(summary_h["has_cluster"]).lower()],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "auto", "h5"],
        cellLoc="center",
        bbox=[0.00, 0.08, 0.64, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["border"])
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(COLORS["light_blue"])
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        else:
            cell.set_facecolor(COLORS["card"])
            if c == 0:
                cell.set_text_props(weight="bold", color=COLORS["ink"], ha="left")
            else:
                cell.set_text_props(color=COLORS["ink"])

    rounded_note(
        ax,
        (0.71, 0.46),
        0.25,
        0.22,
        f"{summary_a['n_cells']:,}\nvalidated cells",
        fc=COLORS["card"],
        ec=COLORS["teal"],
        fontsize=15,
        weight="bold",
        color=COLORS["ink"],
    )
    rounded_note(
        ax,
        (0.71, 0.18),
        0.25,
        0.18,
        "Smoke-test outputs:\nsummary.json, report.md,\nCSV summaries",
        fc=COLORS["light_gold"],
        ec=COLORS["gold"],
        fontsize=9.2,
    )


def draw_panel_b(ax, top_rna: pd.DataFrame):
    style_axis(ax)
    panel_label(ax, "B", "Top RNA features from the validated run")

    plot_df = top_rna.copy()
    plot_df["total_counts_m"] = plot_df["total_counts"] / 1_000_000
    y = range(len(plot_df))
    bars = ax.barh(y, plot_df["total_counts_m"], color=COLORS["teal"], height=0.58)
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["feature"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Total RNA counts (millions)", fontsize=9.5, color=COLORS["ink"])
    ax.set_xlim(0, plot_df["total_counts_m"].max() * 1.18)
    ax.text(0.00, 0.98, "Smoke test: prefer='auto'", transform=ax.transAxes, fontsize=8.7, color=COLORS["muted"], ha="left", va="top")

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        x = bar.get_width()
        y_mid = bar.get_y() + bar.get_height() / 2
        ax.text(
            x + plot_df["total_counts_m"].max() * 0.02,
            y_mid,
            f"{row['total_counts_m']:.2f}M\n{int(row['detected_cells']):,} cells",
            ha="left",
            va="center",
            fontsize=8.6,
            color=COLORS["ink"],
        )


def draw_panel_c(ax, top_protein: pd.DataFrame):
    style_axis(ax)
    panel_label(ax, "C", "Top protein markers from the validated run")

    plot_df = top_protein.copy()
    y = range(len(plot_df))
    bars = ax.barh(y, plot_df["mean_signal"], color=COLORS["gold"], height=0.58)
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["marker"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean protein signal", fontsize=9.5, color=COLORS["ink"])
    ax.set_xlim(0, plot_df["mean_signal"].max() * 1.24)
    ax.text(0.00, 0.98, "Smoke test: prefer='auto'", transform=ax.transAxes, fontsize=8.7, color=COLORS["muted"], ha="left", va="top")

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        x = bar.get_width()
        y_mid = bar.get_y() + bar.get_height() / 2
        ax.text(
            x + plot_df["mean_signal"].max() * 0.03,
            y_mid,
            f"{row['mean_signal']:.1f}\n{int(row['positive_cells']):,} positive",
            ha="left",
            va="center",
            fontsize=8.6,
            color=COLORS["ink"],
        )


def draw_panel_d(ax, clusters: pd.DataFrame, partial_only: dict, pytest_text: str):
    ax.set_axis_off()
    panel_label(ax, "D", "Largest clusters and partial-loader output")

    left = ax.inset_axes([0.00, 0.08, 0.47, 0.80])
    style_axis(left)
    left.set_title("Largest graph clusters", fontsize=10, loc="left", color=COLORS["ink"], pad=8)
    left.bar(range(len(clusters)), clusters["n_cells"] / 1000, color=COLORS["blue"], width=0.62)
    left.set_xticks(range(len(clusters)))
    left.set_xticklabels([str(c) for c in clusters["cluster"]], fontsize=9)
    left.set_ylabel("Cells (thousands)", fontsize=9.3, color=COLORS["ink"])
    left.set_xlabel("Cluster label", fontsize=9.3, color=COLORS["ink"])
    left.grid(axis="y", color=COLORS["grid"], linewidth=0.9)
    left.grid(axis="x", visible=False)
    for idx, value in enumerate(clusters["n_cells"]):
        left.text(idx, value / 1000 + 2.2, f"{value/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, color=COLORS["ink"])

    right = ax.inset_axes([0.55, 0.08, 0.43, 0.80])
    style_axis(right)
    right.set_title("Real MEX-only partial load", fontsize=10, loc="left", color=COLORS["ink"], pad=8)
    feature_counts = partial_only["feature_type_counts"]
    labels = [
        "Gene expression",
        "Protein expression",
        "Neg. control codeword",
        "Unassigned codeword",
        "Neg. control probe",
        "Genomic control",
    ]
    values = [
        feature_counts["Gene Expression"],
        feature_counts["Protein Expression"],
        feature_counts["Negative Control Codeword"],
        feature_counts["Unassigned Codeword"],
        feature_counts["Negative Control Probe"],
        feature_counts["Genomic Control"],
    ]
    colors = [COLORS["teal"], COLORS["gold"], "#97A5B2", "#AEB9C3", "#C0C9D0", "#D5DCE1"]
    y = range(len(labels))
    bars = right.barh(list(y), values, color=colors, height=0.56)
    right.set_yticks(list(y))
    right.set_yticklabels(labels, fontsize=8.7)
    right.invert_yaxis()
    right.set_xlabel("Recovered features", fontsize=9.2, color=COLORS["ink"])
    right.set_xlim(0, max(values) * 1.22)
    for bar, value in zip(bars, values):
        x = bar.get_width()
        y_mid = bar.get_y() + bar.get_height() / 2
        right.text(x + max(values) * 0.025, y_mid, f"{value}", ha="left", va="center", fontsize=8.3, color=COLORS["ink"])

    right.text(
        0.00,
        -0.22,
        f"{partial_only['shape'][0]:,} x {partial_only['shape'][1]} | counts-only partial load",
        transform=right.transAxes,
        fontsize=8.0,
        color=COLORS["muted"],
        ha="left",
    )


def make_figure():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    smoke_auto = load_json(EVIDENCE / "smoke_auto" / "summary.json")
    smoke_h5 = load_json(EVIDENCE / "smoke_h5" / "summary.json")
    top_rna = pd.read_csv(EVIDENCE / "smoke_auto" / "top_rna_features.csv")
    top_protein = pd.read_csv(EVIDENCE / "smoke_auto" / "top_protein_markers.csv")
    clusters = pd.read_csv(EVIDENCE / "smoke_auto" / "largest_clusters.csv")
    partial_only = load_json(EVIDENCE / "partial_loader_mex_only.json")
    pytest_text = (EVIDENCE / "pytest_q.txt").read_text(encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": COLORS["card"],
            "figure.facecolor": COLORS["bg"],
            "savefig.facecolor": COLORS["bg"],
        }
    )

    fig = plt.figure(figsize=(13.4, 10.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.07,
        hspace=0.28,
        wspace=0.20,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, smoke_auto, smoke_h5)
    draw_panel_b(ax_b, top_rna)
    draw_panel_c(ax_c, top_protein)
    draw_panel_d(ax_d, clusters, partial_only, pytest_text)

    fig.suptitle(
        "pyXenium validation on a public 10x Xenium RNA+Protein dataset",
        fontsize=14.8,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.955,
    )

    fig.savefig(PNG_PATH, dpi=400)
    fig.savefig(PDF_PATH)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    make_figure()
