"""Build a Nature-style short communication package for TopoLink-CCI.

This script follows the local ``nature-figure`` plotting contract:

- Python/Matplotlib only.
- Editable SVG text and Type 42 PDF fonts.
- One clear claim per panel.
- Source-data TSVs for every panel.
- Separate two-main-figure and one-main-figure-fallback outputs.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D

try:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Inches, Pt
except Exception:  # pragma: no cover
    Document = None


REPO = Path(__file__).resolve().parents[4]
CCI_ROOT = REPO / "benchmarking" / "cci_2026_atera"
PACKAGE_ROOT = CCI_ROOT / "topolink_cci_short_communication" / "nature_short_communication_20260516"
FIG_DIR = PACKAGE_ROOT / "figures"
SOURCE_DIR = PACKAGE_ROOT / "source_data"
META_DIR = PACKAGE_ROOT / "metadata"
MANUSCRIPT_DIR = PACKAGE_ROOT / "manuscript"

METHOD_MATRIX = CCI_ROOT / "results" / "method_completion_matrix.tsv"
SYNTHETIC_TRUTH = (
    CCI_ROOT
    / "results"
    / "publication_benchmark_24h_20260511"
    / "pdc_collected"
    / "publication_benchmark_24h_20260511"
    / "synthetic_truth"
    / "tables"
    / "synthetic_truth_metrics.tsv"
)
EVIDENCE = CCI_ROOT / "pdc_validation_v2_collected" / "topolink_cci_validation_v2" / "tables" / "topolink_cci_validation_v2_evidence.tsv"
BREAST_TOP = CCI_ROOT / "results" / "preview_completed_20260511" / "source_data" / "fig3_breast_topolink_top_axes.tsv"
CROSS_DATASET = CCI_ROOT / "results" / "preview_completed_20260511" / "source_data" / "fig4_breast_cervical_topolink_comparison.tsv"
CANONICAL_RANK = CCI_ROOT / "results" / "cross_method_comparison_20260511" / "source_data" / "fig6_canonical_pair_rank_heatmap.tsv"
COMPONENTS = CCI_ROOT / "vwf_selp_deep_dive" / "tables" / "component_decomposition.tsv"
HOTSPOT_SUMMARY = CCI_ROOT / "vwf_selp_deep_dive" / "tables" / "hotspot_summary.tsv"
HOTSPOT_IMAGE = CCI_ROOT / "vwf_selp_deep_dive" / "figures" / "vwf_selp_hotspot_spatial_overlay.png"


COLORS = {
    "text": "#111827",
    "grid": "#E5E7EB",
    "muted": "#6B7280",
    "teal": "#0F766E",
    "blue": "#2563EB",
    "orange": "#D97706",
    "red": "#B91C1C",
    "green": "#15803D",
    "purple": "#6D28D9",
    "brown": "#8A5A2B",
    "gray": "#6B7280",
    "full": "#86EFAC",
    "bounded": "#FBBF24",
    "failure": "#FCA5A5",
}

THEME = {
    "VWF-SELP": "vascular",
    "VWF-LRP1": "vascular",
    "MMRN2-CD93": "vascular",
    "CD48-CD2": "immune",
    "DLL4-NOTCH3": "notch",
    "CXCL12-CXCR4": "stromal",
    "JAG1-NOTCH2": "tumor",
}

THEME_COLORS = {
    "vascular": COLORS["teal"],
    "stromal": COLORS["brown"],
    "immune": COLORS["green"],
    "notch": COLORS["purple"],
    "tumor": COLORS["gray"],
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.2,
            "axes.titlesize": 6.4,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 600,
        }
    )


def ensure_dirs() -> None:
    for path in [FIG_DIR, SOURCE_DIR, META_DIR, MANUSCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def load_data() -> dict[str, pd.DataFrame]:
    evidence = read_table(EVIDENCE)
    evidence["axis"] = evidence["ligand"] + "-" + evidence["receptor"]
    evidence["theme"] = evidence["axis"].map(THEME).fillna("tumor")
    data = {
        "method": read_table(METHOD_MATRIX),
        "synthetic": read_table(SYNTHETIC_TRUTH),
        "evidence": evidence,
        "breast_top": read_table(BREAST_TOP),
        "cross_dataset": read_table(CROSS_DATASET),
        "canonical_rank": read_table(CANONICAL_RANK),
        "components": read_table(COMPONENTS),
        "hotspot_summary": read_table(HOTSPOT_SUMMARY),
    }
    return data


def save_source(panel_id: str, df: pd.DataFrame) -> Path:
    path = SOURCE_DIR / f"{panel_id}.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for ext in ["pdf", "svg", "png", "tiff"]:
        path = FIG_DIR / f"{stem}.{ext}"
        if ext == "tiff":
            fig.savefig(path, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=600, bbox_inches="tight")
        outputs[ext] = str(path)
    return outputs


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.07,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
        va="top",
    )


def rounded_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    w, h = wh
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc="white",
        ec=color,
        lw=0.7,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=4.8, color=COLORS["text"])


def draw_problem_schematic(ax: plt.Axes) -> pd.DataFrame:
    panel_label(ax, "a")
    ax.set_title("False-positive risks", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        ("co-expression\nfalse positives", 0.06, 0.66, COLORS["red"]),
        ("cell abundance\nbias", 0.58, 0.66, COLORS["orange"]),
        ("proximity without\nmolecular support", 0.06, 0.28, COLORS["blue"]),
        ("CCI resource\nambiguity", 0.58, 0.28, COLORS["purple"]),
    ]
    for text, x, y, color in boxes:
        rounded_box(ax, (x, y), (0.36, 0.16), text, color)
    ax.text(0.50, 0.50, "False-positive\nrisk", ha="center", va="center", fontsize=7, fontweight="bold", color=COLORS["text"])
    for _, x, y, color in boxes:
        ax.annotate("", xy=(0.50, 0.50), xytext=(x + 0.18, y + 0.08), arrowprops=dict(arrowstyle="->", lw=0.7, color=color))
    return pd.DataFrame(boxes, columns=["risk", "x", "y", "color"])


def draw_score_architecture(ax: plt.Axes) -> pd.DataFrame:
    panel_label(ax, "b")
    ax.set_title("Score architecture", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    components = [
        ("sender\nanchor", 0.08, 0.68, COLORS["teal"]),
        ("receiver\nanchor", 0.38, 0.68, COLORS["purple"]),
        ("structure\nbridge", 0.68, 0.68, COLORS["blue"]),
        ("sender\nexpression", 0.08, 0.36, COLORS["green"]),
        ("receiver\nexpression", 0.38, 0.36, COLORS["green"]),
        ("local\ncontact", 0.68, 0.36, COLORS["orange"]),
    ]
    for text, x, y, color in components:
        rounded_box(ax, (x, y), (0.22, 0.14), text, color)
        ax.annotate("", xy=(0.50, 0.20), xytext=(x + 0.11, y), arrowprops=dict(arrowstyle="->", lw=0.55, color=color))
    ax.text(
        0.50,
        0.13,
        "prior-weighted geometric mean\n= discovery score, not proof",
        ha="center",
        va="center",
        fontsize=6,
        bbox=dict(fc="#F9FAFB", ec="#D1D5DB", boxstyle="round,pad=0.25"),
    )
    return pd.DataFrame(components, columns=["component", "x", "y", "color"])


def draw_workflow(ax: plt.Axes) -> pd.DataFrame:
    panel_label(ax, "c")
    ax.set_title("Workflow", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Xenium\nWTA", 0.04, 0.42),
        ("topology\nmap", 0.24, 0.42),
        ("CCI\nranking", 0.44, 0.42),
        ("controls", 0.64, 0.42),
        ("evidence\nclass", 0.82, 0.42),
    ]
    for i, (text, x, y) in enumerate(steps):
        rounded_box(ax, (x, y), (0.15, 0.18), text, COLORS["teal"] if i < 3 else COLORS["orange"])
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][1] - 0.01, y + 0.09), xytext=(x + 0.16, y + 0.09), arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["text"]))
    ax.text(0.50, 0.19, "Breast WTA: 170,057 cells; 3,299 common CCI pairs; 1,319,600 hypotheses", ha="center", fontsize=5.8)
    return pd.DataFrame(steps, columns=["step", "x", "y"])


def draw_synthetic(ax: plt.Axes, synthetic: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "d")
    plot = synthetic.copy()
    label_map = {
        "TopoLink-CCI": "TopoLink\nCCI",
        "expression_only": "expr.",
        "contact_only": "contact",
        "topology_anchor_only": "anchor",
    }
    plot["display"] = plot["method"].map(label_map).fillna(plot["method"])
    x = np.arange(len(plot))
    width = 0.34
    ax.bar(x - width / 2, plot["auroc"], width=width, color=COLORS["blue"], label="AUROC")
    ax.bar(x + width / 2, plot["auprc"], width=width, color=COLORS["teal"], label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["display"])
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("metric")
    ax.set_title("Synthetic Truth", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.legend(loc="lower left")
    top = plot.loc[plot["method"].eq("TopoLink-CCI")].iloc[0]
    anchor = plot.loc[plot["method"].eq("topology_anchor_only")].iloc[0]
    ax.text(
        0.03,
        0.95,
        f"TopoLink-CCI\nAUROC {top.auroc:.4f}\nAUPRC {top.auprc:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        bbox=dict(fc="white", ec="#D1D5DB", boxstyle="round,pad=0.25"),
    )
    ax.text(0.98, 0.52, f"anchor-only\nAUPRC {anchor.auprc:.4f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=COLORS["orange"])
    return plot[["method", "auroc", "auprc", "precision_at_5", "recall_at_5", "f1_at_5"]]


def draw_evidence_matrix(ax: plt.Axes, evidence: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "e")
    layers = [
        ("expr", "expression_specificity_support"),
        ("label", "cell_label_permutation_support"),
        ("spatial", "spatial_null_support"),
        ("matched", "matched_gene_control_support"),
        ("target", "downstream_target_support"),
        ("signal", "functional_received_signal_support"),
        ("cross", "cross_method_support"),
        ("boot", "bootstrap_stability_support"),
    ]
    df = evidence.sort_values("pyxenium_rank").head(7).copy()
    matrix = df[[col for _, col in layers]].astype(bool).to_numpy()
    ax.set_title("Validation controls", loc="left", fontweight="bold")
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([name for name, _ in layers], rotation=45, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["axis"].tolist())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            supported = matrix[i, j]
            ax.scatter(
                j,
                i,
                s=52,
                marker="o" if supported else "X",
                color=COLORS["green"] if supported else COLORS["orange"],
                edgecolor="white",
                linewidth=0.45,
            )
    ax.grid(color=COLORS["grid"], lw=0.45)
    ax.tick_params(length=0)
    ax.text(0.99, 0.02, "7/7 strong; 0 artifact risk", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, fontweight="bold")
    keep = ["axis", "pyxenium_rank", "CCI_score", "support_count", "evidence_class"] + [col for _, col in layers]
    return df[keep]


def draw_breast_top_axes(ax: plt.Axes, breast_top: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "a")
    plot = breast_top.copy().sort_values("CCI_score").tail(8)
    plot["theme"] = plot["axis"].map(THEME).fillna("tumor")
    colors = [THEME_COLORS[t] for t in plot["theme"]]
    ax.barh(plot["axis"], plot["CCI_score"], color=colors)
    ax.set_xlim(0.63, 0.82)
    ax.set_xlabel("CCI score")
    ax.set_title("Breast top axes", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    return plot[["dataset", "axis", "sender_receiver", "CCI_score", "local_contact", "cross_edge_count", "theme"]]


def draw_vwf_decomposition(ax: plt.Axes, components: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "b")
    order = ["sender_anchor", "receiver_anchor", "structure_bridge", "sender_expr", "receiver_expr", "local_contact"]
    df = components.set_index("component").loc[order].reset_index()
    colors = [COLORS["teal"], COLORS["purple"], COLORS["blue"], COLORS["green"], COLORS["green"], COLORS["orange"]]
    ax.barh(df["component"].str.replace("_", "\n"), df["value"], color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("component value")
    ax.set_title("VWF-SELP score components", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.text(
        0.05,
        0.10,
        "CCI_score 0.791\nlocal_contact 0.291\n12,779 endothelial-endothelial edges",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        bbox=dict(fc="white", ec="#D1D5DB", boxstyle="round,pad=0.25"),
    )
    return df


def draw_hotspot(ax: plt.Axes, hotspot_summary: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "c")
    ax.set_title("Spatial hotspots", loc="left", fontweight="bold")
    image = plt.imread(HOTSPOT_IMAGE)
    ax.imshow(image, rasterized=True)
    ax.axis("off")
    n_hot = int(hotspot_summary.loc[0, "n_hotspot_endothelial_cells"])
    ax.text(
        0.03,
        0.96,
        f"{n_hot} hotspot endothelial cells\nRNA-level spatial evidence",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        bbox=dict(fc="white", ec="#D1D5DB", boxstyle="round,pad=0.25"),
    )
    return hotspot_summary.assign(image_path=str(HOTSPOT_IMAGE))


def draw_method_status(ax: plt.Axes, method: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "d")
    breast = method.loc[method["dataset"].eq("atera_breast_wta")].copy()
    order = ["full_result", "bounded_subset_result", "reproducible_failure_card"]
    counts = breast["status"].value_counts().reindex(order).fillna(0).astype(int)
    labels = ["full", "bounded", "failure\ncard"]
    colors = [COLORS["full"], COLORS["bounded"], COLORS["failure"]]
    ax.bar(labels, counts.values, color=colors, edgecolor="#374151", linewidth=0.5)
    ax.set_ylim(0, max(counts.values) + 2)
    ax.set_ylabel("methods")
    ax.set_title("Benchmark status", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.25, str(value), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(0.98, 0.96, "0 deferred", transform=ax.transAxes, ha="right", va="top", fontsize=6, fontweight="bold")
    return counts.rename_axis("status").reset_index(name="n_methods")


def draw_canonical_rank(ax: plt.Axes, canonical: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "e")
    selected_methods = ["TopoLink-CCI", "CellPhoneDB", "LARIS", "LIANA+", "SpatialDM", "stLearn", "Squidpy"]
    canonical_ids = ["VWF|SELP", "VWF|LRP1", "MMRN2|CD93", "DLL4|NOTCH3", "CXCL12|CXCR4"]
    df = canonical.loc[canonical["method"].isin(selected_methods) & canonical["canonical_id"].isin(canonical_ids)].copy()
    pivot = df.pivot_table(index="method", columns="canonical_id", values="rank_capped", aggfunc="min").reindex(selected_methods)
    pivot = pivot.reindex(columns=canonical_ids)
    ax.set_xlim(-0.5, len(canonical_ids) - 0.5)
    ax.set_ylim(len(selected_methods) - 0.5, -0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("|", "-").replace("-", "\n") for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Canonical ranks", loc="left", fontweight="bold")
    ax.grid(True, color=COLORS["grid"], lw=0.45)
    for i, method in enumerate(pivot.index):
        for j, axis_id in enumerate(pivot.columns):
            value = pivot.loc[method, axis_id]
            if pd.isna(value):
                ax.plot(j, i, marker="x", color="#BDBDBD", markersize=4.0, mew=0.8)
                continue
            if value <= 50:
                color, size = COLORS["teal"], 58
            elif value <= 500:
                color, size = "#4D96A8", 42
            elif value <= 5000:
                color, size = COLORS["orange"], 26
            else:
                color, size = "#C9C9C9", 14
            ax.scatter(j, i, s=size, color=color, edgecolor="white", linewidth=0.35, zorder=3)
            ax.text(j, i + 0.29, f"{int(value)}", ha="center", va="center", fontsize=3.8, color=COLORS["text"])
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["teal"], markeredgecolor="white", markersize=5, label="top 50"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4D96A8", markeredgecolor="white", markersize=4, label="top 500"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["orange"], markeredgecolor="white", markersize=3, label="top 5000"),
        Line2D([0], [0], marker="x", color="#BDBDBD", markersize=4, label="not detected"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    return df


def draw_cross_dataset(ax: plt.Axes, cross: pd.DataFrame) -> pd.DataFrame:
    panel_label(ax, "f")
    top = cross.groupby("dataset", group_keys=False).head(3).copy()
    top = top.sort_values(["dataset", "CCI_score"], ascending=[True, True])
    top["display"] = top["dataset"].str.title() + ": " + top["axis"]
    colors = [COLORS["teal"] if d == "breast" else COLORS["purple"] for d in top["dataset"]]
    ax.barh(top["display"], top["CCI_score"], color=colors)
    ax.set_xlim(0.60, max(0.84, float(top["CCI_score"].max()) + 0.02))
    ax.set_xlabel("CCI score")
    ax.set_title("Breast vs cervical", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.text(
        0.02,
        0.03,
        "Breast top: VWF-SELP endothelial activation\nCervical top: DSC2-DSG3 differentiating tumor adhesion\nCervical full: 2,404,971 rows",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        bbox=dict(fc="white", ec="#D1D5DB", boxstyle="round,pad=0.25"),
    )
    return top[["dataset", "axis", "sender_receiver", "CCI_score", "local_contact", "cross_edge_count"]]


def make_figure_1(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    fig = plt.figure(figsize=(7.15, 5.35), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.0, 1.2])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.91, wspace=0.50, hspace=0.42)
    sources = {
        "fig1a_problem_schematic": save_source("fig1a_problem_schematic", draw_problem_schematic(fig.add_subplot(gs[0, 0]))),
        "fig1b_score_architecture": save_source("fig1b_score_architecture", draw_score_architecture(fig.add_subplot(gs[0, 1]))),
        "fig1c_workflow": save_source("fig1c_workflow", draw_workflow(fig.add_subplot(gs[0, 2]))),
        "fig1d_synthetic_truth": save_source("fig1d_synthetic_truth", draw_synthetic(fig.add_subplot(gs[1, 0]), data["synthetic"])),
        "fig1e_evidence_matrix": save_source("fig1e_evidence_matrix", draw_evidence_matrix(fig.add_subplot(gs[1, 1:]), data["evidence"])),
    }
    fig.text(0.07, 0.985, "Figure 1 | TopoLink-CCI method and validation logic", fontsize=7.2, fontweight="bold", va="top")
    outputs = save_figure(fig, "figure_1_topolink_cci_method_validation")
    plt.close(fig)
    write_manifest("figure_1", outputs, sources, "TopoLink-CCI needs topology-expression-contact concordance plus independent controls.")
    return outputs


def make_figure_2(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    fig = plt.figure(figsize=(7.15, 8.45), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.12, 1.0])
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.075, top=0.94, wspace=0.62, hspace=0.78)
    sources = {
        "fig2a_breast_top_axes": save_source("fig2a_breast_top_axes", draw_breast_top_axes(fig.add_subplot(gs[0, 0]), data["breast_top"])),
        "fig2b_vwf_selp_components": save_source("fig2b_vwf_selp_components", draw_vwf_decomposition(fig.add_subplot(gs[0, 1]), data["components"])),
        "fig2c_vwf_selp_hotspots": save_source("fig2c_vwf_selp_hotspots", draw_hotspot(fig.add_subplot(gs[1, 0]), data["hotspot_summary"])),
        "fig2d_method_status": save_source("fig2d_method_status", draw_method_status(fig.add_subplot(gs[1, 1]), data["method"])),
        "fig2e_canonical_rank": save_source("fig2e_canonical_rank", draw_canonical_rank(fig.add_subplot(gs[2, 0]), data["canonical_rank"])),
        "fig2f_cross_dataset": save_source("fig2f_cross_dataset", draw_cross_dataset(fig.add_subplot(gs[2, 1]), data["cross_dataset"])),
    }
    fig.text(0.10, 0.987, "Figure 2 | Whole-dataset discovery and biological interpretation", fontsize=7.2, fontweight="bold", va="top")
    outputs = save_figure(fig, "figure_2_topolink_cci_discovery_benchmark")
    plt.close(fig)
    write_manifest("figure_2", outputs, sources, "Whole-dataset TopoLink-CCI outputs recover interpretable and tissue-specific CCI axes.")
    return outputs


def make_fallback(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    fig = plt.figure(figsize=(7.15, 7.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.2, 1.05], width_ratios=[1.0, 1.15])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.07, top=0.94, wspace=0.52, hspace=0.72)
    sources = {
        "fallback_a_score": save_source("fallback_a_score_architecture", draw_score_architecture(fig.add_subplot(gs[0, 0]))),
        "fallback_b_synthetic": save_source("fallback_b_synthetic_truth", draw_synthetic(fig.add_subplot(gs[0, 1]), data["synthetic"])),
        "fallback_c_evidence": save_source("fallback_c_evidence_matrix", draw_evidence_matrix(fig.add_subplot(gs[1, :]), data["evidence"])),
        "fallback_d_status": save_source("fallback_d_method_status", draw_method_status(fig.add_subplot(gs[2, 0]), data["method"])),
        "fallback_e_cross_dataset": save_source("fallback_e_cross_dataset", draw_cross_dataset(fig.add_subplot(gs[2, 1]), data["cross_dataset"])),
    }
    fig.text(0.08, 0.987, "TopoLink-CCI prioritizes topology-supported spatial CCI hypotheses", fontsize=7.2, fontweight="bold", va="top")
    outputs = save_figure(fig, "figure_1_fallback_single_main")
    plt.close(fig)
    write_manifest("figure_1_fallback", outputs, sources, "Single-display-item fallback combining method, validation and discovery evidence.")
    return outputs


def write_manifest(figure_id: str, outputs: dict[str, str], sources: dict[str, Path], claim: str) -> None:
    manifest = {
        "figure_id": figure_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim": claim,
        "outputs": outputs,
        "source_data": {key: str(value) for key, value in sources.items()},
        "figure_contract": {
            "backend": "Python/Matplotlib",
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
            "source_data_required": True,
            "journal_target": "Nature Methods Brief Communication",
        },
    }
    (META_DIR / f"{figure_id}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_manuscripts() -> dict[str, Path]:
    abstract = (
        "Spatial cell-cell interaction inference is confounded by co-expression, cell abundance and incomplete molecular resources. "
        "We introduce TopoLink-CCI, a topology-guided framework that integrates tissue topology, expression specificity and local contact, "
        "then challenges candidates with orthogonal controls. In Xenium WTA breast and cervical cancers, TopoLink-CCI scales to whole datasets "
        "and prioritizes interpretable vascular, stromal, immune and tumor interaction axes."
    )
    main_text = f"""# Topology-guided prioritization of spatial cell-cell interaction axes

**Target journal:** Nature Methods Brief Communication
**Fallback journal:** Nature Biotechnology Brief Communication
**Method:** TopoLink-CCI

## Abstract

{abstract}

## Main text

Spatial transcriptomics has made it possible to infer cell-cell interaction (CCI) axes in intact tissues, but high-resolution whole-transcriptome imaging also increases false-positive risk. Ligand and receptor genes can appear biologically plausible because both are strongly expressed, because the corresponding cell types are abundant, or because a curated molecular resource includes extracellular-matrix, adhesion, scavenger-receptor and shared activation-state relationships that are not classical secreted signaling events. Spatial proximity improves interpretation but does not by itself prove molecular exchange, receptor activation or causal signaling. The central problem is therefore not only to rank candidate molecular pairs, but to prioritize CCI hypotheses that are compatible with tissue topology, local organization and independent controls.

We developed TopoLink-CCI as a topology-guided CCI prioritization framework in pyXenium. In CCI-resource mode, each ligand, receptor, sender and receiver combination is scored by six retained components: sender topology anchor, receiver topology anchor, sender-receiver structure bridge, sender expression support, receiver expression support and local contact support. The discovery score is a prior-weighted geometric mean of these components, so a high score requires concordance across topology, expression and spatial contact rather than any single high-expression feature. Because every component is exported, users can diagnose whether an axis is topology-driven, expression-driven, contact-sensitive or potentially dominated by a resource prior. This design explicitly treats the score as a discovery statistic, not as proof of protein-level communication.

We first evaluated the scoring logic in a topology-preserving Synthetic Truth benchmark. The full TopoLink-CCI model achieved AUROC 0.9919 and AUPRC 0.8333, whereas the topology-anchor-only model retained high AUROC (0.9839) but dropped to AUPRC 0.5833. We then applied TopoLink-CCI to Atera Xenium WTA breast cancer, generating 1,319,600 common-resource CCI hypotheses from 170,057 cells. The expanded breast benchmark includes 18 terminalized methods: nine full results, six bounded subset results and three reproducible failure cards, with no deferred methods. Seven representative axes were evaluated with orthogonal controls inspired by established CCI methods, including cell-label permutation, spatial nulls, matched-expression gene controls, downstream target support, received-signal association, cross-method consensus, component ablation and bootstrap stability.

TopoLink-CCI prioritized biologically interpretable axes spanning vascular activation, stromal matrix biology, immune adhesion, Notch signaling and tumor-intrinsic adhesion. The top breast axis, VWF-SELP from endothelial cells to endothelial cells, is best interpreted as an endothelial activation and vascular adhesion niche consistent with Weibel-Palade body biology, not as direct proof of VWF/P-selectin protein release. Cross-dataset application to Xenium WTA cervical cancer produced 2,404,971 hypotheses and a distinct top tumor-adhesion axis, DSC2-DSG3 in differentiating tumor cells, supporting tissue-context-specific prioritization. TopoLink-CCI is therefore a spatial CCI hypothesis-prioritization framework: it narrows a large molecular search space to axes consistent with tissue topology, expression specificity, local contact and independent computational controls, while leaving causal signaling and protein-level mechanism to orthogonal experimental validation.

## Figure legends

**Figure 1 | TopoLink-CCI method and validation logic.** **a,** Spatial CCI inference is vulnerable to co-expression, cell-abundance, proximity and resource-composition false positives. **b,** TopoLink-CCI scores each candidate using sender and receiver topology anchors, structure bridging, expression support and local contact. **c,** Workflow from Xenium WTA and pyXenium topology maps to ranked CCI axes and validation controls. **d,** Synthetic Truth evaluation shows that the full model achieves AUROC 0.9919 and AUPRC 0.8333, whereas topology-anchor-only scoring loses precision-recall performance. **e,** Orthogonal evidence matrix for seven interpretable breast cancer axes.

**Figure 2 | Whole-dataset discovery and biological interpretation.** **a,** Top interpretable Breast WTA axes ranked by TopoLink-CCI score. **b,** VWF-SELP component decomposition shows joint topology, expression and local-contact support. **c,** VWF-SELP endothelial hotspots are shown as RNA-level spatial evidence. **d,** The expanded breast benchmark terminalizes 18 methods into full, bounded and reproducible-failure tiers. **e,** Canonical recovery is compared by within-method rank rather than raw score. **f,** Breast and cervical WTA datasets yield tissue-context-specific top axes.

## Guardrails

- TopoLink-CCI score is a discovery score, not proof of ligand binding, protein secretion, receptor activation or causal signaling.
- Raw scores are not compared across methods.
- Full whole-dataset and bounded subset methods are explicitly separated.
- F1, AUROC and AUPRC are used only for Synthetic Truth or predefined canonical truth sets.
"""
    methods = """# Online Methods outline

## TopoLink-CCI score

TopoLink-CCI evaluates each candidate molecular interaction axis using six retained components: sender topology anchor, receiver topology anchor, structure bridge, sender expression, receiver expression and local contact. The discovery score is computed as a prior-weighted geometric mean of these components.

## Topology map and expression support

Topology anchors are derived from pyXenium topology outputs linking genes, cell groups and tissue structures. Expression support is evaluated within sender and receiver cell groups and exported as diagnostic fields.

## Local contact graph

Local contact support is computed on a spatial neighbor graph and records active-edge support between sender and receiver populations.

## Synthetic Truth benchmark

Synthetic Truth data preserve tissue topology while implanting known CCI axes. AUROC, AUPRC and top-k precision/recall are used only because positive and negative axes are defined by construction.

## Benchmark status tiers

Methods are classified as full result, bounded subset result or reproducible failure card. Bounded methods are not treated as equivalent to whole-dataset full methods.
"""
    inquiry = """# Presubmission inquiry draft

Dear Editors,

We would like to ask whether you would consider a Brief Communication describing TopoLink-CCI, a topology-guided framework for spatial cell-cell interaction inference in whole-transcriptome imaging data.

TopoLink-CCI addresses a central limitation of current CCI analysis: co-expression, cell abundance and spatial proximity can generate plausible but weakly controlled interaction hypotheses. The method integrates tissue topology, expression specificity and local contact into an interpretable discovery score, then evaluates candidates with orthogonal false-positive controls.

In Atera Xenium WTA breast cancer, TopoLink-CCI generated 1,319,600 CCI hypotheses and prioritized vascular, stromal, immune and Notch axes with strong computational support. In a Synthetic Truth benchmark it achieved AUROC 0.9919 and AUPRC 0.8333, outperforming topology-anchor-only scoring in precision-recall ranking. Cross-dataset application to Xenium WTA cervical cancer produced 2,404,971 hypotheses and a distinct top tumor-adhesion axis, supporting tissue-context-specific prioritization.

We believe this work fits Nature Methods because it presents a concise method and validation framework for spatial omics, with broad relevance to tissue-scale CCI analysis. The proposed submission would include two main figures, online methods, source data and a complete benchmark status table.

Sincerely,
"""
    paths = {
        "main": MANUSCRIPT_DIR / "topolink_cci_short_communication.md",
        "methods": MANUSCRIPT_DIR / "online_methods.md",
        "inquiry": MANUSCRIPT_DIR / "presubmission_inquiry.md",
    }
    paths["main"].write_text(main_text, encoding="utf-8", newline="\n")
    paths["methods"].write_text(methods, encoding="utf-8", newline="\n")
    paths["inquiry"].write_text(inquiry, encoding="utf-8", newline="\n")
    return paths


def write_supplement_manifest(data: dict[str, pd.DataFrame]) -> Path:
    rows = [
        ("Supplementary Table 1", "method_completion_matrix", str(METHOD_MATRIX), "Full/bounded/failure status for all benchmarked methods."),
        ("Supplementary Table 2", "validation_evidence", str(EVIDENCE), "False-positive controls and evidence classes for seven CCI axes."),
        ("Supplementary Table 3", "synthetic_truth_metrics", str(SYNTHETIC_TRUTH), "Synthetic Truth AUROC/AUPRC and top-k metrics."),
        ("Supplementary Table 4", "breast_cervical_top_axes", str(CROSS_DATASET), "Cross-dataset TopoLink-CCI top axes."),
        ("Source image", "vwf_selp_hotspot_overlay", str(HOTSPOT_IMAGE), "Spatial hotspot image used in Figure 2c."),
    ]
    manifest = pd.DataFrame(rows, columns=["item", "name", "path", "description"])
    path = MANUSCRIPT_DIR / "supplementary_data_manifest.tsv"
    manifest.to_csv(path, sep="\t", index=False)
    return path


def write_docx(manuscript_paths: dict[str, Path]) -> Path | None:
    if Document is None:
        return None
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(1.7)
    section.right_margin = Cm(1.7)
    for style_name, size in [("Normal", 9), ("Title", 16), ("Heading 1", 12), ("Heading 2", 10)]:
        style = doc.styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
    main_text = manuscript_paths["main"].read_text(encoding="utf-8")
    for line in main_text.splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            p = doc.add_paragraph(style="Title")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(line[2:]).bold = True
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("- "):
            doc.add_paragraph(line[2:], style="List Bullet")
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.05
            p.add_run(line.replace("**", ""))
    doc.add_page_break()
    doc.add_heading("Main Figures", level=1)
    for figure in ["figure_1_topolink_cci_method_validation.png", "figure_2_topolink_cci_discovery_benchmark.png"]:
        path = FIG_DIR / figure
        if path.exists():
            doc.add_picture(str(path), width=Inches(6.9))
    out = MANUSCRIPT_DIR / "topolink_cci_short_communication_review_copy.docx"
    doc.save(out)
    return out


def write_package_manifest(outputs: dict[str, dict[str, str]], manuscripts: dict[str, Path], docx_path: Path | None, supplement: Path) -> Path:
    review_pdf = docx_path.with_suffix(".pdf") if docx_path else None
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "Nature Methods Brief Communication",
        "fallback_target": "Nature Biotechnology Brief Communication",
        "backend": "Python/Matplotlib via nature-figure contract",
        "figures": outputs,
        "manuscripts": {key: str(path) for key, path in manuscripts.items()},
        "docx": str(docx_path) if docx_path else None,
        "review_pdf": str(review_pdf) if review_pdf and review_pdf.exists() else None,
        "supplementary_manifest": str(supplement),
    }
    path = PACKAGE_ROOT / "package_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def validate_outputs(outputs: dict[str, dict[str, str]]) -> None:
    required = [
        ("Synthetic Truth AUROC", SYNTHETIC_TRUTH, "TopoLink-CCI"),
        ("method completion matrix", METHOD_MATRIX, "FastCCC"),
        ("VWF-SELP evidence", EVIDENCE, "VWF-SELP"),
    ]
    for label, path, token in required:
        text = path.read_text(encoding="utf-8", errors="replace")
        if token not in text:
            raise RuntimeError(f"Validation failed for {label}: missing {token}")
    for fig_outputs in outputs.values():
        for path in fig_outputs.values():
            if not Path(path).exists() or Path(path).stat().st_size == 0:
                raise RuntimeError(f"Missing figure output: {path}")


def main() -> None:
    configure_matplotlib()
    ensure_dirs()
    data = load_data()
    outputs = {
        "figure_1": make_figure_1(data),
        "figure_2": make_figure_2(data),
        "one_figure_fallback": make_fallback(data),
    }
    manuscripts = write_manuscripts()
    supplement = write_supplement_manifest(data)
    docx_path = write_docx(manuscripts)
    package_manifest = write_package_manifest(outputs, manuscripts, docx_path, supplement)
    validate_outputs(outputs)
    print(json.dumps({"package": str(PACKAGE_ROOT), "manifest": str(package_manifest)}, indent=2))


if __name__ == "__main__":
    main()
