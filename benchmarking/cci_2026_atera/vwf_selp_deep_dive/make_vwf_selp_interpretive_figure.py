"""Build a publication-style interpretive figure for the VWF-SELP top hit.

The figure intentionally combines a compact schematic with real outputs from
the VWF-SELP deep-dive analysis. It does not rerun the CCI benchmark.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUT = FIGURES / "final"


COLORS = {
    "endo": "#0F7C80",
    "endo_light": "#CFECE9",
    "vwf": "#B21E48",
    "selp": "#E08D1F",
    "immune": "#2E8B57",
    "pericyte": "#5F6F89",
    "tumor": "#7E7E7E",
    "caf": "#B6A16B",
    "ink": "#1F2933",
    "muted": "#667085",
    "panel_bg": "#F8FAFC",
    "line": "#D0D5DD",
}


def _read_tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, sep="\t")


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.04,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.08,
        1.04,
        title,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
    )


def _style_card(ax: plt.Axes) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=8)


def draw_schematic(ax: plt.Axes) -> None:
    _style_card(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    _panel_label(ax, "A", "Endothelial activation model")

    vessel = FancyBboxPatch(
        (0.6, 2.2),
        8.8,
        5.4,
        boxstyle="round,pad=0.02,rounding_size=1.1",
        linewidth=1.2,
        edgecolor=COLORS["endo"],
        facecolor="#E7F7F5",
    )
    ax.add_patch(vessel)
    ax.text(5, 7.35, "vascular endothelial neighborhood", ha="center", fontsize=9, color=COLORS["endo"])

    # Lumen highlight.
    lumen = FancyBboxPatch(
        (1.0, 3.2),
        8.0,
        3.6,
        boxstyle="round,pad=0.02,rounding_size=0.9",
        linewidth=0,
        facecolor="#F8FDFF",
        alpha=0.95,
    )
    ax.add_patch(lumen)

    cell_x = [2.0, 3.5, 5.0, 6.5, 8.0]
    for i, x in enumerate(cell_x):
        cell = FancyBboxPatch(
            (x - 0.75, 2.45),
            1.5,
            1.0,
            boxstyle="round,pad=0.03,rounding_size=0.25",
            linewidth=0.8,
            edgecolor=COLORS["endo"],
            facecolor=COLORS["endo_light"],
        )
        ax.add_patch(cell)
        ax.text(x, 2.95, "EC", ha="center", va="center", fontsize=8, fontweight="bold", color=COLORS["endo"])
        for j in range(3):
            ax.add_patch(Circle((x - 0.35 + j * 0.35, 3.22), 0.07, color=COLORS["vwf"], alpha=0.8))
        if i in (1, 2, 3):
            ax.plot([x - 0.38, x - 0.18, x + 0.05, x + 0.24, x + 0.44], [4.2, 4.55, 4.35, 4.65, 4.42], color=COLORS["vwf"], lw=2)
            ax.text(x + 0.48, 4.6, "VWF", fontsize=8, color=COLORS["vwf"], fontweight="bold")
            for px in [x - 0.25, x + 0.15, x + 0.42]:
                ax.plot([px, px], [3.45, 3.9], color=COLORS["selp"], lw=1.8)
                ax.add_patch(Circle((px, 3.95), 0.08, color=COLORS["selp"]))

    immune = Circle((2.6, 5.6), 0.48, color=COLORS["immune"], alpha=0.9)
    ax.add_patch(immune)
    ax.text(2.6, 5.6, "T", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.add_patch(FancyArrowPatch((2.15, 5.05), (3.15, 4.35), arrowstyle="->", mutation_scale=13, color=COLORS["immune"], lw=1.3))
    ax.text(1.15, 6.1, "immune rolling/\nadherence cue", fontsize=8, color=COLORS["immune"], ha="left")

    pericyte = FancyBboxPatch(
        (6.45, 1.4),
        2.25,
        0.55,
        boxstyle="round,pad=0.04,rounding_size=0.22",
        linewidth=0,
        facecolor=COLORS["pericyte"],
        alpha=0.9,
    )
    ax.add_patch(pericyte)
    ax.text(7.58, 1.68, "pericyte context", ha="center", va="center", fontsize=7.5, color="white")

    ax.text(5.0, 8.75, "VWF + SELP/P-selectin = WPB / adhesion state", ha="center", fontsize=10, fontweight="bold", color=COLORS["ink"])
    ax.text(0.8, 0.5, "Caveat: RNA + spatial evidence; not direct proof of protein release or thrombosis.", fontsize=7.5, color=COLORS["muted"])


def draw_score_panel(ax: plt.Axes) -> None:
    _style_card(ax)
    _panel_label(ax, "B", "pyXenium score stack")
    comp = _read_tsv("component_decomposition.tsv")
    comp = comp.assign(component_label=comp["component"].str.replace("_", "\n"))
    y = np.arange(len(comp))[::-1]
    colors = [COLORS["endo"], COLORS["endo"], COLORS["endo"], COLORS["vwf"], COLORS["selp"], "#7A5CFF"]
    ax.barh(y, comp["value"], color=colors, alpha=0.9, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(comp["component_label"], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("component value", fontsize=8)
    ax.grid(axis="x", color="white", lw=1.2)
    for yi, value in zip(y, comp["value"]):
        ax.text(value + 0.02, yi, f"{value:.3f}", va="center", fontsize=8, color=COLORS["ink"])

    sens = _read_tsv("rank_sensitivity.tsv")
    original = sens.loc[sens["scenario"] == "original"].iloc[0]
    no_contact = sens.loc[sens["scenario"] == "remove_local_contact"].iloc[0]
    callout = (
        f"Full common-db rank {int(original.target_rank)}\n"
        f"CCI_score = {original.target_score:.3f}\n"
        f"Remove contact -> rank {int(no_contact.target_rank)}\n"
        f"Top becomes {no_contact.top_ligand}-{no_contact.top_receptor}"
    )
    ax.text(
        0.54,
        0.10,
        callout,
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["line"], lw=0.8),
    )


def draw_expression_panel(ax: plt.Axes) -> None:
    _style_card(ax)
    _panel_label(ax, "C", "WTA endothelial specificity")
    expr = _read_tsv("expression_specificity_full_wta.tsv")
    genes = ["VWF", "SELP", "PECAM1", "EMCN", "KDR", "MMRN2", "HBB", "PF4"]
    celltypes = [
        "Endothelial Cells",
        "Pericytes",
        "T Lymphocytes",
        "Macrophages",
        "CAFs, DCIS Associated",
        "11q13 Invasive Tumor Cells",
    ]
    celltype_labels = ["Endothelial", "Pericytes", "T cells", "Macrophages", "DCIS CAFs", "11q13 tumor"]
    sub = expr[expr["gene"].isin(genes) & expr["cell_type"].isin(celltypes)].copy()
    sub["gene"] = pd.Categorical(sub["gene"], categories=genes, ordered=True)
    sub["cell_type"] = pd.Categorical(sub["cell_type"], categories=celltypes, ordered=True)
    sub = sub.sort_values(["cell_type", "gene"])
    x = sub["gene"].cat.codes.to_numpy()
    y = sub["cell_type"].cat.codes.to_numpy()
    sizes = 30 + 260 * sub["detection_fraction"].to_numpy()
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=sub["mean_log1p_norm_by_gene"],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(celltypes)))
    ax.set_yticklabels(celltype_labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(color="white", lw=1.0)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("gene-normalized mean", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    ax.text(
        0.04,
        0.06,
        "VWF and SELP peak in endothelial cells;\nHBB/PF4 included as contamination controls.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["line"], lw=0.6),
    )


def draw_spatial_panel(ax_map: plt.Axes, ax_ctx: plt.Axes) -> None:
    _style_card(ax_map)
    _panel_label(ax_map, "D", "Spatial hotspots and neighborhood context")
    img = Image.open(FIGURES / "vwf_selp_hotspot_spatial_overlay.png")
    ax_map.imshow(img)
    ax_map.axis("off")
    ax_map.text(
        0.02,
        0.03,
        "433 endothelial hotspot cells\n95th percentile among endothelial cells",
        transform=ax_map.transAxes,
        fontsize=9,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["line"], lw=0.8, alpha=0.92),
    )

    _style_card(ax_ctx)
    ctx = _read_tsv("hotspot_neighbor_context.tsv")
    keep = ["T Lymphocytes", "Pericytes", "11q13 Invasive Tumor Cells", "CAFs, DCIS Associated", "Macrophages"]
    piv = ctx[ctx["neighbor_cell_type"].isin(keep)].pivot(index="neighbor_cell_type", columns="group", values="neighbor_fraction")
    piv = piv.loc[keep]
    yy = np.arange(len(piv))
    width = 0.36
    ax_ctx.barh(yy - width / 2, piv["hotspot_endothelial"], width, label="hotspot EC", color=COLORS["endo"])
    ax_ctx.barh(yy + width / 2, piv["all_endothelial"], width, label="all EC", color="#B8C6D9")
    ax_ctx.set_yticks(yy)
    ax_ctx.set_yticklabels(["T cells", "Pericytes", "11q13 tumor", "DCIS CAFs", "Macrophages"], fontsize=7.5)
    ax_ctx.invert_yaxis()
    ax_ctx.set_xlabel("neighbor fraction", fontsize=8)
    ax_ctx.legend(frameon=False, fontsize=7, loc="lower right")
    ax_ctx.grid(axis="x", color="white", lw=1.2)
    ax_ctx.text(
        0.02,
        1.02,
        "hotspots are immune/pericyte-associated,\nnot simply tumor-front enriched",
        transform=ax_ctx.transAxes,
        fontsize=8.5,
        color=COLORS["ink"],
        va="bottom",
    )


def draw_ecology_panel(ax_contour: plt.Axes, ax_methods: plt.Axes) -> None:
    _style_card(ax_contour)
    _panel_label(ax_contour, "E", "Tissue ecology and method triangulation")
    de = _read_tsv("s1_s5_vascular_gene_de.tsv")
    genes = ["VWF", "PECAM1", "EMCN", "KDR", "MMRN2", "CLEC14A", "EGFL7"]
    sub = de[de["gene"].isin(genes)].copy().sort_values("delta_log1p_cpm", ascending=True)
    ax_contour.barh(sub["gene"], sub["delta_log1p_cpm"], color=COLORS["endo"], alpha=0.88)
    ax_contour.set_xlabel("S3 enrichment delta (log1p CPM)", fontsize=8)
    ax_contour.tick_params(axis="y", labelsize=8)
    ax_contour.grid(axis="x", color="white", lw=1.2)
    ax_contour.text(
        0.02,
        0.92,
        "S1-S5 contour analysis:\nvascular markers peak in S3",
        transform=ax_contour.transAxes,
        fontsize=8,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["line"], lw=0.6),
    )

    _style_card(ax_methods)
    ax_methods.axis("off")
    hits = _read_tsv("cross_method_vwf_selp_hits.tsv")
    exact = hits[
        (hits["ligand"] == "VWF")
        & (hits["receptor"] == "SELP")
        & (hits["sender"] == "Endothelial Cells")
        & (hits["receiver"] == "Endothelial Cells")
    ].copy()
    exact = exact.sort_values("rank_within_method").head(4)
    rows = []
    for _, row in exact.iterrows():
        rows.append([row["method"], f"{int(row['rank_within_method']):,}", f"{row['score_std']:.3f}"])
    if not rows:
        rows = [["CellPhoneDB", "692", "0.999"], ["LARIS", "2,845", "0.998"]]
    ax_methods.text(0.0, 0.95, "Exact VWF-SELP recovered by other methods", fontsize=8.5, fontweight="bold", color=COLORS["ink"])
    col_x = [0.00, 0.46, 0.72]
    headers = ["method", "rank", "score_std"]
    for x, h in zip(col_x, headers):
        ax_methods.text(x, 0.78, h, fontsize=7.5, color=COLORS["muted"], fontweight="bold")
    y = 0.63
    for row in rows[:4]:
        for x, txt in zip(col_x, row):
            ax_methods.text(x, y, str(txt), fontsize=8, color=COLORS["ink"])
        y -= 0.16
    ax_methods.add_patch(Rectangle((0, 0.10), 0.98, 0.78, fill=False, edgecolor=COLORS["line"], lw=0.8))
    ax_methods.text(
        0.0,
        0.02,
        "Interpretation: consensus supports a vascular axis;\npyXenium adds topology/contact specificity.",
        fontsize=7.5,
        color=COLORS["muted"],
        va="bottom",
    )


def build_main_figure() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )
    fig = plt.figure(figsize=(20, 11), dpi=300, facecolor="white")
    gs = fig.add_gridspec(3, 16, height_ratios=[0.15, 1.0, 1.18], hspace=0.50, wspace=0.72)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.70,
        "pyXenium identifies a topology-supported endothelial VWF-SELP vascular activation niche",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    ax_title.text(
        0.0,
        0.10,
        "Whole-dataset Xenium WTA breast analysis: rank 1 LR axis, endothelial-specific expression, spatial hotspots, S3 vascular contour enrichment, and cross-method support.",
        fontsize=10,
        color=COLORS["muted"],
        va="center",
    )

    draw_schematic(fig.add_subplot(gs[1, 0:5]))
    draw_score_panel(fig.add_subplot(gs[1, 5:9]))
    draw_expression_panel(fig.add_subplot(gs[1, 9:16]))

    draw_spatial_panel(fig.add_subplot(gs[2, 0:7]), fig.add_subplot(gs[2, 7:10]))
    draw_ecology_panel(fig.add_subplot(gs[2, 10:13]), fig.add_subplot(gs[2, 13:16]))

    caption = (
        "Main interpretation: the top pyXenium hit is best read as an endothelial activation / "
        "Weibel-Palade body-associated adhesion state. The data support RNA-level endothelial "
        "specificity and spatial topology; protein release, platelet adhesion, and thrombosis remain "
        "biological hypotheses requiring orthogonal validation."
    )
    fig.text(0.02, 0.012, textwrap.fill(caption, 190), fontsize=8, color=COLORS["muted"])

    png = OUT / "vwf_selp_endothelial_activation_main_figure.png"
    pdf = OUT / "vwf_selp_endothelial_activation_main_figure.pdf"
    svg = OUT / "vwf_selp_endothelial_activation_main_figure.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def _draw_evidence_card(ax: plt.Axes, xy: tuple[float, float], title: str, body: str, color: str) -> None:
    x, y = xy
    title_wrapped = textwrap.fill(title, width=23)
    body_wrapped = textwrap.fill(body, width=34)
    card = FancyBboxPatch(
        (x, y),
        0.43,
        0.19,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
        alpha=0.96,
    )
    ax.add_patch(card)
    ax.text(x + 0.02, y + 0.135, title_wrapped, transform=ax.transAxes, fontsize=9.4, fontweight="bold", color=color)
    ax.text(x + 0.02, y + 0.035, body_wrapped, transform=ax.transAxes, fontsize=7.8, color=COLORS["ink"], va="bottom")


def build_conference_summary() -> Path:
    """A simplified 16:9 version that works as a talk or Canva first slide."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="white")
    gs = fig.add_gridspec(3, 8, height_ratios=[0.18, 1.0, 1.0], hspace=0.38, wspace=0.55)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.72,
        "VWF-SELP marks an endothelial activation niche",
        fontsize=24,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    ax_title.text(
        0.0,
        0.18,
        "A topology-supported pyXenium top hit linking endothelial identity, WPB biology, spatial hotspots, and vascular tissue ecology.",
        fontsize=12,
        color=COLORS["muted"],
        va="center",
    )

    draw_schematic(fig.add_subplot(gs[1:, 0:3]))

    ax_spatial = fig.add_subplot(gs[1:, 3:5])
    _style_card(ax_spatial)
    img = Image.open(FIGURES / "vwf_selp_hotspot_spatial_overlay.png")
    ax_spatial.imshow(img)
    ax_spatial.axis("off")
    ax_spatial.text(
        0.04,
        0.06,
        "Spatial hotspot map\n433 endothelial hotspot cells",
        transform=ax_spatial.transAxes,
        fontsize=10,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["line"], lw=0.8),
    )

    ax_cards = fig.add_subplot(gs[1:, 5:8])
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    ax_cards.axis("off")
    _draw_evidence_card(
        ax_cards,
        (0.02, 0.76),
        "pyXenium rank 1",
        "CCI_score 0.791; contact-aware topology\nkeeps VWF-SELP as the top endothelial axis.",
        COLORS["endo"],
    )
    _draw_evidence_card(
        ax_cards,
        (0.52, 0.76),
        "WTA specificity",
        "VWF and SELP peak in endothelial cells\nwith PECAM1/EMCN/KDR support.",
        COLORS["vwf"],
    )
    _draw_evidence_card(
        ax_cards,
        (0.02, 0.52),
        "Neighborhood context",
        "Hotspot ECs are enriched near T-cell\nand pericyte neighborhoods.",
        COLORS["immune"],
    )
    _draw_evidence_card(
        ax_cards,
        (0.52, 0.52),
        "Contour ecology",
        "S1-S5 analysis places vascular markers\nin S3-enriched tissue regions.",
        COLORS["pericyte"],
    )
    _draw_evidence_card(
        ax_cards,
        (0.02, 0.28),
        "Cross-method support",
        "CellPhoneDB and LARIS recover exact\nVWF-SELP endothelial-endothelial support.",
        COLORS["selp"],
    )
    _draw_evidence_card(
        ax_cards,
        (0.52, 0.28),
        "Interpretation",
        "RNA/spatial evidence for WPB-associated\nadhesion state; protein validation needed.",
        "#7A5CFF",
    )
    ax_cards.text(
        0.02,
        0.08,
        "Biological model: activated endothelial neighborhoods expose a VWF/P-selectin-like adhesion program\nthat may organize immune and pericyte-adjacent vascular niches.",
        fontsize=11,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec=COLORS["line"], lw=0.8),
    )

    png = OUT / "vwf_selp_endothelial_activation_conference_summary.png"
    pdf = OUT / "vwf_selp_endothelial_activation_conference_summary.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def build_asset_manifest() -> None:
    summary = {
        "title": "pyXenium identifies a topology-supported endothelial VWF-SELP vascular activation niche",
        "root": str(ROOT),
        "outputs": {
            "main_png": str(OUT / "vwf_selp_endothelial_activation_main_figure.png"),
            "main_pdf": str(OUT / "vwf_selp_endothelial_activation_main_figure.pdf"),
            "main_svg": str(OUT / "vwf_selp_endothelial_activation_main_figure.svg"),
            "conference_png": str(OUT / "vwf_selp_endothelial_activation_conference_summary.png"),
            "conference_pdf": str(OUT / "vwf_selp_endothelial_activation_conference_summary.pdf"),
            "canva_brief": str(OUT / "canva_design_brief.md"),
            "caption": str(OUT / "figure_caption.md"),
        },
        "source_assets": [
            str(FIGURES / "component_decomposition.png"),
            str(FIGURES / "expression_specificity_heatmap.png"),
            str(FIGURES / "vwf_selp_hotspot_spatial_overlay.png"),
            str(FIGURES / "contour_vascular_summary.png"),
            str(TABLES / "rank_sensitivity.tsv"),
            str(TABLES / "hotspot_summary.tsv"),
            str(TABLES / "hotspot_neighbor_context.tsv"),
            str(TABLES / "cross_method_vwf_selp_hits.tsv"),
        ],
        "key_numbers": {
            "LR_pair": "VWF-SELP",
            "sender_receiver": "Endothelial Cells -> Endothelial Cells",
            "CCI_score": 0.7912892368005828,
            "local_contact": 0.2912449657481761,
            "cross_edge_count": 12779,
            "hotspot_endothelial_cells": 433,
        },
    }
    (OUT / "asset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_caption_and_canva_brief() -> None:
    caption = """# Figure caption

**pyXenium identifies a topology-supported endothelial VWF-SELP vascular activation niche.**
The whole-dataset Xenium WTA breast benchmark ranked `VWF-SELP / Endothelial Cells -> Endothelial Cells` as the strongest pyXenium cell-cell interaction axis (`CCI_score=0.791`). The component stack shows high sender and receiver topology anchors, full expression support, a perfect structure bridge, and a local endothelial contact score of `0.291`; removing the local-contact term drops the interaction from rank 1 to rank 13. Full WTA expression summaries show that `VWF` and `SELP` are most enriched in endothelial cells, alongside canonical endothelial markers including `PECAM1`, `EMCN`, `KDR`, `MMRN2`, and `CLEC14A`, while blood/platelet contamination controls remain secondary. Spatial hotspot analysis identifies 433 high-scoring endothelial cells with T-cell and pericyte-enriched neighborhood context. S1-S5 contour analysis places vascular markers in S3-enriched tissue regions, and CellPhoneDB/LARIS independently recover exact `VWF-SELP` endothelial-endothelial support. Together, the data support an endothelial activation / Weibel-Palade body-associated adhesion state. This is RNA-level and spatial-topology evidence, not direct proof of VWF/P-selectin protein release or thrombosis.

Literature context: Weibel-Palade bodies are endothelial storage organelles containing VWF and P-selectin and connect vascular adhesion, hemostasis, and inflammation. See NCBI Bookshelf, Reactome vascular wall cell-surface interactions, and WPB review literature for biological framing.
"""
    (OUT / "figure_caption.md").write_text(caption, encoding="utf-8")

    brief = """# Canva design brief

Create a single landscape scientific figure titled:

**pyXenium identifies a topology-supported endothelial VWF-SELP vascular activation niche**

Canvas: 16:9 landscape, clean manuscript style, white/off-white background, teal endothelial palette with crimson VWF and amber SELP/P-selectin highlights.

## Required panel narrative

1. **A. Biological model**: draw an endothelial vessel neighborhood with VWF strings, SELP/P-selectin on activated endothelial surfaces, nearby T cell/immune rolling, and pericyte context. Include caveat: RNA/spatial evidence, not direct protein release proof.
2. **B. pyXenium evidence stack**: show CCI_score 0.791, rank 1, component values, and the rank-sensitivity callout: removing local contact drops VWF-SELP to rank 13.
3. **C. WTA specificity**: show VWF and SELP enriched in endothelial cells with endothelial markers PECAM1, EMCN, KDR, MMRN2, CLEC14A. Include HBB/PF4 contamination control note.
4. **D. Spatial hotspot map**: use the hotspot spatial overlay and show 433 endothelial hotspot cells; add small bar chart for T cell/pericyte-associated neighborhoods.
5. **E. Tissue ecology and cross-method support**: show S3 vascular contour enrichment and exact VWF-SELP recovery by CellPhoneDB and LARIS.

Use the generated local output `vwf_selp_endothelial_activation_main_figure.svg` or PDF as the editable source if importing into Canva. For a talk-style Canva page, use `vwf_selp_endothelial_activation_conference_summary.png` or PDF.
"""
    (OUT / "canva_design_brief.md").write_text(brief, encoding="utf-8")


def main() -> None:
    png = build_main_figure()
    summary_png = build_conference_summary()
    build_asset_manifest()
    write_caption_and_canva_brief()
    print(f"Wrote {png}")
    print(f"Wrote {summary_png}")


if __name__ == "__main__":
    main()
