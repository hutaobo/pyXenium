"""Create draft Nature Methods Brief Communication figures for TopoLink-CCI.

The script uses the PDC-clean TopoLink-CCI validation outputs plus the
VWF-SELP deep-dive artifacts. It is intentionally self-contained so the
manuscript figure drafts can be regenerated after validation table updates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "pdc_validation_v2_collected" / "topolink_cci_validation_v2"
DEEP_DIVE = ROOT / "vwf_selp_deep_dive"
OUT = ROOT / "topolink_cci_short_communication" / "figures"


COLORS = {
    "vascular": "#0f766e",
    "stromal": "#8a5a2b",
    "immune": "#3f7f3f",
    "notch": "#5b5f97",
    "tumor": "#7a7a7a",
    "pass": "#138a72",
    "partial": "#e69f00",
    "fail": "#b23a48",
    "blue": "#2563eb",
    "light": "#f4f7f6",
    "text": "#1f2933",
}


BIOLOGY_THEME = {
    "VWF-SELP": "vascular",
    "VWF-LRP1": "vascular",
    "MMRN2-CD93": "vascular",
    "CD48-CD2": "immune",
    "DLL4-NOTCH3": "notch",
    "CXCL12-CXCR4": "stromal",
    "JAG1-NOTCH2": "tumor",
}


def load_evidence() -> pd.DataFrame:
    path = VALIDATION / "tables" / "topolink_cci_validation_v2_evidence.tsv"
    df = pd.read_csv(path, sep="\t")
    df["pair"] = df["ligand"] + "-" + df["receptor"]
    df["short_axis"] = df["pair"] + "\n" + df["sender"].str.replace(" Cells", "", regex=False) + " -> " + df["receiver"].str.replace(" Cells", "", regex=False)
    df["theme"] = df["pair"].map(BIOLOGY_THEME).fillna("tumor")
    return df.sort_values("pyxenium_rank")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.06,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )


def rounded_box(ax, xy, width, height, text, fc="#ffffff", ec="#425466", fontsize=9):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        fc=fc,
        ec=ec,
        lw=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        wrap=True,
    )
    return box


def draw_figure_1(evidence: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.2], height_ratios=[1, 1.15])

    # A. Conceptual topology/contact model.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "a")
    ax.set_title("TopoLink-CCI links topology, expression and contact", fontsize=12, fontweight="bold", loc="left")
    for x, y, color, label in [
        (0.26, 0.55, COLORS["vascular"], "sender\ncell"),
        (0.64, 0.55, COLORS["notch"], "receiver\ncell"),
        (0.45, 0.25, COLORS["blue"], "local\nneighbor graph"),
    ]:
        circ = patches.Circle((x, y), 0.12, fc=color, ec="white", lw=2, alpha=0.9)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(0.52, 0.55), xytext=(0.38, 0.55), arrowprops=dict(arrowstyle="->", lw=2.5, color=COLORS["text"]))
    ax.text(0.45, 0.61, "molecular\ninteraction axis", ha="center", fontsize=9)
    rounded_box(ax, (0.05, 0.78), 0.35, 0.13, "ligand-sender\ntopology anchor", fc="#e0f2f1", ec=COLORS["vascular"])
    rounded_box(ax, (0.58, 0.78), 0.35, 0.13, "receptor-receiver\ntopology anchor", fc="#ebe9fb", ec=COLORS["notch"])
    rounded_box(ax, (0.25, 0.03), 0.43, 0.13, "structure bridge +\nspatial local contact", fc="#eff6ff", ec=COLORS["blue"])
    ax.plot([0.22, 0.26], [0.78, 0.67], color=COLORS["vascular"], lw=1.5)
    ax.plot([0.74, 0.64], [0.78, 0.67], color=COLORS["notch"], lw=1.5)
    ax.plot([0.46, 0.45], [0.16, 0.36], color=COLORS["blue"], lw=1.5)

    # B. Score formula.
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    add_panel_label(ax, "b")
    ax.set_title("Discovery score uses a prior-weighted geometric mean", fontsize=12, fontweight="bold", loc="left")
    formula = (
        r"$\mathrm{TopoLink\!-\!CCI}_{l,r,s,t} = \pi_{l,r} \times$" "\n"
        r"$\mathrm{GM}(A_\mathrm{sender}, A_\mathrm{receiver}, B_\mathrm{structure},$" "\n"
        r"$E_\mathrm{sender}, E_\mathrm{receiver}, C_\mathrm{local})$"
    )
    ax.text(0.5, 0.74, formula, ha="center", va="center", fontsize=18, color=COLORS["text"])
    components = [
        ("sender anchor", COLORS["vascular"]),
        ("receiver anchor", COLORS["notch"]),
        ("structure bridge", COLORS["blue"]),
        ("sender expression", "#2f855a"),
        ("receiver expression", "#2f855a"),
        ("local contact", "#c2410c"),
    ]
    for i, (name, color) in enumerate(components):
        x = 0.08 + (i % 3) * 0.3
        y = 0.34 - (i // 3) * 0.18
        rounded_box(ax, (x, y), 0.23, 0.12, name, fc="#ffffff", ec=color, fontsize=9)
    ax.text(0.5, 0.05, "High scores are hypotheses; validation gates test false-positive risk.", ha="center", fontsize=10, color="#4b5563")

    # C. Workflow.
    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "c")
    ax.set_title("Brief Communication workflow", fontsize=12, fontweight="bold", loc="left")
    steps = [
        ("Xenium WTA\ncells + genes + xy", "#e8f5e9"),
        ("pyXenium\ntopology map", "#e0f2f1"),
        ("TopoLink-CCI\ncandidate axes", "#eff6ff"),
        ("orthogonal\nfalse-positive controls", "#fff7ed"),
        ("evidence class\nstrong / moderate / risk", "#f4f7f6"),
    ]
    ys = np.linspace(0.78, 0.22, len(steps))
    for i, (text, fc) in enumerate(steps):
        rounded_box(ax, (0.2, ys[i]), 0.6, 0.11, text, fc=fc, ec="#425466", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, ys[i + 1] + 0.12),
                xytext=(0.5, ys[i]),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=COLORS["text"]),
            )
    ax.text(
        0.5,
        0.06,
        "Primary dataset: Atera Xenium WTA breast cancer\n170,057 cells; 20 cell groups; 3,299 common interaction pairs",
        ha="center",
        fontsize=9,
    )

    # D. Evidence matrix.
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "d")
    layers = [
        ("expression", "expression_specificity_support"),
        ("label\nperm.", "cell_label_permutation_support"),
        ("spatial\nnull", "spatial_null_support"),
        ("matched\ngenes", "matched_gene_control_support"),
        ("downstream", "downstream_target_support"),
        ("received\nsignal", "functional_received_signal_support"),
        ("cross\nmethod", "cross_method_support"),
        ("ablation", "component_ablation_support"),
        ("bootstrap", "bootstrap_stability_support"),
    ]
    matrix = evidence[[col for _, col in layers]].astype(bool).to_numpy()
    n_rows, n_cols = matrix.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([x for x, _ in layers], fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(evidence["pair"].tolist(), fontsize=9)
    ax.set_title("Orthogonal evidence matrix (7/7 strong)", fontsize=12, fontweight="bold", loc="left")
    for i in range(n_rows):
        for j in range(n_cols):
            color = COLORS["pass"] if matrix[i, j] else COLORS["partial"]
            marker = "o" if matrix[i, j] else "X"
            ax.scatter(j, i, s=230, marker=marker, color=color, edgecolor="white", linewidth=1.4)
    ax.grid(True, color="#e5e7eb", lw=0.8)
    ax.tick_params(length=0)
    ax.text(0.99, -0.16, "orange = weak/axis-specific caveat, not artifact risk", transform=ax.transAxes, ha="right", fontsize=8, color="#6b7280")

    fig.suptitle("Figure 1 | TopoLink-CCI method and validation logic", fontsize=16, fontweight="bold")
    return fig


def draw_figure_2(evidence: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1, 1.15], height_ratios=[1, 1])

    # A. Ranked interpretable axes.
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a")
    plot_df = evidence.sort_values("CCI_score", ascending=True)
    bar_colors = [COLORS.get(BIOLOGY_THEME.get(pair, "tumor"), COLORS["tumor"]) for pair in plot_df["pair"]]
    ax.barh(plot_df["pair"], plot_df["CCI_score"].astype(float), color=bar_colors)
    ax.set_xlabel("TopoLink-CCI score")
    ax.set_title("Top interpretable axes", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlim(0.58, 0.82)
    ax.grid(axis="x", color="#e5e7eb")
    for idx, (score, rank) in enumerate(zip(plot_df["CCI_score"], plot_df["pyxenium_rank"])):
        ax.text(float(score) + 0.004, idx, f"rank {int(rank)}", va="center", fontsize=8)

    # B. VWF-SELP score decomposition.
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b")
    comp = pd.read_csv(DEEP_DIVE / "tables" / "component_decomposition.tsv", sep="\t")
    comp_order = ["sender_anchor", "receiver_anchor", "structure_bridge", "sender_expr", "receiver_expr", "local_contact"]
    comp = comp.set_index("component").loc[comp_order].reset_index()
    ax.barh(comp["component"].str.replace("_", "\n"), comp["value"].astype(float), color=[COLORS["vascular"]] * 5 + ["#c2410c"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("component value")
    ax.set_title("VWF-SELP is rank 1 by combined support", fontsize=12, fontweight="bold", loc="left")
    ax.text(0.05, 0.08, "CCI_score = 0.791\nlocal contact = 0.291\n12,779 endothelial-endothelial edges", transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec="#d1d5db", boxstyle="round,pad=0.35"))
    ax.grid(axis="x", color="#e5e7eb")

    # C. Existing spatial hotspot overlay.
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "c")
    img_path = DEEP_DIVE / "figures" / "vwf_selp_hotspot_spatial_overlay.png"
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title("Endothelial VWF-SELP hotspots in tissue", fontsize=12, fontweight="bold", loc="left")

    # D. Cross-method and validation strength.
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "d")
    metrics = evidence[["pair", "support_count", "cross_method_same_lr_count", "bootstrap_rank_median"]].copy()
    metrics = metrics.sort_values("support_count", ascending=True)
    y = np.arange(len(metrics))
    ax.barh(y - 0.18, metrics["support_count"], height=0.32, color=COLORS["pass"], label="validation layers")
    ax.barh(y + 0.18, metrics["cross_method_same_lr_count"], height=0.32, color=COLORS["blue"], label="same-CCI method support")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics["pair"], fontsize=9)
    ax.set_xlim(0, 9)
    ax.set_xlabel("count")
    ax.set_title("Support is not single-algorithm self-consistency", fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color="#e5e7eb")

    # E. VWF/SELP expression specificity heatmap.
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "e")
    img_path = DEEP_DIVE / "figures" / "expression_specificity_heatmap.png"
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title("WTA expression supports endothelial identity", fontsize=12, fontweight="bold", loc="left")

    # F. Biological model summary.
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "f")
    ax.set_title("Lead biological interpretation", fontsize=12, fontweight="bold", loc="left")
    vessel = patches.FancyBboxPatch((0.12, 0.38), 0.76, 0.2, boxstyle="round,pad=0.04,rounding_size=0.12", fc="#d6f5ef", ec=COLORS["vascular"], lw=2)
    ax.add_patch(vessel)
    ax.text(0.5, 0.49, "endothelial activation niche", ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["vascular"])
    ax.text(0.22, 0.68, "VWF", color="#b91c1c", fontsize=16, fontweight="bold")
    ax.text(0.67, 0.68, "SELP", color="#b45309", fontsize=16, fontweight="bold")
    ax.annotate("", xy=(0.44, 0.58), xytext=(0.28, 0.66), arrowprops=dict(arrowstyle="->", lw=2, color="#b91c1c"))
    ax.annotate("", xy=(0.60, 0.58), xytext=(0.69, 0.66), arrowprops=dict(arrowstyle="->", lw=2, color="#b45309"))
    rounded_box(ax, (0.08, 0.12), 0.84, 0.17, "Topology-supported molecular interaction axis;\ncomputational validation, not protein-level proof.", fc="#fff7ed", ec="#c2410c", fontsize=10)
    ax.text(0.5, 0.88, "Recovered themes: vascular adhesion, ECM/stroma,\nimmune recruitment and Notch signaling", ha="center", fontsize=10)

    fig.suptitle("Figure 2 | Whole-dataset discovery and biological interpretation", fontsize=16, fontweight="bold")
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    evidence = load_evidence()
    save_figure(draw_figure_1(evidence), "figure_1_topolink_cci_method_validation")
    save_figure(draw_figure_2(evidence), "figure_2_topolink_cci_discovery_biology")

    source = evidence[
        [
            "pair",
            "sender",
            "receiver",
            "biology_label",
            "CCI_score",
            "pyxenium_rank",
            "support_count",
            "evidence_class",
            "cell_label_perm_fdr",
            "spatial_null_fdr",
            "matched_gene_z",
            "downstream_target_fdr",
            "cross_method_same_lr_count",
            "bootstrap_rank_median",
        ]
    ].copy()
    source.to_csv(OUT / "figure_source_data.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
