from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, patches
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_INPUT_DIR = Path(
    r"D:/GitHub/pyXenium/benchmarking/cci_2026_atera/pdc_validation_v2_collected/topolink_cci_validation_v2"
)
SUPPORT_COLS = [
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
SUPPORT_LABELS = [
    "Expression\nspecificity",
    "Label\npermutation",
    "Spatial\nnull",
    "Matched\ngenes",
    "Downstream\ntargets",
    "Received\nsignal",
    "Cross-method\nconsensus",
    "Component\nablation",
    "Bootstrap\nstability",
]
EXPECTED_ORDER = [
    "VWF-SELP",
    "VWF-LRP1",
    "MMRN2-CD93",
    "CD48-CD2",
    "DLL4-NOTCH3",
    "CXCL12-CXCR4",
    "JAG1-NOTCH2",
]
BIOLOGY_COLORS = {
    "WPB / endothelial activation": "#0f766e",
    "vascular-stromal matrix/scavenger axis": "#2c7fb8",
    "CD93-MMRN2 angiogenesis": "#41ab5d",
    "T-cell adhesion/co-stimulation": "#31a354",
    "endothelial-pericyte Notch": "#756bb1",
    "CAF-immune chemokine recruitment": "#d95f0e",
    "tumor-intrinsic Notch signaling": "#7f1d1d",
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def neglog10(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return -math.log10(max(value, 1e-300))


def wrap(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def load_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    tables = input_dir / "tables"
    required = {
        "evidence": tables / "topolink_cci_validation_v2_evidence.tsv",
        "cell_perm": tables / "cell_label_permutation.tsv",
        "spatial": tables / "spatial_neighborhood_controls.tsv",
        "matched": tables / "topolink_cci_validation_v2_false_positive_controls.tsv",
        "downstream": tables / "downstream_target_support.tsv",
        "bootstrap": tables / "bootstrap_stability_summary.tsv",
        "ablation": tables / "component_ablation.tsv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing validation table(s):\n" + "\n".join(missing))
    data = {name: pd.read_csv(path, sep="\t") for name, path in required.items()}
    evidence = data["evidence"].copy()
    evidence["lr_pair"] = evidence["ligand"].astype(str) + "-" + evidence["receptor"].astype(str)
    evidence["axis_label_short"] = evidence["lr_pair"] + "\n" + evidence["sender"].astype(str) + " -> " + evidence["receiver"].astype(str)
    for col in SUPPORT_COLS:
        evidence[col] = as_bool(evidence[col])
    evidence = evidence.sort_values("pyxenium_rank", kind="mergesort").reset_index(drop=True)
    data["evidence"] = evidence
    return data


def validate_inputs(evidence: pd.DataFrame) -> None:
    observed = evidence["lr_pair"].tolist()
    if observed != EXPECTED_ORDER:
        raise AssertionError(f"Unexpected LR order: {observed}")
    top = evidence.iloc[0]
    if top["lr_pair"] != "VWF-SELP" or int(top["pyxenium_rank"]) != 1 or round(float(top["CCI_score"]), 3) != 0.791:
        raise AssertionError("VWF-SELP is not shown as rank 1 with CCI_score=0.791.")
    support_sums = evidence[SUPPORT_COLS].sum(axis=1).astype(int)
    if not np.array_equal(support_sums.to_numpy(), evidence["support_count"].astype(int).to_numpy()):
        raise AssertionError("support_count does not match evidence matrix columns.")
    cxcl = evidence.loc[evidence["lr_pair"] == "CXCL12-CXCR4"].iloc[0]
    if bool(cxcl["spatial_null_support"]):
        raise AssertionError("CXCL12-CXCR4 should remain spatial-null negative in the main figure.")
    if int((evidence["evidence_class"] == "strong").sum()) != 7:
        raise AssertionError("Expected 7 strong axes in PDC clean v2 results.")


def add_panel_label(ax: plt.Axes, label: str, title: str | None = None) -> None:
    ax.text(-0.04, 1.05, label, transform=ax.transAxes, fontsize=13, weight="bold", va="bottom", ha="right")
    if title:
        ax.text(0.0, 1.05, title, transform=ax.transAxes, fontsize=10.5, weight="bold", va="bottom", ha="left")


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "A", "Classic LR/CCC false-positive controls applied to TopoLink-CCI discoveries")
    steps = [
        ("pyXenium\nLR discovery", "#0f766e"),
        ("7 classic\ncandidate axes", "#155e75"),
        ("Orthogonal\nvalidation gates", "#1d4ed8"),
        ("Evidence class\nstrong/moderate", "#166534"),
    ]
    x_positions = np.linspace(0.05, 0.92, len(steps))
    y = 0.55
    for i, ((text, color), x) in enumerate(zip(steps, x_positions)):
        box = patches.FancyBboxPatch(
            (x - 0.09, y - 0.13),
            0.18,
            0.26,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.3, color="white", weight="bold", transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.115, y),
                xytext=(x + 0.11, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#334155"),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    gates = [
        "label permutation",
        "spatial null",
        "matched genes",
        "downstream targets",
        "cross-method consensus",
        "ablation + bootstrap",
    ]
    for i, gate in enumerate(gates):
        x = 0.37 + (i % 3) * 0.16
        yy = 0.20 - (i // 3) * 0.12
        ax.scatter([x], [yy], s=95, color="#e0f2fe", edgecolor="#0369a1", transform=ax.transAxes, zorder=3)
        ax.text(x + 0.025, yy, gate, ha="left", va="center", fontsize=8.2, color="#0f172a", transform=ax.transAxes)
    ax.text(
        0.02,
        0.04,
        "Principle: CCI_score nominates candidates; independent controls reduce false-positive risk.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#475569",
    )


def draw_panel_b(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    add_panel_label(ax, "B", "Nine-layer evidence matrix")
    mat = evidence[SUPPORT_COLS].to_numpy(dtype=float)
    ax.imshow(np.ones_like(mat), cmap="Greys", vmin=0, vmax=1, alpha=0.08, aspect="auto")
    for i, row in evidence.iterrows():
        color = BIOLOGY_COLORS.get(row["biology_label"], "#64748b")
        for j, passed in enumerate(mat[i]):
            face = "#0f766e" if passed else "#f8fafc"
            edge = color if passed else "#cbd5e1"
            ax.scatter(j, i, s=165, marker="o", facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3)
            if passed:
                ax.text(j, i, "✓", ha="center", va="center", fontsize=8.2, color="white", weight="bold")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color="#f59e0b", weight="bold")
    ax.set_xticks(np.arange(len(SUPPORT_LABELS)))
    ax.set_xticklabels(SUPPORT_LABELS, rotation=45, ha="right", fontsize=7.2)
    ax.set_yticks(np.arange(len(evidence)))
    ax.set_yticklabels(evidence["lr_pair"], fontsize=8.4)
    ax.set_xlim(-0.7, len(SUPPORT_LABELS) - 0.3)
    ax.set_ylim(len(evidence) - 0.4, -0.6)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.text(
        0.98,
        1.05,
        "7/7 strong; 0 artifact risk",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#166534",
        weight="bold",
    )


def draw_panel_c(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    add_panel_label(ax, "C", "Quantitative control strength")
    metrics = [
        ("Label perm\n-log10 FDR", evidence["cell_label_perm_fdr"].map(neglog10), "cell_label_permutation_support"),
        ("Spatial null\n-log10 FDR", evidence["spatial_null_fdr"].map(neglog10), "spatial_null_support"),
        ("Matched\ngene z", pd.to_numeric(evidence["matched_gene_z"]), "matched_gene_control_support"),
        ("Downstream\n-log10 FDR", evidence["downstream_target_fdr"].map(neglog10), "downstream_target_support"),
    ]
    y = np.arange(len(evidence))
    for j, (label, values, support_col) in enumerate(metrics):
        vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        vmax = np.nanmax(finite) if len(finite) else 1.0
        for i, val in enumerate(vals):
            supported = bool(evidence.iloc[i][support_col])
            color = BIOLOGY_COLORS.get(evidence.iloc[i]["biology_label"], "#64748b") if supported else "#f59e0b"
            size = 45 + 120 * min(max(val / max(vmax, 1e-9), 0), 1) if np.isfinite(val) else 45
            ax.scatter(j, i, s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3, alpha=0.92)
            text = f"{val:.1f}" if "z" in label else f"{val:.2f}"
            ax.text(j + 0.13, i, text, va="center", ha="left", fontsize=6.8, color="#334155")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(evidence["lr_pair"], fontsize=8.1)
    ax.set_xlim(-0.45, len(metrics) - 0.1)
    ax.set_ylim(len(evidence) - 0.4, -0.6)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.06, 5, "spatial-null weak\nbut other layers strong", fontsize=6.8, color="#d97706", ha="left", va="center")


def draw_panel_d(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    add_panel_label(ax, "D", "Robustness and component ablation")
    y = np.arange(len(evidence))
    boot = pd.to_numeric(evidence["bootstrap_rank_median"], errors="coerce").to_numpy(dtype=float)
    iqr = pd.to_numeric(evidence["bootstrap_rank_iqr"], errors="coerce").to_numpy(dtype=float)
    ablation = pd.to_numeric(evidence["max_rank_after_ablation"], errors="coerce").to_numpy(dtype=float)
    ax.errorbar(boot, y - 0.14, xerr=np.vstack([iqr / 2, iqr / 2]), fmt="o", color="#0f766e", ecolor="#99f6e4", ms=5, label="Bootstrap median rank")
    ax.scatter(ablation, y + 0.14, marker="^", s=48, color="#d97706", edgecolor="white", linewidth=0.7, label="Worst rank after ablation")
    ax.axvline(750, color="#f59e0b", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1200)
    ax.set_yticks(y)
    ax.set_yticklabels(evidence["lr_pair"], fontsize=8.1)
    ax.set_xlabel("Rank (lower is better; log scale)", fontsize=8.5)
    ax.set_ylim(len(evidence) - 0.4, -0.6)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.text(1.02, 0, "VWF-SELP\nmedian rank 1", fontsize=6.9, color="#0f766e", va="center")


def draw_panel_e(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "E", "Biological axes retained after validation")
    y0 = 0.93
    h = 0.12
    for i, row in evidence.iterrows():
        color = BIOLOGY_COLORS.get(row["biology_label"], "#64748b")
        y = y0 - i * h
        box = patches.FancyBboxPatch(
            (0.02, y - 0.085),
            0.96,
            0.092,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.add_patch(patches.Rectangle((0.025, y - 0.079), 0.014, 0.079, color=color, transform=ax.transAxes))
        ax.text(0.05, y - 0.018, row["lr_pair"], fontsize=8.4, weight="bold", color="#0f172a", transform=ax.transAxes, va="center")
        ax.text(0.23, y - 0.018, wrap(row["biology_label"], 25), fontsize=6.9, color="#334155", transform=ax.transAxes, va="center")
        ax.text(
            0.70,
            y - 0.018,
            f"rank {int(row['pyxenium_rank'])} | LR {float(row['CCI_score']):.3f} | {int(row['support_count'])}/9",
            fontsize=7.0,
            color="#0f172a",
            transform=ax.transAxes,
            va="center",
        )
    ax.text(
        0.02,
        0.02,
        "Computational validation; not protein-level or functional proof.",
        fontsize=7.4,
        color="#b45309",
        transform=ax.transAxes,
    )


def make_main_figure(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    evidence = data["evidence"]
    fig = plt.figure(figsize=(11.8, 8.4), constrained_layout=False)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.9, 1.7, 1.55], width_ratios=[1.35, 1.0], hspace=0.58, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    draw_panel_a(ax_a)
    draw_panel_b(ax_b, evidence)
    draw_panel_c(ax_c, evidence)
    draw_panel_d(ax_d, evidence)
    draw_panel_e(ax_e, evidence)
    fig.suptitle(
        "TopoLink-CCI discoveries pass multi-layer computational false-positive controls",
        fontsize=14,
        weight="bold",
        y=0.985,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "topolink_cci_validation_main_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "topolink_cci_validation_main_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "topolink_cci_validation_main_figure.svg", bbox_inches="tight")
    plt.close(fig)


def draw_readable_panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "A", "Validation concept: discovery is separated from credibility checks")
    ax.text(
        0.02,
        0.78,
        "pyXenium nominates LR axes",
        fontsize=13,
        weight="bold",
        color="#0f766e",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.58,
        "CCI_score ranks candidate biology, but does not by itself prove cell-cell communication.",
        fontsize=9.5,
        color="#334155",
        transform=ax.transAxes,
    )
    ax.annotate(
        "",
        xy=(0.37, 0.67),
        xytext=(0.29, 0.67),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#475569"),
        xycoords=ax.transAxes,
    )
    ax.text(
        0.40,
        0.78,
        "Independent false-positive controls",
        fontsize=13,
        weight="bold",
        color="#1d4ed8",
        transform=ax.transAxes,
    )
    gate_groups = [
        ("Permutation", "Cell labels\ncommunication probability", "#dbeafe", "#1d4ed8"),
        ("Spatial", "neighbor null\nmatched genes", "#dcfce7", "#166534"),
        ("Biology", "downstream targets\nreceived signal", "#fef3c7", "#b45309"),
        ("Consensus", "other methods\nbootstrap/ablation", "#ede9fe", "#6d28d9"),
    ]
    for i, (title, body, face, edge) in enumerate(gate_groups):
        x = 0.40 + i * 0.145
        box = patches.FancyBboxPatch(
            (x, 0.31),
            0.125,
            0.31,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x + 0.0625, 0.53, title, ha="center", va="center", fontsize=9.1, weight="bold", color=edge, transform=ax.transAxes)
        ax.text(x + 0.0625, 0.40, body, ha="center", va="center", fontsize=7.7, color="#334155", transform=ax.transAxes)
    ax.annotate(
        "",
        xy=(0.94, 0.67),
        xytext=(0.86, 0.67),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#475569"),
        xycoords=ax.transAxes,
    )
    ax.text(0.965, 0.78, "Evidence class", ha="right", fontsize=13, weight="bold", color="#166534", transform=ax.transAxes)
    ax.text(0.965, 0.58, "strong / moderate /\nhypothesis / artifact risk", ha="right", fontsize=9.0, color="#334155", transform=ax.transAxes)
    ax.text(
        0.02,
        0.08,
        "Main-message rule: a high LR axis is credible only when multiple orthogonal checks agree.",
        fontsize=9.0,
        color="#7c2d12",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed", edgecolor="#fed7aa"),
    )


def draw_readable_panel_b(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "B", "Seven classic LR axes pass the validation framework")
    cols = [
        ("LR axis", 0.03),
        ("Biology", 0.19),
        ("Rank | CCI_score", 0.43),
        ("Passed layers", 0.58),
        ("Label", 0.71),
        ("Spatial", 0.78),
        ("Matched", 0.85),
        ("Target", 0.92),
        ("Notes", 0.98),
    ]
    for label, x in cols:
        ax.text(x, 0.92, label, ha="right" if label == "Notes" else "left", va="center", fontsize=8.6, weight="bold", color="#0f172a", transform=ax.transAxes)
    ax.plot([0.02, 0.985], [0.885, 0.885], color="#cbd5e1", lw=1.0, transform=ax.transAxes)
    y_start = 0.80
    row_h = 0.105
    for i, row in evidence.iterrows():
        y = y_start - i * row_h
        color = BIOLOGY_COLORS.get(row["biology_label"], "#64748b")
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.02, y - 0.045),
                0.965,
                0.078,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor=bg,
                edgecolor="#e2e8f0",
                linewidth=0.6,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(patches.Rectangle((0.024, y - 0.039), 0.010, 0.066, color=color, transform=ax.transAxes))
        ax.text(0.04, y, row["lr_pair"], ha="left", va="center", fontsize=9.0, weight="bold", color="#0f172a", transform=ax.transAxes)
        ax.text(0.19, y, wrap(row["biology_label"], 29), ha="left", va="center", fontsize=7.2, color="#334155", transform=ax.transAxes)
        ax.text(0.43, y, f"#{int(row['pyxenium_rank'])} | {float(row['CCI_score']):.3f}", ha="left", va="center", fontsize=8.1, color="#0f172a", transform=ax.transAxes)
        ax.text(0.60, y, f"{int(row['support_count'])}/9 strong", ha="left", va="center", fontsize=8.1, color="#166534", weight="bold", transform=ax.transAxes)
        compact_checks = [
            ("cell_label_permutation_support", 0.725),
            ("spatial_null_support", 0.795),
            ("matched_gene_control_support", 0.865),
            ("downstream_target_support", 0.932),
        ]
        failed = []
        for col, x in compact_checks:
            passed = bool(row[col])
            ax.scatter([x], [y], s=105, color="#0f766e" if passed else "#f59e0b", edgecolor="white", linewidth=0.8, transform=ax.transAxes, zorder=3)
            ax.text(x, y, "✓" if passed else "–", ha="center", va="center", fontsize=7.6, color="white", weight="bold", transform=ax.transAxes)
            if not passed:
                failed.append(col)
        note = "all key gates pass"
        if row["lr_pair"] == "CXCL12-CXCR4":
            note = "spatial-null weak"
        if row["lr_pair"] == "JAG1-NOTCH2":
            note = "target/matched weaker"
        ax.text(0.98, y, note, ha="right", va="center", fontsize=7.2, color="#b45309" if failed else "#475569", transform=ax.transAxes)
    ax.text(
        0.02,
        0.035,
        "Shown checks are the most interpretable gates; full 9-layer evidence is in source data and supplementary figure.",
        fontsize=7.6,
        color="#64748b",
        transform=ax.transAxes,
    )


def draw_readable_panel_c(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "C", "Take-home evidence")
    left = ax.inset_axes([0.02, 0.12, 0.49, 0.76])
    right = ax.inset_axes([0.56, 0.12, 0.42, 0.76])

    plot = evidence.sort_values("support_count", ascending=True)
    y = np.arange(len(plot))
    colors = [BIOLOGY_COLORS.get(label, "#64748b") for label in plot["biology_label"]]
    left.barh(y, plot["support_count"].astype(float), color=colors, alpha=0.92)
    left.axvline(5, color="#334155", linestyle="--", lw=1.0)
    left.text(5.05, len(plot) - 0.25, "strong threshold", fontsize=7.0, color="#334155", va="top")
    left.set_yticks(y)
    left.set_yticklabels(plot["lr_pair"], fontsize=8)
    left.set_xlim(0, 9.5)
    left.set_xlabel("Supported evidence layers (of 9)", fontsize=8.4)
    left.set_title("All selected axes reach strong support", fontsize=9.2, weight="bold")
    left.grid(axis="x", color="#e2e8f0", linewidth=0.6)
    left.spines[["top", "right", "left"]].set_visible(False)
    left.tick_params(axis="y", length=0)

    right.set_axis_off()
    bullets = [
        ("Discovery", "Top axes remain biologically coherent after controls."),
        ("Specificity", "Cell-label permutation FDR is significant for all 7 axes."),
        ("Spatiality", "6/7 pass spatial-null; CXCL12-CXCR4 is retained by other evidence."),
        ("Safety", "No platelet/RBC contamination flag in the selected axes."),
        ("Caveat", "Computational support, not protein-level functional proof."),
    ]
    for i, (title, body) in enumerate(bullets):
        yy = 0.90 - i * 0.18
        color = "#0f766e" if title != "Caveat" else "#b45309"
        right.scatter([0.035], [yy], s=75, color=color, transform=right.transAxes)
        right.text(0.085, yy + 0.025, title, fontsize=8.7, weight="bold", color="#0f172a", transform=right.transAxes, va="center")
        right.text(0.085, yy - 0.045, wrap(body, 44), fontsize=7.4, color="#334155", transform=right.transAxes, va="center")


def make_readable_main_figure(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    evidence = data["evidence"]
    fig = plt.figure(figsize=(11.8, 9.2), constrained_layout=False)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.05, 2.35, 1.55], hspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    draw_readable_panel_a(ax_a)
    draw_readable_panel_b(ax_b, evidence)
    draw_readable_panel_c(ax_c, evidence)
    fig.suptitle(
        "Readable summary: TopoLink-CCI axes are supported by orthogonal computational controls",
        fontsize=14.5,
        weight="bold",
        y=0.99,
    )
    fig.savefig(output_dir / "topolink_cci_validation_main_figure_readable.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "topolink_cci_validation_main_figure_readable.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "topolink_cci_validation_main_figure_readable.svg", bbox_inches="tight")
    plt.close(fig)


def write_readable_caption(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    evidence = data["evidence"]
    caption = f"""# Readable figure caption

**Figure. Orthogonal computational controls support TopoLink-CCI discoveries.**
**A**, TopoLink-CCI scores are used to nominate candidate cell-cell interaction axes, which are then tested with independent control layers adapted from classic LR/CCC methods: permutation, spatial nulls, matched-gene controls, downstream/received-signal support, cross-method consensus, component ablation, and bootstrap stability.
**B**, Readable evidence table for seven biologically interpretable LR axes, ordered by pyXenium rank. The compact gates show label permutation, spatial null, matched-gene control, and downstream target support; full nine-layer support is included in source data. All seven axes are classified as strong. `CXCL12-CXCR4` is spatial-null weak but supported by other evidence layers; `JAG1-NOTCH2` has weaker matched-gene/downstream support but still meets the strong threshold.
**C**, Summary support counts and interpretive take-home points. The selected LR axes reach the pre-specified strong threshold of at least five independent computational evidence layers, and none are flagged for platelet/RBC contamination.

This figure summarizes computational credibility only; it does not prove protein-level binding, secretion, or functional signaling.
"""
    (output_dir / "figure_caption_readable.md").write_text(caption, encoding="utf-8")


def z_from_summary(observed: float, mean: float, sd: float) -> float:
    if not all(np.isfinite([observed, mean, sd])) or sd <= 0:
        return np.nan
    return (observed - mean) / sd


def draw_null_summary_page(data: dict[str, pd.DataFrame]) -> plt.Figure:
    evidence = data["evidence"]
    cell = data["cell_perm"].set_index("axis_id")
    spatial = data["spatial"].set_index("axis_id")
    controls = data["matched"]
    matched = controls[controls["control_type"] == "matched_gene_expression_control"].set_index("axis_id")
    fig, axes = plt.subplots(len(evidence), 3, figsize=(9.2, 10.5), sharex=True, sharey=True)
    x = np.linspace(-4, 4, 300)
    y = np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    for i, row in evidence.iterrows():
        axis_id = row["axis_id"]
        z_values = [
            z_from_summary(cell.loc[axis_id, "cell_label_comm_prob"], cell.loc[axis_id, "cell_label_perm_mean"], cell.loc[axis_id, "cell_label_perm_sd"]) if axis_id in cell.index else np.nan,
            z_from_summary(spatial.loc[axis_id, "spatial_lr_edge_score"], spatial.loc[axis_id, "spatial_perm_mean"], spatial.loc[axis_id, "spatial_perm_sd"]) if axis_id in spatial.index else np.nan,
            float(matched.loc[axis_id, "z_or_rank"]) if axis_id in matched.index and pd.notna(matched.loc[axis_id, "z_or_rank"]) else np.nan,
        ]
        for j, (title, zval) in enumerate(zip(["Label perm", "Spatial perm", "Matched genes"], z_values)):
            ax = axes[i, j]
            ax.plot(x, y, color="#94a3b8", lw=1.0)
            ax.fill_between(x, 0, y, color="#e2e8f0", alpha=0.8)
            clipped = np.clip(zval, -4, 4) if np.isfinite(zval) else np.nan
            if np.isfinite(clipped):
                ax.axvline(clipped, color="#dc2626", lw=1.4)
                label = f"z={zval:.1f}" if abs(zval) < 100 else "z>100"
                ax.text(0.97, 0.75, label, transform=ax.transAxes, ha="right", va="center", fontsize=6.3, color="#991b1b")
            if i == 0:
                ax.set_title(title, fontsize=8.2)
            if j == 0:
                ax.set_ylabel(row["lr_pair"], fontsize=7.8, rotation=0, ha="right", va="center", labelpad=42)
            ax.set_yticks([])
            ax.set_xticks([-4, 0, 4])
            ax.tick_params(labelsize=6.5, length=2)
            ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Supplementary Fig. S1: null summary distributions (observed red line; z-space)", fontsize=12, weight="bold", y=0.995)
    fig.text(0.5, 0.015, "Standard deviations from saved null summaries; raw permutation draws were not retained.", ha="center", fontsize=7.2, color="#475569")
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.97])
    return fig


def draw_validation_card_page(evidence: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 10.5))
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()
    for i, row in evidence.iterrows():
        ax = axes[i]
        color = BIOLOGY_COLORS.get(row["biology_label"], "#64748b")
        card = patches.FancyBboxPatch((0.02, 0.06), 0.96, 0.86, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#ffffff", edgecolor="#cbd5e1", transform=ax.transAxes)
        ax.add_patch(card)
        ax.add_patch(patches.Rectangle((0.02, 0.84), 0.96, 0.08, color=color, transform=ax.transAxes))
        ax.text(0.05, 0.88, row["lr_pair"], color="white", weight="bold", fontsize=10, transform=ax.transAxes, va="center")
        ax.text(0.05, 0.77, f"{row['sender']} -> {row['receiver']}", fontsize=7.7, color="#334155", transform=ax.transAxes)
        ax.text(0.05, 0.66, wrap(row["biology_label"], 32), fontsize=7.8, color="#0f172a", transform=ax.transAxes)
        ax.text(0.05, 0.50, f"rank {int(row['pyxenium_rank'])} | CCI_score {float(row['CCI_score']):.3f}", fontsize=8.0, color="#0f172a", transform=ax.transAxes)
        ax.text(0.05, 0.39, f"class: {row['evidence_class']} | support {int(row['support_count'])}/9", fontsize=8.0, color="#166534", transform=ax.transAxes, weight="bold")
        failed = [label.replace("\n", " ") for label, col in zip(SUPPORT_LABELS, SUPPORT_COLS) if not bool(row[col])]
        caveat = "No weak layer" if not failed else "Weak: " + ", ".join(failed)
        ax.text(0.05, 0.25, wrap(caveat, 42), fontsize=7.1, color="#b45309" if failed else "#475569", transform=ax.transAxes)
        ax.text(0.05, 0.12, "Computational support only.", fontsize=6.9, color="#64748b", transform=ax.transAxes)
    fig.suptitle("Supplementary Fig. S2: per-axis validation cards", fontsize=12, weight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_supplementary_figures(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    null_fig = draw_null_summary_page(data)
    card_fig = draw_validation_card_page(data["evidence"])
    with PdfPages(output_dir / "topolink_cci_validation_supplementary_controls.pdf") as pdf:
        pdf.savefig(null_fig, bbox_inches="tight")
        pdf.savefig(card_fig, bbox_inches="tight")
    null_fig.savefig(output_dir / "topolink_cci_validation_supplementary_null_summary.png", dpi=300, bbox_inches="tight")
    card_fig.savefig(output_dir / "topolink_cci_validation_supplementary_cards.png", dpi=300, bbox_inches="tight")
    plt.close(null_fig)
    plt.close(card_fig)


def write_caption(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    evidence = data["evidence"]
    top = evidence.iloc[0]
    caption = f"""# Figure caption

**Figure. TopoLink-CCI discoveries pass multi-layer computational false-positive controls.**
**A**, Schematic of the validation framework. pyXenium nominates candidate cell-cell interaction axes, which are then evaluated with validation principles adapted from CellPhoneDB/Squidpy, CellChat, stLearn/SpatialDM, NicheNet, LIANA, and pyXenium-specific robustness checks.
**B**, Evidence matrix across seven biologically interpretable LR axes. Filled circles indicate evidence layers that passed the pre-specified thresholds. All seven axes were classified as strong and none were flagged as contamination/artifact risk.
**C**, Quantitative strength of selected controls: cell-label permutation FDR, spatial-null FDR, matched-gene z-score, and downstream target FDR. The top-ranked axis, {top['lr_pair']}, had CCI_score={float(top['CCI_score']):.3f}, label-permutation FDR={float(top['cell_label_perm_fdr']):.3g}, spatial-null FDR={float(top['spatial_null_fdr']):.3g}, and matched-gene z={float(top['matched_gene_z']):.2f}.
**D**, Bootstrap and component-ablation robustness. Bootstrap ranks are medians from five 80% stratified resamples; ablation shows the worst rank after removing one pyXenium score component at a time.
**E**, Biological interpretation cards for the retained LR axes. These results support computational credibility, but do not prove protein-level cell-cell interaction binding, secretion, or functional causality.

Supplementary Fig. S1 visualizes saved null summaries in standardized z-space; raw permutation draws were not retained. Supplementary Fig. S2 shows per-axis validation cards and weak evidence layers.
"""
    (output_dir / "figure_caption.md").write_text(caption, encoding="utf-8")


def write_source_data(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    evidence = data["evidence"]
    cols = [
        "lr_pair",
        "sender",
        "receiver",
        "biology_label",
        "CCI_score",
        "pyxenium_rank",
        "evidence_class",
        "support_count",
        "cell_label_perm_fdr",
        "spatial_null_fdr",
        "matched_gene_z",
        "downstream_target_fdr",
        "bootstrap_rank_median",
        "bootstrap_rank_iqr",
        "max_rank_after_ablation",
        "contamination_flag",
        *SUPPORT_COLS,
    ]
    evidence[cols].to_csv(output_dir / "topolink_cci_validation_publication_source_data.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication figures for TopoLink-CCI validation v2.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="PDC clean validation v2 directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to INPUT_DIR/publication_figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "publication_figures"
    data = load_tables(input_dir)
    validate_inputs(data["evidence"])
    make_main_figure(data, output_dir)
    make_readable_main_figure(data, output_dir)
    make_supplementary_figures(data, output_dir)
    write_caption(data, output_dir)
    write_readable_caption(data, output_dir)
    write_source_data(data, output_dir)
    expected = [
        "topolink_cci_validation_main_figure.png",
        "topolink_cci_validation_main_figure.pdf",
        "topolink_cci_validation_main_figure.svg",
        "topolink_cci_validation_main_figure_readable.png",
        "topolink_cci_validation_main_figure_readable.pdf",
        "topolink_cci_validation_main_figure_readable.svg",
        "topolink_cci_validation_supplementary_controls.pdf",
        "figure_caption.md",
        "figure_caption_readable.md",
    ]
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected figure outputs: {missing}")
    print(f"Wrote publication figures to {output_dir}")


if __name__ == "__main__":
    main()
