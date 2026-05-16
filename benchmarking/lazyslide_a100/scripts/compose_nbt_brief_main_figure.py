"""Compose the composite NBT Brief Communication main figure."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "tutorials"
    / "multimodal_histoseg_lazyslide_breast_wta"
    / "naturebiotech_package"
)
FINAL_DIR = PACKAGE_ROOT / "FINAL_SUBMISSION_NBT_20260513"
SOURCE_DIR = FINAL_DIR / "Source_Tables"
OUT_DIR = FINAL_DIR / "Nature_Enhanced_Assets"

PROGRAM_FAMILY_ORDER = [
    "endocrine/epithelial identity",
    "metabolic/stress",
    "stromal-remodeling/CAF/ECM",
    "immune ecology/TLS/immune exclusion",
    "invasion/boundary/EMT",
]
PROGRAM_FAMILY_LABELS = {
    "endocrine/epithelial identity": "Endocrine /\nepithelial",
    "metabolic/stress": "Metabolic /\nstress",
    "stromal-remodeling/CAF/ECM": "Stromal /\nCAF-ECM",
    "immune ecology/TLS/immune exclusion": "Immune /\nTLS-exclusion",
    "invasion/boundary/EMT": "Invasion /\nEMT",
}
SHORT_PROGRAM_LABELS = {
    "luminal_estrogen_response": "Luminal ER",
    "unfolded_protein_response": "UPR",
    "oxidative_phosphorylation": "OXPHOS",
    "collagen_ecm_organization": "Collagen ECM",
    "tls_b_cell_plasma": "TLS B/plasma",
    "t_cell_exhaustion_checkpoint": "T-cell checkpoint",
    "myofibroblast_caf_activation": "myCAF",
    "tls_adjacent_activation": "TLS adjacent",
    "emt_invasive_front": "Invasive front",
    "epithelial_identity": "Epithelial",
    "stromal_encapsulation": "Stromal cap",
    "immune_exclusion": "Immune exclusion",
}
PALETTE = {
    "black": "#272727",
    "gray": "#6F6F6F",
    "light_grid": "#ECEFF3",
    "blue": "#2F70B7",
    "blue_light": "#A9C8E8",
    "red": "#C1464A",
    "red_light": "#E9AAA5",
    "green": "#56966E",
    "yellow": "#F3DFA2",
    "purple": "#A68ACB",
    "teal": "#8EC7CC",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.3,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def panel_label(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(x, y, label, fontsize=8.5, fontweight="bold", ha="left", va="top", color=PALETTE["black"])


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    color: str,
    w: float = 0.18,
    h: float = 0.58,
    fontsize: float = 6.1,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        fc=color,
        ec=PALETTE["black"],
        lw=0.85,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def draw_framework(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = [
        (0.72, "Atera WTA\ncells + clusters", "#F7ECC9", "HistoSeg\ncontours", "#D8E7F7"),
        (0.40, "H&E WSI", "#F4C2CD", "LazySlide\nPLIP / UNI", "#F4D7DE"),
        (0.08, "Atera WTA", "#F7ECC9", "Program\nscores", "#F1E3A8"),
    ]
    for y, source_label, source_color, derived_label, derived_color in rows:
        add_box(ax, (0.02, y), source_label, source_color, w=0.22, h=0.18, fontsize=4.9)
        add_box(ax, (0.31, y), derived_label, derived_color, w=0.18, h=0.18, fontsize=4.9)
        ax.add_patch(
            FancyArrowPatch(
                (0.24, y + 0.09),
                (0.31, y + 0.09),
                arrowstyle="->",
                mutation_scale=8,
                lw=0.75,
                color=PALETTE["black"],
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (0.49, y + 0.09),
                (0.62, 0.50),
                arrowstyle="->",
                mutation_scale=8,
                lw=0.75,
                color=PALETTE["black"],
                connectionstyle="arc3,rad=0.0",
            )
        )
    add_box(ax, (0.62, 0.36), "Contour-level\ntable", "#DCEDE1", w=0.17, h=0.28, fontsize=5.1)
    add_box(ax, (0.84, 0.36), "Residual\ndecoding", "#E7D8F4", w=0.14, h=0.28, fontsize=5.1)
    ax.add_patch(
        FancyArrowPatch(
            (0.79, 0.50),
            (0.84, 0.50),
            arrowstyle="->",
            mutation_scale=9,
            lw=0.85,
            color=PALETTE["black"],
        )
    )


def read_patch_square(path_string: str, size: int = 850) -> Image.Image:
    image = Image.open(Path(path_string)).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def draw_patch_plate(fig: plt.Figure) -> None:
    hero = pd.read_csv(SOURCE_DIR / "Figure2_Luminal_ER_HeroPatch_4Pairs.csv")
    left_x = 0.080
    right_x = 0.252
    top_y = 0.620
    w = 0.142
    h = 0.142
    row_step = 0.188
    high_color = PALETTE["red"]
    low_color = "#0D559B"

    fig.text(left_x + w / 2, 0.788, "High luminal ER", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=high_color)
    fig.text(right_x + w / 2, 0.788, "Low luminal ER", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=low_color)

    for pair in [1, 2]:
        rows = hero[hero["pair"] == pair]
        high = rows[rows["polarity"] == "high"].iloc[0]
        low = rows[rows["polarity"] == "low"].iloc[0]
        y = top_y - (pair - 1) * row_step
        for record, x, color in [(high, left_x, high_color), (low, right_x, low_color)]:
            ax = fig.add_axes([x, y, w, h])
            ax.imshow(read_patch_square(record["patch_file"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(0.9)
            ax.text(
                0.02,
                0.02,
                f"WTA {record['wta_program_z']:+.1f}\nH&E {record['oriented_he_embedding_z']:+.1f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=3.8,
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.48, "linewidth": 0},
            )
        fig.text(0.035, y + h * 0.56, f"Pair {pair}", fontsize=5.3, ha="left", va="center", color=PALETTE["gray"])


def draw_spatial_defense(ax: plt.Axes) -> None:
    perm = pd.read_csv(SOURCE_DIR / "Spatial_Permutation_Defense_Report.csv")
    keep_programs = [
        "luminal_estrogen_response",
        "unfolded_protein_response",
        "oxidative_phosphorylation",
        "myofibroblast_caf_activation",
        "emt_invasive_front",
        "immune_exclusion",
    ]
    perm = perm[perm["program"].isin(keep_programs)].copy()
    perm["label"] = perm["dataset"].str.title() + ": " + perm["program"].map(
        lambda value: SHORT_PROGRAM_LABELS.get(value, value.replace("_", " "))
    )
    perm["abs_rho"] = perm["recomputed_partial_rho"].abs()
    perm["null99"] = perm["null_abs_rho_q99"]
    perm = perm.sort_values("abs_rho")
    y = np.arange(len(perm))
    colors = [PALETTE["blue"] if dataset == "breast" else PALETTE["red_light"] for dataset in perm["dataset"]]
    ax.hlines(y, perm["null99"], perm["abs_rho"], color=PALETTE["gray"], lw=0.8, zorder=1)
    ax.scatter(perm["null99"], y, marker="|", s=85, color=PALETTE["gray"], lw=1.4, zorder=2)
    ax.scatter(perm["abs_rho"], y, s=42, color=colors, edgecolor=PALETTE["black"], lw=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(perm["label"], fontsize=4.9)
    ax.set_xlabel("|partial rho| vs 99% spatial-null threshold", fontsize=5.4)
    ax.set_xlim(0.15, 0.72)
    ax.set_ylim(-0.45, len(perm) - 0.30)
    ax.grid(axis="x", color=PALETTE["light_grid"], lw=0.6)
    ax.tick_params(axis="both", labelsize=4.8, length=2)
    ax.text(0.995, 0.04, "10,000 permutations", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.7, color=PALETTE["gray"])


def dot_alpha(call: str) -> float:
    return {"strong": 0.96, "moderate": 0.72, "weak": 0.42, "not_detected": 0.20}.get(str(call), 0.55)


def draw_candidate_signature_summary(ax: plt.Axes) -> None:
    table = pd.read_csv(SOURCE_DIR / "CrossCancer_Morphological_Signature_Table.csv")
    row_order = [("breast", "PLIP"), ("breast", "UNI"), ("cervical", "PLIP"), ("cervical", "UNI")]
    row_labels = ["Breast PLIP", "Breast UNI", "Cervical PLIP", "Cervical UNI"]
    dataset_colors = {"breast": PALETTE["blue"], "cervical": PALETTE["red_light"]}
    x_spacing = 0.88
    y_spacing = 0.70
    x_values = np.arange(len(PROGRAM_FAMILY_ORDER)) * x_spacing
    y_values = np.arange(len(row_order)) * y_spacing

    ax.set_xlim(-0.52, x_values[-1] + 0.52)
    ax.set_ylim(y_values[-1] + 0.48, -0.45)
    for x in x_values:
        ax.axvline(x, color=PALETTE["light_grid"], lw=0.55, zorder=0)
    for y in y_values:
        ax.axhline(y, color=PALETTE["light_grid"], lw=0.55, zorder=0)

    for y_idx, (dataset, model) in enumerate(row_order):
        for x_idx, family in enumerate(PROGRAM_FAMILY_ORDER):
            row = table[
                (table["dataset"] == dataset)
                & (table["model"] == model)
                & (table["program_family"] == family)
            ]
            if row.empty:
                continue
            record = row.iloc[0]
            x = x_values[x_idx]
            y = y_values[y_idx]
            rho = record["max_abs_partial_rho"]
            if pd.isna(rho):
                ax.scatter(x, y, marker="x", s=42, color=PALETTE["gray"], lw=1.0)
                ax.text(x, y + 0.23, "not detected", ha="center", va="top", fontsize=4.3, color=PALETTE["gray"])
                continue
            ax.scatter(
                x,
                y,
                s=90 + 360 * float(rho),
                color=dataset_colors[dataset],
                alpha=dot_alpha(record["support_call"]),
                edgecolor=PALETTE["black"],
                lw=0.45,
                zorder=4,
            )
            ax.text(x, y, f"{rho:.2f}", ha="center", va="center", fontsize=5.1, fontweight="bold", color=PALETTE["black"], zorder=5)
            label = SHORT_PROGRAM_LABELS.get(str(record["top_pathway"]), str(record["top_pathway"]).replace("_", " "))
            ax.text(x, y + 0.245, label, ha="center", va="top", fontsize=3.95, color=PALETTE["gray"], zorder=5)

    ax.set_xticks(x_values)
    ax.set_xticklabels([PROGRAM_FAMILY_LABELS[item] for item in PROGRAM_FAMILY_ORDER], fontsize=5.1)
    ax.set_yticks(y_values)
    ax.set_yticklabels(row_labels, fontsize=5.2)
    for tick, (dataset, _model) in zip(ax.get_yticklabels(), row_order, strict=True):
        tick.set_color(dataset_colors[dataset])
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        1.0,
        1.02,
        "Candidate program-family signatures; dot area = max |partial rho|",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.5,
        color=PALETTE["gray"],
    )


def draw_maz(ax: plt.Axes) -> None:
    maz = pd.read_csv(SOURCE_DIR / "MAZ_QC_Table_v2.csv")
    focus = [
        "emt_invasive_front",
        "myofibroblast_caf_activation",
        "collagen_ecm_organization",
        "immune_exclusion",
        "oxidative_phosphorylation",
    ]
    maz = maz[(maz["program"].isin(focus)) & (maz["passes_ring_qc"])].copy()
    maz["abs_ring"] = maz["ring_profile_spearman_rho"].abs()
    maz = maz.sort_values(["dataset", "abs_ring"], ascending=[True, False]).drop_duplicates(["dataset", "program"]).head(8)
    maz = maz.sort_values("ring_profile_spearman_rho")
    y = np.arange(len(maz))
    colors = [PALETTE["blue"] if dataset == "breast" else PALETTE["red_light"] for dataset in maz["dataset"]]
    ax.axvline(0, color=PALETTE["gray"], lw=0.65)
    ax.hlines(y, 0, maz["ring_profile_spearman_rho"], color=PALETTE["gray"], lw=0.75)
    ax.scatter(maz["ring_profile_spearman_rho"], y, s=38, color=colors, edgecolor=PALETTE["black"], lw=0.4, zorder=3)
    labels = [dataset.title() + ": " + SHORT_PROGRAM_LABELS.get(program, program.replace("_", " ")) for dataset, program in zip(maz["dataset"], maz["program"], strict=True)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=4.7)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Ring-level H&E-WTA coupling", fontsize=5.4)
    ax.grid(axis="x", color=PALETTE["light_grid"], lw=0.55)
    ax.tick_params(axis="both", labelsize=4.8, length=2)


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    outputs = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ]:
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        outputs.append(path)
    return outputs


def compose_main_figure() -> list[Path]:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.20, 7.35))

    panel_label(fig, 0.015, 0.982, "a")
    ax_a = fig.add_axes([0.055, 0.825, 0.895, 0.120])
    draw_framework(ax_a)

    panel_label(fig, 0.015, 0.800, "b")
    draw_patch_plate(fig)

    panel_label(fig, 0.405, 0.812, "c")
    ax_c = fig.add_axes([0.535, 0.620, 0.405, 0.175])
    draw_spatial_defense(ax_c)

    panel_label(fig, 0.445, 0.560, "d")
    ax_e = fig.add_axes([0.552, 0.390, 0.388, 0.145])
    draw_maz(ax_e)

    panel_label(fig, 0.015, 0.330, "e")
    ax_d = fig.add_axes([0.125, 0.066, 0.790, 0.215])
    draw_candidate_signature_summary(ax_d)

    outputs = save_figure(fig, OUT_DIR / "Figure_1_mTM_NBT_Brief_Composite")
    plt.close(fig)
    return outputs


def update_single_figure_upload(outputs: list[Path]) -> None:
    upload_dir = PACKAGE_ROOT / "NBT_INITIAL_SUBMISSION_UPLOAD_20260515"
    figure_dir = upload_dir / "Figures"
    if not figure_dir.exists():
        return
    by_suffix = {path.suffix: path for path in outputs}
    for suffix in [".svg", ".pdf", ".png"]:
        src = by_suffix[suffix]
        shutil.copy2(src, figure_dir / f"Figure_1{suffix}")


def main() -> None:
    outputs = compose_main_figure()
    update_single_figure_upload(outputs)
    print("Generated composite main figure:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
