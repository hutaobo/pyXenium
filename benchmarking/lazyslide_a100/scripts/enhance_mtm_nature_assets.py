"""Generate Nature-style enhanced assets for the mTM WTA submission package."""

from __future__ import annotations

import hashlib
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


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
SOURCE_TABLE_DIR = FINAL_DIR / "Source_Tables"
OUT_DIR = FINAL_DIR / "Nature_Enhanced_Assets"
NATURE_SKILL_DIR = Path.home() / ".codex" / "skills"
NATURE_SKILLS_COMMIT = "2549f85aa928aa35edb757ca6b0dc66b401fa55b"
NATURE_SKILLS_REPO = "https://github.com/Yuan1z0825/nature-skills.git"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "red_soft": "#E9A6A1",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

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
SHORT_PATHWAY_LABELS = {
    "luminal_estrogen_response": "Luminal ER",
    "unfolded_protein_response": "UPR",
    "oxidative_phosphorylation": "OXPHOS",
    "collagen_ecm_organization": "Collagen ECM",
    "tls_b_cell_plasma": "TLS B/plasma",
    "t_cell_exhaustion_checkpoint": "T-cell checkpoint",
    "not_detected": "not detected",
    "emt_invasion": "EMT",
    "epithelial_identity": "Epithelial",
    "myofibroblast_caf_activation": "myCAF",
    "stromal_encapsulation": "Stromal cap",
    "tls_adjacent_activation": "TLS adjacent",
    "immune_exclusion": "Immune exclusion",
    "emt_invasive_front": "Invasive front",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def save_pub_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ]:
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    return outputs


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.06,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["neutral_black"],
    )


def support_alpha(call: str) -> float:
    return {
        "strong": 0.95,
        "moderate": 0.78,
        "weak": 0.48,
        "not_detected": 0.18,
    }.get(str(call), 0.55)


def generate_figure4() -> list[Path]:
    apply_nature_style()
    table = pd.read_csv(SOURCE_TABLE_DIR / "CrossCancer_Morphological_Signature_Table.csv")
    row_order = [("breast", "PLIP"), ("breast", "UNI"), ("cervical", "PLIP"), ("cervical", "UNI")]
    row_labels = ["Breast\nPLIP", "Breast\nUNI", "Cervical\nPLIP", "Cervical\nUNI"]
    dataset_colors = {"breast": PALETTE["blue_secondary"], "cervical": PALETTE["red_soft"]}
    x_spacing = 0.88
    y_spacing = 0.70
    x_values = np.arange(len(PROGRAM_FAMILY_ORDER)) * x_spacing
    y_values = np.arange(len(row_order)) * y_spacing

    fig, ax = plt.subplots(figsize=(5.35, 2.28))
    ax.set_xlim(-0.50, x_values[-1] + 0.50)
    ax.set_ylim(y_values[-1] + 0.50, -0.48)
    ax.invert_yaxis()
    ax.set_xticks(x_values)
    ax.set_xticklabels([PROGRAM_FAMILY_LABELS[item] for item in PROGRAM_FAMILY_ORDER], fontsize=6.4)
    ax.set_yticks(y_values)
    ax.set_yticklabels(row_labels, fontsize=6.8)
    ax.tick_params(axis="both", length=0)
    ax.spines[["left", "bottom"]].set_visible(False)

    for x in x_values:
        ax.axvline(x, color="#F0F1F3", linewidth=0.7, zorder=0)
    for y in y_values:
        ax.axhline(y, color="#F0F1F3", linewidth=0.7, zorder=0)

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
            call = str(record["support_call"])
            top_pathway = str(record["top_pathway"])
            if pd.isna(rho):
                ax.scatter(
                    x,
                    y,
                    s=32,
                    marker="x",
                    color=PALETTE["neutral_mid"],
                    linewidth=0.9,
                    zorder=4,
                )
                ax.text(
                    x,
                    y + 0.24,
                    "not detected",
                    ha="center",
                    va="top",
                    fontsize=4.9,
                    color=PALETTE["neutral_mid"],
                )
                continue

            size = 45 + 420 * float(rho)
            ax.scatter(
                x,
                y,
                s=size,
                color=dataset_colors[dataset],
                alpha=support_alpha(call),
                edgecolor=PALETTE["neutral_black"],
                linewidth=0.45,
                zorder=5,
            )
            ax.text(
                x,
                y,
                f"{rho:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color=PALETTE["neutral_black"],
                zorder=6,
            )
            ax.text(
                x,
                y + 0.24,
                SHORT_PATHWAY_LABELS.get(top_pathway, top_pathway.replace("_", " ")),
                ha="center",
                va="top",
                fontsize=4.6,
                color=PALETTE["neutral_dark"],
                zorder=6,
            )

    for tick, (dataset, _model) in zip(ax.get_yticklabels(), row_order, strict=True):
        tick.set_color(dataset_colors[dataset])
        tick.set_fontweight("bold")
    fig.tight_layout(pad=0.12)
    outputs = save_pub_figure(fig, "Figure_4_CrossCancer_Morphological_Signature_Table_Nature")
    plt.close(fig)
    return outputs


def read_patch(path_string: str):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(path)


def read_patch_square(path_string: str, size: int = 1024) -> Image.Image:
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def generate_figure2() -> list[Path]:
    apply_nature_style()
    hero = pd.read_csv(SOURCE_TABLE_DIR / "Figure2_Luminal_ER_HeroPatch_4Pairs.csv")
    fig_width, fig_height = 4.55, 6.28
    fig = plt.figure(figsize=(fig_width, fig_height))
    high_color = PALETTE["red_strong"]
    low_color = PALETTE["blue_main"]
    pair_labels = ["A", "B", "C", "D"]
    square_in = 1.32
    image_width = square_in / fig_width
    image_height = square_in / fig_height
    left_x = 0.115
    right_x = 0.468
    top_y = 0.724
    row_step = 0.224
    label_x = 0.025

    fig.text(
        left_x + image_width / 2,
        0.968,
        "High luminal ER",
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=high_color,
    )
    fig.text(
        right_x + image_width / 2,
        0.968,
        "Low luminal ER",
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=low_color,
    )

    for pair in sorted(hero["pair"].unique()):
        pair_rows = hero[hero["pair"] == pair]
        high = pair_rows[pair_rows["polarity"] == "high"].iloc[0]
        low = pair_rows[pair_rows["polarity"] == "low"].iloc[0]
        row_idx = int(pair) - 1
        y0 = top_y - row_idx * row_step

        for col_idx, record, color in [
            (0, high, high_color),
            (1, low, low_color),
        ]:
            x0 = left_x if col_idx == 0 else right_x
            ax = fig.add_axes([x0, y0, image_width, image_height])
            ax.imshow(read_patch_square(record["patch_file"]))
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color(color)
            if col_idx == 0:
                fig.text(
                    label_x,
                    y0 + image_height - 0.008,
                    pair_labels[row_idx],
                    fontsize=8.2,
                    fontweight="bold",
                    ha="left",
                    va="top",
                    color=PALETTE["neutral_black"],
                )
                fig.text(
                    label_x,
                    y0 + image_height - 0.055,
                    f"Pair {pair}",
                    fontsize=5.4,
                    ha="left",
                    va="top",
                    color=PALETTE["neutral_dark"],
                )
            ax.text(
                0.02,
                0.02,
                (
                    f"{record['contour_id']}\n"
                    f"WTA {record['wta_program_z']:+.2f} | "
                    f"H&E {record['oriented_he_embedding_z']:+.2f} | "
                    f"n={int(record['n_cells'])}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=4.2,
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "linewidth": 0},
            )

    outputs = save_pub_figure(fig, "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature")
    plt.close(fig)
    return outputs


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.strip().lower()).strip("-")
    return slug or "section"


def build_toc(markdown: str) -> list[str]:
    toc = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        indent = "  " * max(level - 1, 0)
        toc.append(f"{indent}- [{title}](#{slugify(title)})")
    return toc


def add_heading_anchors(markdown: str) -> str:
    output = []
    seen: dict[str, int] = {}
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if match:
            title = match.group(2).strip()
            slug = slugify(title)
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            if count:
                slug = f"{slug}-{count + 1}"
            output.append(f'<a id="{slug}"></a>')
        output.append(line)
    return "\n".join(output)


def make_reader() -> Path:
    source = FINAL_DIR / "Main_Text.md"
    manuscript = source.read_text(encoding="utf-8")
    toc = "\n".join(build_toc(manuscript))
    anchored = add_heading_anchors(manuscript)
    links = re.findall(r"https?://[^\s)]+", manuscript)
    provided_references = [
        ("STPath", "https://www.nature.com/articles/s41746-025-02020-3"),
        ("10x Atera", "https://www.10xgenomics.com/platforms/atera"),
        ("spEMO", "https://www.nature.com/articles/s41551-025-01602-6.pdf"),
        ("PAST", "https://huggingface.co/papers/2507.06418"),
    ]
    figure_links = [
        ("Figure 2 Nature SVG", "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.svg"),
        ("Figure 4 Nature SVG", "Figure_4_CrossCancer_Morphological_Signature_Table_Nature.svg"),
        ("Original Figure 1 PNG", "../Figure_1_mTM_Framework.png"),
        ("Original Figure 3 PNG", "../Figure_3_MAZ_Stable_Coupled_Profiles.png"),
    ]
    lines = [
        "# Full Markdown Reader: mTM WTA NBT Submission",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        "This reader reorganizes the final `Main_Text.md` for mobile review. It does not add new claims, new statistics or unverified references.",
        "",
        "## Table of Contents",
        "",
        toc,
        "",
        "## Figure Callouts",
        "",
    ]
    lines.extend([f"- [{label}]({target})" for label, target in figure_links])
    lines.extend(
        [
            "",
            "## Reference Link Audit",
            "",
            "Reference anchors currently named in the manuscript or project prompt:",
            "",
        ]
    )
    lines.extend([f"- [{label}]({url})" for label, url in provided_references])
    lines.extend(
        [
            "",
            "Direct URLs already present in `Main_Text.md`:",
            "",
            *(f"- {url}" for url in links),
        ]
    )
    if not links:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Main Text with Anchors",
            "",
            anchored,
            "",
        ]
    )
    output = OUT_DIR / "Main_Text_Full_Markdown_Reader.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def svg_has_text(path: Path) -> bool:
    return "<text" in path.read_text(encoding="utf-8", errors="ignore")


def write_report(outputs: list[Path]) -> Path:
    figure_skill = NATURE_SKILL_DIR / "nature-figure"
    reader_skill = NATURE_SKILL_DIR / "nature-reader"
    svg_checks = {path.name: svg_has_text(path) for path in outputs if path.suffix == ".svg"}
    lines = [
        "# Nature Skills Integration Report",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        "## Installed skills",
        "",
        f"- Repository: `{NATURE_SKILLS_REPO}`",
        f"- Recorded HEAD commit: `{NATURE_SKILLS_COMMIT}`",
        f"- `nature-figure`: `{figure_skill}` ({'present' if figure_skill.exists() else 'missing'})",
        f"- `nature-reader`: `{reader_skill}` ({'present' if reader_skill.exists() else 'missing'})",
        "- Restart Codex to expose newly installed skills in the session skill registry.",
        "",
        "## Figure contract",
        "",
        "- Backend: Python/matplotlib only.",
        "- Figure 4: quantitative grid; claim is program-family recovery across cancer types and models.",
        "- Figure 2: image plate plus annotation; claim is within-S3 morphomolecular divergence, not visual indistinguishability.",
        "- Export contract: SVG primary, PDF and PNG secondary.",
        "",
        "## SVG editable text checks",
        "",
    ]
    lines.extend([f"- `{name}`: {'pass' if passed else 'fail'}" for name, passed in svg_checks.items()])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            *(f"- `{path.name}`" for path in outputs),
        ]
    )
    report = OUT_DIR / "Nature_Skills_Integration_Report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_manifest(outputs: list[Path]) -> Path:
    records = []
    for path in sorted(outputs):
        records.append(
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "generated_utc": utc_now(),
            }
        )
    manifest = OUT_DIR / "Nature_Enhanced_Assets_Manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def verify_numeric_consistency() -> None:
    hero = pd.read_csv(SOURCE_TABLE_DIR / "Figure2_Luminal_ER_HeroPatch_4Pairs.csv")
    signature = pd.read_csv(SOURCE_TABLE_DIR / "CrossCancer_Morphological_Signature_Table.csv")
    if hero["wta_program_z"].isna().any():
        raise ValueError("Figure 2 source table contains missing WTA z-scores.")
    required_columns = {"max_abs_partial_rho", "top_pathway", "support_call"}
    missing = required_columns.difference(signature.columns)
    if missing:
        raise ValueError(f"Figure 4 source table missing columns: {sorted(missing)}")


def wrap_for_log(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=96))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify_numeric_consistency()
    outputs: list[Path] = []
    outputs.extend(generate_figure4())
    outputs.extend(generate_figure2())
    outputs.append(make_reader())
    report = write_report(outputs)
    outputs.append(report)
    manifest = write_manifest(outputs)
    outputs.append(manifest)
    print(
        wrap_for_log(
            f"Generated {len(outputs)} Nature-enhanced assets in {OUT_DIR}. "
            "Original final-submission files were left untouched."
        )
    )


if __name__ == "__main__":
    main()
