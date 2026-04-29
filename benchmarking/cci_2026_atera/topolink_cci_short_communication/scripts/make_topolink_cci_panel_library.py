"""Generate Illustrator-friendly standalone panels for TopoLink-CCI.

This script is intentionally stricter than an exploratory plotting script:

- PDF text is exported as editable TrueType/Type 42 text when supported.
- SVG text is kept as <text> rather than converted to paths.
- Every panel is saved through PanelExporter, which requires source data.
- Dense spatial layers are rasterized while axes and labels remain vector.
- A panel manifest records outputs, source hashes, style policy and checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patches


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "pdc_validation_v2_collected" / "topolink_cci_validation_v2"
DEEP_DIVE = ROOT / "vwf_selp_deep_dive"
PDC_PYXENIUM = ROOT / "pdc_collected" / "pdc_20260426_1327" / "runs" / "full_common" / "pyxenium"
OUT_ROOT = ROOT / "topolink_cci_short_communication" / "figures" / "panels"


COLORS = {
    "vascular": "#0F766E",
    "stromal": "#8A5A2B",
    "immune": "#3F7F3F",
    "notch": "#5B5F97",
    "tumor": "#7A7A7A",
    "pass": "#138A72",
    "weak": "#E69F00",
    "risk": "#B23A48",
    "blue": "#2563EB",
    "gray": "#6B7280",
    "light_gray": "#F3F4F6",
    "text": "#1F2933",
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

SIZE_MM = {
    "single": (85, 65),
    "wide": (180, 65),
    "tall": (85, 110),
    "square": (85, 85),
}

STYLE_POINTS = {
    "panel_label": 8,
    "title": 7,
    "axis_label": 6,
    "tick_label": 5,
    "legend": 5,
    "annotation": 5,
    "body": 6,
}


def mm_to_inch(value: float) -> float:
    return value / 25.4


def file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ensure_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "pass"})


@dataclass(frozen=True)
class FigureStyle:
    font_family: str
    pdf_fonttype: int = 42
    ps_fonttype: int = 42
    svg_fonttype: str = "none"

    @staticmethod
    def choose_font() -> str:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        return "Arial" if "Arial" in installed else "DejaVu Sans"

    @classmethod
    def apply(cls) -> "FigureStyle":
        font = cls.choose_font()
        plt.rcParams.update(
            {
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "font.family": font,
                "font.sans-serif": [font, "Arial", "DejaVu Sans"],
                "font.size": STYLE_POINTS["body"],
                "axes.titlesize": STYLE_POINTS["title"],
                "axes.labelsize": STYLE_POINTS["axis_label"],
                "xtick.labelsize": STYLE_POINTS["tick_label"],
                "ytick.labelsize": STYLE_POINTS["tick_label"],
                "legend.fontsize": STYLE_POINTS["legend"],
                "axes.linewidth": 0.45,
                "lines.linewidth": 0.75,
                "grid.linewidth": 0.35,
                "patch.linewidth": 0.65,
                "savefig.dpi": 300,
                "figure.dpi": 300,
                "text.usetex": False,
            }
        )
        return cls(font_family=font)

    def as_dict(self) -> dict[str, object]:
        return {
            "font_family": self.font_family,
            "pdf.fonttype": self.pdf_fonttype,
            "ps.fonttype": self.ps_fonttype,
            "svg.fonttype": self.svg_fonttype,
            "panel_label_pt": STYLE_POINTS["panel_label"],
            "title_pt": STYLE_POINTS["title"],
            "axis_label_pt": STYLE_POINTS["axis_label"],
            "tick_label_pt": STYLE_POINTS["tick_label"],
            "legend_pt": STYLE_POINTS["legend"],
        }


class PanelExporter:
    def __init__(self, output_root: Path, style: FigureStyle):
        self.output_root = output_root
        self.style = style
        self.manifest_rows: list[dict[str, object]] = []
        for sub in ["pdf", "svg", "png", "source_data", "metadata"]:
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)

    def save_panel(
        self,
        fig: plt.Figure,
        panel_id: str,
        slug: str,
        source_data_df: pd.DataFrame,
        metadata: dict[str, object],
    ) -> None:
        if not isinstance(source_data_df, pd.DataFrame):
            raise TypeError(f"{panel_id}_{slug} source_data_df must be a pandas DataFrame")
        if source_data_df.empty:
            raise ValueError(f"{panel_id}_{slug} source_data_df is empty")

        stem = f"{panel_id}_{slug}"
        pdf_path = self.output_root / "pdf" / f"{stem}.pdf"
        svg_path = self.output_root / "svg" / f"{stem}.svg"
        png_path = self.output_root / "png" / f"{stem}.png"
        source_path = self.output_root / "source_data" / f"{stem}.tsv"
        metadata_path = self.output_root / "metadata" / f"{stem}.json"

        source_data_df.to_csv(source_path, sep="\t", index=False)
        fig.savefig(pdf_path, dpi=300)
        fig.savefig(svg_path, dpi=300)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

        svg_text_count = 0
        svg_path_count = 0
        if svg_path.exists():
            text = svg_path.read_text(encoding="utf-8", errors="ignore")
            svg_text_count = text.count("<text")
            svg_path_count = text.count("<path")

        pdffonts_status = "pdffonts_not_available"
        pdffonts_output = ""
        pdffonts = shutil.which("pdffonts")
        if pdffonts:
            try:
                result = subprocess.run(
                    [pdffonts, str(pdf_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=20,
                )
                pdffonts_status = f"exit_{result.returncode}"
                pdffonts_output = (result.stdout + "\n" + result.stderr).strip()[:4000]
            except Exception as exc:  # pragma: no cover - diagnostic only
                pdffonts_status = f"pdffonts_error:{exc}"

        width_in, height_in = fig.get_size_inches()
        source_tables = [Path(p) for p in metadata.get("source_tables", [])]
        source_hashes = {str(path): file_hash(path) for path in source_tables}
        rasterized = bool(metadata.get("rasterized_layers", False))
        complete_metadata = {
            **metadata,
            "panel_id": panel_id,
            "slug": slug,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "outputs": {
                "pdf": str(pdf_path),
                "svg": str(svg_path),
                "png": str(png_path),
                "source_data": str(source_path),
                "metadata": str(metadata_path),
            },
            "width_mm": round(width_in * 25.4, 3),
            "height_mm": round(height_in * 25.4, 3),
            "font_policy": self.style.as_dict(),
            "svg_text_count": svg_text_count,
            "svg_path_count": svg_path_count,
            "pdffonts_status": pdffonts_status,
            "pdffonts_output": pdffonts_output,
            "source_hashes": source_hashes,
            "source_data_rows": int(len(source_data_df)),
        }
        metadata_path.write_text(json.dumps(complete_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        self.manifest_rows.append(
            {
                "panel_id": panel_id,
                "slug": slug,
                "title": metadata.get("title", ""),
                "pdf": str(pdf_path),
                "svg": str(svg_path),
                "png": str(png_path),
                "source_data": str(source_path),
                "metadata": str(metadata_path),
                "width_mm": complete_metadata["width_mm"],
                "height_mm": complete_metadata["height_mm"],
                "font_family": self.style.font_family,
                "pdf_fonttype": self.style.pdf_fonttype,
                "svg_fonttype": self.style.svg_fonttype,
                "svg_text_count": svg_text_count,
                "svg_path_count": svg_path_count,
                "pdffonts_status": pdffonts_status,
                "rasterized_layers": rasterized,
                "source_data_rows": int(len(source_data_df)),
                "source_hashes": json.dumps(source_hashes, sort_keys=True),
                "biological_message": metadata.get("biological_message", ""),
                "caveat": metadata.get("caveat", ""),
            }
        )

    def write_manifest(self) -> Path:
        if not self.manifest_rows:
            raise RuntimeError("No panels were saved")
        manifest = pd.DataFrame(self.manifest_rows)
        path = self.output_root / "panel_manifest.tsv"
        manifest.to_csv(path, sep="\t", index=False)
        return path


def make_figure(size_key: str) -> tuple[plt.Figure, plt.Axes]:
    width_mm, height_mm = SIZE_MM[size_key]
    fig, ax = plt.subplots(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)), layout="constrained")
    return fig, ax


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.30,
        label,
        transform=ax.transAxes,
        fontsize=STYLE_POINTS["panel_label"],
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, width=0.45)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.35)


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    fontsize: float = 5.5,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        fc=fc,
        ec=ec,
        lw=0.7,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["text"])


def load_evidence() -> pd.DataFrame:
    path = VALIDATION / "tables" / "topolink_cci_validation_v2_evidence.tsv"
    df = pd.read_csv(path, sep="\t")
    df["pair"] = df["ligand"] + "-" + df["receptor"]
    df["axis_short"] = df["pair"] + "\n" + df["sender"].str.replace(" Cells", "", regex=False) + " -> " + df["receiver"].str.replace(" Cells", "", regex=False)
    df["theme"] = df["pair"].map(THEME).fillna("tumor")
    for col in [c for c in df.columns if c.endswith("_support")]:
        df[col] = ensure_bool(df[col])
    return df.sort_values("pyxenium_rank")


def read_scores_for_landscape() -> pd.DataFrame:
    path = PDC_PYXENIUM / "pyxenium_scores.tsv"
    cols = ["ligand", "receptor", "sender_celltype", "receiver_celltype", "CCI_score"]
    df = pd.read_csv(path, sep="\t", usecols=cols)
    df["pair"] = df["ligand"] + "-" + df["receptor"]
    df["rank"] = df["CCI_score"].rank(ascending=False, method="first").astype(int)
    return df


def read_full_coords() -> pd.DataFrame:
    h5ad_path = PDC_PYXENIUM / "input_cache" / "adata_full_from_sparse_bundle.h5ad"
    if not h5ad_path.exists():
        return pd.DataFrame()
    import h5py

    with h5py.File(h5ad_path, "r") as h5:
        coords = np.asarray(h5["obsm"]["spatial"][:])
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "layer": "all_cells"})


def source_meta(title: str, source_tables: Iterable[Path], message: str, caveat: str = "", rasterized: bool = False) -> dict[str, object]:
    return {
        "title": title,
        "source_tables": [str(p) for p in source_tables],
        "rasterized_layers": rasterized,
        "biological_message": message,
        "caveat": caveat,
        "output_formats": ["pdf", "svg", "png"],
    }


def fig1a_problem_schematic() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "Fig. 1a")
    ax.text(0.02, 0.92, "False-positive problem in spatial CCI", fontsize=STYLE_POINTS["title"], fontweight="bold")

    rounded_box(ax, 0.06, 0.57, 0.22, 0.2, "expression-only\ncandidate", "#FEF3C7", COLORS["weak"])
    rounded_box(ax, 0.37, 0.57, 0.22, 0.2, "database\nmembership", "#FEE2E2", COLORS["risk"])
    rounded_box(ax, 0.68, 0.57, 0.22, 0.2, "spatial\nproximity", "#DBEAFE", COLORS["blue"])
    ax.annotate("", xy=(0.34, 0.67), xytext=(0.28, 0.67), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(0.65, 0.67), xytext=(0.59, 0.67), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.5, 0.43, "Any one layer alone can be misleading", ha="center", fontsize=STYLE_POINTS["body"], color=COLORS["gray"])
    rounded_box(ax, 0.18, 0.14, 0.64, 0.18, "TopoLink-CCI requires concordant topology + expression + local contact,\nthen tests candidates with orthogonal controls.", "#ECFDF5", COLORS["pass"])

    source = pd.DataFrame(
        [
            {"element": "box", "x": 0.17, "y": 0.67, "label": "expression-only candidate", "category": "false_positive_source", "value": 1},
            {"element": "box", "x": 0.48, "y": 0.67, "label": "database membership", "category": "false_positive_source", "value": 1},
            {"element": "box", "x": 0.79, "y": 0.67, "label": "spatial proximity", "category": "false_positive_source", "value": 1},
            {"element": "box", "x": 0.5, "y": 0.23, "label": "TopoLink-CCI combined evidence", "category": "solution", "value": 1},
        ]
    )
    return fig, source, source_meta(
        "False-positive problem in spatial CCI",
        [],
        "Spatial CCI needs orthogonal evidence beyond expression, database membership or proximity alone.",
        "Schematic panel; source data records graphical elements.",
    )


def fig1b_topolink_score_formula() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "Fig. 1b")
    ax.text(0.02, 0.92, "TopoLink-CCI discovery score", fontsize=STYLE_POINTS["title"], fontweight="bold")
    ax.text(
        0.5,
        0.72,
        "TopoLink-CCI(l,r,s,t) = prior(l,r) x geometric mean( six evidence components )",
        ha="center",
        fontsize=7,
        color=COLORS["text"],
    )
    components = [
        ("sender anchor", "ligand fits sender topology", COLORS["vascular"]),
        ("receiver anchor", "receptor fits receiver topology", COLORS["notch"]),
        ("structure bridge", "sender-receiver tissue relation", COLORS["blue"]),
        ("sender expression", "ligand expressed in sender", "#2F855A"),
        ("receiver expression", "receptor expressed in receiver", "#2F855A"),
        ("local contact", "neighbor edges support interaction", "#C2410C"),
    ]
    xs = [0.06, 0.37, 0.68, 0.06, 0.37, 0.68]
    ys = [0.45, 0.45, 0.45, 0.18, 0.18, 0.18]
    rows = []
    for i, (name, desc, color) in enumerate(components):
        rounded_box(ax, xs[i], ys[i], 0.25, 0.16, f"{name}\n{desc}", "#FFFFFF", color, fontsize=5.4)
        rows.append({"element": "component", "x": xs[i] + 0.125, "y": ys[i] + 0.08, "label": name, "category": "score_component", "description": desc, "value": i + 1})
    return fig, pd.DataFrame(rows), source_meta(
        "TopoLink-CCI score formula",
        [],
        "The discovery score is a prior-weighted geometric mean of six diagnostic components.",
        "Schematic formula uses editable text rather than math outlines.",
    )


def fig1c_local_contact_model() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "Fig. 1c")
    ax.text(0.04, 0.92, "Local contact support", fontsize=STYLE_POINTS["title"], fontweight="bold")
    senders = [(0.25, 0.68), (0.20, 0.48), (0.32, 0.32)]
    receivers = [(0.68, 0.70), (0.76, 0.50), (0.62, 0.34)]
    rows = []
    for i, (x, y) in enumerate(senders):
        ax.scatter(x, y, s=80, color=COLORS["vascular"], edgecolor="white", linewidth=0.4, zorder=3)
        rows.append({"element": "node", "x": x, "y": y, "label": f"sender_{i+1}", "category": "sender", "value": 1})
    for i, (x, y) in enumerate(receivers):
        ax.scatter(x, y, s=80, color=COLORS["notch"], edgecolor="white", linewidth=0.4, zorder=3)
        rows.append({"element": "node", "x": x, "y": y, "label": f"receiver_{i+1}", "category": "receiver", "value": 1})
    edges = [(senders[0], receivers[0], 0.9), (senders[0], receivers[1], 0.5), (senders[1], receivers[1], 0.7), (senders[2], receivers[2], 0.8), (senders[1], receivers[2], 0.2)]
    for i, (a, b, strength) in enumerate(edges):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#374151", alpha=0.25 + strength * 0.55, lw=0.4 + strength * 1.2, zorder=1)
        rows.append({"element": "edge", "x": (a[0]+b[0])/2, "y": (a[1]+b[1])/2, "label": f"edge_{i+1}", "category": "neighbor_edge", "value": strength})
    rounded_box(ax, 0.08, 0.06, 0.84, 0.16, "local contact = f(edge strength, active-edge coverage, edge-count support)", "#F9FAFB", COLORS["gray"], fontsize=5.1)
    rows.append({"element": "formula", "x": 0.5, "y": 0.14, "label": "local contact support", "category": "summary", "value": 1})
    return fig, pd.DataFrame(rows), source_meta(
        "Local contact model",
        [],
        "Local contact links candidate molecular axes to the actual cell-cell neighbor graph.",
        "Schematic panel; source data records nodes and edges.",
    )


def fig1d_validation_gate_map() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "Fig. 1d")
    ax.text(0.02, 0.92, "Orthogonal validation gates adapted from CCC methods", fontsize=STYLE_POINTS["title"], fontweight="bold")
    gates = [
        ("CellPhoneDB/\nSquidpy", "cell-label\npermutation", COLORS["blue"]),
        ("CellChat", "group\nspecificity", COLORS["notch"]),
        ("stLearn/\nSpatialDM", "spatial null +\nmatched genes", COLORS["vascular"]),
        ("NicheNet", "receiver target\nsupport", COLORS["immune"]),
        ("LIANA", "cross-method\nconsensus", COLORS["stromal"]),
        ("pyXenium", "ablation +\nbootstrap", COLORS["pass"]),
    ]
    rows = []
    for i, (method, gate, color) in enumerate(gates):
        x = 0.04 + i * 0.155
        rounded_box(ax, x, 0.52, 0.13, 0.18, method, "#FFFFFF", color)
        ax.annotate("", xy=(x + 0.065, 0.42), xytext=(x + 0.065, 0.52), arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["text"]))
        rounded_box(ax, x, 0.22, 0.13, 0.16, gate, "#F9FAFB", color)
        rows.append({"element": "gate", "x": x + 0.065, "y": 0.3, "label": gate.replace("\n", " "), "category": method.replace("\n", " "), "value": i + 1})
    return fig, pd.DataFrame(rows), source_meta(
        "Validation gate map",
        [],
        "TopoLink-CCI reports candidates with controls inspired by established CCC methods.",
        "Schematic mapping, not a claim of identical software reimplementation.",
    )


def fig1e_evidence_matrix(evidence: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    add_panel_label(ax, "Fig. 1e")
    layers = [
        ("expression", "expression_specificity_support"),
        ("label perm.", "cell_label_permutation_support"),
        ("spatial null", "spatial_null_support"),
        ("matched genes", "matched_gene_control_support"),
        ("downstream", "downstream_target_support"),
        ("received signal", "functional_received_signal_support"),
        ("cross method", "cross_method_support"),
        ("ablation", "component_ablation_support"),
        ("bootstrap", "bootstrap_stability_support"),
    ]
    pairs = evidence["pair"].tolist()
    source_rows = []
    for i, pair in enumerate(pairs):
        for j, (label, col) in enumerate(layers):
            passed = bool(evidence.iloc[i][col])
            ax.scatter(j, i, s=50, marker="o" if passed else "X", color=COLORS["pass"] if passed else COLORS["weak"], edgecolor="white", linewidth=0.35)
            source_rows.append({"x": label, "y": pair, "value": int(passed), "category": "pass" if passed else "weak", "label": f"{pair}:{label}"})
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([x for x, _ in layers], rotation=35, ha="right")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_title("Evidence matrix for representative axes", loc="left", fontsize=STYLE_POINTS["title"])
    ax.grid(color="#E5E7EB", linewidth=0.35)
    ax.tick_params(length=0)
    return fig, pd.DataFrame(source_rows), source_meta(
        "Evidence matrix",
        [VALIDATION / "tables" / "topolink_cci_validation_v2_evidence.tsv"],
        "Seven representative TopoLink-CCI axes have strong computational support.",
        "Orange marks are axis-specific weak evidence layers, not artifact-risk calls.",
    )


def fig1f_false_positive_controls_quantitative(evidence: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    add_panel_label(ax, "Fig. 1f")
    metric_defs = [
        ("label perm. FDR", "cell_label_perm_fdr", "-log10"),
        ("spatial null FDR", "spatial_null_fdr", "-log10"),
        ("matched gene z", "matched_gene_z", "z"),
        ("downstream FDR", "downstream_target_fdr", "-log10"),
    ]
    rows = []
    for i, row in evidence.iterrows():
        pair = row["pair"]
        for j, (label, col, transform) in enumerate(metric_defs):
            raw = float(row[col])
            value = -math.log10(max(raw, 1e-300)) if transform == "-log10" else raw
            rows.append({"x": label, "y": pair, "value": value, "raw_value": raw, "category": transform, "label": pair})
    src = pd.DataFrame(rows)
    pairs = evidence["pair"].tolist()
    for _, row in src.iterrows():
        x = [m[0] for m in metric_defs].index(row["x"])
        y = pairs.index(row["y"])
        size = 10 + min(float(row["value"]), 12) * 8
        ax.scatter(x, y, s=size, color=COLORS["pass"], alpha=0.85, edgecolor="white", linewidth=0.25)
    ax.set_xticks(range(len(metric_defs)))
    ax.set_xticklabels([m[0] for m in metric_defs], rotation=25, ha="right")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_title("Quantitative false-positive controls", loc="left", fontsize=STYLE_POINTS["title"])
    ax.grid(color="#E5E7EB", linewidth=0.35)
    ax.tick_params(length=0)
    return fig, src, source_meta(
        "Quantitative false-positive controls",
        [VALIDATION / "tables" / "topolink_cci_validation_v2_evidence.tsv"],
        "Representative axes pass multiple quantitative controls, not just the TopoLink-CCI score.",
        "Bubble sizes use transformed metrics; raw values are in source data.",
    )


def fig2a_whole_dataset_rank_landscape(evidence: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    add_panel_label(ax, "Fig. 2a")
    scores = read_scores_for_landscape()
    bins = np.linspace(0, scores["CCI_score"].max(), 120)
    counts, edges = np.histogram(scores["CCI_score"], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, counts, color="#CBD5E1", step="mid")
    ax.set_yscale("log")
    ax.set_xlabel("TopoLink-CCI score")
    ax.set_ylabel("candidate axes")
    ax.set_title("Whole-dataset score landscape", loc="left", fontsize=STYLE_POINTS["title"])
    clean_axis(ax, "y")
    highlight_rows = []
    for _, row in evidence.iterrows():
        score = float(row["CCI_score"])
        pair = row["pair"]
        color = COLORS[THEME.get(pair, "tumor")]
        ax.axvline(score, color=color, lw=0.9, alpha=0.85)
        highlight_rows.append({"x": score, "y": np.nan, "value": int(row["pyxenium_rank"]), "category": row["theme"], "label": pair})
    hist_src = pd.DataFrame({"x": centers, "y": counts, "value": counts, "category": "histogram", "label": "CCI_score_bin"})
    src = pd.concat([hist_src, pd.DataFrame(highlight_rows)], ignore_index=True)
    return fig, src, source_meta(
        "Whole-dataset score landscape",
        [PDC_PYXENIUM / "pyxenium_scores.tsv", VALIDATION / "tables" / "topolink_cci_validation_v2_evidence.tsv"],
        "TopoLink-CCI ranks 1,319,600 whole-dataset candidate axes and highlights validated discoveries.",
        "Histogram bins are source data; highlighted lines mark seven representative axes.",
    )


def fig2b_top_interpretable_axes(evidence: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    add_panel_label(ax, "Fig. 2b")
    plot = evidence.sort_values("CCI_score", ascending=True)
    colors = [COLORS[THEME.get(pair, "tumor")] for pair in plot["pair"]]
    y = np.arange(len(plot))
    ax.barh(y, plot["CCI_score"], color=colors, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["pair"])
    ax.set_xlabel("TopoLink-CCI score")
    ax.set_xlim(0.60, 0.82)
    clean_axis(ax, "x")
    ax.set_title("Top interpretable axes", loc="left", fontsize=STYLE_POINTS["title"])
    for idx, row in enumerate(plot.itertuples()):
        ax.text(float(row.CCI_score) + 0.004, idx, f"r{int(row.pyxenium_rank)}", va="center", fontsize=STYLE_POINTS["tick_label"])
    src = plot[["pair", "sender", "receiver", "CCI_score", "pyxenium_rank", "biology_label", "theme"]].rename(columns={"CCI_score": "x", "pyxenium_rank": "value"})
    src["y"] = src["pair"]
    src["category"] = src["theme"]
    src["label"] = src["pair"]
    return fig, src, source_meta(
        "Top interpretable axes",
        [VALIDATION / "tables" / "topolink_cci_validation_v2_evidence.tsv"],
        "High-scoring axes span vascular, immune, stromal, Notch and tumor-intrinsic biology.",
    )


def fig2c_vwf_selp_component_decomposition() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    add_panel_label(ax, "Fig. 2c")
    path = DEEP_DIVE / "tables" / "component_decomposition.tsv"
    comp = pd.read_csv(path, sep="\t")
    order = ["sender_anchor", "receiver_anchor", "structure_bridge", "sender_expr", "receiver_expr", "local_contact"]
    comp = comp.set_index("component").loc[order].reset_index()
    y = np.arange(len(comp))
    colors = [COLORS["vascular"]] * 5 + ["#C2410C"]
    ax.barh(y, comp["value"], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(comp["component"].str.replace("_", " "))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("component value")
    clean_axis(ax, "x")
    ax.set_title("VWF-SELP component support", loc="left", fontsize=STYLE_POINTS["title"])
    ax.text(0.04, 0.05, "score 0.791\n12,779 local edges", transform=ax.transAxes, fontsize=STYLE_POINTS["annotation"], bbox=dict(fc="white", ec="#D1D5DB", lw=0.5))
    src = comp.rename(columns={"component": "label", "value": "x"})
    src["y"] = src["label"]
    src["value"] = src["x"]
    src["category"] = ["component"] * len(src)
    return fig, src, source_meta(
        "VWF-SELP component decomposition",
        [path],
        "The lead endothelial axis combines topology, expression and local contact support.",
    )


def fig2d_vwf_selp_rank_sensitivity() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    add_panel_label(ax, "Fig. 2d")
    path = DEEP_DIVE / "tables" / "rank_sensitivity.tsv"
    data = pd.read_csv(path, sep="\t").head(10)
    y = np.arange(len(data))[::-1]
    ax.scatter(data["target_rank"], y, color=COLORS["vascular"], s=28, zorder=3)
    ax.plot(data["target_rank"], y, color=COLORS["vascular"], lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(data["scenario"].str.replace("_", " "))
    ax.set_xlabel("VWF-SELP rank")
    ax.set_xlim(0.5, max(3, data["target_rank"].max() + 1))
    ax.set_title("Rank sensitivity", loc="left", fontsize=STYLE_POINTS["title"])
    clean_axis(ax, "x")
    src = data.copy()
    src["x"] = src["target_rank"]
    src["y"] = src["scenario"]
    src["value"] = src["target_score"]
    src["category"] = "rank_sensitivity"
    src["label"] = src["scenario"]
    return fig, src, source_meta(
        "VWF-SELP rank sensitivity",
        [path],
        "Component perturbations test whether the top axis is driven by one isolated term.",
    )


def fig2e_expression_specificity_dotplot() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    add_panel_label(ax, "Fig. 2e")
    path = DEEP_DIVE / "tables" / "expression_specificity_full_wta.tsv"
    df = pd.read_csv(path, sep="\t")
    genes = ["VWF", "SELP", "PECAM1", "EMCN", "KDR", "FLT1", "MMRN2", "CLEC14A", "PPBP", "PF4", "ITGA2B", "GP1BA", "HBB"]
    plot = df[df["gene"].isin(genes)].copy()
    cell_order = plot.groupby("cell_type")["mean_log1p_norm_by_gene"].max().sort_values(ascending=False).index.tolist()
    gene_order = genes
    x_map = {c: i for i, c in enumerate(cell_order)}
    y_map = {g: i for i, g in enumerate(gene_order)}
    plot["x"] = plot["cell_type"].map(x_map)
    plot["y"] = plot["gene"].map(y_map)
    sizes = 5 + 45 * plot["detection_fraction"].clip(0, 1)
    sc = ax.scatter(plot["x"], plot["y"], s=sizes, c=plot["mean_log1p_norm_by_gene"], cmap="viridis", vmin=0, vmax=1, edgecolor="none")
    ax.set_xticks(range(len(cell_order)))
    ax.set_xticklabels(cell_order, rotation=55, ha="right")
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(gene_order)
    ax.set_title("Expression specificity and contamination controls", loc="left", fontsize=STYLE_POINTS["title"])
    ax.tick_params(length=0)
    ax.grid(color="#F3F4F6", linewidth=0.35)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("gene-normalized expression", fontsize=STYLE_POINTS["axis_label"])
    cbar.ax.tick_params(labelsize=STYLE_POINTS["tick_label"], length=2)
    plot["value"] = plot["mean_log1p_norm_by_gene"]
    plot["category"] = np.where(plot["gene"].isin(["PPBP", "PF4", "ITGA2B", "GP1BA", "HBB"]), "contamination_control", "endothelial_or_axis_marker")
    plot["label"] = plot["gene"] + "|" + plot["cell_type"]
    return fig, plot, source_meta(
        "Expression specificity dotplot",
        [path],
        "VWF-SELP interpretation is supported by endothelial marker specificity and contamination controls.",
        "RNA expression specificity does not prove protein localization.",
    )


def fig2f_vwf_selp_spatial_hotspots() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("square")
    add_panel_label(ax, "Fig. 2f")
    hotspot_path = DEEP_DIVE / "tables" / "vwf_selp_hotspot_endothelial_cells.tsv"
    hotspots = pd.read_csv(hotspot_path, sep="\t")
    coords = read_full_coords()
    rasterized = False
    rows = []
    if not coords.empty:
        rasterized = True
        ax.scatter(coords["x"], coords["y"], s=0.08, color="#94A3B8", alpha=0.22, linewidths=0, rasterized=True)
        rows.append(coords.assign(value=np.nan, category="background", label="all_cells")[["x", "y", "value", "category", "label"]])
    sc = ax.scatter(
        hotspots["x"],
        hotspots["y"],
        c=hotspots["hotspot_score"],
        cmap="magma",
        vmin=0,
        vmax=1,
        s=5,
        edgecolor="none",
        rasterized=len(hotspots) > 5000,
    )
    rows.append(hotspots.assign(value=hotspots["hotspot_score"], category="hotspot", label=hotspots["cell_id"])[["x", "y", "value", "category", "label"]])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("VWF-SELP endothelial hotspots", loc="left", fontsize=STYLE_POINTS["title"])
    clean_axis(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("hotspot score", fontsize=STYLE_POINTS["axis_label"])
    cbar.ax.tick_params(labelsize=STYLE_POINTS["tick_label"], length=2)
    src = pd.concat(rows, ignore_index=True)
    return fig, src, source_meta(
        "VWF-SELP spatial hotspots",
        [hotspot_path, PDC_PYXENIUM / "input_cache" / "adata_full_from_sparse_bundle.h5ad"],
        "VWF-SELP hotspots localize to a subset of endothelial neighborhoods.",
        "Dense background cells are rasterized to keep Illustrator files usable.",
        rasterized=rasterized,
    )


def fig2g_hotspot_neighborhood_context() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    add_panel_label(ax, "Fig. 2g")
    path = DEEP_DIVE / "tables" / "hotspot_neighbor_context.tsv"
    data = pd.read_csv(path, sep="\t").sort_values("neighbor_fraction", ascending=True)
    y = np.arange(len(data))
    ax.barh(y, data["neighbor_fraction"], color=COLORS["vascular"], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(data["neighbor_cell_type"].str.replace(" Cells", ""))
    ax.set_xlabel("neighbor fraction")
    ax.set_title("Hotspot neighborhood context", loc="left", fontsize=STYLE_POINTS["title"])
    clean_axis(ax, "x")
    data["x"] = data["neighbor_fraction"]
    data["y"] = data["neighbor_cell_type"]
    data["value"] = data["neighbor_fraction"]
    data["category"] = data["group"]
    data["label"] = data["neighbor_cell_type"]
    return fig, data, source_meta(
        "Hotspot neighborhood context",
        [path],
        "Endothelial VWF-SELP hotspots can be interpreted through their neighboring cell ecology.",
    )


def fig2h_pathway_context() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    add_panel_label(ax, "Fig. 2h")
    path = DEEP_DIVE / "tables" / "vwf_selp_pathway_correlations.tsv"
    data = pd.read_csv(path, sep="\t").sort_values("spearman_rho_vs_vwf_selp_joint", ascending=True)
    y = np.arange(len(data))
    ax.barh(y, data["spearman_rho_vs_vwf_selp_joint"], color=COLORS["vascular"], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(data["pathway"].str.replace("_", "\n"))
    ax.set_xlabel("Spearman rho")
    ax.set_title("Pathway context", loc="left", fontsize=STYLE_POINTS["title"])
    clean_axis(ax, "x")
    data["x"] = data["spearman_rho_vs_vwf_selp_joint"]
    data["y"] = data["pathway"]
    data["value"] = data["spearman_rho_vs_vwf_selp_joint"]
    data["category"] = "pathway_correlation"
    data["label"] = data["pathway"]
    return fig, data, source_meta(
        "Pathway context",
        [path],
        "VWF-SELP hotspots align with vascular identity and thromboinflammatory pathway panels.",
        "Pathway panels are RNA-level summaries.",
    )


def fig2i_contour_ecology_context() -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("single")
    add_panel_label(ax, "Fig. 2i")
    path = DEEP_DIVE / "tables" / "contour_hypothesis_ranking.tsv"
    data = pd.read_csv(path, sep="\t").sort_values("mean_score", ascending=True)
    colors = [COLORS["vascular"] if "vascular" in p else COLORS["gray"] for p in data["program"]]
    y = np.arange(len(data))
    ax.barh(y, data["mean_score"], color=colors, height=0.62)
    ax.axvline(0, color="#111827", lw=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(data["program"].str.replace("_", "\n"))
    ax.set_xlabel("mean contour score")
    ax.set_title("Contour ecology context", loc="left", fontsize=STYLE_POINTS["title"])
    clean_axis(ax, "x")
    data["x"] = data["mean_score"]
    data["y"] = data["program"]
    data["value"] = data["mean_score"]
    data["category"] = np.where(data["program"].str.contains("vascular"), "vascular_contour", "other_contour")
    data["label"] = data["program"]
    return fig, data, source_meta(
        "Contour ecology context",
        [path],
        "Vascular interaction axes can be interpreted in the broader contour and tissue-ecology context.",
    )


def fig2j_cross_method_consensus(evidence: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, dict[str, object]]:
    fig, ax = make_figure("wide")
    add_panel_label(ax, "Fig. 2j")
    path = VALIDATION / "tables" / "cross_method_support_detail.tsv"
    data = pd.read_csv(path, sep="\t")
    data["pair"] = data["axis_id"].str.split("|", regex=False).str[0]
    methods = sorted(data["method"].unique())
    pairs = evidence["pair"].tolist()
    x_map = {m: i for i, m in enumerate(methods)}
    y_map = {p: i for i, p in enumerate(pairs)}
    data["exact_support"] = ensure_bool(data["exact_support"])
    data["same_lr_any_celltype_support"] = ensure_bool(data["same_lr_any_celltype_support"])
    data["support_level"] = np.where(data["exact_support"], 2, np.where(data["same_lr_any_celltype_support"], 1, 0))
    for _, row in data.iterrows():
        if row["pair"] not in y_map:
            continue
        marker = "s" if row["support_level"] == 2 else "o" if row["support_level"] == 1 else "x"
        color = COLORS["pass"] if row["support_level"] == 2 else COLORS["blue"] if row["support_level"] == 1 else "#D1D5DB"
        ax.scatter(x_map[row["method"]], y_map[row["pair"]], s=45, marker=marker, color=color, edgecolor="white", linewidth=0.35)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_title("Cross-method consensus", loc="left", fontsize=STYLE_POINTS["title"])
    ax.grid(color="#E5E7EB", linewidth=0.35)
    ax.tick_params(length=0)
    data["x"] = data["method"]
    data["y"] = data["pair"]
    data["value"] = data["support_level"]
    data["category"] = np.where(data["support_level"] == 2, "exact", np.where(data["support_level"] == 1, "same_lr", "none"))
    data["label"] = data["axis_id"] + "|" + data["method"]
    return fig, data, source_meta(
        "Cross-method consensus",
        [path],
        "TopoLink-CCI discoveries are triangulated against independent method outputs.",
        "Methods differ in score definitions; this panel shows support categories, not raw score equivalence.",
    )


PanelFunc = Callable[..., tuple[plt.Figure, pd.DataFrame, dict[str, object]]]


def build_registry(evidence: pd.DataFrame) -> dict[str, tuple[str, str, Callable[[], tuple[plt.Figure, pd.DataFrame, dict[str, object]]]]]:
    return {
        "fig1a": ("fig1a", "problem_schematic", fig1a_problem_schematic),
        "fig1b": ("fig1b", "topolink_score_formula", fig1b_topolink_score_formula),
        "fig1c": ("fig1c", "local_contact_model", fig1c_local_contact_model),
        "fig1d": ("fig1d", "validation_gate_map", fig1d_validation_gate_map),
        "fig1e": ("fig1e", "evidence_matrix", lambda: fig1e_evidence_matrix(evidence)),
        "fig1f": ("fig1f", "false_positive_controls_quantitative", lambda: fig1f_false_positive_controls_quantitative(evidence)),
        "fig2a": ("fig2a", "whole_dataset_rank_landscape", lambda: fig2a_whole_dataset_rank_landscape(evidence)),
        "fig2b": ("fig2b", "top_interpretable_axes", lambda: fig2b_top_interpretable_axes(evidence)),
        "fig2c": ("fig2c", "vwf_selp_component_decomposition", fig2c_vwf_selp_component_decomposition),
        "fig2d": ("fig2d", "vwf_selp_rank_sensitivity", fig2d_vwf_selp_rank_sensitivity),
        "fig2e": ("fig2e", "expression_specificity_dotplot", fig2e_expression_specificity_dotplot),
        "fig2f": ("fig2f", "vwf_selp_spatial_hotspots", fig2f_vwf_selp_spatial_hotspots),
        "fig2g": ("fig2g", "hotspot_neighborhood_context", fig2g_hotspot_neighborhood_context),
        "fig2h": ("fig2h", "pathway_context", fig2h_pathway_context),
        "fig2i": ("fig2i", "contour_ecology_context", fig2i_contour_ecology_context),
        "fig2j": ("fig2j", "cross_method_consensus", lambda: fig2j_cross_method_consensus(evidence)),
    }


def run_quality_checks(output_root: Path, expected_panels: int) -> None:
    pdf_n = len(list((output_root / "pdf").glob("*.pdf")))
    svg_n = len(list((output_root / "svg").glob("*.svg")))
    png_n = len(list((output_root / "png").glob("*.png")))
    src_n = len(list((output_root / "source_data").glob("*.tsv")))
    meta_n = len(list((output_root / "metadata").glob("*.json")))
    if len({pdf_n, svg_n, png_n, src_n, meta_n, expected_panels}) != 1:
        raise RuntimeError(
            f"Panel output count mismatch: pdf={pdf_n}, svg={svg_n}, png={png_n}, "
            f"source={src_n}, metadata={meta_n}, expected={expected_panels}"
        )
    manifest = pd.read_csv(output_root / "panel_manifest.tsv", sep="\t")
    if len(manifest) != expected_panels:
        raise RuntimeError(f"Manifest has {len(manifest)} rows, expected {expected_panels}")
    fig2f = manifest[manifest["panel_id"] == "fig2f"]
    if fig2f.empty or not bool(fig2f.iloc[0]["rasterized_layers"]):
        raise RuntimeError("fig2f must be marked as rasterized_layers=true")
    if (manifest["svg_text_count"] <= 0).any():
        bad = manifest.loc[manifest["svg_text_count"] <= 0, "panel_id"].tolist()
        raise RuntimeError(f"SVG text was not retained for panels: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", help="Panel id to render, e.g. fig1a. Repeatable. Default: all panels.")
    parser.add_argument("--all", action="store_true", help="Render all panels. This is the default when --panel is omitted.")
    args = parser.parse_args()

    style = FigureStyle.apply()
    evidence = load_evidence()
    registry = build_registry(evidence)
    selected = args.panel or list(registry.keys())
    unknown = [panel for panel in selected if panel not in registry]
    if unknown:
        raise ValueError(f"Unknown panel ids: {unknown}. Available: {sorted(registry)}")

    exporter = PanelExporter(OUT_ROOT, style)
    for key in selected:
        panel_id, slug, fn = registry[key]
        fig, source, metadata = fn()
        exporter.save_panel(fig, panel_id, slug, source, metadata)
    exporter.write_manifest()
    if len(selected) == len(registry):
        run_quality_checks(OUT_ROOT, expected_panels=len(registry))


if __name__ == "__main__":
    main()
