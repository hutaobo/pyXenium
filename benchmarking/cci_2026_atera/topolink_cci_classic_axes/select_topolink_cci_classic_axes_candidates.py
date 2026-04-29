"""Select high-scoring, biologically interpretable classic LR axes.

This script scans the full pyXenium whole-dataset LR score table and creates
curated outputs for manuscript-style interpretation. It uses CCI_score as the
primary ranking signal and applies a predefined classic-biology annotation
layer rather than learning or changing scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT.parent
INPUT = (
    BENCHMARK_ROOT
    / "pdc_collected"
    / "pdc_20260426_1327"
    / "runs"
    / "full_common"
    / "pyxenium"
    / "pyxenium_scores.tsv"
)
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

USECOLS = [
    "ligand",
    "receptor",
    "sender_celltype",
    "receiver_celltype",
    "sender_anchor",
    "receiver_anchor",
    "structure_bridge",
    "sender_expr",
    "receiver_expr",
    "local_contact",
    "contact_strength_raw",
    "contact_strength_normalized",
    "contact_coverage",
    "cross_edge_count",
    "prior_confidence",
    "CCI_score",
]


@dataclass(frozen=True)
class PairAnnotation:
    category: str
    biology_label: str
    interpretation: str
    evidence_note: str
    reference_key: str
    headline: bool = False


REFERENCES = {
    "wpb": "Weibel-Palade Bodies, NCBI Bookshelf: https://www.ncbi.nlm.nih.gov/books/NBK535353/",
    "cd93_mmrn2": "CD93 promotes beta1 integrin activation and fibronectin fibrillogenesis during tumor angiogenesis: https://pmc.ncbi.nlm.nih.gov/articles/PMC6063507/",
    "notch_pericyte": "Notch1 and Notch3 coordinate for pericyte-induced stabilization of vasculature: https://pubmed.ncbi.nlm.nih.gov/34878922/",
    "cxcl12": "Chemokine signaling in cancer-stroma communications: https://pmc.ncbi.nlm.nih.gov/articles/PMC8222467/",
    "cd48_cd2": "Enhanced murine CD4+ T cell responses induced by the CD2 ligand CD48: https://pubmed.ncbi.nlm.nih.gov/9862369/",
    "jag1_notch": "The Notch ligand Jagged1 as a target for anti-tumor therapy: https://pmc.ncbi.nlm.nih.gov/articles/PMC4174884/",
    "ecm_lrp1": "Interpreted as ECM/scavenger-receptor biology in the pyXenium WTA context; use as dataset-supported hypothesis.",
    "vegf": "Reactome VEGFA-VEGFR2 pathway and VEGF-family angiogenesis context: https://reactome.org/content/detail/R-HSA-4420097",
    "immune_checkpoint": "KLRB1/CLEC2D is interpreted as T/NK immune-regulatory biology; use as an immune-cell hypothesis.",
}


PAIR_ANNOTATIONS: dict[tuple[str, str], PairAnnotation] = {
    ("VWF", "SELP"): PairAnnotation(
        "vascular_WPB",
        "WPB / endothelial activation",
        "Endothelial VWF-SELP suggests a Weibel-Palade body/P-selectin-like vascular adhesion state.",
        "VWF and P-selectin are core endothelial Weibel-Palade body cargos linked to hemostasis, inflammation, and leukocyte rolling.",
        "wpb",
        True,
    ),
    ("VWF", "LRP1"): PairAnnotation(
        "vascular_stromal_ECM",
        "vascular-stromal matrix/scavenger axis",
        "Endothelial VWF toward CAF LRP1 highlights a vascular-to-stromal matrix/scavenger-receptor interface.",
        "This is a high-scoring dataset-supported vascular-stromal hypothesis rather than a single canonical pathway claim.",
        "ecm_lrp1",
        True,
    ),
    ("EFNB2", "PECAM1"): PairAnnotation(
        "vascular_identity",
        "endothelial identity / vascular adhesion",
        "Endothelial EFNB2-PECAM1 supports a vascular identity and endothelial adhesion/state axis.",
        "Both genes mark vascular/endothelial biology; interpret as endothelial organization rather than secreted paracrine signaling.",
        "vegf",
        True,
    ),
    ("MMRN2", "CLEC14A"): PairAnnotation(
        "angiogenesis_matrix",
        "tumor endothelial angiogenic complex",
        "MMRN2-CLEC14A points to tumor endothelial matrix remodeling and angiogenic vessel state.",
        "CD93/CLEC14A/MMRN2 family biology is linked to tumor vasculature, matrix organization, and endothelial angiogenic programs.",
        "cd93_mmrn2",
        True,
    ),
    ("HSPG2", "LRP1"): PairAnnotation(
        "vascular_stromal_ECM",
        "basement membrane / ECM receptor",
        "Endothelial HSPG2 to CAF LRP1 suggests basement-membrane-rich vascular-stromal remodeling.",
        "LRP1 frequently acts as a scavenger/ECM-associated receptor; here the strongest support is spatial WTA context.",
        "ecm_lrp1",
        True,
    ),
    ("COL4A2", "CD93"): PairAnnotation(
        "angiogenesis_matrix",
        "basement membrane angiogenesis",
        "COL4A2-CD93 links endothelial basement membrane signal with angiogenic endothelial receptor state.",
        "CD93 is implicated in endothelial matrix organization and tumor angiogenesis.",
        "cd93_mmrn2",
        True,
    ),
    ("MMRN2", "CD93"): PairAnnotation(
        "angiogenesis_matrix",
        "CD93-MMRN2 angiogenesis",
        "MMRN2-CD93 is a direct interpretable tumor-endothelial matrix/angiogenesis axis.",
        "CD93-MMRN2 signaling has reported roles in beta1 integrin activation and fibronectin fibrillogenesis during tumor angiogenesis.",
        "cd93_mmrn2",
        True,
    ),
    ("VWF", "ITGA9"): PairAnnotation(
        "vascular_WPB",
        "vascular adhesion / integrin",
        "VWF-ITGA9 extends the VWF vascular adhesion theme within endothelial neighborhoods.",
        "Interpret as an integrin-associated vascular adhesion hypothesis supported by endothelial spatial contact.",
        "wpb",
        True,
    ),
    ("MMP2", "PECAM1"): PairAnnotation(
        "CAF_ECM_remodeling",
        "CAF-endothelial remodeling",
        "CAF MMP2 toward endothelial PECAM1 supports a stromal-remodeling interface around vasculature.",
        "MMP2 is a classic ECM remodeling enzyme; PECAM1 anchors the receiver identity as endothelial.",
        "ecm_lrp1",
        True,
    ),
    ("COL4A1", "CD93"): PairAnnotation(
        "angiogenesis_matrix",
        "basement membrane angiogenesis",
        "COL4A1-CD93 is another basement-membrane/CD93 vascular angiogenesis axis.",
        "Kept in appendix/main table but treated as partially redundant with COL4A2-CD93.",
        "cd93_mmrn2",
    ),
    ("CXCL12", "CXCR4"): PairAnnotation(
        "immune_recruitment",
        "canonical CAF-immune chemokine axis",
        "CAF CXCL12 to T-cell CXCR4 recovers a canonical stromal chemokine immune-recruitment axis.",
        "CXCL12/CXCR4 is a well-studied cancer-stroma chemokine pathway with immune and tumor-microenvironment relevance.",
        "cxcl12",
        True,
    ),
    ("CXCL12", "ITGA4"): PairAnnotation(
        "immune_recruitment",
        "CAF-T cell immune recruitment/adhesion",
        "CAF CXCL12 to T-cell ITGA4 suggests chemokine-linked immune adhesion/recruitment.",
        "Interpret as a CXCL12-associated immune/stromal neighborhood, not as a replacement for CXCL12-CXCR4.",
        "cxcl12",
    ),
    ("CD48", "CD2"): PairAnnotation(
        "T_cell_signaling",
        "T-cell adhesion/co-stimulation",
        "T-cell CD48-CD2 indicates lymphocyte-local adhesion/co-stimulation biology.",
        "CD48-CD2 has experimental support in T-cell adhesion and activation contexts.",
        "cd48_cd2",
        True,
    ),
    ("CLEC2D", "KLRB1"): PairAnnotation(
        "T_cell_signaling",
        "T/NK immune-regulatory interaction",
        "CLEC2D-KLRB1 suggests a T/NK immune-regulatory neighborhood.",
        "Useful as an immune hypothesis; less central than CD48-CD2 or CXCL12-CXCR4.",
        "immune_checkpoint",
    ),
    ("CD34", "SELP"): PairAnnotation(
        "vascular_WPB",
        "endothelial adhesion",
        "CD34-SELP supports an endothelial adhesion program aligned with the VWF-SELP top hit.",
        "SELP/P-selectin biology links this to endothelial activation and adhesion.",
        "wpb",
        True,
    ),
    ("VEGFC", "FLT1"): PairAnnotation(
        "angiogenesis_growth_factor",
        "VEGF/angiogenesis",
        "VEGFC-FLT1 supports a vascular growth-factor/angiogenesis interpretation.",
        "Canonical VEGF-family CCI biology makes this a straightforward angiogenic axis.",
        "vegf",
        True,
    ),
    ("DLL4", "NOTCH3"): PairAnnotation(
        "Notch_pericyte",
        "endothelial-pericyte Notch",
        "Endothelial DLL4 to pericyte NOTCH3 supports vascular stabilization and pericyte-endothelial Notch biology.",
        "Notch1/Notch3 and DLL4-linked endothelial-pericyte signaling are reported in vascular stabilization contexts.",
        "notch_pericyte",
        True,
    ),
    ("DLL4", "NOTCH4"): PairAnnotation(
        "Notch_vascular",
        "endothelial Notch/angiogenesis",
        "DLL4-NOTCH4 extends the endothelial Notch angiogenesis module.",
        "Treat as supportive endothelial Notch biology and partially redundant with DLL4-NOTCH3.",
        "notch_pericyte",
    ),
    ("DLL1", "NOTCH3"): PairAnnotation(
        "Notch_pericyte",
        "endothelial-pericyte Notch",
        "DLL1-NOTCH3 provides another endothelial-to-pericyte Notch axis.",
        "Treat as supportive Notch biology and partially redundant with DLL4-NOTCH3.",
        "notch_pericyte",
    ),
    ("JAG1", "NOTCH2"): PairAnnotation(
        "tumor_Notch",
        "tumor Notch signaling",
        "Tumor-intrinsic JAG1-NOTCH2 suggests a Notch signaling state within 11q13 invasive tumor cells.",
        "JAG1/Notch signaling is broadly linked to cancer cell state, angiogenesis, stemness, EMT, and therapy resistance.",
        "jag1_notch",
        True,
    ),
    ("C3", "LRP1"): PairAnnotation(
        "complement_scavenger",
        "complement-ECM/scavenger receptor",
        "C3-LRP1 supports complement/scavenger-receptor biology in CAF-rich tissue neighborhoods.",
        "Dataset-supported complement/scavenger hypothesis; interpret with cell-type specificity checks.",
        "ecm_lrp1",
    ),
    ("A2M", "LRP1"): PairAnnotation(
        "complement_scavenger",
        "protease inhibitor/scavenger receptor",
        "A2M-LRP1 highlights protease-inhibitor/scavenger-receptor signaling around vascular-stromal niches.",
        "LRP1 is a plausible receptor for protease-inhibitor/scavenger biology.",
        "ecm_lrp1",
    ),
    ("C1QB", "LRP1"): PairAnnotation(
        "complement_scavenger",
        "macrophage-CAF complement/scavenger axis",
        "Macrophage C1QB to CAF LRP1 suggests complement/scavenger biology in macrophage-stromal niches.",
        "Useful as a myeloid-stromal hypothesis rather than a vascular headline axis.",
        "ecm_lrp1",
    ),
    ("THBS2", "CD36"): PairAnnotation(
        "CAF_ECM_remodeling",
        "stromal matrix angiogenesis",
        "CAF THBS2 to endothelial CD36 supports matrix-rich stromal angiogenesis biology.",
        "THBS/CD36 biology is interpretable in matrix remodeling, angiogenesis, and stromal context.",
        "ecm_lrp1",
        True,
    ),
    ("CCN1", "CAV1"): PairAnnotation(
        "CAF_ECM_remodeling",
        "mechanovascular signaling",
        "CCN1-CAV1 suggests a CAF/endothelial mechanovascular or matrix-associated signaling axis.",
        "Treat as a topology-supported mechanovascular hypothesis.",
        "ecm_lrp1",
    ),
}

HEADLINE_KEYS = [
    ("VWF", "SELP"),
    ("VWF", "LRP1"),
    ("EFNB2", "PECAM1"),
    ("MMRN2", "CLEC14A"),
    ("HSPG2", "LRP1"),
    ("COL4A2", "CD93"),
    ("MMRN2", "CD93"),
    ("VWF", "ITGA9"),
    ("MMP2", "PECAM1"),
    ("CXCL12", "CXCR4"),
    ("DLL4", "NOTCH3"),
    ("CD48", "CD2"),
    ("VEGFC", "FLT1"),
    ("THBS2", "CD36"),
    ("JAG1", "NOTCH2"),
]

NONCLASSIC_CAVEATS = {
    ("GNAS", "ADCY1"): "High score but more intracellular second-messenger pathway than interpretable extracellular LR.",
    ("CDH1", "IGF1R"): "Interesting tumor epithelial adhesion/growth receptor context, but less clean as a classic LR headline.",
    ("CDH1", "PTPRF"): "Cell adhesion/phosphatase context; biologically plausible but not a primary classic LR discovery.",
    ("DSC3", "DSG3"): "Desmosomal adhesion within DCIS; important structure biology but not a classic cell-cell interaction signaling axis.",
    ("DSG1", "DSC3"): "Desmosomal adhesion within DCIS; important structure biology but not a classic cell-cell interaction signaling axis.",
    ("JAG1", "CD46"): "Potential tumor/complement regulatory candidate; lower confidence than canonical JAG1-NOTCH.",
    ("PSAP", "CELSR1"): "Potentially interpretable but not prioritized as a classic cancer LR axis here.",
    ("APP", "LRP10"): "High-score tumor-intrinsic axis; less directly tied to canonical spatial LR interpretation for this report.",
    ("ADAM10", "NOTCH2"): "Notch proteolysis context rather than a cell-cell interaction pair; kept as caveat not headline.",
    ("ADAM15", "ITGB1"): "Adhesion/protease-integrin biology; plausible but less classic than selected ECM/vascular axes.",
    ("SPTAN1", "PTPRA"): "Cytoskeletal/phosphatase context; not treated as a classic extracellular LR headline.",
    ("TNC", "SDC4"): "ECM-syndecan axis is plausible but not in the preregistered classic headline set.",
}

CATEGORY_ORDER = [
    "vascular_WPB",
    "vascular_identity",
    "vascular_stromal_ECM",
    "angiogenesis_matrix",
    "angiogenesis_growth_factor",
    "CAF_ECM_remodeling",
    "Notch_pericyte",
    "tumor_Notch",
    "immune_recruitment",
    "T_cell_signaling",
    "complement_scavenger",
]

CATEGORY_COLORS = {
    "vascular_WPB": "#B21E48",
    "vascular_identity": "#0F7C80",
    "vascular_stromal_ECM": "#6A994E",
    "angiogenesis_matrix": "#2A9D8F",
    "angiogenesis_growth_factor": "#3A86FF",
    "CAF_ECM_remodeling": "#B08968",
    "Notch_pericyte": "#7B2CBF",
    "tumor_Notch": "#D00000",
    "immune_recruitment": "#2E8B57",
    "T_cell_signaling": "#F4A261",
    "complement_scavenger": "#5C677D",
}


def annotate_row(row: pd.Series, global_rank: int) -> dict[str, object]:
    key = (row["ligand"], row["receptor"])
    ann = PAIR_ANNOTATIONS[key]
    return {
        "global_rank": global_rank,
        "ligand": row["ligand"],
        "receptor": row["receptor"],
        "lr_pair": f"{row['ligand']}-{row['receptor']}",
        "sender": row["sender_celltype"],
        "receiver": row["receiver_celltype"],
        "sender_receiver": f"{row['sender_celltype']} -> {row['receiver_celltype']}",
        "CCI_score": row["CCI_score"],
        "sender_anchor": row["sender_anchor"],
        "receiver_anchor": row["receiver_anchor"],
        "structure_bridge": row["structure_bridge"],
        "sender_expr": row["sender_expr"],
        "receiver_expr": row["receiver_expr"],
        "local_contact": row["local_contact"],
        "contact_strength_raw": row["contact_strength_raw"],
        "contact_strength_normalized": row["contact_strength_normalized"],
        "contact_coverage": row["contact_coverage"],
        "cross_edge_count": row["cross_edge_count"],
        "prior_confidence": row["prior_confidence"],
        "category": ann.category,
        "biology_label": ann.biology_label,
        "interpretation": ann.interpretation,
        "evidence_note": ann.evidence_note,
        "reference_key": ann.reference_key,
        "reference": REFERENCES[ann.reference_key],
        "headline_predefined": key in HEADLINE_KEYS,
    }


def scan_scores() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    allowed_keys = set(PAIR_ANNOTATIONS)
    caveat_keys = set(NONCLASSIC_CAVEATS)
    selected: list[dict[str, object]] = []
    caveats: list[dict[str, object]] = []
    row_offset = 0

    for chunk in pd.read_csv(INPUT, sep="\t", usecols=USECOLS, chunksize=200_000):
        for idx, row in chunk.iterrows():
            global_rank = row_offset + (idx - chunk.index[0]) + 1
            key = (row["ligand"], row["receptor"])
            if key in allowed_keys:
                selected.append(annotate_row(row, global_rank))
            elif key in caveat_keys and len(caveats) < 80:
                caveats.append(
                    {
                        "global_rank": global_rank,
                        "ligand": row["ligand"],
                        "receptor": row["receptor"],
                        "lr_pair": f"{row['ligand']}-{row['receptor']}",
                        "sender": row["sender_celltype"],
                        "receiver": row["receiver_celltype"],
                        "CCI_score": row["CCI_score"],
                        "local_contact": row["local_contact"],
                        "cross_edge_count": row["cross_edge_count"],
                        "reason_not_headline": NONCLASSIC_CAVEATS[key],
                    }
                )
        row_offset += len(chunk)

    selected_df = pd.DataFrame(selected).sort_values(["CCI_score", "global_rank"], ascending=[False, True])
    caveat_df = pd.DataFrame(caveats).sort_values(["CCI_score", "global_rank"], ascending=[False, True])
    return selected_df, caveat_df, row_offset


def choose_headline(selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in HEADLINE_KEYS:
        lig, rec = key
        sub = selected_df[(selected_df["ligand"] == lig) & (selected_df["receptor"] == rec)]
        if sub.empty:
            raise RuntimeError(f"Missing headline CCI pair {lig}-{rec}")
        rows.append(sub.sort_values(["CCI_score", "global_rank"], ascending=[False, True]).iloc[0])
    headline = pd.DataFrame(rows).sort_values(["CCI_score", "global_rank"], ascending=[False, True]).reset_index(drop=True)
    headline.insert(0, "classic_cci_rank", range(1, len(headline) + 1))
    return headline


def write_tables(headline: pd.DataFrame, selected_df: pd.DataFrame, caveat_df: pd.DataFrame, n_rows: int) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    headline.to_csv(TABLES / "topolink_cci_classic_axes_candidates.tsv", sep="\t", index=False)
    selected_df.to_csv(TABLES / "topolink_cci_classic_axes_candidates_all.tsv", sep="\t", index=False)
    caveat_df.to_csv(TABLES / "pyxenium_high_score_nonclassic_caveats.tsv", sep="\t", index=False)
    summary = {
        "input": str(INPUT),
        "input_rows": n_rows,
        "headline_rows": int(len(headline)),
        "all_classic_rows": int(len(selected_df)),
        "caveat_rows": int(len(caveat_df)),
        "headline_top": headline[["lr_pair", "sender_receiver", "CCI_score"]].head(5).to_dict(orient="records"),
    }
    (TABLES / "topolink_cci_classic_axes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_figure(headline: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = headline.sort_values("CCI_score", ascending=True)
    labels = [f"{r.lr_pair}\n{r.sender} -> {r.receiver}" for r in df.itertuples()]
    colors = [CATEGORY_COLORS.get(c, "#7E7E7E") for c in df["category"]]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=220)
    ax.barh(range(len(df)), df["CCI_score"], color=colors, alpha=0.92)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("pyXenium CCI_score", fontsize=11)
    ax.set_title("High-scoring classic TopoLink-CCI axes with strong biological interpretability", fontsize=13, weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.set_axisbelow(True)
    for i, score in enumerate(df["CCI_score"]):
        ax.text(score + 0.006, i, f"{score:.3f}", va="center", fontsize=8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0, max(0.84, df["CCI_score"].max() + 0.05))

    handles = []
    seen = []
    for category in headline["category"]:
        if category not in seen:
            seen.append(category)
            handles.append(plt.Line2D([0], [0], marker="s", linestyle="", color=CATEGORY_COLORS[category], label=category.replace("_", " ")))
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "topolink_cci_classic_axes_top_axes.png", bbox_inches="tight")
    fig.savefig(FIGURES / "topolink_cci_classic_axes_top_axes.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(headline: pd.DataFrame, selected_df: pd.DataFrame, caveat_df: pd.DataFrame, n_rows: int) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# pyXenium high-scoring classic LR candidates")
    lines.append("")
    lines.append(f"Input table: `{INPUT}`")
    lines.append(f"Rows scanned: `{n_rows:,}`")
    lines.append("")
    lines.append(
        "Selection strategy: candidates are sorted by `CCI_score` and filtered to predefined cell-cell interaction pairs with established or strongly interpretable biology. The score is not changed; biological interpretability is an annotation layer."
    )
    lines.append("")
    lines.append("## Headline candidates")
    lines.append("")
    lines.append("| Rank | CCI pair | Sender -> Receiver | CCI_score | Category | Interpretation |")
    lines.append("|---:|---|---|---:|---|---|")
    for row in headline.itertuples(index=False):
        lines.append(
            f"| {row.classic_cci_rank} | `{row.lr_pair}` | {row.sender_receiver} | {row.CCI_score:.3f} | {row.biology_label} | {row.interpretation} |"
        )

    lines.append("")
    lines.append("## Biology groups")
    for category in CATEGORY_ORDER:
        sub = headline[headline["category"] == category]
        if sub.empty:
            continue
        lines.append("")
        lines.append(f"### {category.replace('_', ' ')}")
        for row in sub.itertuples(index=False):
            lines.append(
                f"- `{row.lr_pair}` ({row.sender_receiver}, CCI_score={row.CCI_score:.3f}): {row.evidence_note}"
            )

    lines.append("")
    lines.append("## High-score non-classic caveats")
    lines.append("")
    lines.append("These rows scored highly but were not used as headline classic LR discoveries because their interpretation is less clean as an extracellular cell-cell interaction axis.")
    lines.append("")
    lines.append("| Global rank | CCI pair | Sender -> Receiver | CCI_score | Reason |")
    lines.append("|---:|---|---|---:|---|")
    for row in caveat_df.head(15).itertuples(index=False):
        lines.append(
            f"| {row.global_rank} | `{row.lr_pair}` | {row.sender} -> {row.receiver} | {row.CCI_score:.3f} | {row.reason_not_headline} |"
        )

    lines.append("")
    lines.append("## References")
    lines.append("")
    used_keys = list(dict.fromkeys(headline["reference_key"].tolist()))
    for key in used_keys:
        lines.append(f"- {REFERENCES[key]}")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `tables/topolink_cci_classic_axes_candidates.tsv`: headline representatives.")
    lines.append("- `tables/topolink_cci_classic_axes_candidates_all.tsv`: all matching classic LR rows across sender-receiver contexts.")
    lines.append("- `tables/pyxenium_high_score_nonclassic_caveats.tsv`: high-scoring rows deliberately not used as headline classic LR discoveries.")
    lines.append("- `figures/topolink_cci_classic_axes_top_axes.png`: compact ranked figure.")

    (REPORTS / "topolink_cci_classic_axes_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(headline: pd.DataFrame, selected_df: pd.DataFrame, caveat_df: pd.DataFrame, n_rows: int) -> None:
    if n_rows != 1_319_600:
        raise RuntimeError(f"Expected 1,319,600 rows, found {n_rows:,}")
    if headline["CCI_score"].is_monotonic_decreasing is False:
        raise RuntimeError("Headline candidates are not sorted by CCI_score descending.")
    required = {
        "classic_cci_rank",
        "global_rank",
        "lr_pair",
        "sender_receiver",
        "CCI_score",
        "sender_anchor",
        "receiver_anchor",
        "structure_bridge",
        "sender_expr",
        "receiver_expr",
        "local_contact",
        "cross_edge_count",
        "biology_label",
        "interpretation",
    }
    missing = required.difference(headline.columns)
    if missing:
        raise RuntimeError(f"Missing required headline columns: {sorted(missing)}")
    if headline["biology_label"].isna().any() or headline["interpretation"].isna().any():
        raise RuntimeError("Every headline candidate must have biology label and interpretation.")
    if selected_df.empty:
        raise RuntimeError("No classic LR rows found.")
    if caveat_df.empty:
        raise RuntimeError("No caveat rows found; expected at least GNAS-ADCY1/CDH1-related caveats.")


def main() -> None:
    selected_df, caveat_df, n_rows = scan_scores()
    headline = choose_headline(selected_df)
    validate(headline, selected_df, caveat_df, n_rows)
    write_tables(headline, selected_df, caveat_df, n_rows)
    make_figure(headline)
    write_report(headline, selected_df, caveat_df, n_rows)
    print(f"Scanned {n_rows:,} rows")
    print(f"Wrote {len(headline)} headline candidates and {len(selected_df):,} all classic rows")
    print(f"Top candidate: {headline.iloc[0]['lr_pair']} ({headline.iloc[0]['CCI_score']:.6f})")


if __name__ == "__main__":
    main()
