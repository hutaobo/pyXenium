"""Create the final NBT submission-ready package for the mTM WTA project."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "tutorials"
    / "multimodal_histoseg_lazyslide_breast_wta"
    / "naturebiotech_package"
)
DEFENSE_DIR = PACKAGE_ROOT / "autopilot_20260512_defense"
FINAL_DIR = PACKAGE_ROOT / "FINAL_SUBMISSION_NBT_20260513"
SOURCE_TABLE_DIR = FINAL_DIR / "Source_Tables"
PROVENANCE_DIR = FINAL_DIR / "Provenance"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def table_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    display = frame if max_rows is None else frame.head(max_rows)
    return display.to_markdown(index=False, floatfmt=".3g")


def create_figure1() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    colors = {
        "he": "#F7C6D0",
        "contour": "#DDEBFF",
        "embedding": "#E6F4EA",
        "wta": "#FFF3CD",
        "residual": "#EADCF8",
    }
    boxes = [
        (0.45, 4.35, 2.25, 1.35, "H&E WSI", "Direct LazySlide\nPLIP / UNI"),
        (3.15, 4.35, 2.25, 1.35, "HistoSeg", "Biological contours\nS1-S5 / cervical regions"),
        (5.85, 4.35, 2.25, 1.35, "Contour\nsummary", "Tile embeddings\n+ tissue geometry"),
        (8.55, 4.35, 2.25, 1.35, "Atera WTA", "18,000-gene\nsingle-cell programs"),
        (
            4.35,
            1.65,
            4.6,
            1.35,
            "Contour-constrained\nresidual decoding",
            "Program ~ H&E embedding\n+ structure + x/y + boundary distance",
        ),
        (9.95, 1.65, 3.05, 1.35, "Outputs", "Hidden continua\nMAZ\ncross-cancer dictionary"),
    ]
    box_colors = [colors["he"], colors["contour"], colors["embedding"], colors["wta"], colors["residual"], "#EAF7F9"]
    for idx, (x, y, w, h, title, subtitle) in enumerate(boxes):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.2,
            edgecolor="#222222",
            facecolor=box_colors[idx],
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + 0.32, subtitle, ha="center", va="bottom", fontsize=8.4)

    arrows = [
        ((2.75, 5.03), (3.1, 5.03)),
        ((5.45, 5.03), (5.8, 5.03)),
        ((8.15, 5.03), (8.5, 5.03)),
        ((7.0, 4.3), (6.65, 3.05)),
        ((9.3, 4.3), (7.3, 3.05)),
        ((9.0, 2.33), (9.9, 2.33)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

    ax.text(
        0.55,
        0.48,
        "mTM anchors foundation-model morphology to biologically meaningful contours, then tests which molecular programs remain encoded after anatomical labels and spatial covariates are fixed.",
        fontsize=9.0,
        color="#333333",
    )
    fig.tight_layout(pad=0.8)
    for path in [DEFENSE_DIR / "Figure1_mTM_Framework.png", FINAL_DIR / "Figure_1_mTM_Framework.png"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
    for path in [DEFENSE_DIR / "Figure1_mTM_Framework.pdf", FINAL_DIR / "Figure_1_mTM_Framework.pdf"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def create_maz_summary_figure() -> None:
    """Create a clean publication-facing MAZ summary panel.

    The older exploratory profile plot is intentionally not reused here because
    its title/legend layout is cramped. This summary keeps the same evidence but
    presents only the QC-passing focus programs needed for visual review.
    """
    maz = pd.read_csv(DEFENSE_DIR / "MAZ_QC_Table_v2.csv")
    focus = maz[maz["is_focus"].astype(bool) | maz["passes_ring_qc"].astype(bool)].copy()
    focus["abs_ring_rho"] = focus["ring_profile_spearman_rho"].abs()
    focus = focus.sort_values(["dataset", "is_focus", "abs_ring_rho"], ascending=[True, False, False])
    focus = focus.groupby(["dataset", "program"], as_index=False).head(1)
    focus = focus.sort_values(["dataset", "abs_ring_rho"], ascending=[True, False]).head(14)

    labels = [
        f"{row.dataset}: {row.assigned_structure}\n{row.program.replace('_', ' ')}"
        for row in focus.itertuples(index=False)
    ]
    y_positions = list(range(len(focus)))
    colors = focus["dataset"].map({"breast": "#4E79A7", "cervical": "#E15759"}).fillna("#6B7280")
    sizes = 120 + 320 * focus["abs_ring_rho"].clip(0, 1)

    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.42 * len(focus) + 1.6)))
    ax.axvline(0, color="#555555", linewidth=1.0)
    ax.hlines(y_positions, 0, focus["ring_profile_spearman_rho"], color="#C8CDD4", linewidth=1.2)
    ax.scatter(
        focus["ring_profile_spearman_rho"],
        y_positions,
        s=sizes,
        c=colors,
        edgecolor="#1F2937",
        linewidth=0.6,
        alpha=0.92,
    )
    for y, row in zip(y_positions, focus.itertuples(index=False), strict=True):
        lag = getattr(row, "molecular_minus_image_peak_um")
        text = f"{lag:+.0f} um" if pd.notna(lag) else "lag n/a"
        x = row.ring_profile_spearman_rho
        ax.text(
            x + (0.045 if x >= 0 else -0.045),
            y,
            text,
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8.2,
            color="#333333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlim(-1.08, 1.08)
    ax.set_xlabel("Ring-level morphology-to-WTA coupling (Spearman rho)")
    ax.set_title("")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, color="#D4D8DD")
    ax.spines[["top", "right", "left"]].set_visible(False)

    breast_handle = plt.Line2D([0], [0], marker="o", color="w", label="Breast WTA", markerfacecolor="#4E79A7", markersize=8)
    cervical_handle = plt.Line2D([0], [0], marker="o", color="w", label="Cervical WTA", markerfacecolor="#E15759", markersize=8)
    ax.legend(
        handles=[breast_handle, cervical_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=9.2,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for path in [
        DEFENSE_DIR / "Figure3_MAZ_Boundary_Coupling_Summary.png",
        FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.png",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
    for path in [
        DEFENSE_DIR / "Figure3_MAZ_Boundary_Coupling_Summary.pdf",
        FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.pdf",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def create_supplementary_information() -> None:
    permutation = pd.read_csv(DEFENSE_DIR / "Spatial_Permutation_Defense_Report.csv")
    bootstrap = pd.read_csv(DEFENSE_DIR / "Spatial_BlockBootstrap_CI.csv")
    maz = pd.read_csv(DEFENSE_DIR / "MAZ_QC_Table_v2.csv")
    signature = pd.read_csv(DEFENSE_DIR / "CrossCancer_Morphological_Signature_Table.csv")
    hero = pd.read_csv(DEFENSE_DIR / "Figure2_Luminal_ER_HeroPatch_4Pairs.csv")

    methods = (DEFENSE_DIR / "Methods_Statistics_Defense.md").read_text(encoding="utf-8").strip()
    tech = (DEFENSE_DIR / "Technical_Manifest_v1.md").read_text(encoding="utf-8").strip()
    maz_text = (DEFENSE_DIR / "MAZ_Biology_Narrative_v2.md").read_text(encoding="utf-8").strip()

    perm_cols = [
        "dataset",
        "model",
        "program",
        "program_family",
        "n_contours",
        "recomputed_partial_rho",
        "permutations",
        "permutation_empirical_p",
        "null_abs_rho_q95",
        "null_abs_rho_q99",
        "passes_permutation_99",
    ]
    boot_cols = [
        "dataset",
        "model",
        "program",
        "n_bootstrap",
        "bootstrap_rho_median",
        "bootstrap_rho_ci_low",
        "bootstrap_rho_ci_high",
    ]
    maz_focus = maz[maz["is_focus"].astype(bool) | maz["passes_ring_qc"].astype(bool)].copy()
    maz_cols = [
        "dataset",
        "program",
        "assigned_structure",
        "ring_profile_spearman_rho",
        "ring_profile_spearman_p_value",
        "lead_lag_class",
        "passes_ring_qc",
    ]
    hero_cols = ["pair", "polarity", "contour_id", "wta_program_z", "oriented_he_embedding_z", "hidden_program_score"]

    lines = [
        "# Supplementary Information",
        "",
        "## Supplementary Methods",
        "",
        methods,
        "",
        "## Technical Provenance",
        "",
        tech,
        "",
        "## Spatial Permutation and Block Bootstrap Logic",
        "",
        "For each candidate image-program association, molecular residuals were shuffled within predeclared strata that preserve HistoSeg structure, centroid bins and boundary-distance bins where available. This breaks contour-wise H&E-to-WTA pairing while retaining coarse spatial organization. A candidate was considered spatially defended when the observed absolute residual correlation exceeded the 95% spatial-null threshold; the strongest reported candidates also exceeded the 99% threshold.",
        "",
        "### Supplementary Table 1. Spatial permutation defense",
        "",
        table_to_markdown(permutation[perm_cols]),
        "",
        "### Supplementary Table 2. Spatial block-bootstrap confidence intervals",
        "",
        table_to_markdown(bootstrap[boot_cols]),
        "",
        "## Molecularly Active Zone Analysis",
        "",
        maz_text,
        "",
        "### Supplementary Table 3. MAZ QC subset",
        "",
        table_to_markdown(maz_focus[maz_cols].head(20)),
        "",
        "## Cross-Cancer Morphological Signature Table",
        "",
        "The final Figure 4 signature table compares PLIP and UNI within breast and cervical WTA at the program-family level. It does not compare raw embedding dimensions across models; it compares program-family recovery under the same contour-constrained residual-decoding design.",
        "",
        "### Supplementary Table 4. Program-family recovery by cancer and model",
        "",
        table_to_markdown(signature),
        "",
        "## Figure 2 Hero Patch Selection",
        "",
        "Hero patches were selected from Breast S3 luminal estrogen-response contours by pairing the strongest high-program residual contours with the strongest low-program residual contours. Patches are center-cropped to equal square fields and labeled with WTA program z-score and oriented H&E embedding z-score.",
        "",
        "### Supplementary Table 5. Final luminal ER hero patch pairs",
        "",
        table_to_markdown(hero[hero_cols]),
        "",
        "## Supplementary Figure Captions",
        "",
        "- Supplementary Fig. 1: Spatial permutation defense summary (`Final_SpatialPermutation_Defense.pdf`).",
        "- Supplementary Fig. 2: Breast S3 luminal ER hero-patch pairs (`Figure2_Luminal_ER_HeroPatch_4Pairs.pdf`).",
        "- Supplementary Fig. 3: MAZ boundary-coupling profiles (`Final_Figure3_MAZ_v2.pdf`).",
        "- Supplementary Fig. 4: Cross-cancer morphological signature table (`Final_Figure4_CrossCancer_Morphological_Signature_Table.pdf`).",
        "",
        "## Availability Notes",
        "",
        "No IHC or protein validation is claimed. The WTA layer is used as transcriptomic discovery evidence and as a basis for future orthogonal validation.",
    ]
    out = DEFENSE_DIR / "Supplementary_Information_v1.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_review_guide() -> None:
    lines = [
        "# Final Review Guide for User",
        "",
        "This checklist is designed for a five-minute final sign-off before submission packaging.",
        "",
        "## 1. Figure 2 luminal ER hero-patch pair",
        "",
        "Open `Figure_2_Luminal_ER_HeroPatch_4Pairs.png`. The most visually compelling contrast is Pair 1: `S3 S3 #424.1` has WTA luminal ER z = +3.57, while `S3 S3 #550.1` has WTA luminal ER z = -1.14. The high side is cellular and tumor-rich; the low side is collagen-dominant. This is the most intuitive pathology-facing example of hidden endocrine state inside one S3 compartment.",
        "",
        "## 2. Figure 4 cross-cancer signature table",
        "",
        "Open `Figure_4_CrossCancer_Morphological_Signature_Table.png`. Check that the story is immediately visible: Breast WTA is strongest for endocrine/epithelial and metabolic/stress programs, while Cervical WTA is strongest for stromal/CAF/ECM, immune ecology and invasion/EMT. PLIP and UNI are shown side by side, which is the cleanest model-agnostic visual.",
        "",
        "## 3. Spatial defense and MAZ boundary panel",
        "",
        "Open `Supplementary_Figure_SpatialPermutation_Defense.pdf` and `Figure_3_MAZ_Boundary_Coupling.pdf`. These are the defensive backbone: the first addresses spatial autocorrelation, and the second makes the boundary-ecology claim conservative rather than causal.",
        "",
        "## Required final human checks",
        "",
        "- Confirm no wording claims IHC/protein validation.",
        "- Confirm the title and abstract are ambitious enough for NBT but still framed as `Contour-constrained residual decoding`.",
        "- Confirm Figure 2 low-program patches with low cell counts are described as visual examples, not quantitative proof by themselves.",
    ]
    (DEFENSE_DIR / "FINAL_REVIEW_GUIDE_FOR_USER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_submission_files() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    copies = [
        (DEFENSE_DIR / "Full_Manuscript_v3.md", FINAL_DIR / "Main_Text.md"),
        (DEFENSE_DIR / "Cover_Letter_Draft_v2.md", FINAL_DIR / "Cover_Letter.md"),
        (DEFENSE_DIR / "Response_to_Likely_Reviewers_v2.md", FINAL_DIR / "Response_to_Reviewers.md"),
        (DEFENSE_DIR / "Supplementary_Information_v1.md", FINAL_DIR / "Supplementary_Information.md"),
        (DEFENSE_DIR / "FINAL_REVIEW_GUIDE_FOR_USER.md", FINAL_DIR / "FINAL_REVIEW_GUIDE_FOR_USER.md"),
        (DEFENSE_DIR / "Figure1_mTM_Framework.pdf", FINAL_DIR / "Figure_1_mTM_Framework.pdf"),
        (DEFENSE_DIR / "Figure1_mTM_Framework.png", FINAL_DIR / "Figure_1_mTM_Framework.png"),
        (PACKAGE_ROOT / "Final_Figure2_Pack.pdf", FINAL_DIR / "Figure_2_Breast_S3_Hidden_Continua.pdf"),
        (DEFENSE_DIR / "Figure2_Luminal_ER_HeroPatch_4Pairs.pdf", FINAL_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs.pdf"),
        (DEFENSE_DIR / "Figure2_Luminal_ER_HeroPatch_4Pairs.png", FINAL_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs.png"),
        (DEFENSE_DIR / "Final_Figure3_MAZ_v2.pdf", FINAL_DIR / "Figure_3_MAZ_Boundary_Coupling.pdf"),
        (DEFENSE_DIR / "Figure3_MAZ_Boundary_Coupling_Summary.png", FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.png"),
        (DEFENSE_DIR / "Figure3_MAZ_Boundary_Coupling_Summary.pdf", FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.pdf"),
        (
            DEFENSE_DIR / "Final_Figure4_CrossCancer_Morphological_Signature_Table.pdf",
            FINAL_DIR / "Figure_4_CrossCancer_Morphological_Signature_Table.pdf",
        ),
        (
            DEFENSE_DIR / "Final_Figure4_CrossCancer_Morphological_Signature_Table.png",
            FINAL_DIR / "Figure_4_CrossCancer_Morphological_Signature_Table.png",
        ),
        (
            DEFENSE_DIR / "Final_SpatialPermutation_Defense.pdf",
            FINAL_DIR / "Supplementary_Figure_SpatialPermutation_Defense.pdf",
        ),
        (DEFENSE_DIR / "Interactive_Hero_Contour_Browser.html", FINAL_DIR / "Interactive_Hero_Contour_Browser.html"),
        (DEFENSE_DIR / "Technical_Manifest_v1.md", PROVENANCE_DIR / "Technical_Manifest.md"),
        (DEFENSE_DIR / "Technical_Manifest_File_Hashes.csv", PROVENANCE_DIR / "Technical_Manifest_File_Hashes.csv"),
        (DEFENSE_DIR / "Self_Consistency_Check_Report.md", PROVENANCE_DIR / "Self_Consistency_Check_Report.md"),
    ]
    for src, dst in copies:
        copy_file(src, dst)
    optional_provenance = DEFENSE_DIR / "A100_ReadOnly_Cleanup_Check.md"
    if optional_provenance.exists():
        copy_file(optional_provenance, PROVENANCE_DIR / "A100_ReadOnly_Cleanup_Check.md")

    source_tables = [
        "Spatial_Permutation_Defense_Report.csv",
        "Spatial_BlockBootstrap_CI.csv",
        "CrossCancer_Morphological_Signature_Table.csv",
        "CrossCancer_Morphomolecular_Dictionary.csv",
        "MAZ_QC_Table_v2.csv",
        "Figure2_Luminal_ER_HeroPatch_4Pairs.csv",
        "interactive_assets_manifest.csv",
    ]
    for name in source_tables:
        copy_file(DEFENSE_DIR / name, SOURCE_TABLE_DIR / name)


def write_submission_manifest() -> None:
    lines = [
        "# FINAL SUBMISSION NBT 20260513",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        "## Core files",
        "",
        "- `Main_Text.md`",
        "- `Cover_Letter.md`",
        "- `Response_to_Reviewers.md`",
        "- `Supplementary_Information.md`",
        "- `FINAL_REVIEW_GUIDE_FOR_USER.md`",
        "",
        "## Figures",
        "",
        "- `Figure_1_mTM_Framework.pdf/png`",
        "- `Figure_2_Breast_S3_Hidden_Continua.pdf`",
        "- `Figure_2_Luminal_ER_HeroPatch_4Pairs.pdf/png`",
        "- `Figure_3_MAZ_Boundary_Coupling.pdf`",
        "- `Figure_3_MAZ_Stable_Coupled_Profiles.pdf/png`",
        "- `Figure_4_CrossCancer_Morphological_Signature_Table.pdf/png`",
        "- `Supplementary_Figure_SpatialPermutation_Defense.pdf`",
        "",
        "## Nature-enhanced assets",
        "",
        "- `Nature_Enhanced_Assets/Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.svg/pdf/png`",
        "- `Nature_Enhanced_Assets/Figure_4_CrossCancer_Morphological_Signature_Table_Nature.svg/pdf/png`",
        "- `Nature_Enhanced_Assets/Main_Text_Full_Markdown_Reader.md`",
        "",
        "## Provenance",
        "",
        "Hashes are recorded in `Submission_File_Manifest.csv` and `Provenance/Technical_Manifest_File_Hashes.csv`.",
    ]
    (FINAL_DIR / "README_SUBMISSION_PACKAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = []
    for path in sorted(FINAL_DIR.rglob("*")):
        if path.is_file() and path.name != "Submission_File_Manifest.csv":
            records.append(
                {
                    "relative_path": str(path.relative_to(FINAL_DIR)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(records)
    manifest.to_csv(FINAL_DIR / "Submission_File_Manifest.csv", index=False)
    write_final_index()


def write_final_index() -> None:
    important = [
        DEFENSE_DIR / "Supplementary_Information_v1.md",
        DEFENSE_DIR / "FINAL_REVIEW_GUIDE_FOR_USER.md",
        DEFENSE_DIR / "Supplementary_Information_v1.md",
        DEFENSE_DIR / "Deep_Clean_Report.md",
        DEFENSE_DIR / "A100_ReadOnly_Cleanup_Check.md",
        DEFENSE_DIR / "Submission_Readiness_Status.md",
        DEFENSE_DIR / "Figure1_mTM_Framework.pdf",
        DEFENSE_DIR / "Figure1_mTM_Framework.png",
        FINAL_DIR / "Main_Text.md",
        FINAL_DIR / "Cover_Letter.md",
        FINAL_DIR / "Response_to_Reviewers.md",
        FINAL_DIR / "Supplementary_Information.md",
        FINAL_DIR / "Figure_1_mTM_Framework.pdf",
        FINAL_DIR / "Figure_2_Breast_S3_Hidden_Continua.pdf",
        FINAL_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs.png",
        FINAL_DIR / "Figure_3_MAZ_Boundary_Coupling.pdf",
        FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.png",
        FINAL_DIR / "Figure_4_CrossCancer_Morphological_Signature_Table.pdf",
        FINAL_DIR / "Submission_File_Manifest.csv",
    ]
    lines = ["# Final Deliverables Index", "", f"Updated UTC: {utc_now()}", ""]
    for path in important:
        lines.append(f"- `{path}`: {'present' if path.exists() else 'missing'}")
    (DEFENSE_DIR / "Final_Deliverables_Index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def conservative_deep_clean() -> None:
    """Remove only redundant generated snapshots/log handles, not scientific evidence."""
    cleanup_records: list[dict[str, str | int]] = []
    cleanup_dirs = [DEFENSE_DIR, PACKAGE_ROOT / "autopilot_20260511"]
    for directory in cleanup_dirs:
        if not directory.exists():
            continue
        manifests = sorted(directory.glob("Package_File_Manifest_*.csv"), key=lambda p: p.stat().st_mtime)
        for path in manifests[:-1]:
            cleanup_records.append(
                {
                    "action": "deleted_redundant_manifest_snapshot",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
            )
            path.unlink()

    for path in [DEFENSE_DIR / "local_supervisor.pid", DEFENSE_DIR / "local_supervisor_stdout.log", DEFENSE_DIR / "local_supervisor_stderr.log"]:
        if path.exists():
            cleanup_records.append({"action": "deleted_stopped_supervisor_runtime_file", "path": str(path), "bytes": path.stat().st_size})
            path.unlink()

    remote_note = "Remote A100 cleanup was not destructive from this local packager. The final package no longer depends on remote temporary files."
    report_lines = [
        "# Deep Clean Report",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        "Cleanup policy: only generated manifest snapshots and stopped-supervisor runtime files were deleted. Scientific CSV/PDF/PNG/HTML/Markdown evidence was preserved.",
        "",
        f"Remote note: {remote_note}",
        "",
        "## Local cleanup actions",
        "",
    ]
    if cleanup_records:
        for record in cleanup_records:
            report_lines.append(f"- {record['action']}: `{record['path']}` ({record['bytes']} bytes)")
    else:
        report_lines.append("- No redundant local cleanup candidates were present.")
    report_lines.extend(
        [
            "",
            "## Preserved evidence",
            "",
            "- Final submission directory",
            "- Defense package CSV/PDF/PNG/HTML/Markdown artifacts",
            "- `LOG_FOR_BOSS.md`, `Autopilot_Decision_Log.txt`, and final state JSON",
        ]
    )
    (DEFENSE_DIR / "Deep_Clean_Report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    copy_file(DEFENSE_DIR / "Deep_Clean_Report.md", PROVENANCE_DIR / "Deep_Clean_Report.md")


def git_status_summary() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {exc}"


def main() -> None:
    DEFENSE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    create_figure1()
    create_maz_summary_figure()
    create_supplementary_information()
    create_review_guide()
    copy_submission_files()
    write_submission_manifest()
    conservative_deep_clean()
    write_submission_manifest()
    status_path = DEFENSE_DIR / "Submission_Readiness_Status.md"
    status_path.write_text(
        "\n".join(
            [
                "# Submission Readiness Status",
                "",
                f"Generated UTC: {utc_now()}",
                "",
                "Final NBT submission package generated.",
                "",
                f"Final package: `{FINAL_DIR}`",
                "",
                "Git status summary:",
                "",
                "```text",
                git_status_summary(),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_final_index()


if __name__ == "__main__":
    main()
