"""Prepare a clean Nature Biotechnology initial-submission upload directory."""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


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
NATURE_DIR = FINAL_DIR / "Nature_Enhanced_Assets"
COMPOSITE_FIGURE_BASE = NATURE_DIR / "Figure_1_mTM_NBT_Brief_Composite"
UPLOAD_DIR = PACKAGE_ROOT / "NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE"
ADMIN_DIR = PACKAGE_ROOT / "NBT_INTERNAL_ADMIN_20260515"
FIGURE_DIR = UPLOAD_DIR / "Figures"
SOURCE_DATA_DIR = UPLOAD_DIR / "Source_Data"
SUPPLEMENTARY_DIR = UPLOAD_DIR / "Supplementary_Files"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def reset_upload_dir() -> None:
    resolved = UPLOAD_DIR.resolve()
    allowed_parent = PACKAGE_ROOT.resolve()
    if allowed_parent not in resolved.parents or not resolved.name.startswith("NBT_INITIAL_SUBMISSION_UPLOAD_"):
        raise RuntimeError(f"Refusing to reset unexpected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    FIGURE_DIR.mkdir(parents=True)
    SOURCE_DATA_DIR.mkdir(parents=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True)


def reset_admin_dir() -> None:
    resolved = ADMIN_DIR.resolve()
    allowed_parent = PACKAGE_ROOT.resolve()
    if allowed_parent not in resolved.parents or not resolved.name.startswith("NBT_INTERNAL_ADMIN_"):
        raise RuntimeError(f"Refusing to reset unexpected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    ADMIN_DIR.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_source_csv(
    src: Path,
    dst: Path,
    *,
    column_renames: dict[str, str] | None = None,
    value_replacements: dict[str, dict[str, str]] | None = None,
    basename_columns: set[str] | None = None,
) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    column_renames = column_renames or {}
    value_replacements = value_replacements or {}
    basename_columns = basename_columns or set()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {src}")
        fieldnames = [column_renames.get(name, name) for name in reader.fieldnames]
        with dst.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                cleaned: dict[str, str] = {}
                for old_name, value in row.items():
                    new_name = column_renames.get(old_name, old_name)
                    if old_name in basename_columns and value:
                        value = Path(value).name
                    if old_name in value_replacements:
                        value = value_replacements[old_name].get(value, value)
                    cleaned[new_name] = value
                writer.writerow(cleaned)


def extract_figure_legends(main_text: str) -> tuple[str, dict[str, str]]:
    marker_match = re.search(r"^## Figure legends?\s*$", main_text, flags=re.MULTILINE)
    if marker_match is None:
        return main_text, {}
    body = main_text[: marker_match.start()]
    legend_block = main_text[marker_match.end() :]
    matches = list(re.finditer(r"^### Figure ([1-4])\s*$", legend_block, flags=re.MULTILINE))
    legends: dict[str, str] = {}
    for idx, match in enumerate(matches):
        figure_id = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(legend_block)
        legends[figure_id] = legend_block[start:end].strip()
    return body.rstrip(), legends


def copy_submission_figures() -> None:
    composite_files = [
        (COMPOSITE_FIGURE_BASE.with_suffix(".svg"), FIGURE_DIR / "Figure_1.svg"),
        (COMPOSITE_FIGURE_BASE.with_suffix(".pdf"), FIGURE_DIR / "Figure_1.pdf"),
        (COMPOSITE_FIGURE_BASE.with_suffix(".png"), FIGURE_DIR / "Figure_1.png"),
    ]
    if all(src.exists() for src, _dst in composite_files):
        copies = composite_files
    else:
        copies = [
            (FINAL_DIR / "Figure_1_mTM_Framework.pdf", FIGURE_DIR / "Figure_1.pdf"),
            (FINAL_DIR / "Figure_1_mTM_Framework.png", FIGURE_DIR / "Figure_1.png"),
        ]
    supplementary_copies = [
        (
            FINAL_DIR / "Supplementary_Figure_SpatialPermutation_Defense.pdf",
            SUPPLEMENTARY_DIR / "Supplementary_Figure_1_SpatialPermutation_Defense.pdf",
        ),
        (
            NATURE_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.svg",
            SUPPLEMENTARY_DIR / "Supplementary_Figure_2_HeroPatch_Examples.svg",
        ),
        (
            NATURE_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.pdf",
            SUPPLEMENTARY_DIR / "Supplementary_Figure_2_HeroPatch_Examples.pdf",
        ),
        (
            NATURE_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.png",
            SUPPLEMENTARY_DIR / "Supplementary_Figure_2_HeroPatch_Examples.png",
        ),
        (
            FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.pdf",
            SUPPLEMENTARY_DIR / "Supplementary_Figure_3_MAZ.pdf",
        ),
        (
            FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.png",
            SUPPLEMENTARY_DIR / "Supplementary_Figure_3_MAZ.png",
        ),
    ]
    for src, dst in [*copies, *supplementary_copies]:
        copy_file(src, dst)


def copy_legacy_submission_figures() -> None:
    copies = [
        (FINAL_DIR / "Figure_1_mTM_Framework.pdf", FIGURE_DIR / "Figure_1.pdf"),
        (FINAL_DIR / "Figure_1_mTM_Framework.png", FIGURE_DIR / "Figure_1.png"),
        (NATURE_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.svg", FIGURE_DIR / "Figure_2.svg"),
        (NATURE_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.pdf", FIGURE_DIR / "Figure_2.pdf"),
        (NATURE_DIR / "Figure_2_Luminal_ER_HeroPatch_4Pairs_Nature.png", FIGURE_DIR / "Figure_2.png"),
        (FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.pdf", FIGURE_DIR / "Figure_3.pdf"),
        (FINAL_DIR / "Figure_3_MAZ_Stable_Coupled_Profiles.png", FIGURE_DIR / "Figure_3.png"),
        (NATURE_DIR / "Figure_4_CrossCancer_Morphological_Signature_Table_Nature.svg", FIGURE_DIR / "Figure_4.svg"),
        (NATURE_DIR / "Figure_4_CrossCancer_Morphological_Signature_Table_Nature.pdf", FIGURE_DIR / "Figure_4.pdf"),
        (NATURE_DIR / "Figure_4_CrossCancer_Morphological_Signature_Table_Nature.png", FIGURE_DIR / "Figure_4.png"),
    ]
    for src, dst in copies:
        copy_file(src, dst)


def copy_source_data() -> None:
    copy_source_csv(
        FINAL_DIR / "Source_Tables" / "Figure2_Luminal_ER_HeroPatch_4Pairs.csv",
        SOURCE_DATA_DIR / "Figure_2_Hero_Patches_Source_Data.csv",
        column_renames={"patch_file": "patch_image"},
        basename_columns={"patch_file"},
    )
    copy_source_csv(
        FINAL_DIR / "Source_Tables" / "CrossCancer_Morphological_Signature_Table.csv",
        SOURCE_DATA_DIR / "Figure_4_CrossCancer_Signature_Source_Data.csv",
    )
    copy_source_csv(
        FINAL_DIR / "Source_Tables" / "MAZ_QC_Table_v2.csv",
        SOURCE_DATA_DIR / "Figure_3_MAZ_QC_Source_Data.csv",
        column_renames={"lead_lag_class": "boundary_profile_class"},
        value_replacements={
            "lead_lag_class": {
                "molecular_lead": "offset_boundary_zone",
                "morphology_lead": "offset_boundary_zone",
            }
        },
    )
    copy_source_csv(
        FINAL_DIR / "Source_Tables" / "Spatial_Permutation_Defense_Report.csv",
        SOURCE_DATA_DIR / "Supplementary_Spatial_Permutation_Source_Data.csv",
    )
    copy_source_csv(
        FINAL_DIR / "Source_Tables" / "Spatial_BlockBootstrap_CI.csv",
        SOURCE_DATA_DIR / "Supplementary_BlockBootstrap_Source_Data.csv",
    )


def make_manuscript_markdown() -> Path:
    main_text = (FINAL_DIR / "Main_Text.md").read_text(encoding="utf-8")
    body, legends = extract_figure_legends(main_text)
    composite_legend = legends.get("1", "")
    lines = [
        body,
        "",
        "## Figure legend",
        "",
        composite_legend,
        "",
    ]
    path = UPLOAD_DIR / "Manuscript_Initial_Submission.md"
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def convert_with_pandoc(src: Path, dst: Path) -> bool:
    try:
        subprocess.run(["pandoc", str(src), "-o", str(dst)], cwd=UPLOAD_DIR, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return dst.exists()


def prepare_documents() -> None:
    copy_file(FINAL_DIR / "Cover_Letter.md", UPLOAD_DIR / "Cover_Letter.md")
    copy_file(FINAL_DIR / "Online_Methods.md", UPLOAD_DIR / "Online_Methods.md")
    copy_file(FINAL_DIR / "Supplementary_Information.md", UPLOAD_DIR / "Supplementary_Information.md")
    copy_file(FINAL_DIR / "Response_to_Reviewers.md", ADMIN_DIR / "Response_to_Likely_Reviewers_INTERNAL_NOT_FOR_INITIAL_UPLOAD.md")
    copy_file(FINAL_DIR / "Interactive_Hero_Contour_Browser.html", ADMIN_DIR / "Interactive_Hero_Contour_Browser_OPTIONAL.html")
    manuscript_md = make_manuscript_markdown()
    conversions = [
        (manuscript_md, UPLOAD_DIR / "Manuscript_Initial_Submission.docx"),
        (UPLOAD_DIR / "Online_Methods.md", UPLOAD_DIR / "Online_Methods.docx"),
        (UPLOAD_DIR / "Cover_Letter.md", UPLOAD_DIR / "Cover_Letter.docx"),
        (UPLOAD_DIR / "Supplementary_Information.md", UPLOAD_DIR / "Supplementary_Information.docx"),
    ]
    conversion_lines = []
    for src, dst in conversions:
        status = "created" if convert_with_pandoc(src, dst) else "not_created"
        conversion_lines.append(f"- `{dst.name}`: {status}")
    (ADMIN_DIR / "Pandoc_Conversion_Report.md").write_text("\n".join(["# Pandoc Conversion Report", "", *conversion_lines, ""]), encoding="utf-8")


def write_admin_readme() -> None:
    lines = [
        "# NBT Initial Submission Upload Package",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        "## Upload first",
        "",
        "- `Manuscript_Initial_Submission.docx`: main text plus embedded title-free figures and captions.",
        "- `Online_Methods.docx`: online methods.",
        "- `Cover_Letter.docx`: cover letter.",
        "- `Supplementary_Information.docx`: supplementary information.",
        "- `Source_Data/`: CSV source data for quantitative figure panels and statistical defense.",
        "",
        "## Upload if the submission system requests separate figure files",
        "",
        "- `Figures/Figure_1.svg`, `Figures/Figure_1.pdf`, and `Figures/Figure_1.png`: one composite main-text display item.",
        "- Legacy separate panels are in `Supplementary_Files/` for optional supplementary use, not as main-text display items.",
        "",
        "## Optional / internal",
        "",
        "- `Interactive_Hero_Contour_Browser_OPTIONAL.html` can be held back unless the journal accepts interactive supplementary files.",
        "- `Response_to_Likely_Reviewers_INTERNAL_NOT_FOR_INITIAL_UPLOAD.md` is an internal preparation file, not an initial-submission file.",
        "",
        "## What is deliberately not included",
        "",
        "- Hash manifests and provenance reports are not submission files. They remain in the working final package for internal reproducibility checks only.",
        "- Figure files do not contain standalone titles. Figure titles and interpretation are in the captions.",
        "",
        "## Guideline basis checked",
        "",
        "- Nature Biotechnology submission guidelines: https://www.nature.com/nbt/submission-guidelines",
        "- Nature Portfolio initial submission guidance: https://www.nature.com/nature/for-authors/initial-submission",
        "- Nature Research Figure Guide: https://research-figure-guide.nature.com/",
    ]
    (ADMIN_DIR / "UPLOAD_README_INTERNAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    reset_upload_dir()
    reset_admin_dir()
    copy_submission_figures()
    copy_source_data()
    prepare_documents()
    write_admin_readme()
    print(f"Prepared upload package: {UPLOAD_DIR}")


if __name__ == "__main__":
    main()
