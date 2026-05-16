#!/usr/bin/env python3
"""Validate main mTM/NBT numbers from a processed-data archive.

This script intentionally works from deposited source-data tables. It does not
require raw 10x Genomics files, remote compute access or private runtime
directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_DIR = (
    REPO_ROOT
    / "manuscript/mtm_wta_nbt_replication/processed_data_archive_20260516"
)
FALLBACK_UPLOAD_DIR = (
    REPO_ROOT
    / "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
    "naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE"
)

EXPECTED_SOURCE_DATA = [
    "Figure_1b_Hero_Patches_Source_Data.csv",
    "Figure_1c_Spatial_Permutation_Source_Data.csv",
    "Figure_1c_BlockBootstrap_Source_Data.csv",
    "Figure_1d_MAZ_QC_Source_Data.csv",
    "Figure_1e_CrossCancer_Signature_Source_Data.csv",
]

EXPECTED_MAIN_VALUES = {
    ("breast", "plip", "luminal_estrogen_response"): {
        "reported_partial_rho_rounded": -0.639,
        "permutation_empirical_p": 9.999e-05,
        "n_contours": 157,
    },
    ("breast", "plip", "unfolded_protein_response"): {
        "reported_partial_rho_rounded": 0.515,
        "permutation_empirical_p": 9.999e-05,
        "n_contours": 157,
    },
    ("breast", "plip", "oxidative_phosphorylation"): {
        "reported_partial_rho_rounded": 0.531,
        "permutation_empirical_p": 9.999e-05,
        "n_contours": 157,
    },
}


class CheckError(RuntimeError):
    """Raised when a validation check fails."""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CheckError(f"Missing file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_data_dir(archive_dir: Path) -> Path:
    if (archive_dir / "source_data").is_dir():
        return archive_dir / "source_data"
    if (archive_dir / "Source_Data").is_dir():
        return archive_dir / "Source_Data"
    if (FALLBACK_UPLOAD_DIR / "Source_Data").is_dir():
        return FALLBACK_UPLOAD_DIR / "Source_Data"
    raise CheckError("Could not locate source-data directory.")


def supplementary_path(archive_dir: Path) -> Path | None:
    candidates = [
        archive_dir / "upload_package" / "Supplementary_Information.md",
        archive_dir / "Supplementary_Information.md",
        FALLBACK_UPLOAD_DIR / "Supplementary_Information.md",
    ]
    return next((path for path in candidates if path.exists()), None)


def table_after(markdown: str, heading: str) -> list[list[str]]:
    if heading not in markdown:
        raise CheckError(f"Missing SI heading: {heading}")
    start = markdown.index(heading)
    nxt = markdown.find("\n### ", start + len(heading))
    end = nxt if nxt != -1 else len(markdown)
    lines = [line for line in markdown[start:end].splitlines() if line.startswith("|")]
    if len(lines) < 2:
        raise CheckError(f"No markdown table after {heading}")
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def compare_source_data_to_si(source_dir: Path, si_path: Path) -> dict[str, int]:
    si = si_path.read_text(encoding="utf-8")
    mismatches: list[str] = []

    perm = read_csv(source_dir / "Figure_1c_Spatial_Permutation_Source_Data.csv")
    t1 = table_after(si, "### Supplementary Table 1")
    if len(t1) != len(perm):
        mismatches.append(f"Table 1 row count {len(t1)} != permutation CSV {len(perm)}")
    else:
        for cells, row in zip(t1, perm):
            expected_rho = f"{float(row['reported_partial_rho']):.3f}"
            if cells[0].lower() != row["dataset"] or cells[1].lower() != row["model"]:
                mismatches.append(f"Table 1 identity mismatch for {row['program']}")
            if cells[2] != row["program"] or cells[5] != expected_rho:
                mismatches.append(f"Table 1 value mismatch for {row['program']}")

    boot = read_csv(source_dir / "Figure_1c_BlockBootstrap_Source_Data.csv")
    t2 = table_after(si, "### Supplementary Table 2")
    if len(t2) != len(boot):
        mismatches.append(f"Table 2 row count {len(t2)} != bootstrap CSV {len(boot)}")
    else:
        for cells, row in zip(t2, boot):
            ci = f"{float(row['bootstrap_rho_ci_low']):.3f} to {float(row['bootstrap_rho_ci_high']):.3f}"
            if cells[4] != f"{float(row['bootstrap_rho_median']):.3f}" or cells[5] != ci:
                mismatches.append(f"Table 2 value mismatch for {row['program']}")

    sig = read_csv(source_dir / "Figure_1e_CrossCancer_Signature_Source_Data.csv")
    t3 = table_after(si, "### Supplementary Table 3")
    if len(t3) != len(sig):
        mismatches.append(f"Table 3 row count {len(t3)} != signature CSV {len(sig)}")
    else:
        for cells, row in zip(t3, sig):
            if cells[0].lower() != row["dataset"] or cells[1] != row["program_family"]:
                mismatches.append(f"Table 3 identity mismatch for {row}")
            if row["max_abs_partial_rho"]:
                if cells[3] != f"{float(row['max_abs_partial_rho']):.3f}":
                    mismatches.append(f"Table 3 rho mismatch for {row['dataset']}/{row['program_family']}")
            elif cells[3] not in {"NA", "not_detected", "not detected", ""}:
                mismatches.append(f"Table 3 missing-value mismatch for {row['dataset']}/{row['program_family']}")
            if cells[4] != row["top_pathway"] or cells[5] != row["support_call"]:
                mismatches.append(f"Table 3 label mismatch for {row['dataset']}/{row['program_family']}")

    hero = read_csv(source_dir / "Figure_1b_Hero_Patches_Source_Data.csv")
    t4 = table_after(si, "### Supplementary Table 4")
    if len(t4) != len(hero):
        mismatches.append(f"Table 4 row count {len(t4)} != hero CSV {len(hero)}")
    else:
        for cells, row in zip(t4, hero):
            if cells[2] != row["contour_id"]:
                mismatches.append(f"Table 4 identity mismatch for {row['contour_id']}")
            if cells[3] != f"{float(row['wta_program_z']):.2f}":
                mismatches.append(f"Table 4 WTA z mismatch for {row['contour_id']}")
            if cells[4] != f"{float(row['oriented_he_embedding_z']):.2f}":
                mismatches.append(f"Table 4 H&E z mismatch for {row['contour_id']}")

    if mismatches:
        raise CheckError("; ".join(mismatches))
    return {
        "supplementary_table_1_rows": len(perm),
        "supplementary_table_2_rows": len(boot),
        "supplementary_table_3_rows": len(sig),
        "supplementary_table_4_rows": len(hero),
    }


def check_private_paths(archive_dir: Path) -> int:
    patterns = [
        re.compile(r"[A-Za-z]:\\"),
        re.compile(r"/data/taobo\.hu/"),
        re.compile(r"/mnt/taobo\.hu/"),
        re.compile(r"sscb-a100\.scilifelab\.se"),
    ]
    offenders = []
    for path in archive_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json", ".txt", ".tsv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in patterns):
            offenders.append(path.relative_to(archive_dir).as_posix())
    if offenders:
        raise CheckError("Private/internal path tokens found in archive: " + ", ".join(offenders))
    return 0


def validate(archive_dir: Path, *, check_privacy: bool) -> dict[str, object]:
    source_dir = source_data_dir(archive_dir)
    present = sorted(path.name for path in source_dir.glob("*_Source_Data.csv"))
    if present != sorted(EXPECTED_SOURCE_DATA):
        raise CheckError(f"Unexpected source-data files: {present}")

    perm_rows = read_csv(source_dir / "Figure_1c_Spatial_Permutation_Source_Data.csv")
    lookup = {(row["dataset"], row["model"], row["program"]): row for row in perm_rows}
    main_results = {}
    for key, expected in EXPECTED_MAIN_VALUES.items():
        row = lookup.get(key)
        if row is None:
            raise CheckError(f"Missing permutation row: {key}")
        observed_rho = float(row["reported_partial_rho"])
        observed_p = float(row["permutation_empirical_p"])
        observed_n = int(row["n_contours"])
        if round(observed_rho, 3) != expected["reported_partial_rho_rounded"]:
            raise CheckError(f"rho mismatch for {key}: {observed_rho}")
        if abs(observed_p - expected["permutation_empirical_p"]) > 1e-8:
            raise CheckError(f"empirical P mismatch for {key}: {observed_p}")
        if observed_n != expected["n_contours"]:
            raise CheckError(f"n_contours mismatch for {key}: {observed_n}")
        main_results["/".join(key)] = {
            "reported_partial_rho": observed_rho,
            "reported_partial_rho_rounded": round(observed_rho, 3),
            "recomputed_partial_rho": float(row["recomputed_partial_rho"]),
            "reported_minus_recomputed": float(row["reported_minus_recomputed"]),
            "permutation_empirical_p": observed_p,
            "n_contours": observed_n,
        }

    si_path = supplementary_path(archive_dir)
    si_summary = {}
    if si_path is not None:
        si_summary = compare_source_data_to_si(source_dir, si_path)

    privacy_offenders = None
    if check_privacy and archive_dir.exists():
        privacy_offenders = check_private_paths(archive_dir)

    return {
        "status": "pass",
        "archive_dir": str(archive_dir),
        "source_data_dir": str(source_dir),
        "main_results": main_results,
        "si_consistency": si_summary,
        "privacy_offender_count": privacy_offenders,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--write-report", type=Path, default=None)
    parser.add_argument(
        "--skip-privacy-check",
        action="store_true",
        help="Skip archive privacy-token scan.",
    )
    args = parser.parse_args()

    report = validate(args.archive_dir.resolve(), check_privacy=not args.skip_privacy_check)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckError as exc:
        print(f"CHECK FAILED: {exc}")
        raise SystemExit(1)
