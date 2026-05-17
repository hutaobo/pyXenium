#!/usr/bin/env python3
"""Build a reviewer-facing processed-data archive for the mTM/NBT package.

The archive is designed for DOI deposition. It includes the manuscript upload
package, source-data CSVs, sanitized processed summary tables, public provenance
metadata and scripts needed to validate the reported main numbers. It does not
include raw 10x Genomics Atera WTA/H&E files or remote runtime intermediates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    REPO_ROOT
    / "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
    "naturebiotech_package"
)
FINAL_DIR = PACKAGE_ROOT / "FINAL_SUBMISSION_NBT_20260513"
UPLOAD_DIR = PACKAGE_ROOT / "NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE"
SOURCE_TABLES_DIR = FINAL_DIR / "Source_Tables"
RUN_MANIFEST = (
    REPO_ROOT
    / "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
    "run_manifest.json"
)
DEFAULT_ARCHIVE_DIR = (
    REPO_ROOT
    / "manuscript/mtm_wta_nbt_replication/processed_data_archive_20260516"
)

SOURCE_DATA_FILES = [
    "Figure_1b_Hero_Patches_Source_Data.csv",
    "Figure_1c_Spatial_Permutation_Source_Data.csv",
    "Figure_1c_BlockBootstrap_Source_Data.csv",
    "Figure_1d_MAZ_QC_Source_Data.csv",
    "Figure_1e_CrossCancer_Signature_Source_Data.csv",
    "Supplementary_Table_5_SpatialSensitivity_Source_Data.csv",
    "Supplementary_Table_6_GeneComponent_Summary_Source_Data.csv",
    "Supplementary_Table_6_GeneComponent_Long_Source_Data.csv",
    "Supplementary_Table_7_RegistrationPerturbation_Summary_Source_Data.csv",
    "Supplementary_Table_7_RegistrationPerturbation_Long_Source_Data.csv",
    "Supplementary_Table_8_NestedSpatialHoldout_Summary_Source_Data.csv",
    "Supplementary_Table_8_NestedSpatialHoldout_Long_Source_Data.csv",
]

PROCESSED_TABLE_FILES = [
    "Spatial_Permutation_Defense_Report.csv",
    "Spatial_BlockBootstrap_CI.csv",
    "CrossCancer_Morphological_Signature_Table.csv",
    "CrossCancer_Morphomolecular_Dictionary.csv",
    "MAZ_QC_Table_v2.csv",
    "Figure2_Luminal_ER_HeroPatch_4Pairs.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sanitize_cell(cell: str) -> str:
    value = str(cell)
    if ":\\" in value or "\\naturebiotech_package\\" in value:
        name = PureWindowsPath(value).name
        return f"source_not_redistributed/{name}" if name else "source_not_redistributed"
    if "/data/taobo.hu/" in value:
        name = Path(value).name
        return f"remote_path_redacted/{name}" if name else "remote_path_redacted"
    if "/mnt/taobo.hu/" in value:
        name = Path(value).name
        return f"raw_input_path_redacted/{name}" if name else "raw_input_path_redacted"
    return value


def sanitize_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8") as in_handle:
        rows = list(csv.reader(in_handle))
    with dst.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerows([[sanitize_cell(cell) for cell in row] for row in rows])


def copy_upload_package(out_dir: Path, *, include_binaries: bool) -> None:
    dst_root = out_dir / "upload_package"
    dst_root.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {".md", ".csv", ".txt"}
    if include_binaries:
        allowed_suffixes |= {".docx", ".pdf", ".png", ".svg"}

    for src in UPLOAD_DIR.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in allowed_suffixes:
            continue
        rel = src.relative_to(UPLOAD_DIR)
        copy_file(src, dst_root / rel)


def copy_source_data(out_dir: Path) -> None:
    for name in SOURCE_DATA_FILES:
        copy_file(UPLOAD_DIR / "Source_Data" / name, out_dir / "source_data" / name)


def copy_processed_tables(out_dir: Path) -> None:
    for name in PROCESSED_TABLE_FILES:
        sanitize_csv(SOURCE_TABLES_DIR / name, out_dir / "processed_tables" / name)


def public_run_manifest() -> dict[str, object]:
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    gpu = manifest.get("gpu", {})
    config = manifest.get("config", {})
    outputs = manifest.get("outputs", {})
    model_status = manifest.get("model_status", {})
    return {
        "schema": "mtm_wta_public_run_manifest_v1",
        "source": "sanitized local run_manifest.json",
        "sample_context": "10x Genomics Atera WTA preview breast cancer sample",
        "model": config.get("model"),
        "wsi_reader": config.get("wsi_reader"),
        "tile_px": config.get("tile_px"),
        "mpp": config.get("mpp"),
        "program_library": config.get("program_library"),
        "gpu_available": bool(gpu.get("available")),
        "gpu_device_count": len(gpu.get("devices", [])),
        "gpu_hardware_class": "remote NVIDIA GPU",
        "model_status": {
            "status": model_status.get("status"),
            "n_tiles": model_status.get("n_tiles"),
            "wsi_source": model_status.get("wsi_source"),
        },
        "outputs": {
            "n_tiles": outputs.get("n_tiles"),
            "n_assigned_tiles": outputs.get("n_assigned_tiles"),
            "assignment_fraction": outputs.get("assignment_fraction"),
            "n_contours": outputs.get("n_contours"),
        },
        "runtime_seconds": manifest.get("runtime_seconds"),
        "private_paths_removed": True,
    }


def write_text_files(out_dir: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# mTM WTA processed-data archive",
                "",
                f"Generated: {now}",
                "",
                "This archive supports reviewer-side validation of the one-figure",
                "Nature Biotechnology initial-submission package for mTM/Atera WTA.",
                "",
                "Contents:",
                "",
                "- `upload_package/`: manuscript, methods, supplementary information,",
                "  cover letter and figure/source-data files from the upload package.",
                "- `source_data/`: quantitative CSVs used for Fig. 1 and Supplementary",
                "  Figs. 1-3.",
                "- `processed_tables/`: sanitized processed summary tables used to",
                "  generate source data and supplementary tables.",
                "- `provenance/`: public provenance metadata with private compute/storage and",
                "  local filesystem paths removed.",
                "- `scripts/recompute_main_numbers.py`: validation script for the main",
                "  reported numbers and source-data/SI consistency.",
                "",
                "Raw 10x Genomics Atera WTA/H&E files and remote intermediate runtime",
                "outputs are not redistributed here. The raw input files should be",
                "obtained from the original public/vendor dataset pages.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    checklist = {
        "archive_schema": "mtm_wta_processed_data_archive_v1",
        "generated_utc": now,
        "expected_source_data_files": SOURCE_DATA_FILES,
        "expected_values": {
            "breast_s3_luminal_estrogen_response_partial_rho": -0.639,
            "breast_s3_unfolded_protein_response_partial_rho": 0.515,
            "breast_s3_oxidative_phosphorylation_partial_rho": 0.531,
            "spatial_permutation_empirical_p": 9.999e-05,
        },
        "numeric_tolerances": {
            "reported_partial_rho_rounding_decimals": 3,
            "empirical_p_abs_tolerance": 1e-8,
        },
        "privacy_policy": {
            "raw_10x_inputs_redistributed": False,
            "remote_gpu_intermediates_redistributed": False,
            "private_paths_removed_from_public_provenance": True,
        },
    }
    (out_dir / "reproducibility_checklist.json").write_text(
        json.dumps(checklist, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provenance_dir = out_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "public_run_manifest_breast_plip.json").write_text(
        json.dumps(public_run_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (provenance_dir / "public_provenance_summary.md").write_text(
        "\n".join(
            [
                "# Public provenance summary",
                "",
                "The manuscript-facing result tables were generated from paired",
                "Atera WTA and H&E whole-slide image analysis. HistoSeg contours",
                "were derived from spatial-transcriptomics cell-coordinate and",
                "cluster information, not from H&E segmentation. H&E foundation-model",
                "embeddings were assigned to the same contour geometry and tested",
                "against residual WTA program variation.",
                "",
                "This public archive intentionally excludes raw vendor input files,",
                "local filesystem paths, remote runtime directories and storage",
                "paths. Those locations are internal provenance only and are not",
                "required for reviewer-side validation of the deposited source-data",
                "tables.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(out_dir: Path) -> None:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if rel in {"MANIFEST.tsv", "checksums.sha256"}:
            continue
        rows.append((rel, path.stat().st_size, sha256(path)))

    with (out_dir / "MANIFEST.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(rows)

    with (out_dir / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for rel, _size, digest in rows:
            handle.write(f"{digest}  {rel}\n")

    summary = {
        "file_count": len(rows),
        "total_bytes": sum(size for _rel, size, _digest in rows),
        "manifest_sha256": sha256(out_dir / "MANIFEST.tsv"),
    }
    (out_dir / "archive_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_zip(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    return zip_path


def build_archive(out_dir: Path, *, include_binaries: bool, make_zip: bool) -> Path:
    clean_dir(out_dir)
    copy_upload_package(out_dir, include_binaries=include_binaries)
    copy_source_data(out_dir)
    copy_processed_tables(out_dir)
    scripts_dir = out_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    copy_file(Path(__file__).with_name("recompute_main_numbers.py"), scripts_dir / "recompute_main_numbers.py")
    write_text_files(out_dir)
    write_manifest(out_dir)
    if make_zip:
        create_zip(out_dir)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument(
        "--skip-binaries",
        action="store_true",
        help="Only copy text/CSV files from the upload package.",
    )
    parser.add_argument("--no-zip", action="store_true", help="Do not create a .zip archive.")
    args = parser.parse_args()

    out_dir = build_archive(
        args.out_dir.resolve(),
        include_binaries=not args.skip_binaries,
        make_zip=not args.no_zip,
    )
    print(f"Built processed-data archive: {out_dir}")
    if not args.no_zip:
        print(f"Built zip archive: {out_dir.with_suffix('.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
