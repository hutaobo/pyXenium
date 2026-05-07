from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon

from .xenium_artifacts import read_he_image
from .xenium_slide_builder import (
    DEFAULT_MAX_CROP_SIDE_PX,
    _crop_image_level,
    _extract_feature_patch,
    _geometry_xenium_pixel_to_image_xy,
    _select_pyramid_level,
    _slug,
    _to_rgb,
    load_contour_geojson,
)


DEFAULT_FLATTENED_SLIDE_ROOT = Path(r"D:\GitHub\stGPT\outputs\xenium_slides")
DEFAULT_SOURCE_CACHE_ROOT = Path(r"D:\GitHub\stGPT\outputs\xenium_source_cache\10x_public")
DEFAULT_ESTIMATED_PATCH_BYTES = 350_000
TENX_CDN_PATTERN = re.compile(r"https://cf\.10xgenomics\.com/[^\"'<> )]+")


@dataclass(frozen=True)
class BackfillInspectionResult:
    slide_root: Path
    output_dir: Path
    summary: dict[str, Any]
    backfill_plan: Path
    alignment_audit_plan: Path
    storage_estimate: Path
    readiness_summary: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_root": str(self.slide_root),
            "output_dir": str(self.output_dir),
            "summary": self.summary,
            "backfill_plan": str(self.backfill_plan),
            "alignment_audit_plan": str(self.alignment_audit_plan),
            "storage_estimate": str(self.storage_estimate),
            "readiness_summary": str(self.readiness_summary),
        }


@dataclass(frozen=True)
class L3UpgradeInspectionResult:
    slide_root: Path
    output_dir: Path
    summary: dict[str, Any]
    l3_upgrade_plan: Path
    source_asset_resolution_report: Path
    alignment_verdicts: Path
    training_manifest_l3: Path
    blocked_cases: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_root": str(self.slide_root),
            "output_dir": str(self.output_dir),
            "summary": self.summary,
            "l3_upgrade_plan": str(self.l3_upgrade_plan),
            "source_asset_resolution_report": str(self.source_asset_resolution_report),
            "alignment_verdicts": str(self.alignment_verdicts),
            "training_manifest_l3": str(self.training_manifest_l3),
            "blocked_cases": str(self.blocked_cases),
        }


def inspect_backfill_needs(
    slide_root: str | Path = DEFAULT_FLATTENED_SLIDE_ROOT,
    *,
    output_dir: str | Path | None = None,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    average_patch_bytes: int | None = None,
    update_registry: bool = True,
) -> BackfillInspectionResult:
    """Write readiness and contour-patch backfill plans for a XeniumSlide root."""

    root = Path(slide_root).expanduser()
    out = Path(output_dir).expanduser() if output_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    records, registry_path = _load_registry(root)
    patch_bytes = int(average_patch_bytes or _average_existing_patch_bytes(root))

    rows = [
        _build_plan_row(
            record,
            slide_root=root,
            average_patch_bytes=patch_bytes,
            max_crop_side_px=max_crop_side_px,
        )
        for record in records
        if bool(record.get("selected_for_build"))
    ]
    plan = pd.DataFrame(rows)
    plan_path = out / "backfill_plan.csv"
    plan_json_path = out / "backfill_plan.json"
    audit_path = out / "alignment_audit_plan.csv"
    storage_path = out / "storage_estimate.json"
    summary_path = out / "readiness_summary.json"

    plan.to_csv(plan_path, index=False)
    _write_json(plan_json_path, rows)
    audit_rows = _alignment_audit_rows(rows)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    storage = _storage_estimate(rows, average_patch_bytes=patch_bytes, max_crop_side_px=max_crop_side_px)
    _write_json(storage_path, storage)

    if update_registry:
        _update_registry_readiness(records, rows)
        _write_registry(registry_path, records)

    summary = {
        "slide_root": str(root),
        "selected_cases": int(len(rows)),
        "foundation_ready": int(sum(bool(row["foundation_ready"]) for row in rows)),
        "structure_ready": int(sum(bool(row["structure_ready"]) for row in rows)),
        "morphology_ready": int(sum(bool(row["morphology_ready"]) for row in rows)),
        "needs_contour_patch_extraction": int(
            sum(row["backfill_action"] == "extract_contour_patches" for row in rows)
        ),
        "missing_image_context": int(sum(row["backfill_action"] == "missing_image_context" for row in rows)),
        "missing_contours": int(sum(row["backfill_action"] == "generate_or_select_contours" for row in rows)),
        "average_patch_bytes": int(patch_bytes),
        "generated_at": _utc_now(),
    }
    _write_json(summary_path, summary)
    return BackfillInspectionResult(
        slide_root=root,
        output_dir=out,
        summary=summary,
        backfill_plan=plan_path,
        alignment_audit_plan=audit_path,
        storage_estimate=storage_path,
        readiness_summary=summary_path,
    )


def inspect_l3_upgrade(
    slide_root: str | Path = DEFAULT_FLATTENED_SLIDE_ROOT,
    *,
    output_dir: str | Path | None = None,
    source_cache_root: str | Path = DEFAULT_SOURCE_CACHE_ROOT,
    allow_network: bool = True,
    download_source_assets: bool = False,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    average_patch_bytes: int | None = None,
    update_registry: bool = True,
) -> L3UpgradeInspectionResult:
    """Inspect strict L3 morphology readiness and write auditable upgrade artifacts."""

    inspection = inspect_backfill_needs(
        slide_root,
        output_dir=output_dir,
        max_crop_side_px=max_crop_side_px,
        average_patch_bytes=average_patch_bytes,
        update_registry=update_registry,
    )
    root = inspection.slide_root
    out = inspection.output_dir
    source_cache = Path(source_cache_root).expanduser()
    rows = json.loads((out / "backfill_plan.json").read_text(encoding="utf-8"))
    source_rows = [
        _resolve_source_assets(
            row,
            source_cache_root=source_cache,
            allow_network=allow_network,
            download=download_source_assets,
        )
        for row in rows
    ]
    by_case = {row["case_leaf"]: row for row in source_rows}
    snapshot_failures = _load_snapshot_failures(out)
    l3_rows = [
        _l3_plan_row(row, by_case.get(row["case_leaf"], {}), snapshot_failures.get(str(row["case_leaf"])))
        for row in rows
    ]
    blocked_rows = [row for row in l3_rows if row["l3_status"] == "blocked"]
    training_rows = [_training_manifest_row(row) for row in l3_rows if row["morphology_ready"]]

    l3_plan_csv = out / "l3_upgrade_plan.csv"
    source_csv = out / "source_asset_resolution_report.csv"
    blocked_csv = out / "blocked_cases.csv"
    training_csv = out / "training_manifest_l3.csv"
    verdicts_csv = out / "alignment_verdicts.csv"
    summary_json = out / "l3_upgrade_summary.json"

    _write_table_and_json(l3_plan_csv, l3_rows)
    _write_table_and_json(source_csv, source_rows)
    _write_table_and_json(blocked_csv, blocked_rows)
    _write_table_and_json(training_csv, training_rows)
    _write_alignment_verdict_template(verdicts_csv, [row for row in l3_rows if row["requires_alignment_verdict"]])

    if update_registry:
        records, registry_path = _load_registry(root)
        _update_registry_l3(records, l3_rows)
        _write_registry(registry_path, records)

    summary = {
        "slide_root": str(root),
        "source_cache_root": str(source_cache),
        "selected_cases": int(len(l3_rows)),
        "morphology_ready": int(sum(bool(row["morphology_ready"]) for row in l3_rows)),
        "pending_alignment_verdict": int(sum(row["l3_status"] == "pending_alignment_verdict" for row in l3_rows)),
        "pending_alignment_snapshot": int(sum(row["l3_status"] == "pending_alignment_snapshot" for row in l3_rows)),
        "blocked": int(len(blocked_rows)),
        "training_manifest_cases": int(len(training_rows)),
        "download_source_assets": bool(download_source_assets),
        "generated_at": _utc_now(),
    }
    _write_json(summary_json, summary)
    return L3UpgradeInspectionResult(
        slide_root=root,
        output_dir=out,
        summary=summary,
        l3_upgrade_plan=l3_plan_csv,
        source_asset_resolution_report=source_csv,
        alignment_verdicts=verdicts_csv,
        training_manifest_l3=training_csv,
        blocked_cases=blocked_csv,
    )


def run_l3_upgrade(
    slide_root: str | Path = DEFAULT_FLATTENED_SLIDE_ROOT,
    *,
    output_dir: str | Path | None = None,
    source_cache_root: str | Path = DEFAULT_SOURCE_CACHE_ROOT,
    cases: Iterable[str] | None = None,
    audit_only: bool = True,
    workers: int = 1,
    overwrite: bool = False,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    max_snapshot_side_px: int = 1600,
    require_verdict_pass: bool = True,
    allow_network: bool = True,
    download_source_assets: bool = False,
    update_registry: bool = True,
) -> dict[str, Any]:
    """Run strict L3 snapshot generation or verdict-gated contour patch extraction."""

    l3 = inspect_l3_upgrade(
        slide_root,
        output_dir=output_dir,
        source_cache_root=source_cache_root,
        allow_network=allow_network,
        download_source_assets=download_source_assets,
        max_crop_side_px=max_crop_side_px,
        update_registry=update_registry,
    )
    payload = backfill_contour_patches(
        slide_root=slide_root,
        output_dir=l3.output_dir,
        cases=cases,
        audit_only=audit_only,
        workers=workers,
        overwrite=overwrite,
        max_crop_side_px=max_crop_side_px,
        max_snapshot_side_px=max_snapshot_side_px,
        update_registry=update_registry,
        alignment_verdicts=l3.alignment_verdicts,
        require_verdict_pass=require_verdict_pass,
    )
    refreshed = inspect_l3_upgrade(
        slide_root,
        output_dir=l3.output_dir,
        source_cache_root=source_cache_root,
        allow_network=False,
        download_source_assets=False,
        max_crop_side_px=max_crop_side_px,
        update_registry=update_registry,
    )
    payload["l3_inspection"] = refreshed.to_dict()
    payload["l3_summary"] = refreshed.summary
    return payload


def backfill_contour_patches(
    slide_root: str | Path = DEFAULT_FLATTENED_SLIDE_ROOT,
    *,
    output_dir: str | Path | None = None,
    cases: Iterable[str] | None = None,
    audit_only: bool = True,
    workers: int = 1,
    overwrite: bool = False,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    max_snapshot_side_px: int = 1600,
    update_registry: bool = True,
    alignment_verdicts: str | Path | None = None,
    require_verdict_pass: bool = True,
) -> dict[str, Any]:
    """Generate alignment snapshots, and optionally contour H&E patch manifests."""

    inspection = inspect_backfill_needs(
        slide_root,
        output_dir=output_dir,
        max_crop_side_px=max_crop_side_px,
        update_registry=update_registry,
    )
    plan_rows = json.loads((inspection.output_dir / "backfill_plan.json").read_text(encoding="utf-8"))
    case_filter = {str(case) for case in cases or []}
    candidates = [
        row
        for row in plan_rows
        if row.get("backfill_action") == "extract_contour_patches"
        and (not case_filter or row.get("case_name") in case_filter or row.get("case_leaf") in case_filter)
    ]
    verdicts = _load_alignment_verdicts(Path(alignment_verdicts)) if alignment_verdicts is not None else {}
    skipped_by_verdict: list[dict[str, Any]] = []
    if not audit_only and require_verdict_pass:
        approved = []
        for row in candidates:
            verdict = verdicts.get(str(row.get("case_leaf")), "")
            if verdict == "pass":
                approved.append(row)
            else:
                skipped_by_verdict.append(
                    {
                        "case_name": row.get("case_name"),
                        "case_leaf": row.get("case_leaf"),
                        "output_dir": row.get("output_dir"),
                        "status": "skipped_by_verdict",
                        "verdict": verdict or "missing",
                    }
                )
        candidates = approved

    results: list[dict[str, Any]] = list(skipped_by_verdict)
    if workers <= 1 or len(candidates) <= 1:
        for row in candidates:
            results.append(
                _run_backfill_case(
                    row,
                    audit_only=audit_only,
                    overwrite=overwrite,
                    max_crop_side_px=max_crop_side_px,
                    max_snapshot_side_px=max_snapshot_side_px,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
            futures = [
                executor.submit(
                    _run_backfill_case,
                    row,
                    audit_only=audit_only,
                    overwrite=overwrite,
                    max_crop_side_px=max_crop_side_px,
                    max_snapshot_side_px=max_snapshot_side_px,
                )
                for row in candidates
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results = sorted(results, key=lambda row: str(row.get("case_leaf") or row.get("case_name") or ""))
    results_path = inspection.output_dir / "backfill_results.csv"
    results_json_path = inspection.output_dir / "backfill_results.json"
    pd.DataFrame(results).to_csv(results_path, index=False)
    _write_json(results_json_path, results)

    if update_registry and not audit_only:
        inspection = inspect_backfill_needs(
            slide_root,
            output_dir=inspection.output_dir,
            max_crop_side_px=max_crop_side_px,
            update_registry=True,
        )

    return {
        "slide_root": str(inspection.slide_root),
        "audit_only": bool(audit_only),
        "candidate_cases": int(len(candidates)),
        "completed": int(
            sum(
                row.get("status")
                in {"snapshot_written", "patches_written", "patches_written_with_skips", "skipped_existing"}
                for row in results
            )
        ),
        "failed": int(sum(row.get("status") == "failed" for row in results)),
        "skipped_by_verdict": int(sum(row.get("status") == "skipped_by_verdict" for row in results)),
        "results_csv": str(results_path),
        "results_json": str(results_json_path),
        "inspection": inspection.to_dict(),
    }


def _build_plan_row(
    record: Mapping[str, Any],
    *,
    slide_root: Path,
    average_patch_bytes: int,
    max_crop_side_px: int,
) -> dict[str, Any]:
    output_dir = _case_output_dir(record, slide_root)
    manifest = _read_json(output_dir / "slide_manifest.json")
    qc_report = _read_json(output_dir / "qc_report.json")
    contour_source = _read_json(output_dir / "contour_source_manifest.json")
    patch_manifest_path = output_dir / "contour_patches_manifest.json"
    patches = _read_patch_manifest(patch_manifest_path)
    patch_image_count = _existing_patch_image_count(patches)
    missing_patch_images = _missing_patch_image_count(patches)
    counts = dict(manifest.get("counts", {}) if isinstance(manifest, dict) else {})
    cells = _safe_int(record.get("cells") or counts.get("cells"))
    genes = _safe_int(record.get("genes") or counts.get("genes"))
    contours = _safe_int(counts.get("contours") or record.get("contours"))
    assigned_cells = _safe_int(record.get("assigned_cells") or counts.get("assigned_cells"))
    contour_patches = _safe_int(record.get("contour_patches") or counts.get("contour_patches") or len(patches))
    source_geojson = (
        record.get("selected_geojson")
        or contour_source.get("selected_geojson")
        or manifest.get("contours", {}).get("source_geojson")
    )
    image_artifacts = manifest.get("image_artifacts", {}) if isinstance(manifest, dict) else {}
    he_artifact = image_artifacts.get("he", {}) if isinstance(image_artifacts, dict) else {}
    has_he = bool(record.get("has_he") or he_artifact.get("path"))
    has_alignment = bool(record.get("has_alignment") or he_artifact.get("image_to_xenium_affine"))
    qc_status = str(record.get("qc_status") or qc_report.get("status") or "")
    foundation_ready = bool((output_dir / "xenium_slide.zarr").exists() and qc_status == "pass" and cells > 0 and genes > 0)
    structure_ready = bool(foundation_ready and assigned_cells > 0 and source_geojson)
    morphology_ready = bool(structure_ready and contour_patches > 0 and patch_image_count > 0 and missing_patch_images == 0)
    readiness_level = _readiness_level(foundation_ready, structure_ready, morphology_ready)
    backfill_action = _backfill_action(
        foundation_ready=foundation_ready,
        structure_ready=structure_ready,
        morphology_ready=morphology_ready,
        has_he=has_he,
        has_alignment=has_alignment,
        source_geojson=source_geojson,
    )
    estimated_patch_count = contours if backfill_action == "extract_contour_patches" else 0
    return {
        "case_name": record.get("case_name"),
        "case_leaf": output_dir.name,
        "relative_path": record.get("relative_path"),
        "xenium_root": record.get("xenium_root"),
        "output_dir": str(output_dir),
        "qc_status": qc_status,
        "cells": cells,
        "genes": genes,
        "contours": contours,
        "assigned_cells": assigned_cells,
        "contour_patches": contour_patches,
        "existing_contour_patch_images": patch_image_count,
        "missing_contour_patch_images": missing_patch_images,
        "has_he": has_he,
        "has_alignment": has_alignment,
        "source_geojson": source_geojson,
        "foundation_ready": foundation_ready,
        "structure_ready": structure_ready,
        "morphology_ready": morphology_ready,
        "readiness_level": readiness_level,
        "backfill_action": backfill_action,
        "estimated_patch_count": int(estimated_patch_count),
        "estimated_patch_bytes": int(estimated_patch_count * average_patch_bytes),
        "max_crop_side_px": int(max_crop_side_px),
        "alignment_snapshot": str(output_dir / "alignment_snapshot.jpg"),
    }


def _resolve_source_assets(
    row: Mapping[str, Any],
    *,
    source_cache_root: Path,
    allow_network: bool,
    download: bool,
) -> dict[str, Any]:
    output_dir = Path(str(row["output_dir"]))
    case_leaf = str(row["case_leaf"])
    cache_dir = source_cache_root / case_leaf
    metadata = _read_json(output_dir / "metadata_10x.json")
    source_url = metadata.get("source_url")
    local_complete = bool(row.get("has_he") and row.get("has_alignment"))
    cached = _cached_source_assets(cache_dir)
    links: list[str] = []
    if source_url and allow_network:
        links = _fetch_10x_source_links(str(source_url))
    direct = _classify_source_links(row, links)
    status = _source_status(local_complete=local_complete, cached=cached, direct=direct, source_url=source_url)
    downloaded: list[str] = []
    if download and not local_complete and direct["he_url"] and direct["alignment_url"] and status == "direct_sources_complete":
        downloaded = _download_direct_source_assets(
            row,
            cache_dir=cache_dir,
            he_url=str(direct["he_url"]),
            alignment_url=str(direct["alignment_url"]),
            keypoints_url=direct.get("keypoints_url"),
        )
        cached = _cached_source_assets(cache_dir)
        if cached["has_he"] and cached["has_alignment"]:
            status = "cache_complete"
    return {
        "case_name": row.get("case_name"),
        "case_leaf": case_leaf,
        "relative_path": row.get("relative_path"),
        "source_url": source_url,
        "source_cache_dir": str(cache_dir),
        "source_status": status,
        "local_has_he": bool(row.get("has_he")),
        "local_has_alignment": bool(row.get("has_alignment")),
        "cache_has_he": bool(cached["has_he"]),
        "cache_has_alignment": bool(cached["has_alignment"]),
        "he_url": direct.get("he_url"),
        "alignment_url": direct.get("alignment_url"),
        "keypoints_url": direct.get("keypoints_url"),
        "outs_zip_url": direct.get("outs_zip_url"),
        "matched_link_count": int(direct.get("matched_link_count") or 0),
        "candidate_link_count": int(len(links)),
        "downloaded_files": ";".join(downloaded),
        "blocked_reason": _source_blocked_reason(status, row),
    }


def _l3_plan_row(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    snapshot_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    action = str(row.get("backfill_action") or "")
    morphology_ready = bool(row.get("morphology_ready"))
    source_status = str(source.get("source_status") or "")
    blocked_reason = ""
    requires_alignment_verdict = action == "extract_contour_patches" and not morphology_ready
    if morphology_ready:
        l3_status = "morphology_ready"
    elif snapshot_failure:
        l3_status = "blocked"
        blocked_reason = f"Alignment snapshot generation failed: {snapshot_failure.get('error')}"
        requires_alignment_verdict = False
    elif requires_alignment_verdict:
        snapshot_path = Path(str(row.get("alignment_snapshot") or ""))
        l3_status = "pending_alignment_verdict" if snapshot_path.exists() else "pending_alignment_snapshot"
    elif action in {"missing_image_context", "generate_or_select_contours"}:
        if source_status in {"direct_sources_complete", "cache_complete"}:
            l3_status = "pending_source_asset_ingest"
        elif row.get("has_he") and row.get("has_alignment") and action == "generate_or_select_contours":
            l3_status = "pending_contour_generation"
        else:
            l3_status = "blocked"
            blocked_reason = str(source.get("blocked_reason") or "No official H&E/alignment source assets are currently available.")
    else:
        l3_status = "blocked"
        blocked_reason = f"Unsupported L3 action: {action}"
    merged = dict(row)
    merged.update(
        {
            "l3_status": l3_status,
            "requires_alignment_verdict": bool(requires_alignment_verdict),
            "source_status": source_status,
            "source_url": source.get("source_url"),
            "source_cache_dir": source.get("source_cache_dir"),
            "he_url": source.get("he_url"),
            "alignment_url": source.get("alignment_url"),
            "outs_zip_url": source.get("outs_zip_url"),
            "blocked_reason": blocked_reason,
        }
    )
    return merged


def _training_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(row["output_dir"]))
    return {
        "case_name": row.get("case_name"),
        "case_leaf": row.get("case_leaf"),
        "relative_path": row.get("relative_path"),
        "xenium_root": row.get("xenium_root"),
        "slide_store": str(output_dir / "xenium_slide.zarr"),
        "slide_manifest": str(output_dir / "slide_manifest.json"),
        "qc_report": str(output_dir / "qc_report.json"),
        "contour_patches_manifest": str(output_dir / "contour_patches_manifest.json"),
        "cell_to_contour": str(output_dir / "cell_to_contour.parquet"),
        "structure_assignments": str(output_dir / "structure_assignments.csv"),
        "readiness_level": "morphology_ready",
        "contour_patches": int(row.get("contour_patches") or 0),
        "existing_contour_patch_images": int(row.get("existing_contour_patch_images") or 0),
        "source_url": row.get("source_url"),
    }


def _run_backfill_case(
    row: Mapping[str, Any],
    *,
    audit_only: bool,
    overwrite: bool,
    max_crop_side_px: int,
    max_snapshot_side_px: int,
) -> dict[str, Any]:
    case_name = str(row.get("case_name") or row.get("case_leaf") or "")
    output_dir = Path(str(row["output_dir"]))
    snapshot_path = output_dir / "alignment_snapshot.jpg"
    try:
        snapshot = write_alignment_snapshot(
            xenium_root=Path(str(row["xenium_root"])),
            contour_geojson=Path(str(row["source_geojson"])),
            output_path=snapshot_path,
            max_snapshot_side_px=max_snapshot_side_px,
            overwrite=overwrite,
        )
        result = {
            "case_name": case_name,
            "case_leaf": row.get("case_leaf"),
            "output_dir": str(output_dir),
            "alignment_snapshot": str(snapshot_path),
            "snapshot_status": snapshot["status"],
        }
        if audit_only:
            result["status"] = "snapshot_written" if snapshot["status"] in {"written", "exists"} else "failed"
            return result
        existing = _read_patch_manifest(output_dir / "contour_patches_manifest.json")
        if existing and not overwrite:
            result.update({"status": "skipped_existing", "contour_patch_count": len(existing)})
            return result
        _, features = load_contour_geojson(Path(str(row["source_geojson"])))
        he_image = read_he_image(str(row["xenium_root"]))
        if he_image is None:
            raise RuntimeError("No aligned H&E image was found.")
        patches, patch_failures = _extract_contour_patches_with_failures(
            features=features,
            he_image=he_image,
            output_dir=output_dir / "contour_patches",
            contour_geojson=Path(str(row["source_geojson"])),
            max_crop_side_px=max_crop_side_px,
        )
        if not patches:
            _write_patch_failures(output_dir, patch_failures)
            raise RuntimeError("No contour patches could be extracted from the approved H&E/contour alignment.")
        _write_json(output_dir / "contour_patches_manifest.json", patches)
        _write_patch_failures(output_dir, patch_failures)
        _repair_case_patch_counts(output_dir, len(patches), skipped_count=len(patch_failures))
        result.update(
            {
                "status": "patches_written_with_skips" if patch_failures else "patches_written",
                "contour_patch_count": len(patches),
                "skipped_contour_count": len(patch_failures),
            }
        )
        return result
    except Exception as exc:
        return {
            "case_name": case_name,
            "case_leaf": row.get("case_leaf"),
            "output_dir": str(output_dir),
            "alignment_snapshot": str(snapshot_path),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _extract_contour_patches_with_failures(
    *,
    features: list[dict[str, Any]],
    he_image: Any,
    output_dir: str | Path,
    contour_geojson: str | Path,
    max_crop_side_px: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = Path(output_dir).expanduser()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        props = dict(feature["properties"])
        contour_id = str(props.get("contour_id") or f"contour_{index:05d}")
        patch_path = out / f"{index:05d}_{_slug(contour_id)}.png"
        try:
            if patch_path.exists():
                patch_meta = _patch_meta_from_existing_image(
                    geometry=feature["geometry"],
                    he_image=he_image,
                    output_path=patch_path,
                    max_side_px=max_crop_side_px,
                )
            else:
                patch_meta = _extract_feature_patch(
                    geometry=feature["geometry"],
                    he_image=he_image,
                    output_path=patch_path,
                    max_side_px=max_crop_side_px,
                )
        except Exception as exc:
            failures.append(
                {
                    "feature_index": int(index),
                    "contour_id": contour_id,
                    "structure_id": props.get("structure_id"),
                    "structure_label": props.get("structure_label"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "bbox": {
                        "x0": props.get("bbox_x0"),
                        "y0": props.get("bbox_y0"),
                        "x1": props.get("bbox_x1"),
                        "y1": props.get("bbox_y1"),
                        "coordinate_space": "xenium",
                    },
                    "source_geojson": str(Path(contour_geojson).expanduser()),
                }
            )
            continue
        records.append(
            {
                "contour_id": contour_id,
                "structure_id": props.get("structure_id"),
                "structure_label": props.get("structure_label"),
                "structure_name": props.get("structure_label"),
                "image_path": str(patch_path),
                "bbox": {
                    "x0": props.get("bbox_x0"),
                    "y0": props.get("bbox_y0"),
                    "x1": props.get("bbox_x1"),
                    "y1": props.get("bbox_y1"),
                    "coordinate_space": "xenium",
                },
                "pyramid_level": patch_meta["pyramid_level"],
                "transform": he_image.transform_metadata(),
                "source_geojson": str(Path(contour_geojson).expanduser()),
                "patch": patch_meta,
            }
        )
    return records, failures


def _patch_meta_from_existing_image(
    *,
    geometry: Polygon | MultiPolygon,
    he_image: Any,
    output_path: Path,
    max_side_px: int,
) -> dict[str, Any]:
    image_geometry_level0 = _geometry_xenium_pixel_to_image_xy(geometry, he_image)
    min_x, min_y, max_x, max_y = image_geometry_level0.bounds
    bbox_level0 = (
        int(np.floor(min_x)),
        int(np.floor(min_y)),
        int(np.ceil(max_x)),
        int(np.ceil(max_y)),
    )
    level_index, scale_x, scale_y = _select_pyramid_level(
        he_image,
        bbox_level0=bbox_level0,
        max_side_px=max_side_px,
    )
    level = he_image.levels[level_index]
    shape_at_level = tuple(int(value) for value in getattr(level, "shape", np.shape(level)))
    image_width = shape_at_level[he_image.axes.index("x")]
    image_height = shape_at_level[he_image.axes.index("y")]
    bbox = (
        max(int(np.floor(min_x / scale_x)), 0),
        max(int(np.floor(min_y / scale_y)), 0),
        min(int(np.ceil(max_x / scale_x)), image_width),
        min(int(np.ceil(max_y / scale_y)), image_height),
    )
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Contour bbox does not intersect H&E image at level {level_index}: {bbox}")
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Reading existing H&E contour patches requires Pillow.") from exc
    with Image.open(output_path) as image:
        saved_width, saved_height = image.size
        rgb = np.asarray(image.convert("RGB"))
    nonzero = float(np.count_nonzero(rgb.sum(axis=-1) > 3) / max(rgb.shape[0] * rgb.shape[1], 1))
    return {
        "path": str(output_path),
        "original_width": int(x1 - x0),
        "original_height": int(y1 - y0),
        "saved_width": int(saved_width),
        "saved_height": int(saved_height),
        "nonzero_fraction": nonzero,
        "pyramid_level": int(level_index),
        "level_downsample_x": float(scale_x),
        "level_downsample_y": float(scale_y),
        "bbox_level_xy": [int(value) for value in bbox],
        "bbox_level0_xy": [int(value) for value in bbox_level0],
    }


def _write_patch_failures(output_dir: Path, failures: list[dict[str, Any]]) -> None:
    json_path = output_dir / "contour_patch_failures.json"
    csv_path = output_dir / "contour_patch_failures.csv"
    if failures:
        _write_json(json_path, failures)
        pd.DataFrame(failures).to_csv(csv_path, index=False)
        return
    for path in (json_path, csv_path):
        if path.exists():
            path.unlink()


def write_alignment_snapshot(
    *,
    xenium_root: str | Path,
    contour_geojson: str | Path,
    output_path: str | Path,
    max_snapshot_side_px: int = 1600,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render contour outlines over a downsampled aligned H&E image."""

    out = Path(output_path).expanduser()
    if out.exists() and not overwrite:
        return {"status": "exists", "path": str(out)}
    he_image = read_he_image(str(xenium_root))
    if he_image is None:
        raise RuntimeError("No aligned H&E image was found.")
    _, features = load_contour_geojson(contour_geojson)
    level_index = _snapshot_level_index(he_image, max_snapshot_side_px=max_snapshot_side_px)
    level = he_image.levels[level_index]
    shape_at_level = tuple(int(value) for value in getattr(level, "shape", np.shape(level)))
    x_index = he_image.axes.index("x")
    y_index = he_image.axes.index("y")
    width = shape_at_level[x_index]
    height = shape_at_level[y_index]
    rgb, sampling_step = _read_snapshot_rgb(
        level,
        axes=he_image.axes,
        width=width,
        height=height,
        max_snapshot_side_px=max_snapshot_side_px,
    )
    from PIL import Image, ImageDraw

    pil = Image.fromarray(rgb)
    render_scale = min(float(max_snapshot_side_px) / max(pil.size), 1.0)
    if render_scale < 1.0:
        new_size = (max(int(pil.size[0] * render_scale), 1), max(int(pil.size[1] * render_scale), 1))
        pil = pil.resize(new_size, Image.Resampling.LANCZOS)
    shapes = he_image.multiscale_shapes()
    level0_shape = shapes[0]
    scale_x = float(level0_shape[x_index]) / float(shape_at_level[x_index]) * float(sampling_step)
    scale_y = float(level0_shape[y_index]) / float(shape_at_level[y_index]) * float(sampling_step)
    draw = ImageDraw.Draw(pil)
    for feature in features:
        image_geometry = _geometry_xenium_pixel_to_image_xy(feature["geometry"], he_image)
        for polygon in _iter_polygons(image_geometry):
            coords = [
                (float(x) / scale_x * render_scale, float(y) / scale_y * render_scale)
                for x, y in polygon.exterior.coords
            ]
            if len(coords) >= 2:
                draw.line(coords, fill=(255, 0, 0), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out, quality=92)
    return {
        "status": "written",
        "path": str(out),
        "contours": int(len(features)),
        "pyramid_level": int(level_index),
        "sampling_step": int(sampling_step),
        "snapshot_width": int(pil.size[0]),
        "snapshot_height": int(pil.size[1]),
    }


def _load_registry(slide_root: Path) -> tuple[list[dict[str, Any]], Path]:
    registry_path = slide_root / "dataset_registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload], registry_path
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return [dict(row) for row in records], registry_path


def _case_output_dir(record: Mapping[str, Any], slide_root: Path) -> Path:
    output_dir = record.get("output_dir") or record.get("reused_output_dir")
    if output_dir:
        return Path(str(output_dir))
    xenium_root = record.get("xenium_root")
    leaf = Path(str(xenium_root)).name if xenium_root else str(record.get("case_slug") or record.get("case_name"))
    return slide_root / leaf


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_patch_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("patches") or payload.get("contour_patches") or []
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _existing_patch_image_count(patches: list[dict[str, Any]]) -> int:
    return int(sum(1 for row in patches if row.get("image_path") and Path(str(row["image_path"])).exists()))


def _missing_patch_image_count(patches: list[dict[str, Any]]) -> int:
    return int(sum(1 for row in patches if row.get("image_path") and not Path(str(row["image_path"])).exists()))


def _average_existing_patch_bytes(slide_root: Path) -> int:
    sizes: list[int] = []
    for manifest in slide_root.glob("*/contour_patches_manifest.json"):
        for row in _read_patch_manifest(manifest):
            image_path = row.get("image_path")
            if image_path and Path(str(image_path)).exists():
                sizes.append(int(Path(str(image_path)).stat().st_size))
    if not sizes:
        return DEFAULT_ESTIMATED_PATCH_BYTES
    return max(int(sum(sizes) / len(sizes)), 1)


def _readiness_level(foundation_ready: bool, structure_ready: bool, morphology_ready: bool) -> str:
    if morphology_ready:
        return "morphology_ready"
    if structure_ready:
        return "structure_ready"
    if foundation_ready:
        return "foundation_ready"
    return "not_ready"


def _backfill_action(
    *,
    foundation_ready: bool,
    structure_ready: bool,
    morphology_ready: bool,
    has_he: bool,
    has_alignment: bool,
    source_geojson: str | None,
) -> str:
    if morphology_ready:
        return "none"
    if not foundation_ready:
        return "repair_foundation_contract"
    if structure_ready and has_he and has_alignment and source_geojson:
        return "extract_contour_patches"
    if structure_ready:
        return "missing_image_context"
    if source_geojson:
        return "repair_contour_assignment"
    return "generate_or_select_contours"


def _alignment_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_rows = []
    for row in rows:
        if row["backfill_action"] != "extract_contour_patches":
            continue
        snapshot = Path(str(row["alignment_snapshot"]))
        audit_rows.append(
            {
                "case_name": row["case_name"],
                "case_leaf": row["case_leaf"],
                "relative_path": row["relative_path"],
                "xenium_root": row["xenium_root"],
                "output_dir": row["output_dir"],
                "source_geojson": row["source_geojson"],
                "alignment_snapshot": str(snapshot),
                "audit_status": "existing_snapshot" if snapshot.exists() else "pending",
                "estimated_patch_count": row["estimated_patch_count"],
                "estimated_patch_bytes": row["estimated_patch_bytes"],
            }
        )
    return audit_rows


def _storage_estimate(rows: list[dict[str, Any]], *, average_patch_bytes: int, max_crop_side_px: int) -> dict[str, Any]:
    candidates = [row for row in rows if row["backfill_action"] == "extract_contour_patches"]
    estimated_count = int(sum(int(row["estimated_patch_count"]) for row in candidates))
    estimated_bytes = int(sum(int(row["estimated_patch_bytes"]) for row in candidates))
    return {
        "candidate_cases": int(len(candidates)),
        "estimated_patch_count": estimated_count,
        "average_patch_bytes": int(average_patch_bytes),
        "estimated_bytes": estimated_bytes,
        "estimated_gib": float(estimated_bytes / (1024**3)),
        "max_crop_side_px": int(max_crop_side_px),
        "generated_at": _utc_now(),
    }


def _update_registry_readiness(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    by_output = {str(row["output_dir"]): row for row in rows}
    for record in records:
        output_dir = record.get("output_dir") or record.get("reused_output_dir")
        row = by_output.get(str(output_dir))
        if row is None:
            continue
        for key in (
            "foundation_ready",
            "structure_ready",
            "morphology_ready",
            "readiness_level",
            "backfill_action",
            "estimated_patch_count",
            "estimated_patch_bytes",
            "existing_contour_patch_images",
            "missing_contour_patch_images",
        ):
            record[key] = row[key]


def _update_registry_l3(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    by_output = {str(row["output_dir"]): row for row in rows}
    for record in records:
        output_dir = record.get("output_dir") or record.get("reused_output_dir")
        row = by_output.get(str(output_dir))
        if row is None:
            continue
        for key in (
            "l3_status",
            "source_status",
            "source_cache_dir",
            "blocked_reason",
            "requires_alignment_verdict",
        ):
            record[key] = row.get(key)


def _write_registry(path: Path, records: list[dict[str, Any]]) -> None:
    _write_json(path, records)
    pd.DataFrame(records).to_csv(path.with_suffix(".csv"), index=False)


def _write_table_and_json(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _write_json(csv_path.with_suffix(".json"), rows)


def _write_alignment_verdict_template(path: Path, rows: list[dict[str, Any]]) -> None:
    existing = _load_alignment_verdict_rows(path)
    existing_by_case = {row["case_leaf"]: row for row in existing if row.get("case_leaf")}
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        case_leaf = str(row.get("case_leaf") or "")
        previous = existing_by_case.get(case_leaf, {})
        output_rows.append(
            {
                "case_leaf": case_leaf,
                "alignment_snapshot": row.get("alignment_snapshot"),
                "verdict": previous.get("verdict", ""),
                "notes": previous.get("notes", ""),
                "reviewer": previous.get("reviewer", ""),
                "reviewed_at": previous.get("reviewed_at", ""),
            }
        )
    pd.DataFrame(output_rows).to_csv(path, index=False)


def _load_alignment_verdict_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, dtype=str).fillna("")
    return [dict(row) for row in frame.to_dict(orient="records")]


def _load_alignment_verdicts(path: Path) -> dict[str, str]:
    rows = _load_alignment_verdict_rows(path)
    return {str(row.get("case_leaf")): str(row.get("verdict", "")).strip().lower() for row in rows}


def _load_snapshot_failures(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / "backfill_results.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("case_leaf")): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("status") == "failed" and row.get("case_leaf")
    }


def _cached_source_assets(cache_dir: Path) -> dict[str, bool]:
    he_files = list(cache_dir.glob("*he_image*.ome.tif")) + list(cache_dir.glob("*he_image*.ome.tiff"))
    alignment_files = (
        list(cache_dir.glob("*imagealignment*.csv"))
        + list(cache_dir.glob("*image_alignment*.csv"))
        + list(cache_dir.glob("*alignment*.csv"))
    )
    return {"has_he": bool(he_files), "has_alignment": bool(alignment_files)}


def _fetch_10x_source_links(source_url: str) -> list[str]:
    try:
        response = requests.get(source_url, timeout=30)
        response.raise_for_status()
    except Exception:
        return []
    return sorted(set(TENX_CDN_PATTERN.findall(response.text)))


def _classify_source_links(row: Mapping[str, Any], links: list[str]) -> dict[str, Any]:
    matched = _matched_case_links(row, links)
    he_links = [link for link in matched if _is_he_image_url(link)]
    alignment_links = [link for link in matched if _is_alignment_url(link)]
    keypoints_links = [link for link in matched if "keypoint" in link.lower()]
    outs_links = [link for link in matched if link.lower().endswith("_outs.zip") or link.lower().endswith("outs.zip")]
    return {
        "matched_link_count": len(matched),
        "he_url": _unique_or_none(he_links),
        "alignment_url": _unique_or_none(alignment_links),
        "keypoints_url": _unique_or_none(keypoints_links),
        "outs_zip_url": _unique_or_none(outs_links),
        "ambiguous_he_count": len(he_links) if _unique_or_none(he_links) is None else 0,
        "ambiguous_alignment_count": len(alignment_links) if _unique_or_none(alignment_links) is None else 0,
    }


def _matched_case_links(row: Mapping[str, Any], links: list[str]) -> list[str]:
    leaf = str(row.get("case_leaf") or Path(str(row.get("xenium_root") or "")).name)
    stem = leaf[:-5] if leaf.lower().endswith("_outs") else leaf
    case_name = str(row.get("case_name") or "")
    needles = {_norm_token(leaf), _norm_token(stem), _norm_token(case_name)}
    needles = {needle for needle in needles if len(needle) >= 8}
    matched = [link for link in links if any(needle in _norm_token(link) for needle in needles)]
    if matched:
        return matched
    return links if len(links) == 1 else []


def _source_status(
    *,
    local_complete: bool,
    cached: Mapping[str, bool],
    direct: Mapping[str, Any],
    source_url: str | None,
) -> str:
    if local_complete:
        return "local_complete"
    if cached.get("has_he") and cached.get("has_alignment"):
        return "cache_complete"
    if direct.get("he_url") and direct.get("alignment_url"):
        return "direct_sources_complete"
    if direct.get("he_url") and direct.get("outs_zip_url"):
        return "direct_he_with_outs_zip_needs_alignment_extraction"
    if direct.get("he_url"):
        return "direct_he_only_needs_alignment"
    if direct.get("outs_zip_url"):
        return "outs_zip_available_needs_asset_extraction"
    if not source_url:
        return "missing_source_url"
    if direct.get("matched_link_count"):
        return "no_he_alignment_links_matched"
    return "unresolved"


def _source_blocked_reason(status: str, row: Mapping[str, Any]) -> str:
    if status in {"local_complete", "cache_complete", "direct_sources_complete"}:
        return ""
    if status == "direct_he_only_needs_alignment":
        return "Official page has an H&E image link, but no direct alignment CSV link was found."
    if status == "direct_he_with_outs_zip_needs_alignment_extraction":
        return "Official page has H&E and outs.zip links; alignment extraction from zip is not implemented in this pass."
    if status == "outs_zip_available_needs_asset_extraction":
        return "Official page has only an outs.zip source candidate; H&E/alignment assets are not directly resolvable."
    if status == "missing_source_url":
        return "No official 10x source URL is recorded for this case."
    return f"No official H&E/alignment source assets matched case {row.get('case_leaf')}."


def _download_direct_source_assets(
    row: Mapping[str, Any],
    *,
    cache_dir: Path,
    he_url: str,
    alignment_url: str,
    keypoints_url: str | None,
) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = [
        str(_download_url_to_dir(he_url, cache_dir)),
        str(_download_url_to_dir(alignment_url, cache_dir)),
    ]
    if keypoints_url:
        downloaded.append(str(_download_url_to_dir(keypoints_url, cache_dir)))
    experiment = Path(str(row.get("xenium_root") or "")) / "experiment.xenium"
    if experiment.exists():
        shutil.copy2(experiment, cache_dir / "experiment.xenium")
    return downloaded


def _download_url_to_dir(url: str, target_dir: Path) -> Path:
    target = target_dir / Path(url.split("?", 1)[0]).name
    if target.exists() and target.stat().st_size > 0:
        return target
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    return target


def _is_he_image_url(url: str) -> bool:
    lower = url.lower()
    return "he_image" in lower and lower.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff"))


def _is_alignment_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith(".csv") and ("alignment" in lower or "imagealignment" in lower)


def _unique_or_none(values: list[str]) -> str | None:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else None


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _repair_case_patch_counts(output_dir: Path, patch_count: int, *, skipped_count: int = 0) -> None:
    manifest_path = output_dir / "slide_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest:
        manifest.setdefault("counts", {})["contour_patches"] = int(patch_count)
        manifest.setdefault("artifacts", {})["contour_patches_manifest"] = str(output_dir / "contour_patches_manifest.json")
        _write_json(manifest_path, manifest)
    qc_path = output_dir / "qc_report.json"
    qc = _read_json(qc_path)
    if qc:
        qc.setdefault("metrics", {})["contour_patch_count"] = int(patch_count)
        warnings = [
            warning
            for warning in qc.get("warnings", [])
            if "Contour patch extraction was not run or produced no patches" not in str(warning)
            and "contours were skipped during contour patch extraction" not in str(warning)
        ]
        if skipped_count:
            warnings.append(
                f"{skipped_count} contours were skipped during contour patch extraction; "
                "see contour_patch_failures.json."
            )
        qc["warnings"] = warnings
        _write_json(qc_path, qc)


def _snapshot_level_index(he_image: Any, *, max_snapshot_side_px: int) -> int:
    shapes = he_image.multiscale_shapes()
    x_index = he_image.axes.index("x")
    y_index = he_image.axes.index("y")
    selected = len(shapes) - 1
    for index, shape_at_level in enumerate(shapes):
        if max(int(shape_at_level[x_index]), int(shape_at_level[y_index])) <= int(max_snapshot_side_px):
            selected = index
            break
    return selected


def _read_snapshot_rgb(
    level: Any,
    *,
    axes: str,
    width: int,
    height: int,
    max_snapshot_side_px: int,
) -> tuple[np.ndarray, int]:
    step = max(int(np.ceil(max(width, height) / max(float(max_snapshot_side_px), 1.0))), 1)
    try:
        if step <= 1:
            image = _crop_image_level(level, axes=axes, bbox=(0, 0, width, height))
            return _to_rgb(image, axes), 1
        slices = [slice(None)] * len(level.shape)
        slices[axes.index("x")] = slice(0, width, step)
        slices[axes.index("y")] = slice(0, height, step)
        if hasattr(level, "open_zarr_source"):
            store, source = level.open_zarr_source()
            try:
                image = np.asarray(source[tuple(slices)]).copy()
            finally:
                if hasattr(store, "close"):
                    store.close()
        else:
            image = np.asarray(level[tuple(slices)]).copy()
        return _to_rgb(image, axes), step
    except Exception:
        if not hasattr(level, "open_zarr_source"):
            raise
        return _read_snapshot_rgb_by_tiles(
            level,
            axes=axes,
            width=width,
            height=height,
            step=step,
        ), step


def _read_snapshot_rgb_by_tiles(
    level: Any,
    *,
    axes: str,
    width: int,
    height: int,
    step: int,
) -> np.ndarray:
    """Best-effort low-resolution read that tolerates corrupt TIFF tiles."""

    store, source = level.open_zarr_source()
    try:
        chunks = tuple(int(value) for value in getattr(source, "chunks", ()) or ())
        x_index = axes.index("x")
        y_index = axes.index("y")
        chunk_x = chunks[x_index] if len(chunks) > x_index and chunks[x_index] > 0 else 1024
        chunk_y = chunks[y_index] if len(chunks) > y_index and chunks[y_index] > 0 else 1024
        out_width = int(np.ceil(float(width) / float(max(step, 1))))
        out_height = int(np.ceil(float(height) / float(max(step, 1))))
        rgb = np.full((out_height, out_width, 3), 255, dtype=np.uint8)
        for y0 in range(0, height, chunk_y):
            y1 = min(y0 + chunk_y, height)
            y_offset = (step - (y0 % step)) % step
            if y0 + y_offset >= y1:
                continue
            out_y0 = (y0 + y_offset) // step
            for x0 in range(0, width, chunk_x):
                x1 = min(x0 + chunk_x, width)
                x_offset = (step - (x0 % step)) % step
                if x0 + x_offset >= x1:
                    continue
                slices = [slice(None)] * len(level.shape)
                slices[x_index] = slice(x0, x1)
                slices[y_index] = slice(y0, y1)
                try:
                    tile = np.asarray(source[tuple(slices)]).copy()
                except Exception:
                    continue
                tile_rgb = _to_rgb(tile, axes)
                sampled = tile_rgb[y_offset::step, x_offset::step, :]
                out_x0 = (x0 + x_offset) // step
                rgb[out_y0 : out_y0 + sampled.shape[0], out_x0 : out_x0 + sampled.shape[1], :] = sampled
        return rgb
    finally:
        if hasattr(store, "close"):
            store.close()


def _iter_polygons(geometry: Polygon | MultiPolygon) -> Iterable[Polygon]:
    if isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    else:
        yield geometry


def _safe_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
