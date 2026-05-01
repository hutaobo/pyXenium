from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from pyXenium.__main__ import app
from pyXenium.io import backfill_contour_patches, inspect_backfill_needs, inspect_l3_upgrade


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_case(
    root: Path,
    *,
    leaf: str,
    xenium_root: Path,
    cells: int = 10,
    genes: int = 5,
    contours: int = 0,
    assigned_cells: int = 0,
    contour_patches: int = 0,
    source_geojson: Path | None = None,
) -> Path:
    case_dir = root / leaf
    (case_dir / "xenium_slide.zarr").mkdir(parents=True)
    image_artifacts = {}
    if source_geojson is not None:
        image_artifacts = {"he": {"path": str(xenium_root / "he.ome.tif"), "image_to_xenium_affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
    _write_json(
        case_dir / "slide_manifest.json",
        {
            "counts": {
                "cells": cells,
                "genes": genes,
                "contours": contours,
                "assigned_cells": assigned_cells,
                "contour_patches": contour_patches,
            },
            "image_artifacts": image_artifacts,
            "contours": {"source_geojson": str(source_geojson) if source_geojson is not None else None},
        },
    )
    _write_json(case_dir / "qc_report.json", {"status": "pass", "warnings": []})
    _write_json(
        case_dir / "contour_source_manifest.json",
        {"status": "existing_selected" if source_geojson is not None else "missing_or_failed", "selected_geojson": str(source_geojson) if source_geojson is not None else None},
    )
    patches = []
    if contour_patches:
        patch = case_dir / "contour_patches" / "00001_contour.png"
        patch.parent.mkdir(parents=True)
        patch.write_bytes(b"png")
        patches.append({"contour_id": "c1", "image_path": str(patch)})
    _write_json(case_dir / "contour_patches_manifest.json", patches)
    return case_dir


def test_inspect_backfill_needs_writes_readiness_contract(tmp_path):
    slide_root = tmp_path / "slides"
    source_geojson = tmp_path / "contours.geojson"
    source_geojson.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    foundation = _write_case(slide_root, leaf="foundation_case", xenium_root=tmp_path / "raw" / "foundation_case")
    structure = _write_case(
        slide_root,
        leaf="structure_case",
        xenium_root=tmp_path / "raw" / "structure_case",
        contours=5,
        assigned_cells=4,
        source_geojson=source_geojson,
    )
    morphology = _write_case(
        slide_root,
        leaf="morphology_case",
        xenium_root=tmp_path / "raw" / "morphology_case",
        contours=1,
        assigned_cells=1,
        contour_patches=1,
        source_geojson=source_geojson,
    )
    records = [
        {
            "case_name": path.name,
            "case_slug": path.name,
            "relative_path": path.name,
            "xenium_root": str(tmp_path / "raw" / path.name),
            "output_dir": str(path),
            "selected_for_build": True,
            "qc_status": "pass",
            "has_he": path.name != "foundation_case",
            "has_alignment": path.name != "foundation_case",
            "selected_geojson": str(source_geojson) if path.name != "foundation_case" else None,
        }
        for path in (foundation, structure, morphology)
    ]
    _write_json(slide_root / "dataset_registry.json", records)

    result = inspect_backfill_needs(slide_root, average_patch_bytes=100, update_registry=True)

    assert result.summary["selected_cases"] == 3
    assert result.summary["foundation_ready"] == 3
    assert result.summary["structure_ready"] == 2
    assert result.summary["morphology_ready"] == 1
    assert result.summary["needs_contour_patch_extraction"] == 1
    storage = json.loads(result.storage_estimate.read_text(encoding="utf-8"))
    assert storage["estimated_patch_count"] == 5
    assert storage["estimated_bytes"] == 500
    registry = json.loads((slide_root / "dataset_registry.json").read_text(encoding="utf-8"))
    assert {row["readiness_level"] for row in registry} == {
        "foundation_ready",
        "structure_ready",
        "morphology_ready",
    }


def test_inspect_backfill_cli(tmp_path):
    slide_root = tmp_path / "slides"
    case_dir = _write_case(slide_root, leaf="foundation_case", xenium_root=tmp_path / "raw" / "foundation_case")
    _write_json(
        slide_root / "dataset_registry.json",
        [
            {
                "case_name": "foundation_case",
                "relative_path": "foundation_case",
                "xenium_root": str(tmp_path / "raw" / "foundation_case"),
                "output_dir": str(case_dir),
                "selected_for_build": True,
                "qc_status": "pass",
            }
        ],
    )

    result = CliRunner().invoke(app, ["slide", "inspect-backfill", "--slide-root", str(slide_root)])

    assert result.exit_code == 0
    assert (slide_root / "backfill_plan.csv").exists()
    payload = json.loads(result.output)
    assert payload["summary"]["foundation_ready"] == 1


def test_backfill_contour_patches_audit_only_uses_candidates(tmp_path, monkeypatch):
    slide_root = tmp_path / "slides"
    source_geojson = tmp_path / "contours.geojson"
    source_geojson.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    case_dir = _write_case(
        slide_root,
        leaf="structure_case",
        xenium_root=tmp_path / "raw" / "structure_case",
        contours=5,
        assigned_cells=4,
        source_geojson=source_geojson,
    )
    _write_json(
        slide_root / "dataset_registry.json",
        [
            {
                "case_name": "structure_case",
                "case_slug": "structure_case",
                "relative_path": "structure_case",
                "xenium_root": str(tmp_path / "raw" / "structure_case"),
                "output_dir": str(case_dir),
                "selected_for_build": True,
                "qc_status": "pass",
                "has_he": True,
                "has_alignment": True,
                "selected_geojson": str(source_geojson),
            }
        ],
    )

    from pyXenium.io import backfill as backfill_module

    monkeypatch.setattr(backfill_module, "write_alignment_snapshot", lambda **kwargs: {"status": "written", "path": str(kwargs["output_path"])})

    result = backfill_contour_patches(slide_root, audit_only=True)

    assert result["candidate_cases"] == 1
    assert result["completed"] == 1
    rows = json.loads((slide_root / "backfill_results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "snapshot_written"


def test_inspect_l3_upgrade_writes_blocked_and_training_manifests(tmp_path):
    slide_root = tmp_path / "slides"
    source_geojson = tmp_path / "contours.geojson"
    source_geojson.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    blocked = _write_case(slide_root, leaf="blocked_case", xenium_root=tmp_path / "raw" / "blocked_case")
    morphology = _write_case(
        slide_root,
        leaf="morphology_case",
        xenium_root=tmp_path / "raw" / "morphology_case",
        contours=1,
        assigned_cells=1,
        contour_patches=1,
        source_geojson=source_geojson,
    )
    _write_json(
        slide_root / "dataset_registry.json",
        [
            {
                "case_name": "blocked_case",
                "case_slug": "blocked_case",
                "relative_path": "blocked_case",
                "xenium_root": str(tmp_path / "raw" / "blocked_case"),
                "output_dir": str(blocked),
                "selected_for_build": True,
                "qc_status": "pass",
            },
            {
                "case_name": "morphology_case",
                "case_slug": "morphology_case",
                "relative_path": "morphology_case",
                "xenium_root": str(tmp_path / "raw" / "morphology_case"),
                "output_dir": str(morphology),
                "selected_for_build": True,
                "qc_status": "pass",
                "has_he": True,
                "has_alignment": True,
                "selected_geojson": str(source_geojson),
            },
        ],
    )

    result = inspect_l3_upgrade(slide_root, allow_network=False)

    assert result.summary["selected_cases"] == 2
    assert result.summary["morphology_ready"] == 1
    assert result.summary["blocked"] == 1
    assert (slide_root / "l3_upgrade_plan.csv").exists()
    assert (slide_root / "source_asset_resolution_report.csv").exists()
    assert (slide_root / "alignment_verdicts.csv").exists()
    training = json.loads((slide_root / "training_manifest_l3.json").read_text(encoding="utf-8"))
    blocked_rows = json.loads((slide_root / "blocked_cases.json").read_text(encoding="utf-8"))
    assert [row["case_leaf"] for row in training] == ["morphology_case"]
    assert [row["case_leaf"] for row in blocked_rows] == ["blocked_case"]


def test_extract_patches_requires_pass_verdict(tmp_path, monkeypatch):
    slide_root = tmp_path / "slides"
    source_geojson = tmp_path / "contours.geojson"
    source_geojson.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    case_dir = _write_case(
        slide_root,
        leaf="structure_case",
        xenium_root=tmp_path / "raw" / "structure_case",
        contours=5,
        assigned_cells=4,
        source_geojson=source_geojson,
    )
    _write_json(
        slide_root / "dataset_registry.json",
        [
            {
                "case_name": "structure_case",
                "case_slug": "structure_case",
                "relative_path": "structure_case",
                "xenium_root": str(tmp_path / "raw" / "structure_case"),
                "output_dir": str(case_dir),
                "selected_for_build": True,
                "qc_status": "pass",
                "has_he": True,
                "has_alignment": True,
                "selected_geojson": str(source_geojson),
            }
        ],
    )
    _write_json(slide_root / "alignment_verdicts.json", [])
    (slide_root / "alignment_verdicts.csv").write_text(
        "case_leaf,alignment_snapshot,verdict,notes,reviewer,reviewed_at\n"
        "structure_case,,fail,,,\n",
        encoding="utf-8",
    )

    from pyXenium.io import backfill as backfill_module

    monkeypatch.setattr(backfill_module, "write_alignment_snapshot", lambda **kwargs: {"status": "written", "path": str(kwargs["output_path"])})

    result = backfill_contour_patches(
        slide_root,
        audit_only=False,
        alignment_verdicts=slide_root / "alignment_verdicts.csv",
    )

    assert result["candidate_cases"] == 0
    assert result["skipped_by_verdict"] == 1
    rows = json.loads((slide_root / "backfill_results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "skipped_by_verdict"
