from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from pyXenium.__main__ import app
from pyXenium.io import (
    build_10x_public_slides,
    discover_10x_xenium_datasets,
    generate_missing_contours_with_histoseg,
    resolve_10x_dataset_metadata,
    select_primary_contour_geojson,
)


def _write_geojson(path: Path, *, label: str = "tumor") -> Path:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"assigned_structure": label},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [0.0, 0.0],
                        [10.0, 0.0],
                        [10.0, 10.0],
                        [0.0, 10.0],
                        [0.0, 0.0],
                    ]],
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_inventory_case(root: Path, *, uuid: str, run_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.xenium").write_text(
        json.dumps(
            {
                "experiment_uuid": uuid,
                "analysis_uuid": f"analysis-{uuid}",
                "run_name": run_name,
                "panel_name": "Xenium synthetic panel",
            }
        ),
        encoding="utf-8",
    )
    (root / "cell_feature_matrix.h5").write_bytes(b"placeholder")
    (root / "cells.parquet").write_bytes(b"placeholder")
    return root


def test_select_primary_contour_geojson_prefers_generated(tmp_path):
    root = tmp_path / "case"
    _write_geojson(root / "other_annotation.geojson", label="fallback")
    _write_geojson(root / "xenium_explorer_annotations.generated.geojson", label="generated")
    (root / "._xenium_explorer_annotations.geojson").write_text("{}", encoding="utf-8")

    selected = select_primary_contour_geojson(root)

    assert selected["status"] == "selected"
    assert selected["selected_geojson"].endswith("xenium_explorer_annotations.generated.geojson")
    assert len(selected["candidates"]) == 2


def test_discover_10x_filters_nonbuildable_and_marks_duplicate(tmp_path):
    xenium_root = tmp_path / "Xenium"
    preferred = _make_inventory_case(
        xenium_root / "Xenium_Prime" / "preferred_outs",
        uuid="same-uuid",
        run_name="Xenium_Prime_Human_Breast_Cancer_FFPE_outs",
    )
    _write_geojson(preferred / "xenium_explorer_annotations.geojson")
    _make_inventory_case(
        xenium_root / "Xenium_Prime_Copy" / "preferred_outs",
        uuid="same-uuid",
        run_name="Xenium_Prime_Human_Breast_Cancer_FFPE_outs",
    )
    nonbuildable = xenium_root / "Broken" / "unzipped"
    nonbuildable.mkdir(parents=True)
    (nonbuildable / "experiment.xenium").write_text("{}", encoding="utf-8")

    records = discover_10x_xenium_datasets(xenium_root, include_duplicates=True)

    assert len(records) == 3
    selected = [record for record in records if record["selected_for_build"]]
    assert len(selected) == 2
    duplicate = [record for record in records if record["duplicate_of"]]
    assert len(duplicate) == 1
    assert duplicate[0]["build_status"] if "build_status" in duplicate[0] else True


def test_resolve_10x_dataset_metadata_uses_cache_and_unresolved_fallback(tmp_path):
    record = {
        "experiment_uuid": "uuid-1",
        "case_name": "xenium_prime_human_breast_cancer_ffpe",
        "xenium_root": str(tmp_path / "case"),
    }
    cache = {
        "records": {
            "uuid-1": {
                "metadata_status": "resolved",
                "source_url": "https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer",
                "title": "FFPE Human Breast Cancer",
                "fields": {"Species": "Human", "Anatomical Entity": "Breast"},
            }
        }
    }

    resolved = resolve_10x_dataset_metadata(record, metadata_cache=cache, allow_network=False)
    unresolved = resolve_10x_dataset_metadata({**record, "experiment_uuid": "missing"}, allow_network=False)

    assert resolved["metadata_status"] == "resolved"
    assert resolved["fields"]["Anatomical Entity"] == "Breast"
    assert unresolved["metadata_status"] == "unresolved"
    assert unresolved["source_index_url"] == "https://www.10xgenomics.com/datasets"


def test_generate_missing_contours_without_inputs_returns_missing_inputs(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    (root / "experiment.xenium").write_text("{}", encoding="utf-8")

    result = generate_missing_contours_with_histoseg(root, tmp_path / "out", histoseg_root=tmp_path / "missing")

    assert result["status"] == "missing_inputs"
    assert result["selected_geojson"] is None
    assert not (root / "histoseg_contours").exists()


def test_build_10x_public_slides_writes_registry_for_nonbuildable_case(tmp_path):
    xenium_root = tmp_path / "Xenium"
    case = xenium_root / "Broken" / "case_outs"
    case.mkdir(parents=True)
    (case / "experiment.xenium").write_text("{}", encoding="utf-8")

    summary = build_10x_public_slides(
        xenium_root=xenium_root,
        output_root=tmp_path / "out",
        generate_missing_contours=False,
    )

    assert summary["records"] == 1
    assert summary["not_buildable"] == 1
    assert (tmp_path / "out" / "dataset_registry.json").exists()
    assert (tmp_path / "out" / "failed_cases.csv").exists()


def test_build_10x_public_slides_repairs_existing_contract_artifacts(tmp_path):
    xenium_root = tmp_path / "Xenium"
    _make_inventory_case(
        xenium_root / "Xenium_Test" / "case_outs",
        uuid="uuid-existing",
        run_name="Xenium_Prime_Human_Breast_Cancer_FFPE_outs",
    )
    output_case = tmp_path / "out" / "Xenium_Test" / "case"
    output_case.mkdir(parents=True)
    (output_case / "xenium_slide.zarr").mkdir()
    (output_case / "slide_manifest.json").write_text(
        json.dumps(
            {
                "counts": {"cells": 3, "genes": 2, "assigned_cells": 0, "contour_patches": 0},
                "artifacts": {},
                "tenx_source": {},
                "contour_source": {},
            }
        ),
        encoding="utf-8",
    )
    (output_case / "qc_report.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    cache = {
        "records": {
            "uuid-existing": {
                "metadata_status": "resolved",
                "source_url": "https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer",
                "title": "FFPE Human Breast Cancer",
                "fields": {"Species": "Human", "Anatomical Entity": "Breast"},
            }
        }
    }

    summary = build_10x_public_slides(
        xenium_root=xenium_root,
        output_root=tmp_path / "out",
        metadata_cache=cache,
        generate_missing_contours=False,
    )

    assert summary["completed_outputs"] == 1
    assert summary["skipped_existing"] == 1
    for name in (
        "metadata_10x.json",
        "contour_source_manifest.json",
        "cell_to_contour.parquet",
        "structure_assignments.csv",
        "contour_patches_manifest.json",
    ):
        assert (output_case / name).exists()
    registry = json.loads((tmp_path / "out" / "dataset_registry.json").read_text(encoding="utf-8"))
    assert registry[0]["metadata_status"] == "resolved"
    assert registry[0]["contour_status"] == "missing_or_failed"


def test_discover_10x_cli_writes_inventory(tmp_path):
    xenium_root = tmp_path / "Xenium"
    _make_inventory_case(
        xenium_root / "Xenium_Prime" / "case_outs",
        uuid="uuid-cli",
        run_name="Xenium_Prime_Human_Breast_Cancer_FFPE_outs",
    )
    output = tmp_path / "inventory.json"

    result = CliRunner().invoke(
        app,
        ["slide", "discover-10x", "--xenium-root", str(xenium_root), "--output", str(output)],
    )

    assert result.exit_code == 0
    records = json.loads(output.read_text(encoding="utf-8"))
    assert records[0]["case_slug"] == "case"


def test_new_10x_public_module_uses_xeniumslide_name_only():
    source = Path("src/pyXenium/io/tenx_public_slides.py").read_text(encoding="utf-8")
    assert "XeniumSData" not in source
