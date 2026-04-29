import json
import sys

import pyXenium
from pyXenium.perturb import (
    SpatialPerturbBridgeConfig,
    build_spatialperturb_handoff,
    spatialperturb_status,
    write_spatialperturb_handoff,
)


def test_perturb_import_does_not_import_external_spatialperturb():
    sys.modules.pop("spatialperturb", None)

    import pyXenium.perturb as perturb

    assert perturb.SpatialPerturbBridgeConfig is SpatialPerturbBridgeConfig
    assert "spatialperturb" not in sys.modules


def test_spatialperturb_status_is_json_serializable():
    status = spatialperturb_status()

    json.dumps(status)
    assert status["bridge"] == "SpatialPerturb Bridge"
    assert status["distribution"] == "SpatialPerturb"
    assert status["import_name"] == "spatialperturb"
    assert status["minimum_version"] == "0.3"
    assert isinstance(status["python_compatible"], bool)


def test_build_spatialperturb_handoff_includes_optional_inputs(tmp_path):
    config = SpatialPerturbBridgeConfig(
        xenium_path=tmp_path / "xenium outs",
        output_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        cell_group_path=tmp_path / "groups.csv",
        roi_geojson_path=tmp_path / "roi.geojson",
        sample_name="breast_case_01",
        reference_datasets=("gse241115_breast_cropseq", "gse281048_pathway_atlas"),
    )

    spec = build_spatialperturb_handoff(config)

    assert spec["bridge"] == "SpatialPerturb Bridge"
    assert spec["outputs"]["prepared_h5ad"] == str(tmp_path / "reports" / "spatialperturb_xenium.h5ad")
    assert spec["reference_datasets"] == ["gse241115_breast_cropseq", "gse281048_pathway_atlas"]
    assert spec["inputs"]["cell_group_path"] == str(tmp_path / "groups.csv")
    assert spec["inputs"]["roi_geojson_path"] == str(tmp_path / "roi.geojson")
    assert spec["inputs"]["sample_name"] == "breast_case_01"
    assert spec["commands"]["prepare_xenium"] == [
        "spatialperturb",
        "prepare-xenium",
        str(tmp_path / "xenium outs"),
        str(tmp_path / "reports" / "spatialperturb_xenium.h5ad"),
        "--cell-group-path",
        str(tmp_path / "groups.csv"),
        "--roi-geojson-path",
        str(tmp_path / "roi.geojson"),
        "--sample-name",
        "breast_case_01",
    ]
    assert spec["commands"]["run_reference_benchmark"] == [
        "spatialperturb",
        "run-reference-benchmark",
        str(tmp_path / "reports" / "spatialperturb_xenium.h5ad"),
        str(tmp_path / "reports"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--reference-datasets",
        "gse241115_breast_cropseq,gse281048_pathway_atlas",
    ]
    assert spec["command_text"]["install"] == 'python -m pip install "SpatialPerturb>=0.3"'
    assert f'"{tmp_path / "xenium outs"}"' in spec["command_text"]["prepare_xenium"]
    assert "Perturb-seq-derived program similarity" in spec["interpretation_caveat"]


def test_write_spatialperturb_handoff_round_trips_json(tmp_path):
    config = SpatialPerturbBridgeConfig(
        xenium_path=tmp_path / "xenium",
        output_dir=tmp_path / "reports",
    )

    output_path = tmp_path / "handoff" / "spatialperturb_bridge.json"
    spec = write_spatialperturb_handoff(config, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == spec


def test_build_spatialperturb_handoff_accepts_comma_separated_reference_string(tmp_path):
    spec = build_spatialperturb_handoff(
        {
            "xenium_path": tmp_path / "xenium",
            "output_dir": tmp_path / "reports",
            "reference_datasets": "gse241115_breast_cropseq,gse281048_pathway_atlas",
        }
    )

    assert spec["reference_datasets"] == ["gse241115_breast_cropseq", "gse281048_pathway_atlas"]
    assert spec["commands"]["run_reference_benchmark"][-1] == "gse241115_breast_cropseq,gse281048_pathway_atlas"


def test_top_level_lazy_exports_include_spatialperturb_bridge():
    assert pyXenium.SpatialPerturbBridgeConfig is SpatialPerturbBridgeConfig
    assert pyXenium.build_spatialperturb_handoff is build_spatialperturb_handoff
    assert pyXenium.spatialperturb_status is spatialperturb_status
    assert pyXenium.write_spatialperturb_handoff is write_spatialperturb_handoff
    assert pyXenium.perturb.SpatialPerturbBridgeConfig is SpatialPerturbBridgeConfig
