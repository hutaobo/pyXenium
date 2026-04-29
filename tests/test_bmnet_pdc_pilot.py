from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyXenium.multimodal import (
    DeterministicBreastBMNetLikeBackend,
    build_bmnet_pilot_backend,
    run_bmnet_morphology_increment_pilot,
)
from pyXenium.multimodal.bmnet_pdc import _resolve_contour_id_key

from test_contour_boundary_ecology import _make_boundary_ecology_sdata


def test_deterministic_bmnet_like_backend_emits_breast_probabilities():
    backend = DeterministicBreastBMNetLikeBackend()
    image = np.full((32, 32, 3), [220, 150, 170], dtype=np.uint8)
    features = backend.extract_features(image)

    assert set(features) >= {
        "normal_prob",
        "benign_prob",
        "in_situ_prob",
        "invasive_prob",
        "tumor_prob",
        "invasive_margin",
        "prediction_entropy",
    }
    total = sum(features[f"{label}_prob"] for label in ("normal", "benign", "in_situ", "invasive"))
    assert np.isclose(total, 1.0)
    assert backend.metadata()["semantic_status"] == "smoke_test_only_not_biological_evidence"
    tiny = backend.extract_features(np.asarray([[[220, 150, 170]]], dtype=np.uint8))
    assert np.isclose(sum(tiny[f"{label}_prob"] for label in ("normal", "benign", "in_situ", "invasive")), 1.0)


def test_bmnet_pilot_runner_writes_increment_artifacts(tmp_path: Path):
    sdata = _make_boundary_ecology_sdata()
    result = run_bmnet_morphology_increment_pilot(
        sdata,
        output_dir=tmp_path / "bmnet_pilot",
        contour_key="tumor_boundary_contours",
        backend="deterministic-smoke",
        max_contours=2,
        min_contours=3,
    )

    out = Path(result["artifact_dir"])
    features = result["feature_table"]["contour_features"]
    assert len(features) == 2
    assert "bmnet__whole__invasive_prob" in features.columns
    assert "bmnet__outer_minus_inner__invasive_prob" in features.columns
    assert (out / "contour_features_with_bmnet.csv").exists()
    assert (out / "bmnet_patch_predictions.csv").exists()
    assert (out / "incremental_prediction.csv").exists()

    summary = json.loads((out / "bmnet_pdc_run_summary.json").read_text(encoding="utf-8"))
    assert summary["n_contours"] == 2
    assert summary["model_metadata"]["backend"] == "deterministic-smoke"
    increment_summary = json.loads((out / "morphology_increment_summary.json").read_text(encoding="utf-8"))
    assert increment_summary["model_metadata"]["semantic_status"] == "smoke_test_only_not_biological_evidence"


def test_bmnet_backend_factory_rejects_missing_local_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        build_bmnet_pilot_backend("bmnet-local")


def test_contour_id_key_falls_back_to_unique_geojson_name(tmp_path: Path):
    path = tmp_path / "contours.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"name": "S1 #1"}, "geometry": None},
                    {"type": "Feature", "properties": {"name": "S1 #2"}, "geometry": None},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _resolve_contour_id_key(path, requested="polygon_id") == "name"
