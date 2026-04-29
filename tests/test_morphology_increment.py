from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyXenium.contour import build_contour_feature_table
from pyXenium.multimodal import BMNetBackend, compare_he_vs_xenium_morphology_sources, score_contour_boundary_programs

from test_contour_boundary_ecology import _make_boundary_ecology_sdata


def _fake_bmnet_predict(image_patch, **_metadata):
    patch = np.asarray(image_patch, dtype=float)
    mean_r = float(np.nanmean(patch[:, :, 0])) / 255.0 if patch.size else 0.0
    mean_b = float(np.nanmean(patch[:, :, 2])) / 255.0 if patch.size else 0.0
    invasive = np.clip(mean_r - 0.55, 0.0, 1.0)
    in_situ = np.clip(mean_b - 0.35, 0.0, 1.0)
    benign = max(0.0, 0.25 - invasive * 0.2)
    normal = max(0.0, 1.0 - invasive - in_situ - benign)
    return {
        "normal": normal,
        "benign": benign,
        "in_situ": in_situ,
        "invasive": invasive,
    }


def _add_synthetic_boundaries(sdata):
    cell_rows = []
    nucleus_rows = []
    spatial = np.asarray(sdata.table.obsm["spatial"], dtype=float)
    for cell_id, (x, y) in zip(sdata.table.obs_names.astype(str), spatial):
        cell_rows.extend(_square_boundary(cell_id, x, y, radius=5.0))
        nucleus_rows.extend(_square_boundary(cell_id, x, y, radius=2.5))
    sdata.shapes["cell_boundaries"] = pd.DataFrame(cell_rows)
    sdata.shapes["nucleus_boundaries"] = pd.DataFrame(nucleus_rows)
    return sdata


def _square_boundary(cell_id: str, x: float, y: float, *, radius: float):
    coords = [
        (x - radius, y - radius),
        (x + radius, y - radius),
        (x + radius, y + radius),
        (x - radius, y + radius),
    ]
    return [
        {
            "cell_id": cell_id,
            "vertex_id": index,
            "x": float(px),
            "y": float(py),
        }
        for index, (px, py) in enumerate(coords)
    ]


def test_bmnet_backend_adds_named_contour_features():
    sdata = _make_boundary_ecology_sdata()
    backend = BMNetBackend(predict_fn=_fake_bmnet_predict)
    feature_table = build_contour_feature_table(
        sdata,
        contour_key="tumor_boundary_contours",
        pathology_backends=[backend],
    )

    features = feature_table["contour_features"]
    assert "bmnet__whole__invasive_prob" in features.columns
    assert "bmnet__outer_rim__invasive_prob" in features.columns
    assert "bmnet__outer_minus_inner__invasive_prob" in features.columns
    assert "bmnet" in feature_table["feature_columns"]
    assert feature_table["context"]["pathology_backends"] == ["bmnet"]

    scored = score_contour_boundary_programs(
        sdata,
        contour_key="tumor_boundary_contours",
        feature_table=feature_table,
        program_library="breast_boundary_bmnet_v1",
    )
    weights = scored["program_feature_weights"]
    assert (weights["feature"] == "bmnet__outer_rim__invasive_prob").any()
    assert weights.loc[weights["feature"] == "bmnet__outer_rim__invasive_prob", "available"].any()


def test_morphology_increment_graceful_without_boundaries(tmp_path: Path):
    sdata = _make_boundary_ecology_sdata()
    backend = BMNetBackend(predict_fn=_fake_bmnet_predict)
    feature_table = build_contour_feature_table(
        sdata,
        contour_key="tumor_boundary_contours",
        pathology_backends=[backend],
    )
    result = compare_he_vs_xenium_morphology_sources(
        sdata,
        contour_key="tumor_boundary_contours",
        feature_table=feature_table,
        output_dir=tmp_path / "increment",
        min_contours=8,
    )

    assert result["summary"]["xenium_native_available"] is False
    assert result["summary"]["has_bmnet_features"] is True
    assert not result["he_morphology_features"].empty
    assert (tmp_path / "increment" / "incremental_prediction.csv").exists()
    assert (tmp_path / "increment" / "morphology_increment_summary.json").exists()


def test_morphology_increment_uses_xenium_native_boundaries():
    sdata = _add_synthetic_boundaries(_make_boundary_ecology_sdata())
    backend = BMNetBackend(predict_fn=_fake_bmnet_predict)
    feature_table = build_contour_feature_table(
        sdata,
        contour_key="tumor_boundary_contours",
        pathology_backends=[backend],
    )
    program_scores = score_contour_boundary_programs(
        sdata,
        contour_key="tumor_boundary_contours",
        feature_table=feature_table,
        program_library="breast_boundary_bmnet_v1",
    )["program_scores"]

    result = compare_he_vs_xenium_morphology_sources(
        sdata,
        contour_key="tumor_boundary_contours",
        feature_table=feature_table,
        program_scores=program_scores,
        min_contours=8,
    )

    native = result["xenium_native_morphology"]
    assert result["summary"]["xenium_native_available"] is True
    assert "xenium_native__whole__cell_area__mean" in native.columns
    assert "baseline_plus_he_shuffle" in set(result["incremental_prediction"]["model"])
    assert {"baseline_plus_xenium_native", "baseline_plus_he", "baseline_plus_both"}.issubset(
        set(result["incremental_prediction"]["model"])
    )
