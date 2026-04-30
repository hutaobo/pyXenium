"""Contract-schema tests for pyXenium integration artifacts.

These tests validate that every handoff artifact produced or consumed by
pyXenium satisfies the schema specified in the four-package design doc:

  HistoSeg → pyXenium  : GeoJSON contours + segmentation QC JSON
  pyXenium → stGPT     : handoff manifest (JSON), tabular files (CSV/Parquet)
  stGPT → pyXenium     : embedding CSV/Parquet keyed by cell_id / contour_id
  pyXenium → SPatho    : case manifest JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pyXenium.contour import add_contours_from_geojson, import_histoseg_segmentation_qc
from pyXenium.io import XeniumSData
from pyXenium.multimodal import (
    build_spatho_manifest,
    compare_programs_with_embeddings,
    export_for_stgpt,
    import_stgpt_embeddings,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "pyxenium-bridge-v1"
_REQUIRED_PACKAGE_KEYS = {"pyXenium", "stGPT", "HistoSeg", "SPatho"}


def _toy_sdata() -> XeniumSData:
    adata = ad.AnnData(
        X=sparse.csr_matrix([[1, 0, 3], [0, 2, 1], [4, 0, 0], [0, 1, 5]], dtype=float),
        obs=pd.DataFrame(
            {
                "cell_type": ["tumor", "immune", "tumor", "stromal"],
                "sample_id": ["s1"] * 4,
                "cell_id": ["s1_c1", "s1_c2", "s1_c3", "s1_c4"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame({"feature_id": ["f_epcam", "f_cd3d", "f_vim"]}, index=["EPCAM", "CD3D", "VIM"]),
    )
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    adata.uns["sample_id"] = "s1"
    shapes = {
        "roi": pd.DataFrame(
            {
                "contour_id": ["r1", "r1", "r2", "r2"],
                "vertex_id": [0, 1, 0, 1],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 1.0, 2.0, 3.0],
                "classification_name": ["tumor", "tumor", "stroma", "stroma"],
            }
        )
    }
    return XeniumSData(table=adata, shapes=shapes, metadata={"sample_id": "s1"})


def _minimal_geojson() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": "struct_a",
                    "segmentation_source": "test_model",
                    "metadata": {
                        "assigned_structure": "Tumor",
                        "classification_name": "tumor_core",
                        "annotation_source": "auto",
                        "structure_id": "1",
                        "name": "Tumor",
                        "object_type": "polygon_unit",
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]]
                    ],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# HistoSeg → pyXenium contract: GeoJSON import
# ---------------------------------------------------------------------------


class TestGeoJSONImportContract:
    """GeoJSON produced by HistoSeg must be importable by pyXenium."""

    def test_geojson_feature_collection_required(self, tmp_path):
        geojson_path = tmp_path / "contours.geojson"
        geojson_path.write_text(json.dumps(_minimal_geojson()), encoding="utf-8")
        sdata = _toy_sdata()

        add_contours_from_geojson(
            sdata,
            geojson_path,
            key="histoseg",
            coordinate_space="xenium_um",
        )

        assert "histoseg" in sdata.shapes
        assert "histoseg" in sdata.metadata.get("contours", {})

    def test_geojson_contour_metadata_preserved(self, tmp_path):
        geojson_path = tmp_path / "contours.geojson"
        geojson_path.write_text(json.dumps(_minimal_geojson()), encoding="utf-8")
        sdata = _toy_sdata()

        add_contours_from_geojson(
            sdata,
            geojson_path,
            key="histoseg",
            coordinate_space="xenium_um",
        )

        contour_meta = sdata.metadata["contours"]["histoseg"]
        assert isinstance(contour_meta, dict)
        assert "n_contours" in contour_meta or "source_path" in contour_meta or contour_meta  # non-empty

    def test_missing_geojson_raises(self, tmp_path):
        sdata = _toy_sdata()
        with pytest.raises(FileNotFoundError):
            add_contours_from_geojson(
                sdata,
                tmp_path / "missing.geojson",
                key="histoseg",
                coordinate_space="xenium_um",
            )

    def test_duplicate_shape_key_raises(self, tmp_path):
        geojson_path = tmp_path / "contours.geojson"
        geojson_path.write_text(json.dumps(_minimal_geojson()), encoding="utf-8")
        sdata = _toy_sdata()
        add_contours_from_geojson(sdata, geojson_path, key="histoseg", coordinate_space="xenium_um")
        with pytest.raises(KeyError):
            add_contours_from_geojson(sdata, geojson_path, key="histoseg", coordinate_space="xenium_um")


# ---------------------------------------------------------------------------
# HistoSeg → pyXenium contract: segmentation QC JSON
# ---------------------------------------------------------------------------


class TestHistoSegQCImportContract:
    """HistoSeg segmentation QC JSON must be importable into XeniumSData."""

    def _write_qc_json(self, path: Path, payload: dict | None = None) -> Path:
        if payload is None:
            payload = {
                "n_structures": 1,
                "segmentation_source": "test_model",
                "model_name": "histoseg_v1",
                "mean_iou": 0.82,
                "qc_flags": [],
            }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _sdata_with_contour(self, tmp_path: Path) -> XeniumSData:
        sdata = _toy_sdata()
        geojson_path = tmp_path / "c.geojson"
        geojson_path.write_text(json.dumps(_minimal_geojson()), encoding="utf-8")
        add_contours_from_geojson(sdata, geojson_path, key="seg", coordinate_space="xenium_um")
        return sdata

    def test_qc_json_attached_to_contour_registry(self, tmp_path):
        sdata = self._sdata_with_contour(tmp_path)
        qc_path = self._write_qc_json(tmp_path / "qc.json")

        payload = import_histoseg_segmentation_qc(sdata, qc_path, shape_key="seg")

        assert payload["n_structures"] == 1
        assert sdata.metadata["contours"]["seg"]["histoseg_qc"]["n_structures"] == 1
        assert sdata.metadata["contours"]["seg"]["histoseg_qc"]["segmentation_source"] == "test_model"

    def test_qc_json_extra_keys_preserved(self, tmp_path):
        sdata = self._sdata_with_contour(tmp_path)
        qc_path = self._write_qc_json(tmp_path / "qc.json", {"n_structures": 2, "custom_metric": 0.95})

        payload = import_histoseg_segmentation_qc(sdata, qc_path, shape_key="seg")

        assert payload["custom_metric"] == 0.95
        assert sdata.metadata["contours"]["seg"]["histoseg_qc"]["custom_metric"] == 0.95

    def test_missing_shape_key_raises(self, tmp_path):
        sdata = _toy_sdata()
        qc_path = self._write_qc_json(tmp_path / "qc.json")
        with pytest.raises(KeyError, match="not found"):
            import_histoseg_segmentation_qc(sdata, qc_path, shape_key="nonexistent")

    def test_missing_file_raises(self, tmp_path):
        sdata = self._sdata_with_contour(tmp_path)
        with pytest.raises(FileNotFoundError):
            import_histoseg_segmentation_qc(sdata, tmp_path / "missing.json", shape_key="seg")

    def test_invalid_json_raises(self, tmp_path):
        sdata = self._sdata_with_contour(tmp_path)
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not json {{", encoding="utf-8")
        with pytest.raises(ValueError, match="Could not parse"):
            import_histoseg_segmentation_qc(sdata, bad_path, shape_key="seg")

    def test_non_dict_json_raises(self, tmp_path):
        sdata = self._sdata_with_contour(tmp_path)
        list_path = tmp_path / "list.json"
        list_path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            import_histoseg_segmentation_qc(sdata, list_path, shape_key="seg")


# ---------------------------------------------------------------------------
# pyXenium → stGPT contract: handoff manifest JSON schema
# ---------------------------------------------------------------------------


class TestStGPTHandoffManifestSchema:
    """The stGPT handoff manifest must satisfy the integration contract schema."""

    _REQUIRED_MANIFEST_KEYS = {
        "schema_version",
        "kind",
        "sample_id",
        "table_format",
        "package_boundaries",
        "inputs",
        "files",
        "stgpt_expected_outputs",
        "interpretation_boundary",
    }
    _REQUIRED_INPUT_KEYS = {"n_cells", "n_features", "has_spatial", "contour_key"}
    _REQUIRED_STGPT_OUTPUT_KEYS = {"cell_embeddings", "contour_embeddings", "optional_columns"}

    def test_manifest_has_all_required_top_level_keys(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path, contour_key="roi", neighbor_k=2)
        assert self._REQUIRED_MANIFEST_KEYS.issubset(manifest.keys())

    def test_manifest_kind_is_canonical(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path)
        assert manifest["kind"] == "pyxenium_to_stgpt_handoff"

    def test_manifest_schema_version(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path)
        assert manifest["schema_version"] == _SCHEMA_VERSION

    def test_manifest_package_boundaries_has_all_four_packages(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path)
        assert _REQUIRED_PACKAGE_KEYS.issubset(manifest["package_boundaries"].keys())

    def test_manifest_inputs_schema(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path)
        assert self._REQUIRED_INPUT_KEYS.issubset(manifest["inputs"].keys())
        assert isinstance(manifest["inputs"]["n_cells"], int)
        assert isinstance(manifest["inputs"]["n_features"], int)
        assert isinstance(manifest["inputs"]["has_spatial"], bool)

    def test_manifest_stgpt_expected_outputs_schema(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path)
        assert self._REQUIRED_STGPT_OUTPUT_KEYS.issubset(manifest["stgpt_expected_outputs"].keys())

    def test_manifest_files_all_exist_on_disk(self, tmp_path):
        sdata = _toy_sdata()
        manifest = export_for_stgpt(sdata, tmp_path, contour_key="roi", neighbor_k=2)
        for key, fname in manifest["files"].items():
            assert (tmp_path / fname).exists(), f"File {fname!r} for key {key!r} not on disk."

    def test_manifest_is_valid_json_on_disk(self, tmp_path):
        export_for_stgpt(_toy_sdata(), tmp_path)
        saved_path = tmp_path / "stgpt_handoff_manifest.json"
        assert saved_path.exists()
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        assert payload["kind"] == "pyxenium_to_stgpt_handoff"

    def test_csv_cell_table_has_required_columns(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path, neighbor_k=2)
        cells = pd.read_csv(tmp_path / manifest["files"]["cell_table"])
        assert "cell_id" in cells.columns
        assert "sample_id" in cells.columns

    def test_csv_feature_table_has_required_columns(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path)
        features = pd.read_csv(tmp_path / manifest["files"]["features"])
        assert "feature_id" in features.columns

    def test_parquet_cell_table_has_required_columns(self, tmp_path):
        manifest = export_for_stgpt(_toy_sdata(), tmp_path, neighbor_k=2, table_format="parquet")
        cells = pd.read_parquet(tmp_path / manifest["files"]["cell_table"])
        assert "cell_id" in cells.columns
        assert "sample_id" in cells.columns

    def test_invalid_table_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="table_format"):
            export_for_stgpt(_toy_sdata(), tmp_path, table_format="xlsx")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# stGPT → pyXenium contract: embedding CSV/Parquet schema
# ---------------------------------------------------------------------------


class TestEmbeddingImportSchema:
    """stGPT embedding tables must satisfy the pyXenium import contract."""

    def _write_cell_embeddings(self, path: Path, *, fmt: str = "csv") -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "cell_id": ["c1", "c2", "c3", "c4"],
                "z0": [0.1, 0.2, 0.3, 0.4],
                "z1": [1.0, 0.8, 0.2, 0.1],
                "niche": ["a", "a", "b", "b"],
                "uncertainty": [0.01, 0.02, 0.03, 0.04],
            }
        )
        if fmt == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        return frame

    def test_csv_embedding_attached_to_anndata(self, tmp_path):
        sdata = _toy_sdata()
        path = tmp_path / "emb.csv"
        self._write_cell_embeddings(path, fmt="csv")

        result = import_stgpt_embeddings(path, target=sdata, obsm_key="X_stgpt")

        assert result["attached_to_anndata"] is True
        assert result["embedding_columns"] == ["z0", "z1"]
        assert "X_stgpt" in sdata.table.obsm
        assert sdata.table.obsm["X_stgpt"].shape == (4, 2)

    def test_parquet_embedding_attached_to_anndata(self, tmp_path):
        sdata = _toy_sdata()
        path = tmp_path / "emb.parquet"
        self._write_cell_embeddings(path, fmt="parquet")

        result = import_stgpt_embeddings(path, target=sdata, obsm_key="X_stgpt")

        assert result["attached_to_anndata"] is True
        assert sdata.table.obsm["X_stgpt"].shape == (4, 2)

    def test_niche_and_uncertainty_added_to_obs(self, tmp_path):
        sdata = _toy_sdata()
        path = tmp_path / "emb.csv"
        self._write_cell_embeddings(path)

        import_stgpt_embeddings(path, target=sdata, obsm_key="X_stgpt")

        assert "stgpt_niche" in sdata.table.obs.columns
        assert "stgpt_uncertainty" in sdata.table.obs.columns

    def test_result_schema(self, tmp_path):
        sdata = _toy_sdata()
        path = tmp_path / "emb.csv"
        self._write_cell_embeddings(path)

        result = import_stgpt_embeddings(path, target=sdata)

        assert result["schema_version"] == _SCHEMA_VERSION
        assert result["kind"] == "stgpt_embedding_import"
        assert isinstance(result["embedding_columns"], list)
        assert isinstance(result["n_rows"], int)
        assert isinstance(result["attached_to_anndata"], bool)

    def test_no_numeric_columns_raises(self, tmp_path):
        path = tmp_path / "bad_emb.csv"
        pd.DataFrame({"cell_id": ["c1", "c2"], "label": ["a", "b"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="No numeric embedding columns"):
            import_stgpt_embeddings(path)

    def test_missing_id_column_raises(self, tmp_path):
        path = tmp_path / "no_id.csv"
        pd.DataFrame({"z0": [0.1, 0.2], "z1": [0.3, 0.4]}).to_csv(path, index=False)
        with pytest.raises(KeyError):
            import_stgpt_embeddings(path, id_column="cell_id")


# ---------------------------------------------------------------------------
# pyXenium → SPatho contract: case manifest JSON schema
# ---------------------------------------------------------------------------


class TestSPathoManifestSchema:
    """The SPatho case manifest must satisfy the integration contract schema."""

    _REQUIRED_MANIFEST_KEYS = {
        "schema_version",
        "kind",
        "sample_id",
        "report_intent",
        "package_boundaries",
        "inputs",
        "artifacts",
        "review_targets",
        "interpretation_boundary",
    }
    _REQUIRED_ARTIFACT_KEYS = {"histoseg", "pyxenium", "stgpt"}

    def test_manifest_has_all_required_keys(self, tmp_path):
        manifest = build_spatho_manifest(
            tmp_path / "manifest.json",
            sample_id="s1",
        )
        assert self._REQUIRED_MANIFEST_KEYS.issubset(manifest.keys())

    def test_manifest_kind_is_canonical(self, tmp_path):
        manifest = build_spatho_manifest(tmp_path / "manifest.json")
        assert manifest["kind"] == "pyxenium_to_spatho_manifest"

    def test_manifest_schema_version(self, tmp_path):
        manifest = build_spatho_manifest(tmp_path / "manifest.json")
        assert manifest["schema_version"] == _SCHEMA_VERSION

    def test_manifest_package_boundaries_has_all_four_packages(self, tmp_path):
        manifest = build_spatho_manifest(tmp_path / "manifest.json")
        assert _REQUIRED_PACKAGE_KEYS.issubset(manifest["package_boundaries"].keys())

    def test_manifest_artifacts_has_three_upstream_sections(self, tmp_path):
        manifest = build_spatho_manifest(tmp_path / "manifest.json")
        assert self._REQUIRED_ARTIFACT_KEYS.issubset(manifest["artifacts"].keys())

    def test_manifest_review_targets_are_list(self, tmp_path):
        manifest = build_spatho_manifest(
            tmp_path / "manifest.json",
            review_targets=[{"kind": "contour", "id": "r1"}],
        )
        assert isinstance(manifest["review_targets"], list)
        assert manifest["review_targets"][0]["kind"] == "contour"

    def test_manifest_is_valid_json_on_disk(self, tmp_path):
        output = tmp_path / "spatho_manifest.json"
        build_spatho_manifest(output, sample_id="s1")
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["kind"] == "pyxenium_to_spatho_manifest"
        assert payload["sample_id"] == "s1"

    def test_artifact_paths_are_strings_in_json(self, tmp_path):
        output = tmp_path / "spatho_manifest.json"
        build_spatho_manifest(
            output,
            histoseg_artifacts={"contours": tmp_path / "c.geojson"},
            stgpt_artifacts={"embeddings": tmp_path / "emb.parquet"},
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(payload["artifacts"]["histoseg"]["contours"], str)
        assert isinstance(payload["artifacts"]["stgpt"]["embeddings"], str)


# ---------------------------------------------------------------------------
# pyXenium concordance contract: program vs embedding comparison schema
# ---------------------------------------------------------------------------


class TestConcordanceSchema:
    """compare_programs_with_embeddings result must satisfy the concordance schema."""

    def _make_inputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        scores = pd.DataFrame(
            {
                "contour_id": ["r1", "r2", "r3", "r4"],
                "prog_a": [0.9, 0.8, 0.1, 0.0],
                "prog_b": [0.0, 0.2, 0.7, 0.9],
                "top_program": ["prog_a", "prog_a", "prog_b", "prog_b"],
            }
        )
        embeddings = pd.DataFrame(
            {
                "contour_id": ["r1", "r2", "r3", "r4"],
                "z0": [1.0, 0.8, 0.2, 0.0],
                "z1": [0.0, 0.2, 0.8, 1.0],
                "niche": ["A", "A", "B", "B"],
            }
        )
        return scores, embeddings

    def test_result_has_required_keys(self):
        scores, embeddings = self._make_inputs()
        result = compare_programs_with_embeddings(scores, embeddings)
        assert {"summary", "merged", "correlations", "label_summary", "concordance"}.issubset(result.keys())

    def test_summary_schema(self):
        scores, embeddings = self._make_inputs()
        result = compare_programs_with_embeddings(scores, embeddings)
        summary = result["summary"]
        assert summary["kind"] == "program_embedding_comparison"
        assert summary["schema_version"] == _SCHEMA_VERSION
        assert isinstance(summary["n_overlap"], int)
        assert isinstance(summary["program_columns"], list)
        assert isinstance(summary["embedding_columns"], list)

    def test_correlations_has_expected_columns(self):
        scores, embeddings = self._make_inputs()
        result = compare_programs_with_embeddings(scores, embeddings)
        corr = result["correlations"]
        assert not corr.empty
        for col in ("program", "embedding", "spearman_rho", "p_value"):
            assert col in corr.columns

    def test_concordance_has_expected_columns(self):
        scores, embeddings = self._make_inputs()
        result = compare_programs_with_embeddings(scores, embeddings)
        conc = result["concordance"]
        assert not conc.empty
        for col in ("label", "dominant_program", "dominant_fraction", "n_items"):
            assert col in conc.columns

    def test_no_overlap_raises(self):
        scores = pd.DataFrame({"contour_id": ["x"], "prog_a": [1.0]})
        embeddings = pd.DataFrame({"contour_id": ["y"], "z0": [0.5]})
        with pytest.raises(ValueError, match="No rows overlapped"):
            compare_programs_with_embeddings(scores, embeddings)
