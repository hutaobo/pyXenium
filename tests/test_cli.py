import json

from click.testing import CliRunner
from pyXenium.__main__ import app
from pyXenium.io.io import load_toy


def test_demo():
    r = CliRunner().invoke(app, ["demo"])
    assert r.exit_code == 0
    assert "Loaded groups" in r.output


def test_load_toy():
    toy = load_toy()
    assert set(toy) == {"analysis", "cells", "transcripts"}


def test_datasets_command_copies_bundled_files(tmp_path):
    r = CliRunner().invoke(app, ["datasets", "--dest", str(tmp_path)])
    assert r.exit_code == 0

    target = tmp_path / "toy_slide"
    assert target.exists()
    assert (target / "analysis.zarr.zip").exists()
    assert (target / "cells.zarr.zip").exists()
    assert (target / "transcripts.zarr.zip").exists()


def test_validate_renal_ffpe_protein_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run_validated_renal_ffpe_smoke(**kwargs):
        captured.update(kwargs)
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        (output_dir / "report.md").write_text("# Smoke report\n", encoding="utf-8")
        return {
            "summary": {
                "dataset_title": "Example dataset",
                "dataset_url": "https://example.org",
                "base_path": kwargs["base_path"],
                "prefer": kwargs["prefer"],
                "n_cells": 10,
                "n_rna_features": 5,
                "n_protein_markers": 2,
                "x_nnz": 42,
                "has_spatial": True,
                "has_cluster": True,
                "obsm_keys": ["protein", "spatial"],
                "metrics_summary_num_cells_detected": 10,
                "top_rna_features_by_total_counts": [],
                "top_protein_markers_by_mean_signal": [],
                "largest_clusters": [],
            },
            "validated_reference": {
                "expected_cells": 10,
                "expected_rna_features": 5,
                "expected_protein_markers": 2,
            },
            "issues": [],
        }

    monkeypatch.setattr(
        "pyXenium.__main__.run_validated_renal_ffpe_smoke",
        fake_run_validated_renal_ffpe_smoke,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate-renal-ffpe-protein",
            str(tmp_path / "dataset"),
            "--prefer",
            "h5",
            "--top-n",
            "3",
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["prefer"] == "h5"
    assert captured["top_n"] == 3
    assert captured["base_path"] == str(tmp_path / "dataset")


def test_renal_immune_resistance_pilot_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run_renal_immune_resistance_pilot(**kwargs):
        captured.update(kwargs)
        return {
            "payload": {
                "sample_id": kwargs["sample_id"],
                "dataset_title": "Example dataset",
                "dataset_url": "https://example.org",
                "base_path": kwargs["base_path"],
                "prefer": kwargs["prefer"],
                "n_cells": 10,
                "n_rna_features": 5,
                "n_protein_markers": 2,
                "resistant_niches": ["myeloid_vascular", "epithelial_emt_front"],
                "sample_summary": {"mean_joint_score": 1.2},
                "joint_cell_states": [],
                "top_marker_discordance": [],
                "top_pathway_discordance": [],
                "top_spatial_niches": [],
                "model_comparison": [],
                "hypotheses": [],
            }
        }

    monkeypatch.setattr(
        "pyXenium.__main__.run_renal_immune_resistance_pilot",
        fake_run_renal_immune_resistance_pilot,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "renal-immune-resistance-pilot",
            str(tmp_path / "dataset"),
            "--prefer",
            "zarr",
            "--sample-id",
            "pilot_1",
            "--n-neighbors",
            "9",
            "--region-bins",
            "12",
            "--manuscript-mode",
            "--manuscript-root",
            str(tmp_path / "manuscript"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["sample_id"] == "pilot_1"
    assert captured["n_neighbors"] == 9
    assert captured["region_bins"] == 12
    assert captured["manuscript_mode"] is True
