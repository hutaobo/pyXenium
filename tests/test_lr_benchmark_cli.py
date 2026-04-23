from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from pyXenium.__main__ import app


def _write_adapter_dry_run_fixture(tmp_path: Path) -> Path:
    benchmark = tmp_path / "benchmark"
    (benchmark / "configs").mkdir(parents=True)
    (benchmark / "data").mkdir(parents=True)
    methods_yaml = """methods:
  - slug: squidpy
    display_name: Squidpy ligrec
    language: python
    env_name: pyx-lr-squidpy
    runner: runners/python/run_squidpy.py
  - slug: liana
    display_name: LIANA+ bivariate
    language: python
    env_name: pyx-lr-liana
    runner: runners/python/run_liana.py
  - slug: commot
    display_name: COMMOT
    language: python
    env_name: pyx-lr-commot
    runner: runners/python/run_commot.py
  - slug: cellchat
    display_name: CellChat v3 / Spatial CellChat
    language: r
    env_name: r-lr-cellchat
    runner: runners/r/run_cellchat.R
"""
    (benchmark / "configs" / "methods.yaml").write_text(methods_yaml, encoding="utf-8")
    (benchmark / "envs").mkdir()
    (benchmark / "scripts").mkdir()
    (benchmark / "runners").mkdir()
    (benchmark / "logs").mkdir()
    (benchmark / "runs").mkdir()
    (benchmark / "results").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("toy\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("toy\n", encoding="utf-8")
    (tmp_path / "src").mkdir(exist_ok=True)
    lr_db = benchmark / "data" / "lr_db_common.tsv"
    smoke_panel = benchmark / "data" / "atera_smoke_panel.tsv"
    (benchmark / "data" / "smoke").mkdir()
    (benchmark / "data" / "smoke" / "adata_smoke.h5ad").touch()
    lr_db.write_text("ligand\treceptor\nCXCL12\tCXCR4\n", encoding="utf-8")
    smoke_panel.write_text("ligand\treceptor\nCXCL12\tCXCR4\n", encoding="utf-8")
    manifest = {
        "smoke_h5ad": str(benchmark / "data" / "smoke" / "adata_smoke.h5ad"),
        "full_h5ad": None,
        "lr_db_common_tsv": str(lr_db),
        "atera_smoke_panel_tsv": str(smoke_panel),
        "smoke_bundle": {},
        "full_bundle": {"counts_symbol_mtx": str(benchmark / "data" / "full" / "counts_symbol.mtx")},
    }
    (benchmark / "data" / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return benchmark


def test_benchmark_prepare_command(monkeypatch, tmp_path):
    captured = {}

    def fake_prepare_atera_lr_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "dataset_root": kwargs["dataset_root"],
            "benchmark_root": str(tmp_path / "benchmark"),
            "smoke_n_cells": kwargs["smoke_n_cells"],
        }

    monkeypatch.setattr("pyXenium.__main__.prepare_atera_lr_benchmark", fake_prepare_atera_lr_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "prepare",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--benchmark-root",
            str(tmp_path / "benchmark"),
            "--smoke-n-cells",
            "123",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["smoke_n_cells"] == 123
    assert captured["dataset_root"] == str(tmp_path / "dataset")
    assert captured["write_full_h5ad"] is True


def test_benchmark_prepare_command_accepts_xenium_root_alias(monkeypatch, tmp_path):
    captured = {}

    def fake_prepare_atera_lr_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "dataset_root": kwargs["dataset_root"],
            "benchmark_root": str(tmp_path / "benchmark"),
            "smoke_n_cells": kwargs["smoke_n_cells"],
        }

    monkeypatch.setattr("pyXenium.__main__.prepare_atera_lr_benchmark", fake_prepare_atera_lr_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "prepare",
            "--xenium-root",
            str(tmp_path / "dataset"),
            "--benchmark-root",
            str(tmp_path / "benchmark"),
            "--smoke-n-cells",
            "123",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["smoke_n_cells"] == 123
    assert captured["dataset_root"] == str(tmp_path / "dataset")
    assert captured["write_full_h5ad"] is True


def test_benchmark_smoke_pyxenium_command(monkeypatch, tmp_path):
    captured = {}

    def fake_resolve_layout(*, relative_root):
        root = Path(relative_root)
        return type(
            "Layout",
            (),
            {
                "root": root,
                "data_dir": root / "data",
                "runs_dir": root / "runs",
                "results_dir": root / "results",
                "reports_dir": root / "reports",
                "config_dir": root / "configs",
            },
        )()

    def fake_run_pyxenium_smoke(**kwargs):
        captured.update(kwargs)
        return {"method": "pyxenium", "output_dir": str(kwargs["output_dir"])}

    monkeypatch.setattr("pyXenium.__main__.resolve_layout", fake_resolve_layout)
    monkeypatch.setattr("pyXenium.__main__.run_pyxenium_smoke", fake_run_pyxenium_smoke)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "smoke-pyxenium",
            "--benchmark-root",
            str(tmp_path / "benchmark"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "pyxenium"
    assert captured["input_h5ad"] == tmp_path / "benchmark" / "data" / "smoke" / "adata_smoke.h5ad"


def test_benchmark_report_command(monkeypatch, tmp_path):
    combined = tmp_path / "combined.tsv"
    combined.write_text(
        "method\tdatabase_mode\tligand\treceptor\tsender\treceiver\tscore_raw\tscore_std\trank_within_method\trank_fraction\tfdr_or_pvalue\tresolution\tspatial_support_type\tartifact_path\n"
        "pyxenium\tcommon-db\tCXCL12\tCXCR4\tCAFs, DCIS Associated\tT Lymphocytes\t1.0\t0.0\t1\t1.0\t\tcelltype_pair\ttopology\tartifacts\n",
        encoding="utf-8",
    )

    def fake_resolve_layout(*, relative_root):
        root = Path(relative_root)
        (root / "configs").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(parents=True, exist_ok=True)
        return type(
            "Layout",
            (),
            {
                "root": root,
                "data_dir": root / "data",
                "runs_dir": root / "runs",
                "results_dir": root / "results",
                "reports_dir": root / "reports",
                "config_dir": root / "configs",
            },
        )()

    monkeypatch.setattr("pyXenium.__main__.resolve_layout", fake_resolve_layout)
    monkeypatch.setattr("pyXenium.__main__.compute_canonical_recovery", lambda combined, canonical_axes: (None, pd.DataFrame([{"method": "pyxenium", "database_mode": "common-db", "canonical_recovery_score": 1.0}])))
    monkeypatch.setattr("pyXenium.__main__.compute_pathway_relevance", lambda combined, pathway_config: (None, pd.DataFrame([{"method": "pyxenium", "database_mode": "common-db", "pathway_relevance_score": 1.0}])))
    monkeypatch.setattr("pyXenium.__main__.compute_spatial_coherence", lambda combined: pd.DataFrame([{"method": "pyxenium", "database_mode": "common-db", "spatial_coherence_score": 1.0}]))
    monkeypatch.setattr("pyXenium.__main__.compute_novelty_support", lambda combined: (None, pd.DataFrame([{"method": "pyxenium", "database_mode": "common-db", "novelty_support_score": 0.0}])))
    monkeypatch.setattr("pyXenium.__main__.compute_robustness", lambda combined: pd.DataFrame([{"method": "pyxenium", "database_mode": "common-db", "robustness_score": 0.0}]))
    monkeypatch.setattr("pyXenium.__main__.score_biological_performance", lambda **kwargs: pd.DataFrame([{"method": "pyxenium", "database_mode": "common-db", "biology_score": 0.75, "spatial_coherence_score": 1.0}]))
    monkeypatch.setattr("pyXenium.__main__.render_atera_lr_benchmark_report", lambda **kwargs: "# report\n")

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "report",
            "--benchmark-root",
            str(tmp_path / "benchmark"),
            "--combined-results",
            str(combined),
            "--output-path",
            str(tmp_path / "benchmark" / "reports" / "benchmark_report.md"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert Path(payload["report_md"]).exists()


def test_benchmark_run_method_dry_run_command(tmp_path):
    benchmark = _write_adapter_dry_run_fixture(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "run-method",
            "--method",
            "squidpy",
            "--benchmark-root",
            str(benchmark),
            "--dry-run",
            "--max-lr-pairs",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["method"] == "squidpy"
    assert payload["will_execute"] is True
    assert payload["lr_pairs"] == 1
    assert payload["output_dir"].endswith("squidpy_smoke_common_db")


def test_benchmark_smoke_core_dry_run_command(tmp_path):
    benchmark = _write_adapter_dry_run_fixture(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "smoke-core",
            "--benchmark-root",
            str(benchmark),
            "--methods",
            "squidpy,liana,commot,cellchat",
            "--dry-run",
            "--max-lr-pairs",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "completed"
    assert [item["method"] for item in payload["methods"]] == ["squidpy", "liana", "commot", "cellchat"]
    assert all(item["status"] == "dry-run" for item in payload["methods"])


def test_benchmark_stage_a100_plan_only_command(tmp_path, monkeypatch):
    benchmark = _write_adapter_dry_run_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "stage-a100",
            "--benchmark-root",
            str(benchmark),
            "--plan-only",
            "--transfer-mode",
            "tar-scp",
            "--remote-xenium-root",
            "/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "a100_stage_plan"
    assert payload["requires_host"] is True
    assert payload["stage_data"] is False
    assert payload["transfer_mode"] == "tar-scp"
    assert payload["readonly_xenium_root"].startswith("/mnt/taobo.hu")
    assert payload["path_policy"]["valid"] is True


def test_benchmark_prepare_a100_bundle_command(tmp_path, monkeypatch):
    benchmark = _write_adapter_dry_run_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "prepare-a100-bundle",
            "--benchmark-root",
            str(benchmark),
            "--methods",
            "squidpy",
            "--allow-missing-full",
            "--transfer-mode",
            "tar-scp",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] in {"ready", "ready_after_remote_prepare"}
    assert payload["remote_xenium_root"].startswith("/mnt/taobo.hu")
    assert payload["job_manifest"]["jobs"]
    assert payload["job_manifest"]["jobs"][0]["job_id"] == "prepare_full_bundle"
    assert payload["stage_plan"]["kind"] == "a100_stage_plan"
    assert payload["stage_plan"]["transfer_mode"] == "tar-scp"
    assert payload["path_policy"]["valid"] is True


def test_benchmark_run_a100_plan_dry_run_command(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "kind": "a100_job_manifest",
                "jobs": [
                    {
                        "job_id": "full_common_db_squidpy",
                        "method": "squidpy",
                        "command": "echo squidpy",
                        "status": "planned",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "run-a100-plan",
            "--plan-json",
            str(plan),
            "--remote",
            "--host",
            "sscb-a100.scilifelab.se",
            "--user",
            "taobo.hu",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["remote"] is True
    assert payload["jobs"][0]["status"] == "dry-run"
    assert payload["jobs"][0]["wrapper_command"].startswith("ssh taobo.hu@sscb-a100.scilifelab.se ")


def test_benchmark_collect_a100_results_dry_run_command(tmp_path, monkeypatch):
    benchmark = _write_adapter_dry_run_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "atera-lr",
            "collect-a100-results",
            "--benchmark-root",
            str(benchmark),
            "--transfer-mode",
            "scp",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["transfer_mode"] == "scp"
    assert len(payload["commands"]) == 4
