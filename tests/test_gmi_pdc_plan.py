from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from pyXenium.__main__ import app
from pyXenium.gmi import (
    DEFAULT_GMI_PDC_ROOT,
    DEFAULT_GMI_PDC_XENIUM_ROOT,
    build_gmi_pdc_plan,
    summarize_gmi_pdc_runs,
    validate_pdc_gmi_path_policy,
)


def test_pdc_path_policy_requires_separate_klemming_scratch_root():
    valid = validate_pdc_gmi_path_policy(
        pdc_xenium_root=DEFAULT_GMI_PDC_XENIUM_ROOT,
        pdc_root=DEFAULT_GMI_PDC_ROOT,
    )
    home = validate_pdc_gmi_path_policy(
        pdc_xenium_root=DEFAULT_GMI_PDC_XENIUM_ROOT,
        pdc_root="/cfs/klemming/home/h/hutaobo/pyxenium_gmi",
    )
    cci_root = validate_pdc_gmi_path_policy(
        pdc_xenium_root=DEFAULT_GMI_PDC_XENIUM_ROOT,
        pdc_root="/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/runs/gmi",
    )
    inside_data = validate_pdc_gmi_path_policy(
        pdc_xenium_root=DEFAULT_GMI_PDC_XENIUM_ROOT,
        pdc_root=DEFAULT_GMI_PDC_XENIUM_ROOT + "/gmi_outputs",
    )

    assert valid["valid"] is True
    assert home["valid"] is False
    assert any("scratch" in issue or "home" in issue for issue in home["issues"])
    assert cci_root["valid"] is False
    assert any("separate from the CCI benchmark root" in issue for issue in cci_root["issues"])
    assert inside_data["valid"] is False
    assert any("source cache" in issue for issue in inside_data["issues"])


def test_build_gmi_pdc_plan_uses_slurm_chain_and_expected_stages():
    payload = build_gmi_pdc_plan(account="naiss-test", pdc_root=DEFAULT_GMI_PDC_ROOT)

    assert payload["backend"] == "pdc-dardel-slurm"
    assert payload["host"] == "dardel.pdc.kth.se"
    assert payload["path_policy"]["valid"] is True
    assert payload["conda_prefix"].endswith("/conda/envs/pyx-gmi")
    assert payload["conda_pkgs_dir"].endswith("/conda/pkgs")
    assert payload["histoseg_root"].endswith("/external/HistoSeg")
    assert payload["contour_geojson"].endswith("/xenium_explorer_annotations.s1_s5.generated.geojson")
    assert "prepare_pdc_inputs.sh" in payload["prepare_contours_command"]
    assert any(path.endswith("/cells.parquet") for path in payload["required_dataset_files"])
    assert any(path.endswith("/analysis/analysis/clustering/gene_expression_graphclust/clusters.csv") for path in payload["required_dataset_files"])
    assert len(payload["stages"]) == 8
    assert [stage["stage_id"] for stage in payload["stages"]] == [
        "smoke_contour_top200_spatial50",
        "full_contour_top500_spatial100",
        "full_contour_top500_spatial100_stability",
        "validation_rna_only_qc20",
        "validation_spatial_only_qc20",
        "validation_no_coordinate_qc20",
        "sensitivity_top1000_spatial100_qc20",
        "sensitivity_all_nonempty_top500_spatial100",
    ]

    smoke = payload["stages"][0]
    full = payload["stages"][1]
    stability = payload["stages"][2]
    assert smoke["resources"]["partition"] == "shared"
    assert smoke["resources"]["mem"] == "64GB"
    assert full["resources"]["partition"] == "main"
    assert full["resources"]["mem"] == "220GB"
    assert "--dependency" not in " ".join(smoke["sbatch"])
    assert "--dependency=afterok:${full_contour_top500_spatial100_JOB_ID}" in " ".join(stability["sbatch"])
    assert all("--account=naiss-test" in stage["sbatch"] for stage in payload["stages"])
    assert all("--contour-geojson" in stage["command"] for stage in payload["stages"])
    assert all("/pyxenium_cci_benchmark_2026-04/" not in stage["output_dir"] for stage in payload["stages"])
    assert all("nohup" not in stage["sbatch_command"] for stage in payload["stages"])
    assert all("sbatch" == stage["sbatch"][0] for stage in payload["stages"])


def test_gmi_pdc_plan_cli_writes_json(tmp_path):
    output = tmp_path / "pdc_plan.json"
    result = CliRunner().invoke(
        app,
        [
            "gmi",
            "pdc-plan",
            "--account",
            "naiss-test",
            "--output-json",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["path_policy"]["valid"] is True
    assert len(payload["stages"]) == 8
    assert "dardel.pdc.kth.se" in result.output
    assert "sensitivity_all_nonempty_top500_spatial100" in result.output


def test_gmi_pdc_monitor_summarizes_local_stage_outputs(tmp_path):
    run = tmp_path / "runs" / "validation_rna_only_qc20"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "n_contours": 80,
                "n_features": 500,
                "n_rna_features": 500,
                "n_spatial_features": 0,
                "selected_main_effects": 2,
                "selected_interactions": 0,
                "train_metrics": {"auc": 1.0},
                "top_main_effects": [{"feature": "NIBAN1"}, {"feature": "SORL1"}],
                "cv_folds_completed": 5,
                "bootstrap_repeats_requested": 10,
            }
        ),
        encoding="utf-8",
    )
    (run / "main_effects.tsv").write_text("feature_index\tfeature\tcoefficient\n1\tNIBAN1\t-1\n", encoding="utf-8")

    payload = summarize_gmi_pdc_runs(tmp_path)

    assert payload["n_stage_dirs"] == 1
    assert payload["n_completed"] == 1
    stage = payload["stages"][0]
    assert stage["stage_id"] == "validation_rna_only_qc20"
    assert stage["top_main_effects"][0]["feature"] == "NIBAN1"
    assert stage["main_effects_head"][0]["feature"] == "NIBAN1"


def test_gmi_pdc_scaffold_files_exist():
    root = Path(__file__).resolve().parents[1] / "benchmarking" / "gmi_pdc"
    expected = [
        "README.md",
        "envs/pyx-gmi-pdc.yml",
        "scripts/bootstrap_pdc_env.sh",
        "scripts/prepare_pdc_inputs.sh",
        "scripts/prepare_s1_s5_geojson.py",
        "scripts/run_pdc_stage.sh",
        "scripts/submit_pdc_chain.sh",
        "scripts/monitor_pdc_gmi.sh",
        "scripts/collect_pdc_results.py",
    ]
    for relpath in expected:
        assert (root / relpath).exists(), relpath
