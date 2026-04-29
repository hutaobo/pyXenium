from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_GMI_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04"
DEFAULT_GMI_PDC_XENIUM_ROOT = (
    "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/"
    "data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs"
)
DEFAULT_GMI_PDC_CONTOUR_GEOJSON = "xenium_explorer_annotations.s1_s5.generated.geojson"
DEFAULT_GMI_PDC_HISTOSEG_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04/external/HistoSeg"
DEFAULT_GMI_PDC_MODULES = ("PDC/24.11", "miniconda3/25.3.1-1-cpeGNU-24.11")
DEFAULT_GMI_PDC_REQUIRED_DATASET_FILES = (
    "cell_feature_matrix.h5",
    "cells.parquet",
    "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv",
    "analysis/analysis/clustering/gene_expression_graphclust/clusters.csv",
)


def _clean_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def validate_pdc_gmi_path_policy(*, pdc_xenium_root: str, pdc_root: str) -> dict[str, Any]:
    issues: list[str] = []
    xenium = _clean_path(pdc_xenium_root)
    root = _clean_path(pdc_root)

    if not xenium.startswith("/cfs/klemming/"):
        issues.append("pdc_xenium_root should point to a Klemming /cfs/klemming dataset path.")
    if not root.startswith("/cfs/klemming/scratch/"):
        issues.append("pdc_root should be under /cfs/klemming/scratch for PDC GMI runs.")
    if root.startswith("/cfs/klemming/home/"):
        issues.append("pdc_root must not write into the PDC home area.")
    if "/pyxenium_cci_benchmark_2026-04/" in root:
        issues.append("pdc_root must be separate from the CCI benchmark root.")
    if root == xenium or root.startswith(xenium + "/"):
        issues.append("pdc_root must not be inside the read-only Xenium source cache.")
    return {"valid": not issues, "issues": issues, "pdc_xenium_root": xenium, "pdc_root": root}


def _stage_specs(pdc_root: str, pdc_xenium_root: str) -> list[dict[str, Any]]:
    base = {"cpus_per_task": 16}
    full = {"partition": "main", "nodes": 1, "ntasks": 1, "mem": "220GB", "time": "24:00:00", **base}
    smoke = {"partition": "shared", "nodes": 1, "ntasks": 1, "mem": "64GB", "time": "04:00:00", **base}
    return [
        {
            "stage_id": "smoke_contour_top200_spatial50",
            "kind": "smoke",
            "resources": smoke,
            "args": "--rna-feature-count 200 --spatial-feature-count 50",
        },
        {
            "stage_id": "full_contour_top500_spatial100",
            "kind": "full",
            "resources": full,
            "args": "--rna-feature-count 500 --spatial-feature-count 100",
        },
        {
            "stage_id": "full_contour_top500_spatial100_stability",
            "kind": "stability",
            "resources": full,
            "args": (
                "--rna-feature-count 500 --spatial-feature-count 100 "
                "--spatial-cv-folds 5 --bootstrap-repeats 10 "
                "--label-permutation-control --coordinate-shuffle-control --spatial-feature-shuffle-control"
            ),
        },
        {
            "stage_id": "validation_rna_only_qc20",
            "kind": "validation",
            "resources": full,
            "args": "--rna-feature-count 500 --spatial-feature-count 0 --spatial-cv-folds 5 --bootstrap-repeats 10",
        },
        {
            "stage_id": "validation_spatial_only_qc20",
            "kind": "validation",
            "resources": full,
            "args": "--rna-feature-count 0 --spatial-feature-count 100 --spatial-cv-folds 5 --bootstrap-repeats 10",
        },
        {
            "stage_id": "validation_no_coordinate_qc20",
            "kind": "validation",
            "resources": full,
            "args": (
                "--rna-feature-count 500 --spatial-feature-count 100 "
                "--exclude-coordinate-spatial-features --spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
        {
            "stage_id": "sensitivity_top1000_spatial100_qc20",
            "kind": "sensitivity",
            "resources": full,
            "args": "--rna-feature-count 1000 --spatial-feature-count 100 --spatial-cv-folds 5 --bootstrap-repeats 10",
        },
        {
            "stage_id": "sensitivity_all_nonempty_top500_spatial100",
            "kind": "sensitivity",
            "resources": full,
            "args": (
                "--rna-feature-count 500 --spatial-feature-count 100 --min-cells-per-contour 1 "
                "--spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
    ]


def build_gmi_pdc_plan(
    *,
    pdc_xenium_root: str = DEFAULT_GMI_PDC_XENIUM_ROOT,
    pdc_root: str = DEFAULT_GMI_PDC_ROOT,
    repo_dir: str | None = None,
    conda_prefix: str | None = None,
    conda_pkgs_dir: str | None = None,
    histoseg_root: str | None = None,
    account: str | None = None,
    module_versions: tuple[str, ...] = DEFAULT_GMI_PDC_MODULES,
) -> dict[str, Any]:
    policy = validate_pdc_gmi_path_policy(pdc_xenium_root=pdc_xenium_root, pdc_root=pdc_root)
    root = policy["pdc_root"]
    xenium = policy["pdc_xenium_root"]
    repo = _clean_path(repo_dir or _join(root, "repo"))
    env_prefix = _clean_path(conda_prefix or _join(root, "conda", "envs", "pyx-gmi"))
    pkgs = _clean_path(conda_pkgs_dir or _join(root, "conda", "pkgs"))
    histoseg = _clean_path(histoseg_root or _join(root, "external", "HistoSeg"))
    contour_geojson = _join(xenium, DEFAULT_GMI_PDC_CONTOUR_GEOJSON)
    scripts_dir = _join(repo, "benchmarking", "gmi_pdc", "scripts")

    stages = []
    previous_stage: str | None = None
    for order, spec in enumerate(_stage_specs(root, xenium), start=1):
        stage_id = spec["stage_id"]
        output_dir = _join(root, "runs", stage_id)
        log_file = _join(root, "logs", f"{stage_id}.%j.log")
        command = (
            f"{scripts_dir}/run_pdc_stage.sh "
            f"--stage-id {stage_id} "
            f"--pdc-root {root} "
            f"--dataset-root {xenium} "
            f"--repo-dir {repo} "
            f"--conda-prefix {env_prefix} "
            f"--contour-geojson {contour_geojson} "
            f"-- {spec['args']}"
        )
        sbatch = [
            "sbatch",
            f"--job-name=pyxgmi_{stage_id[:24]}",
            f"--partition={spec['resources']['partition']}",
            f"--nodes={spec['resources']['nodes']}",
            f"--ntasks={spec['resources']['ntasks']}",
            f"--cpus-per-task={spec['resources']['cpus_per_task']}",
            f"--mem={spec['resources']['mem']}",
            f"--time={spec['resources']['time']}",
            f"--output={log_file}",
        ]
        if account:
            sbatch.append(f"--account={account}")
        if previous_stage:
            sbatch.append(f"--dependency=afterok:${{{previous_stage}_JOB_ID}}")
        sbatch.extend(["--wrap", command])
        stages.append(
            {
                "order": order,
                "stage_id": stage_id,
                "kind": spec["kind"],
                "output_dir": output_dir,
                "log_file": log_file,
                "resources": spec["resources"],
                "depends_on": previous_stage,
                "gmi_args": spec["args"],
                "command": command,
                "sbatch": sbatch,
                "sbatch_command": " ".join(sbatch),
            }
        )
        previous_stage = stage_id

    return {
        "backend": "pdc-dardel-slurm",
        "host": "dardel.pdc.kth.se",
        "pdc_xenium_root": xenium,
        "pdc_root": root,
        "repo_dir": repo,
        "runs_dir": _join(root, "runs"),
        "logs_dir": _join(root, "logs"),
        "reports_dir": _join(root, "reports"),
        "results_dir": _join(root, "results"),
        "tmp_dir": _join(root, "tmp"),
        "conda_prefix": env_prefix,
        "conda_pkgs_dir": pkgs,
        "histoseg_root": histoseg,
        "contour_geojson": contour_geojson,
        "required_dataset_files": [_join(xenium, path) for path in DEFAULT_GMI_PDC_REQUIRED_DATASET_FILES],
        "prepare_contours_command": (
            f"{scripts_dir}/prepare_pdc_inputs.sh "
            f"--pdc-root {root} "
            f"--dataset-root {xenium} "
            f"--repo-dir {repo} "
            f"--conda-prefix {env_prefix} "
            f"--histoseg-root {histoseg}"
        ),
        "module_versions": list(module_versions),
        "account": account,
        "account_policy": "Use --account, $PDC_PROJECT, or auto-resolve a single active projinfo allocation.",
        "path_policy": policy,
        "env_manifest": "benchmarking/gmi_pdc/envs/pyx-gmi-pdc.yml",
        "stages": stages,
    }


def write_gmi_pdc_plan(payload: dict[str, Any], output_json: str | Path | None) -> Path | None:
    if output_json is None:
        return None
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _read_tsv_head(path: Path, max_rows: int = 8) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
    return rows


def summarize_gmi_pdc_runs(pdc_root: str | Path = DEFAULT_GMI_PDC_ROOT) -> dict[str, Any]:
    root = Path(pdc_root)
    runs_dir = root / "runs"
    stages: list[dict[str, Any]] = []
    if runs_dir.exists():
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
            stage = {
                "stage_id": run_dir.name,
                "status": "completed" if summary else "running_or_incomplete",
                "output_dir": str(run_dir),
                "summary_json": str(summary_path) if summary_path.exists() else None,
                "main_effects_head": _read_tsv_head(run_dir / "main_effects.tsv"),
                "interaction_effects_head": _read_tsv_head(run_dir / "interaction_effects.tsv"),
                "cv_metrics_head": _read_tsv_head(run_dir / "cv_metrics.tsv"),
                "stability_head": _read_tsv_head(run_dir / "stability.tsv"),
            }
            if summary:
                stage.update(
                    {
                        "n_contours": summary.get("n_contours"),
                        "n_features": summary.get("n_features"),
                        "n_rna_features": summary.get("n_rna_features"),
                        "n_spatial_features": summary.get("n_spatial_features"),
                        "selected_main_effects": summary.get("selected_main_effects"),
                        "selected_interactions": summary.get("selected_interactions"),
                        "train_metrics": summary.get("train_metrics"),
                        "top_main_effects": summary.get("top_main_effects", [])[:10],
                        "cv_folds_completed": summary.get("cv_folds_completed"),
                        "bootstrap_repeats_requested": summary.get("bootstrap_repeats_requested"),
                    }
                )
            stages.append(stage)
    return {
        "pdc_root": str(root),
        "runs_dir": str(runs_dir),
        "n_stage_dirs": len(stages),
        "n_completed": sum(1 for stage in stages if stage["status"] == "completed"),
        "stages": stages,
    }
