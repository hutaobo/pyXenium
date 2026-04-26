from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_GMI_A100_REMOTE_ROOT = "/data/taobo.hu/pyxenium_gmi_contour_2026-04"
DEFAULT_GMI_A100_READONLY_XENIUM_ROOT = "/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs"


def validate_a100_gmi_path_policy(*, remote_xenium_root: str, remote_root: str) -> dict[str, Any]:
    issues: list[str] = []
    xenium = str(remote_xenium_root).replace("\\", "/").rstrip("/")
    root = str(remote_root).replace("\\", "/").rstrip("/")
    if not xenium.startswith("/mnt/"):
        issues.append("remote_xenium_root should point to the read-only /mnt dataset path.")
    if root.startswith("/mnt/"):
        issues.append("remote_root must be writable and must not be under /mnt.")
    if not root.startswith("/data/"):
        issues.append("remote_root should be under /data for A100 runs.")
    return {"valid": not issues, "issues": issues, "remote_xenium_root": xenium, "remote_root": root}


def _join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def build_gmi_a100_plan(
    *,
    remote_xenium_root: str = DEFAULT_GMI_A100_READONLY_XENIUM_ROOT,
    remote_root: str = DEFAULT_GMI_A100_REMOTE_ROOT,
    env_name: str = "pyx-gmi",
    repo_dir: str | None = None,
) -> dict[str, Any]:
    policy = validate_a100_gmi_path_policy(remote_xenium_root=remote_xenium_root, remote_root=remote_root)
    repo = repo_dir or _join(remote_root, "repo")
    prefix = (
        'export PATH="$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH" && '
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"; fi && '
        f"export PYTHONPATH={repo}/src:${{PYTHONPATH:-}} && "
        f"cd {repo}"
    )

    def pyx(command: str) -> str:
        return f"{prefix} && conda run --name {env_name} pyxenium {command}"

    stages = [
        {
            "job_id": "gmi_contour_smoke_top200_spatial50",
            "kind": "smoke",
            "output_dir": _join(remote_root, "runs", "smoke_contour_top200_spatial50"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'smoke_contour_top200_spatial50')} "
                "--rna-feature-count 200 --spatial-feature-count 50"
            ),
        },
        {
            "job_id": "gmi_contour_full_top500_spatial100",
            "kind": "full",
            "output_dir": _join(remote_root, "runs", "full_contour_top500_spatial100"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'full_contour_top500_spatial100')} "
                "--rna-feature-count 500 --spatial-feature-count 100"
            ),
        },
        {
            "job_id": "gmi_contour_full_top500_spatial100_stability",
            "kind": "stability",
            "output_dir": _join(remote_root, "runs", "full_contour_top500_spatial100_stability"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'full_contour_top500_spatial100_stability')} "
                "--rna-feature-count 500 --spatial-feature-count 100 --spatial-cv-folds 5 --bootstrap-repeats 10 "
                "--label-permutation-control --coordinate-shuffle-control --spatial-feature-shuffle-control"
            ),
        },
        {
            "job_id": "gmi_contour_rna_only_qc20",
            "kind": "validation",
            "output_dir": _join(remote_root, "runs", "validation_rna_only_qc20"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'validation_rna_only_qc20')} "
                "--rna-feature-count 500 --spatial-feature-count 0 --spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
        {
            "job_id": "gmi_contour_spatial_only_qc20",
            "kind": "validation",
            "output_dir": _join(remote_root, "runs", "validation_spatial_only_qc20"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'validation_spatial_only_qc20')} "
                "--rna-feature-count 0 --spatial-feature-count 100 --spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
        {
            "job_id": "gmi_contour_no_coordinate_qc20",
            "kind": "validation",
            "output_dir": _join(remote_root, "runs", "validation_no_coordinate_qc20"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'validation_no_coordinate_qc20')} "
                "--rna-feature-count 500 --spatial-feature-count 100 --exclude-coordinate-spatial-features "
                "--spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
        {
            "job_id": "gmi_contour_top1000_spatial100_qc20",
            "kind": "sensitivity",
            "output_dir": _join(remote_root, "runs", "sensitivity_top1000_spatial100_qc20"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'sensitivity_top1000_spatial100_qc20')} "
                "--rna-feature-count 1000 --spatial-feature-count 100 --spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
        {
            "job_id": "gmi_contour_all_nonempty_top500_spatial100",
            "kind": "sensitivity",
            "output_dir": _join(remote_root, "runs", "sensitivity_all_nonempty_top500_spatial100"),
            "command": pyx(
                "gmi run "
                f"--dataset-root {remote_xenium_root} "
                f"--output-dir {_join(remote_root, 'runs', 'sensitivity_all_nonempty_top500_spatial100')} "
                "--rna-feature-count 500 --spatial-feature-count 100 --min-cells-per-contour 1 "
                "--spatial-cv-folds 5 --bootstrap-repeats 10"
            ),
        },
    ]
    return {
        "remote_xenium_root": policy["remote_xenium_root"],
        "remote_root": policy["remote_root"],
        "repo_dir": repo,
        "env_name": env_name,
        "path_policy": policy,
        "env_manifest": "benchmarking/gmi_atera/envs/pyx-gmi.yml",
        "stages": stages,
    }


def write_gmi_a100_plan(payload: dict[str, Any], output_json: str | Path | None) -> Path | None:
    if output_json is None:
        return None
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
