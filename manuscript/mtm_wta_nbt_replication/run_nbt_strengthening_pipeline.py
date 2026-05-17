#!/usr/bin/env python3
"""Run the NBT strengthening analyses on A100 with periodic monitoring."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BREAST_PLIP = "/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/runs/direct_lazyslide_plip_full_text_mtm_wta_programs_20260509"
BREAST_UNI = "/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/runs/direct_lazyslide_uni_full_mtm_wta_20260509"
CERVICAL_PLIP = "/data/taobo.hu/pyxenium_lazyslide_cervical_wta_20260511/runs/direct_lazyslide_plip_full_mtm_wta"
CERVICAL_UNI = "/data/taobo.hu/pyxenium_lazyslide_cervical_wta_20260511/runs/direct_lazyslide_uni_full_mtm_wta"
REMOTE_PYTHON = "/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/envs/plip-patch/bin/python"
REMOTE_ROOT = "/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507/runs/nbt_strengthening_20260517"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="a100")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--remote-python", default=REMOTE_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--local-out-dir", type=Path, default=Path("manuscript/mtm_wta_nbt_replication/nbt_strengthening_20260517"))
    parser.add_argument("--skip-upload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    local_out = (repo / args.local_out_dir).resolve()
    local_out.mkdir(parents=True, exist_ok=True)
    status_path = local_out / "monitor_status.jsonl"
    scripts = [
        "run_spatial_sensitivity.py",
        "build_nbt_strengthening_candidates.py",
        "run_registration_perturbation.py",
        "run_nested_spatial_holdout.py",
        "run_morphology_panel_export.py",
    ]
    script_dir = Path(__file__).resolve().parent
    remote_scripts = f"{args.remote_root}/scripts"
    remote_source_data = f"{args.remote_root}/source_data"

    if not args.skip_upload:
        run_checked(["ssh", args.host, f"mkdir -p {q(remote_scripts)} {q(remote_source_data)}"])
        for script in scripts:
            run_checked(["scp", str(script_dir / script), f"{args.host}:{remote_scripts}/"])
        source_dir = repo / "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data"
        run_checked(["scp", str(source_dir / "Figure_1c_Spatial_Permutation_Source_Data.csv"), f"{args.host}:{remote_source_data}/"])
        run_checked(["scp", str(source_dir / "Figure_1e_CrossCancer_Signature_Source_Data.csv"), f"{args.host}:{remote_source_data}/"])

    steps = build_steps(args.remote_python, args.remote_root)
    for step in steps:
        record(status_path, step=step["name"], status="started", command=step["command"])
        code = run_remote_monitored(args.host, args.remote_root, step["command"], status_path, step["name"], args.poll_seconds)
        if code != 0 and not step.get("allow_failure", False):
            record(status_path, step=step["name"], status="failed", exit_code=code)
            return code
        record(status_path, step=step["name"], status="completed" if code == 0 else "allowed_failure", exit_code=code)

    run_checked(["rsync", "-az", "--delete", f"{args.host}:{args.remote_root}/", str(local_out) + "/"])
    record(status_path, step="pull_outputs", status="completed", local_out=str(local_out))
    print(f"Pipeline completed. Pulled outputs to {local_out}")
    return 0


def build_steps(remote_python: str, remote_root: str) -> list[dict[str, object]]:
    scripts = f"{remote_root}/scripts"
    source = f"{remote_root}/source_data"
    candidates = f"{remote_root}/candidates"
    morphology_manifest = f"{BREAST_PLIP}/figure2_hero_patches/hero_patch_manifest.csv"
    return [
        {
            "name": "build_candidates",
            "command": " ".join(
                [
                    q(remote_python),
                    q(f"{scripts}/build_nbt_strengthening_candidates.py"),
                    "--source-data-dir",
                    q(source),
                    "--out-dir",
                    q(candidates),
                    "--breast-plip-run",
                    q(BREAST_PLIP),
                    "--breast-uni-run",
                    q(BREAST_UNI),
                    "--cervical-plip-run",
                    q(CERVICAL_PLIP),
                    "--cervical-uni-run",
                    q(CERVICAL_UNI),
                ]
            ),
        },
        {
            "name": "registration_breast_plip",
            "command": registration_command(remote_python, scripts, BREAST_PLIP, f"{candidates}/nbt_candidates_figure1c_plip.csv", f"{remote_root}/registration_breast_plip", "breast", "plip"),
        },
        {
            "name": "registration_cervical_plip",
            "command": registration_command(remote_python, scripts, CERVICAL_PLIP, f"{candidates}/nbt_candidates_figure1c_plip.csv", f"{remote_root}/registration_cervical_plip", "cervical", "plip"),
        },
        {
            "name": "nested_breast_plip",
            "command": nested_command(remote_python, scripts, BREAST_PLIP, f"{candidates}/nbt_candidates_figure1c_plip.csv", f"{remote_root}/nested_breast_plip", "breast", "plip"),
        },
        {
            "name": "nested_cervical_plip",
            "command": nested_command(remote_python, scripts, CERVICAL_PLIP, f"{candidates}/nbt_candidates_figure1c_plip.csv", f"{remote_root}/nested_cervical_plip", "cervical", "plip"),
        },
        {
            "name": "nested_breast_uni_cross_cancer",
            "command": nested_command(remote_python, scripts, BREAST_UNI, f"{candidates}/nbt_candidates_figure1e_cross_cancer.csv", f"{remote_root}/nested_breast_uni_cross_cancer", "breast", "uni"),
        },
        {
            "name": "nested_cervical_uni_cross_cancer",
            "command": nested_command(remote_python, scripts, CERVICAL_UNI, f"{candidates}/nbt_candidates_figure1e_cross_cancer.csv", f"{remote_root}/nested_cervical_uni_cross_cancer", "cervical", "uni"),
        },
        {
            "name": "morphology_panel_export",
            "command": " ".join(
                [
                    q(remote_python),
                    q(f"{scripts}/run_morphology_panel_export.py"),
                    "--hero-manifest",
                    q(morphology_manifest),
                    "--out-dir",
                    q(f"{remote_root}/morphology_panel"),
                ]
            ),
        },
    ]


def registration_command(remote_python: str, scripts: str, run_dir: str, candidates: str, out_dir: str, dataset: str, model: str) -> str:
    return " ".join(
        [
            q(remote_python),
            q(f"{scripts}/run_registration_perturbation.py"),
            "--run-dir",
            q(run_dir),
            "--candidates",
            q(candidates),
            "--out-dir",
            q(out_dir),
            "--dataset",
            q(dataset),
            "--model",
            q(model),
        ]
    )


def nested_command(remote_python: str, scripts: str, run_dir: str, candidates: str, out_dir: str, dataset: str, model: str) -> str:
    return " ".join(
        [
            q(remote_python),
            q(f"{scripts}/run_nested_spatial_holdout.py"),
            "--run-dir",
            q(run_dir),
            "--candidates",
            q(candidates),
            "--out-dir",
            q(out_dir),
            "--dataset",
            q(dataset),
            "--model",
            q(model),
        ]
    )


def run_remote_monitored(host: str, remote_root: str, command: str, status_path: Path, step: str, poll_seconds: int) -> int:
    wrapped = f"set -o pipefail; mkdir -p {q(remote_root)}/logs; {command} 2>&1 | tee {q(remote_root)}/logs/{q(step)}.log"
    proc = subprocess.Popen(["ssh", host, f"bash -lc {q(wrapped)}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_poll = time.monotonic()
    last_lines: list[str] = []
    while proc.poll() is None:
        if proc.stdout is not None:
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                last_lines.append(line.rstrip())
                last_lines = last_lines[-5:]
        now = time.monotonic()
        if now - last_poll >= poll_seconds:
            record(status_path, step=step, status="running", last_lines=last_lines)
            last_poll = now
        time.sleep(0.1)
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="")
            last_lines.append(line.rstrip())
            last_lines = last_lines[-5:]
    return int(proc.returncode or 0)


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def record(path: Path, **payload: object) -> None:
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def q(value: object) -> str:
    return shlex.quote(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
