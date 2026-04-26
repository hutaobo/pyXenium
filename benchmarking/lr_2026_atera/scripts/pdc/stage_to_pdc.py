from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PDC_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04"
DEFAULT_LOCAL_XENIUM_ROOT = Path(
    r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs"
)
SOURCE_CACHE_REL = "data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs"
REQUIRED_RAW_FILES = (
    "cell_feature_matrix.h5",
    "cells.parquet",
    "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv",
    "experiment.xenium",
    "metrics_summary.csv",
)
REPO_ITEMS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src",
    "benchmarking/lr_2026_atera/configs",
    "benchmarking/lr_2026_atera/envs",
    "benchmarking/lr_2026_atera/runners",
    "benchmarking/lr_2026_atera/scripts",
)
SMOKE_ITEMS = (
    "benchmarking/lr_2026_atera/data/input_manifest.json",
    "benchmarking/lr_2026_atera/data/lr_db_common.tsv",
    "benchmarking/lr_2026_atera/data/atera_smoke_panel.tsv",
    "benchmarking/lr_2026_atera/data/celltype_pairs.tsv",
    "benchmarking/lr_2026_atera/data/smoke",
)
PANEL_ITEMS = (
    "benchmarking/lr_2026_atera/data/lr_db_common.tsv",
    "benchmarking/lr_2026_atera/data/atera_smoke_panel.tsv",
    "benchmarking/lr_2026_atera/data/celltype_pairs.tsv",
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run_command(command: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "RequestTTY=no",
            "-o",
            "RemoteCommand=none",
            host,
            remote_command,
        ],
        check=check,
    )


def remote_exists(host: str, remote_path: str) -> bool:
    completed = ssh(host, f"test -e {q(remote_path)}", check=False)
    return completed.returncode == 0


def local_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".git" in parts or ".pytest_cache" in parts:
        return True
    return path.suffix in {".pyc", ".pyo"}


def add_path_to_archive(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    if not source.exists():
        return
    if source.is_file():
        tar.add(source, arcname=arcname)
        return
    for child in source.rglob("*"):
        if should_skip(child) or not child.is_file():
            continue
        tar.add(child, arcname=str(Path(arcname) / child.relative_to(source)).replace(os.sep, "/"))


def create_archive(repo_root: Path, archive_path: Path, *, include_smoke: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in REPO_ITEMS:
            source = repo_root / item
            if source.exists():
                arcname = str(Path("repo") / item).replace(os.sep, "/")
                add_path_to_archive(tar, source, arcname)
                entries.append({"local": str(source), "archive": arcname, "kind": "repo"})
        for item in PANEL_ITEMS:
            source = repo_root / item
            if source.exists():
                rel = Path(item).relative_to("benchmarking/lr_2026_atera/data")
                arcname = str(Path("data") / "staged_common" / rel).replace(os.sep, "/")
                add_path_to_archive(tar, source, arcname)
                entries.append({"local": str(source), "archive": arcname, "kind": "common_panel"})
        if include_smoke:
            for item in SMOKE_ITEMS:
                source = repo_root / item
                if source.exists():
                    rel = Path(item).relative_to("benchmarking/lr_2026_atera/data")
                    arcname = str(Path("data") / rel).replace(os.sep, "/")
                    add_path_to_archive(tar, source, arcname)
                    entries.append({"local": str(source), "archive": arcname, "kind": "smoke_data"})
    return entries


def build_stage_plan(
    *,
    repo_root: Path,
    local_xenium_root: Path,
    remote_root: str,
    host: str,
    include_smoke: bool,
    include_spatialdata_zarr: bool,
    skip_existing: bool,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    remote_root = remote_root.rstrip("/")
    remote_cache = f"{remote_root}/{SOURCE_CACHE_REL}"
    archive_path = archive_path or repo_root / "benchmarking" / "lr_2026_atera" / "logs" / "pdc_stage_payload.tar.gz"
    raw_items = []
    for name in REQUIRED_RAW_FILES:
        local_path = local_xenium_root / name
        raw_items.append(
            {
                "name": name,
                "local": str(local_path),
                "remote": f"{remote_cache}/{name}",
                "exists": local_path.exists(),
                "is_dir": local_path.is_dir(),
                "bytes": local_size(local_path),
                "required": True,
            }
        )
    if include_spatialdata_zarr:
        local_path = local_xenium_root / "spatialdata.zarr"
        raw_items.append(
            {
                "name": "spatialdata.zarr",
                "local": str(local_path),
                "remote": f"{remote_cache}/spatialdata.zarr",
                "exists": local_path.exists(),
                "is_dir": True,
                "bytes": local_size(local_path),
                "required": False,
            }
        )
    missing = [item["name"] for item in raw_items if item["required"] and not item["exists"]]
    mkdirs = [
        remote_root,
        f"{remote_root}/repo",
        f"{remote_root}/configs",
        f"{remote_root}/envs/python",
        f"{remote_root}/envs/r_libs",
        f"{remote_root}/external_src",
        f"{remote_root}/data/full",
        f"{remote_root}/data/smoke",
        f"{remote_root}/data/repeats",
        remote_cache,
        f"{remote_root}/runs",
        f"{remote_root}/results",
        f"{remote_root}/logs",
        f"{remote_root}/reports",
        f"{remote_root}/tmp",
        f"{remote_root}/slurm",
    ]
    return {
        "kind": "pdc_stage_plan",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "remote_root": remote_root,
        "remote_source_cache": remote_cache,
        "repo_root": str(repo_root),
        "local_xenium_root": str(local_xenium_root),
        "archive_path": str(archive_path),
        "remote_archive": f"{remote_root}/tmp/pdc_stage_payload.tar.gz",
        "include_smoke": include_smoke,
        "include_spatialdata_zarr": include_spatialdata_zarr,
        "skip_existing": skip_existing,
        "mkdirs": mkdirs,
        "raw_items": raw_items,
        "missing_required": missing,
        "path_policy": {
            "output_root": remote_root,
            "source_cache": remote_cache,
            "home_not_used_for_outputs": True,
        },
    }


def write_remote_json(host: str, remote_path: str, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        tmp_name = handle.name
    try:
        ssh(host, f"mkdir -p {q(posixpath.dirname(remote_path))}", check=True)
        run_command(["scp", tmp_name, f"{host}:{remote_path}"])
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def execute_stage_plan(plan: dict[str, Any], *, plan_only: bool = False) -> dict[str, Any]:
    host = str(plan["host"])
    remote_root = str(plan["remote_root"])
    archive_path = Path(str(plan["archive_path"]))
    repo_root = Path(str(plan["repo_root"]))
    if plan["missing_required"]:
        raise FileNotFoundError(f"Missing required local Xenium files: {', '.join(plan['missing_required'])}")
    response: dict[str, Any] = {"plan": plan, "executed": not plan_only, "steps": []}
    if plan_only:
        return response

    mkdir_command = "mkdir -p " + " ".join(q(path) for path in plan["mkdirs"])
    completed = ssh(host, mkdir_command)
    response["steps"].append({"step": "mkdir", "returncode": completed.returncode})

    archive_entries = create_archive(repo_root, archive_path, include_smoke=bool(plan["include_smoke"]))
    response["archive_entries"] = archive_entries
    run_command(["scp", str(archive_path), f"{host}:{plan['remote_archive']}"])
    response["steps"].append({"step": "scp_archive", "archive": str(archive_path)})

    extract = f"tar -xzf {q(plan['remote_archive'])} -C {q(remote_root)} && rm -f {q(plan['remote_archive'])}"
    completed = ssh(host, extract)
    response["steps"].append({"step": "extract_archive", "returncode": completed.returncode})

    copied_raw = []
    for item in plan["raw_items"]:
        if not item["exists"]:
            continue
        if plan["skip_existing"] and remote_exists(host, item["remote"]):
            copied_raw.append({**item, "action": "skipped_existing"})
            continue
        remote_parent = posixpath.dirname(str(item["remote"]))
        ssh(host, f"mkdir -p {q(remote_parent)}")
        scp_command = ["scp"]
        if item["is_dir"]:
            scp_command.append("-r")
        scp_command.extend([item["local"], f"{host}:{remote_parent}/"])
        completed = run_command(scp_command)
        copied_raw.append({**item, "action": "copied", "returncode": completed.returncode})
    response["copied_raw"] = copied_raw

    remote_manifest = f"{plan['remote_root']}/data/source_cache/breast/source_manifest.json"
    source_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_local_root": plan["local_xenium_root"],
        "remote_source_cache": plan["remote_source_cache"],
        "items": copied_raw,
        "stage_plan": {
            "host": plan["host"],
            "remote_root": plan["remote_root"],
            "include_smoke": plan["include_smoke"],
            "include_spatialdata_zarr": plan["include_spatialdata_zarr"],
        },
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(source_manifest, indent=2) + "\n")
        tmp_name = handle.name
    try:
        run_command(["scp", tmp_name, f"{host}:{remote_manifest}"])
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    response["remote_source_manifest"] = remote_manifest
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage pyXenium LR benchmark code and breast source cache to PDC.")
    parser.add_argument("--host", default=DEFAULT_PDC_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--local-xenium-root", default=str(DEFAULT_LOCAL_XENIUM_ROOT))
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--include-smoke", action="store_true", default=True)
    parser.add_argument("--skip-smoke", action="store_false", dest="include_smoke")
    parser.add_argument("--include-spatialdata-zarr", action="store_true", default=True)
    parser.add_argument("--skip-spatialdata-zarr", action="store_false", dest="include_spatialdata_zarr")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    plan = build_stage_plan(
        repo_root=repo_root,
        local_xenium_root=Path(args.local_xenium_root),
        remote_root=args.remote_root,
        host=args.host,
        include_smoke=args.include_smoke,
        include_spatialdata_zarr=args.include_spatialdata_zarr,
        skip_existing=args.skip_existing,
        archive_path=Path(args.archive_path) if args.archive_path else None,
    )
    payload: dict[str, Any]
    if args.execute and not args.plan_only:
        payload = execute_stage_plan(plan)
    else:
        payload = {"plan": plan, "executed": False}
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
