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

import yaml


DEFAULT_PDC_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"
DEFAULT_LOCAL_XENIUM_ROOT = Path(
    r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs"
)
DEFAULT_DATASET_ID = "atera_breast_wta"
DATASETS_CONFIG_REL = Path("benchmarking") / "cci_2026_atera" / "configs" / "datasets.yaml"
BASE_REQUIRED_RAW_FILES = (
    "cell_feature_matrix.h5",
    "cells.parquet",
    "experiment.xenium",
    "metrics_summary.csv",
)
REQUIRED_RAW_FILES = (*BASE_REQUIRED_RAW_FILES, "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv")
REPO_ITEMS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src",
    "benchmarking/cci_2026_atera/configs",
    "benchmarking/cci_2026_atera/envs",
    "benchmarking/cci_2026_atera/runners",
    "benchmarking/cci_2026_atera/scripts",
)
SMOKE_ITEMS = (
    "benchmarking/cci_2026_atera/data/cci_resource_common.tsv",
    "benchmarking/cci_2026_atera/data/atera_smoke_panel.tsv",
    "benchmarking/cci_2026_atera/data/celltype_pairs.tsv",
    "benchmarking/cci_2026_atera/data/smoke",
)
PANEL_ITEMS = (
    "benchmarking/cci_2026_atera/data/cci_resource_common.tsv",
    "benchmarking/cci_2026_atera/data/atera_smoke_panel.tsv",
    "benchmarking/cci_2026_atera/data/celltype_pairs.tsv",
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def load_dataset_entry(repo_root: Path, dataset_id: str, datasets_config: Path | None = None) -> dict[str, Any]:
    path = datasets_config or repo_root / DATASETS_CONFIG_REL
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []):
        if str(dataset.get("id")) == str(dataset_id):
            return dict(dataset)
    available = ", ".join(str(dataset.get("id")) for dataset in payload.get("datasets", []))
    raise ValueError(f"Unknown dataset_id {dataset_id!r}. Available datasets: {available}")


def source_cache_rel(dataset_id: str, local_xenium_root: Path) -> str:
    return f"data/source_cache/{dataset_id}/{local_xenium_root.name}"


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
                rel = Path(item).relative_to("benchmarking/cci_2026_atera/data")
                arcname = str(Path("data") / "staged_common" / rel).replace(os.sep, "/")
                add_path_to_archive(tar, source, arcname)
                entries.append({"local": str(source), "archive": arcname, "kind": "common_panel"})
        if include_smoke:
            for item in SMOKE_ITEMS:
                source = repo_root / item
                if source.exists():
                    rel = Path(item).relative_to("benchmarking/cci_2026_atera/data")
                    arcname = str(Path("data") / rel).replace(os.sep, "/")
                    add_path_to_archive(tar, source, arcname)
                    entries.append({"local": str(source), "archive": arcname, "kind": "smoke_data"})
    return entries


def build_stage_plan(
    *,
    repo_root: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset: dict[str, Any] | None = None,
    local_xenium_root: Path,
    remote_root: str,
    remote_source_root: str | None = None,
    host: str,
    include_smoke: bool,
    include_spatialdata_zarr: bool,
    skip_existing: bool,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    remote_root = remote_root.rstrip("/")
    dataset = dataset or {
        "id": dataset_id,
        "display_name": "Atera Xenium WTA FFPE breast cancer",
        "platform": "Xenium WTA",
        "tissue": "breast cancer",
        "role": "primary_discovery",
        "expected_cells": 170057,
        "cell_groups_relpath": "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv",
    }
    remote_cache = (remote_source_root.rstrip("/") if remote_source_root else f"{remote_root}/{source_cache_rel(dataset_id, local_xenium_root)}")
    local_tbc_results = Path(str(dataset.get("local_tbc_results") or "")) if dataset.get("local_tbc_results") else None
    remote_tbc_results = f"{remote_root}/data/tbc_results"
    archive_path = archive_path or repo_root / "benchmarking" / "cci_2026_atera" / "logs" / "pdc_stage_payload.tar.gz"
    cell_groups_relpath = str(dataset.get("cell_groups_relpath") or "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv")
    required_raw_files = (*BASE_REQUIRED_RAW_FILES, cell_groups_relpath)
    raw_items = []
    for name in dict.fromkeys(required_raw_files):
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
    tbc_items: list[dict[str, Any]] = []
    if local_tbc_results is not None:
        if local_tbc_results.exists():
            for child in sorted(local_tbc_results.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(local_tbc_results).as_posix()
                    tbc_items.append(
                        {
                            "name": rel,
                            "local": str(child),
                            "remote": f"{remote_tbc_results}/{rel}",
                            "exists": True,
                            "bytes": local_size(child),
                            "required": True,
                        }
                    )
        else:
            missing.append(f"tbc_results:{local_tbc_results}")
    mkdirs = [
        remote_root,
        f"{remote_root}/repo",
        f"{remote_root}/configs",
        f"{remote_root}/envs/python",
        f"{remote_root}/envs/r_libs",
        f"{remote_root}/external_src",
        f"{remote_root}/data/full",
        f"{remote_root}/data/smoke",
        remote_tbc_results,
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
        "dataset_id": dataset_id,
        "display_name": dataset.get("display_name"),
        "tissue": dataset.get("tissue"),
        "platform": dataset.get("platform"),
        "role": dataset.get("role"),
        "expected_cells": dataset.get("expected_cells"),
        "cell_groups_relpath": cell_groups_relpath,
        "local_tbc_results": dataset.get("local_tbc_results"),
        "remote_tbc_results": remote_tbc_results,
        "host": host,
        "remote_root": remote_root,
        "remote_benchmark_root": remote_root,
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
        "tbc_items": tbc_items,
        "missing_required": missing,
        "path_policy": {
            "output_root": remote_root,
            "source_cache": remote_cache,
            "tbc_results": remote_tbc_results,
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

    copied_tbc = []
    for item in plan.get("tbc_items", []):
        if plan["skip_existing"] and remote_exists(host, item["remote"]):
            copied_tbc.append({**item, "action": "skipped_existing"})
            continue
        remote_parent = posixpath.dirname(str(item["remote"]))
        ssh(host, f"mkdir -p {q(remote_parent)}")
        completed = run_command(["scp", item["local"], f"{host}:{remote_parent}/"])
        copied_tbc.append({**item, "action": "copied", "returncode": completed.returncode})
    response["copied_tbc"] = copied_tbc

    remote_manifest = f"{plan['remote_root']}/data/source_cache/{plan['dataset_id']}/source_manifest.json"
    remote_root_manifest = f"{plan['remote_root']}/data/source_manifest.json"
    remote_env = f"{plan['remote_root']}/data/pdc_dataset_env.sh"
    source_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": plan["dataset_id"],
        "display_name": plan.get("display_name"),
        "expected_cells": plan.get("expected_cells"),
        "cell_groups_relpath": plan.get("cell_groups_relpath"),
        "source_local_root": plan["local_xenium_root"],
        "remote_source_cache": plan["remote_source_cache"],
        "local_tbc_results": plan.get("local_tbc_results"),
        "remote_tbc_results": plan.get("remote_tbc_results"),
        "items": copied_raw,
        "tbc_items": copied_tbc,
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
        ssh(host, f"mkdir -p {q(posixpath.dirname(remote_manifest))} {q(posixpath.dirname(remote_root_manifest))}")
        run_command(["scp", tmp_name, f"{host}:{remote_manifest}"])
        run_command(["scp", tmp_name, f"{host}:{remote_root_manifest}"])
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    response["remote_source_manifest"] = remote_manifest
    env_text = "\n".join(
        [
            f"export PDC_CCI_DATASET_ID={q(str(plan['dataset_id']))}",
            f"export PDC_CCI_SOURCE_ROOT={q(str(plan['remote_source_cache']))}",
            f"export PDC_CCI_CELL_GROUPS_RELPATH={q(str(plan['cell_groups_relpath']))}",
            f"export PDC_CCI_TBC_RESULTS={q(str(plan['remote_tbc_results']))}",
            f"export PDC_CCI_EXPECTED_CELLS={q('' if plan.get('expected_cells') is None else str(plan.get('expected_cells')))}",
            f"export PDC_CCI_LOCAL_TBC_RESULTS={q('' if plan.get('local_tbc_results') is None else str(plan.get('local_tbc_results')))}",
            "",
        ]
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".sh", delete=False) as handle:
        handle.write(env_text)
        tmp_name = handle.name
    try:
        ssh(host, f"mkdir -p {q(posixpath.dirname(remote_env))}")
        run_command(["scp", tmp_name, f"{host}:{remote_env}"])
        # PDC shell scripts must be LF-only. Windows CRLF here silently appends
        # "\r" to sourced variables and turns valid dataset paths into misses.
        ssh(host, f"sed -i 's/\\r$//' {q(remote_env)}")
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    response["remote_dataset_env"] = remote_env
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage pyXenium CCI benchmark code and dataset source cache to PDC.")
    parser.add_argument("--host", default=DEFAULT_PDC_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--datasets-config", default=None)
    parser.add_argument("--local-xenium-root", default=None)
    parser.add_argument("--remote-source-root", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--include-smoke", action="store_true", default=True)
    parser.add_argument("--skip-smoke", action="store_false", dest="include_smoke")
    parser.add_argument("--include-spatialdata-zarr", action="store_true", default=False)
    parser.add_argument("--skip-spatialdata-zarr", action="store_false", dest="include_spatialdata_zarr")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    dataset = load_dataset_entry(repo_root, args.dataset_id, Path(args.datasets_config) if args.datasets_config else None)
    local_xenium_root = Path(args.local_xenium_root) if args.local_xenium_root else Path(str(dataset.get("local_xenium_root") or DEFAULT_LOCAL_XENIUM_ROOT))
    plan = build_stage_plan(
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        dataset=dataset,
        local_xenium_root=local_xenium_root,
        remote_root=args.remote_root,
        remote_source_root=args.remote_source_root,
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
