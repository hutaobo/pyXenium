from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04"
DEFAULT_PDC_DATASET_ROOT = "/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs"
DEFAULT_LOCAL_DATASET_ROOT = Path(r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs")
REQUIRED_DATA_FILES = (
    "cell_feature_matrix.h5",
    "cells.parquet",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv",
    "experiment.xenium",
    "metrics_summary.csv",
)
REPO_ITEMS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src",
    "docs/conf.py",
    "docs/tutorials/index.md",
    "docs/tutorials/gmi.md",
    "docs/tutorials/mechanostress_atera_pdc.ipynb",
    "docs/_static/tutorials/mechanostress_atera_pdc",
    "benchmarking/mechanostress_pdc",
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_command], check=check)


def remote_size(host: str, remote_path: str) -> int | None:
    completed = ssh(host, f"stat -c %s {q(remote_path)}", check=False)
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".git" in parts or ".pytest_cache" in parts or "_build" in parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def add_path_to_archive(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    if not source.exists():
        return
    if source.is_file():
        tar.add(source, arcname=arcname)
        return
    for child in source.rglob("*"):
        if child.is_file() and not should_skip(child):
            tar.add(child, arcname=str(Path(arcname) / child.relative_to(source)).replace(os.sep, "/"))


def create_repo_archive(repo_root: Path, archive_path: Path) -> list[dict[str, str]]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in REPO_ITEMS:
            source = repo_root / item
            if not source.exists():
                continue
            arcname = str(Path("repo") / item).replace(os.sep, "/")
            add_path_to_archive(tar, source, arcname)
            entries.append({"local": str(source), "archive": arcname})
    return entries


def build_plan(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    pdc_root = args.pdc_root.rstrip("/")
    pdc_dataset_root = args.pdc_dataset_root.rstrip("/")
    local_root = Path(args.local_dataset_root)
    archive_path = Path(args.archive_path) if args.archive_path else repo_root / ".codex_deps" / "mechanostress_pdc_stage_payload.tar.gz"
    data_items = []
    for name in REQUIRED_DATA_FILES:
        local_path = local_root / name
        remote_path = posixpath.join(pdc_dataset_root, name)
        local_bytes = local_path.stat().st_size if local_path.exists() else None
        remote_bytes = remote_size(args.host, remote_path)
        data_items.append(
            {
                "name": name,
                "local": str(local_path),
                "remote": remote_path,
                "local_bytes": local_bytes,
                "remote_bytes": remote_bytes,
                "copy_needed": local_bytes is not None and local_bytes != remote_bytes,
            }
        )
    return {
        "host": args.host,
        "pdc_root": pdc_root,
        "pdc_dataset_root": pdc_dataset_root,
        "repo_dir": posixpath.join(pdc_root, "repo"),
        "archive_path": str(archive_path),
        "data_items": data_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--pdc-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--pdc-dataset-root", default=DEFAULT_PDC_DATASET_ROOT)
    parser.add_argument("--local-dataset-root", default=str(DEFAULT_LOCAL_DATASET_ROOT))
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    plan = build_plan(args, repo_root)
    missing_local = [item["name"] for item in plan["data_items"] if item["local_bytes"] is None]
    if missing_local:
        raise FileNotFoundError(f"Missing local required files: {missing_local}")

    archive_entries = create_repo_archive(repo_root, Path(plan["archive_path"]))
    plan["archive_entries"] = archive_entries
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    ssh(args.host, f"mkdir -p {q(plan['pdc_root'])} {q(plan['pdc_dataset_root'])}")
    remote_archive = posixpath.join(plan["pdc_root"], "mechanostress_pdc_stage_payload.tar.gz")
    run(["scp", plan["archive_path"], f"{args.host}:{remote_archive}"])
    ssh(args.host, f"cd {q(plan['pdc_root'])} && tar -xzf {q(remote_archive)}")
    for item in plan["data_items"]:
        if item["copy_needed"]:
            run(["scp", item["local"], f"{args.host}:{item['remote']}"])
    print("[mechanostress-pdc] staging complete")


if __name__ == "__main__":
    main()
