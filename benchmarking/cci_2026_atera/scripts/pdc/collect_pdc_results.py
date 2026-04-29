from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PDC_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"
REMOTE_DIRS = ("runs", "results", "logs", "reports", "data/source_cache/breast/source_manifest.json")


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_command],
        check=check,
    )


def remote_path_exists(host: str, path: str) -> bool:
    return ssh(host, f"test -e {q(path)}", check=False).returncode == 0


def collect(
    *,
    host: str,
    remote_root: str,
    local_benchmark_root: Path,
    tag: str,
    execute: bool,
    since_last: bool,
) -> dict[str, Any]:
    remote_root = remote_root.rstrip("/")
    destination_root = local_benchmark_root / "pdc_collected" / tag
    commands = []
    copied = []
    for rel in REMOTE_DIRS:
        source = f"{remote_root}/{rel}"
        if not remote_path_exists(host, source):
            copied.append({"remote": source, "status": "missing"})
            continue
        local_dest = destination_root / rel.replace("/", "_")
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        if since_last and local_dest.exists():
            copied.append({"remote": source, "local": str(local_dest), "status": "skipped_existing"})
            continue
        command = ["scp", "-r", f"{host}:{source}", str(local_dest)]
        commands.append(command)
        if execute:
            completed = run_command(command)
            copied.append({"remote": source, "local": str(local_dest), "status": "copied", "returncode": completed.returncode})
        else:
            copied.append({"remote": source, "local": str(local_dest), "status": "dry-run", "command": " ".join(command)})
    payload = {
        "kind": "pdc_collect_results",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "remote_root": remote_root,
        "local_destination": str(destination_root),
        "execute": execute,
        "since_last": since_last,
        "items": copied,
        "commands": [" ".join(command) for command in commands],
    }
    manifest_path = destination_root / "collect_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PDC CCI benchmark outputs to the local benchmark directory.")
    parser.add_argument("--host", default=DEFAULT_PDC_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--benchmark-root", default="benchmarking/cci_2026_atera")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--since-last", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("pdc_%Y%m%d_%H%M%S")
    payload = collect(
        host=args.host,
        remote_root=args.remote_root,
        local_benchmark_root=Path(args.benchmark_root),
        tag=tag,
        execute=args.execute,
        since_last=args.since_last,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
