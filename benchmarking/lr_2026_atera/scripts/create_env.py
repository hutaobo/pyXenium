from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


def _run(command: list[str], *, dry_run: bool) -> dict[str, object]:
    rendered = " ".join(command)
    if dry_run:
        return {"command": rendered, "returncode": 0, "stdout": "", "stderr": ""}
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": rendered,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _resolve_solver(preferred: str) -> str:
    solver = str(preferred).strip().lower()
    if solver == "mamba":
        return "mamba"
    if solver == "conda":
        return "conda"
    return "mamba" if shutil.which("mamba") else "conda"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and bootstrap an isolated benchmark environment.")
    parser.add_argument("--method", required=True, help="Environment or method name from methods.yaml or the env filename stem.")
    parser.add_argument("--methods-config", default=None)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--solver", choices=["auto", "conda", "mamba"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root(Path.cwd())
    benchmark_root = repo_root / (args.benchmark_root or Path("benchmarking") / "lr_2026_atera")
    methods_path = Path(args.methods_config) if args.methods_config else benchmark_root / "configs" / "methods.yaml"
    methods = yaml.safe_load(methods_path.read_text(encoding="utf-8")).get("methods", [])

    selected = None
    for method in methods:
        if method.get("slug") == args.method or method.get("env_name") == args.method:
            selected = method
            break
    if selected is None:
        raise SystemExit(f"Could not find method or environment {args.method!r} in {methods_path}")

    env_file = benchmark_root / selected["env_file"]
    env_name = selected["env_name"]
    bootstrap_script = benchmark_root / "envs" / "bootstrap_env.py"
    solver = _resolve_solver(args.solver)

    commands = [
        [solver, "env", "create", "--file", str(env_file), "--name", str(env_name)],
        [
            "conda",
            "run",
            "--name",
            str(env_name),
            "python",
            str(bootstrap_script),
            "--method",
            selected["slug"],
            "--repo-root",
            str(repo_root),
            "--benchmark-root",
            str(benchmark_root),
        ],
    ]
    payload = {
        "method": selected["slug"],
        "env_name": env_name,
        "env_file": str(env_file),
        "solver": solver,
        "commands": [],
    }
    for command in commands:
        payload["commands"].append(_run(command, dry_run=args.dry_run))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
