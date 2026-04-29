from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, object]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, cwd=str(cwd) if cwd else None)
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _render_r_install_expr(method: dict[str, object], project_dir: Path) -> str:
    install = dict(method.get("install", {}))
    cran = [pkg for pkg in install.get("cran", [])]
    github = [pkg for pkg in install.get("github", [])]
    bioc = [pkg for pkg in install.get("bioc", [])]
    commands = [
        "options(repos = c(CRAN = 'https://cloud.r-project.org'))",
        "if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv')",
        "if (!requireNamespace('remotes', quietly = TRUE)) install.packages('remotes')",
        "renv::consent(provided = TRUE)",
        f"dir.create('{project_dir.as_posix()}', recursive = TRUE, showWarnings = FALSE)",
        f"setwd('{project_dir.as_posix()}')",
        "if (!file.exists('renv.lock')) renv::init(bare = TRUE)",
    ]
    for pkg in cran:
        commands.append(f"if (!requireNamespace('{pkg}', quietly = TRUE)) install.packages('{pkg}')")
    if bioc:
        commands.append("if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')")
        for pkg in bioc:
            commands.append(f"if (!requireNamespace('{pkg}', quietly = TRUE)) BiocManager::install('{pkg}', ask = FALSE, update = FALSE)")
    for pkg in github:
        repo_pkg = pkg.split('/')[-1]
        commands.append(f"if (!requireNamespace('{repo_pkg}', quietly = TRUE)) remotes::install_github('{pkg}', upgrade = 'never')")
    commands.append("renv::snapshot(prompt = FALSE)")
    return "; ".join(commands)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the active method environment.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--methods-config", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve() if args.benchmark_root else repo_root / "benchmarking" / "cci_2026_atera"
    methods_path = Path(args.methods_config) if args.methods_config else benchmark_root / "configs" / "methods.yaml"
    methods = yaml.safe_load(methods_path.read_text(encoding="utf-8")).get("methods", [])
    method = next((item for item in methods if item.get("slug") == args.method or item.get("env_name") == args.method), None)
    if method is None:
        raise SystemExit(f"Unknown method {args.method!r}")

    install = dict(method.get("install", {}))
    payload = {"method": method["slug"], "steps": []}

    if method.get("language") == "python":
        if install.get("pip_editable_repo"):
            payload["steps"].append(_run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo_root)]))
        for pkg in install.get("pip", []):
            payload["steps"].append(_run([sys.executable, "-m", "pip", "install", pkg]))
        for pkg in install.get("pip_git", []):
            payload["steps"].append(_run([sys.executable, "-m", "pip", "install", pkg]))
    else:
        rscript = os.environ.get("RSCRIPT", "Rscript")
        project_dir = benchmark_root / "envs" / f"{method['env_name']}_project"
        expr = _render_r_install_expr(method, project_dir)
        payload["steps"].append(_run([rscript, "-e", expr], cwd=project_dir.parent))

    if install.get("manual_setup_required"):
        payload["manual_setup_required"] = True
        payload["notes"] = install.get("notes", "")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
