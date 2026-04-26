from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PDC_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04"
DEFAULT_ACCOUNT = "naiss2026-4-680"
ALL_METHODS = (
    "pyxenium",
    "squidpy",
    "liana",
    "spatialdm",
    "laris",
    "cellphonedb",
    "stlearn",
    "cellchat",
    "commot",
    "giotto",
    "spatalk",
    "niches",
    "cellnest",
    "cellagentchat",
    "scild",
)


@dataclass(frozen=True)
class MethodConfig:
    language: str
    partition: str
    cpus: int
    memory: str
    time: str
    pip: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    r_packages: tuple[str, ...] = ()
    github: tuple[str, ...] = ()
    pdc_note: str = ""


METHODS: dict[str, MethodConfig] = {
    "pyxenium": MethodConfig("python", "shared", 16, "96G", "06:00:00", imports=("pyXenium",)),
    "squidpy": MethodConfig("python", "shared", 16, "160G", "12:00:00", pip=("setuptools<81", "squidpy", "omnipath", "zarr<3"), imports=("squidpy",)),
    "liana": MethodConfig("python", "memory", 16, "300G", "18:00:00", pip=("liana", "omnipath", "mudata", "decoupler"), imports=("liana",)),
    "spatialdm": MethodConfig("python", "memory", 16, "300G", "18:00:00", pip=("git+https://github.com/StatBiomed/SpatialDM.git", "SparseAEH"), imports=("spatialdm",)),
    "laris": MethodConfig("python", "shared", 16, "192G", "18:00:00", pip=("laris",), imports=("laris",)),
    "cellphonedb": MethodConfig("python", "shared", 16, "160G", "12:00:00", pip=("cellphonedb",), imports=("cellphonedb",)),
    "stlearn": MethodConfig("python", "memory", 16, "300G", "18:00:00", pip=("stlearn",), imports=("stlearn",)),
    "commot": MethodConfig("python", "memory", 16, "300G", "24:00:00", pip=("commot",), imports=("commot",)),
    "cellnest": MethodConfig(
        "python",
        "shared",
        16,
        "160G",
        "12:00:00",
        pip=("git+https://github.com/schwartzlab-methods/CellNEST.git",),
        imports=("cellnest", "CellNEST"),
        pdc_note="Dardel GPU is not NVIDIA/A100; run CPU/bounded adapter only unless ROCm support is confirmed.",
    ),
    "cellagentchat": MethodConfig(
        "python",
        "shared",
        16,
        "160G",
        "12:00:00",
        pip=("git+https://github.com/mcgilldinglab/CellAgentChat.git",),
        imports=("cellagentchat", "CellAgentChat"),
        pdc_note="Dardel GPU is not NVIDIA/A100; run CPU/bounded adapter only unless ROCm support is confirmed.",
    ),
    "scild": MethodConfig(
        "python",
        "shared",
        16,
        "160G",
        "12:00:00",
        pip=("git+https://github.com/jiatingyu-amss/SCILD.git",),
        imports=("SCILD", "Models.SCILD_main"),
        pdc_note="SCILD source layout may not be pip-installable; failure card is acceptable if import/API mapping fails.",
    ),
    "cellchat": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "future", "igraph", "NMF"),
        github=("jinworks/CellChat",),
    ),
    "giotto": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "igraph"),
        github=("drieslab/Giotto",),
    ),
    "spatalk": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "igraph"),
        github=("ZJUFanLab/SpaTalk",),
    ),
    "niches": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "SeuratObject"),
        github=("msraredon/NICHES",),
    ),
}


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
        ["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_command],
        check=check,
    )


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def python_env_setup(method: str, cfg: MethodConfig, root: str) -> str:
    env = f"{root}/envs/python/{method}"
    pip_packages = " ".join(q(pkg) for pkg in cfg.pip)
    import_checks = "\n".join(
        [
            "ok = False",
            f"for name in {list(cfg.imports)!r}:",
            "    try:",
            "        __import__(name)",
            "        print(f'import_ok {name}')",
            "        ok = True",
            "        break",
            "    except Exception as exc:",
            "        print(f'import_failed {name}: {exc}')",
            "if not ok:",
            "    raise SystemExit('no configured import succeeded')",
        ]
        if cfg.imports
        else ["import pyXenium; print('import_ok pyXenium')"]
    )
    return f"""
METHOD={q(method)}
OUT="$ROOT/runs/env_audit/{method}"
mkdir -p "$OUT"
trap 'rc=$?; mkdir -p "$OUT"; printf "# %s PDC method card\\n\\n- Status: failed\\n- Stage: env_setup\\n- Exit code: %s\\n- Note: {cfg.pdc_note}\\n" "$METHOD" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
module load python/3.12.3 >/dev/null 2>&1 || true
PYTHON_BIN="$(command -v python3 || command -v python)"
if [ ! -x {q(env)}/bin/python ]; then
  "$PYTHON_BIN" -m venv {q(env)}
fi
. {q(env)}/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$ROOT/repo"
if [ -n {q(pip_packages)} ]; then
  python -m pip install {pip_packages}
fi
python - <<'PY'
{import_checks}
PY
python -m pip freeze > "$OUT/pip_freeze.txt"
printf '{{"method":"{method}","status":"success","stage":"env_setup","language":"python"}}\\n' > "$OUT/run_summary.json"
"""


def r_env_setup(method: str, cfg: MethodConfig, root: str) -> str:
    r_lib = f"{root}/envs/r_libs/{method}"
    r_packages = ", ".join(repr(pkg) for pkg in cfg.r_packages)
    github = ", ".join(repr(repo) for repo in cfg.github)
    return f"""
METHOD={q(method)}
OUT="$ROOT/runs/env_audit/{method}"
mkdir -p "$OUT" {q(r_lib)}
trap 'rc=$?; mkdir -p "$OUT"; printf "# %s PDC method card\\n\\n- Status: failed\\n- Stage: r_env_setup\\n- Exit code: %s\\n" "$METHOD" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
module load cray-R/4.4.0 >/dev/null 2>&1 || true
export R_LIBS_USER={q(r_lib)}
Rscript - <<'RS'
repos <- c(CRAN = "https://cloud.r-project.org")
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
pkgs <- c({r_packages})
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]
if (length(missing)) install.packages(missing, lib = lib, repos = repos, Ncpus = 8)
if (!requireNamespace("remotes", quietly = TRUE, lib.loc = lib)) install.packages("remotes", lib = lib, repos = repos)
.libPaths(c(lib, .libPaths()))
repos_gh <- c({github})
for (repo in repos_gh) {{
  pkg <- basename(repo)
  if (!requireNamespace(pkg, quietly = TRUE)) {{
    remotes::install_github(repo, lib = lib, upgrade = "never", dependencies = TRUE)
  }}
}}
sessionInfo()
RS
Rscript -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); sink(file.path("{r_lib}", "sessionInfo.txt")); print(sessionInfo()); sink()'
printf '{{"method":"{method}","status":"success","stage":"env_setup","language":"r"}}\\n' > "$OUT/run_summary.json"
"""


def activate_for_method(method: str, cfg: MethodConfig, root: str) -> str:
    if cfg.language == "r":
        return f"""
module load cray-R/4.4.0 >/dev/null 2>&1 || true
. {q(root)}/envs/python/prep/bin/activate
export R_LIBS_USER={q(root)}/envs/r_libs/{method}
export PATH="$(dirname "$(command -v Rscript)"):$PATH"
"""
    return f"""
module load python/3.12.3 >/dev/null 2>&1 || true
. {q(root)}/envs/python/{method}/bin/activate
"""


def method_run_command(method: str, cfg: MethodConfig, root: str, stage: str) -> str:
    phase = "smoke" if stage == "smoke" else "full"
    max_lr = {"smoke": 100, "pilot": 500, "full": None}[stage]
    out_dir = f"{root}/runs/{stage}_common/{method}"
    extra = ""
    if method == "commot" and stage in {"pilot", "full"}:
        extra += " --chunk-id 0 --num-chunks 16"
        out_dir = f"{root}/runs/{stage}_common/{method}_chunk_0"
    if max_lr is not None:
        extra += f" --max-lr-pairs {max_lr}"
    if stage == "pilot":
        extra += " --bounded-mode full_cells_lr500_pilot"
    if cfg.language == "r":
        extra += " --rscript Rscript"
    return f"""
OUT={q(out_dir)}
mkdir -p "$OUT"
trap 'rc=$?; mkdir -p "$OUT"; printf "# {method} PDC method card\\n\\n- Status: failed\\n- Stage: {stage}\\n- Exit code: %s\\n- PDC note: {cfg.pdc_note}\\n" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
{activate_for_method(method, cfg, root)}
python "$ROOT/repo/benchmarking/lr_2026_atera/scripts/run_method.py" \
  --method {q(method)} \
  --input-manifest "$ROOT/data/input_manifest.json" \
  --benchmark-root "$ROOT" \
  --database-mode common-db \
  --phase {phase} \
  --output-dir "$OUT" \
  --gzip-standardized \
  --job-id {q(stage + "_common_" + method)}{extra}
"""


def prepare_command(root: str) -> str:
    return f"bash {q(root)}/repo/benchmarking/lr_2026_atera/scripts/pdc/prepare_pdc_bundle.sh"


def job_script(
    *,
    job_id: str,
    root: str,
    account: str,
    partition: str,
    cpus: int,
    memory: str,
    time_limit: str,
    body: str,
) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH -t {time_limit}
#SBATCH -c {cpus}
#SBATCH --mem={memory}
#SBATCH -J {job_id[:64]}
#SBATCH -o {root}/logs/{job_id}.stdout.log
#SBATCH -e {root}/logs/{job_id}.stderr.log

set -euo pipefail
export ROOT={q(root)}
export PDC_LR_ROOT="$ROOT"
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$ROOT/repo/src:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
mkdir -p "$ROOT"/{{logs,runs,results,reports,tmp,slurm,envs/python,envs/r_libs}}
mkdir -p "$ROOT/configs"
cp -f "$ROOT"/repo/benchmarking/lr_2026_atera/configs/* "$ROOT/configs/" 2>/dev/null || true
cd "$ROOT/repo"
/usr/bin/time -v bash -lc {q(body)} 2> >(tee -a "$ROOT/logs/{job_id}.resource.log" >&2)
"""


def build_jobs(methods: list[str], root: str, account: str, stages: set[str], include_full: bool) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if "prepare" in stages:
        jobs.append(
            {
                "job_id": "pdc_prepare_full_bundle",
                "job_type": "prepare",
                "method": "prep",
                "partition": "shared",
                "cpus": 16,
                "memory": "96G",
                "time": "02:00:00",
                "dependencies": [],
                "script": job_script(
                    job_id="pdc_prepare_full_bundle",
                    root=root,
                    account=account,
                    partition="shared",
                    cpus=16,
                    memory="96G",
                    time_limit="02:00:00",
                    body=prepare_command(root),
                ),
            }
        )
    for method in methods:
        cfg = METHODS[method]
        env_id = f"pdc_env_{method}"
        if "env" in stages:
            body = python_env_setup(method, cfg, root) if cfg.language == "python" else r_env_setup(method, cfg, root)
            jobs.append(
                {
                    "job_id": env_id,
                    "job_type": "env_setup",
                    "method": method,
                    "partition": "shared" if cfg.language == "python" else "memory",
                    "cpus": 8 if cfg.language == "python" else 16,
                    "memory": "64G" if cfg.language == "python" else "160G",
                    "time": "04:00:00" if cfg.language == "python" else "12:00:00",
                    "dependencies": [],
                    "script": job_script(
                        job_id=env_id,
                        root=root,
                        account=account,
                        partition="shared" if cfg.language == "python" else "memory",
                        cpus=8 if cfg.language == "python" else 16,
                        memory="64G" if cfg.language == "python" else "160G",
                        time_limit="04:00:00" if cfg.language == "python" else "12:00:00",
                        body=body,
                    ),
                }
            )
        prior_stage_id = None
        for stage in ("smoke", "pilot", "full"):
            if stage == "full" and not include_full:
                continue
            if stage not in stages and not (stage == "full" and include_full):
                continue
            run_id = f"pdc_{stage}_common_{method}"
            deps = []
            if "prepare" in stages:
                deps.append("pdc_prepare_full_bundle")
            if "env" in stages:
                deps.append(env_id)
            if prior_stage_id:
                deps.append(prior_stage_id)
            jobs.append(
                {
                    "job_id": run_id,
                    "job_type": f"{stage}_common",
                    "method": method,
                    "partition": cfg.partition,
                    "cpus": cfg.cpus,
                    "memory": cfg.memory,
                    "time": cfg.time,
                    "dependencies": deps,
                    "script": job_script(
                        job_id=run_id,
                        root=root,
                        account=account,
                        partition=cfg.partition,
                        cpus=cfg.cpus,
                        memory=cfg.memory,
                        time_limit=cfg.time,
                        body=method_run_command(method, cfg, root, stage),
                    ),
                }
            )
            prior_stage_id = run_id
    return jobs


def write_remote_text(host: str, remote_path: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".sbatch", delete=False) as handle:
        handle.write(text)
        tmp_name = handle.name
    try:
        ssh(host, f"mkdir -p {q(posixpath.dirname(remote_path))}")
        run_command(["scp", tmp_name, f"{host}:{remote_path}"])
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def submit_jobs(host: str, root: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    submitted: list[dict[str, Any]] = []
    slurm_ids: dict[str, str] = {}
    for job in jobs:
        script_path = f"{root}/slurm/{job['job_id']}.sbatch"
        write_remote_text(host, script_path, str(job["script"]))
        missing_deps = [dep for dep in job.get("dependencies", []) if dep not in slurm_ids]
        if missing_deps:
            submitted.append(
                {k: v for k, v in job.items() if k != "script"}
                | {"script_path": script_path, "status": "skipped_missing_dependency", "missing_dependencies": missing_deps}
            )
            continue
        dep_ids = [slurm_ids[dep] for dep in job.get("dependencies", []) if dep in slurm_ids]
        command = f"sbatch --parsable"
        if dep_ids:
            command += " --dependency=afterok:" + ":".join(dep_ids)
        command += f" {q(script_path)}"
        completed = ssh(host, command, check=False)
        if completed.returncode != 0:
            submitted.append(
                {k: v for k, v in job.items() if k != "script"}
                | {
                    "script_path": script_path,
                    "status": "submit_failed",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            continue
        slurm_id = completed.stdout.strip().splitlines()[-1]
        slurm_ids[str(job["job_id"])] = slurm_id
        submitted.append(
            {k: v for k, v in job.items() if k != "script"}
            | {"script_path": script_path, "status": "submitted", "slurm_job_id": slurm_id}
        )
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and submit a PDC Slurm matrix for LR benchmarking.")
    parser.add_argument("--host", default=DEFAULT_PDC_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--stages", default="prepare,env,smoke,pilot")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    stages = {item.strip().lower() for item in args.stages.split(",") if item.strip()}
    jobs = build_jobs(methods, args.remote_root.rstrip("/"), args.account, stages, args.include_full)
    manifest: dict[str, Any] = {
        "kind": "pdc_job_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "remote_root": args.remote_root.rstrip("/"),
        "account": args.account,
        "methods": methods,
        "stages": sorted(stages),
        "include_full": args.include_full,
        "jobs": [{k: v for k, v in job.items() if k != "script"} for job in jobs],
    }
    if args.submit:
        submitted = submit_jobs(args.host, args.remote_root.rstrip("/"), jobs)
        manifest["submitted_jobs"] = submitted
        manifest["jobs"] = submitted
        remote_manifest = f"{args.remote_root.rstrip('/')}/logs/pdc_job_matrix_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
            tmp_name = handle.name
        try:
            run_command(["scp", tmp_name, f"{args.host}:{remote_manifest}"])
            manifest["remote_manifest"] = remote_manifest
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
