from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RUN_ID = "publication_benchmark_24h_20260511"
PDC_HOST = "pdc"
PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"
PDC_BREAST_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04"
A100_HOST = "taobo.hu@sscb-a100.scilifelab.se"
A100_LEGACY_ROOT = "/data/taobo.hu/pyxenium_lr_benchmark_2026-04"
A100_ROOT = "/data/taobo.hu/pyxenium_cci_benchmark_2026-04"
PDC_ACCOUNT = "naiss2026-4-680"

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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def benchmark_root() -> Path:
    return repo_root() / "benchmarking" / "cci_2026_atera"


def local_run_root() -> Path:
    return benchmark_root() / "results" / RUN_ID


def pdc_run_root() -> str:
    return f"{PDC_ROOT}/runs/{RUN_ID}"


def a100_run_root() -> str:
    return f"{A100_ROOT}/runs/{RUN_ID}"


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], *, check: bool = False, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, timeout=timeout)


def ssh(host: str, command: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, command],
        timeout=timeout,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **payload}, sort_keys=True) + "\n")


def event(payload: dict[str, Any]) -> None:
    append_jsonl(local_run_root() / "job_events.jsonl", payload)


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" in parts or ".git" in parts or ".pytest_cache" in parts or path.suffix in {".pyc", ".pyo"}


def stage_repo_to_pdc(force: bool = False) -> dict[str, Any]:
    root = repo_root()
    archive = local_run_root() / "staging" / "pdc_repo_payload.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(archive, "w:gz") as tar:
        for item in REPO_ITEMS:
            source = root / item
            if not source.exists():
                continue
            arcbase = Path("repo") / item
            if source.is_file():
                tar.add(source, arcname=str(arcbase).replace(os.sep, "/"))
                entries.append(str(source))
            else:
                for child in source.rglob("*"):
                    if child.is_file() and not should_skip(child):
                        tar.add(child, arcname=str(arcbase / child.relative_to(source)).replace(os.sep, "/"))
                        entries.append(str(child))

    ssh(PDC_HOST, f"mkdir -p {q(PDC_ROOT)}/tmp {q(PDC_ROOT)}/repo {q(PDC_ROOT)}/logs {q(PDC_ROOT)}/slurm {q(pdc_run_root())}")
    completed = run_command(["scp", str(archive), f"{PDC_HOST}:{PDC_ROOT}/tmp/{archive.name}"])
    if completed.returncode == 0:
        extract = f"tar -xzf {q(PDC_ROOT + '/tmp/' + archive.name)} -C {q(PDC_ROOT)} && rm -f {q(PDC_ROOT + '/tmp/' + archive.name)}"
        extract_result = ssh(PDC_HOST, extract)
    else:
        extract_result = completed
    payload = {
        "action": "stage_repo_to_pdc",
        "archive": str(archive),
        "entries": len(entries),
        "scp_returncode": completed.returncode,
        "extract_returncode": extract_result.returncode,
        "stdout": completed.stdout + extract_result.stdout,
        "stderr": completed.stderr + extract_result.stderr,
    }
    write_json(local_run_root() / "stage_pdc_repo.json", payload)
    event(payload)
    return payload


def _write_remote_text(host: str, remote_path: str, text: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        tmp = handle.name
    try:
        ssh(host, f"mkdir -p {q(posixpath.dirname(remote_path))}")
        return run_command(["scp", tmp, f"{host}:{remote_path}"])
    finally:
        Path(tmp).unlink(missing_ok=True)


def _submit_pdc_job(job_id: str, body: str, *, partition: str, cpus: int, memory: str, time_limit: str) -> dict[str, Any]:
    script_path = f"{PDC_ROOT}/slurm/{RUN_ID}_{job_id}.sbatch"
    script = f"""#!/bin/bash
#SBATCH -A {PDC_ACCOUNT}
#SBATCH -p {partition}
#SBATCH -t {time_limit}
#SBATCH -c {cpus}
#SBATCH --mem={memory}
#SBATCH -J {job_id[:64]}
#SBATCH -o {PDC_ROOT}/logs/{RUN_ID}_{job_id}.stdout.log
#SBATCH -e {PDC_ROOT}/logs/{RUN_ID}_{job_id}.stderr.log

set -euo pipefail
export ROOT={q(PDC_ROOT)}
export BREAST_ROOT={q(PDC_BREAST_ROOT)}
export RUN_ROOT={q(pdc_run_root())}
export TMPDIR="$ROOT/tmp/{RUN_ID}_{job_id}"
mkdir -p "$TMPDIR" "$RUN_ROOT" "$ROOT/logs"
{body}
"""
    upload = _write_remote_text(PDC_HOST, script_path, script)
    if upload.returncode != 0:
        payload = {"job_id": job_id, "status": "script_upload_failed", "returncode": upload.returncode, "stderr": upload.stderr}
        event(payload)
        return payload
    submitted = ssh(PDC_HOST, f"sbatch --parsable {q(script_path)}")
    payload = {
        "job_id": job_id,
        "status": "submitted" if submitted.returncode == 0 else "submit_failed",
        "slurm_job_id": submitted.stdout.strip(),
        "returncode": submitted.returncode,
        "script_path": script_path,
        "stderr": submitted.stderr,
    }
    event(payload)
    return payload


def _python_env_bootstrap(env_name: str) -> str:
    return f"""
module load python/3.12.3 >/dev/null 2>&1 || module load cray-python/3.11.7 >/dev/null 2>&1 || module load cray-python/3.11.5 >/dev/null 2>&1 || module load python/3.10 >/dev/null 2>&1 || module load cray-python/3.10.10 >/dev/null 2>&1 || true
PYTHON_BIN="$(command -v python3 || command -v python)"
ENV="$ROOT/envs/publication24h/{env_name}"
if [ ! -x "$ENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$ENV"
fi
. "$ENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
if ! python -m pip install -e "$ROOT/repo"; then
  echo "editable install with dependencies failed; falling back to no-deps plus minimum runtime packages" >&2
  python -m pip install -e "$ROOT/repo" --no-deps
  python -m pip install numpy pandas scipy matplotlib PyYAML anndata scanpy zarr h5py pyarrow scikit-learn
fi
"""


def submit_pdc_24h_jobs(force: bool = False, only_jobs: set[str] | None = None) -> list[dict[str, Any]]:
    submitted_path = local_run_root() / "submitted_pdc_jobs.json"
    if submitted_path.exists() and not force and not only_jobs:
        payload = json.loads(submitted_path.read_text(encoding="utf-8"))
        event({"action": "submit_pdc_24h_jobs", "status": "skipped_existing_manifest", "path": str(submitted_path)})
        return payload

    ssh(PDC_HOST, f"mkdir -p {q(pdc_run_root())}/{{synthetic_truth,false_positive_controls_validation_v2,method_audits}} {q(PDC_ROOT)}/logs {q(PDC_ROOT)}/slurm")
    jobs: list[dict[str, Any]] = []
    synthetic_body = _python_env_bootstrap("synthetic") + """
python "$ROOT/repo/benchmarking/cci_2026_atera/scripts/run_synthetic_truth_benchmark.py" \
  --output-dir "$RUN_ROOT/synthetic_truth" \
  --cells-per-type 180 \
  --seed 20260511
"""
    if only_jobs is None or "synthetic" in only_jobs or "synthetic_truth" in only_jobs:
        jobs.append(_submit_pdc_job("pub24h_synthetic_truth", synthetic_body, partition="shared", cpus=8, memory="80G", time_limit="04:00:00"))

    validation_body = _python_env_bootstrap("validation") + """
python "$ROOT/repo/benchmarking/cci_2026_atera/scripts/topolink_cci_validation_v2.py" \
  --root "$BREAST_ROOT" \
  --output-dir "$RUN_ROOT/false_positive_controls_validation_v2" \
  --n-label-permutations 500 \
  --n-spatial-permutations 300 \
  --n-spatial-matched-pairs 120 \
  --n-matched-controls 250 \
  --n-downstream-permutations 120 \
  --n-bootstraps 5
"""
    if only_jobs is None or "false_positive" in only_jobs or "validation" in only_jobs:
        jobs.append(_submit_pdc_job("pub24h_false_positive_controls", validation_body, partition="memory", cpus=32, memory="300G", time_limit="24:00:00"))

    for method, install_cmd, import_expr in [
        ("fastccc", "python -m pip install fastccc FastCCC || true", "import importlib.util; assert importlib.util.find_spec('fastccc') or importlib.util.find_spec('FastCCC')"),
        ("copulacci", "python -m pip install copulacci || true", "import importlib.util; assert importlib.util.find_spec('copulacci')"),
    ]:
        body = _python_env_bootstrap(f"audit_{method}") + f"""
OUT="$RUN_ROOT/method_audits/{method}"
mkdir -p "$OUT"
set +e
{install_cmd} > "$OUT/install.stdout.log" 2> "$OUT/install.stderr.log"
python - <<'PY' > "$OUT/audit.stdout.log" 2> "$OUT/audit.stderr.log"
{import_expr}
print("import_ok")
PY
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  printf '{{"method":"{method}","status":"success","stage":"env_api_audit"}}\\n' > "$OUT/run_summary.json"
else
  printf '# {method} method card\\n\\n- Status: env_or_api_audit_failed\\n- Scope: publication 24h subset candidate\\n- Install log: install.stderr.log\\n- Audit log: audit.stderr.log\\n' > "$OUT/method_card.md"
fi
exit 0
"""
        if only_jobs is None or method in only_jobs or f"audit_{method}" in only_jobs:
            jobs.append(_submit_pdc_job(f"pub24h_audit_{method}", body, partition="shared", cpus=4, memory="32G", time_limit="02:00:00"))

    nichenet_body = """
module load cray-R/4.4.0 >/dev/null 2>&1 || true
OUT="$RUN_ROOT/method_audits/nichenet"
RLIB="$ROOT/envs/publication24h/audit_nichenet/Rlib"
mkdir -p "$OUT" "$RLIB"
export R_LIBS_USER="$RLIB"
set +e
Rscript - <<'RS' > "$OUT/audit.stdout.log" 2> "$OUT/audit.stderr.log"
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")
if (!requireNamespace("nichenetr", quietly=TRUE)) remotes::install_github("saeyslab/nichenetr", upgrade="never")
stopifnot(requireNamespace("nichenetr", quietly=TRUE))
cat("import_ok\\n")
RS
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  printf '{"method":"nichenet","status":"success","stage":"env_api_audit"}\\n' > "$OUT/run_summary.json"
else
  printf '# NicheNet method card\\n\\n- Status: env_or_api_audit_failed\\n- Scope: publication 24h subset candidate\\n- Audit log: audit.stderr.log\\n' > "$OUT/method_card.md"
fi
exit 0
"""
    if only_jobs is None or "nichenet" in only_jobs or "audit_nichenet" in only_jobs:
        jobs.append(_submit_pdc_job("pub24h_audit_nichenet", nichenet_body, partition="shared", cpus=8, memory="64G", time_limit="04:00:00"))
    if only_jobs is None:
        write_json(submitted_path, jobs)
    else:
        write_json(local_run_root() / f"submitted_pdc_jobs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json", jobs)
    return jobs


def pdc_monitor() -> dict[str, Any]:
    output_json = local_run_root() / "pdc_live_status.json"
    output_md = local_run_root() / "pdc_live_status.md"
    cmd = [
        "python",
        str(benchmark_root() / "scripts" / "pdc" / "monitor_pdc_jobs.py"),
        "--host",
        PDC_HOST,
        "--remote-root",
        PDC_ROOT,
        "--local-root",
        str(benchmark_root()),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    completed = run_command(cmd, timeout=120)
    payload = {"action": "pdc_monitor", "returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr": completed.stderr}
    event(payload)
    return payload


def a100_monitor() -> dict[str, Any]:
    rescue_root = posixpath.join(A100_LEGACY_ROOT, "runs", "final_closeout_20260511", "a100_rescue")
    remote = f"""
set +e
echo '---HOST---'
hostname
echo '---ROOTS---'
for d in {q(A100_ROOT)} {q(A100_LEGACY_ROOT)}; do test -d "$d" && echo "exists $d" || echo "missing $d"; done
echo '---GPU---'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo '---RESCUE_PROCESSES---'
ps -eo pid,ppid,etime,rss,cmd | grep -E 'a100_r_method_rescue|niches|spatalk' | grep -v grep | head -80 || true
echo '---RECENT_SUMMARIES---'
timeout 25s find {q(rescue_root)} -maxdepth 5 -name run_summary.json -printf '%T@\\t%p\\n' 2>/dev/null | sort -nr | head -20 || true
"""
    try:
        completed = ssh(A100_HOST, remote, timeout=90)
        parsed = {"action": "a100_monitor", "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as exc:
        parsed = {
            "action": "a100_monitor",
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": f"A100 monitor timed out after {exc.timeout}s; preserving last known remote job state until next heartbeat.",
        }
    write_json(local_run_root() / "a100_live_status.json", parsed)
    event({"action": "a100_monitor", "returncode": parsed["returncode"], "stderr": parsed["stderr"]})
    return parsed


def collect_pdc() -> dict[str, Any]:
    destination = local_run_root() / "pdc_collected"
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = run_command(["scp", "-r", f"{PDC_HOST}:{pdc_run_root()}", str(destination)], timeout=600)
    payload = {"action": "collect_pdc", "returncode": completed.returncode, "destination": str(destination), "stderr": completed.stderr}
    event(payload)
    return payload


def refresh_local_figures() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root() / "src")
    commands = [
        ["python", str(benchmark_root() / "scripts" / "make_cross_method_comparison_figures.py")],
    ]
    rows = []
    for command in commands:
        completed = subprocess.run(command, cwd=repo_root(), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        rows.append({"command": " ".join(command), "returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]})
        event({"action": "local_command", **rows[-1]})
    return {"action": "refresh_local_figures", "commands": rows}


def summarize() -> dict[str, Any]:
    run = local_run_root()
    rows = []
    for path in sorted(run.rglob("run_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"status": "unreadable", "error": str(exc)}
        rows.append({"path": str(path.relative_to(run)), **payload})
    cards = [str(path.relative_to(run)) for path in sorted(run.rglob("method_card.md"))]
    pdc_status = {}
    pdc_json = run / "pdc_live_status.json"
    if pdc_json.exists():
        pdc_status = json.loads(pdc_json.read_text(encoding="utf-8"))
    a100_status = {}
    a100_json = run / "a100_live_status.json"
    if a100_json.exists():
        a100_status = json.loads(a100_json.read_text(encoding="utf-8"))

    summary = {
        "run_id": RUN_ID,
        "updated_at": now(),
        "local_run_root": str(run),
        "pdc_run_root": pdc_run_root(),
        "a100_run_root": a100_run_root(),
        "run_summaries": rows,
        "method_cards": cards,
        "pdc_queue_jobs": len(pdc_status.get("squeue", [])) if isinstance(pdc_status, dict) else None,
        "a100_monitor_returncode": a100_status.get("returncode") if isinstance(a100_status, dict) else None,
    }
    write_json(run / "publication_benchmark_24h_summary.json", summary)

    lines = [
        "# Publication Benchmark 24h Summary",
        "",
        f"- Updated: `{summary['updated_at']}`",
        f"- Local root: `{summary['local_run_root']}`",
        f"- PDC root: `{summary['pdc_run_root']}`",
        f"- A100 root: `{summary['a100_run_root']}`",
        f"- Collected run summaries: `{len(rows)}`",
        f"- Method cards: `{len(cards)}`",
        f"- PDC queue jobs visible: `{summary['pdc_queue_jobs']}`",
        "",
        "## Run Summaries",
        "",
        "| path | method | status | stage | rows |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row.get('path','')}` | {row.get('method','')} | {row.get('status','')} | {row.get('stage', row.get('phase', ''))} | {row.get('n_rows','')} |")
    lines.extend(["", "## Method Cards", ""])
    for card in cards:
        lines.append(f"- `{card}`")
    (run / "publication_benchmark_24h_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def snapshot_inputs() -> None:
    run = local_run_root()
    run.mkdir(parents=True, exist_ok=True)
    source = benchmark_root() / "results" / "method_completion_matrix.tsv"
    if source.exists():
        target = run / "method_completion_matrix_start.tsv"
        if not target.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(
        run / "run_config.json",
        {
            "run_id": RUN_ID,
            "created_at": now(),
            "pdc_root": PDC_ROOT,
            "pdc_breast_root": PDC_BREAST_ROOT,
            "a100_root": A100_ROOT,
            "a100_legacy_root": A100_LEGACY_ROOT,
            "policy": {
                "do_not_rerun_completed_full_methods": True,
                "do_not_interrupt_running_a100_rescue": True,
                "max_auto_fix_reruns_per_job": 2,
                "no_silent_changes_to_core_statistics": True,
            },
        },
    )


def run_once(*, submit: bool, stage_pdc: bool, collect: bool, force_submit: bool, only_jobs: set[str] | None = None) -> None:
    snapshot_inputs()
    if stage_pdc:
        stage_repo_to_pdc(force=force_submit)
    if submit:
        submit_pdc_24h_jobs(force=force_submit, only_jobs=only_jobs)
    pdc_monitor()
    a100_monitor()
    if collect:
        collect_pdc()
    refresh_local_figures()
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervise the 24h TopoLink-CCI publication benchmark closeout.")
    parser.add_argument("--submit", action="store_true", help="Submit the PDC 24h job batch if not already submitted.")
    parser.add_argument("--stage-pdc", action="store_true", help="Stage current repo/scripts to the PDC benchmark root.")
    parser.add_argument("--collect", action="store_true", help="Collect PDC 24h output tree to local run root.")
    parser.add_argument("--force-submit", action="store_true", help="Submit jobs even if a local submitted manifest exists.")
    parser.add_argument("--only-jobs", default="", help="Comma-separated subset: synthetic,validation,fastccc,copulacci,nichenet.")
    parser.add_argument("--loop-hours", type=float, default=0.0)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    args = parser.parse_args()

    if args.loop_hours > 0:
        deadline = time.time() + args.loop_hours * 3600.0
        first = True
        while time.time() < deadline:
            only_jobs = {item.strip().lower() for item in args.only_jobs.split(",") if item.strip()}
            run_once(submit=args.submit and first, stage_pdc=args.stage_pdc and first, collect=args.collect, force_submit=args.force_submit, only_jobs=only_jobs or None)
            first = False
            time.sleep(max(60.0, args.interval_minutes * 60.0))
        return
    only_jobs = {item.strip().lower() for item in args.only_jobs.split(",") if item.strip()}
    run_once(submit=args.submit, stage_pdc=args.stage_pdc, collect=args.collect, force_submit=args.force_submit, only_jobs=only_jobs or None)


if __name__ == "__main__":
    main()
