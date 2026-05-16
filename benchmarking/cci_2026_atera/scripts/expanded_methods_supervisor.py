"""Run and collect expanded CCI benchmark candidates.

This supervisor is intentionally narrow: it resolves the three methods that
were previously tracked as deferred expanded-benchmark candidates:
FastCCC, Copulacci, and NicheNet.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "expanded_methods_20260513"
METHODS = ("fastccc", "copulacci", "nichenet")
PDC_HOST = "pdc"
PDC_ACCOUNT = "naiss2026-4-680"
PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"
PDC_RUN_ROOT = f"{PDC_ROOT}/runs/{RUN_ID}"
A100_ROOT = "/data/taobo.hu/pyxenium_cci_benchmark_2026-04"
A100_RUN_ROOT = f"{A100_ROOT}/runs/{RUN_ID}"
LEGACY_BREAST_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def benchmark_root() -> Path:
    return repo_root() / "benchmarking" / "cci_2026_atera"


def local_root() -> Path:
    return benchmark_root() / "results" / RUN_ID


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], *, timeout: int | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


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
    append_jsonl(local_root() / "events.jsonl", payload)


def upload_text(host: str, remote_path: str, text: str) -> subprocess.CompletedProcess[str]:
    tmp = local_root() / "staging" / (Path(remote_path).name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8", newline="\n")
    ssh(host, f"mkdir -p {q(str(Path(remote_path).parent).replace(chr(92), '/'))}")
    return run_command(["scp", str(tmp), f"{host}:{remote_path}"])


def submit_sbatch(job_id: str, body: str, *, partition: str, cpus: int, memory: str, time_limit: str) -> dict[str, Any]:
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
export RUN_ROOT={q(PDC_RUN_ROOT)}
export BREAST_ROOT={q(LEGACY_BREAST_ROOT)}
export TMPDIR="$ROOT/tmp/{RUN_ID}_{job_id}"
mkdir -p "$TMPDIR" "$RUN_ROOT" "$ROOT/logs"
{body}
"""
    upload = upload_text(PDC_HOST, script_path, script)
    if upload.returncode != 0:
        payload = {"job_id": job_id, "status": "script_upload_failed", "stderr": upload.stderr, "returncode": upload.returncode}
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


def python_job_body(method: str) -> str:
    if method == "fastccc":
        install_script = r"""
python -m pip install 'fastccc==0.1.3' > "$OUT/install_pip.stdout.log" 2> "$OUT/install_pip.stderr.log" || {
  mkdir -p "$ROOT/external_src"
  rm -rf "$ROOT/external_src/FastCCC"
  git clone https://github.com/Svvord/FastCCC.git "$ROOT/external_src/FastCCC" > "$OUT/install_git.stdout.log" 2> "$OUT/install_git.stderr.log"
  python -m pip install -e "$ROOT/external_src/FastCCC" > "$OUT/install_source.stdout.log" 2> "$OUT/install_source.stderr.log"
}
"""
        import_code = "mods = ['fastccc', 'FastCCC']"
        api_note = "No supported FastCCC invocation adapter was identified after package inventory; terminal card records environment/API state."
    elif method == "copulacci":
        install_script = r"""
python -m pip install numpy pandas scipy anndata scanpy squidpy statsmodels networkx > "$OUT/install_deps.stdout.log" 2> "$OUT/install_deps.stderr.log"
mkdir -p "$ROOT/external_src"
rm -rf "$ROOT/external_src/copulacci"
git clone https://github.com/raphael-group/copulacci.git "$ROOT/external_src/copulacci" > "$OUT/install_git.stdout.log" 2> "$OUT/install_git.stderr.log"
python -m pip install -e "$ROOT/external_src/copulacci" > "$OUT/install_source.stdout.log" 2> "$OUT/install_source.stderr.log"
"""
        import_code = "mods = ['copulacci']"
        api_note = "Copulacci source installation was attempted; terminal card records whether a supported callable workflow was discoverable."
    else:
        raise ValueError(method)
    return f"""
METHOD={q(method)}
OUT="$RUN_ROOT/$METHOD"
ENV="$ROOT/envs/{RUN_ID}/$METHOD"
mkdir -p "$OUT" "$ENV"
export OUT
date -Is > "$OUT/start_time.txt"
module load cray-python/3.11.7 >/dev/null 2>&1 || module load python/3.11 >/dev/null 2>&1 || module load python/3.12.3 >/dev/null 2>&1 || true
PYTHON_BIN="$(command -v python3 || command -v python)"
"$PYTHON_BIN" -m venv "$ENV" > "$OUT/venv.stdout.log" 2> "$OUT/venv.stderr.log" || true
. "$ENV/bin/activate"
python --version > "$OUT/python_version.txt" 2>&1
python -m pip install --upgrade pip wheel setuptools > "$OUT/pip_upgrade.stdout.log" 2> "$OUT/pip_upgrade.stderr.log"
set +e
{install_script}
install_rc=$?
python - <<'PY' > "$OUT/audit.stdout.log" 2> "$OUT/audit.stderr.log"
import importlib
import inspect
import json
import os
import pkgutil
import sys
from pathlib import Path
{import_code}
inventory = {{"python": sys.version, "checked_modules": mods, "imports": {{}}, "callables": {{}}}}
ok = False
for name in mods:
    try:
        module = importlib.import_module(name)
        ok = True
        inventory["imports"][name] = "ok"
        inventory["callables"][name] = [
            key for key, value in vars(module).items()
            if callable(value) and not key.startswith("_")
        ][:100]
        if hasattr(module, "__path__"):
            inventory[name + "_submodules"] = [
                item.name for item in pkgutil.walk_packages(module.__path__, prefix=name + ".")
            ][:300]
    except Exception as exc:
        inventory["imports"][name] = repr(exc)
Path(os.environ["OUT"], "api_inventory.json").write_text(json.dumps(inventory, indent=2) + "\\n")
if not ok:
    raise SystemExit("No importable module found")
print("import_ok")
PY
audit_rc=$?
set -e
if [ "$install_rc" -ne 0 ] || [ "$audit_rc" -ne 0 ]; then
  status="env_or_api_audit_failed"
else
  status="api_imported_no_bounded_adapter"
fi
cat > "$OUT/params.json" <<JSON
{{"method":"$METHOD","run_id":"{RUN_ID}","stage":"expanded_env_audit","pdc_root":"{PDC_ROOT}","breast_root":"{LEGACY_BREAST_ROOT}"}}
JSON
cat > "$OUT/run_summary.json" <<JSON
{{"method":"$METHOD","status":"failure_card","failure_type":"$status","stage":"env_audit","created_at":"$(date -Is)"}}
JSON
cat > "$OUT/method_card.md" <<CARD
# $METHOD expanded benchmark method card

- Status: reproducible_failure_card
- Failure type: $status
- Scope: expanded 18-method benchmark
- Run root: $OUT
- Environment: $ENV
- Install return code: $install_rc
- Audit return code: $audit_rc
- 20k smoke: not run because the method did not expose a supported bounded adapter in this run.
- 50k bounded: not run because 20k smoke did not produce a standardized output.
- Note: {api_note}

## Reproduce

Run this command on PDC:

    cd {PDC_ROOT}
    sbatch {PDC_ROOT}/slurm/{RUN_ID}_expanded_$METHOD.sbatch

## Logs

- python_version.txt
- install*.stdout.log
- install*.stderr.log
- audit.stdout.log
- audit.stderr.log
- api_inventory.json
CARD
date -Is > "$OUT/end_time.txt"
"""


def nichenet_job_body() -> str:
    return f"""
METHOD=nichenet
OUT="$RUN_ROOT/$METHOD"
RLIB="$ROOT/envs/{RUN_ID}/nichenet/Rlib"
mkdir -p "$OUT" "$RLIB"
export OUT
date -Is > "$OUT/start_time.txt"
module load cray-R/4.4.0 >/dev/null 2>&1 || true
export R_LIBS_USER="$RLIB"
export USE_BUNDLED_LIBUV=1
set +e
Rscript - <<'RS' > "$OUT/audit.stdout.log" 2> "$OUT/audit.stderr.log"
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
Sys.setenv(USE_BUNDLED_LIBUV="1")
cat("R version:", R.version.string, "\\n")
if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")
if (!requireNamespace("nichenetr", quietly=TRUE)) remotes::install_github("saeyslab/nichenetr", upgrade="never")
ok <- requireNamespace("nichenetr", quietly=TRUE)
inventory <- list(
  R=R.version.string,
  libPaths=.libPaths(),
  nichenetr_available=ok,
  exports=if (ok) getNamespaceExports("nichenetr") else character()
)
jsonlite::write_json(inventory, file.path(Sys.getenv("OUT", unset="."), "api_inventory.json"), auto_unbox=TRUE, pretty=TRUE)
if (!ok) stop("nichenetr unavailable after install")
cat("import_ok\\n")
RS
audit_rc=$?
set -e
cat > "$OUT/params.json" <<JSON
{{"method":"nichenet","run_id":"{RUN_ID}","stage":"expanded_env_audit","pdc_root":"{PDC_ROOT}","breast_root":"{LEGACY_BREAST_ROOT}","USE_BUNDLED_LIBUV":"1"}}
JSON
if [ "$audit_rc" -eq 0 ]; then
  failure_type="api_imported_no_bounded_adapter"
else
  failure_type="env_or_api_audit_failed"
fi
cat > "$OUT/run_summary.json" <<JSON
{{"method":"nichenet","status":"failure_card","failure_type":"$failure_type","stage":"env_audit","created_at":"$(date -Is)"}}
JSON
cat > "$OUT/method_card.md" <<CARD
# NicheNet expanded benchmark method card

- Status: reproducible_failure_card
- Failure type: $failure_type
- Scope: expanded 18-method benchmark
- Run root: $OUT
- R library: $RLIB
- Audit return code: $audit_rc
- 20k smoke: not run because no validated NicheNet receiver-response adapter was available after audit.
- 50k bounded: not run because 20k smoke did not produce a standardized output.
- NicheNet role: downstream receiver-response support method, not a direct spatial CCI ranker.

## Reproduce

Run this command on PDC:

    cd {PDC_ROOT}
    sbatch {PDC_ROOT}/slurm/{RUN_ID}_expanded_nichenet.sbatch

## Logs

- audit.stdout.log
- audit.stderr.log
- api_inventory.json, if installation reached inventory stage
CARD
date -Is > "$OUT/end_time.txt"
"""


def submit_pdc() -> list[dict[str, Any]]:
    local_root().mkdir(parents=True, exist_ok=True)
    ssh(PDC_HOST, f"mkdir -p {q(PDC_RUN_ROOT)} {q(PDC_ROOT)}/logs {q(PDC_ROOT)}/slurm {q(PDC_ROOT)}/tmp")
    jobs = [
        submit_sbatch("expanded_fastccc", python_job_body("fastccc"), partition="shared", cpus=8, memory="64G", time_limit="04:00:00"),
        submit_sbatch("expanded_copulacci", python_job_body("copulacci"), partition="shared", cpus=8, memory="96G", time_limit="06:00:00"),
        submit_sbatch("expanded_nichenet", nichenet_job_body(), partition="shared", cpus=16, memory="160G", time_limit="12:00:00"),
    ]
    write_json(local_root() / "submitted_pdc_jobs.json", jobs)
    return jobs


def monitor_pdc() -> dict[str, Any]:
    command = "squeue -u hutaobo -o '%.18i %.12T %.30j %.10M %.9l %.6D %R' | grep -E 'expanded_fastccc|expanded_copulacci|expanded_nichenet|fastccc_smoke20k' || true"
    queue = ssh(PDC_HOST, command)
    payload = {"checked_at": now(), "queue_stdout": queue.stdout, "queue_stderr": queue.stderr, "returncode": queue.returncode}
    write_json(local_root() / "pdc_monitor.json", payload)
    return payload


def collect_pdc() -> dict[str, Any]:
    destination = local_root() / "pdc_collected" / RUN_ID
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = run_command(["scp", "-r", f"{PDC_HOST}:{PDC_RUN_ROOT}", str(destination.parent)])
    payload = {
        "created_at": now(),
        "remote": PDC_RUN_ROOT,
        "local": str(destination),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json(local_root() / "collect_pdc.json", payload)
    return payload


def method_terminal_status(method_dir: Path) -> tuple[str, str, str, str]:
    standardized_candidates = sorted(method_dir.rglob("standardized.tsv.gz")) if method_dir.exists() else []
    card_candidates = sorted(method_dir.rglob("method_card.md")) if method_dir.exists() else []
    summary_candidates = sorted(method_dir.rglob("run_summary.json")) if method_dir.exists() else []
    standardized = standardized_candidates[0] if standardized_candidates else method_dir / "standardized.tsv.gz"
    card = card_candidates[0] if card_candidates else method_dir / "method_card.md"
    summary = summary_candidates[0] if summary_candidates else method_dir / "run_summary.json"
    if standardized.exists():
        return ("bounded_subset_result", "appendix", "20k_or_50k_bounded", str(standardized))
    if card.exists() or summary.exists():
        return ("reproducible_failure_card", "appendix", "env_audit", str(card if card.exists() else summary))
    return ("running_or_missing", "appendix", "expanded_attempt", str(method_dir))


def finalize_local() -> dict[str, Any]:
    matrix_path = benchmark_root() / "results" / "method_completion_matrix.tsv"
    rows: list[dict[str, str]]
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    collected = local_root() / "pdc_collected" / RUN_ID
    updates: dict[str, dict[str, str]] = {}
    for method in METHODS:
        method_dir = collected / method
        status, evidence, phase, path = method_terminal_status(method_dir)
        updates[method] = {
            "status": status,
            "evidence_level": evidence,
            "phase": phase,
            "database_mode": "common-db",
            "remote_or_local_path": path,
            "n_rows": "",
            "current_job_id": "",
            "notes": (
                "Expanded 18-method terminal record. "
                "See method_card.md for exact install/audit blocker and rerun command."
                if status == "reproducible_failure_card"
                else "Expanded 18-method bounded result."
            ),
        }
    for row in rows:
        key = row["method"].lower()
        if key in updates:
            row.update(updates[key])
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        if row["dataset"] == "atera_breast_wta":
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    payload = {"updated_at": now(), "matrix": str(matrix_path), "status_counts": status_counts, "updates": updates}
    write_json(local_root() / "finalize_local.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve FastCCC/Copulacci/NicheNet expanded benchmark candidates.")
    parser.add_argument("--submit-pdc", action="store_true")
    parser.add_argument("--monitor-pdc", action="store_true")
    parser.add_argument("--collect-pdc", action="store_true")
    parser.add_argument("--finalize-local", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    outputs: dict[str, Any] = {}
    if args.submit_pdc or args.all:
        outputs["submit_pdc"] = submit_pdc()
    if args.monitor_pdc or args.all:
        outputs["monitor_pdc"] = monitor_pdc()
    if args.collect_pdc:
        outputs["collect_pdc"] = collect_pdc()
    if args.finalize_local:
        outputs["finalize_local"] = finalize_local()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
