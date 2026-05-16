"""A100 retry supervisor for expanded CCI benchmark methods.

This script retries FastCCC, Copulacci, and NicheNet on the A100 host without
overwriting PDC terminal artifacts.
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


RUN_ID = "expanded_methods_a100_20260514"
METHODS = ("fastccc", "copulacci", "nichenet")
A100_HOST = "taobo.hu@sscb-a100.scilifelab.se"
A100_ROOT = "/data/taobo.hu/pyxenium_cci_benchmark_2026-04"
A100_RUN_ROOT = f"{A100_ROOT}/runs/{RUN_ID}"
A100_INPUT_ROOT = "/data/taobo.hu/pyxenium_lr_benchmark_2026-04"
CONDA = "/home/taobo.hu/miniconda3/bin/conda"


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


def run_command(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def ssh(command: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", A100_HOST, command],
        timeout=timeout,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def upload_text(remote_path: str, text: str) -> subprocess.CompletedProcess[str]:
    tmp = local_root() / "staging" / (Path(remote_path).name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8", newline="\n")
    ssh(f"mkdir -p {q(str(Path(remote_path).parent).replace(chr(92), '/'))}")
    return run_command(["scp", str(tmp), f"{A100_HOST}:{remote_path}"])


def submit_background(job_id: str, script: str) -> dict[str, Any]:
    remote_script = f"{A100_ROOT}/scripts/{RUN_ID}_{job_id}.sh"
    upload = upload_text(remote_script, script)
    if upload.returncode != 0:
        return {"job_id": job_id, "status": "script_upload_failed", "stderr": upload.stderr, "returncode": upload.returncode}
    cmd = (
        f"mkdir -p {q(A100_ROOT)}/logs {q(A100_RUN_ROOT)}; "
        f"chmod +x {q(remote_script)}; "
        f"nohup bash {q(remote_script)} > {q(A100_ROOT + '/logs/' + RUN_ID + '_' + job_id + '.stdout.log')} "
        f"2> {q(A100_ROOT + '/logs/' + RUN_ID + '_' + job_id + '.stderr.log')} "
        f"< /dev/null & echo $!"
    )
    submitted = ssh(cmd)
    payload = {
        "job_id": job_id,
        "status": "submitted" if submitted.returncode == 0 else "submit_failed",
        "pid": submitted.stdout.strip(),
        "returncode": submitted.returncode,
        "remote_script": remote_script,
        "stderr": submitted.stderr,
    }
    return payload


def bash_header(job_id: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
export ROOT={q(A100_ROOT)}
export RUN_ROOT={q(A100_RUN_ROOT)}
export INPUT_ROOT={q(A100_INPUT_ROOT)}
export TMPDIR="$ROOT/tmp/{RUN_ID}_{job_id}"
mkdir -p "$TMPDIR" "$RUN_ROOT" "$ROOT/logs" "$ROOT/envs" "$ROOT/scripts"
"""


def fastccc_script() -> str:
    return bash_header("fastccc") + r"""
METHOD=fastccc
OUT="$RUN_ROOT/$METHOD"
ENV="$ROOT/envs/expanded_methods_a100_20260514_fastccc"
mkdir -p "$OUT"
date -Is > "$OUT/start_time.txt"
export CONDA_PKGS_DIRS="$ROOT/tmp/conda_pkgs_$METHOD"
mkdir -p "$CONDA_PKGS_DIRS"
if [ ! -x "$ENV/bin/python" ]; then
  rm -rf "$ENV"
  /home/taobo.hu/miniconda3/bin/conda create -y -p "$ENV" python=3.11 pip > "$OUT/conda_create.stdout.log" 2> "$OUT/conda_create.stderr.log"
fi
PYTHON="$ENV/bin/python"
"$PYTHON" --version > "$OUT/python_version.txt" 2>&1
"$PYTHON" -m pip install --upgrade pip wheel setuptools > "$OUT/pip_upgrade.stdout.log" 2> "$OUT/pip_upgrade.stderr.log"
"$PYTHON" -m pip install fastccc==0.1.3 anndata pandas numpy scipy statsmodels h5py zarr > "$OUT/install.stdout.log" 2> "$OUT/install.stderr.log"
"$PYTHON" - <<'PY' > "$OUT/run.stdout.log" 2> "$OUT/run.stderr.log"
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from fastccc.core import statistical_analysis_method

start = time.time()
root = Path("/data/taobo.hu/pyxenium_cci_benchmark_2026-04")
input_root = Path("/data/taobo.hu/pyxenium_lr_benchmark_2026-04")
out = root / "runs" / "expanded_methods_a100_20260514" / "fastccc"
smoke = out / "smoke20k"
raw_dir = smoke / "raw"
dbdir = smoke / "cpdb_common_resource"
for d in [smoke, raw_dir, dbdir]:
    d.mkdir(parents=True, exist_ok=True)

lr = pd.read_csv(input_root / "data/lr_db_common.tsv", sep="\t")[["ligand", "receptor"]].dropna().drop_duplicates().reset_index(drop=True)
genes = pd.Index(pd.unique(pd.concat([lr["ligand"], lr["receptor"]], ignore_index=True))).astype(str)
protein_ids = np.arange(1, len(genes) + 1)
gene_table = pd.DataFrame({"protein_id": protein_ids, "hgnc_symbol": genes})
protein_table = pd.DataFrame({"id_protein": protein_ids, "protein_multidata_id": protein_ids})
gene_to_multi = dict(zip(genes, protein_ids))
interaction_ids = [f"CPI-TOPOLINK-{i:05d}" for i in range(len(lr))]
interaction_table = pd.DataFrame({
    "id_cp_interaction": interaction_ids,
    "multidata_1_id": [gene_to_multi[x] for x in lr["ligand"]],
    "multidata_2_id": [gene_to_multi[x] for x in lr["receptor"]],
})
gene_table.to_csv(dbdir / "gene_table.csv", index=False)
protein_table.to_csv(dbdir / "protein_table.csv", index=False)
interaction_table.to_csv(dbdir / "interaction_table.csv", index=False)
pd.DataFrame({"complex_multidata_id": pd.Series(dtype="int64")}).to_csv(dbdir / "complex_table.csv", index=False)
pd.DataFrame({"complex_multidata_id": pd.Series(dtype="int64"), "protein_multidata_id": pd.Series(dtype="int64")}).to_csv(dbdir / "complex_composition_table.csv", index=False)
id_to_pair = dict(zip(interaction_ids, zip(lr["ligand"], lr["receptor"])))

adata = ad.read_h5ad(input_root / "data/smoke/adata_smoke.h5ad")
if "cell_type" not in adata.obs:
    raise RuntimeError("smoke h5ad lacks cell_type")
if sp.issparse(adata.X):
    adata.X = adata.X.copy()
    adata.X.data = np.log1p(adata.X.data)
else:
    adata.X = np.log1p(adata.X)

strength, pvals, perc = statistical_analysis_method(
    str(dbdir), "", adata, convert_type="hgnc_symbol", meta_key="cell_type", style="cpdb", save_path=str(raw_dir)
)

# FastCCC returns rows as sender|receiver and columns as interaction ids.
strength = strength.copy()
if "Unnamed: 0" in strength.columns:
    strength = strength.set_index("Unnamed: 0")
pvals = pvals.copy()
if "Unnamed: 0" in pvals.columns:
    pvals = pvals.set_index("Unnamed: 0")
records = []
for sender_receiver, row in strength.iterrows():
    sender, receiver = str(sender_receiver).split("|", 1) if "|" in str(sender_receiver) else (str(sender_receiver), str(sender_receiver))
    p_row = pvals.loc[sender_receiver] if sender_receiver in pvals.index else None
    for interaction_id, score in row.items():
        if interaction_id not in id_to_pair or pd.isna(score):
            continue
        ligand, receptor = id_to_pair[interaction_id]
        pval = float(p_row[interaction_id]) if p_row is not None and interaction_id in p_row.index and pd.notna(p_row[interaction_id]) else math.nan
        records.append((ligand, receptor, sender, receiver, float(score), pval))
res = pd.DataFrame(records, columns=["ligand", "receptor", "sender", "receiver", "score_raw", "fdr_or_pvalue"])
if res.empty:
    raise RuntimeError("FastCCC A100 smoke produced no rows")
res["method"] = "FastCCC"
res["database_mode"] = "common-db"
res["resolution"] = "celltype_pair"
res["spatial_support_type"] = "nonspatial_analytic_expression_baseline"
res["bounded_mode"] = "20k_smoke_a100"
res["artifact_path"] = str(smoke / "standardized.tsv.gz")
res = res.sort_values(["score_raw", "fdr_or_pvalue"], ascending=[False, True], na_position="last").reset_index(drop=True)
res["rank_within_method"] = np.arange(1, len(res) + 1)
res["score_std"] = 1.0 - (res["rank_within_method"] - 1) / max(len(res) - 1, 1)
cols = ["method", "database_mode", "ligand", "receptor", "sender", "receiver", "score_raw", "score_std", "rank_within_method", "fdr_or_pvalue", "resolution", "spatial_support_type", "bounded_mode", "artifact_path"]
res[cols].to_csv(smoke / "standardized.tsv.gz", sep="\t", index=False, compression="gzip")
summary = {"method": "FastCCC", "status": "success", "phase": "20k_smoke_a100", "n_rows": int(len(res)), "runtime_seconds": time.time() - start, "output": str(smoke / "standardized.tsv.gz")}
(smoke / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(smoke / "params.json").write_text(json.dumps({"method": "FastCCC", "phase": "20k_smoke_a100", "input": str(input_root / "data/smoke/adata_smoke.h5ad")}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
date -Is > "$OUT/end_time.txt"
"""


def copulacci_script() -> str:
    return bash_header("copulacci") + r"""
METHOD=copulacci
OUT="$RUN_ROOT/$METHOD"
ENV="$ROOT/envs/expanded_methods_a100_20260514_copulacci"
mkdir -p "$OUT"
date -Is > "$OUT/start_time.txt"
export CONDA_PKGS_DIRS="$ROOT/tmp/conda_pkgs_$METHOD"
mkdir -p "$CONDA_PKGS_DIRS"
if [ ! -x "$ENV/bin/python" ]; then
  rm -rf "$ENV"
  /home/taobo.hu/miniconda3/bin/conda create -y -p "$ENV" python=3.11 pip > "$OUT/conda_create.stdout.log" 2> "$OUT/conda_create.stderr.log"
fi
PYTHON="$ENV/bin/python"
"$PYTHON" --version > "$OUT/python_version.txt" 2>&1
"$PYTHON" -m pip install --upgrade pip wheel setuptools > "$OUT/pip_upgrade.stdout.log" 2> "$OUT/pip_upgrade.stderr.log"
"$PYTHON" -m pip install numpy pandas scipy anndata scanpy squidpy statsmodels networkx > "$OUT/install_deps.stdout.log" 2> "$OUT/install_deps.stderr.log"
mkdir -p "$ROOT/external_src"
rm -rf "$ROOT/external_src/copulacci"
git clone https://github.com/raphael-group/copulacci.git "$ROOT/external_src/copulacci" > "$OUT/install_git.stdout.log" 2> "$OUT/install_git.stderr.log"
"$PYTHON" -m pip install -e "$ROOT/external_src/copulacci" > "$OUT/install_source.stdout.log" 2> "$OUT/install_source.stderr.log"
"$PYTHON" - <<'PY' > "$OUT/audit.stdout.log" 2> "$OUT/audit.stderr.log"
import importlib, inspect, json
from pathlib import Path
mods = ["copulacci", "copulacci.spatial", "copulacci.model", "copulacci.model2", "copulacci.cci"]
inventory = {}
for name in mods:
    module = importlib.import_module(name)
    inventory[name] = {}
    for key in dir(module):
        if key.startswith("_"):
            continue
        obj = getattr(module, key)
        if callable(obj):
            try:
                inventory[name][key] = str(inspect.signature(obj))
            except Exception:
                inventory[name][key] = "<signature unavailable>"
Path("/data/taobo.hu/pyxenium_cci_benchmark_2026-04/runs/expanded_methods_a100_20260514/copulacci/api_inventory.json").write_text(json.dumps(inventory, indent=2) + "\\n")
print("import_ok")
PY
cat > "$OUT/run_summary.json" <<JSON
{"method":"Copulacci","status":"failure_card","failure_type":"api_imported_no_safe_bounded_adapter","stage":"a100_api_inventory","created_at":"$(date -Is)"}
JSON
cat > "$OUT/method_card.md" <<CARD
# Copulacci A100 retry method card

- Status: reproducible_failure_card
- Failure type: api_imported_no_safe_bounded_adapter
- Scope: expanded 18-method A100 retry
- Run root: $OUT
- Environment: $ENV
- Install source: https://github.com/raphael-group/copulacci
- 20k smoke: not run because native workflow mapping requires method-specific data-list construction and could not be safely mapped to the common-resource benchmark without fabricating a baseline.
- 50k bounded: not run because 20k smoke did not produce standardized output.

## Logs

- api_inventory.json
- audit.stdout.log
- audit.stderr.log
- install*.log
CARD
date -Is > "$OUT/end_time.txt"
"""


def nichenet_script() -> str:
    return bash_header("nichenet") + r"""
METHOD=nichenet
OUT="$RUN_ROOT/$METHOD"
ENV="$ROOT/envs/expanded_methods_a100_20260514_nichenet"
mkdir -p "$OUT"
date -Is > "$OUT/start_time.txt"
export CONDA_PKGS_DIRS="$ROOT/tmp/conda_pkgs_$METHOD"
mkdir -p "$CONDA_PKGS_DIRS"
if [ ! -x "$ENV/bin/Rscript" ]; then
  rm -rf "$ENV"
  /home/taobo.hu/miniconda3/bin/conda create -y -p "$ENV" -c conda-forge r-base=4.3 r-remotes r-jsonlite r-dplyr r-tibble r-magrittr libuv cmake make gcc_linux-64 gxx_linux-64 > "$OUT/conda_create.stdout.log" 2> "$OUT/conda_create.stderr.log"
fi
export R_LIBS_USER="$OUT/Rlib"
export USE_BUNDLED_LIBUV=1
mkdir -p "$R_LIBS_USER"
RSCRIPT="$ENV/bin/Rscript"
set +e
"$RSCRIPT" - <<'RS' > "$OUT/audit.stdout.log" 2> "$OUT/audit.stderr.log"
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
Sys.setenv(USE_BUNDLED_LIBUV="1")
cat("R version:", R.version.string, "\\n")
if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")
if (!requireNamespace("nichenetr", quietly=TRUE)) remotes::install_github("saeyslab/nichenetr", upgrade="never")
ok <- requireNamespace("nichenetr", quietly=TRUE)
inventory <- list(R=R.version.string, nichenetr_available=ok, exports=if (ok) getNamespaceExports("nichenetr") else character())
jsonlite::write_json(inventory, file.path("/data/taobo.hu/pyxenium_cci_benchmark_2026-04/runs/expanded_methods_a100_20260514", "nichenet", "api_inventory.json"), auto_unbox=TRUE, pretty=TRUE)
if (!ok) stop("nichenetr unavailable after install")
cat("import_ok\\n")
RS
rc=$?
set -e
if [ "$rc" -eq 0 ]; then failure_type="api_imported_no_receiver_response_adapter"; else failure_type="env_or_api_audit_failed"; fi
cat > "$OUT/run_summary.json" <<JSON
{"method":"NicheNet","status":"failure_card","failure_type":"$failure_type","stage":"a100_env_audit","created_at":"$(date -Is)"}
JSON
cat > "$OUT/method_card.md" <<CARD
# NicheNet A100 retry method card

- Status: reproducible_failure_card
- Failure type: $failure_type
- Scope: expanded 18-method A100 retry
- Run root: $OUT
- Environment: $ENV
- 20k smoke: not run unless nichenetr import and a validated receiver-response adapter are available.
- 50k bounded: not run because 20k smoke did not produce standardized output.
- NicheNet role: downstream receiver-response support, not a direct spatial CCI ranker.

## Logs

- conda_create.stdout.log
- conda_create.stderr.log
- audit.stdout.log
- audit.stderr.log
- api_inventory.json, if installation reached inventory stage
CARD
date -Is > "$OUT/end_time.txt"
exit 0
"""


def submit_a100(methods: tuple[str, ...] = METHODS) -> list[dict[str, Any]]:
    local_root().mkdir(parents=True, exist_ok=True)
    ssh(f"mkdir -p {q(A100_RUN_ROOT)} {q(A100_ROOT)}/logs {q(A100_ROOT)}/scripts {q(A100_ROOT)}/tmp")
    script_by_method = {
        "fastccc": fastccc_script,
        "copulacci": copulacci_script,
        "nichenet": nichenet_script,
    }
    unknown = sorted(set(methods) - set(script_by_method))
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    jobs = [submit_background(method, script_by_method[method]()) for method in methods]
    write_json(local_root() / "submitted_a100_jobs.json", jobs)
    return jobs


def monitor_a100() -> dict[str, Any]:
    cmd = (
        "ps -eo pid,etime,pcpu,pmem,cmd | "
        "grep -E 'expanded_methods_a100_20260514|fastccc|copulacci|nichenet' | grep -v grep || true"
    )
    proc = ssh(cmd)
    payload = {"checked_at": datetime.now(timezone.utc).isoformat(), "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    write_json(local_root() / "a100_monitor.json", payload)
    return payload


def collect_a100() -> dict[str, Any]:
    destination = local_root() / "a100_collected" / RUN_ID
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_remote = f"{A100_ROOT}/tmp/{RUN_ID}_collect.tar.gz"
    archive_local = local_root() / "a100_collected" / f"{RUN_ID}_collect.tar.gz"
    pack = ssh(
        f"rm -f {q(archive_remote)}; "
        f"tar -C {q(A100_RUN_ROOT)} --exclude='*/Rlib' -czf {q(archive_remote)} ."
    )
    completed = run_command(["scp", f"{A100_HOST}:{archive_remote}", str(archive_local)]) if pack.returncode == 0 else pack
    if completed.returncode == 0:
        destination.mkdir(parents=True, exist_ok=True)
        import tarfile

        with tarfile.open(archive_local, "r:gz") as tar:
            tar.extractall(destination)
    logs_dir = local_root() / "a100_collected" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs = run_command(["scp", f"{A100_HOST}:{A100_ROOT}/logs/{RUN_ID}_*.log", str(logs_dir)])
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "remote": A100_RUN_ROOT,
        "local": str(destination),
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "logs_returncode": logs.returncode,
        "logs_stderr": logs.stderr,
        "logs_local": str(logs_dir),
    }
    write_json(local_root() / "collect_a100.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="A100 retry for FastCCC/Copulacci/NicheNet expanded benchmark methods.")
    parser.add_argument("--submit-a100", action="store_true")
    parser.add_argument("--monitor-a100", action="store_true")
    parser.add_argument("--collect-a100", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated methods for --submit-a100.")
    args = parser.parse_args()
    payload: dict[str, Any] = {}
    if args.submit_a100:
        methods = tuple(part.strip().lower() for part in args.methods.split(",") if part.strip())
        payload["submit_a100"] = submit_a100(methods)
    if args.monitor_a100:
        payload["monitor_a100"] = monitor_a100()
    if args.collect_a100:
        payload["collect_a100"] = collect_a100()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
