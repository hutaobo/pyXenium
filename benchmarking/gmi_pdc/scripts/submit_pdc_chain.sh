#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04}"
PDC_XENIUM_ROOT="${PDC_XENIUM_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-gmi}"
HISTOSEG_ROOT="${HISTOSEG_ROOT:-${PDC_ROOT}/external/HistoSeg}"
ACCOUNT="${PDC_PROJECT:-}"
PLAN_JSON="${PDC_ROOT}/logs/pdc_gmi_plan.json"
DRY_RUN=0
SKIP_PREPARE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdc-root) PDC_ROOT="$2"; shift 2 ;;
    --dataset-root|--pdc-xenium-root) PDC_XENIUM_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX="$2"; shift 2 ;;
    --histoseg-root) HISTOSEG_ROOT="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    *) echo "[gmi-pdc] unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${PDC_ROOT}/logs" "${PDC_ROOT}/runs" "${PDC_ROOT}/results" "${PDC_ROOT}/reports" "${PDC_ROOT}/tmp"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

if [[ -z "${ACCOUNT}" ]]; then
  mapfile -t ACCOUNTS < <(projinfo 2>/dev/null | awk '/^[a-zA-Z0-9_-]+[[:space:]]/ {print $1}' | sort -u)
  if [[ ${#ACCOUNTS[@]} -eq 1 ]]; then
    ACCOUNT="${ACCOUNTS[0]}"
  else
    echo "[gmi-pdc] unable to infer a single Slurm account. Set PDC_PROJECT or pass --account." >&2
    printf '[gmi-pdc] projinfo candidates: %s\n' "${ACCOUNTS[*]:-none}" >&2
    exit 2
  fi
fi

cd "${REPO_DIR}"

CONTOUR_GEOJSON="${PDC_XENIUM_ROOT}/xenium_explorer_annotations.s1_s5.generated.geojson"
INITIAL_JOB_ID=""
if [[ ! -s "${CONTOUR_GEOJSON}" && "${SKIP_PREPARE}" -eq 1 ]]; then
  echo "[gmi-pdc] missing contour GeoJSON: ${CONTOUR_GEOJSON}" >&2
  echo "[gmi-pdc] source cache must contain the S1/S5 GeoJSON or enough GraphClust/cell inputs to generate it." >&2
  exit 3
fi
if [[ ! -s "${CONTOUR_GEOJSON}" ]]; then
  echo "[gmi-pdc] missing contour GeoJSON; submitting a Slurm preflight generation job."
  PREPARE_CMD="${REPO_DIR}/benchmarking/gmi_pdc/scripts/prepare_pdc_inputs.sh --pdc-root ${PDC_ROOT} --dataset-root ${PDC_XENIUM_ROOT} --repo-dir ${REPO_DIR} --conda-prefix ${CONDA_PREFIX} --histoseg-root ${HISTOSEG_ROOT}"
  PREPARE_SBATCH=(
    sbatch
    --job-name=pyxgmi_prepare_s1s5
    --partition=shared
    --nodes=1
    --ntasks=1
    --cpus-per-task=16
    --mem=64GB
    --time=04:00:00
    --output="${PDC_ROOT}/logs/prepare_s1_s5.%j.log"
    --account="${ACCOUNT}"
    --wrap
    "${PREPARE_CMD}"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${PREPARE_SBATCH[@]}"
    printf '\n'
    INITIAL_JOB_ID="DRYRUN_prepare_s1_s5"
  else
    PREPARE_RESULT="$("${PREPARE_SBATCH[@]}")"
    echo "${PREPARE_RESULT}"
    INITIAL_JOB_ID="$(awk '{print $NF}' <<<"${PREPARE_RESULT}")"
  fi
fi

conda run --prefix "${CONDA_PREFIX}" pyxenium gmi pdc-plan \
  --pdc-xenium-root "${PDC_XENIUM_ROOT}" \
  --pdc-root "${PDC_ROOT}" \
  --repo-dir "${REPO_DIR}" \
  --conda-prefix "${CONDA_PREFIX}" \
  --account "${ACCOUNT}" \
  --output-json "${PLAN_JSON}" >/dev/null

python3 - "${PLAN_JSON}" "${DRY_RUN}" "${INITIAL_JOB_ID}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
dry_run = sys.argv[2] == "1"
previous_job_id = sys.argv[3] or None
submitted = []

def with_dependency(cmd: list[str], job_id: str | None, placeholder: str | None) -> list[str]:
    if not job_id:
        return cmd
    if placeholder:
        return [part.replace("${" + placeholder + "_JOB_ID}", job_id) for part in cmd]
    if any(part.startswith("--dependency=") for part in cmd):
        return cmd
    try:
        wrap_index = cmd.index("--wrap")
    except ValueError:
        return [*cmd, f"--dependency=afterok:{job_id}"]
    return [*cmd[:wrap_index], f"--dependency=afterok:{job_id}", *cmd[wrap_index:]]

for stage in plan["stages"]:
    cmd = list(stage["sbatch"])
    cmd = with_dependency(cmd, previous_job_id, stage["depends_on"])
    printable = " ".join(cmd)
    if dry_run:
        print(printable)
        previous_job_id = f"DRYRUN_{stage['stage_id']}"
        submitted.append({"stage_id": stage["stage_id"], "job_id": previous_job_id, "command": printable})
        continue
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(result.stdout.strip())
    previous_job_id = result.stdout.strip().split()[-1]
    submitted.append({"stage_id": stage["stage_id"], "job_id": previous_job_id, "command": printable})
Path(plan["logs_dir"], "pdc_gmi_submitted_jobs.json").write_text(json.dumps(submitted, indent=2) + "\n")
PY
