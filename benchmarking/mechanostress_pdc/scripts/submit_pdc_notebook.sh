#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04}"
PDC_DATASET_ROOT="${PDC_DATASET_ROOT:-/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-mechanostress}"
ACCOUNT="${PDC_PROJECT:-}"
DEFAULT_ACCOUNT="naiss2025-22-606"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdc-root) PDC_ROOT="$2"; shift 2 ;;
    --dataset-root|--pdc-dataset-root) PDC_DATASET_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[mechanostress-pdc] unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${PDC_ROOT}/logs" "${PDC_ROOT}/slurm"

module load PDC/24.11

if [[ -z "${ACCOUNT}" ]]; then
  if projinfo 2>/dev/null | grep -q "${DEFAULT_ACCOUNT}"; then
    ACCOUNT="${DEFAULT_ACCOUNT}"
  else
    mapfile -t ACCOUNTS < <(projinfo 2>/dev/null | awk '/^[a-zA-Z0-9_-]+[[:space:]]/ {print $1}' | sort -u)
    if [[ ${#ACCOUNTS[@]} -ge 1 ]]; then
      ACCOUNT="${ACCOUNTS[0]}"
    else
      echo "[mechanostress-pdc] unable to infer a Slurm account. Set PDC_PROJECT or pass --account." >&2
      exit 2
    fi
  fi
fi

CMD=(
  sbatch
  --job-name=pyx_mechanostress_atera
  --partition=main
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --mem=128GB
  --time=04:00:00
  --output="${PDC_ROOT}/logs/mechanostress_atera.%j.log"
  --account="${ACCOUNT}"
  --export="ALL,PDC_ROOT=${PDC_ROOT},PDC_DATASET_ROOT=${PDC_DATASET_ROOT},REPO_DIR=${REPO_DIR},CONDA_PREFIX=${CONDA_PREFIX}"
  --wrap
  "bash ${REPO_DIR}/benchmarking/mechanostress_pdc/scripts/run_pdc_notebook.sh"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
fi
