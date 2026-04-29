#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04}"
PDC_XENIUM_ROOT="${PDC_XENIUM_ROOT:-/cfs/klemming/scratch/h/hutaobo/topolink_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-bmnet}"
CONTOUR_GEOJSON="${CONTOUR_GEOJSON:-${PDC_XENIUM_ROOT}/xenium_explorer_annotations.s1_s5.generated.geojson}"
BACKEND="${BACKEND:-deterministic-smoke}"
CHECKPOINT="${CHECKPOINT:-}"
HF_MODEL="${HF_MODEL:-1aurent/vit_small_patch8_224.lunit_dino}"
ACCOUNT="${PDC_PROJECT:-}"
SMOKE_MAX_CONTOURS="${SMOKE_MAX_CONTOURS:-50}"
INCLUDE_FULL=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdc-root) PDC_ROOT="$2"; shift 2 ;;
    --dataset-root|--pdc-xenium-root) PDC_XENIUM_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX="$2"; shift 2 ;;
    --contour-geojson) CONTOUR_GEOJSON="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --hf-model) HF_MODEL="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --smoke-max-contours) SMOKE_MAX_CONTOURS="$2"; shift 2 ;;
    --include-full) INCLUDE_FULL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[bmnet-pdc] unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${PDC_ROOT}/logs" "${PDC_ROOT}/runs" "${PDC_ROOT}/tmp"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

if [[ -z "${ACCOUNT}" ]]; then
  mapfile -t ACCOUNTS < <(projinfo 2>/dev/null | awk '/^[a-zA-Z0-9_-]+[[:space:]]/ {print $1}' | sort -u)
  if [[ ${#ACCOUNTS[@]} -eq 1 ]]; then
    ACCOUNT="${ACCOUNTS[0]}"
  else
    echo "[bmnet-pdc] unable to infer a single Slurm account. Set PDC_PROJECT or pass --account." >&2
    printf '[bmnet-pdc] projinfo candidates: %s\n' "${ACCOUNTS[*]:-none}" >&2
    exit 2
  fi
fi

submit_job() {
  local run_id="$1"
  local max_contours="$2"
  local partition="$3"
  local mem="$4"
  local time_limit="$5"
  local dependency="${6:-}"
  local cmd="bash ${REPO_DIR}/benchmarking/bmnet_pdc/scripts/run_pdc_bmnet_pilot.sh --run-id ${run_id} --pdc-root ${PDC_ROOT} --dataset-root ${PDC_XENIUM_ROOT} --repo-dir ${REPO_DIR} --conda-prefix ${CONDA_PREFIX} --contour-geojson ${CONTOUR_GEOJSON} --backend ${BACKEND} --hf-model ${HF_MODEL} --program-library breast_boundary_bmnet_v1"
  if [[ -n "${CHECKPOINT}" ]]; then
    cmd="${cmd} --checkpoint ${CHECKPOINT}"
  fi
  if [[ -n "${max_contours}" ]]; then
    cmd="${cmd} --max-contours ${max_contours}"
  else
    cmd="${cmd} --max-contours ''"
  fi
  local sbatch_cmd=(
    sbatch
    "--job-name=pyxbmnet_${run_id:0:20}"
    "--partition=${partition}"
    "--nodes=1"
    "--ntasks=1"
    "--cpus-per-task=16"
    "--mem=${mem}"
    "--time=${time_limit}"
    "--output=${PDC_ROOT}/logs/${run_id}.%j.log"
    "--account=${ACCOUNT}"
  )
  if [[ -n "${dependency}" ]]; then
    sbatch_cmd+=("--dependency=afterok:${dependency}")
  fi
  sbatch_cmd+=(--wrap "${cmd}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${sbatch_cmd[@]}" >&2
    printf '\n' >&2
    echo "DRYRUN_${run_id}"
  else
    local result
    result="$("${sbatch_cmd[@]}")"
    echo "${result}" >&2
    awk '{print $NF}' <<<"${result}"
  fi
}

SMOKE_ID="$(submit_job "bmnet_smoke_${BACKEND//-/_}" "${SMOKE_MAX_CONTOURS}" "shared" "80GB" "04:00:00")"
if [[ "${INCLUDE_FULL}" -eq 1 ]]; then
  submit_job "bmnet_full_${BACKEND//-/_}" "" "main" "220GB" "24:00:00" "${SMOKE_ID}" >/dev/null
fi
