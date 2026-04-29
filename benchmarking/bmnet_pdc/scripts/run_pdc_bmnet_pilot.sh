#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-bmnet_smoke_$(date +%Y%m%d_%H%M%S)}"
PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04}"
PDC_XENIUM_ROOT="${PDC_XENIUM_ROOT:-/cfs/klemming/scratch/h/hutaobo/topolink_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-bmnet}"
CONTOUR_GEOJSON="${CONTOUR_GEOJSON:-${PDC_XENIUM_ROOT}/xenium_explorer_annotations.s1_s5.generated.geojson}"
BACKEND="${BACKEND:-deterministic-smoke}"
CHECKPOINT="${CHECKPOINT:-}"
HF_MODEL="${HF_MODEL:-1aurent/vit_small_patch8_224.lunit_dino}"
MAX_CONTOURS="${MAX_CONTOURS:-50}"
PROGRAM_LIBRARY="${PROGRAM_LIBRARY:-breast_boundary_bmnet_v1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --pdc-root) PDC_ROOT="$2"; shift 2 ;;
    --dataset-root|--pdc-xenium-root) PDC_XENIUM_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX="$2"; shift 2 ;;
    --contour-geojson) CONTOUR_GEOJSON="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --hf-model) HF_MODEL="$2"; shift 2 ;;
    --max-contours) MAX_CONTOURS="$2"; shift 2 ;;
    --program-library) PROGRAM_LIBRARY="$2"; shift 2 ;;
    *) echo "[bmnet-pdc] unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "${PDC_XENIUM_ROOT}" ]]; then
  echo "[bmnet-pdc] missing dataset root: ${PDC_XENIUM_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CONDA_PREFIX}" ]]; then
  echo "[bmnet-pdc] missing conda prefix: ${CONDA_PREFIX}" >&2
  echo "[bmnet-pdc] run benchmarking/bmnet_pdc/scripts/bootstrap_pdc_env.sh first." >&2
  exit 2
fi
if [[ ! -s "${CONTOUR_GEOJSON}" ]]; then
  echo "[bmnet-pdc] missing contour GeoJSON: ${CONTOUR_GEOJSON}" >&2
  echo "[bmnet-pdc] run benchmarking/gmi_pdc/scripts/prepare_pdc_inputs.sh if the S1/S5 GeoJSON has not been generated." >&2
  exit 2
fi

LOG_DIR="${PDC_ROOT}/logs"
OUT_DIR="${PDC_ROOT}/runs/${RUN_ID}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/${RUN_ID}.log") 2>&1

echo "[bmnet-pdc] run_id=${RUN_ID} started $(date -Is)"
echo "[bmnet-pdc] dataset=${PDC_XENIUM_ROOT}"
echo "[bmnet-pdc] output=${OUT_DIR}"
echo "[bmnet-pdc] backend=${BACKEND}"
echo "[bmnet-pdc] checkpoint=${CHECKPOINT:-none}"
echo "[bmnet-pdc] hf_model=${HF_MODEL}"
echo "[bmnet-pdc] max_contours=${MAX_CONTOURS:-all}"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export TMPDIR="${PDC_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

cd "${REPO_DIR}"

ARGS=(
  benchmarking/bmnet_pdc/scripts/run_bmnet_morphology_increment.py
  --dataset-root "${PDC_XENIUM_ROOT}"
  --output-dir "${OUT_DIR}"
  --contour-geojson "${CONTOUR_GEOJSON}"
  --backend "${BACKEND}"
  --hf-model "${HF_MODEL}"
  --program-library "${PROGRAM_LIBRARY}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${MAX_CONTOURS}" ]]; then
  ARGS+=(--max-contours "${MAX_CONTOURS}")
fi

conda run --prefix "${CONDA_PREFIX}" python "${ARGS[@]}"

echo "[bmnet-pdc] run_id=${RUN_ID} completed $(date -Is)"
