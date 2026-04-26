#!/usr/bin/env bash
set -euo pipefail

STAGE_ID=""
PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04}"
DATASET_ROOT="${PDC_XENIUM_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-gmi}"
CONTOUR_GEOJSON=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-id) STAGE_ID="$2"; shift 2 ;;
    --pdc-root) PDC_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX="$2"; shift 2 ;;
    --contour-geojson) CONTOUR_GEOJSON="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[gmi-pdc] unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${STAGE_ID}" ]]; then
  echo "[gmi-pdc] --stage-id is required" >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[gmi-pdc] missing dataset root: ${DATASET_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CONDA_PREFIX}" ]]; then
  echo "[gmi-pdc] missing conda prefix: ${CONDA_PREFIX}" >&2
  exit 2
fi
if [[ -z "${CONTOUR_GEOJSON}" ]]; then
  CONTOUR_GEOJSON="${DATASET_ROOT}/xenium_explorer_annotations.s1_s5.generated.geojson"
fi
if [[ ! -s "${CONTOUR_GEOJSON}" ]]; then
  echo "[gmi-pdc] missing contour GeoJSON: ${CONTOUR_GEOJSON}" >&2
  echo "[gmi-pdc] run benchmarking/gmi_pdc/scripts/prepare_pdc_inputs.sh before submitting stages." >&2
  exit 2
fi

LOG_DIR="${PDC_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${PDC_ROOT}/runs" "${PDC_ROOT}/results" "${PDC_ROOT}/reports" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/${STAGE_ID}.stage.log") 2>&1

echo "[gmi-pdc] stage=${STAGE_ID} started $(date -Is)"
echo "[gmi-pdc] dataset=${DATASET_ROOT}"
echo "[gmi-pdc] output=${PDC_ROOT}/runs/${STAGE_ID}"
echo "[gmi-pdc] contour_geojson=${CONTOUR_GEOJSON}"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export TMPDIR="${PDC_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

cd "${REPO_DIR}"

conda run --prefix "${CONDA_PREFIX}" pyxenium gmi run \
  --dataset-root "${DATASET_ROOT}" \
  --contour-geojson "${CONTOUR_GEOJSON}" \
  --output-dir "${PDC_ROOT}/runs/${STAGE_ID}" \
  "${EXTRA_ARGS[@]}"

echo "[gmi-pdc] stage=${STAGE_ID} completed $(date -Is)"
