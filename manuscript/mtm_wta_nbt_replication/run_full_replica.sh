#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

required_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

required_env BREAST_DATASET_ROOT
required_env BREAST_HISTOSEG_GEOJSON
required_env BREAST_HE_OME_TIF
required_env BREAST_WORK_DIR

BREAST_DATA_DIR="${BREAST_WORK_DIR}/data"
BREAST_RUN_DIR="${BREAST_WORK_DIR}/runs/direct_lazyslide_plip_full_mtm_wta"
BREAST_PYRAMID="${BREAST_DATA_DIR}/WTA_Preview_FFPE_Breast_Cancer_he_image.tiffslide_pyramid.tif"
mkdir -p "${BREAST_DATA_DIR}" "${BREAST_RUN_DIR}"

"${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/prepare_tiffslide_pyramid.py" \
  --input "${BREAST_HE_OME_TIF}" \
  --output "${BREAST_PYRAMID}" \
  --tile-px 512 \
  --jpeg-quality 90 \
  --verify

"${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/run_histoseg_lazyslide_workflow.py" \
  --dataset-root "${BREAST_DATASET_ROOT}" \
  --histoseg-geojson "${BREAST_HISTOSEG_GEOJSON}" \
  --contour-id-key name \
  --he-source-path "${BREAST_PYRAMID}" \
  --wsi-reader tiffslide \
  --output-dir "${BREAST_RUN_DIR}" \
  --model plip \
  --text-model plip \
  --batch-size "${BATCH_SIZE:-64}" \
  --table-format parquet

if [[ -n "${CERVICAL_DATASET_ROOT:-}" && -n "${CERVICAL_HISTOSEG_GEOJSON:-}" && -n "${CERVICAL_HE_OME_TIF:-}" && -n "${CERVICAL_WORK_DIR:-}" ]]; then
  CERVICAL_DATA_DIR="${CERVICAL_WORK_DIR}/data"
  CERVICAL_RUN_DIR="${CERVICAL_WORK_DIR}/runs/direct_lazyslide_plip_full_mtm_wta"
  CERVICAL_PYRAMID="${CERVICAL_DATA_DIR}/WTA_Preview_FFPE_Cervical_Cancer_he_image.tiffslide_pyramid.tif"
  mkdir -p "${CERVICAL_DATA_DIR}" "${CERVICAL_RUN_DIR}"
  "${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/prepare_tiffslide_pyramid.py" \
    --input "${CERVICAL_HE_OME_TIF}" \
    --output "${CERVICAL_PYRAMID}" \
    --tile-px 512 \
    --jpeg-quality 90 \
    --verify
  "${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/run_histoseg_lazyslide_workflow.py" \
    --dataset-root "${CERVICAL_DATASET_ROOT}" \
    --histoseg-geojson "${CERVICAL_HISTOSEG_GEOJSON}" \
    --contour-id-key "${CERVICAL_CONTOUR_ID_KEY:-name}" \
    --he-source-path "${CERVICAL_PYRAMID}" \
    --wsi-reader tiffslide \
    --output-dir "${CERVICAL_RUN_DIR}" \
    --model plip \
    --text-model plip \
    --batch-size "${BATCH_SIZE:-64}" \
    --table-format parquet
else
  echo "Cervical raw inputs not fully specified; skipping cervical full GPU rerun."
fi

"${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/compose_nbt_brief_main_figure.py"
"${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/enhance_mtm_nature_assets.py"
"${PYTHON}" "${REPO_ROOT}/benchmarking/lazyslide_a100/scripts/prepare_nbt_initial_submission_upload.py"

echo "Replication run completed. Check manuscript/mtm_wta_nbt_replication/expected_outputs.md for expected files."
