#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04}"
PDC_DATASET_ROOT="${PDC_DATASET_ROOT:-/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-mechanostress}"
SAMPLE_ID="${SAMPLE_ID:-WTA_Preview_FFPE_Breast_Cancer_outs}"
STAGING_ROOT="${PDC_ROOT}/staging"
SAMPLE_DIR="${STAGING_ROOT}/${SAMPLE_ID}"
LOG_DIR="${PDC_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${SAMPLE_DIR}" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/prepare_pdc_inputs.log") 2>&1

echo "[mechanostress-pdc] input preparation started $(date -Is)"
echo "[mechanostress-pdc] dataset=${PDC_DATASET_ROOT}"
echo "[mechanostress-pdc] sample_dir=${SAMPLE_DIR}"

REQUIRED=(
  "cell_feature_matrix.h5"
  "cells.parquet"
  "cell_boundaries.parquet"
  "nucleus_boundaries.parquet"
  "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"
  "experiment.xenium"
  "metrics_summary.csv"
)

for name in "${REQUIRED[@]}"; do
  path="${PDC_DATASET_ROOT}/${name}"
  if [[ ! -s "${path}" ]]; then
    echo "[mechanostress-pdc] missing required input: ${path}" >&2
    exit 3
  fi
  ln -sfn "${path}" "${SAMPLE_DIR}/${name}"
done

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

ANNOTATION="${SAMPLE_DIR}/WTA_Preview_FFPE_Breast_Cancer_cell_clusters_with_annotation_and_coord.csv"
conda run --prefix "${CONDA_PREFIX}" python \
  "${REPO_DIR}/benchmarking/mechanostress_pdc/scripts/derive_atera_annotation.py" \
  --dataset-root "${PDC_DATASET_ROOT}" \
  --output-csv "${ANNOTATION}"

echo "[mechanostress-pdc] input preparation completed $(date -Is)"
