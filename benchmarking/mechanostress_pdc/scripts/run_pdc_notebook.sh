#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04}"
PDC_DATASET_ROOT="${PDC_DATASET_ROOT:-/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-mechanostress}"
SAMPLE_ID="${SAMPLE_ID:-WTA_Preview_FFPE_Breast_Cancer_outs}"

mkdir -p "${PDC_ROOT}/logs" "${PDC_ROOT}/notebooks" "${PDC_ROOT}/outputs" "${PDC_ROOT}/static/mechanostress_atera_pdc" "${PDC_ROOT}/tmp"
exec > >(tee -a "${PDC_ROOT}/logs/run_pdc_notebook.log") 2>&1

echo "[mechanostress-pdc] notebook run started $(date -Is)"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export PDC_ROOT PDC_DATASET_ROOT REPO_DIR CONDA_PREFIX SAMPLE_ID
export TMPDIR="${PDC_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

bash "${REPO_DIR}/benchmarking/mechanostress_pdc/scripts/prepare_pdc_inputs.sh"

export PYXENIUM_MECHANOSTRESS_RUN_FULL=1
export PYXENIUM_MECHANOSTRESS_DATASET_ROOT="${PDC_DATASET_ROOT}"
export PYXENIUM_MECHANOSTRESS_COHORT_ROOT="${PDC_ROOT}/staging"
export PYXENIUM_MECHANOSTRESS_SAMPLE_GLOB="${SAMPLE_ID}"
export PYXENIUM_MECHANOSTRESS_OUTPUT_ROOT="${PDC_ROOT}/outputs"
export PYXENIUM_MECHANOSTRESS_STATIC_DIR="${PDC_ROOT}/static/mechanostress_atera_pdc"

TEMPLATE="${REPO_DIR}/docs/tutorials/mechanostress_atera_pdc.ipynb"
EXECUTED="${PDC_ROOT}/notebooks/mechanostress_atera_pdc.executed.ipynb"

conda run --prefix "${CONDA_PREFIX}" python \
  "${REPO_DIR}/benchmarking/mechanostress_pdc/scripts/execute_notebook.py" \
  --input "${TEMPLATE}" \
  --output "${EXECUTED}" \
  --timeout 14400

echo "[mechanostress-pdc] executed notebook: ${EXECUTED}"
echo "[mechanostress-pdc] notebook run completed $(date -Is)"
