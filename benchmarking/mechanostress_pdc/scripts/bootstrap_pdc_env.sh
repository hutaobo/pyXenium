#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-mechanostress}"
CONDA_PKGS_DIR="${CONDA_PKGS_DIR:-${PDC_ROOT}/conda/pkgs}"
LOG_DIR="${PDC_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${PDC_ROOT}/conda/envs" "${CONDA_PKGS_DIR}" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/bootstrap_pdc_env.log") 2>&1

echo "[mechanostress-pdc] bootstrap started $(date -Is)"
echo "[mechanostress-pdc] pdc_root=${PDC_ROOT}"
echo "[mechanostress-pdc] repo_dir=${REPO_DIR}"
echo "[mechanostress-pdc] conda_prefix=${CONDA_PREFIX}"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"
export TMPDIR="${PDC_ROOT}/tmp"

cd "${REPO_DIR}"

if [[ -d "${CONDA_PREFIX}" ]]; then
  echo "[mechanostress-pdc] updating conda prefix"
  conda env update --prefix "${CONDA_PREFIX}" --file benchmarking/mechanostress_pdc/envs/pyx-mechanostress-pdc.yml --prune
else
  echo "[mechanostress-pdc] creating conda prefix"
  conda env create --prefix "${CONDA_PREFIX}" --file benchmarking/mechanostress_pdc/envs/pyx-mechanostress-pdc.yml
fi

echo "[mechanostress-pdc] bootstrap completed $(date -Is)"
