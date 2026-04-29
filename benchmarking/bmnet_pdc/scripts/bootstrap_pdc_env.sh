#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-bmnet}"
CONDA_PKGS_DIR="${CONDA_PKGS_DIR:-${PDC_ROOT}/conda/pkgs}"
LOG_DIR="${PDC_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${PDC_ROOT}/conda/envs" "${CONDA_PKGS_DIR}" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/bootstrap_pdc_env.log") 2>&1

echo "[bmnet-pdc] bootstrap started $(date -Is)"
echo "[bmnet-pdc] pdc_root=${PDC_ROOT}"
echo "[bmnet-pdc] repo_dir=${REPO_DIR}"
echo "[bmnet-pdc] conda_prefix=${CONDA_PREFIX}"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"
export TMPDIR="${PDC_ROOT}/tmp"

cd "${REPO_DIR}"

if [[ -d "${CONDA_PREFIX}" ]]; then
  echo "[bmnet-pdc] updating conda prefix"
  conda env update --prefix "${CONDA_PREFIX}" --file benchmarking/bmnet_pdc/envs/pyx-bmnet-pdc.yml --prune
else
  echo "[bmnet-pdc] creating conda prefix"
  conda env create --prefix "${CONDA_PREFIX}" --file benchmarking/bmnet_pdc/envs/pyx-bmnet-pdc.yml
fi

echo "[bmnet-pdc] bootstrap completed $(date -Is)"
