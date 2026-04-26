#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-gmi}"
CONDA_PKGS_DIR="${CONDA_PKGS_DIR:-${PDC_ROOT}/conda/pkgs}"
LOG_DIR="${PDC_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${PDC_ROOT}/conda/envs" "${CONDA_PKGS_DIR}" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/bootstrap_pdc_env.log") 2>&1

echo "[gmi-pdc] bootstrap started $(date -Is)"
echo "[gmi-pdc] pdc_root=${PDC_ROOT}"
echo "[gmi-pdc] repo_dir=${REPO_DIR}"
echo "[gmi-pdc] conda_prefix=${CONDA_PREFIX}"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"
export TMPDIR="${PDC_ROOT}/tmp"

cd "${REPO_DIR}"

if [[ -d "${CONDA_PREFIX}" ]]; then
  echo "[gmi-pdc] updating conda prefix"
  conda env update --prefix "${CONDA_PREFIX}" --file benchmarking/gmi_pdc/envs/pyx-gmi-pdc.yml --prune
else
  echo "[gmi-pdc] creating conda prefix"
  conda env create --prefix "${CONDA_PREFIX}" --file benchmarking/gmi_pdc/envs/pyx-gmi-pdc.yml
fi

echo "[gmi-pdc] installing vendored Gmi"
conda run --prefix "${CONDA_PREFIX}" Rscript benchmarking/gmi_atera/scripts/bootstrap_gmi_env.R
echo "[gmi-pdc] bootstrap completed $(date -Is)"
