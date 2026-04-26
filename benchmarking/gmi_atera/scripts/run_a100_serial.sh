#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/data/taobo.hu/pyxenium_gmi_contour_2026-04}"
REMOTE_XENIUM_ROOT="${REMOTE_XENIUM_ROOT:-/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs}"
ENV_NAME="${ENV_NAME:-pyx-gmi}"
REPO_DIR="${REPO_DIR:-${REMOTE_ROOT}/repo}"
LOG_DIR="${REMOTE_ROOT}/logs"
MAIN_LOG="${LOG_DIR}/gmi_a100_serial.log"

mkdir -p "${LOG_DIR}" "${REMOTE_ROOT}/runs" "${REMOTE_ROOT}/results" "${REMOTE_ROOT}/reports" "${REMOTE_ROOT}/tmp"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "[gmi-a100] started $(date -Is)"
echo "[gmi-a100] repo=${REPO_DIR}"
echo "[gmi-a100] dataset=${REMOTE_XENIUM_ROOT}"
echo "[gmi-a100] remote_root=${REMOTE_ROOT}"

if [[ ! -d "${REMOTE_XENIUM_ROOT}" ]]; then
  echo "[gmi-a100] missing read-only dataset: ${REMOTE_XENIUM_ROOT}" >&2
  exit 2
fi
if [[ "${REMOTE_ROOT}" == /mnt/* ]]; then
  echo "[gmi-a100] refusing to write under /mnt: ${REMOTE_ROOT}" >&2
  exit 2
fi

export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${PATH}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
export PYTHONNOUSERSITE=1
export TMPDIR="${REMOTE_ROOT}/tmp"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

cd "${REPO_DIR}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[gmi-a100] updating conda env ${ENV_NAME}"
  conda env update --name "${ENV_NAME}" --file benchmarking/gmi_atera/envs/pyx-gmi.yml --prune
else
  echo "[gmi-a100] creating conda env ${ENV_NAME}"
  conda env create --file benchmarking/gmi_atera/envs/pyx-gmi.yml
fi

echo "[gmi-a100] bootstrapping vendored Gmi"
conda run --name "${ENV_NAME}" Rscript benchmarking/gmi_atera/scripts/bootstrap_gmi_env.R

echo "[gmi-a100] writing plan"
conda run --name "${ENV_NAME}" pyxenium gmi a100-plan \
  --remote-xenium-root "${REMOTE_XENIUM_ROOT}" \
  --remote-root "${REMOTE_ROOT}" \
  --output-json benchmarking/gmi_atera/logs/a100_gmi_plan.json

echo "[gmi-a100] smoke run"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/smoke_contour_top200_spatial50" \
  --rna-feature-count 200 \
  --spatial-feature-count 50

echo "[gmi-a100] full run"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/full_contour_top500_spatial100" \
  --rna-feature-count 500 \
  --spatial-feature-count 100

echo "[gmi-a100] stability run"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/full_contour_top500_spatial100_stability" \
  --rna-feature-count 500 \
  --spatial-feature-count 100 \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10 \
  --label-permutation-control \
  --coordinate-shuffle-control \
  --spatial-feature-shuffle-control

echo "[gmi-a100] validation rna-only qc20"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/validation_rna_only_qc20" \
  --rna-feature-count 500 \
  --spatial-feature-count 0 \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10

echo "[gmi-a100] validation spatial-only qc20"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/validation_spatial_only_qc20" \
  --rna-feature-count 0 \
  --spatial-feature-count 100 \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10

echo "[gmi-a100] validation no-coordinate qc20"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/validation_no_coordinate_qc20" \
  --rna-feature-count 500 \
  --spatial-feature-count 100 \
  --exclude-coordinate-spatial-features \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10

echo "[gmi-a100] sensitivity top1000 qc20"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/sensitivity_top1000_spatial100_qc20" \
  --rna-feature-count 1000 \
  --spatial-feature-count 100 \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10

echo "[gmi-a100] sensitivity all-nonempty"
conda run --name "${ENV_NAME}" pyxenium gmi run \
  --dataset-root "${REMOTE_XENIUM_ROOT}" \
  --output-dir "${REMOTE_ROOT}/runs/sensitivity_all_nonempty_top500_spatial100" \
  --rna-feature-count 500 \
  --spatial-feature-count 100 \
  --min-cells-per-contour 1 \
  --spatial-cv-folds 5 \
  --bootstrap-repeats 10

echo "[gmi-a100] completed $(date -Is)"
