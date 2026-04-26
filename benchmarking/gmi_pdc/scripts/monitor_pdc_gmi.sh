#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-gmi}"

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

echo "=== squeue ==="
squeue -u "${USER}" -o "%.18i %.12P %.30j %.8T %.10M %.9l %.6D %R" | grep -E 'JOBID|pyxgmi' || true
echo "=== contour preflight ==="
if [[ -s "${PDC_ROOT}/logs/prepare_pdc_inputs.log" ]]; then
  tail -n 40 "${PDC_ROOT}/logs/prepare_pdc_inputs.log"
else
  echo "no prepare_pdc_inputs.log yet"
fi
echo "=== monitor ==="
cd "${REPO_DIR}"
conda run --prefix "${CONDA_PREFIX}" pyxenium gmi pdc-monitor --pdc-root "${PDC_ROOT}"
