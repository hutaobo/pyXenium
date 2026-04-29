#!/usr/bin/env bash
set -euo pipefail

PDC_ROOT="${PDC_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04}"
PDC_XENIUM_ROOT="${PDC_XENIUM_ROOT:-/cfs/klemming/scratch/h/hutaobo/topolink_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs}"
REPO_DIR="${REPO_DIR:-${PDC_ROOT}/repo}"
CONDA_PREFIX="${CONDA_PREFIX:-${PDC_ROOT}/conda/envs/pyx-gmi}"
HISTOSEG_ROOT="${HISTOSEG_ROOT:-${PDC_ROOT}/external/HistoSeg}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdc-root) PDC_ROOT="$2"; shift 2 ;;
    --dataset-root|--pdc-xenium-root) PDC_XENIUM_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-prefix) CONDA_PREFIX="$2"; shift 2 ;;
    --histoseg-root) HISTOSEG_ROOT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "[gmi-pdc] unknown option: $1" >&2; exit 2 ;;
  esac
done

LOG_DIR="${PDC_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${PDC_ROOT}/contour_generation" "${PDC_ROOT}/tmp"
exec > >(tee -a "${LOG_DIR}/prepare_pdc_inputs.log") 2>&1

echo "[gmi-pdc] input preparation started $(date -Is)"
echo "[gmi-pdc] dataset=${PDC_XENIUM_ROOT}"
echo "[gmi-pdc] repo=${REPO_DIR}"
echo "[gmi-pdc] histoseg_root=${HISTOSEG_ROOT}"

REQUIRED=(
  "${PDC_XENIUM_ROOT}/cell_feature_matrix.h5"
  "${PDC_XENIUM_ROOT}/cells.parquet"
  "${PDC_XENIUM_ROOT}/WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"
  "${PDC_XENIUM_ROOT}/analysis/analysis/clustering/gene_expression_graphclust/clusters.csv"
)

for path in "${REQUIRED[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "[gmi-pdc] missing required input: ${path}" >&2
    echo "[gmi-pdc] the source cache copy is not complete enough for S1/S5 contour generation yet." >&2
    exit 3
  fi
done

CONTOUR_GEOJSON="${PDC_XENIUM_ROOT}/xenium_explorer_annotations.s1_s5.generated.geojson"
if [[ -s "${CONTOUR_GEOJSON}" && "${FORCE}" -eq 0 ]]; then
  echo "[gmi-pdc] contour GeoJSON already exists: ${CONTOUR_GEOJSON}"
  exit 0
fi

if [[ ! -d "${CONDA_PREFIX}" ]]; then
  echo "[gmi-pdc] missing conda prefix: ${CONDA_PREFIX}" >&2
  echo "[gmi-pdc] run benchmarking/gmi_pdc/scripts/bootstrap_pdc_env.sh first." >&2
  exit 3
fi

module load PDC/24.11
module load miniconda3/25.3.1-1-cpeGNU-24.11

export TMPDIR="${PDC_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
if [[ -d "${HISTOSEG_ROOT}/src" ]]; then
  export PYTHONPATH="${HISTOSEG_ROOT}/src:${PYTHONPATH}"
fi

cd "${REPO_DIR}"

ARGS=(
  benchmarking/gmi_pdc/scripts/prepare_s1_s5_geojson.py
  --dataset-root "${PDC_XENIUM_ROOT}"
  --pdc-root "${PDC_ROOT}"
)
if [[ -d "${HISTOSEG_ROOT}" ]]; then
  ARGS+=(--histoseg-root "${HISTOSEG_ROOT}")
fi
if [[ "${FORCE}" -eq 1 ]]; then
  ARGS+=(--force)
fi

conda run --prefix "${CONDA_PREFIX}" python "${ARGS[@]}"

if [[ ! -s "${CONTOUR_GEOJSON}" ]]; then
  echo "[gmi-pdc] contour generation finished but GeoJSON is still missing: ${CONTOUR_GEOJSON}" >&2
  exit 3
fi

echo "[gmi-pdc] input preparation completed $(date -Is)"
