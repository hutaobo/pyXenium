#!/usr/bin/env bash
set -euo pipefail

PYXENIUM_REPO="${PYXENIUM_REPO:-$(pwd)}"
A100_ENV_DIR="${A100_ENV_DIR:-${PYXENIUM_REPO}/.venv-lazyslide-a100}"

if [[ -f "${A100_ENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${A100_ENV_DIR}/bin/activate"
fi

python "${PYXENIUM_REPO}/benchmarking/lazyslide_a100/scripts/run_histoseg_lazyslide_workflow.py" "$@"
