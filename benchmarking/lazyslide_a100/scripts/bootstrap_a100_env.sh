#!/usr/bin/env bash
set -euo pipefail

PYXENIUM_REPO="${PYXENIUM_REPO:-$(pwd)}"
A100_ENV_DIR="${A100_ENV_DIR:-${PYXENIUM_REPO}/.venv-lazyslide-a100}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${A100_ENV_DIR}"
# shellcheck source=/dev/null
source "${A100_ENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e "${PYXENIUM_REPO}[lazyslide]"

# LazySlide spatial-domain examples use the Scanpy/Leiden stack. Keep these in
# the A100 environment rather than pyXenium core.
python -m pip install "igraph>=0.11" "leidenalg>=0.10"

python - <<'PY'
import importlib.metadata as md

for package in ["pyXenium", "lazyslide", "wsidata"]:
    try:
        print(f"{package}=={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}: not installed")
PY
