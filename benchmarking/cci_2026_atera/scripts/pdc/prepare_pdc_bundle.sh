#!/usr/bin/env bash
set -euo pipefail

ROOT="${PDC_CCI_ROOT:-/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04}"
SOURCE_ROOT="${PDC_CCI_SOURCE_ROOT:-$ROOT/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs}"
PY_ENV="${PDC_CCI_PREP_ENV:-$ROOT/envs/python/prep}"
PYTHON_MODULE="${PDC_LR_PYTHON_MODULE:-python/3.12.3}"

mkdir -p "$ROOT"/{data/full,data/smoke,logs,reports,results,runs,tmp,envs/python}
mkdir -p "$ROOT/configs"
cp -f "$ROOT"/repo/benchmarking/cci_2026_atera/configs/* "$ROOT/configs/" 2>/dev/null || true
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$ROOT/repo/src:${PYTHONPATH:-}"

if command -v module >/dev/null 2>&1; then
  module load "$PYTHON_MODULE" >/dev/null 2>&1 || true
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
if [ ! -x "$PY_ENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$PY_ENV"
fi

# shellcheck disable=SC1091
. "$PY_ENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$ROOT/repo"
python -m pip install pyarrow

python "$ROOT/repo/benchmarking/cci_2026_atera/scripts/prepare_data.py" \
  --dataset-root "$SOURCE_ROOT" \
  --benchmark-root "$ROOT" \
  --prefer h5 \
  --skip-full-h5ad \
  --output-json "$ROOT/logs/pdc_prepare_payload.json"

if [ -f "$ROOT/data/staged_common/cci_resource_common.tsv" ]; then
  cp -f "$ROOT/data/staged_common/cci_resource_common.tsv" "$ROOT/data/cci_resource_common.tsv"
  [ -f "$ROOT/data/staged_common/atera_smoke_panel.tsv" ] && cp -f "$ROOT/data/staged_common/atera_smoke_panel.tsv" "$ROOT/data/atera_smoke_panel.tsv"
  [ -f "$ROOT/data/staged_common/celltype_pairs.tsv" ] && cp -f "$ROOT/data/staged_common/celltype_pairs.tsv" "$ROOT/data/celltype_pairs.tsv"
  python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ.get("PDC_CCI_ROOT", "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"))
manifest_path = root / "data" / "input_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
lr_path = root / "data" / "cci_resource_common.tsv"
manifest["cci_resource_common_tsv"] = str(lr_path)
manifest["common_db"] = {"source": "staged_local_cci_db_common", "n_pairs": int(len(pd.read_csv(lr_path, sep="\t")))}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Patched PDC manifest to use staged common CCI resource: {manifest['common_db']}")
PY
fi

python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

root = Path(os.environ.get("PDC_CCI_ROOT", "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"))
manifest = root / "data" / "input_manifest.json"
payload = json.loads(manifest.read_text(encoding="utf-8"))
full_bundle = payload.get("full_bundle") or {}
required = ["counts_symbol_mtx", "barcodes_tsv", "genes_tsv", "meta_tsv", "coords_tsv"]
missing = [key for key in required if not full_bundle.get(key) or not Path(full_bundle[key]).exists()]
if missing:
    raise SystemExit(f"PDC bundle validation failed; missing full_bundle keys/files: {missing}")
if int(payload.get("full_n_cells", 0)) != 170057:
    raise SystemExit(f"Unexpected full_n_cells: {payload.get('full_n_cells')}")
validation = {
    "status": "ok",
    "manifest": str(manifest),
    "full_n_cells": payload.get("full_n_cells"),
    "full_n_genes": payload.get("full_n_genes"),
    "full_bundle": full_bundle,
    "source_root": payload.get("xenium_root"),
    "benchmark_root": payload.get("benchmark_root"),
}
(root / "logs" / "pdc_bundle_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
print(json.dumps(validation, indent=2))
PY
