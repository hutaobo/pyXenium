from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PDC_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04"
DEFAULT_ACCOUNT = "naiss2026-4-680"
ALL_METHODS = (
    "pyxenium",
    "squidpy",
    "liana",
    "spatialdm",
    "laris",
    "cellphonedb",
    "stlearn",
    "cellchat",
    "commot",
    "giotto",
    "spatalk",
    "niches",
    "cellnest",
    "cellagentchat",
    "scild",
)
A100_AUTHORITATIVE_FULL_METHODS = (
    "pyxenium",
    "squidpy",
    "liana",
    "spatialdm",
    "stlearn",
    "cellphonedb",
    "laris",
)
PDC_FULL_BACKFILL_METHODS = tuple(method for method in ALL_METHODS if method not in A100_AUTHORITATIVE_FULL_METHODS)
PDC_REBUILD_FULL_METHODS = ("cellchat", "cellnest", "cellagentchat", "scild")
REBUILD_ENV_TAG = "rebuild_20260428"
SOURCE_PYTHON_REBUILD_METHODS = ("cellnest", "cellagentchat", "scild")
CELLAGENTCHAT_FULL_CHUNKS = 16
PYXENIUM_PY310_RUNTIME_PIP = (
    "PyYAML>=6.0",
    "click>=8.1",
    "fsspec>=2024.6.0",
    "pyarrow>=14,<18",
    "aiohttp",
    "shapely>=2.0",
    "tifffile>=2024.8.10,<2026",
    "imagecodecs>=2024.6.1,<2026",
    "zarr>=2.16,<3",
)
SOURCE_CHECKOUTS = {
    "cellchat": ("https://github.com/jinworks/CellChat.git", "75253cd0c9e68410e6e721a6d3a0419a1d7e358f"),
    "cellnest": ("https://github.com/schwartzlab-methods/CellNEST.git", "2737fa8f54952b4b35a540f6070655a69f2c4999"),
    "cellagentchat": ("https://github.com/mcgilldinglab/CellAgentChat.git", "37e51980cb9ba87684993d8bdae26feac8806bae"),
    "scild": ("https://github.com/jiatingyu-amss/SCILD.git", "683515043df1878f3069c4dd5f887abb5c8976bd"),
}


@dataclass(frozen=True)
class MethodConfig:
    language: str
    partition: str
    cpus: int
    memory: str
    time: str
    pip: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    r_packages: tuple[str, ...] = ()
    bioc_packages: tuple[str, ...] = ()
    github: tuple[str, ...] = ()
    pdc_note: str = ""


METHODS: dict[str, MethodConfig] = {
    "pyxenium": MethodConfig("python", "shared", 16, "96G", "06:00:00", imports=("pyXenium",)),
    "squidpy": MethodConfig("python", "shared", 16, "160G", "12:00:00", pip=("setuptools<81", "squidpy", "omnipath", "zarr<3"), imports=("squidpy",)),
    "liana": MethodConfig("python", "memory", 16, "300G", "18:00:00", pip=("liana", "omnipath", "mudata", "decoupler"), imports=("liana",)),
    "spatialdm": MethodConfig("python", "memory", 16, "300G", "18:00:00", pip=("git+https://github.com/StatBiomed/SpatialDM.git", "SparseAEH"), imports=("spatialdm",)),
    "laris": MethodConfig("python", "shared", 16, "192G", "18:00:00", pip=("laris",), imports=("laris",)),
    "cellphonedb": MethodConfig("python", "shared", 16, "160G", "12:00:00", pip=("cellphonedb",), imports=("cellphonedb",)),
    "stlearn": MethodConfig("python", "memory", 16, "300G", "18:00:00", pip=("stlearn",), imports=("stlearn",)),
    "commot": MethodConfig("python", "memory", 16, "300G", "24:00:00", pip=("numpy<2", "commot"), imports=("commot",)),
    "cellnest": MethodConfig(
        "python",
        "shared",
        16,
        "160G",
        "12:00:00",
        pip=(),
        imports=("torch", "torch_geometric"),
        pdc_note="Source-layout checkout at external_src/cellnest; prefer CellNEST Apptainer/Singularity image and fall back to a pinned Python env.",
    ),
    "cellagentchat": MethodConfig(
        "python",
        "memory",
        32,
        "300G",
        "24:00:00",
        pip=(
            "numpy==1.26.4",
            "pandas>=1.5.0,<2.3",
            "scipy>=1.9.1,<1.14",
            "scanpy>=1.9.6,<1.11",
            "anndata>=0.8.0,<0.11",
            "zarr>=2.16,<3",
            "Mesa==1.0.0",
            "pyslingshot==0.0.2",
            "sparselinear==0.0.5",
            "torch>=1.13,<2.4",
            "seaborn>=0.12",
            "matplotlib>=3.6",
        ),
        imports=("preprocessor", "model_setup", "abm", "Communication"),
        pdc_note="Source-layout checkout at external_src/cellagentchat; PYTHONPATH must include external_src/cellagentchat/src.",
    ),
    "scild": MethodConfig(
        "python",
        "shared",
        16,
        "160G",
        "12:00:00",
        pip=(
            "numpy==1.26.4",
            "pandas>=2.0,<2.4",
            "scipy==1.11.4",
            "scanpy>=1.10,<1.12",
            "anndata>=0.10,<0.12",
            "zarr>=2.16,<3",
            "commot==0.0.3",
            "h5py>=3.10,<3.15",
            "igraph>=0.11,<0.12",
            "leidenalg>=0.10,<0.11",
            "matplotlib>=3.8,<3.11",
            "mpl-chord-diagram>=0.4,<0.5",
            "networkx>=3.2,<3.5",
            "plotly>=5.0,<6.0",
            "scikit-image==0.22.0",
            "scikit-learn>=1.4,<1.8",
            "scikit-misc>=0.5,<0.6",
            "torch>=2.1,<2.8",
            "torch-geometric>=2.5,<2.7",
            "tqdm>=4.66",
        ),
        imports=("Models.SCILD_main",),
        pdc_note="Source-layout checkout at external_src/scild; PYTHONPATH must include external_src/scild.",
    ),
    "cellchat": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "future", "igraph", "NMF"),
        bioc_packages=("BiocGenerics", "Biobase", "BiocNeighbors", "ComplexHeatmap"),
        github=("jinworks/CellChat",),
    ),
    "giotto": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "igraph"),
        github=("drieslab/Giotto",),
    ),
    "spatalk": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "igraph"),
        github=("ZJUFanLab/SpaTalk",),
    ),
    "niches": MethodConfig(
        "r",
        "memory",
        32,
        "300G",
        "24:00:00",
        r_packages=("jsonlite", "Matrix", "data.table", "remotes", "SeuratObject"),
        github=("msraredon/NICHES",),
    ),
}


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run_command(command: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_command],
        check=check,
    )


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def rebuild_env_path(method: str, root: str) -> str:
    return f"{root}/envs/{REBUILD_ENV_TAG}/{method}"


def source_checkout_command(method: str, root: str) -> str:
    if method not in SOURCE_CHECKOUTS:
        return ""
    repo, commit = SOURCE_CHECKOUTS[method]
    src = f"{root}/external_src/{method}"
    return f"""
mkdir -p {q(root + "/external_src")}
if [ ! -d {q(src)}/.git ]; then
  rm -rf {q(src)}
  git clone {q(repo)} {q(src)}
fi
current_commit="$(git -C {q(src)} rev-parse HEAD 2>/dev/null || true)"
if [ "$current_commit" = {q(commit)} ]; then
  echo "source_checkout_ok {method} already at {commit}"
else
  if ! git -C {q(src)} cat-file -e {q(commit + "^{commit}")} 2>/dev/null; then
    git -C {q(src)} fetch --all --tags --prune
  fi
  git -C {q(src)} checkout {q(commit)}
fi
"""


def python_module_load(method: str) -> str:
    if method in SOURCE_PYTHON_REBUILD_METHODS:
        return (
            "module load python/3.10 >/dev/null 2>&1 || "
            "module load python/3.10.13 >/dev/null 2>&1 || "
            "module load cray-python/3.10.10 >/dev/null 2>&1 || "
            "module load python/3.12.3 >/dev/null 2>&1 || true"
        )
    return "module load python/3.12.3 >/dev/null 2>&1 || true"


def pythonpath_for_method(method: str, root: str) -> str:
    entries = [f"{root}/repo/src"]
    if method == "cellagentchat":
        entries.insert(0, f"{root}/external_src/cellagentchat/src")
    elif method == "scild":
        entries.insert(0, f"{root}/external_src/scild")
    elif method == "cellnest":
        entries.insert(0, f"{root}/external_src/cellnest")
    return ":".join(entries) + ":${PYTHONPATH:-}"


def python_env_setup(method: str, cfg: MethodConfig, root: str) -> str:
    env = rebuild_env_path(method, root) if method in PDC_REBUILD_FULL_METHODS else f"{root}/envs/python/{method}"
    pip_packages = " ".join(q(pkg) for pkg in cfg.pip)
    runtime_packages = " ".join(q(pkg) for pkg in PYXENIUM_PY310_RUNTIME_PIP) if method in SOURCE_PYTHON_REBUILD_METHODS else ""
    repo_install = 'python -m pip install -e "$ROOT/repo" --no-deps' if method in SOURCE_PYTHON_REBUILD_METHODS else 'python -m pip install -e "$ROOT/repo"'
    import_checks = "\n".join(
        [
            f"for name in {list(cfg.imports)!r}:",
            "    try:",
            "        __import__(name)",
            "        print(f'import_ok {name}')",
            "    except Exception as exc:",
            "        print(f'import_failed {name}: {exc}')",
            "        raise",
        ]
        if cfg.imports
        else ["import pyXenium; print('import_ok pyXenium')"]
    )
    source_setup = source_checkout_command(method, root)
    source_exports = ""
    if method in SOURCE_PYTHON_REBUILD_METHODS:
        source_exports = f"""
export {method.upper()}_SRC={q(root + "/external_src/" + method)}
export PYTHONPATH="{pythonpath_for_method(method, root)}"
"""
    cellnest_bootstrap = ""
    if method == "cellnest":
        cellnest_bootstrap = f"""
CELLNEST_IMAGE="{env}/cellnest_image.sif"
if command -v apptainer >/dev/null 2>&1; then
  if [ ! -s "$CELLNEST_IMAGE" ]; then apptainer pull "$CELLNEST_IMAGE" library://fatema/collection/cellnest_image.sif:latest || true; fi
elif command -v singularity >/dev/null 2>&1; then
  if [ ! -s "$CELLNEST_IMAGE" ]; then singularity pull "$CELLNEST_IMAGE" library://fatema/collection/cellnest_image.sif:latest || true; fi
fi
if [ -s "$CELLNEST_IMAGE" ]; then
  export CELLNEST_CONTAINER="$CELLNEST_IMAGE"
fi
"""
    cellnest_fallback_pip = ""
    if method == "cellnest":
        cellnest_fallback_pip = """
if [ ! -n "${CELLNEST_CONTAINER:-}" ]; then
  python -m pip install numpy==1.26.4 pandas==2.2.1 scipy==1.12.0 anndata==0.10.6 scanpy==1.9.8 'zarr>=2.16,<3' qnorm==0.8.1 networkx==3.2.1 scikit-learn==1.4.1.post1
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || python -m pip install torch torchvision torchaudio
  python -m pip install torch-geometric || true
fi
"""
        import_checks = """
import os
import shutil
import subprocess
container = os.environ.get("CELLNEST_CONTAINER")
runtime = shutil.which("apptainer") or shutil.which("singularity")
if container and runtime:
    subprocess.check_call([runtime, "exec", container, "python", "-c", "import torch; print('cellnest_container_ok')"])
else:
    import torch
    import torch_geometric
    print("import_ok torch torch_geometric")
"""
    return f"""
METHOD={q(method)}
OUT="$ROOT/runs/env_audit/{method}"
mkdir -p "$OUT"
rm -f "$OUT/method_card.md" "$OUT/run_summary.json" "$OUT/pip_freeze.txt" "$OUT/torch_wheel_tag.env"
trap 'rc=$?; mkdir -p "$OUT"; printf "# %s PDC method card\\n\\n- Status: failed\\n- Stage: env_setup\\n- Exit code: %s\\n- Note: {cfg.pdc_note}\\n" "$METHOD" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
{source_setup}
{python_module_load(method)}
PYTHON_BIN="$(command -v python3 || command -v python)"
if [ {str(method in PDC_REBUILD_FULL_METHODS).lower()} = true ]; then
  rm -rf {q(env)}
fi
if [ ! -x {q(env)}/bin/python ]; then
  "$PYTHON_BIN" -m venv {q(env)}
fi
. {q(env)}/bin/activate
{source_exports}
python -m pip install --upgrade pip setuptools wheel
{repo_install}
if [ -n {q(runtime_packages)} ]; then
  python -m pip install {runtime_packages}
fi
if [ -n {q(pip_packages)} ]; then
  python -m pip install {pip_packages}
fi
if [ "$METHOD" = "cellagentchat" ]; then
  python - <<'PY' > "$OUT/torch_wheel_tag.env"
import torch
base = torch.__version__.split("+", 1)[0]
cuda = torch.version.cuda
tag = "cpu" if not cuda else "cu" + cuda.replace(".", "")
print(f"TORCH_VERSION={{base}}")
print(f"TORCH_WHEEL_TAG={{tag}}")
PY
  . "$OUT/torch_wheel_tag.env"
  python -m pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-${{TORCH_VERSION}}+${{TORCH_WHEEL_TAG}}.html" || python -m pip install torch-scatter torch-sparse
fi
{cellnest_bootstrap}
{cellnest_fallback_pip}
python - <<'PY'
{import_checks}
PY
python -m pip freeze > "$OUT/pip_freeze.txt"
printf '{{"method":"{method}","status":"success","stage":"env_setup","language":"python"}}\\n' > "$OUT/run_summary.json"
"""


def r_env_setup(method: str, cfg: MethodConfig, root: str) -> str:
    r_lib = f"{rebuild_env_path(method, root)}/Rlib" if method in PDC_REBUILD_FULL_METHODS else f"{root}/envs/r_libs/{method}"
    r_packages = ", ".join(repr(pkg) for pkg in cfg.r_packages)
    bioc_packages = ", ".join(repr(pkg) for pkg in cfg.bioc_packages)
    github = ", ".join(repr(repo) for repo in cfg.github)
    source_setup = source_checkout_command(method, root)
    local_source = f"{root}/external_src/{method}" if method in SOURCE_CHECKOUTS else ""
    r_preflight = (
        f"Rscript -e '.libPaths(c(Sys.getenv(\"R_LIBS_USER\"), .libPaths())); library(CellChat); library(NMF); sink(file.path(\"{r_lib}\", \"sessionInfo.txt\")); print(sessionInfo()); sink()'"
        if method == "cellchat"
        else f"Rscript -e '.libPaths(c(Sys.getenv(\"R_LIBS_USER\"), .libPaths())); sink(file.path(\"{r_lib}\", \"sessionInfo.txt\")); print(sessionInfo()); sink()'"
    )
    return f"""
METHOD={q(method)}
OUT="$ROOT/runs/env_audit/{method}"
mkdir -p "$OUT"
rm -f "$OUT/method_card.md" "$OUT/run_summary.json"
trap 'rc=$?; mkdir -p "$OUT"; printf "# %s PDC method card\\n\\n- Status: failed\\n- Stage: r_env_setup\\n- Exit code: %s\\n" "$METHOD" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
{source_setup}
module load libffi/3.4.2 >/dev/null 2>&1 || true
module load cray-R/4.4.0 >/dev/null 2>&1 || true
if [ {str(method in PDC_REBUILD_FULL_METHODS).lower()} = true ]; then
  rm -rf {q(rebuild_env_path(method, root))}
fi
mkdir -p {q(r_lib)}
export R_LIBS_USER={q(r_lib)}
Rscript - <<'RS'
repos <- c(CRAN = "https://cloud.r-project.org")
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
bioc_pkgs <- c({bioc_packages})
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = lib, repos = repos)
if (length(bioc_pkgs)) {{
  missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_bioc)) BiocManager::install(missing_bioc, lib = lib, ask = FALSE, update = FALSE)
  missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_bioc)) stop("missing Bioconductor packages after install: ", paste(missing_bioc, collapse = ", "))
}}
pkgs <- c({r_packages})
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, lib = lib, repos = repos, Ncpus = 8)
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("missing CRAN packages after install: ", paste(missing, collapse = ", "))
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", lib = lib, repos = repos)
if (!requireNamespace("remotes", quietly = TRUE)) stop("missing CRAN package after install: remotes")
repos_gh <- c({github})
install_failures <- character()
for (repo in repos_gh) {{
  pkg <- basename(repo)
  local_source <- "{local_source}"
  if (!requireNamespace(pkg, quietly = TRUE) && nzchar(local_source) && dir.exists(local_source)) {{
    tryCatch(
      remotes::install_local(local_source, lib = lib, upgrade = "never", dependencies = c("Depends", "Imports", "LinkingTo")),
      error = function(exc) {{
        install_failures <<- c(install_failures, sprintf("%s local install failed: %s", repo, conditionMessage(exc)))
      }}
    )
  }}
  if (!requireNamespace(pkg, quietly = TRUE)) {{
    tryCatch(
      remotes::install_github(repo, lib = lib, upgrade = "never", dependencies = c("Depends", "Imports", "LinkingTo")),
      error = function(exc) {{
        install_failures <<- c(install_failures, sprintf("%s install failed: %s", repo, conditionMessage(exc)))
      }}
    )
  }}
  if (!requireNamespace(pkg, quietly = TRUE)) {{
    install_failures <- c(install_failures, sprintf("%s namespace unavailable after install", pkg))
  }}
}}
if (length(install_failures)) stop(paste(install_failures, collapse = "; "))
sessionInfo()
RS
{r_preflight}
printf '{{"method":"{method}","status":"success","stage":"env_setup","language":"r"}}\\n' > "$OUT/run_summary.json"
"""


def activate_for_method(method: str, cfg: MethodConfig, root: str) -> str:
    if cfg.language == "r":
        r_lib = f"{rebuild_env_path(method, root)}/Rlib" if method in PDC_REBUILD_FULL_METHODS else f"{root}/envs/r_libs/{method}"
        r_preflight = (
            "Rscript -e '.libPaths(c(Sys.getenv(\"R_LIBS_USER\"), .libPaths())); library(CellChat); library(NMF); cat(\"r_preflight_ok\\\\n\")'"
            if method == "cellchat"
            else "Rscript -e '.libPaths(c(Sys.getenv(\"R_LIBS_USER\"), .libPaths())); cat(\"r_preflight_ok\\\\n\")'"
        )
        return f"""
module load libffi/3.4.2 >/dev/null 2>&1 || true
module load cray-R/4.4.0 >/dev/null 2>&1 || true
. {q(root)}/envs/python/prep/bin/activate
python - <<'PY'
import ctypes
import anndata
import pandas
print("python_preflight_ok ctypes anndata pandas")
PY
export R_LIBS_USER={q(r_lib)}
export PATH="$(dirname "$(command -v Rscript)"):$PATH"
{r_preflight}
"""
    env = rebuild_env_path(method, root) if method in PDC_REBUILD_FULL_METHODS else f"{root}/envs/python/{method}"
    source_exports = ""
    if method in {"cellnest", "cellagentchat", "scild"}:
        source_exports = f"""
export {method.upper()}_SRC={q(root + "/external_src/" + method)}
export PYTHONPATH="{pythonpath_for_method(method, root)}"
"""
    if method == "cellnest":
        source_exports += f"""
if [ -s {q(env + "/cellnest_image.sif")} ]; then export CELLNEST_CONTAINER={q(env + "/cellnest_image.sif")}; fi
"""
    return f"""
{python_module_load(method)}
. {q(env)}/bin/activate
{source_exports}
"""


def method_run_command(
    method: str,
    cfg: MethodConfig,
    root: str,
    stage: str,
    *,
    chunk_id: int | None = None,
    num_chunks: int | None = None,
) -> str:
    phase = "smoke" if stage == "smoke" else "full"
    max_lr = {"smoke": 25, "pilot": 500, "full": None}[stage]
    out_dir = f"{root}/runs/{stage}_common/{method}"
    extra = ""
    if method == "commot" and stage == "pilot":
        extra += " --chunk-id 0 --num-chunks 16"
        out_dir = f"{root}/runs/{stage}_common/{method}/chunk_000_of_016"
    if chunk_id is not None:
        if num_chunks is None:
            raise ValueError("num_chunks is required when chunk_id is provided.")
        extra += f" --chunk-id {int(chunk_id)} --num-chunks {int(num_chunks)}"
        out_dir = f"{root}/runs/{stage}_common/{method}/chunk_{int(chunk_id):03d}_of_{int(num_chunks):03d}"
    if max_lr is not None:
        extra += f" --max-cci-pairs {max_lr}"
    if stage == "pilot":
        extra += " --bounded-mode full_cells_lr500_pilot"
    if cfg.language == "r":
        extra += " --rscript Rscript"
    job_suffix = "" if chunk_id is None else f"_chunk_{int(chunk_id):03d}_of_{int(num_chunks or 0):03d}"
    cellagentchat_exports = ""
    if method == "cellagentchat":
        cellagentchat_exports = """
export CELLAGENTCHAT_FEATURE_SELECTION="${CELLAGENTCHAT_FEATURE_SELECTION:-0}"
export CELLAGENTCHAT_EPOCHS="${CELLAGENTCHAT_EPOCHS:-10}"
export CELLAGENTCHAT_MAX_STEPS="${CELLAGENTCHAT_MAX_STEPS:-1}"
"""
    return f"""
OUT={q(out_dir)}
mkdir -p "$OUT"
rm -f "$OUT/method_card.md" "$OUT/params.json" "$OUT/run_summary.json" "$OUT"/*_standardized.tsv "$OUT"/*_standardized.tsv.gz "$OUT/standardized_results.tsv" "$OUT/standardized_results.tsv.gz"
trap 'rc=$?; mkdir -p "$OUT"; printf "# {method} PDC method card\\n\\n- Status: failed\\n- Stage: {stage}\\n- Exit code: %s\\n- PDC note: {cfg.pdc_note}\\n" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
{activate_for_method(method, cfg, root)}
{cellagentchat_exports}
python "$ROOT/repo/benchmarking/cci_2026_atera/scripts/run_method.py" \
  --method {q(method)} \
  --input-manifest "$ROOT/data/input_manifest.json" \
  --benchmark-root "$ROOT" \
  --database-mode common-db \
  --phase {phase} \
  --output-dir "$OUT" \
  --gzip-standardized \
  --job-id {q(stage + "_common_" + method + job_suffix)}{extra}
"""


def aggregate_chunk_command(method: str, cfg: MethodConfig, root: str, stage: str, *, num_chunks: int) -> str:
    if method != "cellagentchat" or stage != "full":
        raise ValueError("Only cellagentchat full chunk aggregation is supported.")
    out_dir = f"{root}/runs/{stage}_common/{method}"
    return f"""
OUT={q(out_dir)}
mkdir -p "$OUT"
rm -f "$OUT/method_card.md" "$OUT/run_summary.json" "$OUT/{method}_standardized.tsv" "$OUT/{method}_standardized.tsv.gz" "$OUT/standardized_results.tsv" "$OUT/standardized_results.tsv.gz"
trap 'rc=$?; mkdir -p "$OUT"; printf "# {method} PDC method card\\n\\n- Status: failed\\n- Stage: {stage}_aggregate\\n- Exit code: %s\\n- PDC note: {cfg.pdc_note}\\n" "$rc" > "$OUT/method_card.md"; exit "$rc"' ERR
{activate_for_method(method, cfg, root)}
python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyXenium.benchmarking.cci_atera import STANDARDIZED_RESULT_COLUMNS, _resolve_ordered_rank

out = Path({out_dir!r})
expected = {int(num_chunks)}
failed_statuses = {{"failed", "error", "out_of_memory", "timeout", "cancelled", "dependencyneversatisfied", "dependency_never_satisfied"}}
frames = []
inputs = []
missing = []
blocked = []
for idx in range(expected):
    chunk_dir = out / f"chunk_{{idx:03d}}_of_{{expected:03d}}"
    summary_path = chunk_dir / "run_summary.json"
    summary = {{}}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = str(summary.get("status") or "").strip().lower()
    if status in failed_statuses or (chunk_dir / "method_card.md").exists():
        blocked.append(idx)
        continue
    candidates = sorted(chunk_dir.glob("*standardized.tsv.gz")) + sorted(chunk_dir.glob("*standardized.tsv"))
    if not candidates:
        missing.append(idx)
        continue
    path = candidates[-1]
    table = pd.read_csv(path, sep="\\t", compression="infer")
    missing_columns = [column for column in STANDARDIZED_RESULT_COLUMNS if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Chunk {{idx}} output {{path}} is missing standardized columns: {{missing_columns}}")
    frames.append(table.copy())
    inputs.append(str(path))
if missing or blocked:
    raise RuntimeError(f"Cannot aggregate {method}: missing_chunks={{missing}}, blocked_chunks={{blocked}}")
if not frames:
    raise RuntimeError(f"Cannot aggregate {method}: no chunk tables were found.")
combined = pd.concat(frames, ignore_index=True)
score = pd.to_numeric(combined["score_raw"], errors="coerce")
pvalues = pd.to_numeric(combined["fdr_or_pvalue"], errors="coerce") if "fdr_or_pvalue" in combined.columns else None
rank, rank_fraction = _resolve_ordered_rank(score, pvalues)
combined["rank_within_method"] = rank.astype(float)
combined["rank_fraction"] = rank_fraction.astype(float)
combined["score_std"] = rank_fraction.astype(float)
extra_columns = [column for column in combined.columns if column not in STANDARDIZED_RESULT_COLUMNS]
combined = combined.loc[:, STANDARDIZED_RESULT_COLUMNS + extra_columns].sort_values("rank_within_method").reset_index(drop=True)
standardized_path = out / "{method}_standardized.tsv.gz"
combined.to_csv(standardized_path, sep="\\t", index=False, compression="gzip")
summary = {{
    "method": "{method}",
    "status": "success",
    "stage": "{stage}_aggregate",
    "database_mode": "common-db",
    "standardized_tsv": str(standardized_path),
    "standardized_tsv_gz": str(standardized_path),
    "n_rows": int(len(combined)),
    "num_chunks": expected,
    "chunk_inputs": inputs,
    "top_hit": combined.head(1).to_dict(orient="records"),
}}
(out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
"""


def prepare_command(root: str) -> str:
    return f"bash {q(root)}/repo/benchmarking/cci_2026_atera/scripts/pdc/prepare_pdc_bundle.sh"


def job_script(
    *,
    job_id: str,
    root: str,
    account: str,
    partition: str,
    cpus: int,
    memory: str,
    time_limit: str,
    body: str,
) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH -t {time_limit}
#SBATCH -c {cpus}
#SBATCH --mem={memory}
#SBATCH -J {job_id[:64]}
#SBATCH -o {root}/logs/{job_id}.stdout.log
#SBATCH -e {root}/logs/{job_id}.stderr.log

set -euo pipefail
export ROOT={q(root)}
export PDC_CCI_ROOT="$ROOT"
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$ROOT/repo/src:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
mkdir -p "$ROOT"/{{logs,runs,results,reports,tmp,slurm,envs/python,envs/r_libs,envs/{REBUILD_ENV_TAG},external_src}}
mkdir -p "$ROOT/configs"
cp -f "$ROOT"/repo/benchmarking/cci_2026_atera/configs/* "$ROOT/configs/" 2>/dev/null || true
cd "$ROOT/repo"
/usr/bin/time -v bash -lc {q(body)} 2> >(tee -a "$ROOT/logs/{job_id}.resource.log" >&2)
"""


def build_jobs(
    methods: list[str],
    root: str,
    account: str,
    stages: set[str],
    include_full: bool,
    *,
    commot_chunks: int = 16,
    cellagentchat_chunks: int = CELLAGENTCHAT_FULL_CHUNKS,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if "prepare" in stages:
        jobs.append(
            {
                "job_id": "pdc_prepare_full_bundle",
                "job_type": "prepare",
                "method": "prep",
                "partition": "shared",
                "cpus": 16,
                "memory": "96G",
                "time": "02:00:00",
                "dependencies": [],
                "script": job_script(
                    job_id="pdc_prepare_full_bundle",
                    root=root,
                    account=account,
                    partition="shared",
                    cpus=16,
                    memory="96G",
                    time_limit="02:00:00",
                    body=prepare_command(root),
                ),
            }
        )
    for method in methods:
        cfg = METHODS[method]
        env_id = f"pdc_env_{method}"
        if "env" in stages:
            body = python_env_setup(method, cfg, root) if cfg.language == "python" else r_env_setup(method, cfg, root)
            jobs.append(
                {
                    "job_id": env_id,
                    "job_type": "env_setup",
                    "method": method,
                    "partition": "shared" if cfg.language == "python" else "memory",
                    "cpus": 8 if cfg.language == "python" else 16,
                    "memory": "64G" if cfg.language == "python" else "160G",
                    "time": "04:00:00" if cfg.language == "python" else "12:00:00",
                    "dependencies": [],
                    "script": job_script(
                        job_id=env_id,
                        root=root,
                        account=account,
                        partition="shared" if cfg.language == "python" else "memory",
                        cpus=8 if cfg.language == "python" else 16,
                        memory="64G" if cfg.language == "python" else "160G",
                        time_limit="04:00:00" if cfg.language == "python" else "12:00:00",
                        body=body,
                    ),
                }
            )
        prior_stage_id = None
        for stage in ("smoke", "pilot", "full"):
            if stage == "full" and not include_full:
                continue
            if stage not in stages and not (stage == "full" and include_full):
                continue
            deps = []
            if "prepare" in stages:
                deps.append("pdc_prepare_full_bundle")
            if "env" in stages:
                deps.append(env_id)
            if prior_stage_id:
                deps.append(prior_stage_id)

            chunk_ids: list[int | None]
            num_chunks: int | None
            if method == "commot" and stage == "full" and int(commot_chunks) > 1:
                chunk_ids = list(range(int(commot_chunks)))
                num_chunks = int(commot_chunks)
            elif method == "cellagentchat" and stage == "full" and int(cellagentchat_chunks) > 1:
                chunk_ids = list(range(int(cellagentchat_chunks)))
                num_chunks = int(cellagentchat_chunks)
            else:
                chunk_ids = [None]
                num_chunks = None

            stage_run_ids: list[str] = []
            for chunk_id in chunk_ids:
                chunk_suffix = "" if chunk_id is None else f"_chunk_{int(chunk_id):03d}_of_{int(num_chunks or 0):03d}"
                run_id = f"pdc_{stage}_common_{method}{chunk_suffix}"
                stage_run_ids.append(run_id)
                jobs.append(
                    {
                        "job_id": run_id,
                        "job_type": f"{stage}_common",
                        "method": method,
                        "chunk_id": chunk_id,
                        "num_chunks": num_chunks,
                        "partition": cfg.partition,
                        "cpus": cfg.cpus,
                        "memory": cfg.memory,
                        "time": cfg.time,
                        "dependencies": deps,
                        "script": job_script(
                            job_id=run_id,
                            root=root,
                            account=account,
                            partition=cfg.partition,
                            cpus=cfg.cpus,
                            memory=cfg.memory,
                            time_limit=cfg.time,
                            body=method_run_command(method, cfg, root, stage, chunk_id=chunk_id, num_chunks=num_chunks),
                        ),
                    }
                )
            if method == "cellagentchat" and stage == "full" and num_chunks is not None and len(stage_run_ids) > 1:
                aggregate_id = f"pdc_{stage}_common_{method}_aggregate"
                jobs.append(
                    {
                        "job_id": aggregate_id,
                        "job_type": f"{stage}_common_aggregate",
                        "method": method,
                        "chunk_id": None,
                        "num_chunks": num_chunks,
                        "partition": "shared",
                        "cpus": 4,
                        "memory": "32G",
                        "time": "01:00:00",
                        "dependencies": stage_run_ids,
                        "script": job_script(
                            job_id=aggregate_id,
                            root=root,
                            account=account,
                            partition="shared",
                            cpus=4,
                            memory="32G",
                            time_limit="01:00:00",
                            body=aggregate_chunk_command(method, cfg, root, stage, num_chunks=num_chunks),
                        ),
                    }
                )
                prior_stage_id = aggregate_id
            else:
                prior_stage_id = stage_run_ids[0] if len(stage_run_ids) == 1 else None
    return jobs


def write_remote_text(host: str, remote_path: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".sbatch", delete=False) as handle:
        handle.write(text)
        tmp_name = handle.name
    try:
        ssh(host, f"mkdir -p {q(posixpath.dirname(remote_path))}")
        run_command(["scp", tmp_name, f"{host}:{remote_path}"])
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def dependency_clause(job: dict[str, Any], slurm_ids: dict[str, str]) -> str:
    dep_ids: dict[str, list[str]] = {"afterok": [], "afterany": []}
    for dep in job.get("dependencies", []):
        if dep not in slurm_ids:
            continue
        mode = "afterok"
        dep_ids[mode].append(slurm_ids[dep])
    clauses = [f"{mode}:{':'.join(ids)}" for mode, ids in dep_ids.items() if ids]
    return ",".join(clauses)


def ensure_remote_sources(host: str, root: str, methods: list[str]) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    for method in methods:
        if method not in SOURCE_CHECKOUTS:
            continue
        command = f"set -euo pipefail\nROOT={q(root)}\n" + source_checkout_command(method, root)
        completed = ssh(host, "bash -lc " + q(command), check=False)
        staged.append(
            {
                "method": method,
                "status": "staged" if completed.returncode == 0 else "stage_failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to stage source checkout for {method} on {host}:{root}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
    return staged


def submit_jobs(host: str, root: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    submitted: list[dict[str, Any]] = []
    slurm_ids: dict[str, str] = {}
    script_paths: dict[str, str] = {}
    source_methods = sorted({str(job.get("method")) for job in jobs if str(job.get("method")) in SOURCE_CHECKOUTS})
    ensure_remote_sources(host, root, source_methods)
    for job in jobs:
        script_path = f"{root}/slurm/{job['job_id']}.sbatch"
        write_remote_text(host, script_path, str(job["script"]))
        script_paths[str(job["job_id"])] = script_path
    for job in jobs:
        script_path = script_paths[str(job["job_id"])]
        missing_deps = [dep for dep in job.get("dependencies", []) if dep not in slurm_ids]
        if missing_deps:
            submitted.append(
                {k: v for k, v in job.items() if k != "script"}
                | {"script_path": script_path, "status": "skipped_missing_dependency", "missing_dependencies": missing_deps}
            )
            continue
        command = f"sbatch --parsable"
        deps = dependency_clause(job, slurm_ids)
        if deps:
            command += " --dependency=" + deps
        command += f" {q(script_path)}"
        completed = ssh(host, command, check=False)
        if completed.returncode != 0:
            submitted.append(
                {k: v for k, v in job.items() if k != "script"}
                | {
                    "script_path": script_path,
                    "status": "submit_failed",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            continue
        slurm_id = completed.stdout.strip().splitlines()[-1]
        slurm_ids[str(job["job_id"])] = slurm_id
        submitted.append(
            {k: v for k, v in job.items() if k != "script"}
            | {"script_path": script_path, "status": "submitted", "slurm_job_id": slurm_id}
        )
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and submit a PDC Slurm matrix for CCI benchmarking.")
    parser.add_argument("--host", default=DEFAULT_PDC_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--methods", default=",".join(PDC_REBUILD_FULL_METHODS))
    parser.add_argument("--stages", default="prepare,env")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--commot-chunks", type=int, default=16)
    parser.add_argument("--cellagentchat-chunks", type=int, default=CELLAGENTCHAT_FULL_CHUNKS)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    stages = {item.strip().lower() for item in args.stages.split(",") if item.strip()}
    jobs = build_jobs(
        methods,
        args.remote_root.rstrip("/"),
        args.account,
        stages,
        args.include_full,
        commot_chunks=args.commot_chunks,
        cellagentchat_chunks=args.cellagentchat_chunks,
    )
    manifest: dict[str, Any] = {
        "kind": "pdc_job_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "remote_root": args.remote_root.rstrip("/"),
        "account": args.account,
        "methods": methods,
        "stages": sorted(stages),
        "include_full": args.include_full,
        "rebuild_env_tag": REBUILD_ENV_TAG,
        "rebuild_full_methods": list(PDC_REBUILD_FULL_METHODS),
        "source_checkouts": SOURCE_CHECKOUTS,
        "a100_authoritative_full_methods": list(A100_AUTHORITATIVE_FULL_METHODS),
        "pdc_full_backfill_methods": list(PDC_FULL_BACKFILL_METHODS),
        "commot_chunks": int(args.commot_chunks),
        "cellagentchat_chunks": int(args.cellagentchat_chunks),
        "jobs": [{k: v for k, v in job.items() if k != "script"} for job in jobs],
    }
    if args.submit:
        submitted = submit_jobs(args.host, args.remote_root.rstrip("/"), jobs)
        manifest["submitted_jobs"] = submitted
        manifest["jobs"] = submitted
        remote_manifest = f"{args.remote_root.rstrip('/')}/logs/pdc_job_matrix_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
            tmp_name = handle.name
        try:
            run_command(["scp", tmp_name, f"{args.host}:{remote_manifest}"])
            manifest["remote_manifest"] = remote_manifest
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
