from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .cci_atera import ATERA_BENCHMARK_RELATIVE_ROOT, BenchmarkLayout, load_method_registry, resolve_layout
from .cci_adapters import SUPPORTED_REAL_ADAPTERS, load_input_manifest, validate_input_manifest


DEFAULT_A100_REMOTE_ROOT = "/data/taobo.hu/pyxenium_cci_benchmark_2026-04"
DEFAULT_A100_READONLY_XENIUM_ROOT = "/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs"
DEFAULT_A100_METHOD_ORDER = ("pyxenium", "squidpy", "liana", "commot", "cellchat")
DEFAULT_A100_ALL_METHODS = (
    "pyxenium",
    "squidpy",
    "liana",
    "cellchat",
    "commot",
    "spatialdm",
    "stlearn",
    "giotto",
    "laris",
    "cellphonedb",
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
PDC_FULL_BACKFILL_METHODS = tuple(method for method in DEFAULT_A100_ALL_METHODS if method not in A100_AUTHORITATIVE_FULL_METHODS)
A100_GPU_METHODS = {"cellnest", "cellagentchat", "scild"}
A100_SOURCE_CHECKOUTS = {
    "cellnest": {
        "repo": "https://github.com/schwartzlab-methods/CellNEST.git",
        "commit": "2737fa8f54952b4b35a540f6070655a69f2c4999",
    },
    "cellagentchat": {
        "repo": "https://github.com/mcgilldinglab/CellAgentChat.git",
        "commit": "37e51980cb9ba87684993d8bdae26feac8806bae",
    },
    "scild": {
        "repo": "https://github.com/jiatingyu-amss/SCILD.git",
        "commit": "683515043df1878f3069c4dd5f887abb5c8976bd",
    },
}
A100_R_HEAVY_METHODS = {"cellchat", "spatalk", "niches", "giotto"}
A100_PYTHON_AUDIT_MODULES = {
    "commot": ("commot",),
    "liana": ("liana",),
    "squidpy": ("squidpy",),
    "spatialdm": ("spatialdm",),
    "stlearn": ("stlearn",),
    "cellphonedb": ("cellphonedb",),
    "laris": ("laris",),
    "cellnest": ("cellnest",),
    "cellagentchat": ("CellAgentChat",),
    "scild": ("scild",),
}
DEFAULT_A100_WRITABLE_SUBDIRS = (
    "repo",
    "configs",
    "envs",
    "external_src",
    "scripts",
    "runners",
    "data",
    "data/full",
    "data/smoke",
    "data/repeats",
    "runs",
    "results",
    "logs",
    "reports",
    "tmp",
)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unreadable", "error": f"Could not parse {path}"}


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def _default_transfer_mode() -> str:
    if shutil.which("rsync"):
        return "rsync"
    if platform.system().lower().startswith("win"):
        return "tar-scp"
    return "scp"


def _normalize_transfer_mode(transfer_mode: str | None) -> str:
    mode = str(transfer_mode or "auto").strip().lower()
    if mode == "auto":
        return _default_transfer_mode()
    allowed = {"rsync", "scp", "tar-scp"}
    if mode not in allowed:
        raise ValueError(f"Unsupported transfer mode {transfer_mode!r}. Expected one of {sorted(allowed)}.")
    return mode


def _repo_root_from_layout(layout: BenchmarkLayout) -> Path:
    for candidate in (layout.root, *layout.root.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from benchmark layout: {layout.root}")


def _remote_path(remote_root: str | Path, *parts: str) -> str:
    root = str(remote_root).replace("\\", "/").rstrip("/")
    suffix = "/".join(part.strip("/").replace("\\", "/") for part in parts if part)
    return f"{root}/{suffix}" if suffix else root


def _q(value: str | Path) -> str:
    return shlex.quote(str(value).replace("\\", "/"))


def _join_command(parts: Sequence[str | Path | int]) -> str:
    return " ".join(_q(str(part)) for part in parts)


def _a100_source_checkout_command(method: str, remote_root: str | Path) -> str:
    checkout = A100_SOURCE_CHECKOUTS.get(str(method).lower())
    if not checkout:
        return ""
    repo = str(checkout["repo"])
    commit = str(checkout["commit"])
    external_root = _remote_path(remote_root, "external_src")
    src = _remote_path(external_root, method)
    ok_message = f"source_checkout_ok {method} already at {commit}"
    return (
        f"mkdir -p {_q(external_root)} && "
        f"if [ ! -d {_q(src)}/.git ]; then rm -rf {_q(src)} && git clone {_q(repo)} {_q(src)}; fi && "
        f'current_commit="$(git -C {_q(src)} rev-parse HEAD 2>/dev/null || true)"; '
        f"if [ \"$current_commit\" = {_q(commit)} ]; then echo {_q(ok_message)}; "
        f"else if ! git -C {_q(src)} cat-file -e {_q(commit + '^{commit}')} 2>/dev/null; "
        f"then git -C {_q(src)} fetch --all --tags --prune; fi; "
        f"git -C {_q(src)} checkout {_q(commit)}; fi"
    )


def _remote_target(host: str | None, user: str | None) -> str:
    return f"{user}@{host}" if host and user else "<user>@<host>"


def _db_run_slug(database_mode: str) -> str:
    aliases = {
        "common": "common",
        "common-db": "common",
        "native": "native",
        "native-db": "native",
        "smoke": "smoke_panel",
        "smoke-panel": "smoke_panel",
    }
    key = str(database_mode).strip().lower()
    return aliases.get(key, _safe_slug(database_mode))


def _run_group(phase: str, database_mode: str, repeat_id: str | None = None) -> str:
    db_slug = _db_run_slug(database_mode)
    if repeat_id:
        return f"repeat_{db_slug}/{_safe_slug(repeat_id)}"
    return f"{_safe_slug(phase)}_{db_slug}"


def _a100_prefix(remote_root: str | Path) -> str:
    tmp_dir = _remote_path(remote_root, "tmp")
    repo_dir = _remote_path(remote_root, "repo")
    repo_src = _remote_path(remote_root, "repo", "src")
    return (
        'export PATH="$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH" && '
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"; fi && '
        "export PYTHONNOUSERSITE=1 && "
        f"export TMPDIR={_q(tmp_dir)} && "
        f"mkdir -p { _q(tmp_dir) } && "
        f"export PYTHONPATH={_q(repo_src)}:${{PYTHONPATH:-}} && "
        f"cd {_q(repo_dir)}"
    )


def _resource_profile(method: str) -> dict[str, Any]:
    method = str(method).lower()
    if method in A100_GPU_METHODS:
        return {"gpu_required": True, "max_parallel_group": "gpu", "omp_threads": 8, "mkl_threads": 8}
    if method in A100_R_HEAVY_METHODS:
        return {"gpu_required": False, "max_parallel_group": "r_heavy", "omp_threads": 8, "mkl_threads": 8}
    if method in {"commot", "spatialdm", "stlearn"}:
        return {"gpu_required": False, "max_parallel_group": "cpu_heavy", "omp_threads": 8, "mkl_threads": 8}
    return {"gpu_required": False, "max_parallel_group": "cpu_light", "omp_threads": 4, "mkl_threads": 4}


def _wrap_a100_job(
    inner_command: str,
    *,
    remote_root: str | Path,
    job_id: str,
    env_name: str | None = None,
    gpu_id: int | str | None = None,
    omp_threads: int = 8,
    mkl_threads: int = 8,
) -> tuple[str, str]:
    resource_log = _remote_path(remote_root, "logs", f"{job_id}.resource.log")
    env_prefix = ""
    if env_name:
        env_prefix = f'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/{env_name}/lib:$HOME/miniconda3/lib:${{LD_LIBRARY_PATH:-}}" && '
    cuda_value = "" if gpu_id is None or str(gpu_id) == "" else str(gpu_id)
    thread_prefix = (
        f"export CUDA_VISIBLE_DEVICES={_q(cuda_value)} && "
        f"export OMP_NUM_THREADS={int(omp_threads)} && "
        f"export MKL_NUM_THREADS={int(mkl_threads)} && "
    )
    command = f"{_a100_prefix(remote_root)} && {env_prefix}{thread_prefix}/usr/bin/time -v -o {_q(resource_log)} {inner_command}"
    return command, resource_log


def _method_registry_by_slug(layout: BenchmarkLayout) -> dict[str, dict[str, Any]]:
    return {str(item["slug"]): dict(item) for item in load_method_registry(layout.config_dir / "methods.yaml")}


def _prepare_full_bundle_command(
    *,
    remote_root: str | Path,
    remote_xenium_root: str | Path,
    smoke_n_cells: int,
    seed: int,
    prefer: str,
    job_id: str = "prepare_full_bundle",
) -> tuple[str, str]:
    parts = [
        "conda",
        "run",
        "--name",
        "pyx-cci-prep",
        "python",
        _remote_path(remote_root, "scripts", "prepare_data.py"),
        "--xenium-root",
        str(remote_xenium_root),
        "--benchmark-root",
        _remote_path(remote_root),
        "--smoke-n-cells",
        str(smoke_n_cells),
        "--seed",
        str(seed),
        "--prefer",
        prefer,
        "--skip-full-h5ad",
        "--output-json",
        _remote_path(remote_root, "logs", "prepare_full_bundle.json"),
    ]
    return _wrap_a100_job(
        _join_command(parts),
        remote_root=remote_root,
        job_id=job_id,
        env_name="pyx-cci-prep",
    )


def _method_run_command(
    *,
    method: str,
    method_info: Mapping[str, Any],
    remote_root: str | Path,
    phase: str,
    database_mode: str,
    run_group: str,
    max_cci_pairs: int | None,
    n_perms: int,
    input_manifest: str | Path | None = None,
    repeat_id: str | None = None,
    job_id: str,
    chunk_id: int | None = None,
    num_chunks: int | None = None,
    bounded_mode: str | None = None,
    gpu_id: int | str | None = None,
    gzip_standardized: bool = False,
) -> tuple[str, str]:
    env_name = str(method_info.get("env_name", ""))
    if chunk_id is not None and num_chunks is not None:
        output_name = f"{method}/chunk_{int(chunk_id):03d}_of_{int(num_chunks):03d}"
    elif repeat_id is not None:
        output_name = f"{method}_{repeat_id}"
    else:
        output_name = method
    output_dir = _remote_path(remote_root, "runs", run_group, output_name)
    input_manifest = str(input_manifest) if input_manifest is not None else _remote_path(remote_root, "data", "input_manifest.json")
    python_contract_runner = method_info.get("language") != "r"
    if method_info.get("language") == "r":
        runner_name = "run_cellchat.R" if method == "cellchat" else "run_external_cci_method.R"
        parts = [
            "conda",
            "run",
            "--name",
            env_name,
            "Rscript",
            _remote_path(remote_root, "runners", "r", runner_name),
            "--method",
            method,
            "--input-manifest",
            input_manifest,
            "--output-dir",
            output_dir,
            "--database-mode",
            database_mode,
            "--phase",
            phase,
        ]
    else:
        parts = [
            "conda",
            "run",
            "--name",
            env_name,
            "python",
            _remote_path(remote_root, "scripts", "run_method.py"),
            "--method",
            method,
            "--benchmark-root",
            _remote_path(remote_root),
            "--input-manifest",
            input_manifest,
            "--output-dir",
            output_dir,
            "--database-mode",
            database_mode,
            "--phase",
            phase,
            "--n-perms",
            str(n_perms),
        ]
    if max_cci_pairs is not None:
        parts.extend(["--max-cci-pairs", str(max_cci_pairs)])
    if python_contract_runner:
        if chunk_id is not None:
            parts.extend(["--chunk-id", str(chunk_id)])
        if num_chunks is not None:
            parts.extend(["--num-chunks", str(num_chunks)])
        if bounded_mode:
            parts.extend(["--bounded-mode", str(bounded_mode)])
        if gpu_id is not None and str(gpu_id) != "":
            parts.extend(["--gpu-id", str(gpu_id)])
        parts.extend(["--job-id", str(job_id)])
        if gzip_standardized:
            parts.append("--gzip-standardized")
    inner_command = _join_command(parts)
    if method == "cellagentchat":
        inner_command = (
            'export CELLAGENTCHAT_FEATURE_SELECTION="${CELLAGENTCHAT_FEATURE_SELECTION:-0}" && '
            'export CELLAGENTCHAT_EPOCHS="${CELLAGENTCHAT_EPOCHS:-10}" && '
            'export CELLAGENTCHAT_MAX_STEPS="${CELLAGENTCHAT_MAX_STEPS:-1}" && '
            f"{inner_command}"
        )
    source_setup = _a100_source_checkout_command(method, remote_root)
    if source_setup:
        inner_command = f"{source_setup} && {inner_command}"
    profile = _resource_profile(method)
    return _wrap_a100_job(
        inner_command,
        remote_root=remote_root,
        job_id=job_id,
        env_name=env_name,
        gpu_id=gpu_id if profile["gpu_required"] else None,
        omp_threads=int(profile["omp_threads"]),
        mkl_threads=int(profile["mkl_threads"]),
    )


def _cellagentchat_aggregate_command(
    *,
    remote_root: str | Path,
    run_group: str,
    database_mode: str,
    num_chunks: int,
    job_id: str,
) -> tuple[str, str]:
    output_dir = _remote_path(remote_root, "runs", run_group, "cellagentchat")
    python_code = f"""
import json
from pathlib import Path

import pandas as pd

from pyXenium.benchmarking.cci_atera import STANDARDIZED_RESULT_COLUMNS, _resolve_ordered_rank

root = Path({output_dir!r})
num_chunks = {int(num_chunks)}
paths = []
missing = []
for idx in range(num_chunks):
    chunk_dir = root / f"chunk_{{idx:03d}}_of_{{num_chunks:03d}}"
    candidates = [
        chunk_dir / "cellagentchat_standardized.tsv.gz",
        chunk_dir / "cellagentchat_standardized.tsv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        missing.append(str(chunk_dir))
    else:
        paths.append(path)
if missing:
    raise SystemExit("missing cellagentchat chunk standardized outputs: " + ", ".join(missing))

frames = []
extra_columns = []
for path in paths:
    table = pd.read_csv(path, sep=chr(9), compression="infer")
    missing_columns = [column for column in STANDARDIZED_RESULT_COLUMNS if column not in table.columns]
    if missing_columns:
        raise SystemExit(f"standardized chunk {{path}} is missing columns: {{missing_columns}}")
    for column in table.columns:
        if column not in STANDARDIZED_RESULT_COLUMNS and column not in extra_columns:
            extra_columns.append(column)
    frames.append(table.copy())
if not frames:
    raise SystemExit("no cellagentchat chunk tables to aggregate")

ordered_columns = STANDARDIZED_RESULT_COLUMNS + extra_columns
combined = pd.concat(frames, ignore_index=True).reindex(columns=ordered_columns)
if combined.empty:
    raise SystemExit("cellagentchat chunk aggregation produced an empty table")
scores = pd.to_numeric(combined["score_raw"], errors="coerce")
fill = scores.min(skipna=True) if scores.notna().any() else 0.0
scores = scores.fillna(fill)
pvalues = pd.to_numeric(combined["fdr_or_pvalue"], errors="coerce")
rank, rank_fraction = _resolve_ordered_rank(scores, pvalues)
combined["rank_within_method"] = rank.astype(float)
combined["rank_fraction"] = rank_fraction.astype(float)
combined["score_std"] = rank_fraction.astype(float)
combined = combined.loc[:, ordered_columns].sort_values("rank_within_method").reset_index(drop=True)
root.mkdir(parents=True, exist_ok=True)
output = root / "cellagentchat_standardized.tsv.gz"
combined.to_csv(output, sep=chr(9), index=False, compression="gzip")
summary = {{
    "status": "success",
    "method": "cellagentchat",
    "phase": "full",
    "database_mode": {database_mode!r},
    "num_chunks": num_chunks,
    "chunk_standardized_tsvs": [str(path) for path in paths],
    "standardized_tsv": str(output),
    "n_rows": int(len(combined)),
}}
(root / "run_summary.json").write_text(json.dumps(summary, indent=2) + chr(10), encoding="utf-8")
print(json.dumps(summary, indent=2))
""".strip()
    inner = _join_command(["conda", "run", "--name", "pyx-cci-cellagentchat", "python", "-c", python_code])
    return _wrap_a100_job(
        inner,
        remote_root=remote_root,
        job_id=job_id,
        env_name="pyx-cci-cellagentchat",
        omp_threads=4,
        mkl_threads=4,
    )


def _audit_command(method: str, method_info: Mapping[str, Any], remote_root: str | Path, *, job_id: str) -> tuple[str, str]:
    env_name = str(method_info.get("env_name", ""))
    if method_info.get("language") == "r":
        project_dir = _remote_path(remote_root, "envs", f"{env_name}_project")
        prefix = (
            f"project <- '{project_dir}'; "
            "if (dir.exists(project)) { setwd(project); "
            "if (requireNamespace('renv', quietly=TRUE)) renv::load(project=project, quiet=TRUE) }; "
        )
        expr = prefix + "cat(R.version.string, intToUtf8(10)); cat('R ok', intToUtf8(10))"
        if method == "cellchat":
            expr = prefix + "cat(R.version.string, intToUtf8(10)); if (!requireNamespace('CellChat', quietly=TRUE)) stop('CellChat missing')"
        elif method == "giotto":
            expr = prefix + "cat(R.version.string, intToUtf8(10)); ok <- requireNamespace('Giotto', quietly=TRUE) || requireNamespace('GiottoClass', quietly=TRUE); if (!ok) stop('Giotto missing')"
        elif method == "spatalk":
            expr = prefix + "cat(R.version.string, intToUtf8(10)); if (!requireNamespace('SpaTalk', quietly=TRUE)) stop('SpaTalk missing')"
        elif method == "niches":
            expr = prefix + "cat(R.version.string, intToUtf8(10)); if (!requireNamespace('NICHES', quietly=TRUE)) stop('NICHES missing')"
        inner = _join_command(["conda", "run", "--name", env_name, "Rscript", "-e", expr])
        return _wrap_a100_job(
            inner,
            remote_root=remote_root,
            job_id=job_id,
            env_name=env_name,
        )
    modules = list(A100_PYTHON_AUDIT_MODULES.get(method, ()))
    py_expr = (
        "import importlib.metadata as m, importlib.util; "
        "import pyXenium; "
        "print('pyXenium ok'); "
        "print('pandas', m.version('pandas')); "
        f"mods={modules!r}; "
        "missing=[name for name in mods if importlib.util.find_spec(name) is None]; "
        "[print('module', name, 'ok') for name in mods if name not in missing]; "
        "assert not missing, 'Missing modules: ' + ','.join(missing)"
    )
    inner = _join_command(["conda", "run", "--name", env_name, "python", "-c", py_expr])
    return _wrap_a100_job(
        inner,
        remote_root=remote_root,
        job_id=job_id,
        env_name=env_name,
    )


def _bootstrap_env_command(method: str, method_info: Mapping[str, Any], remote_root: str | Path, *, job_id: str) -> tuple[str, str]:
    env_name = str(method_info.get("env_name", ""))
    env_file = _remote_path(remote_root, str(method_info.get("env_file", "")).lstrip("/"))
    bootstrap_script = _remote_path(remote_root, "envs", "bootstrap_env.py")
    repo_root = _remote_path(remote_root, "repo")
    benchmark_root = str(remote_root)
    env_exists = f"conda env list | awk '{{print $1}}' | grep -qx {_q(env_name)}"
    create_or_update = (
        f"({env_exists} && conda env update --name {_q(env_name)} --file {_q(env_file)} --prune "
        f"|| conda env create --name {_q(env_name)} --file {_q(env_file)})"
    )
    bootstrap = _join_command(
        [
            "conda",
            "run",
            "--name",
            env_name,
            "python",
            bootstrap_script,
            "--method",
            method,
            "--repo-root",
            repo_root,
            "--benchmark-root",
            benchmark_root,
        ]
    )
    inner = _join_command(["bash", "-lc", f"{create_or_update} && {bootstrap}"])
    return _wrap_a100_job(
        inner,
        remote_root=remote_root,
        job_id=job_id,
        env_name=env_name,
    )


def validate_a100_path_policy(
    payload: Mapping[str, Any],
    *,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    readonly_xenium_root: str | Path | None = DEFAULT_A100_READONLY_XENIUM_ROOT,
) -> dict[str, Any]:
    remote = str(remote_root).replace("\\", "/").rstrip("/")
    readonly = str(readonly_xenium_root).replace("\\", "/").rstrip("/") if readonly_xenium_root else None
    output_fields = {"destination", "output_dir", "stdout", "stderr", "resource_log"}
    input_fields = {"input_root", "readonly_xenium_root", "remote_xenium_root"}
    violations: list[str] = []
    readonly_inputs: list[str] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str):
            normalized = value.replace("\\", "/")
            if readonly and normalized.startswith(readonly):
                if key in input_fields:
                    readonly_inputs.append(normalized)
                elif key in output_fields:
                    violations.append(f"write target {key} points into read-only Xenium root: {normalized}")
            if key in output_fields and normalized.startswith("/") and not normalized.startswith(remote):
                violations.append(f"write target {key} is outside writable remote root: {normalized}")

    visit(payload)
    return {
        "valid": not violations,
        "violations": violations,
        "readonly_inputs": sorted(set(readonly_inputs)),
        "writable_remote_root": remote,
        "readonly_xenium_root": readonly,
    }


def _stage_archive_entries(local_items: Sequence[Mapping[str, str]], remote_root: str | Path) -> list[dict[str, str]]:
    remote_base = str(remote_root).replace("\\", "/").rstrip("/") + "/"
    entries: list[dict[str, str]] = []
    for item in local_items:
        destination = str(item["destination"]).replace("\\", "/")
        if not destination.startswith(remote_base):
            raise ValueError(f"Stage destination {destination!r} is not rooted under {remote_base!r}.")
        entries.append(
            {
                "source": str(item["source"]),
                "destination": destination,
                "arcname": destination[len(remote_base):].lstrip("/"),
            }
        )
    return entries


def _build_stage_commands(
    *,
    local_items: Sequence[Mapping[str, str]],
    remote_root: str | Path,
    host: str | None,
    user: str | None,
    transfer_mode: str,
    local_archive: str | Path | None = None,
) -> dict[str, Any]:
    remote_target = _remote_target(host, user)
    mode = _normalize_transfer_mode(transfer_mode)
    if mode == "rsync":
        commands = []
        for item in local_items:
            source = Path(item["source"])
            if source.is_dir():
                commands.append(f'rsync -a "{source}/" {remote_target}:"{str(item["destination"]).rstrip("/")}/"')
            else:
                commands.append(f'rsync -a "{source}" {remote_target}:"{item["destination"]}"')
        return {"transfer_mode": mode, "copy_commands": commands}

    if mode == "scp":
        commands = []
        for item in local_items:
            source = Path(item["source"])
            recursive = "-r " if source.is_dir() else ""
            commands.append(f'scp {recursive}"{source}" {remote_target}:"{item["destination"]}"')
        return {"transfer_mode": mode, "copy_commands": commands}

    archive_entries = _stage_archive_entries(local_items, remote_root)
    archive_path = Path(local_archive) if local_archive else Path(tempfile.gettempdir()) / "pyxenium_a100_stage.tar.gz"
    remote_archive = _remote_path(remote_root, "logs", archive_path.name)
    pack_note = f"Create archive {archive_path} with {len(archive_entries)} staged entries rooted under {remote_root}."
    commands = [
        f"# {pack_note}",
        f'scp "{archive_path}" {remote_target}:"{remote_archive}"',
        f'ssh {remote_target} "tar -xzf {remote_archive} -C {str(remote_root).rstrip("/")} && rm -f {remote_archive}"',
    ]
    return {
        "transfer_mode": mode,
        "copy_commands": commands,
        "local_archive": str(archive_path),
        "remote_archive": remote_archive,
        "archive_entries": archive_entries,
    }


def build_a100_stage_plan(
    *,
    benchmark_root: str | Path | None = None,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    remote_xenium_root: str | Path | None = DEFAULT_A100_READONLY_XENIUM_ROOT,
    stage_data: bool | None = None,
    transfer_mode: str | None = None,
    host: str | None = None,
    user: str | None = None,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    repo_root = _repo_root_from_layout(layout)
    should_stage_data = (remote_xenium_root is None) if stage_data is None else bool(stage_data)
    resolved_transfer_mode = _normalize_transfer_mode(transfer_mode)
    local_items = [
        {"source": str(repo_root / "pyproject.toml"), "destination": _remote_path(remote_root, "repo", "pyproject.toml")},
        {"source": str(repo_root / "README.md"), "destination": _remote_path(remote_root, "repo", "README.md")},
        {"source": str(repo_root / "LICENSE"), "destination": _remote_path(remote_root, "repo", "LICENSE")},
        {"source": str(repo_root / "src"), "destination": _remote_path(remote_root, "repo", "src")},
        {"source": str(layout.config_dir), "destination": _remote_path(remote_root, "configs")},
        {"source": str(layout.env_dir), "destination": _remote_path(remote_root, "envs")},
        {"source": str(layout.scripts_dir), "destination": _remote_path(remote_root, "scripts")},
        {"source": str(layout.runners_dir), "destination": _remote_path(remote_root, "runners")},
    ]
    if should_stage_data:
        local_items.append({"source": str(layout.data_dir), "destination": _remote_path(remote_root, "data")})
    existing_items = [item for item in local_items if Path(item["source"]).exists()]
    remote_target = _remote_target(host, user)
    mkdir_paths = " ".join(_q(_remote_path(remote_root, subdir)) for subdir in DEFAULT_A100_WRITABLE_SUBDIRS)
    mkdir_command = (
        f"ssh {remote_target} \"mkdir -p {mkdir_paths}\""
    )
    stage_commands = _build_stage_commands(
        local_items=existing_items,
        remote_root=remote_root,
        host=host,
        user=user,
        transfer_mode=resolved_transfer_mode,
        local_archive=layout.logs_dir / "a100_stage_payload.tar.gz",
    )
    payload = {
        "kind": "a100_stage_plan",
        "local_benchmark_root": str(layout.root),
        "local_repo_root": str(repo_root),
        "remote_root": str(remote_root),
        "remote_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
        "readonly_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
        "writable_benchmark_root": str(remote_root),
        "stage_data": should_stage_data,
        "transfer_mode": resolved_transfer_mode,
        "writable_subdirs": [_remote_path(remote_root, subdir) for subdir in DEFAULT_A100_WRITABLE_SUBDIRS],
        "host": host,
        "user": user,
        "requires_host": not bool(host and user),
        "mkdir_command": mkdir_command,
        "copy_commands": stage_commands["copy_commands"],
        "local_items": existing_items,
    }
    for key in ("local_archive", "remote_archive", "archive_entries"):
        if key in stage_commands:
            payload[key] = stage_commands[key]
    payload["path_policy"] = validate_a100_path_policy(payload, remote_root=remote_root, readonly_xenium_root=remote_xenium_root)
    return payload


def build_a100_job_manifest(
    *,
    benchmark_root: str | Path | None = None,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    remote_xenium_root: str | Path | None = DEFAULT_A100_READONLY_XENIUM_ROOT,
    methods: Sequence[str] = DEFAULT_A100_METHOD_ORDER,
    database_mode: str = "common-db",
    phase: str = "full",
    max_cci_pairs: int | None = None,
    n_perms: int = 100,
    include_audit: bool = True,
    include_prepare: bool = True,
    smoke_n_cells: int = 20_000,
    seed: int = 0,
    prefer: str = "h5",
    repeat_id: str | None = None,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    registry = _method_registry_by_slug(layout)
    run_group = _run_group(phase, database_mode, repeat_id=repeat_id)
    jobs: list[dict[str, Any]] = []
    if include_prepare and remote_xenium_root is not None and phase == "full" and repeat_id is None:
        prepare_id = "prepare_full_bundle"
        prepare_command, prepare_resource_log = _prepare_full_bundle_command(
            remote_root=remote_root,
            remote_xenium_root=remote_xenium_root,
            smoke_n_cells=smoke_n_cells,
            seed=seed,
            prefer=prefer,
            job_id=prepare_id,
        )
        jobs.append(
            {
                "job_id": prepare_id,
                "job_type": "data_prepare",
                "method": "prep",
                "phase": "prepare",
                "database_mode": database_mode,
                "env_name": "pyx-cci-prep",
                "command": prepare_command,
                "input_root": str(remote_xenium_root),
                "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                "output_dir": _remote_path(remote_root, "data"),
                "stdout": _remote_path(remote_root, "logs", f"{prepare_id}.stdout.log"),
                "stderr": _remote_path(remote_root, "logs", f"{prepare_id}.stderr.log"),
                "resource_log": prepare_resource_log,
                "readonly_xenium_root": str(remote_xenium_root),
                "writable_benchmark_root": str(remote_root),
                "status": "planned",
                "runtime_seconds": None,
                "peak_memory_gb": None,
                "exit_code": None,
            }
        )
    for method in methods:
        method = str(method).strip().lower()
        if not method:
            continue
        if method not in registry:
            raise ValueError(f"Method {method!r} is not registered in methods.yaml.")
        method_info = registry[method]
        if include_audit:
            audit_id = f"audit_{method}"
            audit_command, audit_resource_log = _audit_command(method, method_info, remote_root, job_id=audit_id)
            jobs.append(
                {
                    "job_id": audit_id,
                    "job_type": "env_audit",
                    "method": method,
                    "phase": "audit",
                    "database_mode": database_mode,
                    "env_name": method_info.get("env_name"),
                    "command": audit_command,
                    "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                    "output_dir": _remote_path(remote_root, "logs", audit_id),
                    "stdout": _remote_path(remote_root, "logs", f"{audit_id}.stdout.log"),
                    "stderr": _remote_path(remote_root, "logs", f"{audit_id}.stderr.log"),
                    "resource_log": audit_resource_log,
                    "readonly_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
                    "writable_benchmark_root": str(remote_root),
                    "status": "planned",
                    "runtime_seconds": None,
                    "peak_memory_gb": None,
                    "exit_code": None,
                }
            )
        job_id = f"{phase}_{_safe_slug(database_mode)}_{method}" if repeat_id is None else f"{phase}_{_safe_slug(database_mode)}_{repeat_id}_{method}"
        output_dir = _remote_path(remote_root, "runs", run_group, method if repeat_id is None else f"{method}_{repeat_id}")
        command, resource_log = _method_run_command(
            method=method,
            method_info=method_info,
            remote_root=remote_root,
            phase=phase,
            database_mode=database_mode,
            run_group=run_group,
            max_cci_pairs=max_cci_pairs,
            n_perms=n_perms,
            repeat_id=repeat_id,
            job_id=job_id,
        )
        jobs.append(
            {
                "job_id": job_id,
                "job_type": "method_run",
                "method": method,
                "phase": phase,
                "database_mode": database_mode,
                "repeat_id": repeat_id,
                "env_name": method_info.get("env_name"),
                "command": command,
                "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                "output_dir": output_dir,
                "stdout": _remote_path(remote_root, "logs", f"{job_id}.stdout.log"),
                "stderr": _remote_path(remote_root, "logs", f"{job_id}.stderr.log"),
                "resource_log": resource_log,
                "readonly_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
                "writable_benchmark_root": str(remote_root),
                "status": "planned",
                "runtime_seconds": None,
                "peak_memory_gb": None,
                "exit_code": None,
            }
        )
    payload = {
        "kind": "a100_job_manifest",
        "remote_root": str(remote_root),
        "remote_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
        "readonly_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
        "writable_benchmark_root": str(remote_root),
        "phase": phase,
        "database_mode": database_mode,
        "methods": list(methods),
        "max_cci_pairs": max_cci_pairs,
        "n_perms": n_perms,
        "run_group": run_group,
        "resource_limits": {"max_memory_gb": 64, "max_runtime_seconds": 6 * 60 * 60},
        "jobs": jobs,
    }
    payload["path_policy"] = validate_a100_path_policy(payload, remote_root=remote_root, readonly_xenium_root=remote_xenium_root)
    return payload


def prepare_a100_bundle(
    *,
    benchmark_root: str | Path | None = None,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    remote_xenium_root: str | Path | None = DEFAULT_A100_READONLY_XENIUM_ROOT,
    transfer_mode: str | None = None,
    methods: Sequence[str] = DEFAULT_A100_METHOD_ORDER,
    database_mode: str = "common-db",
    phase: str = "full",
    max_cci_pairs: int | None = None,
    n_perms: int = 100,
    require_full: bool = True,
    include_prepare: bool = True,
    stage_data: bool | None = None,
    smoke_n_cells: int = 20_000,
    seed: int = 0,
    prefer: str = "h5",
    host: str | None = None,
    user: str | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    manifest_path = layout.data_dir / "input_manifest.json"
    manifest = load_input_manifest(manifest_path)
    validation = validate_input_manifest(manifest, require_full=require_full)
    stage_plan = build_a100_stage_plan(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        remote_xenium_root=remote_xenium_root,
        stage_data=stage_data,
        transfer_mode=transfer_mode,
        host=host,
        user=user,
    )
    job_manifest = build_a100_job_manifest(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        remote_xenium_root=remote_xenium_root,
        methods=methods,
        database_mode=database_mode,
        phase=phase,
        max_cci_pairs=max_cci_pairs,
        n_perms=n_perms,
        include_prepare=include_prepare,
        smoke_n_cells=smoke_n_cells,
        seed=seed,
        prefer=prefer,
    )
    can_prepare_remotely = bool(remote_xenium_root and include_prepare and phase == "full")
    status = "ready" if validation["valid"] else "blocked"
    if not validation["valid"] and can_prepare_remotely:
        status = "ready_after_remote_prepare"
    payload = {
        "status": status,
        "validation": validation,
        "input_manifest": str(manifest_path),
        "remote_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
        "readonly_xenium_root": str(remote_xenium_root) if remote_xenium_root is not None else None,
        "writable_benchmark_root": str(remote_root),
        "remote_prepare_required": bool(not validation["valid"] and can_prepare_remotely),
        "stage_plan": stage_plan,
        "job_manifest": job_manifest,
    }
    payload["path_policy"] = validate_a100_path_policy(payload, remote_root=remote_root, readonly_xenium_root=remote_xenium_root)
    default_path = layout.logs_dir / "a100_bundle_plan.json"
    _write_json(output_json or default_path, payload)
    return payload


def execute_a100_stage_plan(
    *,
    stage_plan: Mapping[str, Any] | str | Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    plan = json.loads(Path(stage_plan).read_text(encoding="utf-8")) if isinstance(stage_plan, (str, Path)) else dict(stage_plan)
    transfer_mode = _normalize_transfer_mode(plan.get("transfer_mode"))
    host = plan.get("host")
    user = plan.get("user")
    if not dry_run and not (host and user):
        raise ValueError("host and user are required when executing an A100 stage plan.")

    rows: list[dict[str, Any]] = []
    mkdir_command = str(plan["mkdir_command"])
    if dry_run:
        rows.append({"step": "mkdir", "command": mkdir_command, "status": "dry-run"})
    else:
        completed = subprocess.run(mkdir_command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        rows.append({"step": "mkdir", "command": mkdir_command, "status": "success" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode != 0:
            return {"dry_run": dry_run, "transfer_mode": transfer_mode, "steps": rows}

    if transfer_mode == "tar-scp":
        archive_entries = plan.get("archive_entries", [])
        local_archive = Path(str(plan["local_archive"]))
        remote_archive = str(plan["remote_archive"])
        if dry_run:
            rows.append({"step": "pack", "archive": str(local_archive), "entries": len(archive_entries), "status": "dry-run"})
        else:
            local_archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local_archive, "w:gz") as handle:
                for entry in archive_entries:
                    handle.add(entry["source"], arcname=entry["arcname"])
            rows.append({"step": "pack", "archive": str(local_archive), "entries": len(archive_entries), "status": "success"})

        remote_target = _remote_target(host, user)
        upload_command = f'scp "{local_archive}" {remote_target}:"{remote_archive}"'
        extract_command = f'ssh {remote_target} "tar -xzf {remote_archive} -C {str(plan["remote_root"]).rstrip("/")} && rm -f {remote_archive}"'
        for step, command in (("upload", upload_command), ("extract", extract_command)):
            if dry_run:
                rows.append({"step": step, "command": command, "status": "dry-run"})
                continue
            completed = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            rows.append({"step": step, "command": command, "status": "success" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
            if completed.returncode != 0:
                break
        return {"dry_run": dry_run, "transfer_mode": transfer_mode, "steps": rows}

    for command in plan.get("copy_commands", []):
        if dry_run:
            rows.append({"step": "copy", "command": command, "status": "dry-run"})
            continue
        completed = subprocess.run(str(command), shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        rows.append({"step": "copy", "command": command, "status": "success" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode != 0:
            break
    return {"dry_run": dry_run, "transfer_mode": transfer_mode, "steps": rows}


def run_a100_plan(
    *,
    plan_json: str | Path,
    dry_run: bool = True,
    job_ids: Sequence[str] | None = None,
    remote: bool = False,
    host: str | None = None,
    user: str | None = None,
    ssh_executable: str = "ssh",
) -> dict[str, Any]:
    plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    job_manifest = plan.get("job_manifest", plan)
    selected = set(job_ids or [])
    rows: list[dict[str, Any]] = []
    remote_target = _remote_target(host, user) if remote else None
    if remote and not dry_run and not (host and user):
        raise ValueError("host and user are required when run_a100_plan executes with remote=True.")
    for job in job_manifest.get("jobs", []):
        if selected and job.get("job_id") not in selected:
            continue
        wrapper_command = None
        if remote:
            wrapper_command = f'{ssh_executable} {remote_target} "{job["command"]}"'
        if dry_run:
            row = {**job, "status": "dry-run"}
            if wrapper_command:
                row["wrapper_command"] = wrapper_command
            rows.append(row)
            continue
        started = time.perf_counter()
        if remote:
            completed = subprocess.run([ssh_executable, str(remote_target), str(job["command"])], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        else:
            completed = subprocess.run(str(job["command"]), shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        stdout = Path(str(job.get("stdout", f"{job['job_id']}.stdout.log")))
        stderr = Path(str(job.get("stderr", f"{job['job_id']}.stderr.log")))
        if not remote and not stdout.is_absolute():
            stdout = Path(plan_json).parent / stdout
        if not remote and not stderr.is_absolute():
            stderr = Path(plan_json).parent / stderr
        if not remote:
            stdout.parent.mkdir(parents=True, exist_ok=True)
            stderr.parent.mkdir(parents=True, exist_ok=True)
            stdout.write_text(completed.stdout, encoding="utf-8")
            stderr.write_text(completed.stderr, encoding="utf-8")
        rows.append(
            {
                **job,
                "status": "success" if completed.returncode == 0 else "failed",
                "runtime_seconds": float(time.perf_counter() - started),
                "exit_code": completed.returncode,
                "stdout": str(stdout),
                "stderr": str(stderr),
                "wrapper_command": wrapper_command,
                "remote": remote,
                "host": host,
                "user": user,
            }
        )
    return {"dry_run": dry_run, "remote": remote, "host": host, "user": user, "jobs": rows}


def collect_a100_results(
    *,
    benchmark_root: str | Path | None = None,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    host: str | None = None,
    user: str | None = None,
    transfer_mode: str | None = None,
    dry_run: bool = True,
    since_last: bool = False,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    mode = _normalize_transfer_mode(transfer_mode if transfer_mode is not None else ("scp" if not shutil.which("rsync") else "rsync"))
    remote_target = _remote_target(host, user)
    remote_to_local = [
        (_remote_path(remote_root, "runs"), layout.runs_dir / "a100_collected"),
        (_remote_path(remote_root, "results"), layout.results_dir / "a100_collected"),
        (_remote_path(remote_root, "logs"), layout.logs_dir / "a100_collected"),
        (_remote_path(remote_root, "reports"), layout.reports_dir / "a100_collected"),
    ]
    commands: list[str] = []
    for remote_path_value, local_path in remote_to_local:
        if mode == "rsync":
            commands.append(f'rsync -a {remote_target}:"{remote_path_value}/" "{local_path}"')
        else:
            commands.append(f'scp -r {remote_target}:"{remote_path_value}" "{local_path}"')
    payload = {
        "dry_run": dry_run,
        "since_last": since_last,
        "transfer_mode": mode,
        "remote_root": str(remote_root),
        "host": host,
        "user": user,
        "commands": commands,
        "local_runs_dir": str(layout.runs_dir / "a100_collected"),
        "local_results_dir": str(layout.results_dir / "a100_collected"),
        "local_logs_dir": str(layout.logs_dir / "a100_collected"),
        "local_reports_dir": str(layout.reports_dir / "a100_collected"),
    }
    if dry_run:
        return payload
    if not host or not user:
        raise ValueError("host and user are required when collect_a100_results is not a dry run.")
    completed = []
    for command in commands:
        target_path = None
        if command.endswith('"'):
            try:
                target_path = command.rsplit('"', 2)[1]
            except Exception:
                target_path = None
        if target_path:
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        completed.append({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    payload["completed"] = completed
    return payload


def summarize_run_status(runs_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(Path(runs_dir).glob("**/run_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"summary_json": str(summary_path), "status": "unreadable", "error": str(exc)})
            continue
        params = {}
        params_path = payload.get("params_json")
        if params_path and Path(str(params_path)).exists():
            try:
                params = json.loads(Path(str(params_path)).read_text(encoding="utf-8"))
            except Exception:
                params = {}
        rows.append(
            {
                "method": payload.get("method", params.get("method")),
                "phase": payload.get("phase", params.get("phase")),
                "database_mode": payload.get("database_mode", params.get("database_mode")),
                "status": payload.get("status", "success" if payload.get("standardized_tsv") else "unknown"),
                "n_rows": payload.get("n_rows"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "peak_memory_gb": payload.get("peak_memory_gb"),
                "exit_code": payload.get("exit_code", payload.get("returncode")),
                "standardized_tsv": payload.get("standardized_tsv"),
                "summary_json": str(summary_path),
                "error": payload.get("error", payload.get("reason")),
            }
        )
    return pd.DataFrame(rows)


def build_engineering_summary(run_status: pd.DataFrame) -> pd.DataFrame:
    if run_status.empty:
        return pd.DataFrame()
    table = run_status.copy()
    for column in ("elapsed_seconds", "peak_memory_gb", "n_rows"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.loc[
        :,
        [
            col
            for col in [
                "method",
                "phase",
                "database_mode",
                "status",
                "elapsed_seconds",
                "peak_memory_gb",
                "n_rows",
                "exit_code",
            ]
            if col in table.columns
        ],
    ].sort_values([col for col in ["phase", "database_mode", "method"] if col in table.columns]).reset_index(drop=True)


def build_a100_resource_summary(job_manifest: Mapping[str, Any]) -> pd.DataFrame:
    jobs = job_manifest.get("jobs", [])
    if not jobs:
        return pd.DataFrame()
    return pd.DataFrame(jobs).loc[
        :,
        [
            col
            for col in [
                "job_id",
                "job_type",
                "method",
                "phase",
                "database_mode",
                "status",
                "runtime_seconds",
                "peak_memory_gb",
                "exit_code",
                "resource_log",
            ]
            if col in pd.DataFrame(jobs).columns
        ],
    ]


def write_failed_method_card(output_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method = payload.get("method", "unknown")
    lines = [
        f"# Method Card: {method}",
        "",
        f"- Status: `{payload.get('status', 'failed')}`",
        f"- Phase: `{payload.get('phase', 'unknown')}`",
        f"- Database mode: `{payload.get('database_mode', 'unknown')}`",
        f"- Read-only Xenium root: `{payload.get('readonly_xenium_root', payload.get('remote_xenium_root', 'not recorded'))}`",
        f"- Writable benchmark root: `{payload.get('writable_benchmark_root', 'not recorded')}`",
        f"- Error: `{payload.get('error', payload.get('reason', 'not recorded'))}`",
        f"- Reproduce: `{payload.get('command', payload.get('reproduce', 'See run_summary.json and logs.'))}`",
        "",
    ]
    if payload.get("path_policy"):
        policy = payload["path_policy"]
        lines.extend(
            [
                "## Path Safety",
                "",
                f"- Valid: `{policy.get('valid')}`",
                f"- Violations: `{policy.get('violations', [])}`",
                "",
            ]
        )
    path = output_dir / "method_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_a100_job_matrix(
    *,
    benchmark_root: str | Path | None = None,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    methods: Sequence[str] = DEFAULT_A100_ALL_METHODS,
    database_mode: str = "common-db",
    phase: str = "smoke",
    max_cci_pairs: int | None = None,
    n_perms: int = 100,
    commot_chunks: int = 16,
    cellagentchat_chunks: int = 16,
    gpu_count: int = 8,
    gzip_edge_outputs: bool = True,
    include_bootstrap: bool = False,
    include_audit: bool = False,
    repeat_id: str | None = None,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    registry = _method_registry_by_slug(layout)
    run_group = _run_group(phase, database_mode, repeat_id=repeat_id)
    jobs: list[dict[str, Any]] = []
    gpu_cursor = 0
    bootstrapped_envs: set[str] = set()

    for method_value in methods:
        method = str(method_value).strip().lower()
        if not method:
            continue
        if method not in registry:
            raise ValueError(f"Method {method!r} is not registered in methods.yaml.")
        method_info = registry[method]
        profile = _resource_profile(method)
        env_name = str(method_info.get("env_name", ""))
        if include_bootstrap and env_name and env_name not in bootstrapped_envs:
            bootstrapped_envs.add(env_name)
            bootstrap_id = f"bootstrap_{_safe_slug(env_name)}"
            bootstrap_command, bootstrap_resource_log = _bootstrap_env_command(method, method_info, remote_root, job_id=bootstrap_id)
            jobs.append(
                {
                    "job_id": bootstrap_id,
                    "job_type": "env_bootstrap",
                    "method": method,
                    "phase": "bootstrap",
                    "database_mode": database_mode,
                    "env_name": env_name,
                    "command": bootstrap_command,
                    "output_dir": _remote_path(remote_root, "logs", bootstrap_id),
                    "stdout": _remote_path(remote_root, "logs", f"{bootstrap_id}.stdout.log"),
                    "stderr": _remote_path(remote_root, "logs", f"{bootstrap_id}.stderr.log"),
                    "resource_log": bootstrap_resource_log,
                    "status": "planned",
                    "gpu_required": False,
                    "max_parallel_group": "bootstrap",
                    "omp_threads": 8,
                    "mkl_threads": 8,
                }
            )
        if include_audit:
            audit_id = f"audit_{method}"
            audit_command, audit_resource_log = _audit_command(method, method_info, remote_root, job_id=audit_id)
            jobs.append(
                {
                    "job_id": audit_id,
                    "job_type": "env_audit",
                    "method": method,
                    "phase": "audit",
                    "database_mode": database_mode,
                    "env_name": method_info.get("env_name"),
                    "command": audit_command,
                    "output_dir": _remote_path(remote_root, "logs", audit_id),
                    "stdout": _remote_path(remote_root, "logs", f"{audit_id}.stdout.log"),
                    "stderr": _remote_path(remote_root, "logs", f"{audit_id}.stderr.log"),
                    "resource_log": audit_resource_log,
                    "status": "planned",
                    **profile,
                }
            )

        chunk_ids: list[int | None]
        num_chunks: int | None
        if method == "commot" and phase == "full" and commot_chunks > 1:
            chunk_ids = list(range(int(commot_chunks)))
            num_chunks = int(commot_chunks)
        elif method == "cellagentchat" and phase == "full" and cellagentchat_chunks > 1:
            chunk_ids = list(range(int(cellagentchat_chunks)))
            num_chunks = int(cellagentchat_chunks)
        else:
            chunk_ids = [None]
            num_chunks = None

        for chunk_id in chunk_ids:
            gpu_id = None
            if profile["gpu_required"]:
                gpu_id = gpu_cursor % max(1, int(gpu_count))
                gpu_cursor += 1
            chunk_suffix = "" if chunk_id is None else f"_chunk_{chunk_id:03d}_of_{num_chunks:03d}"
            repeat_suffix = "" if repeat_id is None else f"_{_safe_slug(repeat_id)}"
            job_id = f"{phase}_{_safe_slug(database_mode)}{repeat_suffix}_{method}{chunk_suffix}"
            command, resource_log = _method_run_command(
                method=method,
                method_info=method_info,
                remote_root=remote_root,
                phase=phase,
                database_mode=database_mode,
                run_group=run_group,
                max_cci_pairs=max_cci_pairs,
                n_perms=n_perms,
                repeat_id=repeat_id,
                job_id=job_id,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                bounded_mode=("commot_chunked_full" if method == "commot" and chunk_id is not None else None),
                gpu_id=gpu_id,
                gzip_standardized=gzip_edge_outputs and method in {"commot", "spatialdm", "stlearn", "laris", "cellnest", "cellagentchat", "scild", "niches", "spatalk"},
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "job_type": "method_run",
                    "method": method,
                    "phase": phase,
                    "database_mode": database_mode,
                    "repeat_id": repeat_id,
                    "chunk_id": chunk_id,
                    "num_chunks": num_chunks,
                    "gpu_id": gpu_id,
                    "env_name": method_info.get("env_name"),
                    "command": command,
                    "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                    "output_dir": _remote_path(remote_root, "runs", run_group, method if chunk_id is None else f"{method}/chunk_{chunk_id:03d}_of_{num_chunks:03d}"),
                    "stdout": _remote_path(remote_root, "logs", f"{job_id}.stdout.log"),
                    "stderr": _remote_path(remote_root, "logs", f"{job_id}.stderr.log"),
                    "resource_log": resource_log,
                    "status": "planned",
                    "runtime_seconds": None,
                    "peak_memory_gb": None,
                    "exit_code": None,
                    **profile,
                }
            )
        if method == "cellagentchat" and phase == "full" and num_chunks and num_chunks > 1:
            repeat_suffix = "" if repeat_id is None else f"_{_safe_slug(repeat_id)}"
            aggregate_id = f"{phase}_{_safe_slug(database_mode)}{repeat_suffix}_cellagentchat_aggregate"
            dependencies = [
                f"{phase}_{_safe_slug(database_mode)}{repeat_suffix}_cellagentchat_chunk_{idx:03d}_of_{num_chunks:03d}"
                for idx in range(int(num_chunks))
            ]
            aggregate_command, aggregate_resource_log = _cellagentchat_aggregate_command(
                remote_root=remote_root,
                run_group=run_group,
                database_mode=database_mode,
                num_chunks=int(num_chunks),
                job_id=aggregate_id,
            )
            jobs.append(
                {
                    "job_id": aggregate_id,
                    "job_type": "method_aggregate",
                    "method": "cellagentchat",
                    "phase": phase,
                    "database_mode": database_mode,
                    "repeat_id": repeat_id,
                    "chunk_id": None,
                    "num_chunks": int(num_chunks),
                    "gpu_id": None,
                    "env_name": method_info.get("env_name"),
                    "command": aggregate_command,
                    "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                    "output_dir": _remote_path(remote_root, "runs", run_group, "cellagentchat"),
                    "stdout": _remote_path(remote_root, "logs", f"{aggregate_id}.stdout.log"),
                    "stderr": _remote_path(remote_root, "logs", f"{aggregate_id}.stderr.log"),
                    "resource_log": aggregate_resource_log,
                    "dependencies": dependencies,
                    "status": "planned",
                    "runtime_seconds": None,
                    "peak_memory_gb": None,
                    "exit_code": None,
                    "gpu_required": False,
                    "max_parallel_group": "aggregate",
                    "omp_threads": 4,
                    "mkl_threads": 4,
                }
            )

    payload = {
        "kind": "a100_job_matrix",
        "remote_root": str(remote_root),
        "phase": phase,
        "database_mode": database_mode,
        "methods": [str(item).strip().lower() for item in methods if str(item).strip()],
        "run_group": run_group,
        "max_cci_pairs": max_cci_pairs,
        "n_perms": n_perms,
        "include_bootstrap": include_bootstrap,
        "include_audit": include_audit,
        "parallel_policy": {
            "gpu_count": int(gpu_count),
            "gpu_method_slots": 1,
            "max_r_heavy_parallel": 2,
            "max_cpu_heavy_parallel": 3,
            "max_commot_chunks_parallel": 4,
            "max_cellagentchat_chunks_parallel": int(min(max(1, int(gpu_count)), max(1, int(cellagentchat_chunks)))),
        },
        "cellagentchat_chunks": int(cellagentchat_chunks),
        "jobs": jobs,
    }
    payload["path_policy"] = validate_a100_path_policy(payload, remote_root=remote_root, readonly_xenium_root=DEFAULT_A100_READONLY_XENIUM_ROOT)
    return payload


def _prepare_pilot_bundle_command(
    *,
    remote_root: str | Path,
    n_cells: int = 50_000,
    seed: int = 1,
    force: bool = False,
    job_id: str = "prepare_pilot50k_bundle",
) -> tuple[str, str]:
    root = str(remote_root).replace("\\", "/").rstrip("/")
    force_literal = "True" if force else "False"
    python_code = f"""
import json
from pathlib import Path

root = Path({root!r})
input_manifest_path = root / "data" / "input_manifest.json"
pilot_dir = root / "data" / "pilot50k"
pilot_manifest_path = pilot_dir / "input_manifest.json"
if pilot_manifest_path.exists() and not {force_literal}:
    print(json.dumps({{"status": "exists", "input_manifest": str(pilot_manifest_path)}}))
    raise SystemExit(0)

from pyXenium.benchmarking.cci_adapters import read_sparse_bundle_as_adata
from pyXenium.benchmarking.cci_atera import _bundle_fingerprints, _write_sparse_bundle, stratified_subset

manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
full_bundle = manifest.get("full_bundle") or {{}}
if not full_bundle:
    raise SystemExit("input_manifest.json does not contain a full_bundle for pilot50k creation")

pilot_dir.mkdir(parents=True, exist_ok=True)
adata = read_sparse_bundle_as_adata(full_bundle)
pilot = stratified_subset(adata, n_cells={int(n_cells)}, stratify_key="cell_type", seed={int(seed)})
bundle = _write_sparse_bundle(pilot, pilot_dir)
pilot_manifest = dict(manifest)
pilot_manifest["full_h5ad"] = None
pilot_manifest["full_bundle"] = bundle
pilot_manifest["full_bundle_fingerprints"] = _bundle_fingerprints(bundle)
pilot_manifest["full_n_cells"] = int(pilot.n_obs)
pilot_manifest["full_n_genes"] = int(pilot.n_vars)
pilot_manifest["pilot_source_manifest"] = str(input_manifest_path)
pilot_manifest["pilot_n_cells"] = int(pilot.n_obs)
pilot_manifest["pilot_seed"] = {int(seed)}
pilot_manifest["pilot_note"] = "50k stratified pilot bundle used for A100 CCI method runtime projection."
pilot_manifest_path.write_text(json.dumps(pilot_manifest, indent=2, default=str) + chr(10), encoding="utf-8")
print(json.dumps({{"status": "created", "input_manifest": str(pilot_manifest_path), "n_cells": int(pilot.n_obs)}}))
""".strip()
    inner = _join_command(["conda", "run", "--name", "pyx-cci-prep", "python", "-c", python_code])
    return _wrap_a100_job(
        inner,
        remote_root=remote_root,
        job_id=job_id,
        env_name="pyx-cci-prep",
    )


def build_a100_sidecar_matrix(
    *,
    benchmark_root: str | Path | None = None,
    remote_root: str | Path = DEFAULT_A100_REMOTE_ROOT,
    methods: Sequence[str] = DEFAULT_A100_ALL_METHODS,
    ready_methods: Sequence[str] | None = None,
    database_mode: str = "common-db",
    smoke_max_cci_pairs: int | None = 25,
    pilot_max_cci_pairs: int | None = 100,
    pilot_n_cells: int = 50_000,
    pilot_seed: int = 1,
    include_audit: bool = True,
    include_smoke: bool = True,
    include_pilot: bool = True,
    completed_job_ids: Sequence[str] | None = None,
    running_job_ids: Sequence[str] | None = None,
    gpu_count: int = 8,
    gpu_start: int = 3,
    gzip_edge_outputs: bool = True,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    registry = _method_registry_by_slug(layout)
    requested = [str(item).strip().lower() for item in methods if str(item).strip()]
    ready = {str(item).strip().lower() for item in (ready_methods or requested) if str(item).strip()}
    blocked = set(str(item) for item in (completed_job_ids or [])) | set(str(item) for item in (running_job_ids or []))
    selected_methods = [method for method in requested if method in ready]
    jobs: list[dict[str, Any]] = []
    gpu_cursor = 0

    if include_pilot and "prepare_pilot50k_bundle" not in blocked:
        command, resource_log = _prepare_pilot_bundle_command(
            remote_root=remote_root,
            n_cells=pilot_n_cells,
            seed=pilot_seed,
            job_id="prepare_pilot50k_bundle",
        )
        jobs.append(
            {
                "job_id": "prepare_pilot50k_bundle",
                "job_type": "pilot_bundle",
                "method": "data",
                "phase": "pilot_prep",
                "database_mode": database_mode,
                "env_name": "pyx-cci-prep",
                "command": command,
                "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                "output_dir": _remote_path(remote_root, "data", "pilot50k"),
                "stdout": _remote_path(remote_root, "logs", "prepare_pilot50k_bundle.stdout.log"),
                "stderr": _remote_path(remote_root, "logs", "prepare_pilot50k_bundle.stderr.log"),
                "resource_log": resource_log,
                "status": "planned_sidecar",
                "gpu_required": False,
                "max_parallel_group": "pilot_prep",
                "omp_threads": 8,
                "mkl_threads": 8,
            }
        )

    for method in selected_methods:
        if method not in registry:
            raise ValueError(f"Method {method!r} is not registered in methods.yaml.")
        method_info = registry[method]
        profile = _resource_profile(method)
        if include_audit:
            job_id = f"sidecar_audit_{method}"
            if job_id not in blocked:
                command, resource_log = _audit_command(method, method_info, remote_root, job_id=job_id)
                jobs.append(
                    {
                        "job_id": job_id,
                        "job_type": "env_audit",
                        "method": method,
                        "phase": "audit",
                        "database_mode": database_mode,
                        "env_name": method_info.get("env_name"),
                        "command": command,
                        "output_dir": _remote_path(remote_root, "logs", job_id),
                        "stdout": _remote_path(remote_root, "logs", f"{job_id}.stdout.log"),
                        "stderr": _remote_path(remote_root, "logs", f"{job_id}.stderr.log"),
                        "resource_log": resource_log,
                        "status": "planned_sidecar",
                        **profile,
                    }
                )
        if include_smoke:
            job_id = f"sidecar_smoke_common_db_{method}"
            if job_id not in blocked:
                gpu_id = None
                if profile["gpu_required"]:
                    available = max(1, int(gpu_count) - int(gpu_start))
                    gpu_id = int(gpu_start) + (gpu_cursor % available)
                    gpu_cursor += 1
                command, resource_log = _method_run_command(
                    method=method,
                    method_info=method_info,
                    remote_root=remote_root,
                    phase="smoke",
                    database_mode=database_mode,
                    run_group="smoke_sidecar_common",
                    max_cci_pairs=smoke_max_cci_pairs,
                    n_perms=100,
                    job_id=job_id,
                    bounded_mode="sidecar_smoke_20k",
                    gpu_id=gpu_id,
                    gzip_standardized=gzip_edge_outputs
                    and method in {"commot", "spatialdm", "stlearn", "laris", "cellnest", "cellagentchat", "scild", "niches", "spatalk"},
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "job_type": "method_run",
                        "method": method,
                        "phase": "smoke",
                        "database_mode": database_mode,
                        "bounded_mode": "sidecar_smoke_20k",
                        "gpu_id": gpu_id,
                        "env_name": method_info.get("env_name"),
                        "command": command,
                        "input_manifest": _remote_path(remote_root, "data", "input_manifest.json"),
                        "output_dir": _remote_path(remote_root, "runs", "smoke_sidecar_common", method),
                        "stdout": _remote_path(remote_root, "logs", f"{job_id}.stdout.log"),
                        "stderr": _remote_path(remote_root, "logs", f"{job_id}.stderr.log"),
                        "resource_log": resource_log,
                        "status": "planned_sidecar",
                        "runtime_seconds": None,
                        "peak_memory_gb": None,
                        "exit_code": None,
                        **profile,
                    }
                )
        if include_pilot:
            job_id = f"sidecar_pilot50k_common_db_{method}"
            if job_id not in blocked:
                gpu_id = None
                if profile["gpu_required"]:
                    available = max(1, int(gpu_count) - int(gpu_start))
                    gpu_id = int(gpu_start) + (gpu_cursor % available)
                    gpu_cursor += 1
                command, resource_log = _method_run_command(
                    method=method,
                    method_info=method_info,
                    remote_root=remote_root,
                    phase="full",
                    database_mode=database_mode,
                    run_group="pilot50k_common",
                    max_cci_pairs=pilot_max_cci_pairs,
                    n_perms=100,
                    input_manifest=_remote_path(remote_root, "data", "pilot50k", "input_manifest.json"),
                    job_id=job_id,
                    bounded_mode=f"pilot50k_{pilot_max_cci_pairs or 'all'}cci",
                    gpu_id=gpu_id,
                    gzip_standardized=gzip_edge_outputs
                    and method in {"commot", "spatialdm", "stlearn", "laris", "cellnest", "cellagentchat", "scild", "niches", "spatalk"},
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "job_type": "method_run",
                        "method": method,
                        "phase": "full",
                        "database_mode": database_mode,
                        "bounded_mode": f"pilot50k_{pilot_max_cci_pairs or 'all'}cci",
                        "gpu_id": gpu_id,
                        "env_name": method_info.get("env_name"),
                        "command": command,
                        "input_manifest": _remote_path(remote_root, "data", "pilot50k", "input_manifest.json"),
                        "output_dir": _remote_path(remote_root, "runs", "pilot50k_common", method),
                        "stdout": _remote_path(remote_root, "logs", f"{job_id}.stdout.log"),
                        "stderr": _remote_path(remote_root, "logs", f"{job_id}.stderr.log"),
                        "resource_log": resource_log,
                        "status": "planned_sidecar",
                        "runtime_seconds": None,
                        "peak_memory_gb": None,
                        "exit_code": None,
                        **profile,
                    }
                )

    payload = {
        "kind": "a100_sidecar_matrix",
        "remote_root": str(remote_root),
        "methods": requested,
        "ready_methods": selected_methods,
        "database_mode": database_mode,
        "smoke_run_group": "smoke_sidecar_common",
        "pilot_run_group": "pilot50k_common",
        "smoke_max_cci_pairs": smoke_max_cci_pairs,
        "pilot_max_cci_pairs": pilot_max_cci_pairs,
        "pilot_n_cells": int(pilot_n_cells),
        "pilot_seed": int(pilot_seed),
        "completed_job_ids": sorted(set(completed_job_ids or [])),
        "running_job_ids": sorted(set(running_job_ids or [])),
        "parallel_policy": {
            "profile": "balanced",
            "gpu_count": int(gpu_count),
            "gpu_start": int(gpu_start),
            "gpu_method_slots": 1,
            "max_r_heavy_parallel": 2,
            "max_cpu_heavy_parallel": 4,
            "max_cpu_light_parallel": 4,
        },
        "jobs": jobs,
    }
    payload["path_policy"] = validate_a100_path_policy(payload, remote_root=remote_root, readonly_xenium_root=DEFAULT_A100_READONLY_XENIUM_ROOT)
    return payload


def _remote_background_command(job: Mapping[str, Any]) -> str:
    stdout = _q(str(job["stdout"]))
    stderr = _q(str(job["stderr"]))
    command = str(job["command"])
    return f"mkdir -p $(dirname {stdout}) $(dirname {stderr}) {_q(str(job['output_dir']))} && nohup bash -lc {_q(command)} > {stdout} 2> {stderr} < /dev/null & echo $!"


def submit_a100_matrix(
    *,
    matrix_json: str | Path,
    dry_run: bool = True,
    remote: bool = True,
    host: str | None = None,
    user: str | None = None,
    job_ids: Sequence[str] | None = None,
    job_types: Sequence[str] | None = None,
    ssh_executable: str = "ssh",
) -> dict[str, Any]:
    matrix = json.loads(Path(matrix_json).read_text(encoding="utf-8"))
    selected = set(job_ids or [])
    selected_types = {str(item) for item in (job_types or [])}
    remote_target = _remote_target(host, user) if remote else None
    if remote and not dry_run and not (host and user):
        raise ValueError("host and user are required when submitting A100 jobs remotely.")
    rows: list[dict[str, Any]] = []
    for job in matrix.get("jobs", []):
        if selected and job.get("job_id") not in selected:
            continue
        if selected_types and job.get("job_type") not in selected_types:
            continue
        background = _remote_background_command(job)
        wrapper = f'{ssh_executable} {remote_target} "{background}"' if remote else background
        if dry_run:
            rows.append({**job, "status": "dry-run", "submit_command": wrapper, "background_command": background})
            continue
        started = time.perf_counter()
        if remote:
            completed = subprocess.run([ssh_executable, str(remote_target), background], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        else:
            completed = subprocess.run(background, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        rows.append(
            {
                **job,
                "status": "submitted" if completed.returncode == 0 else "submit_failed",
                "submit_runtime_seconds": float(time.perf_counter() - started),
                "submit_exit_code": completed.returncode,
                "remote_pid": completed.stdout.strip(),
                "submit_stderr": completed.stderr,
                "submit_command": wrapper,
            }
        )
    return {"dry_run": dry_run, "remote": remote, "host": host, "user": user, "jobs": rows}


def monitor_a100_jobs(
    *,
    matrix_json: str | Path,
    benchmark_root: str | Path | None = None,
    output_tsv: str | Path | None = None,
) -> pd.DataFrame:
    matrix = json.loads(Path(matrix_json).read_text(encoding="utf-8"))
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    local_runs = layout.runs_dir / "a100_collected"
    run_status = summarize_run_status(local_runs) if local_runs.exists() else pd.DataFrame()
    status_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not run_status.empty:
        for _, row in run_status.iterrows():
            key = (str(row.get("method")), str(row.get("phase")), str(row.get("database_mode")))
            status_lookup[key] = row.to_dict()
    rows: list[dict[str, Any]] = []
    for job in matrix.get("jobs", []):
        key = (str(job.get("method")), str(job.get("phase")), str(job.get("database_mode")))
        observed = status_lookup.get(key, {})
        rows.append(
            {
                "job_id": job.get("job_id"),
                "method": job.get("method"),
                "phase": job.get("phase"),
                "database_mode": job.get("database_mode"),
                "chunk_id": job.get("chunk_id"),
                "gpu_id": job.get("gpu_id"),
                "planned_status": job.get("status"),
                "observed_status": observed.get("status", "not_collected"),
                "n_rows": observed.get("n_rows"),
                "elapsed_seconds": observed.get("elapsed_seconds"),
                "standardized_tsv": observed.get("standardized_tsv"),
                "stdout": job.get("stdout"),
                "stderr": job.get("stderr"),
                "resource_log": job.get("resource_log"),
            }
        )
    table = pd.DataFrame(rows)
    if output_tsv:
        Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_tsv, sep="\t", index=False)
    return table


FAILED_RUN_STATUSES = {
    "failed",
    "failed_resource_oom",
    "submit_failed",
    "skipped_missing_dependency",
    "unreadable",
}


def _standardized_outputs(run_dir: Path, *, recursive: bool = False) -> list[Path]:
    patterns = ("*standardized.tsv", "*standardized.tsv.gz")
    outputs: list[Path] = []
    for pattern in patterns:
        iterator = run_dir.rglob(pattern) if recursive else run_dir.glob(pattern)
        outputs.extend(path for path in iterator if path.is_file())
    return sorted(set(outputs))


def _run_summary_status(run_dir: Path) -> str:
    summary = _read_json(run_dir / "run_summary.json")
    status = str(summary.get("status") or "").strip().lower()
    if not status:
        return "standardized_present" if _standardized_outputs(run_dir) else "missing_summary"
    return status


def _run_allows_standardized(run_dir: Path) -> bool:
    status = _run_summary_status(run_dir)
    return status not in FAILED_RUN_STATUSES


def _latest_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(records, key=lambda row: (float(row.get("mtime") or 0.0), str(row.get("path") or "")))


def _direct_method_records(
    *,
    run_roots: Sequence[Path],
    method: str,
    platform_name: str,
    source_label_prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in run_roots:
        method_dir = root / method
        if not method_dir.exists() or not _run_allows_standardized(method_dir):
            continue
        for path in _standardized_outputs(method_dir):
            records.append(
                {
                    "method": method,
                    "platform": platform_name,
                    "source_label": source_label_prefix,
                    "path": str(path),
                    "run_dir": str(method_dir),
                    "status": _run_summary_status(method_dir),
                    "chunk_id": None,
                    "num_chunks": None,
                    "mtime": path.stat().st_mtime,
                }
            )
    return records


def _pdc_full_roots(layout: BenchmarkLayout, pdc_tag: str | None = None) -> list[tuple[str, Path]]:
    collected = layout.root / "pdc_collected"
    if not collected.exists():
        return []
    tags = [collected / pdc_tag] if pdc_tag else sorted(path for path in collected.iterdir() if path.is_dir())
    roots: list[tuple[str, Path]] = []
    for tag_dir in tags:
        run_root = tag_dir / "runs" / "full_common"
        if run_root.exists():
            roots.append((tag_dir.name, run_root))
    return roots


def _pdc_direct_method_record(layout: BenchmarkLayout, method: str, *, pdc_tag: str | None) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    for tag, run_root in _pdc_full_roots(layout, pdc_tag=pdc_tag):
        records.extend(
            _direct_method_records(
                run_roots=[run_root],
                method=method,
                platform_name="pdc",
                source_label_prefix=f"pdc:{tag}",
            )
        )
    return _latest_record(records)


def _chunk_number_from_dir(path: Path) -> int | None:
    name = path.name
    if not name.startswith("chunk_") or "_of_" not in name:
        return None
    value = name.split("_of_", 1)[0].replace("chunk_", "")
    try:
        return int(value)
    except ValueError:
        return None


def _pdc_commot_chunk_records(layout: BenchmarkLayout, *, pdc_tag: str | None, expected_chunks: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    best: list[dict[str, Any]] = []
    best_issue: dict[str, Any] | None = None
    for tag, run_root in _pdc_full_roots(layout, pdc_tag=pdc_tag):
        commot_root = run_root / "commot"
        if not commot_root.exists():
            continue
        paths = _standardized_outputs(commot_root, recursive=True)
        by_chunk: dict[int, Path] = {}
        blocked_chunks: list[int] = []
        for path in paths:
            chunk_id = _chunk_number_from_dir(path.parent)
            if chunk_id is None:
                continue
            if not _run_allows_standardized(path.parent):
                blocked_chunks.append(chunk_id)
                continue
            existing = by_chunk.get(chunk_id)
            if existing is None or path.stat().st_mtime > existing.stat().st_mtime:
                by_chunk[chunk_id] = path
        missing = [idx for idx in range(int(expected_chunks)) if idx not in by_chunk]
        if missing or blocked_chunks:
            best_issue = {
                "method": "commot",
                "platform": "pdc",
                "source_label": f"pdc:{tag}",
                "expected_chunks": int(expected_chunks),
                "observed_chunks": sorted(by_chunk),
                "missing_chunks": missing,
                "blocked_chunks": sorted(set(blocked_chunks)),
            }
            continue
        records = [
            {
                "method": "commot",
                "platform": "pdc",
                "source_label": f"pdc:{tag}",
                "path": str(by_chunk[idx]),
                "run_dir": str(by_chunk[idx].parent),
                "status": _run_summary_status(by_chunk[idx].parent),
                "chunk_id": idx,
                "num_chunks": int(expected_chunks),
                "mtime": by_chunk[idx].stat().st_mtime,
            }
            for idx in range(int(expected_chunks))
        ]
        if not best or max(row["mtime"] for row in records) > max(row["mtime"] for row in best):
            best = records
            best_issue = None
    return best, best_issue


def _pdc_failure_cards(layout: BenchmarkLayout, method: str, *, pdc_tag: str | None) -> list[str]:
    cards: list[str] = []
    for _, run_root in _pdc_full_roots(layout, pdc_tag=pdc_tag):
        method_root = run_root / method
        if method_root.exists():
            cards.extend(str(path) for path in sorted(method_root.rglob("method_card.md")) if path.is_file())
    return cards


def select_cci_source_of_truth_inputs(
    *,
    benchmark_root: str | Path | None = None,
    a100_methods: Sequence[str] = A100_AUTHORITATIVE_FULL_METHODS,
    pdc_methods: Sequence[str] = PDC_FULL_BACKFILL_METHODS,
    pdc_tag: str | None = None,
    commot_chunks: int = 16,
) -> dict[str, Any]:
    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    a100_full = layout.runs_dir / "a100_collected" / "full_common"
    selected: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    selected_methods: set[str] = set()

    for method in [str(item).strip().lower() for item in a100_methods if str(item).strip()]:
        record = _latest_record(
            _direct_method_records(
                run_roots=[a100_full],
                method=method,
                platform_name="a100",
                source_label_prefix="a100:full_common",
            )
        )
        if record is None:
            issues.append({"method": method, "platform": "a100", "status": "missing_authoritative_full"})
            continue
        selected.append(record)
        selected_methods.add(method)

    for method in [str(item).strip().lower() for item in pdc_methods if str(item).strip()]:
        if method in selected_methods:
            continue
        if method == "commot":
            records, issue = _pdc_commot_chunk_records(layout, pdc_tag=pdc_tag, expected_chunks=commot_chunks)
            if records:
                selected.extend(records)
                selected_methods.add(method)
            elif issue:
                issue["failure_cards"] = _pdc_failure_cards(layout, method, pdc_tag=pdc_tag)
                issues.append(issue)
            else:
                issues.append(
                    {
                        "method": method,
                        "platform": "pdc",
                        "status": "missing_full_chunks",
                        "expected_chunks": int(commot_chunks),
                        "failure_cards": _pdc_failure_cards(layout, method, pdc_tag=pdc_tag),
                    }
                )
            continue
        record = _pdc_direct_method_record(layout, method, pdc_tag=pdc_tag)
        if record is None:
            issues.append(
                {
                    "method": method,
                    "platform": "pdc",
                    "status": "missing_full_standardized",
                    "failure_cards": _pdc_failure_cards(layout, method, pdc_tag=pdc_tag),
                }
            )
            continue
        selected.append(record)
        selected_methods.add(method)

    selected = sorted(selected, key=lambda row: (str(row["method"]), row.get("chunk_id") is not None, row.get("chunk_id") or -1))
    return {
        "selected": selected,
        "issues": issues,
        "a100_authoritative_full_methods": [str(item).strip().lower() for item in a100_methods if str(item).strip()],
        "pdc_full_backfill_methods": [str(item).strip().lower() for item in pdc_methods if str(item).strip()],
        "selected_methods": sorted(selected_methods),
        "input_paths": [row["path"] for row in selected],
        "commot_chunks": int(commot_chunks),
        "pdc_tag": pdc_tag,
    }


def _schema_validation(result_paths: Sequence[str | Path], combined: pd.DataFrame) -> dict[str, Any]:
    from .cci_atera import STANDARDIZED_RESULT_COLUMNS

    missing_by_file: dict[str, list[str]] = {}
    for path_value in result_paths:
        path = Path(path_value)
        header = pd.read_csv(path, sep="\t", nrows=0, compression="infer")
        missing = [column for column in STANDARDIZED_RESULT_COLUMNS if column not in header.columns]
        if missing:
            missing_by_file[str(path)] = missing
    return {
        "status": "passed" if not missing_by_file else "failed",
        "n_inputs": len(result_paths),
        "n_rows": int(len(combined)),
        "n_methods": int(combined["method"].nunique()) if not combined.empty and "method" in combined else 0,
        "missing_columns_by_file": missing_by_file,
        "required_columns": STANDARDIZED_RESULT_COLUMNS,
    }


def finalize_cci_source_of_truth(
    *,
    benchmark_root: str | Path | None = None,
    output_prefix: str = "source_of_truth_full_common",
    pdc_tag: str | None = None,
    commot_chunks: int = 16,
    render_report: bool = True,
) -> dict[str, Any]:
    from .cci_atera import (
        aggregate_standardized_results,
        build_canonical_rank_matrix,
        compute_canonical_recovery,
        compute_novelty_support,
        compute_pathway_relevance,
        compute_robustness,
        compute_spatial_coherence,
        render_atera_cci_benchmark_report,
        score_biological_performance,
    )

    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    selection = select_cci_source_of_truth_inputs(
        benchmark_root=layout.root,
        pdc_tag=pdc_tag,
        commot_chunks=commot_chunks,
    )
    input_paths = [Path(path) for path in selection["input_paths"]]
    combined_path = layout.results_dir / f"{output_prefix}_combined.tsv"
    combined = aggregate_standardized_results(input_paths, output_path=combined_path) if input_paths else pd.DataFrame()
    validation = _schema_validation(input_paths, combined) if input_paths else {
        "status": "empty",
        "n_inputs": 0,
        "n_rows": 0,
        "n_methods": 0,
        "missing_columns_by_file": {},
        "required_columns": [],
    }

    inputs_path = layout.results_dir / f"{output_prefix}_inputs.tsv"
    pd.DataFrame(selection["selected"]).drop(columns=["mtime"], errors="ignore").to_csv(inputs_path, sep="\t", index=False)
    validation_path = layout.results_dir / f"{output_prefix}_schema_validation.json"
    _write_json(validation_path, validation)

    outputs: dict[str, str] = {
        "combined_results": str(combined_path),
        "inputs": str(inputs_path),
        "schema_validation": str(validation_path),
    }
    if render_report and not combined.empty and (layout.config_dir / "canonical_axes.yaml").exists() and (layout.config_dir / "pathways.yaml").exists():
        canonical_detail, canonical_summary = compute_canonical_recovery(combined, canonical_axes=layout.config_dir / "canonical_axes.yaml")
        pathway_detail, pathway_summary = compute_pathway_relevance(combined, pathway_config=layout.config_dir / "pathways.yaml")
        spatial_summary = compute_spatial_coherence(combined)
        novelty_detail, novelty_summary = compute_novelty_support(combined)
        robustness_summary = compute_robustness(combined)
        biology_summary = score_biological_performance(
            canonical_summary=canonical_summary,
            pathway_summary=pathway_summary,
            spatial_summary=spatial_summary,
            robustness_summary=robustness_summary,
            novelty_summary=novelty_summary,
        )
        rank_matrix = build_canonical_rank_matrix(canonical_detail)
        run_status = pd.DataFrame(selection["selected"]).drop(columns=["mtime"], errors="ignore")
        table_outputs = {
            "canonical_detail": (layout.results_dir / f"{output_prefix}_canonical_detail.tsv", canonical_detail),
            "canonical_summary": (layout.results_dir / f"{output_prefix}_canonical_summary.tsv", canonical_summary),
            "canonical_rank_matrix": (layout.results_dir / f"{output_prefix}_canonical_rank_matrix.tsv", rank_matrix),
            "pathway_detail": (layout.results_dir / f"{output_prefix}_pathway_detail.tsv", pathway_detail),
            "pathway_summary": (layout.results_dir / f"{output_prefix}_pathway_summary.tsv", pathway_summary),
            "spatial_summary": (layout.results_dir / f"{output_prefix}_spatial_coherence.tsv", spatial_summary),
            "novelty_detail": (layout.results_dir / f"{output_prefix}_novelty_top.tsv", novelty_detail),
            "novelty_summary": (layout.results_dir / f"{output_prefix}_novelty_summary.tsv", novelty_summary),
            "robustness_summary": (layout.results_dir / f"{output_prefix}_robustness_summary.tsv", robustness_summary),
            "biology_scoreboard": (layout.results_dir / f"{output_prefix}_biology_scoreboard.tsv", biology_summary),
            "engineering_scoreboard": (layout.results_dir / f"{output_prefix}_engineering_scoreboard.tsv", run_status),
        }
        for key, (path, table) in table_outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, sep="\t", index=False)
            outputs[key] = str(path)
        report = render_atera_cci_benchmark_report(
            combined_results=combined,
            canonical_summary=canonical_summary,
            pathway_summary=pathway_summary,
            biology_summary=biology_summary,
            benchmark_root=layout.root,
            run_status=run_status,
            engineering_summary=run_status,
            canonical_detail=canonical_detail,
            a100_resource_summary=run_status[run_status["platform"] == "a100"] if "platform" in run_status else run_status,
        )
        report_path = layout.reports_dir / f"{output_prefix}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        outputs["report"] = str(report_path)

    payload = {
        "kind": "cci_source_of_truth_finalize",
        "benchmark_root": str(layout.root),
        "n_rows": int(len(combined)),
        "n_methods": int(combined["method"].nunique()) if not combined.empty and "method" in combined.columns else 0,
        "methods": sorted(combined["method"].dropna().unique().tolist()) if not combined.empty and "method" in combined.columns else [],
        "selection": selection,
        "validation": validation,
        "outputs": outputs,
    }
    _write_json(layout.results_dir / f"{output_prefix}_finalize_summary.json", payload)
    return payload


def finalize_a100_all(
    *,
    benchmark_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    from .cci_atera import aggregate_standardized_results

    layout = resolve_layout(relative_root=benchmark_root or ATERA_BENCHMARK_RELATIVE_ROOT)
    discovered = sorted(
        [
            *layout.runs_dir.glob("a100_collected/**/*standardized.tsv"),
            *layout.runs_dir.glob("a100_collected/**/*standardized.tsv.gz"),
        ]
    )
    resolved_output = Path(output_path) if output_path else layout.results_dir / "a100_all_combined_standardized.tsv"
    combined = aggregate_standardized_results(discovered, output_path=resolved_output) if discovered else pd.DataFrame()
    payload = {
        "output_path": str(resolved_output),
        "n_rows": int(len(combined)),
        "n_methods": int(combined["method"].nunique()) if not combined.empty and "method" in combined.columns else 0,
        "inputs": [str(path) for path in discovered],
    }
    _write_json(layout.results_dir / "a100_all_finalize_summary.json", payload)
    return payload
