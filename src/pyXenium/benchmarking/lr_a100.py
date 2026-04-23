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

from .lr_atera import ATERA_BENCHMARK_RELATIVE_ROOT, BenchmarkLayout, load_method_registry, resolve_layout
from .lr_adapters import SUPPORTED_REAL_ADAPTERS, load_input_manifest, validate_input_manifest


DEFAULT_A100_REMOTE_ROOT = "/data/taobo.hu/pyxenium_lr_benchmark_2026-04"
DEFAULT_A100_READONLY_XENIUM_ROOT = "/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs"
DEFAULT_A100_METHOD_ORDER = ("pyxenium", "squidpy", "liana", "commot", "cellchat")
DEFAULT_A100_WRITABLE_SUBDIRS = (
    "repo",
    "configs",
    "envs",
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
        f"export TMPDIR={_q(tmp_dir)} && "
        f"mkdir -p { _q(tmp_dir) } && "
        f"export PYTHONPATH={_q(repo_src)}:${{PYTHONPATH:-}} && "
        f"cd {_q(repo_dir)}"
    )


def _wrap_a100_job(inner_command: str, *, remote_root: str | Path, job_id: str) -> tuple[str, str]:
    resource_log = _remote_path(remote_root, "logs", f"{job_id}.resource.log")
    command = f"{_a100_prefix(remote_root)} && /usr/bin/time -v -o {_q(resource_log)} {inner_command}"
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
        "pyx-lr-prep",
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
    return _wrap_a100_job(_join_command(parts), remote_root=remote_root, job_id=job_id)


def _method_run_command(
    *,
    method: str,
    method_info: Mapping[str, Any],
    remote_root: str | Path,
    phase: str,
    database_mode: str,
    run_group: str,
    max_lr_pairs: int | None,
    n_perms: int,
    repeat_id: str | None = None,
    job_id: str,
) -> tuple[str, str]:
    env_name = str(method_info.get("env_name", ""))
    output_dir = _remote_path(remote_root, "runs", run_group, method if repeat_id is None else f"{method}_{repeat_id}")
    input_manifest = _remote_path(remote_root, "data", "input_manifest.json")
    if method_info.get("language") == "r" and method == "cellchat":
        parts = [
            "conda",
            "run",
            "--name",
            env_name,
            "Rscript",
            _remote_path(remote_root, "runners", "r", "run_cellchat.R"),
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
    if max_lr_pairs is not None:
        parts.extend(["--max-lr-pairs", str(max_lr_pairs)])
    return _wrap_a100_job(_join_command(parts), remote_root=remote_root, job_id=job_id)


def _audit_command(method: str, method_info: Mapping[str, Any], remote_root: str | Path, *, job_id: str) -> tuple[str, str]:
    env_name = str(method_info.get("env_name", ""))
    if method_info.get("language") == "r":
        expr = "cat(R.version.string, '\\n'); cat('R ok\\n')"
        if method == "cellchat":
            expr = "cat(R.version.string, '\\n'); if (!requireNamespace('CellChat', quietly=TRUE)) stop('CellChat missing')"
        inner = _join_command(["conda", "run", "--name", env_name, "Rscript", "-e", expr])
        return _wrap_a100_job(inner, remote_root=remote_root, job_id=job_id)
    py_expr = "import importlib.metadata as m; import pyXenium; print('pyXenium ok'); print('pandas', m.version('pandas'))"
    inner = _join_command(["conda", "run", "--name", env_name, "python", "-c", py_expr])
    return _wrap_a100_job(inner, remote_root=remote_root, job_id=job_id)


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
    max_lr_pairs: int | None = None,
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
                "env_name": "pyx-lr-prep",
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
            max_lr_pairs=max_lr_pairs,
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
        "max_lr_pairs": max_lr_pairs,
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
    max_lr_pairs: int | None = None,
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
        max_lr_pairs=max_lr_pairs,
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
