from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite


ROOT = Path("/data/taobo.hu/pyxenium_lr_benchmark_2026-04")
REPO = ROOT / "repo"
RUN_METHOD = ROOT / "scripts" / "run_method.py"
PYX_ENV = ROOT / "repo" / "src"
REQUIRED_STANDARDIZED_COLUMNS = [
    "method",
    "database_mode",
    "ligand",
    "receptor",
    "sender",
    "receiver",
    "score_raw",
    "score_std",
    "rank_within_method",
    "rank_fraction",
    "fdr_or_pvalue",
    "resolution",
    "spatial_support_type",
    "artifact_path",
]


@dataclass(frozen=True)
class Candidate:
    tag: str
    n_cells: int
    manifest: Path
    full_retry: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_method_card(output_dir: Path, payload: dict[str, Any]) -> None:
    reason = str(payload.get("reason") or payload.get("error") or payload.get("status", "failed"))
    text = (
        f"# Method Card: {payload.get('method', 'unknown')}\n\n"
        f"- Status: `{payload.get('status', 'failed')}`\n"
        f"- Stage: `{payload.get('stage', 'bounded_rescue')}`\n"
        f"- Reason: `{reason}`\n"
        f"- Reproduce: `See run_summary.json and logs in this directory.`\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "method_card.md").write_text(text, encoding="utf-8")


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _bundle_for_split(split: str) -> dict[str, str]:
    base = ROOT / "data" / split
    return {
        "counts_symbol_mtx": str(base / "counts_symbol.mtx"),
        "barcodes_tsv": str(base / "barcodes.tsv"),
        "genes_tsv": str(base / "genes.tsv"),
        "meta_tsv": str(base / "meta.tsv"),
        "coords_tsv": str(base / "coords.tsv"),
    }


def _make_manifest(split: str, n_cells: int, *, source_note: str) -> Path:
    base_manifest = _read_json(ROOT / "data" / "input_manifest.json")
    bundle = _bundle_for_split(split)
    manifest = {
        **base_manifest,
        "full_h5ad": None,
        "smoke_h5ad": None,
        "full_bundle": bundle,
        "smoke_bundle": bundle,
        "subset_tag": split,
        "subset_n_cells": int(n_cells),
        "subset_seed": 1,
        "subset_note": source_note,
        "lr_policy": "full_common_db",
        "full_n_cells": int(n_cells),
        "smoke_n_cells": int(n_cells),
    }
    path = ROOT / "data" / split / "input_manifest.json"
    _write_json(path, manifest)
    return path


def _stratified_indices(labels: pd.Series, n_cells: int, seed: int = 1) -> np.ndarray:
    if n_cells >= len(labels):
        return np.arange(len(labels), dtype=int)
    counts = labels.value_counts().sort_index()
    raw = counts / counts.sum() * n_cells
    alloc = pd.Series(np.floor(raw.to_numpy()).astype(int), index=raw.index)
    alloc = alloc.clip(lower=1)
    while int(alloc.sum()) > n_cells:
        candidates = alloc[alloc > 1]
        label = (raw.loc[candidates.index] - np.floor(raw.loc[candidates.index])).sort_values().index[0]
        alloc[label] -= 1
    while int(alloc.sum()) < n_cells:
        label = (raw - np.floor(raw)).sort_values(ascending=False).index[0]
        alloc[label] += 1
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label, n in alloc.items():
        positions = np.flatnonzero(labels.to_numpy() == label)
        take = min(int(n), len(positions))
        selected.extend(rng.choice(positions, size=take, replace=False).tolist())
    selected_array = np.array(sorted(selected), dtype=int)
    if len(selected_array) != n_cells:
        remaining = np.setdiff1d(np.arange(len(labels), dtype=int), selected_array, assume_unique=False)
        if len(selected_array) < n_cells:
            add = rng.choice(remaining, size=n_cells - len(selected_array), replace=False)
            selected_array = np.array(sorted(np.concatenate([selected_array, add])), dtype=int)
        else:
            selected_array = np.array(sorted(rng.choice(selected_array, size=n_cells, replace=False)), dtype=int)
    return selected_array


def ensure_subset_bundles() -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {
        "full170k": Candidate("full170k", _line_count(ROOT / "data" / "full" / "barcodes.tsv"), ROOT / "data" / "input_manifest.json", full_retry=True),
        "pilot50k": Candidate("pilot50k", 50_000, _make_manifest("pilot50k", 50_000, source_note="existing deterministic pilot50k bundle")),
        "smoke20k": Candidate("smoke20k", 20_000, _make_manifest("smoke", 20_000, source_note="existing deterministic smoke20k bundle")),
    }
    full_bundle = _bundle_for_split("full")
    full_meta = pd.read_csv(full_bundle["meta_tsv"], sep="\t")
    if "cell_type" not in full_meta.columns:
        raise ValueError("Full meta.tsv must include cell_type for stratified bounded rescue subsets.")
    full_barcodes = pd.read_csv(full_bundle["barcodes_tsv"], header=None, sep="\t").iloc[:, 0].astype(str)
    genes = pd.read_csv(full_bundle["genes_tsv"], sep="\t")
    coords = pd.read_csv(full_bundle["coords_tsv"], sep="\t")
    matrix = None
    for split, n_cells in (("subset10k", 10_000), ("subset5k", 5_000)):
        base = ROOT / "data" / split
        manifest_path = base / "input_manifest.json"
        expected = base / "counts_symbol.mtx"
        if not expected.exists() or _line_count(base / "barcodes.tsv") != n_cells:
            if matrix is None:
                matrix = mmread(full_bundle["counts_symbol_mtx"]).tocsr()
            indices = _stratified_indices(full_meta["cell_type"].astype(str), n_cells=n_cells, seed=1)
            base.mkdir(parents=True, exist_ok=True)
            selected_barcodes = full_barcodes.iloc[indices].reset_index(drop=True)
            selected_meta = full_meta.iloc[indices].copy()
            selected_coords = coords.iloc[indices].copy()
            selected_barcodes.to_csv(base / "barcodes.tsv", sep="\t", index=False, header=False)
            selected_meta.to_csv(base / "meta.tsv", sep="\t", index=False)
            selected_coords.to_csv(base / "coords.tsv", sep="\t", index=False)
            genes.to_csv(base / "genes.tsv", sep="\t", index=False)
            if matrix.shape[0] == len(full_barcodes):
                sub_matrix = matrix[indices, :]
            elif matrix.shape[1] == len(full_barcodes):
                sub_matrix = matrix[:, indices]
            else:
                raise ValueError(f"Full matrix shape {matrix.shape} does not match {len(full_barcodes)} barcodes.")
            mmwrite(base / "counts_symbol.mtx", sub_matrix)
        manifest_path = _make_manifest(split, n_cells, source_note=f"stratified subset from full bundle, seed=1, n_cells={n_cells}")
        candidates[split] = Candidate(split, n_cells, manifest_path)
    return candidates


def _sum_rss_for_pgid(pgid: int) -> int:
    completed = subprocess.run(["ps", "-eo", "pgid=,rss="], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    total = 0
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pgid:
            total += int(parts[1])
    return total


def _gpu_snapshot() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip().replace("\n", "; ")


def _standardized_paths(output_dir: Path, method: str) -> list[Path]:
    return [p for p in (output_dir / f"{method}_standardized.tsv", output_dir / f"{method}_standardized.tsv.gz") if p.exists()]


def _validate_standardized(path: Path) -> int:
    table = pd.read_csv(path, sep="\t", compression="infer")
    missing = [col for col in REQUIRED_STANDARDIZED_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing standardized columns: {missing}")
    if table.empty:
        raise ValueError(f"{path} is empty")
    return int(len(table))


def run_monitored(
    *,
    name: str,
    command: str,
    output_dir: Path,
    method: str,
    env: dict[str, str],
    max_rss_gb: float = 900.0,
    poll_seconds: int = 60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("method_card.md",):
        (output_dir / stale).unlink(missing_ok=True)
    stdout_path = output_dir / "bounded_stdout.log"
    stderr_path = output_dir / "bounded_stderr.log"
    resource_path = output_dir / "bounded_resource.csv"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            env=env,
            preexec_fn=os.setsid,
        )
        pgid = os.getpgid(proc.pid)
        killed_for_rss = False
        peak_rss_kb = 0
        with resource_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["elapsed_seconds", "rss_gb", "gpu_snapshot"])
            writer.writeheader()
            while proc.poll() is None:
                rss_kb = _sum_rss_for_pgid(pgid)
                peak_rss_kb = max(peak_rss_kb, rss_kb)
                writer.writerow(
                    {
                        "elapsed_seconds": round(time.time() - start, 3),
                        "rss_gb": round(rss_kb / 1024 / 1024, 6),
                        "gpu_snapshot": _gpu_snapshot(),
                    }
                )
                handle.flush()
                if rss_kb / 1024 / 1024 > max_rss_gb:
                    killed_for_rss = True
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(10)
                    if proc.poll() is None:
                        os.killpg(pgid, signal.SIGKILL)
                    break
                time.sleep(poll_seconds)
        returncode = proc.wait()
    elapsed = time.time() - start
    standardized = _standardized_paths(output_dir, method)
    if returncode == 0 and standardized:
        n_rows = _validate_standardized(standardized[0])
        payload = {
            "method": method,
            "status": "success",
            "stage": "bounded_rescue",
            "job_name": name,
            "standardized_tsv_gz": str(standardized[0]) if str(standardized[0]).endswith(".gz") else None,
            "standardized_tsv": str(standardized[0]),
            "n_rows": n_rows,
            "peak_rss_gb": round(peak_rss_kb / 1024 / 1024, 6),
            "elapsed_seconds": round(elapsed, 3),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "resource_log": str(resource_path),
        }
        _write_json(output_dir / "run_summary.json", payload)
        return payload
    status = "resource_exceeded" if killed_for_rss or returncode in (-9, 137) else "method_api_failure"
    reason = f"bounded command exited {returncode}; standardized output missing"
    payload = {
        "method": method,
        "status": status,
        "stage": "bounded_rescue",
        "job_name": name,
        "reason": reason,
        "returncode": returncode,
        "peak_rss_gb": round(peak_rss_kb / 1024 / 1024, 6),
        "elapsed_seconds": round(elapsed, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "resource_log": str(resource_path),
    }
    _write_json(output_dir / "run_summary.json", payload)
    _write_method_card(output_dir, payload)
    return payload


def _base_env(env_name: str, gpu_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / 'miniconda3/bin'}:{Path.home() / 'miniconda3/condabin'}:{env.get('PATH', '')}"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{PYX_ENV}:{env.get('PYTHONPATH', '')}"
    env["TMPDIR"] = str(ROOT / "tmp")
    env["LD_LIBRARY_PATH"] = f"{Path.home() / 'miniconda3/envs' / env_name / 'lib'}:{Path.home() / 'miniconda3/lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _run_method_command(env_name: str, method: str, manifest: Path, output_dir: Path, *, chunk_id: int | None = None, num_chunks: int | None = None) -> str:
    parts = [
        "conda",
        "run",
        "--name",
        env_name,
        "python",
        str(RUN_METHOD),
        "--method",
        method,
        "--benchmark-root",
        str(ROOT),
        "--input-manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--database-mode",
        "common-db",
        "--phase",
        "full",
        "--n-perms",
        "100",
        "--gzip-standardized",
        "--job-id",
        output_dir.name,
    ]
    if chunk_id is not None:
        parts.extend(["--chunk-id", str(chunk_id)])
    if num_chunks is not None:
        parts.extend(["--num-chunks", str(num_chunks)])
    return " ".join(subprocess.list2cmdline([part]) for part in parts)


def aggregate_cellagentchat(root: Path, *, tag: str, n_cells: int) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    chunk_paths: list[str] = []
    for idx in range(16):
        chunk_dir = root / f"chunk_{idx:03d}_of_016"
        paths = _standardized_paths(chunk_dir, "cellagentchat")
        if not paths:
            raise FileNotFoundError(f"Missing CellAgentChat chunk output: {chunk_dir}")
        _validate_standardized(paths[0])
        frames.append(pd.read_csv(paths[0], sep="\t", compression="infer"))
        chunk_paths.append(str(paths[0]))
    combined = pd.concat(frames, ignore_index=True)
    extra_columns = [col for col in combined.columns if col not in REQUIRED_STANDARDIZED_COLUMNS]
    scores = pd.to_numeric(combined["score_raw"], errors="coerce")
    fill = scores.min(skipna=True) if scores.notna().any() else 0.0
    sortable = pd.DataFrame({"score": scores.fillna(fill)}, index=combined.index)
    pvalues = pd.to_numeric(combined["fdr_or_pvalue"], errors="coerce").fillna(np.inf)
    sortable["pvalue"] = pvalues
    ordered = sortable.sort_values(["score", "pvalue"], ascending=[False, True], kind="mergesort").index
    ranks = pd.Series(np.arange(1, len(ordered) + 1, dtype=float), index=ordered).reindex(combined.index)
    combined["rank_within_method"] = ranks.astype(float)
    combined["rank_fraction"] = (1.0 - ((ranks - 1.0) / float(len(ordered)))).astype(float)
    combined["score_std"] = combined["rank_fraction"]
    combined = combined.loc[:, REQUIRED_STANDARDIZED_COLUMNS + extra_columns].sort_values("rank_within_method").reset_index(drop=True)
    output = root / "cellagentchat_standardized.tsv.gz"
    combined.to_csv(output, sep="\t", index=False, compression="gzip")
    payload = {
        "method": "cellagentchat",
        "status": "success",
        "stage": "bounded_rescue_aggregate",
        "subset_tag": tag,
        "subset_n_cells": int(n_cells),
        "lr_policy": "full_common_db",
        "standardized_tsv_gz": str(output),
        "n_rows": int(len(combined)),
        "chunk_standardized_tsvs": chunk_paths,
    }
    _write_json(root / "run_summary.json", payload)
    return payload


def run_cellagentchat(candidates: dict[str, Candidate], rows: list[dict[str, Any]]) -> None:
    env = _base_env("pyx-lr-cellagentchat", gpu_id=0)
    env.update(
        {
            "CELLAGENTCHAT_FEATURE_SELECTION": "0",
            "CELLAGENTCHAT_EPOCHS": "10",
            "CELLAGENTCHAT_MAX_STEPS": "1",
        }
    )
    sequence = [candidates["full170k"], candidates["pilot50k"], candidates["smoke20k"], candidates["subset10k"], candidates["subset5k"]]
    full_failure_reason = "previous full chunks 0-6 killed during concurrent A100 run"
    for candidate in sequence:
        root = ROOT / "runs" / ("bounded_full_retry" if candidate.full_retry else "bounded_rescue") / "cellagentchat" / candidate.tag
        root.mkdir(parents=True, exist_ok=True)
        failed: dict[str, Any] | None = None
        for idx in range(16):
            chunk_dir = root / f"chunk_{idx:03d}_of_016"
            if _standardized_paths(chunk_dir, "cellagentchat"):
                continue
            command = _run_method_command("pyx-lr-cellagentchat", "cellagentchat", candidate.manifest, chunk_dir, chunk_id=idx, num_chunks=16)
            payload = run_monitored(
                name=f"cellagentchat_{candidate.tag}_chunk_{idx:03d}",
                command=command,
                output_dir=chunk_dir,
                method="cellagentchat",
                env=env,
            )
            if payload["status"] != "success":
                failed = payload
                break
        if failed is None:
            aggregate = aggregate_cellagentchat(root, tag=candidate.tag, n_cells=candidate.n_cells)
            rows.append(
                {
                    "method": "cellagentchat",
                    "subset_n_cells": candidate.n_cells,
                    "subset_tag": candidate.tag,
                    "lr_policy": "full_common_db",
                    "status": "success_full_retry" if candidate.full_retry else "success",
                    "standardized_tsv_gz": aggregate["standardized_tsv_gz"],
                    "peak_rss_gb": "",
                    "elapsed_seconds": "",
                    "full_failure_reason": full_failure_reason,
                }
            )
            return
        rows.append(
            {
                "method": "cellagentchat",
                "subset_n_cells": candidate.n_cells,
                "subset_tag": candidate.tag,
                "lr_policy": "full_common_db",
                "status": failed["status"],
                "standardized_tsv_gz": "",
                "peak_rss_gb": failed.get("peak_rss_gb", ""),
                "elapsed_seconds": failed.get("elapsed_seconds", ""),
                "full_failure_reason": full_failure_reason,
            }
        )


def run_single_method(method: str, env_name: str, candidates: list[Candidate], rows: list[dict[str, Any]], env_updates: dict[str, str], *, ascending_keep_max: bool = False) -> None:
    env = _base_env(env_name, gpu_id=0)
    env.update(env_updates)
    full_failure_reason = {
        "cellnest": "170k full failed with SIGKILL/-9 in run_CellNEST.py",
        "scild": "20k smoke reached ~330GB RSS and did not produce standardized output",
    }.get(method, "")
    best: dict[str, Any] | None = None
    for candidate in candidates:
        output_dir = ROOT / "runs" / "bounded_rescue" / method / candidate.tag
        command = _run_method_command(env_name, method, candidate.manifest, output_dir)
        payload = run_monitored(name=f"{method}_{candidate.tag}", command=command, output_dir=output_dir, method=method, env=env)
        row = {
            "method": method,
            "subset_n_cells": candidate.n_cells,
            "subset_tag": candidate.tag,
            "lr_policy": "full_common_db",
            "status": payload["status"],
            "standardized_tsv_gz": payload.get("standardized_tsv_gz") or payload.get("standardized_tsv") or "",
            "peak_rss_gb": payload.get("peak_rss_gb", ""),
            "elapsed_seconds": payload.get("elapsed_seconds", ""),
            "full_failure_reason": full_failure_reason,
        }
        rows.append(row)
        if payload["status"] == "success":
            best = row
            if not ascending_keep_max:
                return
        elif ascending_keep_max and best is not None:
            return
    if best is None:
        fail_dir = ROOT / "runs" / "bounded_rescue" / method
        payload = {
            "method": method,
            "status": "method_not_scalable_or_adapter_issue",
            "stage": "bounded_rescue",
            "reason": "No bounded candidate produced a non-empty standardized TSV/GZ.",
            "full_failure_reason": full_failure_reason,
        }
        _write_json(fail_dir / "run_summary.json", payload)
        _write_method_card(fail_dir, payload)


def write_summary(rows: list[dict[str, Any]]) -> None:
    out_dir = ROOT / "results" / "bounded_rescue_20260430"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "method",
                "subset_n_cells",
                "subset_tag",
                "lr_policy",
                "status",
                "standardized_tsv_gz",
                "peak_rss_gb",
                "elapsed_seconds",
                "full_failure_reason",
            ]
        )
    table.to_csv(out_dir / "bounded_rescue_summary.tsv", sep="\t", index=False)
    _write_json(out_dir / "bounded_rescue_summary.json", {"rows": rows, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A100 bounded rescue for LR methods that exceed full-data resources.")
    parser.add_argument("--methods", default="cellagentchat,cellnest,scild")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    os.environ["PYTHONPATH"] = f"{PYX_ENV}:{os.environ.get('PYTHONPATH', '')}"
    candidates = ensure_subset_bundles()
    rows: list[dict[str, Any]] = []
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    status_path = ROOT / "logs" / "a100_bounded_rescue_supervisor_status_20260430.json"
    _write_json(status_path, {"status": "started", "methods": methods, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    try:
        if "cellagentchat" in methods:
            run_cellagentchat(candidates, rows)
            write_summary(rows)
        if "cellnest" in methods:
            run_single_method(
                "cellnest",
                "pyx-lr-cellnest",
                [candidates["pilot50k"], candidates["smoke20k"], candidates["subset10k"], candidates["subset5k"]],
                rows,
                {"CELLNEST_NUM_EPOCH": "25", "CELLNEST_HIDDEN": "64"},
            )
            write_summary(rows)
        if "scild" in methods:
            run_single_method(
                "scild",
                "pyx-lr-scild",
                [candidates["subset5k"], candidates["subset10k"], candidates["smoke20k"]],
                rows,
                {"SCILD_NITER_MAX": "20", "SCILD_NEIGHBOR_K": "5"},
                ascending_keep_max=True,
            )
            write_summary(rows)
        _write_json(status_path, {"status": "complete", "methods": methods, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except Exception as exc:
        write_summary(rows)
        _write_json(status_path, {"status": "failed", "error": str(exc), "methods": methods, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        raise


if __name__ == "__main__":
    main()
