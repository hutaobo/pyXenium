from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/taobo.hu/pyxenium_lr_benchmark_2026-04")
TAG = "final_closeout_20260511"
CONDA = Path("/home/taobo.hu/miniconda3/bin/conda")
MAX_ACCEPTED_RSS_GB = 900.0
SPATALK_FULL_MIN_AVAILABLE_GB = 500

STANDARDIZED_RESULT_COLUMNS = [
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
class MethodRuntime:
    env_name: str
    runner: str
    r_libs_user: str | None
    extra_args: tuple[str, ...]


RUNTIMES = {
    "niches": MethodRuntime(
        env_name="r-lr-niches-conda",
        runner="runners/r/run_niches.R",
        r_libs_user="envs/r_libs/niches_conda_bisrna2",
        extra_args=("--k", "4"),
    ),
    "spatalk": MethodRuntime(
        env_name="r-lr-spatalk-conda2",
        runner="runners/r/run_spatalk.R",
        r_libs_user=None,
        extra_args=("--n-cores", "16", "--min-pairs", "5", "--per-num", "10", "--pvalue", "1", "--co-exp-ratio", "0.1"),
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, sep="\t", compression=compression, nrows=nrows)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, sep="\t", index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def parse_resource_log(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"resource_log": str(path), "peak_rss_gb": None, "elapsed_seconds_resource": None, "exit_status_resource": None}
    if not path.exists():
        return payload
    text = path.read_text(encoding="utf-8", errors="replace")
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if rss:
        payload["peak_rss_gb"] = int(rss.group(1)) / 1024 / 1024
    exit_status = re.search(r"Exit status:\s*(-?\d+)", text)
    if exit_status:
        payload["exit_status_resource"] = int(exit_status.group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?:\s*(.+)", text)
    if elapsed:
        payload["elapsed_seconds_resource"] = parse_elapsed(elapsed.group(1).strip())
    return payload


def parse_elapsed(value: str) -> float | None:
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return None


def available_memory_gb() -> int | None:
    try:
        completed = subprocess.run(["free", "-g"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("Mem:") and len(parts) >= 7:
            try:
                return int(parts[6])
            except ValueError:
                return None
    return None


def find_standardized(output_dir: Path, method: str) -> Path | None:
    for suffix in (".tsv.gz", ".tsv"):
        path = output_dir / f"{method}_standardized{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def validate_standardized(path: Path, method: str) -> int:
    df = read_tsv(path)
    missing = [column for column in STANDARDIZED_RESULT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing standardized columns: {missing}")
    if df.empty:
        raise ValueError(f"{path} is empty")
    methods = sorted(str(value) for value in df["method"].dropna().unique())
    if methods and methods != [method]:
        raise ValueError(f"{path} contains method values {methods}; expected {method}")
    return int(len(df))


def rerank(combined: pd.DataFrame) -> pd.DataFrame:
    score = pd.to_numeric(combined["score_raw"], errors="coerce").to_numpy(dtype=float)
    score[~np.isfinite(score)] = -np.inf
    pvalue = pd.to_numeric(combined["fdr_or_pvalue"], errors="coerce").to_numpy(dtype=float)
    pvalue_sort = np.where(np.isfinite(pvalue), pvalue, np.inf)
    order = np.lexsort((pvalue_sort, -score))
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    combined["rank_within_method"] = ranks
    combined["rank_fraction"] = 1.0 if len(combined) == 1 else 1.0 - ((ranks - 1.0) / float(len(combined) - 1))
    combined["score_std"] = combined["rank_fraction"].astype(float)
    return combined.sort_values("rank_within_method", kind="mergesort").reset_index(drop=True)


def aggregate_paths(method: str, paths: list[Path], output_path: Path) -> dict[str, Any]:
    if not paths:
        raise RuntimeError(f"No chunk paths to aggregate for {method}")
    chunks = []
    frames = []
    for path in paths:
        n_rows = validate_standardized(path, method)
        chunks.append({"path": str(path), "n_rows": n_rows})
        frames.append(read_tsv(path))
    combined = pd.concat(frames, ignore_index=True)
    extra_columns = [column for column in combined.columns if column not in STANDARDIZED_RESULT_COLUMNS]
    combined = rerank(combined.loc[:, STANDARDIZED_RESULT_COLUMNS + extra_columns])
    write_tsv(combined, output_path)
    payload = {"method": method, "status": "success", "standardized_tsv_gz": str(output_path), "n_rows": int(len(combined)), "chunks": chunks}
    write_json(output_path.with_name("aggregate_summary.json"), payload)
    return payload


def load_manifest(root: Path) -> dict[str, Any]:
    return read_json(root / "data" / "input_manifest.json")


def load_lr_database(root: Path) -> pd.DataFrame:
    manifest = load_manifest(root)
    path = Path(manifest["lr_db_common_tsv"])
    lr = pd.read_csv(path, sep="\t")
    if not {"ligand", "receptor"}.issubset(lr.columns):
        raise RuntimeError(f"LR database missing ligand/receptor columns: {path}")
    return lr.drop_duplicates(subset=["ligand", "receptor"]).reset_index(drop=True)


def write_chunk_manifest(root: Path, out_dir: Path, lr: pd.DataFrame, start: int, end: int) -> Path:
    manifest = load_manifest(root)
    manifest_dir = out_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    lr_path = manifest_dir / f"lr_{start:04d}_{end:04d}.tsv"
    lr.iloc[start:end].to_csv(lr_path, sep="\t", index=False)
    manifest["lr_db_common_tsv"] = str(lr_path)
    manifest_path = manifest_dir / "input_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def command_env(root: Path, runtime: MethodRuntime) -> dict[str, str]:
    env = os.environ.copy()
    env["TMPDIR"] = str(root / "tmp")
    env["PYX_LR_BENCHMARK_ROOT"] = str(root)
    env["OMP_NUM_THREADS"] = "16"
    env["MKL_NUM_THREADS"] = "16"
    if runtime.r_libs_user:
        env["R_LIBS_USER"] = str(root / runtime.r_libs_user)
    return env


def run_chunk(
    *,
    root: Path,
    method: str,
    manifest_path: Path,
    output_dir: Path,
    max_cells: int | None,
    lr_start: int,
    lr_end: int,
) -> dict[str, Any]:
    runtime = RUNTIMES[method]
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = find_standardized(output_dir, method)
    if existing:
        n_rows = validate_standardized(existing, method)
        payload = {
            "method": method,
            "status": "success",
            "skipped_existing": True,
            "standardized_tsv_gz": str(existing),
            "n_rows": n_rows,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "updated_at": utc_now(),
        }
        write_json(output_dir / "rescue_chunk_summary.json", payload)
        return payload

    stdout = output_dir / "rescue_stdout.log"
    stderr = output_dir / "rescue_stderr.log"
    resource = output_dir / "rescue_resource.log"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(resource),
        str(CONDA),
        "run",
        "-n",
        runtime.env_name,
        "Rscript",
        str(root / runtime.runner),
        "--input-manifest",
        str(manifest_path),
        "--database-mode",
        "common",
        "--phase",
        "full",
        "--output-dir",
        str(output_dir),
        *runtime.extra_args,
    ]
    if max_cells is not None:
        command.extend(["--max-cells", str(max_cells)])

    started = time.time()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(command, stdout=out, stderr=err, env=command_env(root, runtime), check=False)
    elapsed = time.time() - started
    resource_payload = parse_resource_log(resource)
    standardized = find_standardized(output_dir, method)
    peak = resource_payload.get("peak_rss_gb")
    status = "success"
    reason = ""
    n_rows = None
    if completed.returncode != 0:
        status = "failed"
        reason = f"Command exited with {completed.returncode}"
    elif not standardized:
        status = "failed"
        reason = "Command exited successfully but did not create standardized output."
    elif peak is not None and float(peak) > MAX_ACCEPTED_RSS_GB:
        status = "resource_exceeded"
        reason = f"Peak RSS {peak:.3f} GB exceeded {MAX_ACCEPTED_RSS_GB:.1f} GB."
    else:
        n_rows = validate_standardized(standardized, method)

    payload = {
        "method": method,
        "status": status,
        "reason": reason,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "lr_start": lr_start,
        "lr_end": lr_end,
        "max_cells": max_cells,
        "standardized_tsv_gz": str(standardized) if standardized else "",
        "n_rows": n_rows,
        "stdout": str(stdout),
        "stderr": str(stderr),
        **resource_payload,
        "command": command,
        "updated_at": utc_now(),
    }
    write_json(output_dir / "rescue_chunk_summary.json", payload)
    if status != "success":
        write_method_card(output_dir, method, status, reason, payload)
    return payload


def write_method_card(output_dir: Path, method: str, status: str, reason: str, payload: dict[str, Any]) -> None:
    lines = [
        f"# Method Card: {method}",
        "",
        f"- Status: `{status}`",
        f"- Reason: `{reason}`",
        f"- Evidence: `{output_dir / 'rescue_chunk_summary.json'}`",
        "",
    ]
    (output_dir / "method_card.md").write_text("\n".join(lines), encoding="utf-8")


def run_window(
    *,
    root: Path,
    method: str,
    lr: pd.DataFrame,
    base_dir: Path,
    scale_tag: str,
    max_cells: int | None,
    start: int,
    end: int,
    label: str,
) -> dict[str, Any]:
    output_dir = base_dir / scale_tag / label
    manifest_path = write_chunk_manifest(root, output_dir, lr, start, end)
    return run_chunk(root=root, method=method, manifest_path=manifest_path, output_dir=output_dir, max_cells=max_cells, lr_start=start, lr_end=end)


def run_chunked_scale(
    *,
    root: Path,
    method: str,
    lr: pd.DataFrame,
    base_dir: Path,
    scale_tag: str,
    max_cells: int | None,
    chunk_size: int,
    fallback_chunk_size: int | None = None,
) -> dict[str, Any]:
    scale_dir = base_dir / scale_tag
    status_path = scale_dir / "scale_status.json"
    if (scale_dir / f"{method}_standardized.tsv.gz").exists():
        existing = scale_dir / f"{method}_standardized.tsv.gz"
        return {"method": method, "status": "success", "scale_tag": scale_tag, "standardized_tsv_gz": str(existing), "skipped_existing": True}

    outputs: list[Path] = []
    chunks: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    total = len(lr)
    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        end = min(total, start + chunk_size)
        label = f"chunk_{chunk_index:03d}_{start:04d}_{end:04d}"
        payload = run_window(root=root, method=method, lr=lr, base_dir=base_dir, scale_tag=scale_tag, max_cells=max_cells, start=start, end=end, label=label)
        chunks.append(payload)
        if payload.get("status") == "success":
            outputs.append(Path(payload["standardized_tsv_gz"]))
            write_json(status_path, {"method": method, "status": "running", "scale_tag": scale_tag, "completed_chunks": len(outputs), "updated_at": utc_now()})
            continue

        if fallback_chunk_size and fallback_chunk_size < (end - start):
            fallback_ok = True
            for sub_index, sub_start in enumerate(range(start, end, fallback_chunk_size)):
                sub_end = min(end, sub_start + fallback_chunk_size)
                sub_label = f"fallback_{chunk_index:03d}_{sub_index:02d}_{sub_start:04d}_{sub_end:04d}"
                sub_payload = run_window(
                    root=root,
                    method=method,
                    lr=lr,
                    base_dir=base_dir,
                    scale_tag=scale_tag,
                    max_cells=max_cells,
                    start=sub_start,
                    end=sub_end,
                    label=sub_label,
                )
                chunks.append(sub_payload)
                if sub_payload.get("status") == "success":
                    outputs.append(Path(sub_payload["standardized_tsv_gz"]))
                else:
                    fallback_ok = False
                    failure = sub_payload
                    break
            if fallback_ok:
                write_json(status_path, {"method": method, "status": "running", "scale_tag": scale_tag, "completed_chunks": len(outputs), "updated_at": utc_now()})
                continue
        failure = payload
        break

    if failure is not None:
        summary = {
            "method": method,
            "status": failure.get("status", "failed"),
            "scale_tag": scale_tag,
            "reason": failure.get("reason", "chunk failed"),
            "failed_chunk": failure,
            "chunks": chunks,
            "updated_at": utc_now(),
        }
        write_json(status_path, summary)
        write_method_card(scale_dir, method, summary["status"], summary["reason"], summary)
        return summary

    aggregate = aggregate_paths(method, outputs, scale_dir / f"{method}_standardized.tsv.gz")
    peaks = [float(chunk["peak_rss_gb"]) for chunk in chunks if chunk.get("peak_rss_gb") not in (None, "")]
    elapsed = [float(chunk["elapsed_seconds"]) for chunk in chunks if chunk.get("elapsed_seconds") not in (None, "")]
    summary = {
        "method": method,
        "status": "success",
        "scale_tag": scale_tag,
        "subset_n_cells": max_cells or 170057,
        "standardized_tsv_gz": aggregate["standardized_tsv_gz"],
        "n_rows": aggregate["n_rows"],
        "peak_rss_gb": max(peaks) if peaks else None,
        "elapsed_seconds": sum(elapsed) if elapsed else None,
        "chunks": chunks,
        "updated_at": utc_now(),
    }
    write_json(status_path, summary)
    return summary


def run_niches(root: Path, tag: str) -> dict[str, Any]:
    method = "niches"
    lr = load_lr_database(root)
    base = root / "runs" / tag / "a100_rescue" / method
    summary_path = base / "rescue_summary.json"
    existing_summary = read_json(summary_path)
    if existing_summary.get("status") in {"success_full", "success_bounded", "method_api_failure", "failed"}:
        return existing_summary
    summary = {"method": method, "status": "running", "started_at": utc_now(), "base_dir": str(base)}
    write_json(summary_path, summary)

    full = run_chunked_scale(root=root, method=method, lr=lr, base_dir=base, scale_tag="full170k", max_cells=None, chunk_size=100, fallback_chunk_size=50)
    if full.get("status") == "success":
        summary.update({"status": "success_full", "full_success": full, "updated_at": utc_now()})
        write_json(summary_path, summary)
        return summary

    summary["full_failure_reason"] = full.get("reason", "full170k failed")
    for scale_tag, max_cells in (("pilot50k", 50000), ("smoke20k", 20000), ("subset10k", 10000), ("subset5k", 5000)):
        bounded = run_chunked_scale(root=root, method=method, lr=lr, base_dir=base, scale_tag=scale_tag, max_cells=max_cells, chunk_size=100, fallback_chunk_size=50)
        if bounded.get("status") == "success":
            summary.update({"status": "success_bounded", "full_failure": full, "bounded_success": bounded, "updated_at": utc_now()})
            write_json(summary_path, summary)
            return summary
        summary[f"{scale_tag}_failure"] = bounded

    summary.update({"status": "failed", "reason": "NICHES failed full and all bounded rescue scales.", "full_failure": full, "updated_at": utc_now()})
    write_json(summary_path, summary)
    write_method_card(base, method, summary["status"], summary["reason"], summary)
    return summary


def run_spatalk(root: Path, tag: str) -> dict[str, Any]:
    method = "spatalk"
    lr = load_lr_database(root)
    base = root / "runs" / tag / "a100_rescue" / method
    summary_path = base / "rescue_summary.json"
    existing_summary = read_json(summary_path)
    if existing_summary.get("status") in {"success_full", "success_bounded", "method_api_failure", "failed"}:
        return existing_summary
    summary = {"method": method, "status": "running", "started_at": utc_now(), "base_dir": str(base)}
    write_json(summary_path, summary)

    smoke = run_chunked_scale(root=root, method=method, lr=lr, base_dir=base, scale_tag="smoke20k", max_cells=20000, chunk_size=50)
    if smoke.get("status") != "success":
        summary.update({"status": "method_api_failure", "reason": smoke.get("reason", "20k chunked smoke failed"), "smoke20k_failure": smoke, "updated_at": utc_now()})
        write_json(summary_path, summary)
        write_method_card(base, method, summary["status"], summary["reason"], summary)
        return summary

    pilot = run_chunked_scale(root=root, method=method, lr=lr, base_dir=base, scale_tag="pilot50k", max_cells=50000, chunk_size=50)
    if pilot.get("status") != "success":
        summary.update({"status": "success_bounded", "bounded_success": smoke, "full_failure_reason": pilot.get("reason", "50k failed"), "pilot50k_failure": pilot, "updated_at": utc_now()})
        write_json(summary_path, summary)
        return summary

    available_gb = available_memory_gb()
    if available_gb is not None and available_gb < SPATALK_FULL_MIN_AVAILABLE_GB:
        reason = f"Skipped full170k because A100 available memory was {available_gb} GiB, below {SPATALK_FULL_MIN_AVAILABLE_GB} GiB guard."
        summary.update({"status": "success_bounded", "bounded_success": pilot, "full_failure_reason": reason, "memory_guard_available_gb": available_gb, "updated_at": utc_now()})
        write_json(summary_path, summary)
        return summary

    full = run_chunked_scale(root=root, method=method, lr=lr, base_dir=base, scale_tag="full170k", max_cells=None, chunk_size=50)
    if full.get("status") == "success":
        summary.update({"status": "success_full", "full_success": full, "updated_at": utc_now()})
    else:
        summary.update({"status": "success_bounded", "bounded_success": pilot, "full_failure_reason": full.get("reason", "170k failed"), "full_failure": full, "updated_at": utc_now()})
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run serial A100 rescue attempts for R LR methods.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--methods", default="niches,spatalk")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    payload: dict[str, Any] = {"tag": args.tag, "root": str(root), "methods": methods, "started_at": utc_now(), "dry_run": args.dry_run}
    if args.dry_run:
        lr = load_lr_database(root)
        payload["n_lr_pairs"] = int(len(lr))
        payload["planned"] = {
            "niches": {"full_chunk_size": 100, "fallback_chunk_size": 50, "bounded_scales": ["pilot50k", "smoke20k", "subset10k", "subset5k"]},
            "spatalk": {"chunk_size": 50, "scales": ["smoke20k", "pilot50k", "full170k"], "full170k_min_available_gb": SPATALK_FULL_MIN_AVAILABLE_GB},
        }
        print(json.dumps(payload, indent=2))
        return

    results = []
    if "niches" in methods:
        results.append(run_niches(root, args.tag))
    if "spatalk" in methods:
        results.append(run_spatalk(root, args.tag))
    payload["finished_at"] = utc_now()
    payload["results"] = results
    write_json(root / "runs" / args.tag / "a100_rescue" / "supervisor_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
