from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def read_table(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(path, sep="\t", compression=compression)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, sep="\t", index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def parse_chunk_index(path: Path) -> int:
    match = re.search(r"chunk_(\d+)_of_(\d+)", str(path))
    if not match:
        return 10**9
    return int(match.group(1))


def find_chunk_outputs(chunk_root: Path, method: str, pattern: str) -> list[Path]:
    paths = []
    for chunk_dir in sorted(chunk_root.glob(pattern), key=parse_chunk_index):
        if not chunk_dir.is_dir():
            continue
        candidates = sorted(chunk_dir.glob(f"{method}_standardized.tsv*"))
        if candidates:
            paths.append(candidates[0])
    return paths


def validate_chunk(path: Path, method: str, min_rows: int) -> dict[str, Any]:
    header = read_table(path)
    missing = [column for column in STANDARDIZED_RESULT_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing standardized columns: {missing}")
    if len(header) < min_rows:
        raise ValueError(f"{path} has {len(header)} rows; expected at least {min_rows}")
    methods = sorted(str(value) for value in header["method"].dropna().unique())
    if methods and methods != [method]:
        raise ValueError(f"{path} contains method values {methods}; expected {method}")
    return {"path": str(path), "n_rows": int(len(header)), "n_columns": int(len(header.columns))}


def rerank(combined: pd.DataFrame) -> pd.DataFrame:
    score = pd.to_numeric(combined["score_raw"], errors="coerce").to_numpy(dtype=float)
    score[~np.isfinite(score)] = -np.inf
    pvalue = pd.to_numeric(combined["fdr_or_pvalue"], errors="coerce").to_numpy(dtype=float)
    pvalue_sort = np.where(np.isfinite(pvalue), pvalue, np.inf)
    order = np.lexsort((pvalue_sort, -score))
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    combined["rank_within_method"] = ranks
    if len(combined) == 1:
        combined["rank_fraction"] = 1.0
    else:
        combined["rank_fraction"] = 1.0 - ((ranks - 1.0) / float(len(combined) - 1))
    combined["score_std"] = combined["rank_fraction"].astype(float)
    combined["score_raw"] = pd.to_numeric(combined["score_raw"], errors="coerce")
    return combined.sort_values("rank_within_method", kind="mergesort").reset_index(drop=True)


def aggregate_chunks(
    *,
    method: str,
    chunk_root: Path,
    output_path: Path,
    expected_chunks: int | None,
    chunk_pattern: str,
    min_rows: int,
    summary_path: Path,
    allow_existing_output: bool,
) -> dict[str, Any]:
    chunk_paths = find_chunk_outputs(chunk_root, method, chunk_pattern)
    if expected_chunks is not None and len(chunk_paths) != expected_chunks:
        raise RuntimeError(f"Expected {expected_chunks} chunks for {method}, found {len(chunk_paths)} under {chunk_root}")
    if not chunk_paths:
        raise RuntimeError(f"No chunk outputs found for {method} under {chunk_root}")
    if output_path.exists() and not allow_existing_output:
        raise FileExistsError(f"Output already exists: {output_path}")

    chunk_summaries = [validate_chunk(path, method, min_rows) for path in chunk_paths]
    frames = [read_table(path) for path in chunk_paths]
    combined = pd.concat(frames, ignore_index=True)
    extra_columns = [column for column in combined.columns if column not in STANDARDIZED_RESULT_COLUMNS]
    combined = combined.loc[:, STANDARDIZED_RESULT_COLUMNS + extra_columns]
    combined = rerank(combined)
    write_table(combined, output_path)

    payload = {
        "method": method,
        "status": "success",
        "stage": "chunked_standardized_aggregate",
        "chunk_root": str(chunk_root),
        "output_path": str(output_path),
        "expected_chunks": expected_chunks,
        "observed_chunks": len(chunk_paths),
        "n_rows": int(len(combined)),
        "chunk_outputs": chunk_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate chunked standardized LR benchmark outputs.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--chunk-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--expected-chunks", type=int, default=None)
    parser.add_argument("--chunk-pattern", default="chunk_*_of_*")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_name("aggregate_summary.json")
    payload = aggregate_chunks(
        method=args.method,
        chunk_root=Path(args.chunk_root),
        output_path=output_path,
        expected_chunks=args.expected_chunks,
        chunk_pattern=args.chunk_pattern,
        min_rows=args.min_rows,
        summary_path=summary_path,
        allow_existing_output=args.allow_existing_output,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
