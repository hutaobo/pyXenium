from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pyXenium.benchmarking import (
    STANDARDIZED_RESULT_COLUMNS,
    aggregate_standardized_results,
    build_canonical_rank_matrix,
    compute_canonical_recovery,
    compute_novelty_support,
    compute_pathway_relevance,
    compute_robustness,
    compute_spatial_coherence,
    render_atera_lr_benchmark_report,
    resolve_layout,
    score_biological_performance,
)


FIRST_WAVE_METHODS = ("pyxenium", "squidpy", "liana", "cellchat", "commot")


def _repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _parse_elapsed_seconds(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None


def _parse_resource_log(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resource_log": str(path),
        "resource_log_exists": path.exists(),
        "elapsed_seconds_resource": None,
        "peak_memory_gb_resource": None,
        "exit_status_resource": None,
    }
    if not path.exists():
        return payload
    text = path.read_text(encoding="utf-8", errors="replace")
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?:\s*(.+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    exit_status = re.search(r"Exit status:\s*(-?\d+)", text)
    if elapsed:
        payload["elapsed_seconds_resource"] = _parse_elapsed_seconds(elapsed.group(1))
    if rss:
        payload["peak_memory_gb_resource"] = int(rss.group(1)) / 1024 / 1024
    if exit_status:
        payload["exit_status_resource"] = int(exit_status.group(1))
    return payload


def _success_standardized_path(method_dir: Path) -> Path | None:
    summary = _read_json(method_dir / "run_summary.json")
    if summary.get("status") == "failed":
        return None
    candidates = sorted(path for path in method_dir.glob("*standardized.tsv") if path.is_file())
    if not candidates:
        candidates = sorted(path for path in method_dir.glob("*_standardized.tsv") if path.is_file())
    return candidates[0] if candidates else None


def discover_full_common_inputs(collected_root: Path) -> list[Path]:
    full_common = collected_root / "full_common"
    inputs: list[Path] = []
    for method in FIRST_WAVE_METHODS:
        path = full_common / method
        if not path.exists():
            continue
        standardized = _success_standardized_path(path)
        if standardized is not None:
            inputs.append(standardized)
    return inputs


def build_schema_validation(result_paths: list[Path], combined: pd.DataFrame) -> dict[str, Any]:
    missing_by_file: dict[str, list[str]] = {}
    rows_by_method: dict[str, int] = {}
    for path in result_paths:
        header = pd.read_csv(path, sep="\t", nrows=0)
        missing = [col for col in STANDARDIZED_RESULT_COLUMNS if col not in header.columns]
        if missing:
            missing_by_file[str(path)] = missing
    if not combined.empty:
        rows_by_method = {str(k): int(v) for k, v in combined.groupby("method").size().to_dict().items()}
    return {
        "status": "passed" if not missing_by_file else "failed",
        "n_inputs": len(result_paths),
        "n_rows": int(len(combined)),
        "n_methods": int(combined["method"].nunique()) if "method" in combined else 0,
        "rows_by_method": rows_by_method,
        "missing_columns_by_file": missing_by_file,
        "required_columns": STANDARDIZED_RESULT_COLUMNS,
    }


def build_run_status(layout, *, cellchat_status: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    collected = layout.runs_dir / "a100_collected" / "full_common"
    for method in ("pyxenium", "squidpy", "liana", "cellchat"):
        summary_path = collected / method / "run_summary.json"
        summary = _read_json(summary_path)
        if not summary and method == "cellchat":
            summary = {"method": method, "status": cellchat_status}
        elif summary.get("status") == "failed" and method == "cellchat" and cellchat_status == "running":
            # A new CellChat run can be active while the last collected summary
            # still describes an older failed attempt. Do not let stale failure
            # resources leak into the live run-status table.
            summary = {"method": method, "status": "running", "phase": "full", "database_mode": "common-db"}
        if not summary:
            continue
        status = summary.get("status", "success" if summary.get("standardized_tsv") else "unknown")
        resource = (
            {"elapsed_seconds_resource": None, "peak_memory_gb_resource": None, "exit_status_resource": None}
            if status == "running"
            else _parse_resource_log(layout.logs_dir / "a100_collected" / f"full_common_db_{method}.resource.log")
        )
        rows.append(
            {
                "method": summary.get("method", method),
                "phase": summary.get("phase", "full"),
                "database_mode": summary.get("database_mode", "common-db"),
                "status": status,
                "n_rows": summary.get("n_rows"),
                "elapsed_seconds": summary.get("elapsed_seconds", resource.get("elapsed_seconds_resource")),
                "peak_memory_gb": resource.get("peak_memory_gb_resource"),
                "exit_code": resource.get("exit_status_resource"),
                "standardized_tsv": summary.get("standardized_tsv"),
                "summary_json": str(summary_path),
                "error": summary.get("error", summary.get("reason")),
            }
        )
    commot_card = layout.reports_dir / "method_cards" / "commot_full_common.md"
    rows.append(
        {
            "method": "commot",
            "phase": "full",
            "database_mode": "common-db",
            "status": "appendix_gated",
            "n_rows": None,
            "elapsed_seconds": None,
            "peak_memory_gb": None,
            "exit_code": None,
            "standardized_tsv": None,
            "summary_json": str(commot_card),
            "error": "Full common-db gated by smoke runtime extrapolation.",
        }
    )
    return pd.DataFrame(rows)


def write_commot_gating(layout) -> dict[str, Any]:
    smoke_summary = _read_json(layout.runs_dir / "a100_collected" / "debug" / "commot_smoke_100" / "run_summary.json")
    smoke_params = _read_json(layout.runs_dir / "a100_collected" / "debug" / "commot_smoke_100" / "params.json")
    manifest = _read_json(layout.data_dir / "input_manifest.json")
    full_cells = int(manifest.get("full_n_cells") or 170057)
    smoke_cells = int(manifest.get("smoke_n_cells") or 20000)
    full_pairs = int((manifest.get("common_db") or {}).get("n_pairs") or 3299)
    smoke_pairs = int(smoke_params.get("lr_pairs") or smoke_params.get("max_lr_pairs") or 100)
    smoke_elapsed = float(smoke_summary.get("elapsed_seconds") or 0.0)
    estimate = smoke_elapsed * (full_cells / smoke_cells) * (full_pairs / smoke_pairs) if smoke_elapsed else math.nan
    payload = {
        "method": "commot",
        "status": "appendix_gated",
        "database_mode": "common-db",
        "phase": "full",
        "smoke_elapsed_seconds": smoke_elapsed,
        "smoke_cells": smoke_cells,
        "smoke_lr_pairs": smoke_pairs,
        "full_cells": full_cells,
        "full_lr_pairs": full_pairs,
        "linear_runtime_estimate_seconds": estimate,
        "linear_runtime_estimate_hours": estimate / 3600 if not math.isnan(estimate) else None,
        "resource_limit_hours": 6,
        "decision": "Do not promote COMMOT full 170k x common-db run into the first-wave main lane; keep smoke result and full gating evidence in appendix.",
        "reproduce_smoke": smoke_params.get("runner") or "See commot smoke run_summary.json and params.json.",
    }
    results_path = layout.results_dir / "commot_full_common_gating.json"
    _write_json(results_path, payload)
    card_dir = layout.reports_dir / "method_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    card = card_dir / "commot_full_common.md"
    lines = [
        "# Method Card: COMMOT full common-db",
        "",
        "- Status: `appendix_gated`",
        f"- Smoke benchmark: `{smoke_cells}` cells x `{smoke_pairs}` LR pairs in `{smoke_elapsed:.1f}` seconds.",
        f"- Full target: `{full_cells}` cells x `{full_pairs}` LR pairs.",
        f"- Linear lower-bound runtime estimate: `{payload['linear_runtime_estimate_hours']:.1f}` hours.",
        "- Decision: keep COMMOT in appendix unless a bounded full strategy is explicitly requested.",
        "- Reason: the estimate is far above the 6 hour promotion gate, and COMMOT communication tensors are expected to scale worse than this simple lower bound.",
        "",
    ]
    card.write_text("\n".join(lines), encoding="utf-8")
    payload["method_card"] = str(card)
    return payload


def write_tables(layout, *, combined: pd.DataFrame, run_status: pd.DataFrame) -> dict[str, str]:
    canonical_detail, canonical_summary = compute_canonical_recovery(
        combined,
        canonical_axes=layout.config_dir / "canonical_axes.yaml",
    )
    pathway_detail, pathway_summary = compute_pathway_relevance(
        combined,
        pathway_config=layout.config_dir / "pathways.yaml",
    )
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
    engineering = run_status.copy()

    outputs = {
        "canonical_detail": layout.results_dir / "a100_full_common_canonical_detail.tsv",
        "canonical_summary": layout.results_dir / "a100_full_common_canonical_summary.tsv",
        "canonical_rank_matrix": layout.results_dir / "a100_full_common_canonical_rank_matrix.tsv",
        "pathway_detail": layout.results_dir / "a100_full_common_pathway_detail.tsv",
        "pathway_summary": layout.results_dir / "a100_full_common_pathway_summary.tsv",
        "spatial_summary": layout.results_dir / "a100_full_common_spatial_coherence.tsv",
        "novelty_detail": layout.results_dir / "a100_full_common_novelty_top.tsv",
        "novelty_summary": layout.results_dir / "a100_full_common_novelty_summary.tsv",
        "robustness_summary": layout.results_dir / "a100_full_common_robustness_summary.tsv",
        "biology_scoreboard": layout.results_dir / "a100_full_common_biology_scoreboard.tsv",
        "engineering_scoreboard": layout.results_dir / "a100_full_common_engineering_scoreboard.tsv",
        "run_status": layout.results_dir / "a100_full_common_run_status.tsv",
    }
    table_map = {
        "canonical_detail": canonical_detail,
        "canonical_summary": canonical_summary,
        "canonical_rank_matrix": rank_matrix,
        "pathway_detail": pathway_detail,
        "pathway_summary": pathway_summary,
        "spatial_summary": spatial_summary,
        "novelty_detail": novelty_detail,
        "novelty_summary": novelty_summary,
        "robustness_summary": robustness_summary,
        "biology_scoreboard": biology_summary,
        "engineering_scoreboard": engineering,
        "run_status": run_status,
    }
    for key, table in table_map.items():
        outputs[key].parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(outputs[key], sep="\t", index=False)

    markdown = render_atera_lr_benchmark_report(
        combined_results=combined,
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        biology_summary=biology_summary,
        benchmark_root=layout.root,
        run_status=run_status,
        engineering_summary=engineering,
        canonical_detail=canonical_detail,
        a100_resource_summary=run_status,
    )
    report_path = layout.reports_dir / "a100_full_common_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    outputs["report"] = report_path
    return {key: str(path) for key, path in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize collected A100 full common-db LR benchmark outputs.")
    parser.add_argument("--benchmark-root", default="benchmarking/lr_2026_atera")
    parser.add_argument("--cellchat-status", default="running", choices=["running", "failed", "pending"])
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    repo_root = _repo_root(Path.cwd())
    layout = resolve_layout(repo_root=repo_root, relative_root=args.benchmark_root)
    collected_root = layout.runs_dir / "a100_collected"
    result_paths = discover_full_common_inputs(collected_root)
    if not result_paths:
        raise SystemExit(f"No successful A100 full common standardized TSVs found under {collected_root}")

    combined_path = layout.results_dir / "a100_full_common_combined.tsv"
    combined = aggregate_standardized_results(result_paths, output_path=combined_path)
    validation = build_schema_validation(result_paths, combined)
    validation_path = layout.results_dir / "a100_full_common_schema_validation.json"
    _write_json(validation_path, validation)

    run_status = build_run_status(layout, cellchat_status=args.cellchat_status)
    commot_gating = write_commot_gating(layout)
    table_outputs = write_tables(layout, combined=combined, run_status=run_status)

    payload = {
        "combined_results": str(combined_path),
        "schema_validation": str(validation_path),
        "n_rows": int(len(combined)),
        "methods": sorted(combined["method"].dropna().unique().tolist()),
        "inputs": [str(path) for path in result_paths],
        "commot_gating": commot_gating,
        "outputs": table_outputs,
    }
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
