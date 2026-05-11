from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

ALL_METHODS = [
    "pyxenium",
    "squidpy",
    "liana",
    "spatialdm",
    "stlearn",
    "cellphonedb",
    "laris",
    "cellchat",
    "commot",
    "giotto",
    "spatalk",
    "niches",
    "cellnest",
    "cellagentchat",
    "scild",
]

A100_AUTHORITATIVE_FULL = {
    "pyxenium": "runs/full_common/pyxenium/pyxenium_standardized.tsv",
    "squidpy": "runs/full_common/squidpy/squidpy_standardized.tsv",
    "liana": "runs/full_common/liana/liana_standardized.tsv",
    "spatialdm": "runs/full_common/spatialdm/spatialdm_standardized.tsv.gz",
    "stlearn": "runs/full_common/stlearn/stlearn_standardized.tsv.gz",
    "cellphonedb": "runs/full_common/cellphonedb/cellphonedb_standardized.tsv",
    "laris": "runs/full_common/laris/laris_standardized.tsv.gz",
    "cellchat": "runs/full_common/cellchat_conda_full/cellchat_standardized.tsv.gz",
}

KNOWN_FULL_FAILURES = {
    "giotto": {
        "status": "resource_exceeded_matrix_limit",
        "reason": "Full 170k Giotto failed with R Matrix/TsparseMatrix 2^31-1 index limit.",
        "evidence": "runs/full_common/giotto_real/run_summary.json",
    },
    "cellnest": {
        "status": "resource_exceeded",
        "reason": "Full 170k CellNEST failed with SIGKILL/-9 in run_CellNEST.py.",
        "evidence": "runs/full_common/cellnest/run_summary.json",
    },
    "cellagentchat": {
        "status": "resource_exceeded",
        "reason": "Full 170k serial chunk retry exceeded resources; 50k bounded rescue succeeded.",
        "evidence": "runs/bounded_full_retry/cellagentchat/full170k/chunk_000_of_016/run_summary.json",
    },
    "scild": {
        "status": "method_not_scalable_or_adapter_issue",
        "reason": "SCILD bounded rescue failed at 5k/10k/20k and produced no usable standardized output.",
        "evidence": "results/bounded_rescue_20260430/bounded_rescue_summary.tsv",
    },
}


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_standardized(path: Path, method: str) -> int:
    header = read_tsv(path, nrows=0)
    missing = [column for column in STANDARDIZED_RESULT_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"{path} missing standardized columns: {missing}")
    df = read_tsv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    methods = sorted(str(value) for value in df["method"].dropna().unique())
    if methods and methods != [method]:
        raise ValueError(f"{path} contains method values {methods}; expected {method}")
    return int(len(df))


def collect_full_inputs(root: Path, commot_path: Path | None) -> tuple[list[Path], list[dict[str, Any]]]:
    inputs: list[Path] = []
    rows: list[dict[str, Any]] = []
    for method, rel_path in A100_AUTHORITATIVE_FULL.items():
        path = root / rel_path
        if not path.exists():
            rows.append({"method": method, "full_status": "missing", "source": "a100", "standardized_tsv": str(path), "n_rows": None})
            continue
        n_rows = validate_standardized(path, method)
        inputs.append(path)
        rows.append({"method": method, "full_status": "success", "source": "a100", "standardized_tsv": str(path), "n_rows": n_rows})

    if commot_path and commot_path.exists():
        n_rows = validate_standardized(commot_path, "commot")
        inputs.append(commot_path)
        rows.append({"method": "commot", "full_status": "success", "source": "pdc_aggregate", "standardized_tsv": str(commot_path), "n_rows": n_rows})
    else:
        rows.append(
            {
                "method": "commot",
                "full_status": "pending_aggregate",
                "source": "pdc",
                "standardized_tsv": str(commot_path) if commot_path else "",
                "n_rows": None,
            }
        )
    return inputs, rows


def discover_rescue_status(root: Path, tag: str, method: str) -> dict[str, Any]:
    base = root / "runs" / tag / "a100_rescue" / method
    summary = base / "rescue_summary.json"
    if summary.exists():
        return read_json(summary)
    return {"method": method, "status": "not_started", "base_dir": str(base)}


def build_bounded_table(root: Path, tag: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bounded_path = root / "results" / "bounded_rescue_20260430" / "bounded_rescue_summary.tsv"
    if bounded_path.exists():
        existing = read_tsv(bounded_path)
        rows.extend(existing.to_dict(orient="records"))

    giotto_path = root / "runs" / "pilot50k_common" / "giotto_real_retry_maxcells" / "giotto_standardized.tsv.gz"
    giotto_summary = read_json(root / "runs" / "pilot50k_common" / "giotto_real_retry_maxcells" / "run_summary.json")
    if giotto_path.exists():
        rows.append(
            {
                "method": "giotto",
                "subset_n_cells": 50000,
                "subset_tag": "pilot50k",
                "lr_policy": "full_common_db",
                "status": "success",
                "standardized_tsv_gz": str(giotto_path),
                "peak_rss_gb": "",
                "elapsed_seconds": giotto_summary.get("elapsed_seconds", ""),
                "full_failure_reason": KNOWN_FULL_FAILURES["giotto"]["reason"],
            }
        )

    for method in ("niches", "spatalk"):
        rescue = discover_rescue_status(root, tag, method)
        bounded = rescue.get("bounded_success")
        if isinstance(bounded, dict) and bounded.get("standardized_tsv_gz"):
            rows.append(
                {
                    "method": method,
                    "subset_n_cells": bounded.get("subset_n_cells"),
                    "subset_tag": bounded.get("subset_tag"),
                    "lr_policy": "full_common_db",
                    "status": "success",
                    "standardized_tsv_gz": bounded.get("standardized_tsv_gz"),
                    "peak_rss_gb": bounded.get("peak_rss_gb", ""),
                    "elapsed_seconds": bounded.get("elapsed_seconds", ""),
                    "full_failure_reason": rescue.get("full_failure_reason", ""),
                }
            )
    return pd.DataFrame(rows)


def build_failure_table(root: Path, tag: str, full_status: pd.DataFrame) -> pd.DataFrame:
    full_success = set(full_status.loc[full_status["full_status"] == "success", "method"].astype(str))
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        if method in full_success:
            continue
        if method in ("niches", "spatalk"):
            rescue = discover_rescue_status(root, tag, method)
            status = rescue.get("status", "not_started")
            reason = rescue.get("reason") or rescue.get("full_failure_reason") or f"{method} A100 rescue is {status}."
            evidence = rescue.get("summary_json") or rescue.get("base_dir") or ""
            rows.append({"method": method, "status": status, "reason": reason, "evidence": evidence})
            continue
        failure = KNOWN_FULL_FAILURES.get(method, {"status": "not_started", "reason": "No full standardized output.", "evidence": ""})
        evidence = str(root / failure["evidence"]) if failure.get("evidence") else ""
        audit_path = root / "results" / tag / f"{method}_audit.json"
        if audit_path.exists():
            evidence = str(audit_path)
        rows.append({"method": method, "status": failure["status"], "reason": failure["reason"], "evidence": evidence})
    return pd.DataFrame(rows)


def write_report(
    *,
    output_path: Path,
    full_status: pd.DataFrame,
    bounded: pd.DataFrame,
    failures: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    def render_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        return "```text\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# LR Benchmark Final Closeout",
        "",
        f"- Tag: `{summary['tag']}`",
        f"- Created: `{summary['created_at']}`",
        f"- Full source-of-truth methods: `{summary['n_full_success']}/{summary['n_total_methods']}`",
        f"- Bounded rescue success methods: `{summary['n_bounded_success_methods']}`",
        f"- Pending methods: `{summary['n_pending_methods']}`",
        "",
        "## Full Source Of Truth",
        "",
        render_table(full_status),
        "",
        "## Bounded Rescue",
        "",
        render_table(bounded) if not bounded.empty else "_No bounded rescue rows._",
        "",
        "## Non-full Terminal Or Pending Methods",
        "",
        render_table(failures) if not failures.empty else "_No failures or pending methods._",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(root: Path, tag: str, commot_path: Path | None, output_dir: Path) -> dict[str, Any]:
    full_inputs, full_rows = collect_full_inputs(root, commot_path)
    if not full_inputs:
        raise RuntimeError("No full standardized inputs were available.")

    combined = pd.concat([read_tsv(path) for path in full_inputs], ignore_index=True)
    combined = combined.loc[:, [column for column in combined.columns if column in STANDARDIZED_RESULT_COLUMNS] + [column for column in combined.columns if column not in STANDARDIZED_RESULT_COLUMNS]]
    full_status = pd.DataFrame(full_rows)
    bounded = build_bounded_table(root, tag)
    failures = build_failure_table(root, tag, full_status)

    pending_statuses = {"not_started", "pending", "running", "pending_aggregate"}
    n_pending = int(sum(str(status) in pending_statuses for status in failures["status"])) if not failures.empty else 0
    summary = {
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "output_dir": str(output_dir),
        "n_total_methods": len(ALL_METHODS),
        "n_full_success": int(full_status[full_status["full_status"] == "success"]["method"].nunique()),
        "full_success_methods": sorted(full_status.loc[full_status["full_status"] == "success", "method"].astype(str).tolist()),
        "n_bounded_success_methods": int(bounded.loc[bounded["status"].astype(str).str.startswith("success", na=False), "method"].nunique()) if not bounded.empty else 0,
        "n_failure_or_nonfull_methods": int(len(failures)),
        "n_pending_methods": n_pending,
        "all_methods_terminal": n_pending == 0,
        "full_source_of_truth_tsv_gz": str(output_dir / "full_source_of_truth.tsv.gz"),
        "bounded_rescue_tsv": str(output_dir / "bounded_rescue_summary.tsv"),
        "failure_status_tsv": str(output_dir / "failure_status.tsv"),
        "report_md": str(output_dir / "benchmark_closeout_report.md"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(combined, output_dir / "full_source_of_truth.tsv.gz")
    write_tsv(full_status, output_dir / "full_method_status.tsv")
    write_tsv(bounded, output_dir / "bounded_rescue_summary.tsv")
    write_tsv(failures, output_dir / "failure_status.tsv")
    (output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output_path=output_dir / "benchmark_closeout_report.md", full_status=full_status, bounded=bounded, failures=failures, summary=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LR benchmark final source-of-truth and closeout tables.")
    parser.add_argument("--root", default="/data/taobo.hu/pyxenium_lr_benchmark_2026-04")
    parser.add_argument("--tag", default="final_closeout_20260511")
    parser.add_argument("--commot-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    tag = args.tag
    commot_path = Path(args.commot_path) if args.commot_path else root / "runs" / tag / "pdc_commot" / "commot_standardized.tsv.gz"
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / tag
    summary = finalize(root=root, tag=tag, commot_path=commot_path, output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
