from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pyXenium.benchmarking import resolve_layout


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_publication_benchmark_plan(*, benchmark_root: str | Path | None = None) -> pd.DataFrame:
    layout = resolve_layout(relative_root=benchmark_root or Path("benchmarking") / "cci_2026_atera")
    datasets = _load_yaml(layout.config_dir / "datasets.yaml").get("datasets", [])
    readiness = _load_yaml(layout.config_dir / "publication_readiness.yaml").get("publication_readiness", {})
    completed = readiness.get("method_panel", {}).get("already_completed", [])
    must_attempt = readiness.get("method_panel", {}).get("must_attempt", [])
    methods = list(dict.fromkeys([*completed, *must_attempt]))

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset.get("id") == "public_non_xenium_spatial":
            phases = ["select_dataset", "prepare_adapter", "smoke_common", "pilot_common"]
        else:
            phases = ["prepare", "smoke_common", "pilot_common", "full_common", "validation_controls"]
        for method in methods:
            for phase in phases:
                rows.append(
                    {
                        "dataset_id": dataset.get("id"),
                        "dataset_role": dataset.get("role"),
                        "platform": dataset.get("platform"),
                        "method": method,
                        "phase": phase,
                        "target_status": "completed_or_import" if method in completed else "must_attempt",
                        "output_policy": "full_result_or_bounded_result_or_method_card",
                    }
                )
    rows.append(
        {
            "dataset_id": "synthetic_topology_truth",
            "dataset_role": "synthetic_false_positive_control",
            "platform": "simulated",
            "method": "all_methods",
            "phase": "synthetic_truth_auroc_auprc_fdr",
            "target_status": "required",
            "output_policy": "synthetic_truth_metrics.tsv",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reviewer-facing TopoLink-CCI publication benchmark task plan.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--output-tsv", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    plan = build_publication_benchmark_plan(benchmark_root=args.benchmark_root)
    payload = {"n_tasks": int(len(plan)), "methods": sorted(plan["method"].unique().tolist()), "datasets": sorted(plan["dataset_id"].unique().tolist())}
    if args.output_tsv:
        path = Path(args.output_tsv)
        path.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(path, sep="\t", index=False)
        payload["output_tsv"] = str(path)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": payload, "tasks": plan.to_dict(orient="records")}, indent=2) + "\n", encoding="utf-8")
        payload["output_json"] = str(path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
