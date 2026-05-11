from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pyXenium.benchmarking import (
    compute_canonical_recovery,
    compute_novelty_support,
    compute_pathway_relevance,
    compute_spatial_coherence,
    compute_robustness,
    build_a100_resource_summary,
    build_engineering_summary,
    render_atera_lr_benchmark_report,
    resolve_layout,
    score_biological_performance,
    summarize_run_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the benchmark markdown report.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--combined-results", default=None)
    parser.add_argument("--canonical-config", default=None)
    parser.add_argument("--pathway-config", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    layout = resolve_layout(relative_root=args.benchmark_root or Path("benchmarking") / "lr_2026_atera")
    combined_path = Path(args.combined_results) if args.combined_results else layout.results_dir / "combined_standardized.tsv"
    canonical_path = Path(args.canonical_config) if args.canonical_config else layout.config_dir / "canonical_axes.yaml"
    pathway_path = Path(args.pathway_config) if args.pathway_config else layout.config_dir / "pathways.yaml"
    output_path = Path(args.output_path) if args.output_path else layout.reports_dir / "benchmark_report.md"

    combined = pd.read_csv(combined_path, sep="\t")
    canonical_detail, canonical_summary = compute_canonical_recovery(combined, canonical_axes=canonical_path)
    _, pathway_summary = compute_pathway_relevance(combined, pathway_config=pathway_path)
    spatial_summary = compute_spatial_coherence(combined)
    _, novelty_summary = compute_novelty_support(combined)
    robustness_summary = compute_robustness(combined)
    run_status = summarize_run_status(layout.runs_dir)
    engineering_summary = build_engineering_summary(run_status)
    a100_resource_summary = pd.DataFrame()
    a100_plan_path = layout.logs_dir / "a100_bundle_plan.json"
    if a100_plan_path.exists():
        a100_payload = json.loads(a100_plan_path.read_text(encoding="utf-8"))
        a100_resource_summary = build_a100_resource_summary(a100_payload.get("job_manifest", a100_payload))
    biology_summary = score_biological_performance(
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        spatial_summary=spatial_summary,
        robustness_summary=robustness_summary,
        novelty_summary=novelty_summary,
    )
    markdown = render_atera_lr_benchmark_report(
        combined_results=combined,
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        biology_summary=biology_summary,
        benchmark_root=layout.root,
        run_status=run_status,
        engineering_summary=engineering_summary,
        canonical_detail=canonical_detail,
        a100_resource_summary=a100_resource_summary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"report_md": str(output_path), "n_methods": int(combined["method"].nunique()) if not combined.empty else 0}, indent=2))


if __name__ == "__main__":
    main()
