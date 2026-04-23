from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.benchmarking import aggregate_standardized_results, resolve_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate standardized benchmark outputs.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--result-path", action="append", default=[])
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    layout = resolve_layout(relative_root=args.benchmark_root or Path("benchmarking") / "lr_2026_atera")
    result_paths = args.result_path or [str(path) for path in sorted(layout.runs_dir.glob("**/*standardized*.tsv"))]
    if not result_paths:
        raise SystemExit("No standardized result tables were found.")

    output_path = args.output_path or layout.results_dir / "combined_standardized.tsv"
    combined = aggregate_standardized_results(result_paths, output_path=output_path)
    print(json.dumps({"output_path": str(output_path), "n_rows": int(len(combined)), "inputs": result_paths}, indent=2))


if __name__ == "__main__":
    main()
