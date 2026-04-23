from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.benchmarking import prepare_atera_lr_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Atera Xenium LR benchmark input bundle.")
    parser.add_argument("--dataset-root", "--xenium-root", dest="dataset_root", default=None)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--tbc-results", default=None)
    parser.add_argument("--smoke-n-cells", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer", default="h5")
    parser.add_argument("--skip-full-bundle", action="store_true")
    parser.add_argument("--skip-full-h5ad", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    payload = prepare_atera_lr_benchmark(
        dataset_root=args.dataset_root,
        benchmark_root=args.benchmark_root,
        tbc_results=args.tbc_results,
        smoke_n_cells=args.smoke_n_cells,
        seed=args.seed,
        prefer=args.prefer,
        export_full_bundle=not args.skip_full_bundle,
        write_full_h5ad=not args.skip_full_h5ad,
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
