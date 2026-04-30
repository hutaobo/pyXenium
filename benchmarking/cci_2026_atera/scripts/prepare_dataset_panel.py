from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyXenium.benchmarking import (
    ATERA_BENCHMARK_RELATIVE_ROOT,
    prepare_atera_cci_benchmark,
    resolve_dataset_entry,
    resolve_layout,
)


def _dataset_prepare_payload(dataset: dict[str, Any], *, benchmark_root: Path | None) -> dict[str, Any]:
    layout = resolve_layout(relative_root=ATERA_BENCHMARK_RELATIVE_ROOT)
    root = Path(benchmark_root) if benchmark_root is not None else layout.root / str(dataset["benchmark_subdir"])
    return {
        "dataset_id": dataset["id"],
        "dataset_root": dataset.get("local_xenium_root"),
        "tbc_results": dataset.get("local_tbc_results"),
        "cell_groups_relpath": dataset.get("cell_groups_relpath"),
        "benchmark_root": str(root),
        "role": dataset.get("role"),
        "platform": dataset.get("platform"),
        "tissue": dataset.get("tissue"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or dry-run a cross-dataset TopoLink-CCI benchmark bundle.")
    parser.add_argument("--dataset-id", default="atera_cervical_wta", help="Dataset id from configs/datasets.yaml.")
    parser.add_argument("--datasets-config", default=None)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--smoke-n-cells", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer", default="h5")
    parser.add_argument("--skip-full-bundle", action="store_true")
    parser.add_argument("--skip-full-h5ad", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only emit the resolved prepare payload.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    dataset = resolve_dataset_entry(args.dataset_id, args.datasets_config)
    benchmark_root = Path(args.benchmark_root) if args.benchmark_root else None
    plan = _dataset_prepare_payload(dataset, benchmark_root=benchmark_root)

    if args.dry_run:
        payload = {"status": "planned", **plan}
    else:
        if not plan["dataset_root"]:
            raise SystemExit(f"Dataset {args.dataset_id!r} does not define a local_xenium_root.")
        payload = prepare_atera_cci_benchmark(
            dataset_root=plan["dataset_root"],
            benchmark_root=plan["benchmark_root"],
            dataset_id=args.dataset_id,
            tbc_results=plan["tbc_results"],
            clusters_relpath=plan["cell_groups_relpath"],
            smoke_n_cells=args.smoke_n_cells,
            seed=args.seed,
            prefer=args.prefer,
            export_full_bundle=not args.skip_full_bundle,
            write_full_h5ad=not args.skip_full_h5ad,
        )
        payload = {"status": "prepared", "dataset_id": args.dataset_id, **payload}

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
