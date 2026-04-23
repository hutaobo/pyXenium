from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.benchmarking import build_method_run_plan, run_registered_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run one Atera LR benchmark method adapter.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--database-mode", default="common-db")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-lr-pairs", type=int, default=None)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--rscript", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "method": args.method,
        "input_manifest": args.input_manifest,
        "output_dir": Path(args.output_dir) if args.output_dir else None,
        "benchmark_root": args.benchmark_root,
        "database_mode": args.database_mode,
        "phase": args.phase,
        "max_lr_pairs": args.max_lr_pairs,
        "n_perms": args.n_perms,
    }
    if args.dry_run:
        payload = build_method_run_plan(**kwargs)
    else:
        payload = run_registered_method(**kwargs, rscript=args.rscript)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
