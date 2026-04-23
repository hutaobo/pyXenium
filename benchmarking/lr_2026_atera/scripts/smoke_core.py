from __future__ import annotations

import argparse
import json

from pyXenium.benchmarking import run_smoke_core


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run the first-wave Atera LR core smoke benchmark.")
    parser.add_argument("--methods", default="pyxenium,squidpy,liana,commot,cellchat")
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--database-mode", default="common-db")
    parser.add_argument("--max-lr-pairs", type=int, default=None)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = run_smoke_core(
        methods=[item.strip() for item in args.methods.split(",") if item.strip()],
        input_manifest=args.input_manifest,
        benchmark_root=args.benchmark_root,
        database_mode=args.database_mode,
        max_lr_pairs=args.max_lr_pairs,
        n_perms=args.n_perms,
        dry_run=args.dry_run,
        continue_on_error=not args.strict,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
