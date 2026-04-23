from __future__ import annotations

import argparse
import json

from pyXenium.benchmarking import run_registered_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Squidpy ligrec LR benchmark adapter.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="squidpy")
    parser.add_argument("--database-mode", default="common-db")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-lr-pairs", type=int, default=None)
    parser.add_argument("--n-perms", type=int, default=100)
    args = parser.parse_args()

    payload = run_registered_method(
        method="squidpy",
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        database_mode=args.database_mode,
        phase=args.phase,
        max_lr_pairs=args.max_lr_pairs,
        n_perms=args.n_perms,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
