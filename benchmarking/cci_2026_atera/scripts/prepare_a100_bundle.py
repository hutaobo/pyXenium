from __future__ import annotations

import argparse
import json

from pyXenium.benchmarking import DEFAULT_A100_READONLY_XENIUM_ROOT, DEFAULT_A100_REMOTE_ROOT, prepare_a100_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an A100 stage plan and CCI full-run job manifest.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--remote-root", default=DEFAULT_A100_REMOTE_ROOT)
    parser.add_argument("--remote-xenium-root", default=DEFAULT_A100_READONLY_XENIUM_ROOT)
    parser.add_argument("--methods", default="pyxenium,squidpy,liana,commot,cellchat")
    parser.add_argument("--database-mode", default="common-db")
    parser.add_argument("--phase", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-cci-pairs", type=int, default=None)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--stage-data", action="store_true", default=None)
    parser.add_argument("--skip-data", action="store_false", dest="stage_data")
    parser.add_argument("--smoke-n-cells", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer", default="h5")
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--allow-missing-full", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    payload = prepare_a100_bundle(
        benchmark_root=args.benchmark_root,
        remote_root=args.remote_root,
        remote_xenium_root=args.remote_xenium_root,
        methods=[item.strip() for item in args.methods.split(",") if item.strip()],
        database_mode=args.database_mode,
        phase=args.phase,
        max_cci_pairs=args.max_cci_pairs,
        n_perms=args.n_perms,
        require_full=not args.allow_missing_full,
        include_prepare=not args.skip_prepare,
        stage_data=args.stage_data,
        smoke_n_cells=args.smoke_n_cells,
        seed=args.seed,
        prefer=args.prefer,
        host=args.host,
        user=args.user,
        output_json=args.output_json,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
