from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.benchmarking import collect_a100_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or execute A100 result recovery commands.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--remote-root", default="/data/taobo.hu/pyxenium_lr_benchmark_2026-04")
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    payload = collect_a100_results(
        benchmark_root=args.benchmark_root,
        remote_root=args.remote_root,
        host=args.host,
        user=args.user,
        dry_run=not args.execute,
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
