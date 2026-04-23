from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.benchmarking import DEFAULT_A100_READONLY_XENIUM_ROOT, DEFAULT_A100_REMOTE_ROOT, build_a100_stage_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage-to-A100 SSH/SCP commands.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--remote-root", default=DEFAULT_A100_REMOTE_ROOT)
    parser.add_argument("--remote-xenium-root", default=DEFAULT_A100_READONLY_XENIUM_ROOT)
    parser.add_argument("--stage-data", action="store_true", default=None)
    parser.add_argument("--skip-data", action="store_false", dest="stage_data")
    parser.add_argument("--include-path", action="append", default=[])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    payload = build_a100_stage_plan(
        benchmark_root=args.benchmark_root,
        remote_root=args.remote_root,
        remote_xenium_root=args.remote_xenium_root,
        stage_data=args.stage_data,
        host=args.host,
        user=args.user,
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
