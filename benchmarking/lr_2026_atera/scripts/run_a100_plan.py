from __future__ import annotations

import argparse
import json

from pyXenium.benchmarking import run_a100_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or execute an A100 LR benchmark job manifest.")
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    args = parser.parse_args()

    payload = run_a100_plan(
        plan_json=args.plan_json,
        dry_run=not args.execute,
        job_ids=args.job_id or None,
        remote=args.remote,
        host=args.host,
        user=args.user,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
