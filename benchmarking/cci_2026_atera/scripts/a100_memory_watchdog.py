"""A100 memory watchdog for bounded CCI retry jobs.

This watchdog is intentionally narrow: it only targets known retry jobs under
the failure-card reopen run roots and never kills unrelated user processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PATTERNS = (
    "run_copulacci_real_bounded_a100_50k.py",
    "copulacci_real_50k",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_remote_script(
    *,
    remote_log: str,
    min_available_gib: int,
    max_target_rss_gib: int,
    kill_patterns: tuple[str, ...],
) -> str:
    pattern_expr = "|".join(kill_patterns)
    return f"""#!/usr/bin/env bash
set -euo pipefail
LOG={remote_log!r}
mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] watchdog start" >> "$LOG"
avail_gib=$(awk '/MemAvailable/ {{ printf "%d", $2/1024/1024 }}' /proc/meminfo)
echo "[$(date -Is)] MemAvailable_GiB=${{avail_gib}}" >> "$LOG"
ps -u "$USER" -o pid=,etime=,pcpu=,pmem=,rss=,cmd= | grep -E {pattern_expr!r} | grep -v grep >> "$LOG" 2>&1 || true
target_pids=""
while read -r pid etime pcpu pmem rss cmd; do
  [ -z "${{pid:-}}" ] && continue
  rss_gib=$((rss / 1024 / 1024))
  if [ "$avail_gib" -lt {min_available_gib} ] || [ "$rss_gib" -gt {max_target_rss_gib} ]; then
    target_pids="${{target_pids}} $pid"
    echo "[$(date -Is)] selected pid=$pid rss_gib=$rss_gib reason=memory_guard cmd=$cmd" >> "$LOG"
  fi
done < <(ps -u "$USER" -o pid=,etime=,pcpu=,pmem=,rss=,cmd= | grep -E {pattern_expr!r} | grep -v grep || true)
if [ -n "$target_pids" ]; then
  echo "[$(date -Is)] killing:$target_pids" >> "$LOG"
  kill $target_pids >> "$LOG" 2>&1 || true
  sleep 5
  kill -9 $target_pids >> "$LOG" 2>&1 || true
else
  echo "[$(date -Is)] no target kill needed" >> "$LOG"
fi
free -h >> "$LOG" 2>&1 || true
echo "[$(date -Is)] watchdog end" >> "$LOG"
"""


def run_watchdog(args: argparse.Namespace) -> int:
    local_status_path = Path(args.local_status)
    local_status_path.parent.mkdir(parents=True, exist_ok=True)
    remote_log = (
        f"{args.remote_root.rstrip('/')}/logs/failure_card_reopen_20260519/"
        "memory_watchdog.log"
    )
    remote_script = build_remote_script(
        remote_log=remote_log,
        min_available_gib=args.min_available_gib,
        max_target_rss_gib=args.max_target_rss_gib,
        kill_patterns=tuple(args.kill_pattern),
    )
    ssh_target = f"{args.user}@{args.host}" if args.user else args.host
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "RequestTTY=no",
        "-o",
        "RemoteCommand=none",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        ssh_target,
        "bash -s",
    ]
    status: dict[str, object] = {
        "checked_at": _now(),
        "host": ssh_target,
        "remote_root": args.remote_root,
        "remote_log": remote_log,
        "min_available_gib": args.min_available_gib,
        "max_target_rss_gib": args.max_target_rss_gib,
        "kill_patterns": list(args.kill_pattern),
    }
    try:
        proc = subprocess.run(
            cmd,
            input=remote_script,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        status.update(
            {
                "status": "ssh_timeout",
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }
        )
        local_status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        return 124

    status.update(
        {
            "status": "ok" if proc.returncode == 0 else "ssh_or_remote_error",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    )
    local_status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return int(proc.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="sscb-a100.scilifelab.se")
    parser.add_argument("--user", default="taobo.hu")
    parser.add_argument(
        "--remote-root",
        default="/data/taobo.hu/pyxenium_cci_benchmark_2026-04",
    )
    parser.add_argument("--min-available-gib", type=int, default=250)
    parser.add_argument("--max-target-rss-gib", type=int, default=180)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--kill-pattern",
        action="append",
        default=list(DEFAULT_PATTERNS),
        help="Process command substring/regex to guard. Can be repeated.",
    )
    parser.add_argument(
        "--local-status",
        default=(
            "benchmarking/cci_2026_atera/results/"
            "failure_card_reopen_20260519/a100_memory_watchdog_status.json"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_watchdog(parse_args(sys.argv[1:])))
