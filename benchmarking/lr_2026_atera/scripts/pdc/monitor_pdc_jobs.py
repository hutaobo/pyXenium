from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PDC_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_lr_benchmark_2026-04"


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["ssh", "-o", "BatchMode=yes", "-o", "RequestTTY=no", "-o", "RemoteCommand=none", host, remote_command],
        check=check,
    )


def read_remote_jsonl(host: str, remote_root: str) -> dict[str, Any]:
    probe = f"""
set -e
ROOT={q(remote_root)}
echo '---SQUEUE---'
squeue -u "$(whoami)" -h -o '%i|%T|%j|%M|%D|%R' || true
echo '---SUMMARIES---'
find "$ROOT/runs" -path '*/run_summary.json' -type f -maxdepth 5 -print 2>/dev/null | sort | while read -r f; do
  printf '%s\\t' "$f"
  python3 - <<PY "$f" 2>/dev/null || true
import json, sys
p=sys.argv[1]
try:
    d=json.load(open(p))
    print(json.dumps({{k:d.get(k) for k in ("method","status","phase","database_mode","runtime_seconds","standardized_tsv","standardized_tsv_gz","n_rows")}}, sort_keys=True))
except Exception as exc:
    print(json.dumps({{"status":"unreadable","error":str(exc)}}))
PY
done
echo '---CARDS---'
find "$ROOT/runs" -name method_card.md -type f -maxdepth 5 -print 2>/dev/null | sort
echo '---RECENT_LOGS---'
find "$ROOT/logs" -maxdepth 1 -type f \\( -name '*.stderr.log' -o -name '*.stdout.log' -o -name '*.resource.log' \\) -printf '%T@\\t%p\\t%s\\n' 2>/dev/null | sort -nr | head -20
"""
    completed = ssh(host, probe)
    return {"stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode}


def parse_monitor_output(text: str) -> dict[str, Any]:
    sections: dict[str, list[str]] = {"squeue": [], "summaries": [], "cards": [], "recent_logs": []}
    current: str | None = None
    markers = {
        "---SQUEUE---": "squeue",
        "---SUMMARIES---": "summaries",
        "---CARDS---": "cards",
        "---RECENT_LOGS---": "recent_logs",
    }
    for line in text.splitlines():
        if line in markers:
            current = markers[line]
            continue
        if current:
            sections[current].append(line)
    squeue = []
    for line in sections["squeue"]:
        parts = line.split("|")
        if len(parts) >= 6:
            squeue.append(
                {
                    "job_id": parts[0],
                    "state": parts[1],
                    "name": parts[2],
                    "elapsed": parts[3],
                    "nodes": parts[4],
                    "reason": parts[5],
                }
            )
    summaries = []
    for line in sections["summaries"]:
        if "\t" not in line:
            continue
        path, payload = line.split("\t", 1)
        try:
            row = json.loads(payload)
        except json.JSONDecodeError:
            row = {"status": "unparsed", "raw": payload}
        row["path"] = path
        summaries.append(row)
    recent_logs = []
    for line in sections["recent_logs"]:
        parts = line.split("\t")
        if len(parts) == 3:
            recent_logs.append({"mtime_epoch": parts[0], "path": parts[1], "bytes": parts[2]})
    return {
        "squeue": squeue,
        "summaries": summaries,
        "method_cards": [line for line in sections["cards"] if line.strip()],
        "recent_logs": recent_logs,
    }


def render_markdown(status: dict[str, Any], remote_root: str) -> str:
    lines = [
        "# PDC LR Benchmark Live Status",
        "",
        f"- Updated: `{status['checked_at']}`",
        f"- Remote root: `{remote_root}`",
        f"- Queue jobs: `{len(status['squeue'])}`",
        f"- Run summaries: `{len(status['summaries'])}`",
        f"- Method cards: `{len(status['method_cards'])}`",
        "",
        "## Queue",
        "",
        "| job_id | state | name | elapsed | reason |",
        "|---|---|---|---|---|",
    ]
    for row in status["squeue"][:50]:
        lines.append(f"| {row['job_id']} | {row['state']} | {row['name']} | {row['elapsed']} | {row['reason']} |")
    lines.extend(["", "## Completed/Reported Runs", "", "| method | status | phase | database | rows | path |", "|---|---|---|---|---:|---|"])
    for row in status["summaries"][:100]:
        lines.append(
            f"| {row.get('method', '')} | {row.get('status', '')} | {row.get('phase', '')} | "
            f"{row.get('database_mode', '')} | {row.get('n_rows', '')} | `{row.get('path', '')}` |"
        )
    lines.extend(["", "## Method Cards", ""])
    for card in status["method_cards"][:100]:
        lines.append(f"- `{card}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor PDC LR benchmark Slurm jobs and remote run artifacts.")
    parser.add_argument("--host", default=DEFAULT_PDC_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    raw = read_remote_jsonl(args.host, args.remote_root.rstrip("/"))
    parsed = parse_monitor_output(raw["stdout"])
    status = {
        "kind": "pdc_live_status",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "remote_root": args.remote_root.rstrip("/"),
        "raw_stderr": raw["stderr"],
        **parsed,
    }
    local_root = Path(args.local_root) if args.local_root else Path.cwd() / "benchmarking" / "lr_2026_atera"
    output_json = Path(args.output_json) if args.output_json else local_root / "results" / "pdc_live_job_status.json"
    output_md = Path(args.output_md) if args.output_md else local_root / "reports" / "pdc_live_status.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(status, args.remote_root.rstrip("/")), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
