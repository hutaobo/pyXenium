from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.gmi import build_gmi_a100_plan, write_gmi_a100_plan


parser = argparse.ArgumentParser(description="Render the experimental Atera GMI A100 plan.")
parser.add_argument("--remote-xenium-root", default=None)
parser.add_argument("--remote-root", default=None)
parser.add_argument("--repo-dir", default=None)
parser.add_argument("--env-name", default="pyx-gmi")
parser.add_argument("--output-json", default="benchmarking/gmi_atera/logs/a100_gmi_plan.json")
args = parser.parse_args()

kwargs = {"repo_dir": args.repo_dir, "env_name": args.env_name}
if args.remote_xenium_root:
    kwargs["remote_xenium_root"] = args.remote_xenium_root
if args.remote_root:
    kwargs["remote_root"] = args.remote_root
payload = build_gmi_a100_plan(**kwargs)
write_gmi_a100_plan(payload, Path(args.output_json))
print(json.dumps(payload, indent=2))
