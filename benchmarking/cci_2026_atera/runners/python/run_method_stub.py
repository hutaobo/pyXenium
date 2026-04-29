from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder runner for third-party Python CCI methods.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": args.method,
        "status": "scaffolded",
        "message": "This method is registered and environment-managed, but its adapter is still a stub in this repo.",
        "input_manifest": str(Path(args.input_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "run_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
