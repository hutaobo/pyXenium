from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args()

    notebook = nbformat.read(args.input, as_version=4)
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name=args.kernel_name)
    client.execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"[mechanostress-pdc] wrote executed notebook: {args.output}")


if __name__ == "__main__":
    main()
