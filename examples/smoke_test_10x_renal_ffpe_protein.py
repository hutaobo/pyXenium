from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from pyXenium.validation.renal_ffpe_protein import (
        DEFAULT_DATASET_PATH,
        run_validated_renal_ffpe_smoke,
    )
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from pyXenium.validation.renal_ffpe_protein import (
        DEFAULT_DATASET_PATH,
        run_validated_renal_ffpe_smoke,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test pyXenium on the official 10x Genomics FFPE Human Renal Cell Carcinoma "
            "RNA + Protein Xenium dataset."
        )
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        default=os.environ.get("PYXENIUM_DATASET_PATH", DEFAULT_DATASET_PATH),
        help=(
            "Local path to the Xenium dataset directory. Defaults to the "
            "PYXENIUM_DATASET_PATH environment variable or the validated local path."
        ),
    )
    parser.add_argument(
        "--prefer",
        choices=("auto", "zarr", "h5", "mex"),
        default="auto",
        help="Preferred matrix backend passed to load_xenium_gene_protein().",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top RNA features, protein markers, and clusters to report.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Print the summary even if the observed values differ from the validated reference.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for report.md, summary.json, and CSV summaries.",
    )
    parser.add_argument(
        "--write-h5ad",
        default=None,
        help="Optional path to export the loaded AnnData object as an .h5ad file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_validated_renal_ffpe_smoke(
        base_path=args.base_path,
        prefer=args.prefer,
        top_n=args.top_n,
        output_json=args.output_json,
        output_dir=args.output_dir,
        write_h5ad=args.write_h5ad,
    )
    print(json.dumps(payload, indent=2))
    if payload["issues"] and not args.allow_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
