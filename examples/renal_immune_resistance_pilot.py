from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from pyXenium.validation import (
        DEFAULT_DATASET_PATH,
        run_renal_immune_resistance_pilot,
    )
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from pyXenium.validation import (
        DEFAULT_DATASET_PATH,
        run_renal_immune_resistance_pilot,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pyXenium spatial immune-resistance pilot workflow on the public "
            "10x FFPE renal carcinoma Xenium RNA + Protein dataset."
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
    parser.add_argument("--prefer", choices=("auto", "zarr", "h5", "mex"), default="auto")
    parser.add_argument("--sample-id", default="renal_ffpe_public_10x")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--region-bins", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--write-h5ad", default=None)
    parser.add_argument("--manuscript-mode", action="store_true")
    parser.add_argument("--manuscript-root", default="manuscript")
    parser.add_argument("--no-export-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = run_renal_immune_resistance_pilot(
        base_path=args.base_path,
        prefer=args.prefer,
        sample_id=args.sample_id,
        n_neighbors=args.n_neighbors,
        region_bins=args.region_bins,
        output_json=args.output_json,
        output_dir=args.output_dir,
        write_h5ad=args.write_h5ad,
        top_n=args.top_n,
        manuscript_mode=args.manuscript_mode,
        manuscript_root=args.manuscript_root,
        export_figures=not args.no_export_figures,
    )
    print(json.dumps(study["payload"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
