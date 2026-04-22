from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from pyXenium.validation import (
        DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
        run_atera_wta_breast_topology,
    )
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from pyXenium.validation import (
        DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
        run_atera_wta_breast_topology,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed pyXenium Atera WTA breast LR/pathway topology workflow "
            "against the local Xenium export and optional precomputed t_and_c results."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=os.environ.get("PYXENIUM_DATASET_PATH", DEFAULT_ATERA_WTA_BREAST_DATASET_PATH),
        help="Local path to the Atera breast Xenium dataset directory.",
    )
    parser.add_argument("--tbc-results", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manuscript-mode", action="store_true")
    parser.add_argument("--manuscript-root", default="manuscript")
    parser.add_argument("--sample-id", default="atera_wta_ffpe_breast")
    parser.add_argument("--write-h5ad", default=None)
    parser.add_argument("--no-export-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = run_atera_wta_breast_topology(
        dataset_root=args.dataset_root,
        tbc_results=args.tbc_results,
        output_dir=args.output_dir,
        manuscript_mode=args.manuscript_mode,
        manuscript_root=args.manuscript_root,
        sample_id=args.sample_id,
        export_figures=not args.no_export_figures,
        write_h5ad=args.write_h5ad,
    )
    print(json.dumps(study["payload"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
