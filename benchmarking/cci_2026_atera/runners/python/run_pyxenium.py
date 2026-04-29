from __future__ import annotations

import argparse
import json

from pyXenium.benchmarking import run_pyxenium_smoke, run_registered_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner wrapper for the pyXenium CCI benchmark adapter.")
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tbc-results", default=None)
    parser.add_argument("--cci-panel-path", default=None)
    parser.add_argument("--method", default="pyxenium")
    parser.add_argument("--database-mode", default="common-db")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-cci-pairs", type=int, default=None)
    parser.add_argument("--export-figures", action="store_true")
    args = parser.parse_args()

    if args.input_manifest:
        payload = run_registered_method(
            method="pyxenium",
            input_manifest=args.input_manifest,
            output_dir=args.output_dir,
            database_mode=args.database_mode,
            phase=args.phase,
            max_cci_pairs=args.max_cci_pairs,
            tbc_results=args.tbc_results,
            export_figures=args.export_figures,
        )
    else:
        missing = [
            name
            for name, value in {
                "--input-h5ad": args.input_h5ad,
                "--tbc-results": args.tbc_results,
                "--cci-panel-path": args.cci_panel_path,
            }.items()
            if value is None
        ]
        if missing:
            raise SystemExit(f"Missing required legacy pyXenium runner arguments: {', '.join(missing)}")
        payload = run_pyxenium_smoke(
            input_h5ad=args.input_h5ad,
            output_dir=args.output_dir,
            tbc_results=args.tbc_results,
            cci_panel_path=args.cci_panel_path,
            database_mode=args.database_mode,
            export_figures=args.export_figures,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
