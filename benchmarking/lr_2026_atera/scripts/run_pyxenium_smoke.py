from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.benchmarking import resolve_layout, run_pyxenium_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pyXenium smoke benchmark.")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tbc-results", default=None)
    parser.add_argument("--lr-panel-path", default=None)
    parser.add_argument("--database-mode", default="smoke-panel")
    parser.add_argument("--export-figures", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    layout = resolve_layout(relative_root=args.benchmark_root or Path("benchmarking") / "lr_2026_atera")
    payload = run_pyxenium_smoke(
        input_h5ad=args.input_h5ad or layout.data_dir / "smoke" / "adata_smoke.h5ad",
        output_dir=args.output_dir or layout.runs_dir / "pyxenium_smoke",
        tbc_results=args.tbc_results or Path(r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs") / "sfplot_tbc_formal_wta" / "results",
        lr_panel_path=args.lr_panel_path or layout.data_dir / "atera_smoke_panel.tsv",
        database_mode=args.database_mode,
        export_figures=args.export_figures,
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
