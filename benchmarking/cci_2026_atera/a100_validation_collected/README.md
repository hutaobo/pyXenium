# TopoLink-CCI validation run on A100

Purpose: computational false-positive controls for top TopoLink-CCI axes.

This folder intentionally reuses existing benchmark artifacts via symlinks instead of duplicating the full Xenium export.

Layout:
- scripts/: validation framework and launch script.
- data_links/full_sparse_bundle -> /data/taobo.hu/topolink_cci_benchmark_2026-04/data/full
- data_links/full_common_results -> /data/taobo.hu/topolink_cci_benchmark_2026-04/runs/full_common
- data_links/input_manifest.json -> /data/taobo.hu/topolink_cci_benchmark_2026-04/data/input_manifest.json
- outputs/: evidence tables, false-positive controls, scoreboard, figures, run_summary.json.
- logs/: stdout/stderr plus /usr/bin/time resource report.
- tmp/: job-local temporary files.

No writes are made to /mnt. The original local Windows Xenium outs are not re-uploaded because the A100 already has the derived full sparse bundle and benchmark outputs needed for this validation.
