# Atera Xenium LR Benchmark

This workspace packages the reproducible benchmark scaffold for the Atera Xenium WTA breast ligand-receptor study. It is designed around isolated per-method environments, a shared frozen input bundle, and a standardized result schema that focuses on biological discovery quality rather than raw method scores.

## Layout

- `configs/`: canonical axes, pathway panels, method registry, and top-level benchmark settings
- `envs/`: one environment manifest per method plus bootstrap helpers
- `scripts/`: high-level entrypoints to prepare data, create environments, aggregate results, render reports, and stage work to A100
- `runners/`: method-side wrappers that consume the frozen bundle and emit standardized artifacts or run manifests
- `data/`: generated benchmark inputs such as optional `adata_full.h5ad`, `adata_smoke.h5ad`, sparse matrices, and shared LR databases
- `runs/`: per-method execution outputs
- `results/`: merged standardized tables and evaluation summaries
- `reports/`: markdown reports and method cards

## Quick Start

Create the frozen input bundle:

```powershell
python benchmarking/lr_2026_atera/scripts/prepare_data.py `
  --dataset-root "Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs"
```

For local full-pilot runs, export the full sparse bundle while keeping the full `.h5ad` optional:

```powershell
python benchmarking/lr_2026_atera/scripts/prepare_data.py `
  --dataset-root "Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs" `
  --skip-full-h5ad
```

Create and bootstrap the prep environment:

```powershell
python benchmarking/lr_2026_atera/scripts/create_env.py --method pyx-lr-prep
```

Run the built-in `pyXenium` smoke benchmark:

```powershell
python benchmarking/lr_2026_atera/scripts/run_pyxenium_smoke.py
```

Dry-run or execute any real adapter through the unified contract:

```powershell
pyxenium benchmark atera-lr run-method `
  --method squidpy `
  --database-mode common-db `
  --phase smoke `
  --dry-run
```

Run the first-wave core smoke panel:

```powershell
pyxenium benchmark atera-lr smoke-core `
  --methods pyxenium,squidpy,liana,commot,cellchat `
  --database-mode common-db `
  --max-lr-pairs 50
```

Aggregate standardized result tables and build a report:

```powershell
python benchmarking/lr_2026_atera/scripts/aggregate_results.py
python benchmarking/lr_2026_atera/scripts/render_report.py
```

Generate A100 staging commands:

```powershell
python benchmarking/lr_2026_atera/scripts/stage_to_a100.py `
  --plan-only `
  --remote-xenium-root /mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs `
  --remote-root /data/taobo.hu/pyxenium_lr_benchmark_2026-04
```

Build the A100 full-run plan. The generated plan reads the original Xenium export from the read-only `/mnt` path and writes the full sparse bundle, logs, runs, results, and reports only under `/data/taobo.hu/pyxenium_lr_benchmark_2026-04`:

```powershell
python benchmarking/lr_2026_atera/scripts/prepare_a100_bundle.py `
  --methods pyxenium,squidpy,liana,commot,cellchat `
  --phase full `
  --database-mode common-db `
  --remote-xenium-root /mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs `
  --remote-root /data/taobo.hu/pyxenium_lr_benchmark_2026-04
```

Dry-run the A100 job manifest:

```powershell
python benchmarking/lr_2026_atera/scripts/run_a100_plan.py `
  --plan-json benchmarking/lr_2026_atera/logs/a100_bundle_plan.json
```

Generate result recovery commands:

```powershell
python benchmarking/lr_2026_atera/scripts/collect_a100_results.py `
  --host your-a100-host `
  --user taobo.hu
```

## Notes

- Full Xenium matrices are exported in sparse Matrix Market format. A dense `counts_symbol.tsv` is intentionally not emitted because it would be impractically large for the full `170,057 x 18,028` matrix.
- The benchmark prep harmonizes gene identifiers by promoting `adata.var['name']` to benchmark-facing `var_names`, while preserving the original Ensembl IDs in `adata.var['ensembl_id']`.
- The first-wave real adapters now cover `pyXenium`, `Squidpy ligrec`, `LIANA+ spatial bivariate`, `COMMOT`, and `CellChat v3 / SpatialCellChat`.
- Each adapter writes method-native raw artifacts, `params.json`, `run_summary.json`, and a standardized TSV that can be consumed by the existing aggregate/report steps.
- Squidpy is run from its isolated `pyx-lr-squidpy` environment, which pins `zarr<3` to avoid the `ome-zarr`/`FSStore` import conflict seen in some base environments.
- A100 planning never stores passwords or hard-codes hosts. Use `--plan-only` without host/user for a portable plan, then supply SSH details only when staging or collecting results.
- On A100, `/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs` is treated as read-only. The stage/job manifest includes a path-safety check that flags any output path under `/mnt`; all writable paths are organized under `/data/taobo.hu/pyxenium_lr_benchmark_2026-04`.
