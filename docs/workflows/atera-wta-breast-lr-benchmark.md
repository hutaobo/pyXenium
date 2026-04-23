# Atera WTA Breast LR Benchmark

`pyXenium` ships a dedicated benchmark scaffold for comparing ligand-receptor methods on the Atera Xenium WTA breast dataset used by the LR tutorial.

## What This Workflow Produces

- a frozen `AnnData` bundle for the full dataset and a stratified smoke subset
- sparse cross-language matrices and metadata tables for external Python and R methods
- a shared LR resource table for `common-db` benchmarking
- a standardized output schema across methods
- biology-oriented summaries for canonical recovery, pathway relevance, spatial coherence, robustness, and novelty support

## Benchmark Root

The benchmark workspace lives under:

```text
benchmarking/lr_2026_atera/
```

This directory contains:

- `configs/` for method registry and biology scoring panels
- `envs/` for one environment manifest per method
- `scripts/` for preparation, environment creation, aggregation, report rendering, and A100 staging
- `runners/` for method-side adapters

## CLI Entry Points

Prepare the shared input bundle:

```bash
pyxenium benchmark atera-lr prepare
```

Prepare a full sparse bundle locally without requiring a full `.h5ad`:

```bash
pyxenium benchmark atera-lr prepare --skip-full-h5ad
```

Run the built-in `pyXenium` smoke benchmark:

```bash
pyxenium benchmark atera-lr smoke-pyxenium
```

Dry-run one method adapter:

```bash
pyxenium benchmark atera-lr run-method --method squidpy --database-mode common-db --dry-run
```

Run the first-wave core smoke panel:

```bash
pyxenium benchmark atera-lr smoke-core --methods pyxenium,squidpy,liana,commot,cellchat --database-mode common-db
```

Aggregate standardized results:

```bash
pyxenium benchmark atera-lr aggregate
```

Render a markdown report:

```bash
pyxenium benchmark atera-lr report
```

Generate A100 staging commands:

```bash
pyxenium benchmark atera-lr stage-a100 --plan-only \
  --remote-xenium-root /mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs \
  --remote-root /data/taobo.hu/pyxenium_lr_benchmark_2026-04
```

Build and dry-run the A100 full common-db plan. The plan includes a `prepare_full_bundle` job that reads from the read-only `/mnt` Xenium export and writes all bundle/runs/logs/reports under `/data/taobo.hu/pyxenium_lr_benchmark_2026-04`:

```bash
pyxenium benchmark atera-lr prepare-a100-bundle --phase full --database-mode common-db \
  --remote-xenium-root /mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs \
  --remote-root /data/taobo.hu/pyxenium_lr_benchmark_2026-04
pyxenium benchmark atera-lr run-a100-plan --plan-json benchmarking/lr_2026_atera/logs/a100_bundle_plan.json
```

Generate A100 result recovery commands:

```bash
pyxenium benchmark atera-lr collect-a100-results --host <host> --user <user>
```

## Practical Note

The full Xenium matrix is exported as sparse Matrix Market rather than a dense TSV because the full matrix is too large to move or parse safely as a dense text file. The benchmark prep still emits `meta.tsv`, `coords.tsv`, `genes.tsv`, `barcodes.tsv`, and the shared LR tables expected by the method adapters.

The first-wave real adapter contract covers `pyXenium`, `Squidpy ligrec`, `LIANA+ spatial bivariate`, `COMMOT`, and `CellChat v3 / SpatialCellChat`. Third-party package installation remains isolated per method environment; missing packages should fail inside the method run with a reproducible `run_summary.json` rather than changing the shared schema.

Use the declared per-method environments rather than a base Python environment. In particular, the Squidpy environment pins `zarr<3` because current `spatialdata`/`ome-zarr` stacks can fail to import against incompatible zarr releases.

A100 orchestration writes a portable stage/job manifest and never stores passwords. The A100 source/destination split is explicit: `/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs` is read-only input, while `/data/taobo.hu/pyxenium_lr_benchmark_2026-04` is the only writable benchmark root. The report step automatically includes run status, engineering reproducibility, canonical pair rank matrix, and A100 resource summary when the corresponding run summaries or A100 plan exist.
