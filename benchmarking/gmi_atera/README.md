# Atera Contour GMI Benchmark

This workspace is the reproducible scaffold for the experimental `pyXenium.gmi`
S1-vs-S5 contour workflow on the Atera WTA breast Xenium dataset.

The workflow treats the source Xenium export as read-only and writes all A100
artifacts under `/data/taobo.hu/pyxenium_gmi_contour_2026-04`.

## A100 Plan

```powershell
pyxenium gmi a100-plan `
  --remote-xenium-root /mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs `
  --remote-root /data/taobo.hu/pyxenium_gmi_contour_2026-04 `
  --output-json benchmarking/gmi_atera/logs/a100_gmi_plan.json
```

The generated plan includes:

- smoke run: S1/S5 contours, top 200 RNA genes, top 50 spatial features
- full run: S1/S5 contours, top 500 RNA genes, top 100 spatial features
- stability run: complete Atera dataset, 5 spatial CV folds, 10 bootstrap repeats,
  label-permutation control, coordinate-shuffle control, and spatial-feature-shuffle control
- validation runs: RNA-only QC20, spatial-only QC20, and no-coordinate QC20
- sensitivity runs: top 1000 RNA genes and all non-empty contours (`n_cells >= 1`)

The primary QC remains `n_cells >= 20`. All endpoint contours remain in
`sample_metadata.tsv` with `retained` and `drop_reason`; zero-cell and very
small contours are excluded from the main GMI fit because contour pseudo-bulk
logCPM and composition summaries are unstable.

## Local Smoke

```powershell
pyxenium gmi run `
  --dataset-root "Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs" `
  --output-dir benchmarking/gmi_atera/runs/local_smoke `
  --rna-feature-count 200 `
  --spatial-feature-count 50
```

## Vendored Gmi

`pyXenium.gmi` installs the R package only from
`src/pyXenium/_vendor/Gmi`, a pinned source snapshot of
`https://github.com/Moyu-nie/Gmi`. Runtime commands must not install Gmi from
GitHub HEAD.
