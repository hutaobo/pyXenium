# pyXenium mechanostress Atera tutorial on PDC

This scaffold runs the `pyXenium.mechanostress` Atera WTA breast tutorial on
PDC Dardel and collects an executed notebook plus small static artifacts for
ReadTheDocs.

Default dataset:

```text
/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs
```

Default run root:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04
```

## Local staging

From the local repository:

```bash
python benchmarking/mechanostress_pdc/scripts/stage_to_pdc.py --dry-run
python benchmarking/mechanostress_pdc/scripts/stage_to_pdc.py
```

The staging script uploads the repo subset needed for the tutorial and checks
the PDC dataset cache. Missing required Xenium files are copied from:

```text
Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs
```

Only the h5 matrix, cells parquet, cell/nucleus boundary parquet, cell-group
CSV, and small metadata files are copied.

## PDC execution

On PDC:

```bash
export PDC_ROOT=/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04
export PDC_DATASET_ROOT=/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs

bash benchmarking/mechanostress_pdc/scripts/bootstrap_pdc_env.sh
bash benchmarking/mechanostress_pdc/scripts/submit_pdc_notebook.sh --dry-run
bash benchmarking/mechanostress_pdc/scripts/submit_pdc_notebook.sh
```

The notebook job uses one node, one task, 16 CPUs, 128GB RAM, and 4 hours by
default. It writes full artifacts under `${PDC_ROOT}/outputs` and small tutorial
snapshots under `${PDC_ROOT}/static/mechanostress_atera_pdc`.

## Collect results

From the local repository after the Slurm job completes:

```bash
python benchmarking/mechanostress_pdc/scripts/collect_pdc_results.py
```

This replaces `docs/tutorials/mechanostress_atera_pdc.ipynb` with the executed
notebook and syncs small static files into
`docs/_static/tutorials/mechanostress_atera_pdc/`.
