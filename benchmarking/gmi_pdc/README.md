# pyXenium contour-GMI on PDC Dardel

This scaffold runs the canonical contour-GMI Atera S1-vs-S5 validation on PDC
Dardel with Slurm. It reuses the Xenium source cache already being copied for
the CCI benchmark and writes all GMI-specific artifacts to a separate scratch
root.

Default dataset:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs
```

Default GMI root:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04
```

## Workflow

From the staged repo on Dardel:

```bash
export PDC_ROOT=/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04
export PDC_XENIUM_ROOT=/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs

bash benchmarking/gmi_pdc/scripts/bootstrap_pdc_env.sh
bash benchmarking/gmi_pdc/scripts/prepare_pdc_inputs.sh
bash benchmarking/gmi_pdc/scripts/submit_pdc_chain.sh
bash benchmarking/gmi_pdc/scripts/monitor_pdc_gmi.sh
```

If the Slurm project cannot be inferred with `projinfo`, pass it explicitly:

```bash
bash benchmarking/gmi_pdc/scripts/submit_pdc_chain.sh --account <project>
```

The stage chain is serial and uses `afterok` dependencies so each stage gets
its own Slurm walltime allocation.

## Final validation artifacts

The `v0.4.1` PDC validation completed job chain `20008045-20008052` with all
8 stages in `COMPLETED` state. Final summaries are written to:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04/reports/pdc_gmi_validation_summary.json
/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04/reports/pdc_gmi_validation_summary.md
```

The primary QC20 result selects `NIBAN1` and `SORL1`; RNA-only and
no-coordinate controls retain the same genes. Spatial-only is driven by
luminal-like amorphous DCIS composition features, and all-nonempty sensitivity
switches to 11q13 invasive tumor composition, so QC20 remains the primary
biological model.

## S1/S5 contour GeoJSON

The GMI stages require:

```text
xenium_explorer_annotations.s1_s5.generated.geojson
```

If this file is absent from the source cache, `prepare_pdc_inputs.sh` rebuilds
it from the same S1-S5 contour recipe used in the pyXenium tutorial:

- `analysis/analysis/clustering/gene_expression_graphclust/clusters.csv`
- `cells.parquet`
- `WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv`
- HistoSeg source at `${PDC_ROOT}/external/HistoSeg` or an installed
  `histoseg` package

Only the final GeoJSON is copied back into the Xenium source cache; intermediate
contour-generation artifacts stay under `${PDC_ROOT}/contour_generation`.
