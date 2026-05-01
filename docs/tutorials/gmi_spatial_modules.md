# GMI spatial gene modules

This tutorial documents the WTA breast spatial module layer built on top of
`pyXenium.gmi`. The module workflow starts from a completed contour-GMI run and
does not rerun the vendored R `Gmi` fit.

```bash
pyxenium gmi modules \
  --gmi-output-dir pyxenium_gmi_outputs/full_contour_top500_spatial100 \
  --output-dir pyxenium_gmi_outputs/full_contour_top500_spatial100/modules
```

## Biological question

The primary question is whether the validated S1/S5 GMI effects form a coherent
spatial gene module. In the Atera WTA breast task, `S1` is the invasive
tumor/CAF endpoint and `S5` is the apocrine-luminal DCIS endpoint.

The module validation asks:

- whether `NIBAN1` and `SORL1` collapse into one S5/DCIS module;
- whether RNA-only and no-coordinate controls preserve that module;
- whether spatial-only modules are driven by composition, rim/edge, CAF/ECM,
  vascular/pericyte, immune context, or coordinate features;
- whether top1000 and all-nonempty sensitivity runs change the QC20 conclusion.

## Method

Each supervised GMI module is seeded by selected or bootstrap-stable GMI main
effects. The seed is expanded with three evidence sources:

- feature correlation across retained contours;
- contour-neighborhood spatial-lag correlation;
- GMI interaction edges from `interaction_effects.tsv`.

The module score is written per contour, oriented toward the endpoint in
`score_high_label`, and summarized with curated pathway overlaps and Moran's I /
Geary's C spatial autocorrelation.

## Outputs

Each module run writes a `modules/` directory containing:

- `spatial_modules.tsv`
- `module_features.tsv`
- `module_scores.tsv.gz`
- `module_enrichment.tsv`
- `module_interactions.tsv`
- `module_spatial_autocorr.tsv`
- `effect_graph_nodes.tsv`
- `effect_graph_edges.tsv`
- `summary.json`
- `report.md`
- optional score maps under `figures/`

## PDC validation

The final WTA breast module tutorial is generated from a fresh PDC Dardel
8-stage run under:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_modules_2026-04-30
```

The stages are:

| Stage | Purpose |
| --- | --- |
| `smoke_contour_top200_spatial50` | Fast smoke run and module artifact check |
| `full_contour_top500_spatial100` | Primary QC20 RNA + spatial model |
| `full_contour_top500_spatial100_stability` | Spatial CV, bootstrap, and controls |
| `validation_rna_only_qc20` | RNA-only module control |
| `validation_spatial_only_qc20` | Spatial-feature-only module control |
| `validation_no_coordinate_qc20` | Coordinate-confounding control |
| `sensitivity_top1000_spatial100_qc20` | Expanded RNA feature budget |
| `sensitivity_all_nonempty_top500_spatial100` | Low-cell contour sensitivity |

## Current validation status

The fresh PDC module validation is running or pending. Once all 8 stages finish,
this page will be updated with the final stage table, selected modules, module
enrichment, spatial autocorrelation, representative score maps, and biological
interpretation.

