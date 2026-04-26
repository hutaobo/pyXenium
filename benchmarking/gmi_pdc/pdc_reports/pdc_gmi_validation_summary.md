# PDC contour-GMI validation summary

- PDC root: `/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04`
- Completed stages: 8/8
- All expected stages complete: true

## Stage results

| Stage | Contours | Features | Main effects | Train AUC | CV mean AUC | Top selected features |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke_contour_top200_spatial50` | 80 | 250 | 0 | 0.5 | - | - |
| `full_contour_top500_spatial100` | 80 | 600 | 2 | 1.0 | - | NIBAN1, SORL1 |
| `full_contour_top500_spatial100_stability` | 80 | 600 | 2 | 1.0 | 1.0 | NIBAN1, SORL1 |
| `validation_rna_only_qc20` | 80 | 500 | 2 | 1.0 | 1.0 | NIBAN1, SORL1 |
| `validation_spatial_only_qc20` | 80 | 100 | 2 | 0.9772727272727273 | 0.935 | omics__whole__state_fraction__luminal_like_amorphous_dcis_cells, omics__core__state_fraction__luminal_like_amorphous_dcis_cells |
| `validation_no_coordinate_qc20` | 80 | 600 | 2 | 1.0 | 1.0 | NIBAN1, SORL1 |
| `sensitivity_top1000_spatial100_qc20` | 80 | 1100 | 2 | 1.0 | 1.0 | EFHD1, SORL1 |
| `sensitivity_all_nonempty_top500_spatial100` | 102 | 600 | 2 | 0.9498987854251013 | 0.9466666666666667 | omics__core__state_fraction__11q13_invasive_tumor_cells, omics__whole__state_fraction__11q13_invasive_tumor_cells |

## Biological readout

- Primary QC20 model selected: NIBAN1, SORL1
- RNA-only retained NIBAN1/SORL1: True
- No-coordinate retained NIBAN1/SORL1: True
- Spatial-only selected: omics__whole__state_fraction__luminal_like_amorphous_dcis_cells, omics__core__state_fraction__luminal_like_amorphous_dcis_cells
- Spatial-only coordinate-driven: False
- Spatial-only composition-driven: True
- Top1000 new features versus QC20: EFHD1
- All-nonempty feature differences versus QC20: NIBAN1, SORL1, omics__core__state_fraction__11q13_invasive_tumor_cells, omics__whole__state_fraction__11q13_invasive_tumor_cells

The QC20 S1/S5 contrast is primarily driven by an S5/DCIS RNA expression program led by NIBAN1 and SORL1 when those features are retained by the full, RNA-only, and no-coordinate stages. Spatial-only signal should be read as contour context, especially composition, unless coordinate features dominate. All-nonempty sensitivity is a QC stress test and does not replace the QC20 primary result.

## Caveats

- QC20 remains the primary result; all-nonempty is a sensitivity analysis.
- GMI is sparse and sample-size sensitive, so selected features should be interpreted with the controls.
- PDC artifacts are on scratch storage and should be archived if long-term retention is needed.
