# Whole-dataset LR benchmarking

## Overview

This tutorial summarizes whole-dataset ligand-receptor benchmarking on the full
Atera Xenium WTA breast sample (`170,057` cells). The clean PDC `full_common`
runs used a shared human ligand-receptor resource so that methods are compared
by recovered biology and method-internal rank behavior, not by raw score
magnitude.

The benchmark is separate from the basic ligand-receptor tutorial, which focuses
on the fixed smoke/topology panel and the `pyXenium.ligand_receptor` workflow.

## Completed full common-db methods

| Method | Full common-db rows | Highest-level signal recovered |
|---|---:|---|
| `pyXenium` | 1,319,600 | Vascular and topology-supported stromal axes, led by `VWF-SELP`, `VWF-LRP1`, and endothelial-to-CAF/pericyte programs. |
| `CellPhoneDB` | 1,183,456 | Reproducible non-spatial expression baseline, led by `CCN2-ITGB2`, `VWF-LRP1`, and CAF/endothelial/immune expression programs. |
| `LARIS` | 1,304,935 | Diffusion-smoothed tumor-stroma signals, led by `PLAT-LRP1` and `GNAS-ADCY1`. |
| `LIANA+` | 744,209 | Spatial bivariate signals including `CXCL12-CD4`, `HMGB1-THBD`, and `ICOSLG-CTLA4`; top hits require caution because several involve `Unassigned` cells. |
| `SpatialDM` | 446,023 | Spatial co-expression signals dominated by tumor-intrinsic epithelial interactions such as `CDH1-IGF1R`. |
| `stLearn` | 505,281 | Local neighborhood CCI signals dominated by tumor-intrinsic high-expression interactions such as `ADAM17-MUC1`. |

## Canonical Atera axis recovery

| Canonical Atera axis | Benchmark recovery |
|---|---|
| `CXCL12-CXCR4` CAF/DCIS to T cells | Strongly recovered. `pyXenium` ranked the expected `CAFs, DCIS Associated -> T Lymphocytes` interaction at rank 28, with `CellPhoneDB` and `LARIS` also recovering the same sender-receiver axis near the top of their full results. |
| `DLL4-NOTCH3` endothelial to pericytes | Strongly recovered by `pyXenium` at rank 24 with the expected `Endothelial Cells -> Pericytes` direction, and also recovered by `CellPhoneDB` and `LARIS`. |
| `JAG1-NOTCH1` tumor/stromal Notch | Recovered by multiple methods, but with method-dependent receiver compartments. `pyXenium` prioritized a tumor/DCIS axis, while `CellPhoneDB` and `LARIS` favored tumor-to-endothelial interpretations. |
| `CSF1-CSF1R` stromal to macrophages | Not recovered in the clean full common-db outputs; this should be interpreted as a database/expression/filtering limitation rather than proof that the macrophage axis is absent. |
| `TGFB1-TGFBR2` endothelial/stromal TGF-beta | Not recovered in the clean full common-db outputs, again suggesting panel detectability or common-resource limitations. |

## Biological interpretation

Overall, `pyXenium` gave the strongest topology-supported biological discovery
profile because it recovered the expected `CXCL12-CXCR4` and `DLL4-NOTCH3`
axes with the most anatomically plausible sender-receiver assignments.
`CellPhoneDB` is the most useful reproducible non-spatial baseline, and
`LARIS` is a strong diffusion-aware complement.

`SpatialDM` and `stLearn` are best read as supplementary spatial co-expression
methods in this dataset because their top ranks are dominated by tumor-intrinsic
high-expression programs. `LIANA+` produced biologically interesting spatial
bivariate hits, but the strongest calls require caution because several involve
the `Unassigned` compartment.

## Caveats

- Scores are standardized within each method; raw scores are not directly
  comparable across methods.
- This page reports clean PDC `full_common` outputs only, not stale A100 salvage
  runs or smoke-only results.
- Non-recovery of a canonical axis in the common-db benchmark can reflect LR
  resource coverage, expression filtering, or panel detectability rather than
  biological absence.

## Next steps

- Use `pyXenium` when topology-supported biological discovery is the priority.
- Use `CellPhoneDB` as the reproducible non-spatial expression baseline.
- Use `LARIS`, `SpatialDM`, `stLearn`, and `LIANA+` as complementary views whose
  discoveries should be interpreted through their method-specific assumptions.
