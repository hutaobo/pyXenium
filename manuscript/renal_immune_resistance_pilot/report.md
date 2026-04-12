# Renal Immune Resistance Discovery Package

Dataset: Xenium In Situ Gene and Protein Expression data for FFPE Human Renal Cell Carcinoma
Source: https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma
Local path: `Y:\long\10X_datasets\Xenium\Xenium_Renal\Xenium_V1_Human_Kidney_FFPE_Protein`
Backend preference: `auto`

## Core Summary

- Cells: `465545`
- RNA features: `405`
- Protein markers: `27`
- Resistant niches tracked: `myeloid_vascular, epithelial_emt_front`
- Positioning: Branch-specific axis models localize the pilot signal more clearly than pooled summaries; the current claim should emphasize localization rather than universal superiority.

## Joint Cell Classes

- `stromal`: `182783` cells
- `immune`: `128563` cells
- `tumor`: `97202` cells
- `unassigned`: `56997` cells

## Joint Cell States

- `endothelial_perivascular`: `182783` cells
- `unassigned`: `64919` cells
- `tumor_epithelial`: `47704` cells
- `emt_like_tumor`: `45017` cells
- `t_cell_exhausted_cytotoxic`: `41864` cells
- `macrophage_like`: `41765` cells

## Top Marker Discordance

- `alpha_sma` (alphaSMA / ACTA2): mean abs discordance `1.460`, signed bias `0.851`
- `vimentin` (Vimentin / VIM): mean abs discordance `1.348`, signed bias `0.294`
- `e_cadherin` (E-Cadherin / CDH1): mean abs discordance `1.160`, signed bias `-0.382`
- `hla_dr` (HLA-DR / HLA-DRA): mean abs discordance `1.114`, signed bias `0.448`
- `vista` (VISTA / VSIR): mean abs discordance `0.933`, signed bias `0.140`
- `cd68` (CD68 / CD68): mean abs discordance `0.773`, signed bias `0.298`

## Top Pathway Discordance

- `vascular_stromal`: mean abs discordance `0.888`, signed bias `0.426`
- `epithelial_emt`: mean abs discordance `0.767`, signed bias `-0.029`
- `myeloid_activation`: mean abs discordance `0.649`, signed bias `0.249`
- `checkpoint`: mean abs discordance `0.501`, signed bias `0.035`
- `lymphoid_effector`: mean abs discordance `0.323`, signed bias `0.000`

## Branch Summary

- `epithelial_emt_front`: best model `emt_only`, benchmark `0.664`, held-out AUC `0.684`
- `myeloid_vascular`: best model `vascular_only`, benchmark `0.652`, held-out AUC `0.544`

## Top Hypotheses

- `epithelial_emt_front`: Checkpoint-active myeloid enrichment at EMT transition fronts (marker `PanCK`, benchmark `0.664`, validation: mIF/RNAscope for PanCK, Vimentin, alphaSMA, and checkpoint-active myeloid markers at EMT transition fronts.)
- `myeloid_vascular`: Perivascular immune-resistance belt (marker `CD163`, benchmark `0.652`, validation: mIF/IHC for CD68 or CD163 with CD31 and alphaSMA, plus VISTA or HLA-DR at perivascular belts.)
