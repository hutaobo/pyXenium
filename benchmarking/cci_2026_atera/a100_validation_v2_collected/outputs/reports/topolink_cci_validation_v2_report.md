# TopoLink-CCI validation v2

This report applies the main computational false-positive controls used by classic LR/CCC papers to TopoLink-CCI axes. The result should be read as computational validation, not wet-lab proof.

## Summary

- strong: 7
- moderate: 0
- hypothesis_only: 0
- artifact_risk: 0

ligand	receptor	sender	receiver	CCI_score	pyxenium_rank	evidence_class	support_count	cell_label_perm_fdr	spatial_null_fdr	matched_gene_z	downstream_target_fdr	cross_method_same_lr_count	bootstrap_rank_median	contamination_flag
VWF	SELP	Endothelial Cells	Endothelial Cells	0.7854081595451057	1	strong	8	0.001996007984031936	0.004651162790697674	6.674917611699146	0.027184466019417475	7	1.0	False
VWF	LRP1	Endothelial Cells	CAFs, DCIS Associated	0.7148067513709494	5	strong	8	0.001996007984031936	0.004651162790697674	11.234674798948532	0.029131985731272295	7	3.0	False
MMRN2	CD93	Endothelial Cells	Endothelial Cells	0.7055697517643245	6	strong	8	0.001996007984031936	0.004651162790697674	6.198278570208171	0.008323424494649227	7	2.0	False
CD48	CD2	T Lymphocytes	T Lymphocytes	0.6870622750035398	13	strong	8	0.001996007984031936	0.011627906976744186	9.860708047779738	0.008323424494649227	7	5.0	False
DLL4	NOTCH3	Endothelial Cells	Pericytes	0.6628108852003319	20	strong	8	0.001996007984031936	0.004651162790697674	8.301941713984622	0.025494276795005204	7	6.0	False
CXCL12	CXCR4	CAFs, DCIS Associated	T Lymphocytes	0.6338817490290192	29	strong	7	0.001996007984031936	0.5813953488372093	11.218977606863426	0.025494276795005204	7	4.0	False
JAG1	NOTCH2	11q13 Invasive Tumor Cells	11q13 Invasive Tumor Cells	0.6167922205938116	41	strong	6	0.001996007984031936	0.004651162790697674	0.3126368933035661	0.6076099881093936	7	7.0	False


## Evidence layers implemented

- CellPhoneDB/Squidpy-style cell-label permutation of ligand/receptor communication probability.
- CellChat-style sender/receiver group specificity and permutation significance.
- stLearn-style spatial-neighborhood LR co-expression plus matched-expression random gene pairs.
- SpatialDM-style spatial expression null based on ligand/receptor neighborhood coupling.
- NicheNet-style receiver target/pathway support using predefined biology panels.
- COMMOT/SpaTalk-style received-signal association with receiver target programs.
- LIANA-style cross-method consensus across completed benchmark methods.
- pyXenium component ablation and stratified bootstrap stability.

## Caveat

The framework reduces false-positive risk, but protein-level receptor binding, secretion, and functional causality require orthogonal experimental validation.
