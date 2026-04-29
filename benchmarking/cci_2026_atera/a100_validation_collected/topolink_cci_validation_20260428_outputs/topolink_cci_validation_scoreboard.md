# TopoLink-CCI validation evidence scoreboard

This PDC run applies the computational false-positive-control logic used by classic CCC/LR papers: expression specificity, spatial/label nulls, matched-gene controls, component ablation, cross-method triangulation, and receiver-context support. It is computational evidence, not wet-lab proof.

## Evidence classes

- Strong axes: 6
- Moderate axes: 1
- Hypothesis-only axes: 0

axis_label	biology_label	CCI_score	global_rank	evidence_class	support_count	cross_method_same_lr_count	matched_gene_percentile	cross_edge_enrichment_z	max_rank_after_component_removal
"VWF-SELP
Endothelial Cells -> Endothelial Cells"	WPB / endothelial activation	0.7854081595451057	1	strong	6	7	1.0	157.09499064265975	16
"VWF-LRP1
Endothelial Cells -> CAFs, DCIS Associated"	vascular-stromal matrix/scavenger axis	0.7148067513709494	5	strong	5	7	1.0	40.577088952341676	614
"MMRN2-CD93
Endothelial Cells -> Endothelial Cells"	CD93-MMRN2 angiogenesis	0.7055697517643245	6	strong	6	7	1.0	157.09499064265975	13
"CD48-CD2
T Lymphocytes -> T Lymphocytes"	T-cell adhesion/co-stimulation	0.6870622750035398	13	strong	6	7	1.0	331.09056328473696	20
"DLL4-NOTCH3
Endothelial Cells -> Pericytes"	endothelial-pericyte Notch	0.6628108852003319	20	strong	5	7	1.0	141.55074144031974	76
"CXCL12-CXCR4
CAFs, DCIS Associated -> T Lymphocytes"	CAF-immune chemokine recruitment	0.6338817490290192	29	strong	5	7	1.0	24.92068208259856	615
"JAG1-NOTCH2
11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells"	tumor-intrinsic Notch signaling	0.6167922205938116	41	moderate	4	7	0.576	567.2684147856648	329


## Control interpretation

- `spatial_abundance_null`: compares observed sender-receiver graph edges with a label-abundance null; it is conservative about exact geometry but catches cell-type-pair edge enrichment.
- `matched_gene_control`: compares ligand/receptor sender-receiver expression specificity with genes matched by global mean and detection.
- `lr_label_permutation`: reports where the candidate ranks within the same sender-receiver cell-type pair and within the same LR pair across cell-type pairs.
- `component_ablation`: recomputes geometric-mean LR scores after removing one pyXenium component at a time.

## References to validation patterns

- CellPhoneDB: curated LR database, complex filtering, and cell-label permutation specificity.
- CellChat: mass-action communication probability, curated cofactors/complexes, and label permutation.
- NicheNet: downstream receiver target-gene support rather than co-expression alone.
- stLearn/SpatialDM/Squidpy: spatially constrained LR evidence and permutation/random-pair controls.
- LIANA benchmark: multi-method consensus and rank aggregation because no single LR score is ground truth.
