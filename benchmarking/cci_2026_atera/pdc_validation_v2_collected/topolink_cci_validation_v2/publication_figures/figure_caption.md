# Figure caption

**Figure. TopoLink-CCI discoveries pass multi-layer computational false-positive controls.**
**A**, Schematic of the validation framework. pyXenium nominates candidate ligand-receptor axes, which are then evaluated with validation principles adapted from CellPhoneDB/Squidpy, CellChat, stLearn/SpatialDM, NicheNet, LIANA, and pyXenium-specific robustness checks.
**B**, Evidence matrix across seven biologically interpretable LR axes. Filled circles indicate evidence layers that passed the pre-specified thresholds. All seven axes were classified as strong and none were flagged as contamination/artifact risk.
**C**, Quantitative strength of selected controls: cell-label permutation FDR, spatial-null FDR, matched-gene z-score, and downstream target FDR. The top-ranked axis, VWF-SELP, had CCI_score=0.791, label-permutation FDR=0.002, spatial-null FDR=0.00465, and matched-gene z=6.67.
**D**, Bootstrap and component-ablation robustness. Bootstrap ranks are medians from five 80% stratified resamples; ablation shows the worst rank after removing one pyXenium score component at a time.
**E**, Biological interpretation cards for the retained LR axes. These results support computational credibility, but do not prove protein-level ligand-receptor binding, secretion, or functional causality.

Supplementary Fig. S1 visualizes saved null summaries in standardized z-space; raw permutation draws were not retained. Supplementary Fig. S2 shows per-axis validation cards and weak evidence layers.
