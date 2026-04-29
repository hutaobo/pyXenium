# Presubmission Inquiry Draft

**Target:** Nature Methods
**Article type requested:** Brief Communication
**Proposed title:** Topology-guided inference of spatial cell-cell interaction axes

Dear Nature Methods editors,

We would like to ask whether the manuscript described below would be suitable for consideration as a Brief Communication.

Spatial cell-cell interaction and cell-cell communication analysis in spatial transcriptomics is highly susceptible to false positives from co-expression, cell-type abundance, database ambiguity and spatial proximity confounding. We developed **TopoLink-CCI**, a topology-guided spatial cell-cell interaction framework that ranks molecular interaction axes by integrating tissue topology, sender and receiver expression specificity, sender-receiver structural bridging and local spatial contact. We then evaluate candidate axes with orthogonal computational false-positive controls adapted from classical cell-cell communication methods, including cell-label permutation, spatial nulls, matched-gene controls, downstream target support, cross-method consensus, component ablation and bootstrap stability.

Applied to a 170,057-cell Xenium WTA breast cancer dataset, TopoLink-CCI generated 1,319,600 sender-receiver interaction hypotheses and prioritized interpretable vascular, stromal, immune and Notch axes. Seven representative axes, including VWF-SELP, VWF-LRP1, MMRN2-CD93, DLL4-NOTCH3, CXCL12-CXCR4, CD48-CD2 and JAG1-NOTCH2, received strong computational support under independent false-positive controls. The top-ranked VWF-SELP endothelial-endothelial axis was interpreted as a topology-supported endothelial activation and vascular adhesion niche, while explicitly avoiding claims of protein-level signaling or causality.

We believe this work fits Nature Methods because it introduces a concise computational method for a current spatial omics bottleneck: how to prioritize cell-cell interaction hypotheses without treating co-expression or database membership as proof of communication. The manuscript is planned as a Brief Communication with two main figures, an Online Methods section, reproducible code and supplementary validation tables.

Thank you for considering this presubmission inquiry.

Sincerely,

[Authors]
