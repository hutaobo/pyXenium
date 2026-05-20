# Presubmission inquiry draft

Dear Editors,

We would like to ask whether you would consider a Brief Communication describing TopoLink-CCI, a topology-guided framework for spatial cell-cell interaction inference in whole-transcriptome imaging data.

TopoLink-CCI addresses a central limitation of current CCI analysis: co-expression, cell abundance and spatial proximity can generate plausible but weakly controlled interaction hypotheses. The method integrates tissue topology, expression specificity and local contact into an interpretable discovery score, then evaluates candidates with orthogonal false-positive controls.

In the public 10x Genomics Preview Data: Atera In Situ Gene Expression, FFPE Human Breast Cancer dataset generated using Atera Whole Transcriptome Assay (Atera WTA), TopoLink-CCI generated 1,319,600 CCI hypotheses and prioritized vascular, stromal, immune and Notch axes with strong computational support. In a Synthetic Truth benchmark it achieved AUROC 0.9919 and AUPRC 0.8333, outperforming topology-anchor-only scoring in precision-recall ranking. Cross-dataset application to the corresponding Atera WTA FFPE Human Cervical Cancer preview dataset produced 2,404,971 hypotheses and a distinct top tumor-adhesion axis, supporting tissue-context-specific prioritization.

The accompanying expanded benchmark terminalizes all 18 Breast WTA methods, comprising nine full whole-dataset results and nine bounded subset results with no remaining failure or deferred methods. Bounded results are reported as scalability-aware appendix evidence and remain explicitly separated from full whole-dataset comparisons.

We believe this work fits Nature Methods because it presents a concise method and validation framework for spatial omics, with broad relevance to tissue-scale CCI analysis. The proposed submission would include two main figures, online methods, source data and a complete benchmark status table.

Sincerely,
