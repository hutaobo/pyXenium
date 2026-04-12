# Figure Legends

**Figure 1. Evidence-backed summary of pyXenium loading and validation.**  
**(A)** Real-data smoke-test summary for the public 10x Genomics FFPE human renal cell carcinoma Xenium RNA+Protein dataset. The automatic loader path and explicit HDF5 path produced identical summaries, recovering 465,545 cells, 405 RNA features and 27 protein markers, while preserving spatial coordinates and cluster labels and returning no validation issues.  
**(B)** Top five RNA features by total counts in the validated `prefer="auto"` smoke test. Bars show total counts in millions, and text annotations report the number of cells with non-zero counts for each feature.  
**(C)** Top five protein markers by mean signal in the validated `prefer="auto"` smoke test. Bars show mean protein signal, and text annotations report the number of positive cells for each marker.  
**(D)** Additional evidence from the same repository validation workflow. Left: sizes of the five largest graph-based clusters recovered from the validated renal dataset. Right: feature-type composition of the real MEX-only partial load, which returned a 465,545 x 543 counts object containing gene-expression, protein-expression and control features without requiring optional spatial or clustering attachments.
