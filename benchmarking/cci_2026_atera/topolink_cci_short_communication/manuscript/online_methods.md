# Online Methods Draft

## Method Overview

TopoLink-CCI is a topology-guided spatial cell-cell interaction scoring framework implemented in pyXenium. In the analysis reported here, TopoLink-CCI was run in cell-cell interaction-resource mode, where each candidate consists of a ligand or interaction-source gene, a receptor or interaction-target gene, a sender cell group and a receiver cell group. The method ranks candidate molecular interaction axes by combining tissue topology, sender and receiver expression support, sender-receiver structural compatibility and local spatial contact.

## Dataset

The primary dataset was the Atera Xenium WTA FFPE breast cancer dataset. The full benchmark object contained 170,057 cells, 18,028 RNA features and 20 annotated cell groups. Gene symbols were taken from the Xenium feature metadata. Spatial coordinates were taken from the cell-level Xenium output. The main benchmark used a common interaction resource containing 3,299 cell-cell interaction or molecular interaction pairs to support cross-method comparison.

## Topology Inputs

TopoLink-CCI uses pyXenium topology outputs that encode gene-to-cell-type anchoring and cell-type-to-cell-type structural relationships. For a ligand or source gene \(l\), receptor or target gene \(r\), sender cell group \(s\) and receiver cell group \(t\), the topology maps provide ligand-sender distance, receptor-receiver distance and sender-receiver structure distance. These distances are converted into support scores by subtracting each distance from 1 after normalization.

## Expression Support

For each gene and cell group, expression support combines pseudobulk expression share and detection fraction. If \(P(g,c)\) is the fraction of total gene expression assigned to cell group \(c\), and \(F(g,c)\) is the fraction of cells in group \(c\) in which gene \(g\) is detected, the expression support is:

\[
E(g,c) = \mathrm{rowNorm}\left(P(g,c)\sqrt{F(g,c)}\right)
\]

This favors genes that are both abundant and recurrently detected in the relevant cell group, while preserving cell-type specificity.

## Local Contact Support

Local contact support is computed from a cell-cell spatial neighbor graph. In the full benchmark, spatial neighborhoods were constructed from cell coordinates using the configured pyXenium neighbor settings. For each sender-receiver group pair and each candidate gene pair, TopoLink-CCI computes local edge strength from normalized source-gene expression in sender cells and normalized target-gene expression in receiver cells. It also records active-edge coverage, local edge count and an edge-count support term. Candidate contacts with insufficient sender-receiver edges are assigned zero contact support.

## TopoLink-CCI Score

For candidate \((l,r,s,t)\), the final score is:

\[
\mathrm{TopoLink\mbox{-}CCI}_{l,r,s,t}
=
\pi_{l,r}
\times
\mathrm{GM}
\left[
A_{\mathrm{sender}},
A_{\mathrm{receiver}},
B_{\mathrm{structure}},
E_{\mathrm{sender}},
E_{\mathrm{receiver}},
C_{\mathrm{local}}
\right]
\]

where \(A_{\mathrm{sender}}\) is ligand-sender topology support, \(A_{\mathrm{receiver}}\) is receptor-receiver topology support, \(B_{\mathrm{structure}}\) is sender-receiver structural support, \(E_{\mathrm{sender}}\) and \(E_{\mathrm{receiver}}\) are expression support terms, \(C_{\mathrm{local}}\) is local contact support and \(\pi_{l,r}\) is the optional prior confidence assigned to the molecular interaction pair. The geometric mean ensures that a high score requires broad support across topology, expression and contact components.

## Benchmarking

The full common-database TopoLink-CCI run generated 1,319,600 sender-receiver candidate axes. Standardized benchmark outputs were compared with completed full or bounded method outputs from CellPhoneDB, LIANA+, SpatialDM, stLearn, LARIS and related adapters. Raw scores were not compared directly across methods. Instead, each method was normalized within method and interpreted through rank, theme recovery, canonical axis recovery and output provenance.

## False-Positive Controls

Seven biologically interpretable TopoLink-CCI axes were selected for targeted computational validation: VWF-SELP, VWF-LRP1, MMRN2-CD93, CD48-CD2, DLL4-NOTCH3, CXCL12-CXCR4 and JAG1-NOTCH2. The validation framework implemented the main computational principles used in classical cell-cell interaction and cell-cell communication methods:

- Cell-label permutation of sender-receiver communication probability.
- Group specificity checks analogous to CellChat-style cell-group communication scoring.
- Spatial-neighborhood coupling and spatial null controls inspired by spatial CCC methods.
- Matched-expression random gene-pair negative controls.
- Receiver target and pathway support inspired by ligand-target frameworks.
- Received-signal association with receiver target programs.
- Cross-method consensus across benchmarked methods.
- TopoLink-CCI component ablation.
- Stratified bootstrap stability.

Candidate axes were assigned an evidence class based on the number and type of independent evidence layers. `strong` required at least five independent support layers and no contamination or label-artifact flag. `moderate` indicated three to four support layers. `hypothesis_only` indicated limited support outside TopoLink-CCI score. `artifact_risk` indicated failed null controls or contamination risk.

## Lead Biological Axis

The lead axis, VWF-SELP from endothelial cells to endothelial cells, had a TopoLink-CCI score of 0.791289 and rank 1 among all full common-database candidates. Its component values were: sender anchor 0.955713, receiver anchor 0.881913, structure bridge 1.000000, sender expression 1.000000, receiver expression 1.000000, local contact 0.291245 and prior confidence 1.000000. This axis was interpreted as an endothelial activation and vascular adhesion niche. The interpretation is RNA-level and spatially supported; it does not prove protein-level VWF secretion, P-selectin surface presentation or platelet or leukocyte adhesion.

## Reproducibility

Primary outputs used for the manuscript are stored under:

`benchmarking/cci_2026_atera/pdc_collected/pdc_20260426_1327/`

Validation outputs used for false-positive control summaries are stored under:

`benchmarking/cci_2026_atera/pdc_validation_v2_collected/topolink_cci_validation_v2/`

The figure drafts in this manuscript package are regenerated by:

```bash
python benchmarking/cci_2026_atera/topolink_cci_short_communication/scripts/make_topolink_cci_nature_brief_figures.py
```
