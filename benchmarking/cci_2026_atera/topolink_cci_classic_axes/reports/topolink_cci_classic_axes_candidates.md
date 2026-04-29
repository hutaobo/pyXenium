# pyXenium high-scoring classic LR candidates

Input table: `D:\GitHub\pyXenium\benchmarking\cci_2026_atera\pdc_collected\pdc_20260426_1327\runs\full_common\pyxenium\pyxenium_scores.tsv`
Rows scanned: `1,319,600`

Selection strategy: candidates are sorted by `CCI_score` and filtered to predefined cell-cell interaction pairs with established or strongly interpretable biology. The score is not changed; biological interpretability is an annotation layer.

## Headline candidates

| Rank | CCI pair | Sender -> Receiver | CCI_score | Category | Interpretation |
|---:|---|---|---:|---|---|
| 1 | `VWF-SELP` | Endothelial Cells -> Endothelial Cells | 0.791 | WPB / endothelial activation | Endothelial VWF-SELP suggests a Weibel-Palade body/P-selectin-like vascular adhesion state. |
| 2 | `VWF-LRP1` | Endothelial Cells -> CAFs, DCIS Associated | 0.748 | vascular-stromal matrix/scavenger axis | Endothelial VWF toward CAF LRP1 highlights a vascular-to-stromal matrix/scavenger-receptor interface. |
| 3 | `EFNB2-PECAM1` | Endothelial Cells -> Endothelial Cells | 0.747 | endothelial identity / vascular adhesion | Endothelial EFNB2-PECAM1 supports a vascular identity and endothelial adhesion/state axis. |
| 4 | `MMRN2-CLEC14A` | Endothelial Cells -> Endothelial Cells | 0.732 | tumor endothelial angiogenic complex | MMRN2-CLEC14A points to tumor endothelial matrix remodeling and angiogenic vessel state. |
| 5 | `HSPG2-LRP1` | Endothelial Cells -> CAFs, DCIS Associated | 0.728 | basement membrane / ECM receptor | Endothelial HSPG2 to CAF LRP1 suggests basement-membrane-rich vascular-stromal remodeling. |
| 6 | `COL4A2-CD93` | Endothelial Cells -> Endothelial Cells | 0.712 | basement membrane angiogenesis | COL4A2-CD93 links endothelial basement membrane signal with angiogenic endothelial receptor state. |
| 7 | `MMRN2-CD93` | Endothelial Cells -> Endothelial Cells | 0.711 | CD93-MMRN2 angiogenesis | MMRN2-CD93 is a direct interpretable tumor-endothelial matrix/angiogenesis axis. |
| 8 | `VWF-ITGA9` | Endothelial Cells -> Endothelial Cells | 0.708 | vascular adhesion / integrin | VWF-ITGA9 extends the VWF vascular adhesion theme within endothelial neighborhoods. |
| 9 | `MMP2-PECAM1` | CAFs, DCIS Associated -> Endothelial Cells | 0.697 | CAF-endothelial remodeling | CAF MMP2 toward endothelial PECAM1 supports a stromal-remodeling interface around vasculature. |
| 10 | `CD48-CD2` | T Lymphocytes -> T Lymphocytes | 0.685 | T-cell adhesion/co-stimulation | T-cell CD48-CD2 indicates lymphocyte-local adhesion/co-stimulation biology. |
| 11 | `VEGFC-FLT1` | Endothelial Cells -> Endothelial Cells | 0.684 | VEGF/angiogenesis | VEGFC-FLT1 supports a vascular growth-factor/angiogenesis interpretation. |
| 12 | `DLL4-NOTCH3` | Endothelial Cells -> Pericytes | 0.670 | endothelial-pericyte Notch | Endothelial DLL4 to pericyte NOTCH3 supports vascular stabilization and pericyte-endothelial Notch biology. |
| 13 | `CXCL12-CXCR4` | CAFs, DCIS Associated -> T Lymphocytes | 0.662 | canonical CAF-immune chemokine axis | CAF CXCL12 to T-cell CXCR4 recovers a canonical stromal chemokine immune-recruitment axis. |
| 14 | `THBS2-CD36` | CAFs, DCIS Associated -> Endothelial Cells | 0.653 | stromal matrix angiogenesis | CAF THBS2 to endothelial CD36 supports matrix-rich stromal angiogenesis biology. |
| 15 | `JAG1-NOTCH2` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.634 | tumor Notch signaling | Tumor-intrinsic JAG1-NOTCH2 suggests a Notch signaling state within 11q13 invasive tumor cells. |

## Biology groups

### vascular WPB
- `VWF-SELP` (Endothelial Cells -> Endothelial Cells, CCI_score=0.791): VWF and P-selectin are core endothelial Weibel-Palade body cargos linked to hemostasis, inflammation, and leukocyte rolling.
- `VWF-ITGA9` (Endothelial Cells -> Endothelial Cells, CCI_score=0.708): Interpret as an integrin-associated vascular adhesion hypothesis supported by endothelial spatial contact.

### vascular identity
- `EFNB2-PECAM1` (Endothelial Cells -> Endothelial Cells, CCI_score=0.747): Both genes mark vascular/endothelial biology; interpret as endothelial organization rather than secreted paracrine signaling.

### vascular stromal ECM
- `VWF-LRP1` (Endothelial Cells -> CAFs, DCIS Associated, CCI_score=0.748): This is a high-scoring dataset-supported vascular-stromal hypothesis rather than a single canonical pathway claim.
- `HSPG2-LRP1` (Endothelial Cells -> CAFs, DCIS Associated, CCI_score=0.728): LRP1 frequently acts as a scavenger/ECM-associated receptor; here the strongest support is spatial WTA context.

### angiogenesis matrix
- `MMRN2-CLEC14A` (Endothelial Cells -> Endothelial Cells, CCI_score=0.732): CD93/CLEC14A/MMRN2 family biology is linked to tumor vasculature, matrix organization, and endothelial angiogenic programs.
- `COL4A2-CD93` (Endothelial Cells -> Endothelial Cells, CCI_score=0.712): CD93 is implicated in endothelial matrix organization and tumor angiogenesis.
- `MMRN2-CD93` (Endothelial Cells -> Endothelial Cells, CCI_score=0.711): CD93-MMRN2 signaling has reported roles in beta1 integrin activation and fibronectin fibrillogenesis during tumor angiogenesis.

### angiogenesis growth factor
- `VEGFC-FLT1` (Endothelial Cells -> Endothelial Cells, CCI_score=0.684): Canonical VEGF-family CCI biology makes this a straightforward angiogenic axis.

### CAF ECM remodeling
- `MMP2-PECAM1` (CAFs, DCIS Associated -> Endothelial Cells, CCI_score=0.697): MMP2 is a classic ECM remodeling enzyme; PECAM1 anchors the receiver identity as endothelial.
- `THBS2-CD36` (CAFs, DCIS Associated -> Endothelial Cells, CCI_score=0.653): THBS/CD36 biology is interpretable in matrix remodeling, angiogenesis, and stromal context.

### Notch pericyte
- `DLL4-NOTCH3` (Endothelial Cells -> Pericytes, CCI_score=0.670): Notch1/Notch3 and DLL4-linked endothelial-pericyte signaling are reported in vascular stabilization contexts.

### tumor Notch
- `JAG1-NOTCH2` (11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells, CCI_score=0.634): JAG1/Notch signaling is broadly linked to cancer cell state, angiogenesis, stemness, EMT, and therapy resistance.

### immune recruitment
- `CXCL12-CXCR4` (CAFs, DCIS Associated -> T Lymphocytes, CCI_score=0.662): CXCL12/CXCR4 is a well-studied cancer-stroma chemokine pathway with immune and tumor-microenvironment relevance.

### T cell signaling
- `CD48-CD2` (T Lymphocytes -> T Lymphocytes, CCI_score=0.685): CD48-CD2 has experimental support in T-cell adhesion and activation contexts.

## High-score non-classic caveats

These rows scored highly but were not used as headline classic LR discoveries because their interpretation is less clean as an extracellular cell-cell interaction axis.

| Global rank | CCI pair | Sender -> Receiver | CCI_score | Reason |
|---:|---|---|---:|---|
| 4 | `GNAS-ADCY1` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.742 | High score but more intracellular second-messenger pathway than interpretable extracellular LR. |
| 5 | `CDH1-IGF1R` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.741 | Interesting tumor epithelial adhesion/growth receptor context, but less clean as a classic LR headline. |
| 8 | `CDH1-PTPRF` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.713 | Cell adhesion/phosphatase context; biologically plausible but not a primary classic LR discovery. |
| 12 | `DSC3-DSG3` | Basal-like Structured DCIS Cells -> Basal-like Structured DCIS Cells | 0.705 | Desmosomal adhesion within DCIS; important structure biology but not a classic cell-cell interaction signaling axis. |
| 22 | `DSG1-DSC3` | Basal-like Structured DCIS Cells -> Basal-like Structured DCIS Cells | 0.673 | Desmosomal adhesion within DCIS; important structure biology but not a classic cell-cell interaction signaling axis. |
| 23 | `JAG1-CD46` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.672 | Potential tumor/complement regulatory candidate; lower confidence than canonical JAG1-NOTCH. |
| 31 | `PSAP-CELSR1` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.653 | Potentially interpretable but not prioritized as a classic cancer LR axis here. |
| 34 | `APP-LRP10` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.645 | High-score tumor-intrinsic axis; less directly tied to canonical spatial LR interpretation for this report. |
| 35 | `ADAM10-NOTCH2` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.643 | Notch proteolysis context rather than a cell-cell interaction pair; kept as caveat not headline. |
| 36 | `ADAM15-ITGB1` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.643 | Adhesion/protease-integrin biology; plausible but less classic than selected ECM/vascular axes. |
| 43 | `SPTAN1-PTPRA` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells | 0.635 | Cytoskeletal/phosphatase context; not treated as a classic extracellular LR headline. |
| 44 | `TNC-SDC4` | Basal-like Structured DCIS Cells -> 11q13 Invasive Tumor Cells | 0.634 | ECM-syndecan axis is plausible but not in the preregistered classic headline set. |
| 232 | `APP-LRP10` | Basal-like Structured DCIS Cells -> 11q13 Invasive Tumor Cells | 0.544 | High-score tumor-intrinsic axis; less directly tied to canonical spatial LR interpretation for this report. |
| 274 | `TNC-SDC4` | Myoepithelial Cells -> 11q13 Invasive Tumor Cells | 0.532 | ECM-syndecan axis is plausible but not in the preregistered classic headline set. |
| 517 | `GNAS-ADCY1` | 11q13 Invasive Tumor Cells -> 11q13 Invasive Tumor Cells (Mitotic) | 0.489 | High score but more intracellular second-messenger pathway than interpretable extracellular LR. |

## References

- Weibel-Palade Bodies, NCBI Bookshelf: https://www.ncbi.nlm.nih.gov/books/NBK535353/
- Interpreted as ECM/scavenger-receptor biology in the pyXenium WTA context; use as dataset-supported hypothesis.
- Reactome VEGFA-VEGFR2 pathway and VEGF-family angiogenesis context: https://reactome.org/content/detail/R-HSA-4420097
- CD93 promotes beta1 integrin activation and fibronectin fibrillogenesis during tumor angiogenesis: https://pmc.ncbi.nlm.nih.gov/articles/PMC6063507/
- Enhanced murine CD4+ T cell responses induced by the CD2 ligand CD48: https://pubmed.ncbi.nlm.nih.gov/9862369/
- Notch1 and Notch3 coordinate for pericyte-induced stabilization of vasculature: https://pubmed.ncbi.nlm.nih.gov/34878922/
- Chemokine signaling in cancer-stroma communications: https://pmc.ncbi.nlm.nih.gov/articles/PMC8222467/
- The Notch ligand Jagged1 as a target for anti-tumor therapy: https://pmc.ncbi.nlm.nih.gov/articles/PMC4174884/

## Output files

- `tables/topolink_cci_classic_axes_candidates.tsv`: headline representatives.
- `tables/topolink_cci_classic_axes_candidates_all.tsv`: all matching classic LR rows across sender-receiver contexts.
- `tables/pyxenium_high_score_nonclassic_caveats.tsv`: high-scoring rows deliberately not used as headline classic LR discoveries.
- `figures/topolink_cci_classic_axes_top_axes.png`: compact ranked figure.
