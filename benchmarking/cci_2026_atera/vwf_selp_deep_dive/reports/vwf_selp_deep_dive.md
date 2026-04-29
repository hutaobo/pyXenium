# VWF-SELP endothelial-endothelial deep dive

## Working conclusion

The top pyXenium full-common hit, `VWF-SELP / Endothelial Cells -> Endothelial Cells`, is best interpreted as an endothelial vascular activation and adhesion state rather than a simple classical paracrine cell-cell interaction axis. The score is high because topology, expression, endothelial-endothelial structural proximity, and local contact all support the same vascular compartment.

## Direct pyXenium evidence

- `CCI_score`: `0.791289236801`
- `sender_anchor`: `0.955713`
- `receiver_anchor`: `0.881913`
- `structure_bridge`: `1.000000`
- `sender_expr`: `1.000000`
- `receiver_expr`: `1.000000`
- `local_contact`: `0.291245`
- `contact_coverage`: `0.111120`
- `cross_edge_count`: `12779`

### Component decomposition

| component        |    value |   geomean_penalty | target_lr   | target_sender     | target_receiver   |   prior_confidence |   reported_lr_score |   recomputed_lr_score |
|:-----------------|---------:|------------------:|:------------|:------------------|:------------------|-------------------:|--------------------:|----------------------:|
| sender_anchor    | 0.955713 |         0.0452975 | VWF-SELP    | Endothelial Cells | Endothelial Cells |                  1 |            0.791289 |              0.791289 |
| receiver_anchor  | 0.881913 |         0.125662  | VWF-SELP    | Endothelial Cells | Endothelial Cells |                  1 |            0.791289 |              0.791289 |
| structure_bridge | 1        |        -1e-08     | VWF-SELP    | Endothelial Cells | Endothelial Cells |                  1 |            0.791289 |              0.791289 |
| sender_expr      | 1        |        -1e-08     | VWF-SELP    | Endothelial Cells | Endothelial Cells |                  1 |            0.791289 |              0.791289 |
| receiver_expr    | 1        |        -1e-08     | VWF-SELP    | Endothelial Cells | Endothelial Cells |                  1 |            0.791289 |              0.791289 |
| local_contact    | 0.291245 |         1.23359   | VWF-SELP    | Endothelial Cells | Endothelial Cells |                  1 |            0.791289 |              0.791289 |


### Rank sensitivity

| scenario                  |   target_score |   target_rank | top_ligand   | top_receptor   | top_sender        | top_receiver      |   top_score |
|:--------------------------|---------------:|--------------:|:-------------|:---------------|:------------------|:------------------|------------:|
| original                  |       0.791289 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.791289 |
| remove_sender_anchor      |       0.761968 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.761968 |
| set_sender_anchor_to_1    |       0.797286 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.797286 |
| remove_receiver_anchor    |       0.774314 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.774314 |
| set_receiver_anchor_to_1  |       0.808037 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.808037 |
| remove_structure_bridge   |       0.755096 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.755096 |
| set_structure_bridge_to_1 |       0.791289 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.791289 |
| remove_sender_expr        |       0.755096 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.755096 |
| set_sender_expr_to_1      |       0.791289 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.791289 |
| remove_receiver_expr      |       0.755096 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.755096 |
| set_receiver_expr_to_1    |       0.791289 |             1 | VWF          | SELP           | Endothelial Cells | Endothelial Cells |    0.791289 |
| remove_local_contact      |       0.966386 |            13 | EDN1         | ADGRL4         | Endothelial Cells | Endothelial Cells |    0.982396 |
| set_local_contact_to_1    |       0.971909 |            13 | EDN1         | ADGRL4         | Endothelial Cells | Endothelial Cells |    0.985308 |


## Full WTA expression specificity

In the full WTA object, `Endothelial Cells` contain `8624` cells. `VWF` and `SELP` are detectable in the WTA matrix, and the endothelial compartment shows direct expression support:

- `VWF` endothelial mean log1p: `1.6583`, detection fraction: `0.8781`.
- `SELP` endothelial mean log1p: `0.3134`, detection fraction: `0.2595`.

The contamination-control table checks platelet and erythroid markers (`PPBP`, `PF4`, `ITGA2B`, `GP1BA`, `HBB`, `HBA1`, `HBA2`) so the interpretation is not reduced to blood carryover.
 `HBB` is the strongest blood-marker caveat in endothelial cells (detection fraction `0.0553`), so the report treats blood carryover as a caution rather than ignoring it.

### Marker top cell types

| gene    | cell_type                            |   n_cells |   mean_raw |   mean_log1p |   detection_fraction |   mean_log1p_norm_by_gene |   detection_norm_by_gene |
|:--------|:-------------------------------------|----------:|-----------:|-------------:|---------------------:|--------------------------:|-------------------------:|
| CDH5    | Endothelial Cells                    |      8624 |  1.58395   |    0.746726  |            0.668135  |                 1         |                1         |
| CDH5    | Pericytes                            |      8087 |  0.318041  |    0.183313  |            0.208359  |                 0.245489  |                0.311852  |
| CDH5    | Myoepithelial Cells                  |      8438 |  0.112586  |    0.0691466 |            0.0856838 |                 0.0925996 |                0.128243  |
| CDH5    | 11q13 Invasive Tumor Cells (Mitotic) |      2462 |  0.0495532 |    0.0330623 |            0.0450853 |                 0.0442764 |                0.0674793 |
| CDH5    | Myeloid Cells                        |       813 |  0.0455105 |    0.0304839 |            0.0418204 |                 0.0408234 |                0.0625927 |
| CDH5    | Mast Cells                           |       369 |  0.0487805 |    0.0303744 |            0.0379404 |                 0.0406767 |                0.0567855 |
| CLEC14A | Endothelial Cells                    |      8624 |  1.31795   |    0.636903  |            0.589518  |                 1         |                1         |
| CLEC14A | Pericytes                            |      8087 |  0.316434  |    0.182637  |            0.209348  |                 0.286758  |                0.355118  |
| CLEC14A | Myeloid Cells                        |       813 |  0.095941  |    0.0631717 |            0.0848708 |                 0.0991857 |                0.143967  |
| CLEC14A | CAFs, Invasive Associated            |      4001 |  0.0717321 |    0.0434649 |            0.0537366 |                 0.0682441 |                0.0911535 |
| CLEC14A | 11q13 Invasive Tumor Cells (Mitotic) |      2462 |  0.0649878 |    0.0430809 |            0.058489  |                 0.0676411 |                0.0992151 |
| CLEC14A | Mast Cells                           |       369 |  0.0569106 |    0.0394474 |            0.0569106 |                 0.0619363 |                0.0965375 |
| COL4A1  | Endothelial Cells                    |      8624 |  2.70014   |    0.904837  |            0.652597  |                 1         |                1         |
| COL4A1  | CAFs, Invasive Associated            |      4001 |  2.44739   |    0.85095   |            0.621345  |                 0.940445  |                0.95211   |
| COL4A1  | Pericytes                            |      8087 |  1.57228   |    0.627668  |            0.516384  |                 0.69368   |                0.791275  |
| COL4A1  | CAFs, DCIS Associated                |     24442 |  0.621635  |    0.31483   |            0.317118  |                 0.347941  |                0.485932  |
| COL4A1  | Myoepithelial Cells                  |      8438 |  0.400095  |    0.204131  |            0.218891  |                 0.2256    |                0.335415  |
| COL4A1  | Basal-like Structured DCIS Cells     |      8760 |  0.347146  |    0.178712  |            0.190753  |                 0.197508  |                0.292299  |


## Spatial hotspot evidence

The hotspot analysis uses full WTA coordinates and a k-nearest-neighbor endothelial neighborhood summary. It identifies `433` high-scoring endothelial hotspot cells at the 95th percentile of endothelial VWF-SELP neighborhood support.

### Hotspot neighbor context

| group               | neighbor_cell_type         |   neighbor_fraction |   n_query_cells |
|:--------------------|:---------------------------|--------------------:|----------------:|
| hotspot_endothelial | 11q13 Invasive Tumor Cells |           0.016455  |             433 |
| hotspot_endothelial | CAFs, DCIS Associated      |           0.15791   |             433 |
| hotspot_endothelial | CAFs, Invasive Associated  |           0.0262702 |             433 |
| hotspot_endothelial | Pericytes                  |           0.171478  |             433 |
| hotspot_endothelial | Macrophages                |           0.0433025 |             433 |
| hotspot_endothelial | T Lymphocytes              |           0.159353  |             433 |
| all_endothelial     | 11q13 Invasive Tumor Cells |           0.0870101 |            8624 |
| all_endothelial     | CAFs, DCIS Associated      |           0.202081  |            8624 |
| all_endothelial     | CAFs, Invasive Associated  |           0.0435558 |            8624 |
| all_endothelial     | Pericytes                  |           0.164932  |            8624 |
| all_endothelial     | Macrophages                |           0.0374681 |            8624 |
| all_endothelial     | T Lymphocytes              |           0.0666019 |            8624 |


## Vascular and pathway context

The top vascular table places `VWF-SELP` in a broader endothelial program that includes VWF/LRP1, EFNB2/PECAM1, MMRN2/CLEC14A, COL4A2/CD93, VEGFC/FLT1, and other vascular matrix/adhesion axes.

### Top vascular pyXenium hits

| ligand   | receptor   | sender_celltype       | receiver_celltype     |   CCI_score |   sender_anchor |   receiver_anchor |   structure_bridge |   sender_expr |   receiver_expr |   local_contact |   contact_coverage |   cross_edge_count |
|:---------|:-----------|:----------------------|:----------------------|-----------:|----------------:|------------------:|-------------------:|--------------:|----------------:|----------------:|-------------------:|-------------------:|
| VWF      | SELP       | Endothelial Cells     | Endothelial Cells     |   0.791289 |        0.955713 |          0.881913 |           1        |             1 |               1 |        0.291245 |          0.11112   |              12779 |
| VWF      | LRP1       | Endothelial Cells     | CAFs, DCIS Associated |   0.747976 |        0.955713 |          0.734629 |           0.720461 |             1 |               1 |        0.346194 |          0.131904  |              13942 |
| EFNB2    | PECAM1     | Endothelial Cells     | Endothelial Cells     |   0.74692  |        0.864067 |          0.956445 |           1        |             1 |               1 |        0.210105 |          0.0578293 |              12779 |
| MMRN2    | CLEC14A    | Endothelial Cells     | Endothelial Cells     |   0.732209 |        0.984513 |          0.900326 |           1        |             1 |               1 |        0.173856 |          0.0395962 |              12779 |
| HSPG2    | LRP1       | Endothelial Cells     | CAFs, DCIS Associated |   0.727961 |        0.881817 |          0.734629 |           0.720461 |             1 |               1 |        0.318853 |          0.111892  |              13942 |
| COL4A2   | CD93       | Endothelial Cells     | Endothelial Cells     |   0.7119   |        0.884029 |          0.881718 |           1        |             1 |               1 |        0.167    |          0.0367008 |              12779 |
| MMRN2    | CD93       | Endothelial Cells     | Endothelial Cells     |   0.710717 |        0.984513 |          0.881718 |           1        |             1 |               1 |        0.148466 |          0.0288755 |              12779 |
| VWF      | ITGA9      | Endothelial Cells     | Endothelial Cells     |   0.7084   |        0.955713 |          0.940402 |           1        |             1 |               1 |        0.140614 |          0.0259019 |              12779 |
| MMP2     | PECAM1     | CAFs, DCIS Associated | Endothelial Cells     |   0.697128 |        0.718867 |          0.956445 |           0.720461 |             1 |               1 |        0.231716 |          0.0536925 |              16371 |
| COL4A1   | CD93       | Endothelial Cells     | Endothelial Cells     |   0.696575 |        0.776629 |          0.881718 |           1        |             1 |               1 |        0.166826 |          0.0384224 |              12779 |
| CXCL12   | ITGA5      | CAFs, DCIS Associated | Endothelial Cells     |   0.688619 |        0.892411 |          0.950872 |           0.720461 |             1 |               1 |        0.174412 |          0.0304196 |              16371 |
| CD34     | SELP       | Endothelial Cells     | Endothelial Cells     |   0.684228 |        0.95596  |          0.881913 |           1        |             1 |               1 |        0.121714 |          0.0194068 |              12779 |
| VEGFC    | FLT1       | Endothelial Cells     | Endothelial Cells     |   0.683529 |        0.871821 |          0.878254 |           1        |             1 |               1 |        0.133196 |          0.0232413 |              12779 |
| CXCL12   | AVPR1A     | CAFs, DCIS Associated | Pericytes             |   0.677852 |        0.892411 |          0.959527 |           0.720461 |             1 |               1 |        0.157246 |          0.0247262 |              15611 |


### Targeted pathway-style summary

| pathway                        | cell_type                            | present_genes                                 |   n_present_genes |   mean_activity |   q95_activity |   detection_fraction_activity_positive |
|:-------------------------------|:-------------------------------------|:----------------------------------------------|------------------:|----------------:|---------------:|---------------------------------------:|
| Hemostasis_Thromboinflammation | Endothelial Cells                    | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.120806  |      0.279311  |                               0.939471 |
| Hemostasis_Thromboinflammation | 11q13 Invasive Tumor Cells (Mitotic) | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.0814984 |      0.158009  |                               0.963444 |
| Hemostasis_Thromboinflammation | 11q13 Invasive Tumor Cells           | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.0812707 |      0.154581  |                               0.959054 |
| Hemostasis_Thromboinflammation | 11q13 Invasive Tumor Cells (G1/S)    | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.0652213 |      0.12068   |                               0.925021 |
| Hemostasis_Thromboinflammation | Apocrine Cells                       | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.052498  |      0.115858  |                               0.846154 |
| Hemostasis_Thromboinflammation | Basal-like Structured DCIS Cells     | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.0430909 |      0.109482  |                               0.739269 |
| Hemostasis_Thromboinflammation | Pericytes                            | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.0406467 |      0.136079  |                               0.602696 |
| Hemostasis_Thromboinflammation | Luminal-like Amorphous DCIS Cells    | VWF,SELP,THBD,PLAT,SERPINE1,ADAMTS13          |                 6 |       0.038831  |      0.106023  |                               0.664876 |
| VascularIdentity               | Endothelial Cells                    | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.26426   |      0.457242  |                               0.998956 |
| VascularIdentity               | Pericytes                            | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.077699  |      0.208339  |                               0.80277  |
| VascularIdentity               | Plasma Cells                         | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.0365053 |      0.0927355 |                               0.731959 |
| VascularIdentity               | 11q13 Invasive Tumor Cells (Mitotic) | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.0249601 |      0.0782352 |                               0.562957 |
| VascularIdentity               | Dendritic Cells                      | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.0221005 |      0.0692219 |                               0.532435 |
| VascularIdentity               | Myeloid Cells                        | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.0197318 |      0.0805222 |                               0.435424 |
| VascularIdentity               | Macrophages                          | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.0185629 |      0.0703372 |                               0.441854 |
| VascularIdentity               | 11q13 Invasive Tumor Cells           | PECAM1,EMCN,CDH5,KDR,FLT1,MMRN2,CLEC14A,EGFL7 |                 8 |       0.0173026 |      0.0651544 |                               0.426073 |


## Contour and boundary ecology context

Existing contour outputs do not include direct `SELP` contour features, but they do include vascular markers and `vascular_stromal` pathway scores. The S1-S5 contour DE table supports vascular enrichment in S3 for several endothelial markers, including `VWF`, `PECAM1`, `EMCN`, `EGFL7`, `MMRN2`, `CLEC14A`, `KDR`, and `FLT1`.

### S1-S5 vascular gene DE

| gene    | comparison   |   n_groups |   n_tested_groups |   n_contours | top_group   | bottom_group   |   max_mean_log1p_cpm |   min_mean_log1p_cpm |   delta_log1p_cpm |   top_group_mean_cpm |   top_group_mean_density_per_um2 |   bottom_group_mean_density_per_um2 |   delta_mean_density_per_um2 |   statistic |     p_value |         fdr | status   |
|:--------|:-------------|-----------:|------------------:|-------------:|:------------|:---------------|---------------------:|---------------------:|------------------:|---------------------:|---------------------------------:|------------------------------------:|-----------------------------:|------------:|------------:|------------:|:---------|
| EGFL7   | global       |          5 |                 5 |         1139 | S3          | S4             |              4.80468 |             1.33738  |           3.4673  |              423.302 |                      0.00198585  |                         0.000316173 |                  0.00166967  |     432.028 | 3.33295e-92 | 6.00864e-88 | ok       |
| VWF     | global       |          5 |                 5 |         1139 | S3          | S4             |              5.02063 |             1.54424  |           3.47639 |              834.645 |                      0.00389106  |                         0.000295879 |                  0.00359518  |     352.296 | 5.60066e-75 | 2.01937e-71 | ok       |
| KDR     | global       |          5 |                 5 |         1139 | S3          | S4             |              4.22668 |             1.22     |           3.00667 |              309.291 |                      0.00188126  |                         0.000325076 |                  0.00155619  |     330.383 | 3.01246e-70 | 9.05144e-67 | ok       |
| CDH5    | global       |          5 |                 5 |         1139 | S3          | S2             |              3.86393 |             1.09032  |           2.77362 |              210.99  |                      0.00112905  |                         0.000219331 |                  0.000909721 |     320.568 | 3.95485e-68 | 1.01854e-64 | ok       |
| PECAM1  | global       |          5 |                 5 |         1139 | S3          | S4             |              5.15712 |             1.99684  |           3.16027 |              633.394 |                      0.00293508  |                         0.000459929 |                  0.00247515  |     315.916 | 3.99026e-67 | 8.99206e-64 | ok       |
| CLEC14A | global       |          5 |                 5 |         1139 | S3          | S4             |              3.63457 |             0.844623 |           2.78995 |              178.834 |                      0.000964294 |                         5.01874e-05 |                  0.000914106 |     314.555 | 7.84525e-67 | 1.57149e-63 | ok       |
| EMCN    | global       |          5 |                 5 |         1139 | S3          | S4             |              3.85034 |             1.04352  |           2.80683 |              192.955 |                      0.000997506 |                         8.23015e-05 |                  0.000915205 |     303.928 | 1.53949e-64 | 2.13492e-61 | ok       |
| MMRN2   | global       |          5 |                 5 |         1139 | S3          | S2             |              3.47412 |             1.02143  |           2.45269 |              160.24  |                      0.000738363 |                         0.00018533  |                  0.000553033 |     270.845 | 2.09734e-57 | 1.57545e-54 | ok       |
| FLT1    | global       |          5 |                 5 |         1139 | S3          | S4             |              3.93464 |             1.6781   |           2.25654 |              207.578 |                      0.00127332  |                         0.000309394 |                  0.000963929 |     218.698 | 3.57373e-46 | 6.71116e-44 | ok       |


### Boundary program scores

| contour_id                               |   myeloid_vascular_belt |   stromal_encapsulation |   emt_invasive_front | top_program             |   top_program_score |
|:-----------------------------------------|------------------------:|------------------------:|---------------------:|:------------------------|--------------------:|
| S1 11q13 Invasive Tumor Cells #1.1       |               0.433486  |              -0.0579408 |            0.568328  | emt_invasive_front      |            0.568328 |
| S1 11q13 Invasive Tumor Cells #2.1       |              -0.117655  |              -0.619584  |            0.264489  | immune_exclusion        |            0.293735 |
| S1 11q13 Invasive Tumor Cells #3.1       |              -0.0454068 |               0.539854  |            0.8091    | emt_invasive_front      |            0.8091   |
| S1 11q13 Invasive Tumor Cells #4.1       |               0.379348  |              -0.011645  |            0.256764  | tls_adjacent_activation |            0.692764 |
| S2 Basal-like Structured DCIS Cells #1.1 |               0.435594  |              -0.37733   |           -0.186089  | necrotic_hypoxic_rim    |            0.529808 |
| S2 Basal-like Structured DCIS Cells #2.1 |              -0.0285485 |              -0.0651599 |           -0.0304837 | immune_exclusion        |            0.218619 |
| S2 Basal-like Structured DCIS Cells #3.1 |               0.455341  |               0.999417  |            0.948449  | stromal_encapsulation   |            0.999417 |
| S2 Basal-like Structured DCIS Cells #4.1 |               0.24108   |               0.0226242 |            0.271789  | emt_invasive_front      |            0.271789 |


## Mechanostress and tumor-stroma context

Existing mechanostress coupling outputs do not directly include `VWF` or `SELP`, so the current analysis reports available overlap and uses full WTA geometry to relate endothelial VWF-SELP joint support to tumor/CAF proximity.

| gene                            |   present_in_existing_mechanostress_coupling |   n_endothelial_cells |   spearman_rho_vs_inverse_distance |       p_value |   median_distance_um |
|:--------------------------------|---------------------------------------------:|----------------------:|-----------------------------------:|--------------:|---------------------:|
| VWF                             |                                            0 |                   nan |                         nan        | nan           |             nan      |
| SELP                            |                                            0 |                   nan |                         nan        | nan           |             nan      |
| PECAM1                          |                                            0 |                   nan |                         nan        | nan           |             nan      |
| EMCN                            |                                            0 |                   nan |                         nan        | nan           |             nan      |
| KDR                             |                                            0 |                   nan |                         nan        | nan           |             nan      |
| COL4A1                          |                                            0 |                   nan |                         nan        | nan           |             nan      |
| COL4A2                          |                                            0 |                   nan |                         nan        | nan           |             nan      |
| VWF_SELP_joint_vs_nearest_tumor |                                          nan |                  8624 |                          -0.175717 |   9.28729e-61 |              79.1024 |
| VWF_SELP_joint_vs_nearest_caf   |                                          nan |                  8624 |                          -0.130955 |   2.68122e-34 |              17.5913 |


The simple nearest-distance test is negative for both inverse tumor distance and inverse CAF distance, so this first-pass result supports a vascular/endothelial self-state more than a signal concentrated immediately at tumor-stroma contact fronts. That does not rule out boundary-associated vascular niches, but it argues they should be tested with contour and vessel-structure annotations rather than inferred from nearest tumor/CAF distance alone.

## Cross-method triangulation

Other full-common benchmark methods do not need to recover the exact `VWF-SELP` pair for this signal to be meaningful. The useful test is whether they recover the broader vascular/stromal theme, such as `VWF-LRP1` in CellPhoneDB and endothelial/stromal vascular axes in LARIS or spatial methods.

| method_source   | ligand   | receptor   | sender                               | receiver                  |   score_raw |   score_std |   rank_within_method | spatial_support_type           |
|:----------------|:---------|:-----------|:-------------------------------------|:--------------------------|------------:|------------:|---------------------:|:-------------------------------|
| cellphonedb     | VWF      | LRP1       | Endothelial Cells                    | CAFs, DCIS Associated     |     7.08993 |    0.999999 |                    2 | nonspatial_expression_baseline |
| cellphonedb     | VWF      | LRP1       | Endothelial Cells                    | CAFs, Invasive Associated |     6.68008 |    0.999998 |                    3 | nonspatial_expression_baseline |
| cellphonedb     | CCN2     | LRP1       | CAFs, Invasive Associated            | CAFs, DCIS Associated     |     6.64891 |    0.999997 |                    4 | nonspatial_expression_baseline |
| cellphonedb     | PLAT     | LRP1       | 11q13 Invasive Tumor Cells           | CAFs, DCIS Associated     |     6.61272 |    0.999997 |                    5 | nonspatial_expression_baseline |
| cellphonedb     | HSPG2    | LRP1       | Endothelial Cells                    | CAFs, DCIS Associated     |     6.57462 |    0.999996 |                    6 | nonspatial_expression_baseline |
| cellphonedb     | CCN2     | LRP1       | CAFs, Invasive Associated            | CAFs, Invasive Associated |     6.26456 |    0.999995 |                    7 | nonspatial_expression_baseline |
| cellphonedb     | PLAT     | LRP1       | 11q13 Invasive Tumor Cells           | CAFs, Invasive Associated |     6.23045 |    0.999994 |                    8 | nonspatial_expression_baseline |
| cellphonedb     | HSPG2    | LRP1       | Endothelial Cells                    | CAFs, Invasive Associated |     6.19456 |    0.999993 |                    9 | nonspatial_expression_baseline |
| cellphonedb     | COL6A1   | ITGA6      | CAFs, Invasive Associated            | Endothelial Cells         |     5.92833 |    0.999992 |                   11 | nonspatial_expression_baseline |
| cellphonedb     | VWF      | ITGB1      | Endothelial Cells                    | CAFs, Invasive Associated |     5.64568 |    0.99999  |                   13 | nonspatial_expression_baseline |
| cellphonedb     | PLAT     | LRP1       | 11q13 Invasive Tumor Cells (G1/S)    | CAFs, DCIS Associated     |     5.49433 |    0.999989 |                   14 | nonspatial_expression_baseline |
| cellphonedb     | APP      | LRP1       | Basal-like Structured DCIS Cells     | CAFs, DCIS Associated     |     5.43588 |    0.999987 |                   16 | nonspatial_expression_baseline |
| cellphonedb     | HSPG2    | ITGB1      | Endothelial Cells                    | CAFs, Invasive Associated |     5.23534 |    0.999986 |                   18 | nonspatial_expression_baseline |
| cellphonedb     | PLAT     | LRP1       | 11q13 Invasive Tumor Cells (G1/S)    | CAFs, Invasive Associated |     5.17672 |    0.999985 |                   19 | nonspatial_expression_baseline |
| cellphonedb     | PLAT     | LRP1       | 11q13 Invasive Tumor Cells (Mitotic) | CAFs, DCIS Associated     |     5.15436 |    0.999984 |                   20 | nonspatial_expression_baseline |
| cellphonedb     | APP      | LRP1       | Basal-like Structured DCIS Cells     | CAFs, Invasive Associated |     5.12164 |    0.999982 |                   22 | nonspatial_expression_baseline |
| cellphonedb     | PSAP     | LRP1       | Dendritic Cells                      | CAFs, DCIS Associated     |     5.06512 |    0.999981 |                   24 | nonspatial_expression_baseline |
| cellphonedb     | CXCL12   | ITGB1      | CAFs, DCIS Associated                | CAFs, Invasive Associated |     5.02931 |    0.999979 |                   26 | nonspatial_expression_baseline |


## Literature-supported interpretation

- NCBI Bookshelf describes Weibel-Palade bodies as endothelial granules that store VWF and P-selectin, linking them to hemostasis, inflammation, platelet aggregation, leukocyte trafficking, and angiogenesis: https://www.ncbi.nlm.nih.gov/books/NBK535353/
- Reactome places P-selectin binding biology under cell-surface interactions at the vascular wall: https://reactome.org/content/detail/R-HSA-202724
- A Frontiers review describes regulated Weibel-Palade body exocytosis as a mechanism that presents highly multimeric VWF and P-selectin at the endothelial surface after vascular injury or inflammatory stimulation: https://www.frontiersin.org/articles/10.3389/fcell.2021.813995/full
- UniProt VWF entry: https://rest.uniprot.org/uniprotkb/P04275.txt

## Caveats

- `VWF-SELP` in transcriptomics is a state-level inference; it does not prove protein-level VWF/P-selectin co-storage or exocytosis.
- P-selectin is also platelet-associated, so contamination controls are required for interpretation.
- pyXenium's `Endothelial Cells -> Endothelial Cells` direction should be read as an endothelial neighborhood/self-state axis, not necessarily directional ligand secretion.
- Direct validation would require protein staining, vascular morphology, or thromboinflammatory markers beyond WTA RNA.
