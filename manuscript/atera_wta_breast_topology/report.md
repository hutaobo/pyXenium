# Atera WTA Breast Topology Reproducibility Bundle

Sample ID: `atera_wta_ffpe_breast`
Dataset root: `Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs`
t_and_c / StructureMap anchor source: `Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs\sfplot_tbc_formal_wta\results`

## Core Summary

- Cells loaded: `170057`
- RNA features loaded: `18028`
- Cluster count: `20`
- Topology celltype count: `20`
- Unassigned cells: `12`
- panel_num_targets_predesigned: `18028`
- median_transcripts_per_cell: `2116`
- Runtime (s): `56.68`

## LR Smoke Panel

- `CSF1-CSF1R`: top `CAFs, DCIS Associated -> Macrophages` (`0.5074`)
- `CXCL12-CXCR4`: top `CAFs, DCIS Associated -> T Lymphocytes` (`0.6339`)
- `TGFB1-TGFBR2`: top `Endothelial Cells -> Endothelial Cells` (`0.5291`)
- `JAG1-NOTCH1`: top `11q13 Invasive Tumor Cells -> Basal-like Structured DCIS Cells` (`0.5029`)
- `DLL4-NOTCH3`: top `Endothelial Cells -> Pericytes` (`0.6628`)

## LR Acceptance

- `PASS` CSF1-CSF1R top sender should not be Mast Cells
- `PASS` CXCL12-CXCR4 should keep CAFs, DCIS Associated -> T Lymphocytes high-ranking
- `PASS` DLL4-NOTCH3 top hit should be Endothelial Cells -> Pericytes

## Pathway Primary Results

- `MacrophageProgram` -> `Macrophages` (`distance=0.0448`)
- `PlasmaProgram` -> `Plasma Cells` (`distance=0.0542`)
- `VascularProgram` -> `Endothelial Cells` (`distance=0.0534`)
- `BasalDCISProgram` -> `Basal-like Structured DCIS Cells` (`distance=0.0591`)
- `ApocrineProgram` -> `Apocrine Cells` (`distance=0.0996`)
- `LuminalAmorphousProgram` -> `Luminal-like Amorphous DCIS Cells` (`distance=0.2623`)

## Pathway Acceptance

- `PASS` `MacrophageProgram` expected `Macrophages`, observed `Macrophages`
- `PASS` `PlasmaProgram` expected `Plasma Cells`, observed `Plasma Cells`
- `PASS` `VascularProgram` expected `Endothelial Cells, Pericytes`, observed `Endothelial Cells`
- `PASS` `BasalDCISProgram` expected `Basal-like Structured DCIS Cells`, observed `Basal-like Structured DCIS Cells`
- `PASS` `ApocrineProgram` expected `Apocrine Cells`, observed `Apocrine Cells`
- `PASS` `LuminalAmorphousProgram` expected `Luminal-like Amorphous DCIS Cells`, observed `Luminal-like Amorphous DCIS Cells`

## Fixed Output Files

- `ligand_to_cell`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\ligand_to_cell.csv`
- `receptor_to_cell`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\receptor_to_cell.csv`
- `lr_sender_receiver_scores`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\lr_sender_receiver_scores.csv`
- `lr_component_diagnostics`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\lr_component_diagnostics.csv`
- `lr_summary_heatmap`: `2` file(s)
- `lr_hotspot_cells_csv`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\figures\lr_hotspot_cells.csv`
- `lr_hotspot_cells_parquet`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\figures\lr_hotspot_cells.parquet`
- `lr_hotspot_overlay`: `2` file(s)
- `pathway_to_cell`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\pathway_to_cell.csv`
- `pathway_structuremap`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\pathway_structuremap.csv`
- `pathway_activity_to_cell`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\pathway_activity_to_cell.csv`
- `pathway_activity_structuremap`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\pathway_activity_structuremap.csv`
- `pathway_mode_comparison`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\pathway_mode_comparison.csv`
- `pathway_to_cell_heatmap`: `2` file(s)
- `pathway_activity_to_cell_heatmap`: `2` file(s)
- `pathway_hotspot_cells_csv`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\figures\pathway_hotspot_cells.csv`
- `pathway_hotspot_overlay`: `2` file(s)
- `summary_json`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\summary.json`
- `report_md`: `D:\GitHub\pyXenium\manuscript\atera_wta_breast_topology\report.md`
