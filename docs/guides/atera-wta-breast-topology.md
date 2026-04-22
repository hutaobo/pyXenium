# Atera WTA breast topology workflow

`pyXenium` ships a fixed reproducibility workflow for the Atera WTA FFPE breast
Xenium sample. The workflow is intended for manuscript-ready reruns of the
breast-validated ligand-receptor and pathway topology analyses.

Core entrypoints:

- `pyXenium.ligand_receptor.ligand_receptor_topology_analysis`
- `pyXenium.pathway.pathway_topology_analysis`
- `pyXenium.pathway.compute_pathway_activity_matrix`
- `pyXenium.validation.run_atera_wta_breast_topology`

CLI usage:

```bash
pyxenium atera-wta-breast-topology \
  "Y:/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs" \
  --manuscript-mode
```

Python usage:

```python
from pyXenium.validation import run_atera_wta_breast_topology

study = run_atera_wta_breast_topology(
    dataset_root=r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs",
    manuscript_mode=True,
    manuscript_root="manuscript",
)
print(study["payload"]["pathway_primary_best"])
```

By default, the workflow prefers precomputed topology anchors from:

```text
<dataset_root>/sfplot_tbc_formal_wta/results
```

That directory should contain:

- `t_and_c_result*.csv`
- `StructureMap_table*.csv`

The fixed output bundle includes:

- `summary.json`
- `report.md`
- `ligand_to_cell.csv`
- `receptor_to_cell.csv`
- `lr_sender_receiver_scores.csv`
- `lr_component_diagnostics.csv`
- `pathway_to_cell.csv`
- `pathway_structuremap.csv`
- `pathway_activity_to_cell.csv`
- `pathway_activity_structuremap.csv`
- `pathway_mode_comparison.csv`
- `figures/`
