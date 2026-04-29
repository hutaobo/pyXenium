# Atera WTA breast topology workflow

`pyXenium` ships a fixed reproducibility workflow for the Atera WTA FFPE breast Xenium sample.

Core entrypoints:

- `pyXenium.cci.cci_topology_analysis`
- `pyXenium.pathway.pathway_topology_analysis`
- `pyXenium.pathway.compute_pathway_activity_matrix`
- `pyXenium.validation.run_atera_wta_breast_topology`

## CLI usage

```bash
pyxenium atera-wta-breast-topology \
  "Y:/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs" \
  --manuscript-mode
```

## Python usage

```python
from pyXenium.validation import run_atera_wta_breast_topology

study = run_atera_wta_breast_topology(
    dataset_root=r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs",
    manuscript_mode=True,
    manuscript_root="manuscript",
)
```

By default the workflow prefers precomputed topology anchors under:

```text
<dataset_root>/sfplot_tbc_formal_wta/results
```

That directory should contain:

- `t_and_c_result*.csv`
- `StructureMap_table*.csv`
