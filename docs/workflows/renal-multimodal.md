# Renal multimodal workflows

pyXenium ships two renal-focused multimodal workflows under `pyXenium.multimodal.workflows`.

## 1. Validated renal FFPE smoke workflow

Use `run_validated_renal_ffpe_smoke(...)` or:

```bash
pyxenium multimodal validate-renal-ffpe-protein \
  "Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein"
```

This workflow verifies that the canonical Xenium RNA + protein loading path reproduces the validated
public renal dataset dimensions and artifact availability.

## 2. Renal immune-resistance pilot

Use `run_renal_immune_resistance_pilot(...)` or:

```bash
pyxenium multimodal renal-immune-resistance-pilot \
  "Y:/long/10X_datasets/Xenium/Xenium_Renal/Xenium_V1_Human_Kidney_FFPE_Protein" \
  --output-dir ./renal_immune_resistance_outputs
```

The pilot workflow packages:

- joint cell classes and cell states
- RNA/protein discordance summaries
- spatial niche construction
- ranked ROI patches
- report-ready tables, markdown, JSON, and figures

## Python example

```python
from pyXenium.multimodal import run_renal_immune_resistance_pilot

study = run_renal_immune_resistance_pilot(
    base_path="/path/to/xenium_export",
    output_dir="./renal_immune_resistance_outputs",
)
```
