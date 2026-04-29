# TopoLink-CCI Nature Methods Brief Communication Package

This folder contains a draft manuscript package for presenting **TopoLink-CCI** as a Nature Methods Brief Communication.

## Contents

- `manuscript/topolink_cci_brief_communication.md`: title, abstract, four-paragraph main text draft, figure legends and claims.
- `manuscript/online_methods.md`: reproducible method details for the Online Methods section.
- `manuscript/supplementary_information_outline.md`: planned Supplementary Notes, Figures, Tables and Data.
- `manuscript/supplementary_data_manifest.tsv`: source paths for supplementary data and software artifacts.
- `manuscript/presubmission_inquiry.md`: concise Nature Methods presubmission inquiry draft.
- `scripts/make_topolink_cci_nature_brief_figures.py`: reproducible figure-generation script.
- `figures/`: generated Figure 1 and Figure 2 drafts in PNG, PDF and SVG formats.

## Regenerate Figures

From the repository root:

```powershell
python benchmarking\cci_2026_atera\topolink_cci_short_communication\scripts\make_topolink_cci_nature_brief_figures.py
```

## Naming

The formal method name is **TopoLink-CCI**. The current benchmark uses TopoLink-CCI in **cell-cell interaction-resource mode**, so historical `TopoLink-CCI` wording should be treated as a mode name rather than the manuscript-level method name.

## Interpretation Guardrail

TopoLink-CCI prioritizes computationally supported spatial molecular interaction hypotheses. It does not by itself prove protein-level ligand binding, secretion, receptor activation or causal signaling.
