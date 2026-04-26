# Tutorials

```{toctree}
:hidden:
:maxdepth: 1

io
ligand_receptor
pathway
multimodal
contour_index
gmi
mechanostress_atera_pdc
```

The tutorials hub brings the canonical pyXenium tutorial series into one
place. The notebooks below use real pyXenium study outputs to explain what
each module does, how to rerun it on local Xenium data, and why the result
matters biologically.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} pyXenium.io Tutorial
:link: io
:link-type: doc
Load a public renal RNA + protein Xenium study, inspect the `XeniumSData`
components, and see how I/O preserves the structures needed for downstream
biology.
:::

:::{grid-item-card} pyXenium.ligand_receptor Tutorial
:link: ligand_receptor
:link-type: doc
Walk through Atera WTA breast ligand-receptor topology outputs and interpret
sender-receiver programs across tumor, immune, and vascular compartments.
:::

:::{grid-item-card} pyXenium.pathway Tutorial
:link: pathway
:link-type: doc
Compare pathway topology aggregation with activity point-cloud scoring on the
same breast topology study and connect the scores to cell-state programs.
:::

:::{grid-item-card} pyXenium.multimodal Tutorial
:link: multimodal
:link-type: doc
Review the renal immune-resistance pilot from loading through joint states,
discordance, niches, and branch-level hypotheses.
:::

:::{grid-item-card} pyXenium.contour Tutorials
:link: contour_index
:link-type: doc
Generate HistoSeg-backed annotations, run contour-level transcript and cell
composition analyses, and profile barrier-aware boundary density curves.
:::

:::{grid-item-card} pyXenium.gmi Tutorial
:link: gmi
:link-type: doc
Use S1/S5 contours as samples for GMI main-effect, interaction, control, and
heterogeneity analyses.
:::

:::{grid-item-card} pyXenium.mechanostress Tutorial
:link: mechanostress_atera_pdc
:link-type: doc
Run the Atera WTA breast S1/S5 mechanostress workflow on PDC and inspect
fibroblast axis strength, tumor-stroma growth states, polarity, and coupling
artifacts.
:::
::::

## Tutorial Pattern

Each tutorial follows the same research-facing structure:

- `Overview`
- `Biological question`
- `Dataset`
- `Setup`
- `Core workflow`
- `Visual outputs`
- `Biological interpretation`
- `Caveats`
- `Next steps`
