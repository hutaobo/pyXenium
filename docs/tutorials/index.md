# Tutorials

```{toctree}
:hidden:
:maxdepth: 1

pyXenium.io <io>
pyXenium.cci <cci_index>
pyXenium.pathway <pathway>
pyXenium.multimodal <multimodal>
pyXenium.contour <contour_index>
pyXenium.gmi <gmi>
pyXenium.mechanostress <mechanostress_atera_pdc>
pyXenium.spatho <ai_driven_spatial_pathologist>
```

The tutorials hub brings the canonical pyXenium tutorial series into one
place. The notebooks below use real pyXenium study outputs to explain what
each module does, how to rerun it on local Xenium data, and why the result
matters biologically. The final entry documents the optional handoff from
pyXenium's `XeniumSData` structure to the external `spatho` pathology workflow.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} pyXenium.io
:link: io
:link-type: doc
Load a public renal RNA + protein Xenium study, inspect the `XeniumSData`
components, and see how I/O preserves the structures needed for downstream
biology.
:::

:::{grid-item-card} pyXenium.cci
:link: cci_index
:link-type: doc
Walk through Atera WTA breast cell-cell interaction topology outputs, then compare
whole-dataset benchmark results across spatial and non-spatial CCI methods.
:::

:::{grid-item-card} pyXenium.pathway
:link: pathway
:link-type: doc
Compare pathway topology aggregation with activity point-cloud scoring on the
same breast topology study and connect the scores to cell-state programs.
:::

:::{grid-item-card} pyXenium.multimodal
:link: multimodal
:link-type: doc
Choose among renal RNA + protein analysis, RNA + contour + H&E discovery, and
the BM-Net/H&E morphology increment pilot on PDC.
:::

:::{grid-item-card} pyXenium.contour
:link: contour_index
:link-type: doc
Generate HistoSeg-backed annotations, run contour-level transcript and cell
composition analyses, and profile barrier-aware boundary density curves.
:::

:::{grid-item-card} pyXenium.gmi
:link: gmi
:link-type: doc
Use S1/S5 contours as samples for GMI main-effect, interaction, control, and
heterogeneity analyses.
:::

:::{grid-item-card} pyXenium.mechanostress
:link: mechanostress_atera_pdc
:link-type: doc
Run the Atera WTA breast S1/S5 mechanostress workflow on PDC and inspect
fibroblast axis strength, tumor-stroma growth states, polarity, and coupling
artifacts.
:::

:::{grid-item-card} pyXenium.spatho
:link: ai_driven_spatial_pathologist
:link-type: doc
Install and call the external `spatho` workflow, and see how pyXenium's
`XeniumSData` structure supports AI-driven spatial pathology without adding
spatho code to pyXenium.
:::
::::

## Tutorial Pattern

The research-facing notebook tutorials generally follow this structure:

- `Overview`
- `Biological question`
- `Dataset`
- `Setup`
- `Core workflow`
- `Visual outputs`
- `Biological interpretation`
- `Caveats`
- `Next steps`
