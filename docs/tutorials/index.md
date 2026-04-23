# Tutorials

```{toctree}
:hidden:
:maxdepth: 1

io
ligand_receptor
pathway
multimodal
contour
```

The tutorials hub brings the five canonical pyXenium themes into one place.
The five notebooks below use real pyXenium study outputs to explain what each
module does, how to rerun it on local Xenium data, and why the result matters
biologically.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} `pyXenium.io`
:link: io
:link-type: doc
Load a public renal RNA + protein Xenium study, inspect the `XeniumSData`
components, and see how I/O preserves the structures needed for downstream
biology.
:::

:::{grid-item-card} `pyXenium.ligand_receptor`
:link: ligand_receptor
:link-type: doc
Walk through Atera WTA breast ligand-receptor topology outputs and interpret
sender-receiver programs across tumor, immune, and vascular compartments.
:::

:::{grid-item-card} `pyXenium.pathway`
:link: pathway
:link-type: doc
Compare pathway topology aggregation with activity point-cloud scoring on the
same breast topology study and connect the scores to cell-state programs.
:::

:::{grid-item-card} `pyXenium.multimodal`
:link: multimodal
:link-type: doc
Review the renal immune-resistance pilot from loading through joint states,
discordance, niches, and branch-level hypotheses.
:::

:::{grid-item-card} `pyXenium.contour`
:link: contour
:link-type: doc
Generate HistoSeg-backed Atera breast contours, import Xenium Explorer
annotations, expand boundaries, and profile shell-based marker densities.
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
