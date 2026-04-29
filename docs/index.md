# pyXenium

```{toctree}
:hidden:
:maxdepth: 2

quickstart
tutorials/index
user-guide/index
workflows/index
api/index
changelog
```

<div class="pyxenium-hero">
  <p class="pyxenium-tagline">Nine canonical sections for Xenium I/O, multimodal analysis, CCI, pathway topology, contour geometry, GMI inference, mechanostress analysis, and external workflow bridges.</p>
  <p class="pyxenium-lead">
    pyXenium is a Python toolkit for 10x Genomics Xenium organized around nine major sections:
    <code>pyXenium.io</code>, <code>pyXenium.multimodal</code>, <code>pyXenium.cci</code>,
    <code>pyXenium.pathway</code>, <code>pyXenium.contour</code>, <code>pyXenium.gmi</code>,
    <code>pyXenium.mechanostress</code>, the external AI pathology bridge via <code>spatho</code>,
    and the Perturb-seq handoff bridge via <code>pyXenium.perturb</code>.
    This site keeps those sections visible first, then maps them to tutorials, guides,
    workflows, and API pages.
  </p>
  <div class="pyxenium-link-row">
    <a href="quickstart.html">Get started</a>
    <a href="https://pypi.org/project/pyXenium/">PyPI</a>
    <a href="https://github.com/hutaobo/pyXenium">GitHub</a>
    <a href="https://pyxenium.readthedocs.io/en/latest/">Read the Docs</a>
    <a href="https://pyxenium.readthedocs.io/en/latest/changelog.html">Changelog</a>
    <a href="https://github.com/hutaobo/pyXenium/releases">Releases</a>
  </div>
</div>

```{image} _static/branding/pyxenium-horizontal-dark.png
:alt: pyXenium horizontal logo
:class: pyxenium-banner
```

## Version & Build

<p>
  <a href="https://pypi.org/project/pyXenium/"><img src="https://img.shields.io/pypi/v/pyXenium.svg" alt="PyPI version"></a>
  <a href="https://pyxenium.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/pyxenium/badge/?version=latest" alt="Read the Docs"></a>
  <a href="https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml"><img src="https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/pyXenium/"><img src="https://img.shields.io/pypi/pyversions/pyXenium.svg" alt="Python versions"></a>
  <a href="https://github.com/hutaobo/pyXenium/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-non--commercial-d97706.svg" alt="License"></a>
</p>

- Current repository version: `0.4.1`
- Package index: [PyPI](https://pypi.org/project/pyXenium/)
- Documentation site: [Read the Docs latest](https://pyxenium.readthedocs.io/en/latest/)
- Canonical build status: [GitHub Actions CI](https://github.com/hutaobo/pyXenium/actions/workflows/ci.yml)
- Releases: [GitHub releases](https://github.com/hutaobo/pyXenium/releases)
- Changelog: [RTD changelog](https://pyxenium.readthedocs.io/en/latest/changelog.html)

## Nine core sections

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Xenium I/O
:link: guides/xenium-data-loading
:link-type: doc
Load Xenium exports, recover partial bundles, round-trip `XeniumSData`, and export compat stores.
:::

:::{grid-item-card} Multimodal Analysis
:link: user-guide/multimodal-overview
:link-type: doc
Work with the canonical `pyXenium.multimodal` surface for RNA + protein preparation and joint analysis.
:::

:::{grid-item-card} Cell-Cell Interaction
:link: tutorials/cci_index
:link-type: doc
Walk through topology-native cell-cell interaction analyses and compare Atera breast benchmark outputs.
:::

:::{grid-item-card} Pathway Topology
:link: tutorials/pathway
:link-type: doc
Connect pathway topology scoring to cell-state programs on the same Xenium study.
:::

:::{grid-item-card} Contour Geometry
:link: tutorials/contour_index
:link-type: doc
Generate HistoSeg-backed annotations, contour-level transcript summaries, and boundary-aware density profiles.
:::

:::{grid-item-card} GMI Inference
:link: guides/gmi-contour
:link-type: doc
Build contour pseudo-bulk matrices and run GMI main-effect, interaction, and validation workflows.
:::

:::{grid-item-card} Mechanostress
:link: tutorials/mechanostress_atera_pdc
:link-type: doc
Compute fibroblast axis strength, tumor-stroma growth patterning, and cell polarity from Xenium morphology.
:::

:::{grid-item-card} AI-Driven Spatial Pathologist
:link: tutorials/ai_driven_spatial_pathologist
:link-type: doc
Call the external `spatho` workflow on Xenium cases structured by pyXenium `XeniumSData`.
:::

:::{grid-item-card} SpatialPerturb Bridge
:link: tutorials/spatialperturb_bridge
:link-type: doc
Generate handoff specs for Perturb-seq reference projection with the external `SpatialPerturb` CLI.
:::

::::

## Documentation entry points

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation / Quickstart
:link: quickstart
:link-type: doc
Install pyXenium, set up the docs environment, and run your first Xenium workflow.
:::

:::{grid-item-card} Tutorials hub
:link: tutorials/index
:link-type: doc
Browse notebook-style walkthroughs for all nine sections, including the optional external bridges.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc
Read focused guides for Xenium I/O, multimodal preparation, contour analysis, and contour GMI inference.
:::

:::{grid-item-card} Workflows
:link: workflows/index
:link-type: doc
Run packaged renal and Atera workflows with standardized artifact bundles and report-ready outputs.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Browse curated autosummary pages for the canonical pyXenium namespaces and the SpatialPerturb bridge helpers.
:::

:::{grid-item-card} Changelog
:link: changelog
:link-type: doc
Track documentation, branding, and package-level changes.
:::
::::

## Feature areas

- `pyXenium.io`: Xenium artifact loading, partial export recovery, SData I/O, and SpatialData-compatible export.
- `pyXenium.multimodal`: canonical RNA + protein loading, immune-resistance scoring, joint analyses, and packaged multimodal workflows.
- `pyXenium.cci`: topology-native cell-cell interaction analysis primitives.
- `pyXenium.pathway`: pathway topology analysis and pathway activity scoring.
- `pyXenium.contour`: GeoJSON contour import and contour-aware density profiling around polygon annotations.
- `pyXenium.gmi`: contour-level GMI modeling for sparse main-effect and interaction discovery in spatial transcriptomics.
- `pyXenium.mechanostress`: morphology-derived mechanical stress states from cell/nucleus boundaries and tumor-stroma context.
- AI-Driven Spatial Pathologist via `spatho`: optional external AI pathology review workflow built on pyXenium's `XeniumSData` case structure, not a pyXenium runtime dependency.
- `pyXenium.perturb`: SpatialPerturb Bridge for optional Perturb-seq reference projection onto Xenium tissue through the external `SpatialPerturb` package.

:::{admonition} GitHub branding asset
:class: pyxenium-brand-note
Use `docs/_static/branding/pyxenium-horizontal-dark.png` as the repository social preview image in GitHub repository settings. That upload is still a manual GitHub step even though the asset is versioned in the repo.
:::
