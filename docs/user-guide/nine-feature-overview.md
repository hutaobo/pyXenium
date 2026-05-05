# Nine-feature overview

```{figure} ../_static/figures/pyxenium-nine-feature-overview.png
:alt: pyXenium nine-feature overview figure
:class: pyxenium-overview-image
:figclass: pyxenium-overview-figure

Figure 1 summarizes pyXenium as a nine-feature toolkit for Xenium spatial biology.
```

The overview places `pyXenium` at the center because the package keeps Xenium-derived
cell tables, transcript coordinates, morphology, boundaries, and downstream handoff
metadata in one analysis-ready structure. The surrounding modules show the main
ways those data are loaded, analyzed, modeled, and handed to optional external
workflows.

1. **{doc}`Xenium I/O <../guides/xenium-data-loading>`** loads Xenium exports, recovers partial bundles, and writes XeniumSlide stores.
2. **{doc}`Multimodal Analysis <multimodal-overview>`** prepares RNA, protein, morphology, and H&E-derived context for joint analysis.
3. **{doc}`Cell-Cell Interaction <../tutorials/cci_index>`** quantifies topology-aware ligand-receptor and sender-receiver patterns.
4. **{doc}`Pathway Topology <../tutorials/pathway>`** maps pathway scores and pathway activity onto spatial neighborhoods.
5. **{doc}`Contour Geometry <../tutorials/contour_index>`** turns tissue annotations into contour features, shells, densities, and boundary-aware summaries.
6. **{doc}`GMI Inference <../guides/gmi-contour>`** builds contour-level matrices for sparse main-effect and interaction modeling.
7. **{doc}`Mechanostress <../tutorials/mechanostress_atera_pdc>`** extracts morphology-derived polarity, axis strength, and tumor-stroma growth signals.
8. **{doc}`AI-Driven Spatial Pathologist <../tutorials/ai_driven_spatial_pathologist>`** documents the optional external `spatho` review workflow around pyXenium-structured Xenium cases.
9. **{doc}`SpatialPerturb Bridge <../tutorials/spatialperturb_bridge>`** writes handoff specifications for projecting Perturb-seq references onto Xenium tissue with the external `SpatialPerturb` package.

Together, these sections separate the stable pyXenium API surfaces from optional
external bridges while keeping the manuscript and documentation entry points aligned.
