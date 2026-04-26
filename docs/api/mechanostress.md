# `pyXenium.mechanostress`

`pyXenium.mechanostress` is the canonical beta surface for morphology-derived
mechanical stress analysis in Xenium data. It converts cell and nucleus
boundaries into fibroblast axis strength, tumor-stroma growth patterning, and
cell polarity outputs.

```{eval-rst}
.. currentmodule:: pyXenium.mechanostress

.. autosummary::
   :toctree: generated
   :nosignatures:

   AxisStrengthConfig
   TumorStromaGrowthConfig
   PolarityConfig
   MechanostressConfig
   MechanostressCohortResult
   MechanostressResult
   estimate_cell_axes
   summarize_axial_orientation
   compute_ane_density
   classify_tumor_stroma_growth
   summarize_tumor_growth
   compute_distance_expression_coupling
   compute_cell_polarity
   summarize_cell_polarity
   run_mechanostress_cohort
   run_mechanostress_workflow
   write_mechanostress_artifacts
   render_mechanostress_report
   validate_hnscc_mechanostress_outputs
   validate_suzuki_luad_mechanostress_outputs
```
