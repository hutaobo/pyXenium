# `pyXenium.cci`

```{eval-rst}
.. currentmodule:: pyXenium.cci

.. autosummary::
   :toctree: generated
   :nosignatures:

   cci_topology_analysis
```

## Publication-facing optional outputs

`cci_topology_analysis` keeps the original discovery score as the default
behavior. For manuscript-grade analyses, the function can also annotate
interaction modes, apply mechanism-aware distance kernels, compute a lightweight
component-shuffle null calibration (`cci_pvalue`, `cci_fdr`, `null_z`), attach
receiver downstream-response support, and return the top-axis hotspot table.

These additions are designed as computational evidence layers. They prioritize
cell-cell interaction hypotheses but do not prove protein binding, secretion, or
causal signaling.
