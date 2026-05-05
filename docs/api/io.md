# `pyXenium.io`

```{eval-rst}
.. currentmodule:: pyXenium.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_xenium
   write_xenium
   read_slide
   load_anndata_from_partial
   export_xenium_to_slide_zarr
   load_xenium_gene_protein
   XeniumSlide
   XeniumImage
```

`XeniumSlide` is pyXenium's independent in-memory and on-disk slide container. Its
design acknowledges the broader [SpatialData](https://spatialdata.scverse.org/en/stable/)
ecosystem and [Marconato et al., 2024](https://doi.org/10.1038/s41592-024-02212-x), but
pyXenium core slide I/O is fully rewritten and does not depend on `spatialdata` at runtime.
