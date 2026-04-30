from importlib import import_module

from ._xenium_defaults import DEFAULT_XENIUM_PIXEL_SIZE_UM
from .api import (
    DEFAULT_CLUSTER_RELPATH,
    read_sdata,
    read_slide,
    read_xenium,
    read_xenium_slide,
    write_slide,
    write_xenium,
    write_xenium_slide,
)
from .partial_xenium_loader import load_anndata_from_partial
from .sdata_model import XeniumFrameChunkSource, XeniumImage, XeniumSData, XeniumSlide
from .spatialdata_export import DEFAULT_SPATIALDATA_STORE_NAME, export_xenium_to_spatialdata_zarr

__all__ = [
    "DEFAULT_CLUSTER_RELPATH",
    "DEFAULT_SPATIALDATA_STORE_NAME",
    "DEFAULT_XENIUM_PIXEL_SIZE_UM",
    "XeniumFrameChunkSource",
    "XeniumImage",
    "XeniumSData",
    "XeniumSlide",
    "build_atera_slides",
    "build_xenium_slide",
    "export_xenium_to_spatialdata_zarr",
    "load_anndata_from_partial",
    "load_xenium_gene_protein",
    "read_sdata",
    "read_slide",
    "read_xenium",
    "read_xenium_slide",
    "write_slide",
    "write_xenium",
    "write_xenium_slide",
]

_LAZY_EXPORTS = {
    "build_atera_slides": ".xenium_slide_builder",
    "build_xenium_slide": ".xenium_slide_builder",
    "load_xenium_gene_protein": ".xenium_gene_protein_loader",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
