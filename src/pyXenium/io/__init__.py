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
    "build_10x_public_slides",
    "build_xenium_slide",
    "discover_10x_xenium_datasets",
    "export_xenium_to_spatialdata_zarr",
    "generate_missing_contours_with_histoseg",
    "load_anndata_from_partial",
    "load_xenium_gene_protein",
    "read_sdata",
    "read_slide",
    "read_xenium",
    "read_xenium_slide",
    "write_slide",
    "write_xenium",
    "write_xenium_slide",
    "resolve_10x_dataset_metadata",
    "select_primary_contour_geojson",
]

_LAZY_EXPORTS = {
    "build_atera_slides": ".xenium_slide_builder",
    "build_10x_public_slides": ".tenx_public_slides",
    "build_xenium_slide": ".xenium_slide_builder",
    "discover_10x_xenium_datasets": ".tenx_public_slides",
    "generate_missing_contours_with_histoseg": ".tenx_public_slides",
    "load_xenium_gene_protein": ".xenium_gene_protein_loader",
    "resolve_10x_dataset_metadata": ".tenx_public_slides",
    "select_primary_contour_geojson": ".tenx_public_slides",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
