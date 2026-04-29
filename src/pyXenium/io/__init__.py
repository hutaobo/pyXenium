from importlib import import_module

from .api import DEFAULT_CLUSTER_RELPATH, read_sdata, read_xenium, write_xenium
from .partial_xenium_loader import load_anndata_from_partial
from .sdata_model import XeniumFrameChunkSource, XeniumImage, XeniumSData
from .spatialdata_export import DEFAULT_SPATIALDATA_STORE_NAME, export_xenium_to_spatialdata_zarr

__all__ = [
    "DEFAULT_CLUSTER_RELPATH",
    "DEFAULT_SPATIALDATA_STORE_NAME",
    "XeniumFrameChunkSource",
    "XeniumImage",
    "XeniumSData",
    "export_xenium_to_spatialdata_zarr",
    "load_anndata_from_partial",
    "load_xenium_gene_protein",
    "read_sdata",
    "read_xenium",
    "write_xenium",
]

_LAZY_EXPORTS = {
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
