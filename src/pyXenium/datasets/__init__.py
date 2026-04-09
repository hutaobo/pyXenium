"""Bundled example datasets and curated public source metadata shipped with pyXenium."""

from .catalog import (
    PUBLIC_DATASET_SOURCES,
    RENAL_FFPE_PROTEIN_10X_DATASET,
    PublicDatasetSource,
    get_public_dataset_sources,
)

__all__ = [
    "PublicDatasetSource",
    "RENAL_FFPE_PROTEIN_10X_DATASET",
    "PUBLIC_DATASET_SOURCES",
    "get_public_dataset_sources",
]
