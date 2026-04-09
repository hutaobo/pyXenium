from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicDatasetSource:
    slug: str
    title: str
    provider: str
    url: str
    modality: str
    software: str
    species: str
    tissue: str
    preservation_method: str
    disease_state: str
    upstream_data_license: str
    first_published: str
    current_release_date: str
    release_notes: str
    local_validation_summary: str


RENAL_FFPE_PROTEIN_10X_DATASET = PublicDatasetSource(
    slug="xenium-protein-ffpe-human-renal-carcinoma",
    title="Xenium In Situ Gene and Protein Expression data for FFPE Human Renal Cell Carcinoma",
    provider="10x Genomics",
    url="https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma",
    modality="RNA + Protein",
    software="Xenium Onboard Analysis 4.0.0",
    species="Human",
    tissue="Kidney",
    preservation_method="FFPE",
    disease_state="Renal cell carcinoma",
    upstream_data_license="CC BY 4.0",
    first_published="2025-07-17",
    current_release_date="2025-09-25",
    release_notes=(
        "10x Genomics states that the dataset was first published on July 17, 2025, "
        "reanalyzed with the final Xenium Onboard Analysis v4.0 pipeline on August 27, 2025, "
        "and replaced again on September 25, 2025 to fix a bug with no changes to the biological results."
    ),
    local_validation_summary=(
        "pyXenium successfully loaded a local copy of the public bundle through both the Zarr-backed "
        "and HDF5-backed cell_feature_matrix inputs, producing an AnnData object with 465545 cells, "
        "405 RNA features, 27 protein markers, spatial centroids, and merged cluster labels."
    ),
)

PUBLIC_DATASET_SOURCES = (RENAL_FFPE_PROTEIN_10X_DATASET,)


def get_public_dataset_sources() -> tuple[PublicDatasetSource, ...]:
    """Return curated public dataset sources used to validate pyXenium."""
    return PUBLIC_DATASET_SOURCES
