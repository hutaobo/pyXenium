from pyXenium import PUBLIC_DATASET_SOURCES, get_public_dataset_sources
from pyXenium.datasets import RENAL_FFPE_PROTEIN_10X_DATASET


def test_public_dataset_catalog_exposes_10x_validation_source():
    sources = get_public_dataset_sources()

    assert sources == PUBLIC_DATASET_SOURCES
    assert RENAL_FFPE_PROTEIN_10X_DATASET in sources
    assert sources[0].provider == "10x Genomics"
    assert sources[0].upstream_data_license == "CC BY 4.0"
    assert sources[0].url.startswith("https://www.10xgenomics.com/datasets/")
