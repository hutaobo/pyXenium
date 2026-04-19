from __future__ import annotations

import anndata as ad

from .api import DEFAULT_CLUSTER_RELPATH, read_xenium


def load_xenium_gene_protein(
    base_path: str,
    *,
    prefer: str = "auto",
    mex_dirname: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    cells_csv: str = "cells.csv.gz",
    cells_parquet: str | None = None,
    read_morphology: bool = False,
    attach_boundaries: bool = True,
    clusters_relpath: str | None = DEFAULT_CLUSTER_RELPATH,
    cluster_column_name: str = "cluster",
) -> ad.AnnData:
    """
    Compatibility wrapper around :func:`pyXenium.io.api.read_xenium`.

    This legacy entry point preserves the established AnnData return type for
    Xenium RNA + protein runs while delegating all parsing logic to the unified
    Xenium artifact reader.
    """

    return read_xenium(
        base_path,
        as_="anndata",
        prefer=prefer,
        include_transcripts=False,
        include_boundaries=attach_boundaries,
        include_images=read_morphology,
        clusters_relpath=clusters_relpath,
        cluster_column_name=cluster_column_name,
        cells_csv=cells_csv,
        cells_parquet=cells_parquet,
        mex_dirname=mex_dirname,
        mex_matrix_name=mex_matrix_name,
        mex_features_name=mex_features_name,
        mex_barcodes_name=mex_barcodes_name,
    )
