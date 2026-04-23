from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from .sdata_model import XeniumFrameChunkSource, XeniumSData
from .sdata_store import read_xenium_sdata, write_xenium_sdata
from .xenium_artifacts import (
    build_feature_summary,
    discover_image_artifacts,
    extract_spatial_from_obs,
    iter_transcript_chunks,
    join_path,
    load_xenium_analysis,
    read_boundary_tables,
    read_cell_feature_matrix,
    read_cells_table,
    read_cells_zarr_spatial,
    read_he_image,
    read_transcripts_table,
    resolve_transcripts_path,
    split_rna_and_protein,
)


DEFAULT_CLUSTER_RELPATH = "analysis/clustering/gene_expression_graphclust/clusters.csv"
TRANSCRIPT_POINT_COLUMNS = (
    "x",
    "y",
    "gene_identity",
    "gene_name",
    "quality_score",
    "valid",
)
TRANSCRIPT_POINT_COLUMN_TYPES = {
    "x": "float64",
    "y": "float64",
    "gene_identity": "int64",
    "gene_name": "string",
    "quality_score": "float64",
    "valid": "bool",
}


def _metadata_for_source(
    *,
    base_path: str,
    backend: str,
    features: pd.DataFrame,
    cluster_key: str | None,
    image_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_path": str(base_path),
        "backend": backend,
        "units": "micron",
        "cluster_key": cluster_key,
        "feature_summary": build_feature_summary(features),
        "image_artifacts": image_artifacts or {},
        "labels": {},
        "store_version": 1,
    }


def _build_obs(
    *,
    base_path: str,
    barcodes,
    cells_csv: str,
    cells_parquet: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    obs = read_cells_table(
        base_path,
        cells_csv=cells_csv,
        cells_parquet=cells_parquet,
        barcodes=barcodes,
    )
    metadata: dict[str, Any] = {}
    if obs is not None:
        return obs, metadata

    cells_path = join_path(base_path, "cells.zarr.zip")
    if Path(str(base_path)).suffix == ".zip":
        cells_path = str(base_path)

    try:
        spatial_frame, cells_meta = read_cells_zarr_spatial(cells_path)
        obs = pd.DataFrame(index=barcodes)
        obs.index.name = "barcode"
        obs["x"] = spatial_frame.reindex(barcodes)["x"]
        obs["y"] = spatial_frame.reindex(barcodes)["y"]
        metadata["cells_zarr"] = cells_meta
        return obs, metadata
    except Exception:
        obs = pd.DataFrame(index=barcodes)
        obs.index.name = "barcode"
        return obs, metadata


def _transcript_point_source(transcripts_path: str) -> XeniumFrameChunkSource:
    return XeniumFrameChunkSource(
        columns=TRANSCRIPT_POINT_COLUMNS,
        column_types=TRANSCRIPT_POINT_COLUMN_TYPES,
        chunk_iter_factory=lambda: iter_transcript_chunks(transcripts_path),
        attrs={"source_path": transcripts_path},
        preserve_extra_columns=True,
    )


def _assemble_anndata(
    *,
    base_path: str,
    prefer: str,
    mex_dirname: str,
    mex_matrix_name: str,
    mex_features_name: str,
    mex_barcodes_name: str,
    cells_csv: str,
    cells_parquet: str | None,
    clusters_relpath: str | None,
    cluster_column_name: str,
    include_boundaries: bool,
    include_images: bool,
) -> tuple[ad.AnnData, dict[str, pd.DataFrame], dict[str, Any]]:
    matrix, features, barcodes, backend = read_cell_feature_matrix(
        base_path,
        prefer=prefer,
        mex_dirname=mex_dirname,
        mex_matrix_name=mex_matrix_name,
        mex_features_name=mex_features_name,
        mex_barcodes_name=mex_barcodes_name,
    )
    modalities = split_rna_and_protein(matrix, features, barcodes)
    obs, obs_meta = _build_obs(
        base_path=base_path,
        barcodes=barcodes,
        cells_csv=cells_csv,
        cells_parquet=cells_parquet,
    )

    analysis = load_xenium_analysis(
        base_path,
        clusters_relpath=clusters_relpath,
        barcodes=barcodes,
        cluster_column_name=cluster_column_name,
    )
    for key, series in analysis.cluster_series.items():
        column_name = analysis.cluster_columns[key]
        obs[column_name] = pd.Categorical(series.astype("object"))

    adata = ad.AnnData(X=modalities["rna_matrix"], obs=obs, var=modalities["rna_var"])
    adata.layers["rna"] = adata.X.copy()
    adata.obsm["protein"] = modalities["protein_frame"]
    spatial = extract_spatial_from_obs(adata.obs)
    if spatial is not None:
        adata.obsm["spatial"] = spatial
    for method, frame in analysis.projection_frames.items():
        adata.obsm[analysis.projection_keys[method]] = frame.to_numpy(dtype=float, copy=True)

    adata.uns.setdefault("modality", {})
    adata.uns["modality"]["rna"] = {"feature_type": "Gene Expression"}
    if modalities["protein_frame"].shape[1] > 0:
        adata.uns["modality"]["protein"] = {
            "feature_type": "Protein Expression",
            "value": "scaled_mean_intensity",
        }
    adata.uns["xenium_io"] = {
        "backend": backend,
        "source_path": str(base_path),
        **obs_meta,
    }
    adata.uns["xenium_analysis"] = analysis.summary()

    boundaries = (
        read_boundary_tables(base_path, include_cell=True, include_nucleus=True)
        if include_boundaries
        else {}
    )
    for key, frame in boundaries.items():
        adata.uns[key] = frame.copy()

    image_artifacts = discover_image_artifacts(base_path) if include_images else {}
    for key, payload in image_artifacts.items():
        adata.uns[key] = dict(payload)

    metadata = _metadata_for_source(
        base_path=base_path,
        backend=backend,
        features=features,
        cluster_key=analysis.default_cluster_key,
        image_artifacts=image_artifacts,
    )
    return adata, boundaries, metadata


def read_xenium(
    base_path: str,
    *,
    as_: str = "anndata",
    prefer: str = "auto",
    include_transcripts: bool = True,
    stream_transcripts: bool = False,
    include_boundaries: bool = True,
    include_images: bool = False,
    clusters_relpath: str | None = DEFAULT_CLUSTER_RELPATH,
    cluster_column_name: str = "cluster",
    cells_csv: str = "cells.csv.gz",
    cells_parquet: str | None = None,
    mex_dirname: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
) -> ad.AnnData | XeniumSData:
    if as_ not in {"anndata", "sdata"}:
        raise ValueError("as_ must be either 'anndata' or 'sdata'.")

    adata, boundaries, metadata = _assemble_anndata(
        base_path=base_path,
        prefer=prefer,
        mex_dirname=mex_dirname,
        mex_matrix_name=mex_matrix_name,
        mex_features_name=mex_features_name,
        mex_barcodes_name=mex_barcodes_name,
        cells_csv=cells_csv,
        cells_parquet=cells_parquet,
        clusters_relpath=clusters_relpath,
        cluster_column_name=cluster_column_name,
        include_boundaries=include_boundaries,
        include_images=include_images,
    )
    if as_ == "anndata":
        return adata

    points: dict[str, pd.DataFrame] = {}
    point_sources: dict[str, XeniumFrameChunkSource] = {}
    if include_transcripts:
        transcripts_path = resolve_transcripts_path(base_path)
        if transcripts_path is not None:
            if stream_transcripts:
                point_sources["transcripts"] = _transcript_point_source(transcripts_path)
            else:
                points["transcripts"] = read_transcripts_table(transcripts_path)

    images = {}
    if include_images:
        he_image = read_he_image(base_path)
        if he_image is not None:
            images["he"] = he_image

    return XeniumSData(
        table=adata,
        points=points,
        shapes=boundaries,
        images=images,
        metadata=metadata,
        point_sources=point_sources,
    )


def write_xenium(
    obj: ad.AnnData | XeniumSData,
    path: str | Path,
    *,
    format: str = "h5ad",
    overwrite: bool = False,
) -> dict[str, Any]:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    if format == "h5ad":
        table = obj.table if isinstance(obj, XeniumSData) else obj
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output path already exists: {target}. Pass overwrite=True to replace it."
                )
            target.unlink()
        table.write_h5ad(target)
        return {
            "format": "h5ad",
            "output_path": str(target),
            "tables": ["cells"],
            "points": sorted(obj.points.keys()) if isinstance(obj, XeniumSData) else [],
            "shapes": sorted(obj.shapes.keys()) if isinstance(obj, XeniumSData) else [],
            "images": sorted(obj.images.keys()) if isinstance(obj, XeniumSData) else [],
            "contour_images": (
                sorted(obj.contour_images.keys()) if isinstance(obj, XeniumSData) else []
            ),
            "labels": [],
        }

    if format == "sdata":
        sdata = obj if isinstance(obj, XeniumSData) else XeniumSData(table=obj)
        return write_xenium_sdata(sdata, target, overwrite=overwrite)

    raise ValueError("format must be either 'h5ad' or 'sdata'.")


def read_sdata(path: str | Path) -> XeniumSData:
    return read_xenium_sdata(path)


def warn_unsupported_image_export_flags(*, morphology_focus: bool, morphology_mip: bool, aligned_images: bool) -> None:
    if morphology_focus or morphology_mip or aligned_images:
        warnings.warn(
            "pyXenium SData currently exports H&E images only. Legacy morphology/aligned "
            "image flags are retained for compatibility and do not materialize those image groups.",
            stacklevel=2,
        )
