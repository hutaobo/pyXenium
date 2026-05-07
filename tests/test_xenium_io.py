from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
import zarr
from scipy import sparse
from scipy.io import mmwrite

import pyXenium.io.api as xenium_api_module
from pyXenium.io.xenium_artifacts import read_analysis_cell_groups
from pyXenium.io import (
    XeniumSlide,
    XeniumSlide,
    build_xenium_slide,
    export_xenium_to_slide_zarr,
    load_anndata_from_partial,
    load_xenium_gene_protein,
    read_slide,
    read_slide,
    read_xenium,
    read_xenium_slide,
    write_xenium,
    write_xenium_slide,
)
from pyXenium.multimodal import load_rna_protein_anndata

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


BARCODES = pd.Index(["aaaaaaaa-1", "aaaaaaab-1", "aaaaaaac-1"], name="barcode")
FEATURES = pd.DataFrame(
    {
        "id": ["gene_1", "gene_2", "prot_1", "prot_2"],
        "name": ["Gene1", "Gene2", "Prot1", "Prot2"],
        "feature_type": [
            "Gene Expression",
            "Gene Expression",
            "Protein Expression",
            "Protein Expression",
        ],
    }
)
CELL_BY_FEATURE = sparse.csr_matrix(
    np.array(
        [
            [1, 0, 5, 10],
            [0, 2, 0, 3],
            [4, 1, 7, 0],
        ],
        dtype=np.float32,
    )
)
GRAPHCLUST_LABELS = ["Tumor", "Immune", "Tumor"]
KMEANS2_LABELS = ["0", "1", "1"]
NESTED_GRAPHCLUST_LABELS = ["NestedA", "NestedB", "NestedA"]
UMAP_VALUES = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
NESTED_UMAP_VALUES = np.asarray([[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]], dtype=float)
PCA_VALUES = np.asarray(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ],
    dtype=float,
)
TSNE_VALUES = np.asarray([[9.0, 1.0], [8.0, 2.0], [7.0, 3.0]], dtype=float)


def _write_gzip_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(text)


def _make_transcripts_zip(
    path: Path,
    *,
    include_cell_id: bool = True,
    extra_columns: dict[str, np.ndarray] | None = None,
    include_z_coordinate: bool = False,
) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    root.attrs["gene_names"] = ["Gene1", "Gene2"]
    grids = root.require_group("grids")
    level = grids.require_group("0")
    chunk = level.require_group("0,0")
    locations = np.asarray([[10.0, 20.0], [15.0, 25.0], [30.0, 40.0], [35.0, 45.0]])
    if include_z_coordinate:
        locations = np.c_[locations, np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)]
    chunk.create_array(
        "location",
        data=locations,
    )
    chunk.create_array("gene_identity", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
    chunk.create_array("quality_score", data=np.asarray([30.0, 25.0, 18.0, 35.0]))
    chunk.create_array("valid", data=np.asarray([1, 1, 1, 0], dtype=np.uint8))
    if include_cell_id:
        chunk.create_array(
            "cell_id",
            data=np.asarray([BARCODES[0], BARCODES[1], BARCODES[2], BARCODES[0]], dtype=np.str_),
        )
    for key, values in (extra_columns or {}).items():
        chunk.create_array(key, data=np.asarray(values))
    store.close()


def _make_cells_zip(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    root.create_array("cell_id", data=np.asarray([[0, 1], [1, 1], [2, 1]], dtype=np.uint32))
    cell_summary = root.create_array(
        "cell_summary",
        data=np.asarray(
            [
                [10.0, 20.0, 100.0, 10.0, 20.0, 50.0, 0.0, 1.0],
                [20.0, 30.0, 100.0, 20.0, 30.0, 50.0, 0.0, 1.0],
                [40.0, 50.0, 100.0, 40.0, 50.0, 50.0, 0.0, 1.0],
            ]
        ),
    )
    cell_summary.attrs["columns"] = [
        "cell_centroid_x",
        "cell_centroid_y",
        "cell_area",
        "nucleus_centroid_x",
        "nucleus_centroid_y",
        "nucleus_area",
        "z_level",
        "nucleus_count",
    ]
    store.close()


def _make_analysis_zip(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    cell_groups = root.require_group("cell_groups")
    group = cell_groups.require_group("0")
    group.create_array("indices", data=np.asarray([0, 2, 1], dtype=np.int64))
    group.create_array("indptr", data=np.asarray([0, 2, 3], dtype=np.int64))
    cell_groups.attrs["group_names"] = [["Tumor", "Immune"]]
    cell_groups.attrs["grouping_names"] = ["graphclust"]
    store.close()


def _make_official_analysis_zip(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    cell_groups = root.require_group("cell_groups")

    graph = cell_groups.require_group("0")
    graph.create_array("indices", data=np.asarray([0, 2, 1], dtype=np.uint32))
    graph.create_array("indptr", data=np.asarray([0, 2], dtype=np.uint32))

    kmeans = cell_groups.require_group("1")
    kmeans.create_array("indices", data=np.asarray([0, 1, 2], dtype=np.uint32))
    kmeans.create_array("indptr", data=np.asarray([0, 1], dtype=np.uint32))

    cell_groups.attrs["group_names"] = [
        ["Cluster 1", "Cluster 2"],
        ["Cluster 1", "Cluster 2"],
    ]
    cell_groups.attrs["grouping_names"] = [
        "gene_expression_graphclust",
        "gene_expression_kmeans_2_clusters",
    ]
    cell_groups.attrs["major_version"] = 1
    cell_groups.attrs["minor_version"] = 0
    cell_groups.attrs["number_groupings"] = 2
    store.close()


def _make_official_analysis_zip_with_padding(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    cell_groups = root.require_group("cell_groups")
    graph = cell_groups.require_group("0")
    graph.create_array("indices", data=np.asarray([0, 1, 0], dtype=np.uint32))
    graph.create_array("indptr", data=np.asarray([0, 1], dtype=np.uint32))
    cell_groups.attrs["group_names"] = [["Cluster 1", "Cluster 2"]]
    cell_groups.attrs["grouping_names"] = ["gene_expression_graphclust"]
    store.close()


def _make_cell_feature_matrix_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    matrix = CELL_BY_FEATURE.T.tocsr()
    x_group = root.require_group("X")
    x_group.create_array("data", data=matrix.data)
    x_group.create_array("indices", data=matrix.indices)
    x_group.create_array("indptr", data=matrix.indptr)
    x_group.create_array("shape", data=np.asarray(matrix.shape, dtype=np.int64))

    features_group = root.require_group("features")
    features_group.create_array("id", data=FEATURES["id"].to_numpy(dtype=np.str_))
    features_group.create_array("name", data=FEATURES["name"].to_numpy(dtype=np.str_))
    features_group.create_array(
        "feature_type", data=FEATURES["feature_type"].to_numpy(dtype=np.str_)
    )
    root.create_array("barcodes", data=BARCODES.to_numpy(dtype=np.str_))


def _make_cell_feature_matrix_h5(path: Path) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 Xenium fixtures.")
    matrix = CELL_BY_FEATURE.T.tocsr()
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", shape=matrix.data.shape, data=matrix.data)
        group.create_dataset("indices", shape=matrix.indices.shape, data=matrix.indices)
        group.create_dataset("indptr", shape=matrix.indptr.shape, data=matrix.indptr)
        group.create_dataset("shape", shape=np.asarray(matrix.shape, dtype=np.int64).shape, data=np.asarray(matrix.shape, dtype=np.int64))
        features_group = group.create_group("features")
        features_group.create_dataset("id", shape=FEATURES["id"].to_numpy(dtype="S").shape, data=FEATURES["id"].to_numpy(dtype="S"))
        features_group.create_dataset("name", shape=FEATURES["name"].to_numpy(dtype="S").shape, data=FEATURES["name"].to_numpy(dtype="S"))
        features_group.create_dataset(
            "feature_type",
            shape=FEATURES["feature_type"].to_numpy(dtype="S").shape,
            data=FEATURES["feature_type"].to_numpy(dtype="S"),
        )
        group.create_dataset("barcodes", shape=BARCODES.to_numpy(dtype="S").shape, data=BARCODES.to_numpy(dtype="S"))


def _make_he_bundle(base_path: Path) -> np.ndarray:
    image = np.zeros((512, 768, 3), dtype=np.uint8)
    image[..., 0] = np.arange(image.shape[1], dtype=np.uint8)
    image[..., 1] = np.arange(image.shape[0], dtype=np.uint8)[:, None]
    image[..., 2] = 127
    tifffile.imwrite(
        base_path / "sample_he_image.ome.tif",
        image,
        ome=True,
        photometric="rgb",
        metadata={"axes": "YXS"},
    )

    affine = np.asarray(
        [
            [2.0, 0.0, 10.0],
            [0.0, 3.0, 5.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pd.DataFrame(affine).to_csv(
        base_path / "sample_he_alignment.csv",
        index=False,
        header=False,
    )

    alignment_points = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 20.0],
            [100.0, 50.0],
            [220.0, 120.0],
        ],
        dtype=float,
    )
    fixed_points = np.c_[alignment_points, np.ones(len(alignment_points), dtype=float)] @ affine.T
    pd.DataFrame(
        {
            "fixedX": fixed_points[:, 0],
            "fixedY": fixed_points[:, 1],
            "alignmentX": alignment_points[:, 0],
            "alignmentY": alignment_points[:, 1],
        }
    ).to_csv(base_path / "sample_keypoints.csv", index=False)

    with (base_path / "experiment.xenium").open("w", encoding="utf-8") as stream:
        json.dump({"pixel_size": 0.5}, stream)

    return affine


def _write_cluster_csv(path: Path, labels, *, index_column: str, cluster_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({index_column: BARCODES.to_list(), cluster_column: list(labels)}).to_csv(
        path,
        index=False,
    )


def _write_projection_csv(path: Path, columns: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"Barcode": BARCODES.to_list()}
    for idx, column in enumerate(columns):
        payload[column] = np.asarray(values)[:, idx]
    pd.DataFrame(payload).to_csv(path, index=False)


def _write_analysis_outputs(
    base_path: Path,
    *,
    analysis_root: str,
    graph_labels,
    include_extra_clusterings: bool,
    include_projections: bool,
    umap_values: np.ndarray | None = None,
) -> None:
    root = base_path / Path(analysis_root)
    _write_cluster_csv(
        root / "clustering" / "gene_expression_graphclust" / "clusters.csv",
        graph_labels,
        index_column="Barcode",
        cluster_column="Cluster",
    )
    if include_extra_clusterings:
        _write_cluster_csv(
            root / "clustering" / "gene_expression_kmeans_2_clusters" / "clusters.csv",
            KMEANS2_LABELS,
            index_column="cell_id",
            cluster_column="cluster",
        )
    if include_projections:
        _write_projection_csv(
            root / "umap" / "gene_expression_2_components" / "projection.csv",
            ["UMAP-1", "UMAP-2"],
            UMAP_VALUES if umap_values is None else umap_values,
        )
        _write_projection_csv(
            root / "pca" / "gene_expression_3_components" / "projection.csv",
            ["PC-1", "PC-2", "PC-3"],
            PCA_VALUES,
        )
        _write_projection_csv(
            root / "tsne" / "gene_expression_2_components" / "projection.csv",
            ["TSNE-1", "TSNE-2"],
            TSNE_VALUES,
        )


def make_xenium_dataset(
    base_path: Path,
    *,
    include_backends: tuple[str, ...] = ("mex", "zarr", "h5"),
    include_boundaries: bool = True,
    include_transcripts: bool = True,
    include_zarr_sidecars: bool = False,
    include_he_image: bool = False,
    include_analysis: bool = True,
    include_extra_clusterings: bool = False,
    include_projections: bool = False,
    primary_analysis_root: str = "analysis",
    include_nested_analysis_variant: bool = False,
    transcripts_include_cell_id: bool = True,
    transcript_extra_columns: dict[str, np.ndarray] | None = None,
    transcripts_include_z_coordinate: bool = False,
) -> Path:
    base_path.mkdir(parents=True, exist_ok=True)

    cells = pd.DataFrame(
        {
            "cell_id": BARCODES.to_list(),
            "x_centroid": [10.0, 20.0, 40.0],
            "y_centroid": [20.0, 30.0, 50.0],
        }
    )
    cells.to_csv(base_path / "cells.csv.gz", index=False, compression="gzip")

    if include_analysis:
        _write_analysis_outputs(
            base_path,
            analysis_root=primary_analysis_root,
            graph_labels=GRAPHCLUST_LABELS,
            include_extra_clusterings=include_extra_clusterings,
            include_projections=include_projections,
        )
        if include_nested_analysis_variant and primary_analysis_root != "analysis/analysis":
            _write_analysis_outputs(
                base_path,
                analysis_root="analysis/analysis",
                graph_labels=NESTED_GRAPHCLUST_LABELS,
                include_extra_clusterings=False,
                include_projections=include_projections,
                umap_values=NESTED_UMAP_VALUES,
            )

    if include_boundaries:
        cell_boundaries = pd.DataFrame(
            {
                "cell_id": [BARCODES[0], BARCODES[0], BARCODES[1]],
                "vertex_x": [0.0, 1.0, 2.0],
                "vertex_y": [0.0, 1.0, 2.0],
            }
        )
        nucleus_boundaries = pd.DataFrame(
            {
                "cell_id": [BARCODES[0], BARCODES[1], BARCODES[2]],
                "vertex_x": [0.5, 1.5, 2.5],
                "vertex_y": [0.5, 1.5, 2.5],
            }
        )
        cell_boundaries.to_csv(
            base_path / "cell_boundaries.csv.gz", index=False, compression="gzip"
        )
        nucleus_boundaries.to_csv(
            base_path / "nucleus_boundaries.csv.gz", index=False, compression="gzip"
        )

    if include_transcripts:
        _make_transcripts_zip(
            base_path / "transcripts.zarr.zip",
            include_cell_id=transcripts_include_cell_id,
            extra_columns=transcript_extra_columns,
            include_z_coordinate=transcripts_include_z_coordinate,
        )

    if include_zarr_sidecars:
        _make_cells_zip(base_path / "cells.zarr.zip")
        _make_analysis_zip(base_path / "analysis.zarr.zip")

    if "mex" in include_backends:
        mex_dir = base_path / "cell_feature_matrix"
        mex_dir.mkdir(exist_ok=True)
        with gzip.open(mex_dir / "matrix.mtx.gz", "wb") as stream:
            mmwrite(stream, CELL_BY_FEATURE.T)
        _write_gzip_text(
            mex_dir / "features.tsv.gz",
            "\n".join(
                "\t".join(row)
                for row in FEATURES[["id", "name", "feature_type"]].astype(str).to_numpy()
            )
            + "\n",
        )
        _write_gzip_text(
            mex_dir / "barcodes.tsv.gz",
            "\n".join(BARCODES.to_list()) + "\n",
        )

    if "zarr" in include_backends:
        _make_cell_feature_matrix_zarr(base_path / "cell_feature_matrix.zarr")

    if "h5" in include_backends:
        _make_cell_feature_matrix_h5(base_path / "cell_feature_matrix.h5")

    if include_he_image:
        _make_he_bundle(base_path)

    return base_path


def _write_contour_geojson(path: Path) -> Path:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "structure_id": 1,
                    "assigned_structure": "tumor_region",
                    "component_index": 0,
                    "polygon_index": 0,
                    "name": "tumor_region_1",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [15.0, 35.0],
                            [85.0, 35.0],
                            [85.0, 105.0],
                            [15.0, 105.0],
                            [15.0, 35.0],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("prefer", ["mex", "zarr", "h5"])
def test_read_xenium_anndata_backends(tmp_path, prefer):
    if prefer == "h5" and h5py is None:
        pytest.skip("h5py is not available")
    include_backends = (prefer,)
    make_xenium_dataset(tmp_path / prefer, include_backends=include_backends)

    adata = read_xenium(str(tmp_path / prefer), as_="anndata", prefer=prefer)

    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert list(adata.var["name"]) == ["Gene1", "Gene2"]
    assert list(adata.obsm["protein"].columns) == ["Prot1", "Prot2"]
    assert "spatial" in adata.obsm
    assert "cluster" in adata.obs.columns
    assert "cell_boundaries" in adata.uns
    assert adata.uns["xenium_io"]["backend"] == prefer


def test_read_xenium_discovers_prefixed_hnscc_artifacts(tmp_path):
    if h5py is None:
        pytest.skip("h5py is not available")
    base = make_xenium_dataset(
        tmp_path / "prefixed",
        include_backends=("h5",),
        include_boundaries=True,
        include_transcripts=False,
    )
    sample = "GSM000001_Apr_13_24"
    (base / "cell_feature_matrix.h5").rename(base / f"{sample}_cell_feature_matrix.h5")

    cells = pd.read_csv(base / "cells.csv.gz")
    cells.to_parquet(base / f"{sample}_cells.parquet")
    (base / "cells.csv.gz").unlink()

    for stem in ("cell_boundaries", "nucleus_boundaries"):
        frame = pd.read_csv(base / f"{stem}.csv.gz")
        frame.to_parquet(base / f"{sample}_{stem}.parquet")
        (base / f"{stem}.csv.gz").unlink()

    sdata = read_xenium(str(base), as_="slide", prefer="h5", include_transcripts=False)

    assert sdata.table.n_obs == 3
    assert "cell_boundaries" in sdata.shapes
    assert "nucleus_boundaries" in sdata.shapes
    assert {"cell_id", "vertex_id", "x", "y"}.issubset(sdata.shapes["cell_boundaries"].columns)


def test_read_xenium_imports_all_clusterings_and_projections(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_extra_clusterings=True,
        include_projections=True,
    )

    adata = read_xenium(str(dataset), as_="anndata", prefer="zarr")

    assert list(adata.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS
    assert list(adata.obs["cluster__gene_expression_kmeans_2_clusters"].astype(str)) == KMEANS2_LABELS
    np.testing.assert_allclose(adata.obsm["X_umap"], UMAP_VALUES)
    np.testing.assert_allclose(adata.obsm["X_pca"], PCA_VALUES)
    np.testing.assert_allclose(adata.obsm["X_tsne"], TSNE_VALUES)
    assert adata.uns["xenium_analysis"]["default_cluster_key"] == "gene_expression_graphclust"
    assert (
        adata.uns["xenium_analysis"]["cluster_columns"]["gene_expression_kmeans_2_clusters"]
        == "cluster__gene_expression_kmeans_2_clusters"
    )
    assert adata.uns["xenium_analysis"]["projection_keys"] == {
        "pca": "X_pca",
        "tsne": "X_tsne",
        "umap": "X_umap",
    }


def test_read_xenium_falls_back_to_analysis_zarr_zip_when_cluster_csv_missing(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_analysis=False,
    )
    _make_official_analysis_zip(dataset / "analysis.zarr.zip")

    adata = read_xenium(str(dataset), as_="anndata", prefer="zarr")

    assert list(adata.obs["cluster"].astype(str)) == ["1", "2", "1"]
    assert list(adata.obs["cluster__gene_expression_kmeans_2_clusters"].astype(str)) == [
        "1",
        "2",
        "2",
    ]
    assert adata.uns["xenium_analysis"]["default_cluster_key"] == "gene_expression_graphclust"
    assert (
        adata.uns["xenium_analysis"]["cluster_sources"]["gene_expression_graphclust"]["source_type"]
        == "analysis.zarr.zip"
    )


def test_read_xenium_csv_cluster_artifacts_override_analysis_zarr_zip(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_analysis=True,
        include_extra_clusterings=False,
    )
    _make_official_analysis_zip(dataset / "analysis.zarr.zip")

    adata = read_xenium(str(dataset), as_="anndata", prefer="zarr")

    assert list(adata.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS
    assert list(adata.obs["cluster__gene_expression_kmeans_2_clusters"].astype(str)) == [
        "1",
        "2",
        "2",
    ]
    assert (
        adata.uns["xenium_analysis"]["cluster_sources"]["gene_expression_graphclust"]["source_type"]
        == "clusters.csv"
    )
    assert (
        adata.uns["xenium_analysis"]["cluster_sources"]["gene_expression_kmeans_2_clusters"]["source_type"]
        == "analysis.zarr.zip"
    )


def test_read_xenium_accepts_analysis_zarr_zip_as_explicit_cell_groups_path(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_analysis=False,
    )
    _make_official_analysis_zip(dataset / "custom_analysis.zarr.zip")

    adata = read_xenium(
        str(dataset),
        as_="anndata",
        prefer="zarr",
        cell_groups_path="custom_analysis.zarr.zip",
    )

    assert list(adata.obs["cluster"].astype(str)) == ["1", "2", "1"]
    assert (
        adata.uns["xenium_analysis"]["cluster_sources"]["gene_expression_graphclust"]["relpath"]
        == "custom_analysis.zarr.zip"
    )


def test_read_analysis_cell_groups_supports_sentinel_and_official_padding(tmp_path):
    legacy_path = tmp_path / "legacy_analysis.zarr.zip"
    padded_path = tmp_path / "padded_analysis.zarr.zip"
    _make_analysis_zip(legacy_path)
    _make_official_analysis_zip_with_padding(padded_path)

    legacy, legacy_meta = read_analysis_cell_groups(str(legacy_path), n_cells=3)
    padded, padded_meta = read_analysis_cell_groups(str(padded_path), n_cells=3)

    assert list(legacy.astype(str)) == ["Tumor", "Immune", "Tumor"]
    assert legacy_meta["n_clusters"] == 2
    assert list(padded.astype(str)) == ["1", "2", "Unassigned"]
    assert padded_meta["n_clusters"] == 2
    assert padded_meta["n_unassigned"] == 1
    assert padded_meta["n_duplicate_indices"] == 0


def test_read_xenium_prefers_single_layer_analysis_paths(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_projections=True,
        include_nested_analysis_variant=True,
    )

    adata = read_xenium(str(dataset), as_="anndata", prefer="zarr")

    assert list(adata.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS
    np.testing.assert_allclose(adata.obsm["X_umap"], UMAP_VALUES)
    assert (
        adata.uns["xenium_analysis"]["cluster_sources"]["gene_expression_graphclust"]["relpath"]
        == "analysis/clustering/gene_expression_graphclust/clusters.csv"
    )
    assert (
        adata.uns["xenium_analysis"]["projection_sources"]["umap"]["relpath"]
        == "analysis/umap/gene_expression_2_components/projection.csv"
    )


def test_read_xenium_clusters_relpath_overrides_default_cluster_source(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_extra_clusterings=True,
    )

    adata = read_xenium(
        str(dataset),
        as_="anndata",
        prefer="zarr",
        clusters_relpath="analysis/clustering/gene_expression_kmeans_2_clusters/clusters.csv",
    )

    assert list(adata.obs["cluster"].astype(str)) == KMEANS2_LABELS
    assert list(adata.obs["cluster__gene_expression_graphclust"].astype(str)) == GRAPHCLUST_LABELS
    assert adata.uns["xenium_analysis"]["default_cluster_key"] == "gene_expression_kmeans_2_clusters"


def test_read_xenium_accepts_official_cell_groups_barcode_clusters_schema(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")
    _write_cluster_csv(
        dataset / "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv",
        GRAPHCLUST_LABELS,
        index_column="Barcode",
        cluster_column="Clusters",
    )

    adata = read_xenium(
        str(dataset),
        as_="anndata",
        prefer="zarr",
        clusters_relpath="WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv",
    )

    assert list(adata.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS


def test_read_xenium_accepts_official_cell_groups_cellid_group_schema(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")
    _write_cluster_csv(
        dataset / "sample_cell_groups.csv",
        KMEANS2_LABELS,
        index_column="cell_id",
        cluster_column="group",
    )

    adata = read_xenium(
        str(dataset),
        as_="anndata",
        prefer="zarr",
        clusters_relpath="sample_cell_groups.csv",
    )

    assert list(adata.obs["cluster"].astype(str)) == KMEANS2_LABELS


def _write_explicit_table_bundle(dataset: Path, table_dir: Path, matrix_dir: Path) -> tuple[Path, Path, Path]:
    cells = pd.read_csv(dataset / "cells.csv.gz")
    cells = cells.rename(
        columns={
            "cell_id": "Barcode",
            "x_centroid": "center_x",
            "y_centroid": "center_y",
        }
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    cells_path = table_dir / "explicit_cells.parquet"
    cells.to_parquet(cells_path, index=False)

    cell_groups_path = table_dir / "explicit_cell_groups.csv"
    pd.DataFrame(
        {"Barcode": BARCODES.to_list(), "Clusters": KMEANS2_LABELS}
    ).to_csv(cell_groups_path, index=False)

    matrix_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix_path = matrix_dir / "explicit_cell_feature_matrix.h5"
    (dataset / "cell_feature_matrix.h5").replace(feature_matrix_path)
    return cells_path, cell_groups_path, feature_matrix_path


def test_read_xenium_accepts_relative_explicit_table_bundle_paths(tmp_path):
    if h5py is None:
        pytest.skip("h5py is not available")
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("h5",),
        include_analysis=False,
        include_boundaries=False,
        include_transcripts=False,
    )
    _write_explicit_table_bundle(dataset, dataset / "tables", dataset / "matrices")

    adata = read_xenium(
        str(dataset),
        as_="anndata",
        cells_path="tables/explicit_cells.parquet",
        cell_groups_path="tables/explicit_cell_groups.csv",
        feature_matrix_path="matrices/explicit_cell_feature_matrix.h5",
        cell_id_col="Barcode",
        cluster_col="Clusters",
        x_col="center_x",
        y_col="center_y",
        include_boundaries=False,
    )

    assert adata.uns["xenium_io"]["backend"] == "h5"
    assert list(adata.obs["cluster"].astype(str)) == KMEANS2_LABELS
    np.testing.assert_allclose(
        adata.obsm["spatial"],
        np.asarray([[10.0, 20.0], [20.0, 30.0], [40.0, 50.0]]),
    )


def test_read_xenium_accepts_absolute_explicit_table_bundle_paths(tmp_path):
    if h5py is None:
        pytest.skip("h5py is not available")
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("h5",),
        include_analysis=False,
        include_boundaries=False,
        include_transcripts=False,
    )
    cells_path, cell_groups_path, feature_matrix_path = _write_explicit_table_bundle(
        dataset,
        tmp_path / "tables",
        tmp_path / "matrices",
    )

    adata = read_xenium(
        str(dataset),
        as_="anndata",
        cells_path=cells_path,
        cell_groups_path=cell_groups_path,
        feature_matrix_path=feature_matrix_path,
        cell_id_col="Barcode",
        cluster_col="Clusters",
        x_col="center_x",
        y_col="center_y",
        include_boundaries=False,
    )

    assert list(adata.obs["cluster"].astype(str)) == KMEANS2_LABELS
    cluster_sources = adata.uns["xenium_analysis"]["cluster_sources"].values()
    assert any(
        Path(source["path"]).resolve() == cell_groups_path.resolve()
        for source in cluster_sources
    )


def test_read_xenium_slide_contains_expected_components(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_extra_clusterings=True,
        include_projections=True,
    )

    sdata = read_xenium(str(dataset), as_="slide", prefer="mex")

    assert isinstance(sdata, XeniumSlide)
    assert sdata.table.n_obs == 3
    assert "transcripts" in sdata.points
    assert {"x", "y", "gene_identity", "gene_name"}.issubset(sdata.points["transcripts"].columns)
    assert set(sdata.shapes) == {"cell_boundaries", "nucleus_boundaries"}
    assert sdata.metadata["units"] == "micron"
    assert sdata.metadata["feature_summary"]["n_features_total"] == 4
    assert sdata.metadata["cluster_key"] == "gene_expression_graphclust"
    assert "cluster__gene_expression_kmeans_2_clusters" in sdata.table.obs.columns
    np.testing.assert_allclose(sdata.table.obsm["X_umap"], UMAP_VALUES)


def test_xenium_slide_is_canonical_roundtrip_type(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)

    slide = read_xenium(str(dataset), as_="slide", prefer="zarr", include_images=True)

    assert isinstance(slide, XeniumSlide)
    assert slide.table.uns["xenium_slide"]["schema_version"] == 1
    assert "he" in slide.images

    output = tmp_path / "xenium_slide.zarr"
    payload = write_xenium_slide(slide, output)
    reloaded = read_xenium_slide(output)
    legacy_reloaded = read_slide(output)

    assert payload["format"] == "pyxenium.slide"
    assert reloaded.table.shape == slide.table.shape
    assert isinstance(legacy_reloaded, XeniumSlide)


def test_build_xenium_slide_writes_contour_assignments_and_patches(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    contour_geojson = _write_contour_geojson(tmp_path / "contours.geojson")
    output_dir = tmp_path / "slide_out"

    result = build_xenium_slide(
        xenium_root=dataset,
        output_dir=output_dir,
        case_name="synthetic_slide",
        organ="test_organ",
        contour_geojson=contour_geojson,
        extract_contour_images=True,
        max_crop_side_px=64,
    )

    assert result.slide_store.exists()
    assert result.contour_patch_manifest is not None
    assert result.contour_patch_count == 1
    assert (output_dir / "cell_to_contour.parquet").exists()
    assert (output_dir / "structure_assignments.csv").exists()
    assert (output_dir / "contour_patches" / "00001_tumor_region_1.png").exists()

    manifest = json.loads(result.slide_manifest.read_text(encoding="utf-8"))
    qc = json.loads(result.qc_report.read_text(encoding="utf-8"))
    cell_to_contour = pd.read_parquet(output_dir / "cell_to_contour.parquet")
    patch_manifest = json.loads((output_dir / "contour_patches_manifest.json").read_text(encoding="utf-8"))
    slide = read_xenium_slide(result.slide_store)

    assert manifest["schema"]["name"] == "XeniumSlide"
    assert manifest["counts"]["cells"] == 3
    assert manifest["counts"]["contours"] == 1
    assert manifest["counts"]["assigned_cells"] == 3
    assert qc["status"] == "pass"
    assert set(cell_to_contour["assignment_status"]) == {"assigned"}
    assert set(cell_to_contour["contour_coordinate_space"]) == {"xenium_pixel"}
    assert patch_manifest[0]["structure_label"] == "tumor_region"
    assert "contours" in slide.shapes
    assert "contour_id" in slide.table.obs.columns


def test_read_xenium_slide_can_stream_transcripts_for_export(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")

    sdata = read_xenium(str(dataset), as_="slide", prefer="mex", stream_transcripts=True)

    assert isinstance(sdata, XeniumSlide)
    assert "transcripts" not in sdata.points
    assert "transcripts" in sdata.point_sources
    assert sdata.component_summary()["points"] == ["transcripts"]

    streamed = sdata.point_sources["transcripts"].materialize()
    expected = read_xenium(str(dataset), as_="slide", prefer="mex").points["transcripts"]
    pd.testing.assert_frame_equal(streamed, expected)


def test_read_xenium_streamed_transcripts_allow_missing_cell_id(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset_missing_cell_id",
        transcripts_include_cell_id=False,
        transcripts_include_z_coordinate=True,
    )

    sdata = read_xenium(str(dataset), as_="slide", prefer="mex", stream_transcripts=True)
    streamed = sdata.point_sources["transcripts"].materialize()

    assert "cell_id" not in streamed.columns
    assert {"x", "y", "z", "gene_identity", "gene_name", "quality_score", "valid"} <= set(streamed.columns)
    assert streamed["z"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_read_xenium_streamed_transcripts_preserve_extra_columns(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset_extra_columns",
        transcript_extra_columns={
            "status": np.asarray([1, 0, 1, 1], dtype=np.uint8),
            "fov_name": np.asarray(["fov_a", "fov_a", "fov_b", "fov_b"], dtype=np.str_),
            "codeword_identity": np.asarray([[11, 101], [12, 102], [13, 103], [14, 104]], dtype=np.int64),
        },
    )

    sdata = read_xenium(str(dataset), as_="slide", prefer="mex", stream_transcripts=True)
    streamed = sdata.point_sources["transcripts"].materialize()

    assert {"status", "fov_name", "codeword_identity_0", "codeword_identity_1"} <= set(streamed.columns)
    assert streamed["status"].tolist() == [1, 0, 1, 1]
    assert streamed["fov_name"].tolist() == ["fov_a", "fov_a", "fov_b", "fov_b"]
    assert streamed["codeword_identity_0"].tolist() == [11, 12, 13, 14]
    assert streamed["codeword_identity_1"].tolist() == [101, 102, 103, 104]


def test_read_xenium_slide_loads_he_image_and_alignment_metadata(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)

    sdata = read_xenium(str(dataset), as_="slide", prefer="zarr", include_images=True)

    assert "he" in sdata.images
    he = sdata.images["he"]
    assert he.axes == "yxc"
    assert len(he.levels) >= 2
    assert he.pixel_size_um == pytest.approx(0.5)
    assert he.alignment_csv_path is not None
    assert he.alignment_csv_path.endswith("sample_he_alignment.csv")
    np.testing.assert_allclose(
        np.asarray(he.image_to_xenium_affine),
        np.asarray([[2.0, 0.0, 10.0], [0.0, 3.0, 5.0], [0.0, 0.0, 1.0]]),
    )
    assert he.keypoints_validation is not None
    assert he.keypoints_validation["n_keypoints"] == 4
    assert he.keypoints_validation["max_residual"] == pytest.approx(0.0)
    assert he.transform_metadata()["transform_direction"] == "image_pixel_xy_to_xenium_pixel_xy"
    assert he.transform_metadata()["transform_output_space"] == "xenium_pixel_xy"
    assert he.transform_metadata()["transform_output_unit"] == "pixel"
    np.testing.assert_allclose(
        he.image_xy_to_xenium_pixel_xy([[0.0, 0.0], [10.0, 20.0]]),
        np.asarray([[10.0, 5.0], [30.0, 65.0]]),
    )
    np.testing.assert_allclose(
        he.xenium_um_to_image_xy([[5.0, 2.5], [15.0, 32.5]]),
        np.asarray([[0.0, 0.0], [10.0, 20.0]]),
    )
    assert "he" in sdata.metadata["image_artifacts"]


def test_slide_roundtrip(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_extra_clusterings=True,
        include_projections=True,
    )
    sdata = read_xenium(str(dataset), as_="slide", prefer="zarr")

    output = tmp_path / "pyxenium_slide.zarr"
    payload = write_xenium(sdata, output, format="slide")
    reloaded = read_slide(output)

    assert payload["format"] == "pyxenium.slide"
    assert reloaded.table.shape == sdata.table.shape
    assert reloaded.points["transcripts"].shape == sdata.points["transcripts"].shape
    assert reloaded.shapes["cell_boundaries"].shape == sdata.shapes["cell_boundaries"].shape
    assert reloaded.metadata["cluster_key"] == "gene_expression_graphclust"
    assert list(reloaded.table.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS
    assert list(reloaded.table.obs["cluster__gene_expression_kmeans_2_clusters"].astype(str)) == KMEANS2_LABELS
    np.testing.assert_allclose(reloaded.table.obsm["X_umap"], UMAP_VALUES)
    assert reloaded.table.uns["xenium_analysis"]["projection_keys"]["umap"] == "X_umap"


def test_slide_roundtrip_preserves_streamed_transcripts(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")
    sdata = read_xenium(str(dataset), as_="slide", prefer="zarr", stream_transcripts=True)

    output = tmp_path / "pyxenium_slide_streamed.zarr"
    payload = write_xenium(sdata, output, format="slide")
    reloaded = read_slide(output)
    expected = read_xenium(str(dataset), as_="slide", prefer="zarr").points["transcripts"]

    assert payload["points"] == ["transcripts"]
    pd.testing.assert_frame_equal(reloaded.points["transcripts"], expected)


def test_slide_roundtrip_preserves_streamed_transcript_extra_columns(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset_streamed_extra_roundtrip",
        transcripts_include_cell_id=False,
        transcript_extra_columns={
            "status": np.asarray([1, 0, 1, 1], dtype=np.uint8),
            "fov_name": np.asarray(["fov_a", "fov_a", "fov_b", "fov_b"], dtype=np.str_),
        },
        transcripts_include_z_coordinate=True,
    )
    sdata = read_xenium(str(dataset), as_="slide", prefer="zarr", stream_transcripts=True)

    output = tmp_path / "pyxenium_slide_streamed_extra.zarr"
    payload = write_xenium(sdata, output, format="slide")
    reloaded = read_slide(output)

    assert payload["points"] == ["transcripts"]
    assert {"z", "status", "fov_name"} <= set(reloaded.points["transcripts"].columns)
    assert "cell_id" not in reloaded.points["transcripts"].columns


def test_slide_roundtrip_preserves_he_images(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    sdata = read_xenium(str(dataset), as_="slide", prefer="zarr", include_images=True)

    output = tmp_path / "pyxenium_slide_he.zarr"
    payload = write_xenium(sdata, output, format="slide")
    reloaded = read_slide(output)

    assert "he" in payload["images"]
    assert "he" in reloaded.images
    he = reloaded.images["he"]
    assert he.axes == "yxc"
    assert len(he.levels) == len(sdata.images["he"].levels)
    assert tuple(int(value) for value in he.levels[0].shape) == sdata.images["he"].multiscale_shapes()[0]
    assert he.keypoints_validation is not None
    assert he.keypoints_validation["mean_residual"] == pytest.approx(0.0)
    assert he.transform_metadata()["xenium_physical_unit"] == "micron"
    assert he.transform_metadata()["xenium_pixel_size_um"] == pytest.approx(0.5)


def test_load_xenium_gene_protein_wrapper_matches_new_api(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")

    with pytest.warns(DeprecationWarning):
        legacy = load_xenium_gene_protein(str(dataset), prefer="zarr")
    modern = load_rna_protein_anndata(str(dataset), prefer="zarr")

    assert legacy.shape == modern.shape
    assert list(legacy.obsm["protein"].columns) == list(modern.obsm["protein"].columns)
    np.testing.assert_allclose(legacy.obsm["spatial"], modern.obsm["spatial"])


def test_export_xenium_to_slide_zarr_writes_compat_store(tmp_path, monkeypatch):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)

    def _unexpected_materialization(*args, **kwargs):
        raise AssertionError("export_xenium_to_slide_zarr should stream transcripts.")

    monkeypatch.setattr(xenium_api_module, "read_transcripts_table", _unexpected_materialization)

    payload = export_xenium_to_slide_zarr(str(dataset), overwrite=True)
    reloaded = read_slide(payload["output_path"])

    assert payload["format"] == "pyxenium.slide"
    assert payload["tables"] == ["cells"]
    assert payload["points"] == ["transcripts"]
    assert "he" in payload["images"]
    assert "he" in reloaded.images
    assert isinstance(reloaded, XeniumSlide)


def test_read_xenium_slide_gracefully_handles_missing_optional_artifacts(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("mex",),
        include_boundaries=False,
        include_transcripts=False,
        include_analysis=False,
    )

    sdata = read_xenium(str(dataset), as_="slide", prefer="mex")

    assert sdata.table.n_obs == 3
    assert sdata.points == {}
    assert sdata.shapes == {}
    assert "cluster" not in sdata.table.obs.columns
    assert "X_umap" not in sdata.table.obsm


def test_read_xenium_slide_gracefully_handles_missing_he_image(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("mex",),
        include_boundaries=False,
        include_transcripts=False,
        include_he_image=False,
    )

    sdata = read_xenium(str(dataset), as_="slide", prefer="mex", include_images=True)

    assert sdata.images == {}


def test_read_xenium_supports_double_analysis_directory_variant(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_projections=True,
        primary_analysis_root="analysis/analysis",
    )

    adata = read_xenium(str(dataset), as_="anndata", prefer="zarr")

    assert list(adata.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS
    np.testing.assert_allclose(adata.obsm["X_umap"], UMAP_VALUES)
    assert (
        adata.uns["xenium_analysis"]["projection_sources"]["umap"]["relpath"]
        == "analysis/analysis/umap/gene_expression_2_components/projection.csv"
    )


def test_load_anndata_from_partial_reuses_shared_artifact_readers(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "partial",
        include_backends=("mex",),
        include_boundaries=False,
        include_transcripts=False,
        include_zarr_sidecars=True,
    )

    adata = load_anndata_from_partial(
        mex_dir=str(dataset / "cell_feature_matrix"),
        analysis_name=str(dataset / "analysis.zarr.zip"),
        cells_name=str(dataset / "cells.zarr.zip"),
    )

    assert adata.shape == (3, 4)
    assert "cluster" in adata.obs.columns
    assert "spatial" in adata.obsm
    assert adata.uns["analysis"]["n_clusters"] == 2


def test_partial_loader_can_return_empty_anndata_when_counts_missing(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "partial_empty",
        include_backends=(),
        include_boundaries=False,
        include_transcripts=False,
        include_zarr_sidecars=True,
    )

    adata = load_anndata_from_partial(
        analysis_name=str(dataset / "analysis.zarr.zip"),
        cells_name=str(dataset / "cells.zarr.zip"),
        build_counts_if_missing=True,
    )

    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (0, 0)
    assert "analysis" in adata.uns
    assert "cells" in adata.uns
