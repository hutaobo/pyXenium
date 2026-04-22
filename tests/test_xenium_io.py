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
from pyXenium.io import (
    XeniumSData,
    export_xenium_to_spatialdata_zarr,
    load_anndata_from_partial,
    load_xenium_gene_protein,
    read_sdata,
    read_xenium,
    write_xenium,
)

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


def _make_transcripts_zip(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    root.attrs["gene_names"] = ["Gene1", "Gene2"]
    grids = root.require_group("grids")
    level = grids.require_group("0")
    chunk = level.require_group("0,0")
    chunk.create_array(
        "location",
        data=np.asarray([[10.0, 20.0], [15.0, 25.0], [30.0, 40.0], [35.0, 45.0]]),
    )
    chunk.create_array("gene_identity", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
    chunk.create_array("quality_score", data=np.asarray([30.0, 25.0, 18.0, 35.0]))
    chunk.create_array("valid", data=np.asarray([1, 1, 1, 0], dtype=np.uint8))
    chunk.create_array(
        "cell_id",
        data=np.asarray([BARCODES[0], BARCODES[1], BARCODES[2], BARCODES[0]], dtype=np.str_),
    )
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
        _make_transcripts_zip(base_path / "transcripts.zarr.zip")

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


def test_read_xenium_sdata_contains_expected_components(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_extra_clusterings=True,
        include_projections=True,
    )

    sdata = read_xenium(str(dataset), as_="sdata", prefer="mex")

    assert isinstance(sdata, XeniumSData)
    assert sdata.table.n_obs == 3
    assert "transcripts" in sdata.points
    assert {"x", "y", "gene_identity", "gene_name"}.issubset(sdata.points["transcripts"].columns)
    assert set(sdata.shapes) == {"cell_boundaries", "nucleus_boundaries"}
    assert sdata.metadata["units"] == "micron"
    assert sdata.metadata["feature_summary"]["n_features_total"] == 4
    assert sdata.metadata["cluster_key"] == "gene_expression_graphclust"
    assert "cluster__gene_expression_kmeans_2_clusters" in sdata.table.obs.columns
    np.testing.assert_allclose(sdata.table.obsm["X_umap"], UMAP_VALUES)


def test_read_xenium_sdata_can_stream_transcripts_for_export(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")

    sdata = read_xenium(str(dataset), as_="sdata", prefer="mex", stream_transcripts=True)

    assert isinstance(sdata, XeniumSData)
    assert "transcripts" not in sdata.points
    assert "transcripts" in sdata.point_sources
    assert sdata.component_summary()["points"] == ["transcripts"]

    streamed = sdata.point_sources["transcripts"].materialize()
    expected = read_xenium(str(dataset), as_="sdata", prefer="mex").points["transcripts"]
    pd.testing.assert_frame_equal(streamed, expected)


def test_read_xenium_sdata_loads_he_image_and_alignment_metadata(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)

    sdata = read_xenium(str(dataset), as_="sdata", prefer="zarr", include_images=True)

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


def test_sdata_roundtrip(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_extra_clusterings=True,
        include_projections=True,
    )
    sdata = read_xenium(str(dataset), as_="sdata", prefer="zarr")

    output = tmp_path / "pyxenium_sdata.zarr"
    payload = write_xenium(sdata, output, format="sdata")
    reloaded = read_sdata(output)

    assert payload["format"] == "pyxenium.sdata"
    assert reloaded.table.shape == sdata.table.shape
    assert reloaded.points["transcripts"].shape == sdata.points["transcripts"].shape
    assert reloaded.shapes["cell_boundaries"].shape == sdata.shapes["cell_boundaries"].shape
    assert reloaded.metadata["cluster_key"] == "gene_expression_graphclust"
    assert list(reloaded.table.obs["cluster"].astype(str)) == GRAPHCLUST_LABELS
    assert list(reloaded.table.obs["cluster__gene_expression_kmeans_2_clusters"].astype(str)) == KMEANS2_LABELS
    np.testing.assert_allclose(reloaded.table.obsm["X_umap"], UMAP_VALUES)
    assert reloaded.table.uns["xenium_analysis"]["projection_keys"]["umap"] == "X_umap"


def test_sdata_roundtrip_preserves_streamed_transcripts(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")
    sdata = read_xenium(str(dataset), as_="sdata", prefer="zarr", stream_transcripts=True)

    output = tmp_path / "pyxenium_sdata_streamed.zarr"
    payload = write_xenium(sdata, output, format="sdata")
    reloaded = read_sdata(output)
    expected = read_xenium(str(dataset), as_="sdata", prefer="zarr").points["transcripts"]

    assert payload["points"] == ["transcripts"]
    pd.testing.assert_frame_equal(reloaded.points["transcripts"], expected)


def test_sdata_roundtrip_preserves_he_images(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    sdata = read_xenium(str(dataset), as_="sdata", prefer="zarr", include_images=True)

    output = tmp_path / "pyxenium_sdata_he.zarr"
    payload = write_xenium(sdata, output, format="sdata")
    reloaded = read_sdata(output)

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

    legacy = load_xenium_gene_protein(str(dataset), prefer="zarr")
    modern = read_xenium(str(dataset), as_="anndata", prefer="zarr", include_transcripts=False)

    assert legacy.shape == modern.shape
    assert list(legacy.obsm["protein"].columns) == list(modern.obsm["protein"].columns)
    np.testing.assert_allclose(legacy.obsm["spatial"], modern.obsm["spatial"])


def test_export_xenium_to_spatialdata_zarr_writes_compat_store(tmp_path, monkeypatch):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)

    def _unexpected_materialization(*args, **kwargs):
        raise AssertionError("export_xenium_to_spatialdata_zarr should stream transcripts.")

    monkeypatch.setattr(xenium_api_module, "read_transcripts_table", _unexpected_materialization)

    payload = export_xenium_to_spatialdata_zarr(str(dataset), overwrite=True)
    reloaded = read_sdata(payload["output_path"])

    assert payload["format"] == "pyxenium.sdata"
    assert payload["tables"] == ["cells"]
    assert payload["points"] == ["transcripts"]
    assert "he" in payload["images"]
    assert "he" in reloaded.images
    assert isinstance(reloaded, XeniumSData)


def test_read_xenium_sdata_gracefully_handles_missing_optional_artifacts(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("mex",),
        include_boundaries=False,
        include_transcripts=False,
        include_analysis=False,
    )

    sdata = read_xenium(str(dataset), as_="sdata", prefer="mex")

    assert sdata.table.n_obs == 3
    assert sdata.points == {}
    assert sdata.shapes == {}
    assert "cluster" not in sdata.table.obs.columns
    assert "X_umap" not in sdata.table.obsm


def test_read_xenium_sdata_gracefully_handles_missing_he_image(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("mex",),
        include_boundaries=False,
        include_transcripts=False,
        include_he_image=False,
    )

    sdata = read_xenium(str(dataset), as_="sdata", prefer="mex", include_images=True)

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
