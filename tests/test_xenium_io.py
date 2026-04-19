from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy import sparse
from scipy.io import mmwrite

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
    chunk.create_dataset(
        "location",
        data=np.asarray([[10.0, 20.0], [15.0, 25.0], [30.0, 40.0], [35.0, 45.0]]),
    )
    chunk.create_dataset("gene_identity", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
    chunk.create_dataset("quality_score", data=np.asarray([30.0, 25.0, 18.0, 35.0]))
    chunk.create_dataset("valid", data=np.asarray([1, 1, 1, 0], dtype=np.uint8))
    chunk.create_dataset(
        "cell_id",
        data=np.asarray([BARCODES[0], BARCODES[1], BARCODES[2], BARCODES[0]], dtype=np.str_),
    )
    store.close()


def _make_cells_zip(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w")
    root.create_dataset("cell_id", data=np.asarray([[0, 1], [1, 1], [2, 1]], dtype=np.uint32))
    cell_summary = root.create_dataset(
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
    group.create_dataset("indices", data=np.asarray([0, 2, 1], dtype=np.int64))
    group.create_dataset("indptr", data=np.asarray([0, 2, 3], dtype=np.int64))
    cell_groups.attrs["group_names"] = [["Tumor", "Immune"]]
    cell_groups.attrs["grouping_names"] = ["graphclust"]
    store.close()


def _make_cell_feature_matrix_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    matrix = CELL_BY_FEATURE.T.tocsr()
    x_group = root.require_group("X")
    x_group.create_dataset("data", data=matrix.data)
    x_group.create_dataset("indices", data=matrix.indices)
    x_group.create_dataset("indptr", data=matrix.indptr)
    x_group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))

    features_group = root.require_group("features")
    features_group.create_dataset("id", data=FEATURES["id"].to_numpy(dtype=np.str_))
    features_group.create_dataset("name", data=FEATURES["name"].to_numpy(dtype=np.str_))
    features_group.create_dataset(
        "feature_type", data=FEATURES["feature_type"].to_numpy(dtype=np.str_)
    )
    root.create_dataset("barcodes", data=BARCODES.to_numpy(dtype=np.str_))


def _make_cell_feature_matrix_h5(path: Path) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 Xenium fixtures.")
    matrix = CELL_BY_FEATURE.T.tocsr()
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
        features_group = group.create_group("features")
        features_group.create_dataset("id", data=FEATURES["id"].to_numpy(dtype="S"))
        features_group.create_dataset("name", data=FEATURES["name"].to_numpy(dtype="S"))
        features_group.create_dataset(
            "feature_type", data=FEATURES["feature_type"].to_numpy(dtype="S")
        )
        group.create_dataset("barcodes", data=BARCODES.to_numpy(dtype="S"))


def make_xenium_dataset(
    base_path: Path,
    *,
    include_backends: tuple[str, ...] = ("mex", "zarr", "h5"),
    include_boundaries: bool = True,
    include_transcripts: bool = True,
    include_zarr_sidecars: bool = False,
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

    cluster_dir = base_path / "analysis" / "clustering" / "gene_expression_graphclust"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"cell_id": BARCODES.to_list(), "cluster": ["Tumor", "Immune", "Tumor"]}
    ).to_csv(cluster_dir / "clusters.csv", index=False)

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


def test_read_xenium_sdata_contains_expected_components(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")

    sdata = read_xenium(str(dataset), as_="sdata", prefer="mex")

    assert isinstance(sdata, XeniumSData)
    assert sdata.table.n_obs == 3
    assert "transcripts" in sdata.points
    assert {"x", "y", "gene_identity", "gene_name"}.issubset(sdata.points["transcripts"].columns)
    assert set(sdata.shapes) == {"cell_boundaries", "nucleus_boundaries"}
    assert sdata.metadata["units"] == "micron"
    assert sdata.metadata["feature_summary"]["n_features_total"] == 4


def test_sdata_roundtrip(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")
    sdata = read_xenium(str(dataset), as_="sdata", prefer="zarr")

    output = tmp_path / "pyxenium_sdata.zarr"
    payload = write_xenium(sdata, output, format="sdata")
    reloaded = read_sdata(output)

    assert payload["format"] == "pyxenium.sdata"
    assert reloaded.table.shape == sdata.table.shape
    assert reloaded.points["transcripts"].shape == sdata.points["transcripts"].shape
    assert reloaded.shapes["cell_boundaries"].shape == sdata.shapes["cell_boundaries"].shape
    assert reloaded.metadata["cluster_key"] == "cluster"


def test_load_xenium_gene_protein_wrapper_matches_new_api(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")

    legacy = load_xenium_gene_protein(str(dataset), prefer="zarr")
    modern = read_xenium(str(dataset), as_="anndata", prefer="zarr", include_transcripts=False)

    assert legacy.shape == modern.shape
    assert list(legacy.obsm["protein"].columns) == list(modern.obsm["protein"].columns)
    np.testing.assert_allclose(legacy.obsm["spatial"], modern.obsm["spatial"])


def test_export_xenium_to_spatialdata_zarr_writes_compat_store(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset")

    payload = export_xenium_to_spatialdata_zarr(str(dataset), overwrite=True)
    reloaded = read_sdata(payload["output_path"])

    assert payload["format"] == "pyxenium.sdata"
    assert payload["tables"] == ["cells"]
    assert payload["points"] == ["transcripts"]
    assert isinstance(reloaded, XeniumSData)


def test_read_xenium_sdata_gracefully_handles_missing_optional_artifacts(tmp_path):
    dataset = make_xenium_dataset(
        tmp_path / "dataset",
        include_backends=("mex",),
        include_boundaries=False,
        include_transcripts=False,
    )

    sdata = read_xenium(str(dataset), as_="sdata", prefer="mex")

    assert sdata.table.n_obs == 3
    assert sdata.points == {}
    assert sdata.shapes == {}


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
