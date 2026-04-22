from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import pyXenium.contour as contour
from pyXenium.contour import (
    add_contours_from_geojson,
    ring_density,
    smooth_density_by_distance,
)
from pyXenium.io import XeniumFrameChunkSource, XeniumSData, read_sdata, read_xenium, write_xenium

from test_xenium_io import make_xenium_dataset


def _write_geojson(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _simple_contour_geojson() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": "structure4_a",
                    "segmentation_source": "protein_clusters",
                    "metadata": {
                        "assigned_structure": "Structure 4",
                        "classification_name": "Tumor rim",
                        "annotation_source": "manual",
                        "structure_id": "4",
                        "name": "Structure 4",
                        "object_type": "polygon_unit",
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [40.0, 0.0],
                            [40.0, 40.0],
                            [0.0, 40.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": "structure2_b",
                    "segmentation_source": "protein_clusters",
                    "metadata": {
                        "assigned_structure": "Structure 2",
                        "classification_name": "Immune niche",
                        "annotation_source": "manual",
                        "structure_id": "2",
                        "name": "Structure 2",
                        "object_type": "polygon_unit",
                    },
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [
                            [
                                [100.0, 100.0],
                                [120.0, 100.0],
                                [120.0, 120.0],
                                [100.0, 120.0],
                                [100.0, 100.0],
                            ]
                        ],
                        [
                            [
                                [130.0, 100.0],
                                [150.0, 100.0],
                                [150.0, 120.0],
                                [130.0, 120.0],
                                [130.0, 100.0],
                            ]
                        ],
                    ],
                },
            },
        ],
    }


def _make_contour_ready_sdata(*, streamed_transcripts: bool = False) -> XeniumSData:
    obs = pd.DataFrame(
        {
            "x_centroid": [5.0, 9.0, 0.0, 12.0, 14.0, 50.0],
            "y_centroid": [5.0, 5.0, 5.0, 5.0, 5.0, 50.0],
            "cell_class": ["A", "B", "B", "A", "A", "C"],
        },
        index=pd.Index([f"cell_{idx}" for idx in range(6)], name="barcode"),
    )
    adata = ad.AnnData(X=np.zeros((len(obs), 0), dtype=float), obs=obs)

    transcripts = pd.DataFrame(
        {
            "x": [5.0, 9.0, 0.0, 12.0, 14.0, 50.0],
            "y": [5.0, 5.0, 5.0, 5.0, 5.0, 50.0],
            "gene_identity": [0, 0, 0, 0, 0, 0],
            "gene_name": ["VIM", "VIM", "VIM", "VIM", "VIM", "ACTB"],
            "quality_score": [35.0, 30.0, 25.0, 40.0, 50.0, 30.0],
            "valid": [True, True, True, True, True, True],
            "cell_id": ["cell_0", "cell_1", "cell_2", "cell_3", "cell_4", "cell_5"],
        }
    )
    contour_frame = pd.DataFrame(
        {
            "contour_id": ["contour_a"] * 4,
            "part_id": [0, 0, 0, 0],
            "ring_id": [0, 0, 0, 0],
            "is_hole": [False, False, False, False],
            "vertex_id": [0, 1, 2, 3],
            "x": [0.0, 10.0, 10.0, 0.0],
            "y": [0.0, 0.0, 10.0, 10.0],
            "assigned_structure": ["Structure 4"] * 4,
            "classification_name": ["Tumor rim"] * 4,
            "annotation_source": ["manual"] * 4,
            "structure_id": ["4"] * 4,
        }
    )

    if streamed_transcripts:
        midpoint = 3
        chunk_1 = transcripts.iloc[:midpoint].copy()
        chunk_2 = transcripts.iloc[midpoint:].copy()
        point_source = XeniumFrameChunkSource(
            columns=tuple(transcripts.columns),
            column_types={
                "x": "float64",
                "y": "float64",
                "gene_identity": "int64",
                "gene_name": "string",
                "quality_score": "float64",
                "valid": "bool",
                "cell_id": "string",
            },
            chunk_iter_factory=lambda: iter([chunk_1, chunk_2]),
        )
        return XeniumSData(
            table=adata,
            points={},
            shapes={"contours": contour_frame},
            point_sources={"transcripts": point_source},
            metadata={"units": "micron"},
        )

    return XeniumSData(
        table=adata,
        points={"transcripts": transcripts},
        shapes={"contours": contour_frame},
        metadata={"units": "micron"},
    )


def test_contour_public_imports_smoke():
    assert callable(contour.add_contours_from_geojson)
    assert callable(contour.ring_density)
    assert callable(contour.smooth_density_by_distance)


def test_add_contours_from_geojson_imports_scales_and_roundtrips(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    sdata = read_xenium(str(dataset), as_="sdata", include_images=True, prefer="zarr")
    geojson_path = tmp_path / "contours.geojson"
    _write_geojson(geojson_path, _simple_contour_geojson())

    copied = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="protein_cluster_contours",
        copy=True,
    )

    assert "protein_cluster_contours" not in sdata.shapes
    assert "protein_cluster_contours" in copied.shapes
    frame = copied.shapes["protein_cluster_contours"].sort_values(
        ["contour_id", "part_id", "ring_id", "vertex_id"],
        kind="stable",
    ).reset_index(drop=True)
    assert frame["contour_id"].nunique() == 2
    assert frame["x"].max() == 75.0
    assert {
        "assigned_structure",
        "classification_name",
        "annotation_source",
        "structure_id",
    } <= set(frame.columns)
    assert copied.metadata["contours"]["protein_cluster_contours"]["n_contours"] == 2
    assert copied.metadata["contours"]["protein_cluster_contours"]["pixel_size_um"] == 0.5

    add_contours_from_geojson(sdata, geojson_path, key="protein_cluster_contours", copy=False)
    assert "protein_cluster_contours" in sdata.shapes

    output = tmp_path / "contour_sdata.zarr"
    payload = write_xenium(copied, output, format="sdata")
    reloaded = read_sdata(payload["output_path"])
    reloaded_frame = reloaded.shapes["protein_cluster_contours"].sort_values(
        ["contour_id", "part_id", "ring_id", "vertex_id"],
        kind="stable",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame, reloaded_frame)
    assert reloaded.metadata["contours"]["protein_cluster_contours"]["n_contours"] == 2


def test_ring_density_supports_transcripts_and_cells():
    sdata = _make_contour_ready_sdata()

    transcripts_result = ring_density(
        sdata,
        contour_key="contours",
        target="transcripts",
        contour_query='assigned_structure == "Structure 4"',
        target_query="quality_score >= 20",
        feature_values="VIM",
        inward=5.0,
        outward=5.0,
        ring_width=2.5,
    ).sort_values("ring_start", kind="stable").reset_index(drop=True)

    assert transcripts_result["count"].to_list() == [1, 1, 2, 1]
    assert transcripts_result["ring_mid"].to_list() == [-3.75, -1.25, 1.25, 3.75]
    assert transcripts_result["feature_key"].tolist() == ["gene_name"] * 4
    assert transcripts_result["feature_values"].tolist() == ["VIM"] * 4
    assert np.all(transcripts_result["density"].to_numpy(dtype=float) > 0)

    cells_result = ring_density(
        sdata,
        contour_key="contours",
        target="cells",
        contour_query='assigned_structure == "Structure 4"',
        target_query='cell_class != "C"',
        inward=5.0,
        outward=5.0,
        ring_width=2.5,
    ).sort_values("ring_start", kind="stable").reset_index(drop=True)

    assert cells_result["count"].to_list() == [1, 1, 2, 1]
    assert cells_result["feature_key"].isna().all()
    assert cells_result["feature_values"].isna().all()


def test_smooth_density_profiles_are_monotonic_and_support_cells():
    sdata = _make_contour_ready_sdata()

    transcript_result = smooth_density_by_distance(
        sdata,
        contour_key="contours",
        target="transcripts",
        contour_query='assigned_structure == "Structure 4"',
        feature_values="VIM",
        inward=6.0,
        outward=5.0,
        bandwidth=1.0,
        grid_step=1.0,
    ).sort_values("signed_distance", kind="stable").reset_index(drop=True)

    expected_grid = np.arange(-6.0, 6.0, 1.0, dtype=float)
    expected_grid[-1] = 5.0
    assert np.allclose(transcript_result["signed_distance"].to_numpy(dtype=float), expected_grid)
    near_boundary = transcript_result.loc[
        transcript_result["signed_distance"] == 0.0, "count_density"
    ].item()
    far_outward = transcript_result.loc[
        transcript_result["signed_distance"] == 5.0, "count_density"
    ].item()
    assert near_boundary > far_outward
    deep_inward = transcript_result.loc[transcript_result["signed_distance"] == -6.0].iloc[0]
    assert deep_inward["geometry_measure"] == 0.0
    assert np.isnan(deep_inward["density"])

    cell_result = smooth_density_by_distance(
        sdata,
        contour_key="contours",
        target="cells",
        target_query='cell_class != "C"',
        inward=6.0,
        outward=5.0,
        bandwidth=1.0,
        grid_step=1.0,
    ).sort_values("signed_distance", kind="stable").reset_index(drop=True)

    assert cell_result["feature_key"].isna().all()
    assert cell_result["feature_values"].isna().all()
    assert cell_result.loc[cell_result["signed_distance"] == 0.0, "count_density"].item() > cell_result.loc[
        cell_result["signed_distance"] == 5.0, "count_density"
    ].item()


def test_smooth_density_matches_streamed_and_materialized_transcripts():
    materialized = _make_contour_ready_sdata(streamed_transcripts=False)
    streamed = _make_contour_ready_sdata(streamed_transcripts=True)

    materialized_result = smooth_density_by_distance(
        materialized,
        contour_key="contours",
        target="transcripts",
        feature_values="VIM",
        inward=5.0,
        outward=5.0,
        bandwidth=1.0,
        grid_step=1.0,
    ).sort_values("signed_distance", kind="stable").reset_index(drop=True)
    streamed_result = smooth_density_by_distance(
        streamed,
        contour_key="contours",
        target="transcripts",
        feature_values="VIM",
        inward=5.0,
        outward=5.0,
        bandwidth=1.0,
        grid_step=1.0,
    ).sort_values("signed_distance", kind="stable").reset_index(drop=True)

    np.testing.assert_allclose(
        materialized_result["count_density"].to_numpy(dtype=float),
        streamed_result["count_density"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        materialized_result["geometry_measure"].to_numpy(dtype=float),
        streamed_result["geometry_measure"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        materialized_result["density"].fillna(-1.0).to_numpy(dtype=float),
        streamed_result["density"].fillna(-1.0).to_numpy(dtype=float),
    )


def test_add_contours_from_real_geojson_smoke():
    real_geojson = Path("notebook_outputs/renal_ffpe_real_data/protein_cluster_review/polygon_units.geojson")
    if not real_geojson.exists():
        return

    sdata = XeniumSData(
        table=ad.AnnData(X=np.zeros((0, 0), dtype=float)),
        metadata={"image_artifacts": {"he": {"pixel_size_um": 0.2125}}},
    )
    imported = add_contours_from_geojson(
        sdata,
        real_geojson,
        key="protein_cluster_contours",
        copy=True,
    )

    frame = imported.shapes["protein_cluster_contours"]
    assert frame["contour_id"].nunique() > 100
    assert "assigned_structure" in frame.columns
    assert frame["assigned_structure"].notna().any()
