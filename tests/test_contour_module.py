from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from shapely import union_all

import pyXenium.contour as contour
from pyXenium.contour import (
    add_contours_from_geojson,
    expand_contours,
    ring_density,
    smooth_density_by_distance,
)
from pyXenium.contour._geometry import contour_frame_to_geometry_table
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
        metadata={
            "units": "micron",
            "contours": {"contours": {"units": "micron", "annotation_source": "synthetic"}},
        },
    )


def _make_nearby_contour_sdata() -> XeniumSData:
    sdata = _make_contour_ready_sdata()
    contour_frame = pd.DataFrame(
        {
            "contour_id": ["left"] * 4 + ["right"] * 4,
            "part_id": [0] * 8,
            "ring_id": [0] * 8,
            "is_hole": [False] * 8,
            "vertex_id": [0, 1, 2, 3] * 2,
            "x": [0.0, 10.0, 10.0, 0.0, 14.0, 24.0, 24.0, 14.0],
            "y": [0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0],
            "assigned_structure": ["Left"] * 4 + ["Right"] * 4,
            "classification_name": ["Synthetic"] * 8,
            "annotation_source": ["manual"] * 8,
            "structure_id": ["L"] * 4 + ["R"] * 4,
            "segmentation_source": ["unit-test"] * 8,
            "score": [0.25] * 4 + [0.75] * 4,
        }
    )
    sdata.shapes["neighbor_contours"] = contour_frame
    sdata.metadata["contours"]["neighbor_contours"] = {
        "units": "micron",
        "annotation_source": "synthetic",
    }
    return sdata


def _make_complex_contour_sdata() -> XeniumSData:
    sdata = _make_contour_ready_sdata()
    donut_frame = pd.DataFrame(
        {
            "contour_id": ["donut"] * 8,
            "part_id": [0] * 8,
            "ring_id": [0] * 4 + [1] * 4,
            "is_hole": [False] * 4 + [True] * 4,
            "vertex_id": [0, 1, 2, 3] * 2,
            "x": [0.0, 20.0, 20.0, 0.0, 7.0, 13.0, 13.0, 7.0],
            "y": [0.0, 0.0, 20.0, 20.0, 7.0, 7.0, 13.0, 13.0],
            "assigned_structure": ["Donut"] * 8,
            "classification_name": ["Hole rich"] * 8,
            "annotation_source": ["manual"] * 8,
            "structure_id": ["D"] * 8,
        }
    )
    multi_frame = pd.DataFrame(
        {
            "contour_id": ["multi"] * 8,
            "part_id": [0] * 4 + [1] * 4,
            "ring_id": [0] * 8,
            "is_hole": [False] * 8,
            "vertex_id": [0, 1, 2, 3] * 2,
            "x": [40.0, 50.0, 50.0, 40.0, 60.0, 70.0, 70.0, 60.0],
            "y": [0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0],
            "assigned_structure": ["Multi"] * 8,
            "classification_name": ["Two islands"] * 8,
            "annotation_source": ["manual"] * 8,
            "structure_id": ["M"] * 8,
        }
    )
    sdata.shapes["complex_contours"] = pd.concat([donut_frame, multi_frame], ignore_index=True)
    sdata.metadata["contours"]["complex_contours"] = {
        "units": "micron",
        "annotation_source": "synthetic",
    }
    return sdata


def _geometry_table_for_key(sdata: XeniumSData, contour_key: str) -> pd.DataFrame:
    return contour_frame_to_geometry_table(sdata.shapes[contour_key], contour_key=contour_key).sort_values(
        "contour_id",
        kind="stable",
    ).reset_index(drop=True)


def test_contour_public_imports_smoke():
    assert callable(contour.add_contours_from_geojson)
    assert callable(contour.expand_contours)
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


def test_expand_contours_overlap_creates_new_layer_and_preserves_metadata():
    sdata = _make_nearby_contour_sdata()

    expand_contours(
        sdata,
        contour_key="neighbor_contours",
        distance=3.0,
        mode="overlap",
    )

    assert "neighbor_contours_expanded" in sdata.shapes
    expanded_frame = sdata.shapes["neighbor_contours_expanded"]
    assert expanded_frame["contour_id"].tolist().count("left") > 0
    assert expanded_frame["segmentation_source"].eq("unit-test").all()
    assert expanded_frame["score"].notna().all()

    source_table = _geometry_table_for_key(sdata, "neighbor_contours")
    expanded_table = _geometry_table_for_key(sdata, "neighbor_contours_expanded")
    assert np.all(expanded_table["geometry"].map(lambda geom: geom.area).to_numpy() > source_table["geometry"].map(lambda geom: geom.area).to_numpy())

    left_geometry = expanded_table.loc[expanded_table["contour_id"] == "left", "geometry"].item()
    right_geometry = expanded_table.loc[expanded_table["contour_id"] == "right", "geometry"].item()
    assert left_geometry.intersection(right_geometry).area > 0.0

    metadata = sdata.metadata["contours"]["neighbor_contours_expanded"]
    assert metadata["derived_from_key"] == "neighbor_contours"
    assert metadata["generator"] == "expand_contours"
    assert metadata["expansion_mode"] == "overlap"
    assert metadata["expansion_distance"] == 3.0
    assert metadata["expansion_distance_um"] == 3.0


def test_expand_contours_voronoi_is_disjoint_and_matches_overlap_support():
    base = _make_nearby_contour_sdata()
    overlap = expand_contours(
        base,
        contour_key="neighbor_contours",
        distance=3.0,
        mode="overlap",
        output_key="neighbor_overlap",
        copy=True,
    )
    voronoi = expand_contours(
        base,
        contour_key="neighbor_contours",
        distance=3.0,
        mode="voronoi",
        output_key="neighbor_voronoi",
        copy=True,
        voronoi_sample_step=1.0,
    )

    assert overlap is not None
    assert voronoi is not None

    overlap_table = _geometry_table_for_key(overlap, "neighbor_overlap")
    voronoi_table = _geometry_table_for_key(voronoi, "neighbor_voronoi")
    overlap_union = union_all(list(overlap_table["geometry"]))
    voronoi_union = union_all(list(voronoi_table["geometry"]))

    left_geometry = voronoi_table.loc[voronoi_table["contour_id"] == "left", "geometry"].item()
    right_geometry = voronoi_table.loc[voronoi_table["contour_id"] == "right", "geometry"].item()
    assert left_geometry.intersection(right_geometry).area < 1e-8
    assert abs(float(overlap_union.area) - float(voronoi_union.area)) < 1e-6
    assert left_geometry.within(
        overlap_table.loc[overlap_table["contour_id"] == "left", "geometry"].item().buffer(1e-8)
    )
    assert right_geometry.within(
        overlap_table.loc[overlap_table["contour_id"] == "right", "geometry"].item().buffer(1e-8)
    )

    metadata = voronoi.metadata["contours"]["neighbor_voronoi"]
    assert metadata["expansion_mode"] == "voronoi"
    assert metadata["voronoi_sample_step"] == 1.0
    assert metadata["voronoi_sample_step_um"] == 1.0


def test_expand_contours_copy_mode_and_roundtrip(tmp_path):
    sdata = _make_nearby_contour_sdata()

    copied = expand_contours(
        sdata,
        contour_key="neighbor_contours",
        distance=2.0,
        mode="voronoi",
        copy=True,
        output_key="neighbor_voronoi",
        voronoi_sample_step=1.0,
    )

    assert copied is not None
    assert "neighbor_voronoi" not in sdata.shapes
    assert "neighbor_voronoi" in copied.shapes

    output = write_xenium(copied, tmp_path / "contour_roundtrip.zarr", format="sdata")
    reloaded = read_sdata(output["output_path"])
    reloaded_table = _geometry_table_for_key(reloaded, "neighbor_voronoi")
    copied_table = _geometry_table_for_key(copied, "neighbor_voronoi")
    assert set(reloaded_table["contour_id"]) == set(copied_table["contour_id"])
    assert reloaded.metadata["contours"]["neighbor_voronoi"]["derived_from_key"] == "neighbor_contours"


def test_expand_contours_supports_holes_and_multipolygons():
    sdata = _make_complex_contour_sdata()

    expanded = expand_contours(
        sdata,
        contour_key="complex_contours",
        distance=1.0,
        mode="overlap",
        output_key="complex_expanded",
        copy=True,
    )

    assert expanded is not None
    frame = expanded.shapes["complex_expanded"]
    assert frame["contour_id"].nunique() == 2
    assert frame["is_hole"].any()
    multi_parts = frame.loc[frame["contour_id"] == "multi", "part_id"].nunique()
    assert multi_parts == 2

    source_table = _geometry_table_for_key(sdata, "complex_contours")
    expanded_table = _geometry_table_for_key(expanded, "complex_expanded")
    source_areas = source_table.set_index("contour_id")["geometry"].map(lambda geom: geom.area)
    expanded_areas = expanded_table.set_index("contour_id")["geometry"].map(lambda geom: geom.area)
    assert expanded_areas["donut"] > source_areas["donut"]
    assert expanded_areas["multi"] > source_areas["multi"]


def test_expand_contours_voronoi_single_contour_matches_overlap():
    sdata = _make_contour_ready_sdata()

    overlap = expand_contours(
        sdata,
        contour_key="contours",
        distance=2.0,
        mode="overlap",
        output_key="contours_overlap",
        copy=True,
    )
    voronoi = expand_contours(
        sdata,
        contour_key="contours",
        distance=2.0,
        mode="voronoi",
        output_key="contours_voronoi",
        copy=True,
        voronoi_sample_step=1.0,
    )

    assert overlap is not None
    assert voronoi is not None
    overlap_geometry = _geometry_table_for_key(overlap, "contours_overlap").loc[0, "geometry"]
    voronoi_geometry = _geometry_table_for_key(voronoi, "contours_voronoi").loc[0, "geometry"]
    assert overlap_geometry.symmetric_difference(voronoi_geometry).area < 1e-8


def test_expand_contours_validation():
    sdata = _make_contour_ready_sdata()

    with pytest.raises(ValueError, match="`distance` must be greater than 0"):
        expand_contours(sdata, contour_key="contours", distance=0.0)
    with pytest.raises(ValueError, match="`mode` must be one of"):
        expand_contours(sdata, contour_key="contours", distance=1.0, mode="invalid")
    with pytest.raises(KeyError, match="missing_contours"):
        expand_contours(sdata, contour_key="missing_contours", distance=1.0)
    with pytest.raises(ValueError, match="`voronoi_sample_step` must be greater than 0"):
        expand_contours(
            sdata,
            contour_key="contours",
            distance=1.0,
            mode="voronoi",
            output_key="contours_voronoi",
            voronoi_sample_step=0.0,
        )

    expand_contours(sdata, contour_key="contours", distance=1.0)
    with pytest.raises(KeyError, match="contours_expanded"):
        expand_contours(sdata, contour_key="contours", distance=1.0)
