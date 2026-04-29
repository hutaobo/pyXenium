from __future__ import annotations

import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from shapely import union_all
from shapely.geometry import box

import pyXenium.contour as contour
from pyXenium.contour import (
    add_contours_from_geojson,
    compare_contour_cell_composition,
    compare_contour_de,
    compare_contour_transcript_de,
    expand_contours,
    generate_barrier_contour_shells,
    generate_contour_shells,
    generate_xenium_explorer_annotations,
    ring_density,
    smooth_density_by_distance,
    summarize_contour_composition,
    summarize_contour_topology,
)
from pyXenium.contour._geometry import contour_frame_to_geometry_table, geometry_table_to_contour_frame
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


def _hole_contour_geojson() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": "donut",
                    "segmentation_source": "protein_clusters",
                    "metadata": {
                        "assigned_structure": "Donut structure",
                        "classification_name": "Hole rich",
                        "annotation_source": "manual",
                        "structure_id": "D",
                        "name": "Donut structure",
                        "object_type": "polygon_unit",
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [40.0, 40.0],
                            [120.0, 40.0],
                            [120.0, 120.0],
                            [40.0, 120.0],
                            [40.0, 40.0],
                        ],
                        [
                            [60.0, 60.0],
                            [60.0, 100.0],
                            [100.0, 100.0],
                            [100.0, 60.0],
                            [60.0, 60.0],
                        ],
                    ],
                },
            }
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


def _make_topology_sdata() -> XeniumSData:
    sdata = _make_contour_ready_sdata()
    geometry_table = pd.DataFrame(
        {
            "contour_id": ["left", "right", "inner"],
            "geometry": [
                box(0.0, 0.0, 10.0, 10.0),
                box(10.0, 0.0, 20.0, 10.0),
                box(2.0, 2.0, 5.0, 5.0),
            ],
            "assigned_structure": ["S1", "S2", "S4"],
            "classification_name": ["Invasive", "DCIS", "Nested"],
            "annotation_source": ["synthetic"] * 3,
            "structure_id": ["1", "2", "4"],
        }
    )
    sdata.shapes["topology_contours"] = geometry_table_to_contour_frame(geometry_table)
    sdata.metadata["contours"]["topology_contours"] = {
        "units": "micron",
        "annotation_source": "synthetic",
    }
    return sdata


def _make_biology_sdata(*, streamed_transcripts: bool = False) -> XeniumSData:
    obs = pd.DataFrame(
        {
            "x_centroid": [2.0, 5.0, 8.0, 22.0, 25.0, 28.0, 2.0, 5.0, 8.0, 22.0, 25.0, 28.0],
            "y_centroid": [2.0, 5.0, 8.0, 2.0, 5.0, 8.0, 22.0, 25.0, 28.0, 22.0, 25.0, 28.0],
            "cell_type": [
                "Tumor",
                "Tumor",
                "Myeloid",
                "Tumor",
                "Tumor",
                "Tumor",
                "Fibroblast",
                "Fibroblast",
                "Endothelial",
                "Fibroblast",
                "Endothelial",
                "Endothelial",
            ],
        },
        index=pd.Index([f"bio_cell_{idx}" for idx in range(12)], name="barcode"),
    )
    expression = np.asarray(
        [
            [10.0, 1.0, 5.0],
            [10.0, 1.0, 5.0],
            [11.0, 2.0, 5.0],
            [12.0, 1.0, 5.0],
            [12.0, 2.0, 5.0],
            [13.0, 2.0, 5.0],
            [1.0, 9.0, 5.0],
            [1.0, 10.0, 5.0],
            [2.0, 10.0, 5.0],
            [2.0, 11.0, 5.0],
            [2.0, 11.0, 5.0],
            [3.0, 12.0, 5.0],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(["GENE_A", "GENE_B", "HOUSE"], name="gene")),
    )
    adata.layers["scaled"] = expression * 2.0

    contour_frame = pd.DataFrame(
        {
            "contour_id": ["case_1"] * 4 + ["case_2"] * 4 + ["ref_1"] * 4 + ["ref_2"] * 4,
            "part_id": [0] * 16,
            "ring_id": [0] * 16,
            "is_hole": [False] * 16,
            "vertex_id": [0, 1, 2, 3] * 4,
            "x": [0.0, 10.0, 10.0, 0.0, 20.0, 30.0, 30.0, 20.0, 0.0, 10.0, 10.0, 0.0, 20.0, 30.0, 30.0, 20.0],
            "y": [0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 20.0, 20.0, 30.0, 30.0],
            "assigned_structure": ["Tumor"] * 8 + ["Stroma"] * 8,
            "classification_name": ["Tumor nests"] * 8 + ["Stromal islands"] * 8,
            "structure_id": ["T"] * 8 + ["S"] * 8,
        }
    )

    transcript_rows = []
    transcript_specs = {
        "case_1": ((2.0, 2.0), "bio_cell_0", ["GENE_A", "GENE_A", "GENE_A", "GENE_B"]),
        "case_2": ((22.0, 2.0), "bio_cell_3", ["GENE_A", "GENE_A", "GENE_A", "GENE_A", "GENE_B"]),
        "ref_1": ((2.0, 22.0), "bio_cell_6", ["GENE_A", "GENE_B", "GENE_B", "GENE_B"]),
        "ref_2": ((22.0, 22.0), "bio_cell_9", ["GENE_A", "GENE_B", "GENE_B", "GENE_B", "GENE_B"]),
    }
    for contour_id, ((x0, y0), cell_id, genes) in transcript_specs.items():
        for offset, gene in enumerate(genes):
            transcript_rows.append(
                {
                    "x": x0 + 0.2 * offset,
                    "y": y0 + 0.2 * offset,
                    "gene_identity": 1 if gene == "GENE_A" else 2,
                    "gene_name": gene,
                    "quality_score": 40.0,
                    "cell_id": cell_id,
                    "contour_hint": contour_id,
                }
            )
    transcripts = pd.DataFrame.from_records(transcript_rows)

    if streamed_transcripts:
        midpoint = len(transcripts) // 2
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
                "cell_id": "string",
                "contour_hint": "string",
            },
            chunk_iter_factory=lambda: iter([chunk_1, chunk_2]),
        )
        return XeniumSData(
            table=adata,
            shapes={"biology_contours": contour_frame},
            point_sources={"transcripts": point_source},
            metadata={"units": "micron", "contours": {"biology_contours": {"units": "micron"}}},
        )

    return XeniumSData(
        table=adata,
        points={"transcripts": transcripts},
        shapes={"biology_contours": contour_frame},
        metadata={"units": "micron", "contours": {"biology_contours": {"units": "micron"}}},
    )


def _xenium_explorer_geojson() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Macrophage Island",
                    "objectType": "annotation",
                    "assigned_structure": "Macrophages",
                    "structure_id": "macrophage",
                    "classification": {
                        "name": "Macrophages",
                        "color": [34, 139, 34],
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [10.0, 0.0],
                            [10.0, 10.0],
                            [0.0, 10.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            }
        ],
    }


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
    assert callable(contour.generate_xenium_explorer_annotations)
    assert callable(contour.ring_density)
    assert callable(contour.smooth_density_by_distance)
    assert callable(contour.summarize_contour_topology)


def test_summarize_contour_topology_wraps_histoseg_tables():
    pytest.importorskip("histoseg.contour")
    sdata = _make_topology_sdata()

    result = summarize_contour_topology(
        sdata,
        contour_key="topology_contours",
        groupby="assigned_structure",
        boundary_tolerance=0.0,
    )

    assert {
        "boundary_overlap",
        "enclosure",
        "contour_summary",
        "group_boundary_overlap",
        "group_enclosure",
    } == set(result)

    boundary = result["boundary_overlap"]
    left_right = boundary.loc[
        boundary[["contour_id_a", "contour_id_b"]].apply(
            lambda row: set(row) == {"left", "right"}, axis=1
        )
    ].iloc[0]
    assert left_right["shared_boundary_length_um"] == pytest.approx(10.0)
    assert left_right["assigned_structure_a"] in {"S1", "S2"}
    assert bool(left_right["is_boundary_neighbor"]) is True

    enclosure = result["enclosure"]
    assert len(enclosure) == 1
    assert enclosure.iloc[0]["outer_contour_id"] == "left"
    assert enclosure.iloc[0]["inner_contour_id"] == "inner"
    assert enclosure.iloc[0]["inner_area_covered_fraction"] == pytest.approx(1.0)

    contour_summary = result["contour_summary"].set_index("contour_id")
    assert contour_summary.loc["left", "n_boundary_neighbors"] == 1
    assert contour_summary.loc["left", "n_contained_contours"] == 1
    assert len(result["group_boundary_overlap"]) >= 1


def test_summarize_contour_topology_respects_contour_query():
    pytest.importorskip("histoseg.contour")
    sdata = _make_topology_sdata()

    result = summarize_contour_topology(
        sdata,
        contour_key="topology_contours",
        contour_query="assigned_structure in ['S1', 'S2']",
        groupby="assigned_structure",
        boundary_tolerance=0.0,
    )

    assert set(result["contour_summary"]["contour_id"]) == {"left", "right"}
    assert result["enclosure"].empty
    assert len(result["boundary_overlap"]) == 1


def test_summarize_contour_topology_missing_histoseg_error(monkeypatch):
    import pyXenium.contour._histoseg as histoseg_loader

    def fake_import_and_validate(module_name, *, required):
        raise ImportError("missing histoseg")

    monkeypatch.delenv("HISTOSEG_ROOT", raising=False)
    monkeypatch.setattr(histoseg_loader, "_import_and_validate", fake_import_and_validate)

    with pytest.raises(ImportError, match="pass `histoseg_root`"):
        summarize_contour_topology(
            _make_topology_sdata(),
            contour_key="topology_contours",
        )


def test_summarize_contour_topology_falls_back_to_histoseg_root(tmp_path, monkeypatch):
    import pyXenium.contour._histoseg as histoseg_loader

    checkout = tmp_path / "HistoSeg"
    package_dir = checkout / "src" / "histoseg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "contour.py").write_text(
        "\n".join(
            [
                "from types import SimpleNamespace",
                "import pandas as pd",
                "",
                "def summarize_contour_topology(contour_table, **kwargs):",
                "    return SimpleNamespace(",
                "        boundary_overlap=pd.DataFrame({'source': ['local_checkout']}),",
                "        enclosure=pd.DataFrame(),",
                "        contour_summary=contour_table[['contour_id']].copy(),",
                "        group_boundary_overlap=pd.DataFrame(),",
                "        group_enclosure=pd.DataFrame(),",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_root = str(checkout / "src")
    old_modules = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name == "histoseg" or name.startswith("histoseg.")
    }
    real_import_and_validate = histoseg_loader._import_and_validate
    attempts = {"count": 0}

    def fake_import_and_validate(module_name, *, required):
        if attempts["count"] == 0:
            attempts["count"] += 1
            raise ImportError("installed histoseg is missing histoseg.contour")
        return real_import_and_validate(module_name, required=required)

    monkeypatch.setattr(histoseg_loader, "_import_and_validate", fake_import_and_validate)
    try:
        result = summarize_contour_topology(
            _make_topology_sdata(),
            contour_key="topology_contours",
            histoseg_root=checkout,
        )
    finally:
        if source_root in sys.path:
            sys.path.remove(source_root)
        for module_name in list(sys.modules):
            if module_name == "histoseg" or module_name.startswith("histoseg."):
                sys.modules.pop(module_name, None)
        for module_name, module in old_modules.items():
            if module is not None:
                sys.modules[module_name] = module

    assert attempts["count"] == 1
    assert result["boundary_overlap"]["source"].tolist() == ["local_checkout"]
    assert set(result["contour_summary"]["contour_id"]) == {"left", "right", "inner"}


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
    assert copied.metadata["contours"]["protein_cluster_contours"]["pixel_size_um_source"] == "he_image"

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


def test_add_contours_from_geojson_uses_default_xenium_pixel_size(tmp_path):
    sdata = XeniumSData(
        table=ad.AnnData(X=np.zeros((0, 0), dtype=float)),
        metadata={"units": "micron"},
    )
    geojson_path = tmp_path / "contours.geojson"
    _write_geojson(geojson_path, _simple_contour_geojson())

    imported = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="default_scaled_contours",
        copy=True,
    )

    frame = imported.shapes["default_scaled_contours"]
    metadata = imported.metadata["contours"]["default_scaled_contours"]
    assert frame["x"].max() == pytest.approx(150.0 * 0.2125)
    assert metadata["pixel_size_um"] == pytest.approx(0.2125)
    assert metadata["pixel_size_um_source"] == "10x_xenium_default"


def test_add_contours_from_geojson_keeps_xenium_um_coordinates_with_default_metadata(tmp_path):
    sdata = XeniumSData(
        table=ad.AnnData(X=np.zeros((0, 0), dtype=float)),
        metadata={"units": "micron"},
    )
    geojson_path = tmp_path / "contours.geojson"
    _write_geojson(geojson_path, _simple_contour_geojson())

    imported = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="micron_contours",
        coordinate_space="xenium_um",
        copy=True,
    )

    frame = imported.shapes["micron_contours"]
    metadata = imported.metadata["contours"]["micron_contours"]
    assert frame["x"].max() == pytest.approx(150.0)
    assert frame["y"].max() == pytest.approx(120.0)
    assert metadata["pixel_size_um"] == pytest.approx(0.2125)
    assert metadata["pixel_size_um_source"] == "10x_xenium_default"


def test_add_contours_from_geojson_prefers_experiment_xenium_pixel_size(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    sdata = read_xenium(str(dataset), as_="sdata", include_images=False, prefer="zarr")
    geojson_path = tmp_path / "contours.geojson"
    _write_geojson(geojson_path, _simple_contour_geojson())

    imported = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="experiment_scaled_contours",
        copy=True,
    )

    metadata = imported.metadata["contours"]["experiment_scaled_contours"]
    assert imported.shapes["experiment_scaled_contours"]["x"].max() == pytest.approx(75.0)
    assert metadata["pixel_size_um"] == pytest.approx(0.5)
    assert metadata["pixel_size_um_source"] == "experiment_xenium"


def test_add_contours_from_geojson_extracts_he_patches_and_roundtrips(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    sdata = read_xenium(str(dataset), as_="sdata", include_images=True, prefer="zarr")
    geojson_path = tmp_path / "contours.geojson"
    _write_geojson(geojson_path, _simple_contour_geojson())

    copied = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="protein_cluster_contours",
        extract_he_patches=True,
        copy=True,
    )

    assert copied is not None
    assert "protein_cluster_contours" not in sdata.contour_images
    assert "protein_cluster_contours" in copied.contour_images
    assert copied.component_summary()["contour_images"] == ["protein_cluster_contours"]
    assert copied.metadata["contours"]["protein_cluster_contours"]["he_patches_enabled"] is True
    assert copied.metadata["contours"]["protein_cluster_contours"]["he_image_key"] == "he"
    assert copied.metadata["contours"]["protein_cluster_contours"]["n_he_patches"] == 2
    assert copied.metadata["contours"]["protein_cluster_contours"]["storage_group"] == (
        "contour_images/protein_cluster_contours"
    )

    patches = copied.contour_images["protein_cluster_contours"]
    assert set(patches) == {"structure4_a", "structure2_b"}

    edge_patch = patches["structure4_a"]
    assert tuple(int(value) for value in edge_patch.levels[0].shape) == (12, 15, 3)
    assert edge_patch.metadata["bbox_image_xy"] == [0, 0, 15, 12]

    multi_patch = patches["structure2_b"]
    assert len(multi_patch.levels) == 1
    assert tuple(int(value) for value in multi_patch.levels[0].shape) == (8, 25, 3)
    assert multi_patch.metadata["source_image_key"] == "he"
    assert multi_patch.metadata["contour_key"] == "protein_cluster_contours"
    assert multi_patch.metadata["contour_id"] == "structure2_b"
    assert multi_patch.metadata["mask_mode"] == "polygon"
    assert multi_patch.metadata["padding_px"] == 0
    assert multi_patch.metadata["bbox_image_xy"] == [45, 31, 70, 39]
    assert multi_patch.metadata["bbox_xenium_um"] == [50.0, 50.0, 75.0, 60.0]
    assert multi_patch.metadata["transform_input_space"] == "patch_local_image_pixel_xy"
    np.testing.assert_allclose(
        np.asarray(multi_patch.image_to_xenium_affine, dtype=float),
        np.asarray([[2.0, 0.0, 100.0], [0.0, 3.0, 98.0], [0.0, 0.0, 1.0]], dtype=float),
    )
    assert not np.allclose(
        np.asarray(multi_patch.image_to_xenium_affine, dtype=float),
        np.asarray(copied.images["he"].image_to_xenium_affine, dtype=float),
    )
    assert np.count_nonzero(np.asarray(multi_patch.levels[0])[:, 10:15, :]) == 0

    add_contours_from_geojson(
        sdata,
        geojson_path,
        key="protein_cluster_contours",
        extract_he_patches=True,
        copy=False,
    )
    assert "protein_cluster_contours" in sdata.contour_images

    expanded = expand_contours(
        copied,
        contour_key="protein_cluster_contours",
        distance=2.0,
        output_key="protein_cluster_contours_expanded",
        copy=True,
    )
    assert expanded is not None
    assert set(expanded.contour_images["protein_cluster_contours"]) == set(patches)

    output = tmp_path / "contour_he_patches.zarr"
    payload = write_xenium(copied, output, format="sdata")
    reloaded = read_sdata(payload["output_path"])

    assert payload["contour_images"] == ["protein_cluster_contours"]
    assert "protein_cluster_contours" in reloaded.contour_images
    reloaded_patch = reloaded.contour_images["protein_cluster_contours"]["structure2_b"]
    assert tuple(int(value) for value in reloaded_patch.levels[0].shape) == (8, 25, 3)
    assert reloaded_patch.metadata["bbox_image_xy"] == [45, 31, 70, 39]
    np.testing.assert_allclose(
        np.asarray(reloaded_patch.image_to_xenium_affine, dtype=float),
        np.asarray(multi_patch.image_to_xenium_affine, dtype=float),
    )


def test_add_contours_from_geojson_extracts_he_patches_with_polygon_holes(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    sdata = read_xenium(str(dataset), as_="sdata", include_images=True, prefer="zarr")
    geojson_path = tmp_path / "hole_contours.geojson"
    _write_geojson(geojson_path, _hole_contour_geojson())

    copied = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="hole_contours",
        extract_he_patches=True,
        copy=True,
    )

    donut_patch = np.asarray(copied.contour_images["hole_contours"]["donut"].levels[0])
    assert tuple(int(value) for value in donut_patch.shape) == (28, 40, 3)
    assert np.count_nonzero(donut_patch[10:20, 12:28, :]) == 0
    assert np.count_nonzero(donut_patch[2:6, 2:6, :]) > 0


def test_add_contours_from_geojson_patch_extraction_requires_loaded_he_image_and_alignment(tmp_path):
    dataset = make_xenium_dataset(tmp_path / "dataset", include_he_image=True)
    geojson_path = tmp_path / "contours.geojson"
    _write_geojson(geojson_path, _simple_contour_geojson())

    sdata_without_images = read_xenium(str(dataset), as_="sdata", include_images=False, prefer="zarr")
    add_contours_from_geojson(
        sdata_without_images,
        geojson_path,
        key="protein_cluster_contours",
        pixel_size_um=0.5,
        extract_he_patches=False,
        copy=False,
    )
    assert "protein_cluster_contours" in sdata_without_images.shapes
    assert "protein_cluster_contours" not in sdata_without_images.contour_images

    with pytest.raises(ValueError, match="include_images=True"):
        add_contours_from_geojson(
            read_xenium(str(dataset), as_="sdata", include_images=False, prefer="zarr"),
            geojson_path,
            key="protein_cluster_contours",
            extract_he_patches=True,
            copy=True,
        )

    sdata_missing_affine = read_xenium(str(dataset), as_="sdata", include_images=True, prefer="zarr")
    sdata_missing_affine.images["he"].image_to_xenium_affine = None
    with pytest.raises(ValueError, match="image_to_xenium_affine"):
        add_contours_from_geojson(
            sdata_missing_affine,
            geojson_path,
            key="protein_cluster_contours",
            extract_he_patches=True,
            copy=True,
        )

    sdata_missing_pixel_size = read_xenium(
        str(dataset),
        as_="sdata",
        include_images=True,
        prefer="zarr",
    )
    sdata_missing_pixel_size.images["he"].pixel_size_um = None
    with pytest.raises(ValueError, match="pixel_size_um"):
        add_contours_from_geojson(
            sdata_missing_pixel_size,
            geojson_path,
            key="protein_cluster_contours",
            extract_he_patches=True,
            copy=True,
        )


def test_add_contours_from_xenium_explorer_geojson_maps_schema_fields(tmp_path):
    sdata = XeniumSData(
        table=ad.AnnData(X=np.zeros((0, 0), dtype=float)),
        metadata={"image_artifacts": {"he": {"pixel_size_um": 1.0}}},
    )
    geojson_path = tmp_path / "xenium_explorer_annotations.generated.geojson"
    _write_geojson(geojson_path, _xenium_explorer_geojson())

    imported = add_contours_from_geojson(
        sdata,
        geojson_path,
        key="atera_contours",
        id_key="name",
        copy=True,
    )

    frame = imported.shapes["atera_contours"]
    assert frame["contour_id"].unique().tolist() == ["Macrophage Island"]
    assert frame["classification_name"].unique().tolist() == ["Macrophages"]
    assert frame["object_type"].unique().tolist() == ["annotation"]
    assert frame["assigned_structure"].unique().tolist() == ["Macrophages"]
    assert frame["structure_id"].unique().tolist() == ["macrophage"]


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


def test_expand_contours_voronoi_supports_multipart_boundaries():
    sdata = _make_complex_contour_sdata()

    expanded = expand_contours(
        sdata,
        contour_key="complex_contours",
        distance=1.0,
        mode="voronoi",
        output_key="complex_voronoi",
        copy=True,
        voronoi_sample_step=1.0,
    )

    assert expanded is not None
    table = _geometry_table_for_key(expanded, "complex_voronoi")
    assert set(table["contour_id"]) == {"donut", "multi"}


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


def test_summarize_contour_composition_counts_cells_and_genes():
    sdata = _make_biology_sdata()

    result = summarize_contour_composition(
        sdata,
        contour_key="biology_contours",
        genes=["GENE_A", "GENE_B"],
        gene_sets={
            "tumor_program": ["GENE_A"],
            "mixed_program": ["GENE_A", "GENE_B"],
        },
    )

    cell_composition = result["cell_composition"]
    case_1_tumor = cell_composition.loc[
        (cell_composition["contour_id"] == "case_1")
        & (cell_composition["cell_type"] == "Tumor")
    ].iloc[0]
    assert case_1_tumor["n_cells"] == 2
    assert case_1_tumor["fraction"] == pytest.approx(2 / 3)
    assert np.allclose(
        cell_composition.groupby("contour_id")["fraction"].sum().to_numpy(dtype=float),
        np.ones(4),
    )

    gene_composition = result["gene_composition"]
    case_1_gene_a = gene_composition.loc[
        (gene_composition["contour_id"] == "case_1")
        & (gene_composition["gene"] == "GENE_A")
    ].iloc[0]
    assert case_1_gene_a["count"] == 3
    assert case_1_gene_a["total_transcripts"] == 4
    assert case_1_gene_a["fraction"] == pytest.approx(0.75)
    assert case_1_gene_a["transcripts_per_cell"] == pytest.approx(1.0)
    assert case_1_gene_a["transcripts_per_um2"] == pytest.approx(3 / 100)

    program_composition = result["program_composition"]
    case_1_program = program_composition.loc[
        (program_composition["contour_id"] == "case_1")
        & (program_composition["program"] == "tumor_program")
    ].iloc[0]
    assert case_1_program["count"] == 3
    assert case_1_program["fraction"] == pytest.approx(0.75)

    summary = result["contour_summary"].set_index("contour_id")
    assert summary.loc["case_1", "n_cells"] == 3
    assert summary.loc["case_1", "n_transcripts"] == 4
    assert summary.loc["case_1", "dominant_cell_type"] == "Tumor"
    assert summary.loc["case_1", "assigned_structure"] == "Tumor"


def test_summarize_contour_composition_matches_streamed_transcripts():
    materialized = summarize_contour_composition(
        _make_biology_sdata(streamed_transcripts=False),
        contour_key="biology_contours",
        genes=["GENE_A", "GENE_B"],
        gene_sets={"tumor_program": ["GENE_A"]},
    )
    streamed = summarize_contour_composition(
        _make_biology_sdata(streamed_transcripts=True),
        contour_key="biology_contours",
        genes=["GENE_A", "GENE_B"],
        gene_sets={"tumor_program": ["GENE_A"]},
    )

    pd.testing.assert_frame_equal(
        materialized["gene_composition"].sort_values(["contour_id", "gene"]).reset_index(drop=True),
        streamed["gene_composition"].sort_values(["contour_id", "gene"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        materialized["program_composition"].sort_values(["contour_id", "program"]).reset_index(drop=True),
        streamed["program_composition"].sort_values(["contour_id", "program"]).reset_index(drop=True),
    )


def test_compare_contour_de_uses_contour_pseudobulk_replicates():
    sdata = _make_biology_sdata()

    result = compare_contour_de(
        sdata,
        contour_key="biology_contours",
        groupby="assigned_structure",
        case="Tumor",
        reference="Stroma",
        genes=["GENE_A", "GENE_B"],
    ).set_index("gene")

    assert result.loc["GENE_A", "status"] == "ok"
    assert result.loc["GENE_A", "n_case_contours"] == 2
    assert result.loc["GENE_A", "n_reference_contours"] == 2
    assert result.loc["GENE_A", "log2fc"] > 0
    assert np.isfinite(result.loc["GENE_A", "p_value"])
    assert np.isfinite(result.loc["GENE_A", "fdr"])
    assert result.loc["GENE_B", "log2fc"] < 0

    insufficient = compare_contour_de(
        sdata,
        contour_key="biology_contours",
        contour_query='contour_id in ["case_1", "ref_1"]',
        groupby="assigned_structure",
        case="Tumor",
        reference="Stroma",
        genes=["GENE_A"],
    )
    assert insufficient["status"].tolist() == ["insufficient_contour_replicates"]
    assert np.isfinite(insufficient["mean_case"].iloc[0])
    assert np.isfinite(insufficient["mean_reference"].iloc[0])
    assert np.isnan(insufficient["p_value"].iloc[0])


def test_compare_contour_transcript_de_uses_direct_transcript_counts():
    sdata = _make_biology_sdata()

    result = compare_contour_transcript_de(
        sdata,
        contour_key="biology_contours",
        groupby="assigned_structure",
        genes=["GENE_A", "GENE_B"],
        comparisons=("global", "one_vs_rest", "pairwise"),
        include_zero_counts=True,
    )

    contour_gene = result["contour_gene"].set_index(["contour_id", "gene"])
    case_1_gene_a = contour_gene.loc[("case_1", "GENE_A")]
    assert case_1_gene_a["count"] == 3
    assert case_1_gene_a["total_transcripts"] == 4
    assert case_1_gene_a["cpm"] == pytest.approx(750_000.0)
    assert case_1_gene_a["density_per_um2"] == pytest.approx(3 / 100)

    one_vs_rest = result["one_vs_rest_de"].set_index(["case", "gene"])
    assert one_vs_rest.loc[("Tumor", "GENE_A"), "status"] == "ok"
    assert one_vs_rest.loc[("Tumor", "GENE_A"), "log2fc_cpm"] > 0
    assert np.isfinite(one_vs_rest.loc[("Tumor", "GENE_A"), "p_value"])
    assert np.isfinite(one_vs_rest.loc[("Tumor", "GENE_A"), "fdr"])
    assert one_vs_rest.loc[("Tumor", "GENE_B"), "log2fc_cpm"] < 0

    global_de = result["global_de"].set_index("gene")
    assert global_de.loc["GENE_A", "status"] == "ok"
    assert global_de.loc["GENE_A", "top_group"] == "Tumor"
    assert np.isfinite(global_de.loc["GENE_A", "p_value"])

    pairwise = result["pairwise_de"].set_index(["case", "reference", "gene"])
    assert pairwise.loc[("Stroma", "Tumor", "GENE_A"), "log2fc_cpm"] < 0


def test_compare_contour_transcript_de_can_discover_all_transcript_genes_without_long_table():
    sdata = _make_biology_sdata()

    result = compare_contour_transcript_de(
        sdata,
        contour_key="biology_contours",
        groupby="assigned_structure",
        genes=None,
        comparisons=("global", "one_vs_rest"),
        return_contour_gene=False,
    )

    assert result["contour_gene"].empty
    assert {"GENE_A", "GENE_B"} == set(result["global_de"]["gene"])
    one_vs_rest = result["one_vs_rest_de"].set_index(["case", "gene"])
    assert one_vs_rest.loc[("Tumor", "GENE_A"), "log2fc_cpm"] > 0
    assert one_vs_rest.loc[("Tumor", "GENE_B"), "log2fc_cpm"] < 0


def test_compare_contour_transcript_de_supports_cell_id_assignment():
    sdata = _make_biology_sdata()

    coordinate_result = compare_contour_transcript_de(
        sdata,
        contour_key="biology_contours",
        groupby="assigned_structure",
        genes=["GENE_A", "GENE_B"],
        comparisons=("one_vs_rest",),
        include_zero_counts=True,
    )
    cell_id_result = compare_contour_transcript_de(
        sdata,
        contour_key="biology_contours",
        groupby="assigned_structure",
        genes=["GENE_A", "GENE_B"],
        assignment="cell_id",
        comparisons=("one_vs_rest",),
        include_zero_counts=True,
    )

    pd.testing.assert_frame_equal(
        coordinate_result["contour_gene"].sort_values(["contour_id", "gene"]).reset_index(drop=True),
        cell_id_result["contour_gene"].sort_values(["contour_id", "gene"]).reset_index(drop=True),
    )
    one_vs_rest = cell_id_result["one_vs_rest_de"].set_index(["case", "gene"])
    assert one_vs_rest.loc[("Tumor", "GENE_A"), "log2fc_cpm"] > 0
    assert one_vs_rest.loc[("Tumor", "GENE_B"), "log2fc_cpm"] < 0


def test_compare_contour_transcript_de_matches_streamed_transcripts():
    materialized = compare_contour_transcript_de(
        _make_biology_sdata(streamed_transcripts=False),
        contour_key="biology_contours",
        groupby="assigned_structure",
        genes=["GENE_A", "GENE_B"],
        comparisons=("one_vs_rest",),
        include_zero_counts=True,
    )
    streamed = compare_contour_transcript_de(
        _make_biology_sdata(streamed_transcripts=True),
        contour_key="biology_contours",
        groupby="assigned_structure",
        genes=["GENE_A", "GENE_B"],
        comparisons=("one_vs_rest",),
        include_zero_counts=True,
    )

    pd.testing.assert_frame_equal(
        materialized["contour_gene"].sort_values(["contour_id", "gene"]).reset_index(drop=True),
        streamed["contour_gene"].sort_values(["contour_id", "gene"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        materialized["one_vs_rest_de"].sort_values(["case", "gene"]).reset_index(drop=True),
        streamed["one_vs_rest_de"].sort_values(["case", "gene"]).reset_index(drop=True),
    )


def test_compare_contour_cell_composition_reports_quantity_and_pvalues():
    sdata = _make_biology_sdata()

    result = compare_contour_cell_composition(
        sdata,
        contour_key="biology_contours",
        groupby="assigned_structure",
    )

    composition = result["composition"]
    case_1_tumor = composition.loc[
        (composition["contour_id"] == "case_1") & (composition["cell_type"] == "Tumor")
    ].iloc[0]
    assert case_1_tumor["assigned_structure"] == "Tumor"
    assert case_1_tumor["n_cells"] == 2
    assert case_1_tumor["fraction"] == pytest.approx(2 / 3)

    group_summary = result["group_summary"].set_index(["assigned_structure", "cell_type"])
    assert group_summary.loc[("Tumor", "Tumor"), "total_count"] == 5
    assert group_summary.loc[("Stroma", "Endothelial"), "mean_fraction"] == pytest.approx(0.5)

    global_stats = result["global_stats"].set_index(["cell_type", "metric"])
    assert global_stats.loc[("Tumor", "fraction"), "status"] == "ok"
    assert np.isfinite(global_stats.loc[("Tumor", "fraction"), "p_value"])

    pairwise_stats = result["pairwise_stats"].set_index(["cell_type", "metric", "case", "reference"])
    assert pairwise_stats.loc[("Tumor", "fraction", "Stroma", "Tumor"), "mean_case"] < pairwise_stats.loc[
        ("Tumor", "fraction", "Stroma", "Tumor"),
        "mean_reference",
    ]


def test_generate_contour_shells_creates_independent_overlapping_shells():
    sdata = _make_nearby_contour_sdata()

    copied = generate_contour_shells(
        sdata,
        contour_key="neighbor_contours",
        inward=2.0,
        outward=3.0,
        step_size=1.0,
        output_key="neighbor_shells",
        copy=True,
    )

    assert copied is not None
    assert "neighbor_shells" not in sdata.shapes
    assert "neighbor_shells" in copied.shapes
    frame = copied.shapes["neighbor_shells"]
    assert frame["source_contour_id"].nunique() == 2
    assert set(frame["shell_direction"]) == {"inward", "outward"}
    assert {"source_contour_id", "shell_id", "shell_start", "shell_end", "shell_mid"}.issubset(frame.columns)
    assert frame.loc[frame["source_contour_id"] == "left", "assigned_structure"].unique().tolist() == ["Left"]

    table = contour_frame_to_geometry_table(frame, contour_key="neighbor_shells")
    left_outward = table.loc[
        (table["source_contour_id"] == "left") & (table["shell_direction"] == "outward"),
        "geometry",
    ].to_list()
    right_outward = table.loc[
        (table["source_contour_id"] == "right") & (table["shell_direction"] == "outward"),
        "geometry",
    ].to_list()
    assert any(left.intersection(right).area > 0 for left in left_outward for right in right_outward)

    metadata = copied.metadata["contours"]["neighbor_shells"]
    assert metadata["generator"] == "generate_contour_shells"
    assert metadata["shell_mode"] == "per_contour"
    assert metadata["step_size_um"] == 1.0


def test_generate_barrier_contour_shells_blocks_other_contour_interiors():
    sdata = _make_nearby_contour_sdata()

    copied = generate_barrier_contour_shells(
        sdata,
        contour_key="neighbor_contours",
        inward=1.0,
        outward=5.0,
        step_size=1.0,
        output_key="neighbor_barrier_shells",
        copy=True,
    )

    assert copied is not None
    assert "neighbor_barrier_shells" not in sdata.shapes
    assert "neighbor_barrier_shells" in copied.shapes

    source_table = _geometry_table_for_key(copied, "neighbor_contours").set_index("contour_id")
    shell_table = _geometry_table_for_key(copied, "neighbor_barrier_shells")
    left_outward = shell_table.loc[
        (shell_table["source_contour_id"] == "left") & (shell_table["shell_direction"] == "outward"),
        "geometry",
    ]
    right_geometry = source_table.loc["right", "geometry"]
    assert left_outward.map(lambda geom: geom.intersection(right_geometry).area).max() < 1e-8

    frame = copied.shapes["neighbor_barrier_shells"]
    assert {"source_contour_id", "shell_start", "shell_end", "shell_mid", "shell_direction"} <= set(frame.columns)
    assert set(frame["shell_direction"]) == {"inward", "outward"}
    assert frame.loc[frame["source_contour_id"] == "left", "assigned_structure"].unique().tolist() == ["Left"]

    metadata = copied.metadata["contours"]["neighbor_barrier_shells"]
    assert metadata["generator"] == "generate_barrier_contour_shells"
    assert metadata["shell_mode"] == "per_contour_barrier"
    assert metadata["barrier_contour_key"] == "neighbor_contours"


def test_contour_biology_validation_paths():
    sdata = _make_biology_sdata()

    no_cell_type = _make_biology_sdata()
    no_cell_type.table.obs = no_cell_type.table.obs.drop(columns=["cell_type"])
    with pytest.raises(KeyError, match="cell_type_key"):
        summarize_contour_composition(no_cell_type, contour_key="biology_contours")

    broken_transcripts = _make_biology_sdata()
    broken_transcripts.points["transcripts"] = broken_transcripts.points["transcripts"].drop(columns=["gene_name"])
    with pytest.raises(ValueError, match="gene_name"):
        summarize_contour_composition(
            broken_transcripts,
            contour_key="biology_contours",
            genes=["GENE_A"],
        )

    with pytest.raises(KeyError, match="MISSING"):
        compare_contour_de(
            sdata,
            contour_key="biology_contours",
            groupby="assigned_structure",
            case="Tumor",
            reference="Stroma",
            genes=["MISSING"],
        )
    with pytest.raises(KeyError, match="missing_group"):
        compare_contour_de(
            sdata,
            contour_key="biology_contours",
            groupby="missing_group",
            case="Tumor",
            reference="Stroma",
            genes=["GENE_A"],
        )
    with pytest.raises(ValueError, match="`step_size` must be greater than 0"):
        generate_contour_shells(
            sdata,
            contour_key="biology_contours",
            inward=1.0,
            outward=1.0,
            step_size=0.0,
        )
    with pytest.raises(ValueError, match="`comparisons`"):
        compare_contour_transcript_de(
            sdata,
            contour_key="biology_contours",
            groupby="assigned_structure",
            comparisons=("unsupported",),
        )
    with pytest.raises(ValueError, match="`assignment`"):
        compare_contour_transcript_de(
            sdata,
            contour_key="biology_contours",
            groupby="assigned_structure",
            assignment="unsupported",
        )
    with pytest.raises(KeyError, match="missing_group"):
        compare_contour_cell_composition(
            sdata,
            contour_key="biology_contours",
            groupby="missing_group",
        )
    with pytest.raises(KeyError, match="missing_barrier"):
        generate_barrier_contour_shells(
            sdata,
            contour_key="biology_contours",
            barrier_contour_key="missing_barrier",
            inward=1.0,
            outward=1.0,
            step_size=1.0,
        )


def test_generate_xenium_explorer_annotations_error_is_actionable(tmp_path, monkeypatch):
    import pyXenium.contour._histoseg as histoseg_loader

    def _missing_histoseg(module_name, *, required):
        raise ModuleNotFoundError(module_name)

    monkeypatch.delenv("HISTOSEG_ROOT", raising=False)
    monkeypatch.setattr(histoseg_loader, "_import_and_validate", _missing_histoseg)

    with pytest.raises(ImportError, match="pass `histoseg_root`"):
        generate_xenium_explorer_annotations(
            tmp_path,
            structures=[{"structure_name": "Macrophages", "cluster_ids": [1]}],
            output_relpath="contour_exports",
        )
