from __future__ import annotations

import builtins
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Polygon

from pyXenium.contour._analysis import _prepare_contours
from pyXenium.contour._geometry import geometry_table_to_contour_frame
from pyXenium.io import XeniumImage, XeniumSlide
from pyXenium.multimodal import (
    assign_tiles_to_histoseg_structures,
    histoseg_contours_to_image_table,
    run_histoseg_lazyslide_structure_workflow,
)


def _toy_slide(*, image_scale: float = 1.0) -> XeniumSlide:
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            [
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [0.0, 4.0, 1.0],
                [0.0, 3.0, 2.0],
            ]
        ),
        obs=pd.DataFrame(
            {"cell_type": ["tumor", "tumor", "stroma", "stroma"]},
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["EPCAM", "COL1A1", "CD3D"]),
    )
    adata.obsm["spatial"] = np.asarray(
        [
            [2.0, 2.0],
            [3.0, 3.0],
            [12.0, 2.0],
            [13.0, 3.0],
        ],
        dtype=float,
    )
    adata.uns["sample_id"] = "toy_breast"

    contours = pd.DataFrame(
        {
            "contour_id": ["roi_1", "roi_2"],
            "geometry": [
                Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
                Polygon([(10, 0), (18, 0), (18, 8), (10, 8)]),
            ],
            "assigned_structure": ["S1 ductal tumor", "S5 fibrotic stroma"],
            "structure_id": ["S1", "S5"],
            "classification_name": ["ductal tumor", "fibrotic stroma"],
        }
    )
    contour_frame = geometry_table_to_contour_frame(contours)
    image = XeniumImage(
        levels=[np.zeros((32, 32, 3), dtype=np.uint8)],
        axes="yxc",
        dtype="uint8",
        source_path="toy_he.svs",
        image_to_xenium_affine=[
            [float(image_scale), 0.0, 0.0],
            [0.0, float(image_scale), 0.0],
            [0.0, 0.0, 1.0],
        ],
        pixel_size_um=1.0,
    )
    return XeniumSlide(
        table=adata,
        shapes={"histoseg": contour_frame},
        images={"he": image},
        metadata={"sample_id": "toy_breast"},
    )


def test_histoseg_contours_to_image_table_respects_alignment_scale():
    slide = _toy_slide(image_scale=2.0)
    contour_table = _prepare_contours(
        sdata=slide,
        contour_key="histoseg",
        contour_query=None,
    )

    image_table = histoseg_contours_to_image_table(contour_table, he_image=slide.images["he"])

    first = image_table.loc[image_table["contour_id"] == "roi_1"].iloc[0]
    assert first["geometry"].bounds == pytest.approx((0.0, 0.0, 4.0, 4.0))
    assert first["assigned_structure"] == "S1 ductal tumor"


def test_assign_tiles_to_histoseg_structures_uses_tile_centroids():
    slide = _toy_slide()
    contour_table = _prepare_contours(
        sdata=slide,
        contour_key="histoseg",
        contour_query=None,
    )
    image_table = histoseg_contours_to_image_table(contour_table, he_image=slide.images["he"])
    tiles = pd.DataFrame(
        {
            "tile_id": ["t1", "t2", "t3"],
            "x": [2.0, 12.0, 30.0],
            "y": [2.0, 2.0, 2.0],
        }
    )

    assignments = assign_tiles_to_histoseg_structures(tiles, image_table)

    assert list(assignments["assigned"]) == [True, True, False]
    assert list(assignments["assigned_structure"][:2]) == [
        "S1 ductal tumor",
        "S5 fibrotic stroma",
    ]


def test_run_histoseg_lazyslide_structure_workflow_with_precomputed_tiles(tmp_path):
    slide = _toy_slide()
    tiles = pd.DataFrame(
        {
            "tile_id": ["t1", "t2", "t3", "t4", "t5"],
            "x": [2.0, 3.0, 12.0, 13.0, 30.0],
            "y": [2.0, 3.0, 2.0, 3.0, 30.0],
            "z0": [0.9, 0.8, 0.1, 0.2, 0.0],
            "z1": [0.1, 0.2, 0.8, 0.7, 0.0],
            "ductal epithelium": [0.82, 0.78, 0.10, 0.12, 0.20],
            "fibrotic stroma": [0.15, 0.10, 0.75, 0.80, 0.10],
            "domain": ["tumor", "tumor", "stroma", "stroma", "outside"],
        }
    )

    result = run_histoseg_lazyslide_structure_workflow(
        slide,
        contour_key="histoseg",
        output_dir=tmp_path,
        precomputed_tile_features=tiles,
        include_boundary_programs=False,
    )

    tile_features = result["tile_features"]
    assert int(tile_features["assigned"].sum()) == 4
    assert "embedding__z0" in tile_features.columns
    assert "text_similarity__ductal_epithelium" in tile_features.columns
    assert set(result["structure_image_features"]["assigned_structure"]) == {
        "S1 ductal tumor",
        "S5 fibrotic stroma",
    }
    assert not result["structure_differential_features"].empty
    assert not result["structure_rna_summary"].empty
    assert result["run_manifest"]["outputs"]["n_assigned_tiles"] == 4

    saved = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["model_status"]["backend"] == "precomputed"
    assert (tmp_path / saved["files"]["tile_features"]).exists()
    assert (tmp_path / saved["files"]["structure_image_features"]).exists()


def test_missing_lazyslide_dependency_has_clear_message(monkeypatch):
    slide = _toy_slide()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"geopandas", "lazyslide", "wsidata"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pyXenium\[lazyslide\]"):
        run_histoseg_lazyslide_structure_workflow(
            slide,
            contour_key="histoseg",
            include_boundary_programs=False,
        )
