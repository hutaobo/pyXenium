from __future__ import annotations

import builtins
import json
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Polygon

from pyXenium.contour._analysis import _prepare_contours
from pyXenium.contour._geometry import geometry_table_to_contour_frame
from pyXenium.io import XeniumImage, XeniumSlide
import pyXenium.multimodal.histoseg_lazyslide as histoseg_lazyslide_module
from pyXenium.multimodal import (
    HistoSegLazySlideConfig,
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


def test_open_lazyslide_wsi_prefers_internal_slide_store(monkeypatch):
    slide = _toy_slide()
    fake_wsi = types.SimpleNamespace(applied_mpp=None)

    def _set_mpp(value):
        fake_wsi.applied_mpp = float(value)

    fake_wsi.set_mpp = _set_mpp
    monkeypatch.setattr(
        slide,
        "inspect_wsi",
        lambda image_key="he": {"wsi_ready": True, "issues": []},
    )
    monkeypatch.setattr(slide, "to_wsidata", lambda image_key="he": fake_wsi)

    open_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _open_wsi(*args, **kwargs):
        open_calls.append((args, kwargs))
        return object()

    config = HistoSegLazySlideConfig(include_boundary_programs=False, slide_mpp=0.42)
    wsi, source = histoseg_lazyslide_module._open_lazyslide_wsi(
        slide,
        he_image=slide.images["he"],
        config=config,
        open_wsi_func=_open_wsi,
    )

    assert wsi is fake_wsi
    assert open_calls == []
    assert fake_wsi.applied_mpp == pytest.approx(0.42)
    assert source["wsi_source"] == "internal_slide_store"


def test_open_lazyslide_wsi_uses_external_override(monkeypatch):
    slide = _toy_slide()
    fake_wsi = types.SimpleNamespace(applied_mpp=None)

    def _set_mpp(value):
        fake_wsi.applied_mpp = float(value)

    fake_wsi.set_mpp = _set_mpp
    monkeypatch.setattr(
        slide,
        "inspect_wsi",
        lambda image_key="he": {"wsi_ready": False, "issues": ["not used"]},
    )

    open_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _open_wsi(*args, **kwargs):
        open_calls.append((args, kwargs))
        return fake_wsi

    config = HistoSegLazySlideConfig(
        include_boundary_programs=False,
        he_source_path="override.svs",
        wsi_reader="tiffslide",
        slide_mpp=0.31,
    )
    wsi, source = histoseg_lazyslide_module._open_lazyslide_wsi(
        slide,
        he_image=slide.images["he"],
        config=config,
        open_wsi_func=_open_wsi,
    )

    assert wsi is fake_wsi
    assert open_calls == [(("override.svs",), {"reader": "tiffslide"})]
    assert fake_wsi.applied_mpp == pytest.approx(0.31)
    assert source["wsi_source"] == "external_override"


def test_open_lazyslide_wsi_requires_wsi_ready_internal_source(monkeypatch):
    slide = _toy_slide()
    monkeypatch.setattr(
        slide,
        "inspect_wsi",
        lambda image_key="he": {
            "wsi_ready": False,
            "issues": ["Expected at least 2 pyramid levels."],
        },
    )

    with pytest.raises(ValueError, match="not WSI-ready"):
        histoseg_lazyslide_module._open_lazyslide_wsi(
            slide,
            he_image=slide.images["he"],
            config=HistoSegLazySlideConfig(include_boundary_programs=False),
            open_wsi_func=lambda *args, **kwargs: object(),
        )


def test_run_histoseg_lazyslide_structure_workflow_internal_bridge_passes_slide_mpp(monkeypatch):
    slide = _toy_slide()
    contour_table = _prepare_contours(sdata=slide, contour_key="histoseg", contour_query=None)
    image_table = histoseg_contours_to_image_table(contour_table, he_image=slide.images["he"])

    class _FakeWSI:
        def __init__(self):
            self._tables: dict[str, ad.AnnData] = {}
            self.shapes: dict[str, pd.DataFrame] = {}

        def __getitem__(self, key):
            return self._tables[key]

        def set_mpp(self, value):
            self.applied_mpp = float(value)

    fake_wsi = _FakeWSI()
    fake_wsi._tables["plip_histoseg_tiles"] = ad.AnnData(np.asarray([[0.9, 0.1], [0.2, 0.8]]))
    fake_wsi._tables["plip_histoseg_tiles"].obs_names = pd.Index(["t1", "t2"])
    fake_wsi._tables["plip_histoseg_tiles"].var_names = pd.Index(["z0", "z1"])
    fake_wsi._tables["plip_histoseg_tiles"].obs["tile_id"] = ["t1", "t2"]

    monkeypatch.setattr(
        slide,
        "inspect_wsi",
        lambda image_key="he": {"wsi_ready": True, "issues": []},
    )
    monkeypatch.setattr(slide, "to_wsidata", lambda image_key="he": fake_wsi)

    recorded_tile_kwargs: dict[str, object] = {}

    fake_geopandas = types.ModuleType("geopandas")
    fake_geopandas.GeoDataFrame = lambda data, geometry, crs=None: pd.DataFrame(data).assign(geometry=list(geometry))

    fake_lazyslide = types.ModuleType("lazyslide")
    fake_lazyslide.__version__ = "test"
    fake_lazyslide.io = types.SimpleNamespace(
        load_annotations=lambda wsi, annotations=None, join_with=None, key_added=None: wsi.shapes.__setitem__(key_added, annotations)
    )
    fake_lazyslide.pp = types.SimpleNamespace(
        tile_tissues=lambda wsi, tile_px, **kwargs: (
            recorded_tile_kwargs.update({"tile_px": tile_px, **kwargs}),
            wsi.shapes.__setitem__(
                "histoseg_tiles",
                pd.DataFrame(
                    {
                        "tile_id": ["t1", "t2"],
                        "x": [2.0, 12.0],
                        "y": [2.0, 2.0],
                    }
                ),
            ),
        ),
        tile_graph=lambda *args, **kwargs: None,
    )
    fake_lazyslide.tl = types.SimpleNamespace(
        feature_extraction=lambda *args, **kwargs: None,
        text_embedding=lambda *args, **kwargs: [],
        text_image_similarity=lambda *args, **kwargs: None,
        spatial_features=lambda *args, **kwargs: None,
        spatial_domain=lambda *args, **kwargs: None,
    )

    fake_wsidata = types.ModuleType("wsidata")
    fake_wsidata.open_wsi = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("external open_wsi should not be used"))

    monkeypatch.setitem(sys.modules, "geopandas", fake_geopandas)
    monkeypatch.setitem(sys.modules, "lazyslide", fake_lazyslide)
    monkeypatch.setitem(sys.modules, "wsidata", fake_wsidata)

    result = histoseg_lazyslide_module._run_lazyslide_backend(
        sdata=slide,
        image_contours=image_table,
        he_image=slide.images["he"],
        config=HistoSegLazySlideConfig(include_boundary_programs=False, slide_mpp=0.37),
    )

    assert recorded_tile_kwargs["slide_mpp"] == pytest.approx(0.37)
    assert result["model_status"]["wsi_source"] == "internal_slide_store"
    assert len(result["tile_features"]) == 2
