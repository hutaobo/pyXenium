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
    associate_contour_image_molecular_features,
    assign_tiles_to_histoseg_structures,
    benchmark_contour_molecular_prediction,
    histoseg_contours_to_image_table,
    run_histoseg_lazyslide_structure_workflow,
    summarize_morphomolecular_evidence,
    summarize_wta_pathway_partial_correlations,
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
    assert "top_prompt_term" in tile_features.columns
    assert "top_prompt_similarity" in tile_features.columns
    assert set(tile_features["top_prompt_term"].dropna()) <= {"ductal epithelium", "fibrotic stroma"}
    assert tile_features["top_image_label"].equals(tile_features["top_prompt_term"])
    assert set(result["structure_image_features"]["assigned_structure"]) == {
        "S1 ductal tumor",
        "S5 fibrotic stroma",
    }
    assert not result["structure_differential_features"].empty
    assert not result["structure_rna_summary"].empty
    contour_summary = result["contour_multimodal_summary"]
    assert set(contour_summary["contour_id"]) == {"roi_1", "roi_2"}
    assert "image_heterogeneity__embedding_cosine_similarity_variance" in contour_summary.columns
    assert "text_similarity__ductal_epithelium__mean" in contour_summary.columns
    assert "rna__EPCAM__mean" in contour_summary.columns
    assert "cell_density_per_1e6_image_px2" in contour_summary.columns
    assert "cell_boundary_distance_um__mean" in contour_summary.columns
    assert "tile_boundary_distance_px__mean" in contour_summary.columns
    assert result["run_manifest"]["outputs"]["n_assigned_tiles"] == 4

    saved = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["model_status"]["backend"] == "precomputed"
    assert saved["prompt_metadata"]["prompt_set_name"] == "breast_histology_v1"
    assert saved["prompt_metadata"]["prompt_review_status"] == "not pathologist-confirmed"
    assert (tmp_path / saved["files"]["tile_features"]).exists()
    assert (tmp_path / saved["files"]["contour_multimodal_summary"]).exists()
    assert (tmp_path / saved["files"]["wta_pathway_partial_correlations"]).exists()
    assert (tmp_path / saved["files"]["morphomolecular_hero_targets"]).exists()
    assert (tmp_path / saved["files"]["structure_image_features"]).exists()


def test_contour_partial_association_controls_structure_labels():
    frame = pd.DataFrame(
        {
            "contour_id": [f"c{i}" for i in range(12)],
            "assigned_structure": ["S1", "S2"] * 6,
            "centroid_x": np.arange(12, dtype=float),
            "centroid_y": np.arange(12, dtype=float)[::-1],
            "relative_prompt_axis__necrosis_vs_ductal": [
                0.0,
                0.2,
                0.1,
                0.3,
                0.2,
                0.4,
                0.3,
                0.5,
                0.4,
                0.6,
                0.5,
                0.7,
            ],
            "program__hypoxia__mean": [
                0.0,
                0.4,
                0.2,
                0.6,
                0.4,
                0.8,
                0.6,
                1.0,
                0.8,
                1.2,
                1.0,
                1.4,
            ],
        }
    )

    associations = associate_contour_image_molecular_features(
        frame,
        controls=("assigned_structure",),
        min_contours=6,
    )

    top = associations.iloc[0]
    assert top["image_feature"] == "relative_prompt_axis__necrosis_vs_ductal"
    assert top["molecular_feature"] == "program__hypoxia__mean"
    assert top["partial_spearman_rho"] > 0.95
    assert top["controls"] == "assigned_structure"


def test_contour_prediction_benchmark_reports_added_image_value():
    image_axis = np.linspace(-1.0, 1.0, 20)
    frame = pd.DataFrame(
        {
            "contour_id": [f"c{i}" for i in range(20)],
            "assigned_structure": ["S1", "S2"] * 10,
            "centroid_x": np.arange(20, dtype=float),
            "centroid_y": np.arange(20, dtype=float) % 5,
            "relative_prompt_axis__fibrotic_vs_immune": image_axis,
            "program__stromal_activation__mean": image_axis * 2.0,
        }
    )

    benchmark = benchmark_contour_molecular_prediction(
        frame,
        min_contours=12,
        max_targets=1,
    )

    assert benchmark.iloc[0]["target_feature"] == "program__stromal_activation__mean"
    assert benchmark.iloc[0]["n_image_features"] == 1
    assert benchmark.iloc[0]["r2_structure_image"] > benchmark.iloc[0]["r2_structure_only"]


def test_morphomolecular_evidence_selects_hero_targets_and_contours():
    contour_summary = pd.DataFrame(
        {
            "contour_id": [f"c{i}" for i in range(8)],
            "assigned_structure": ["S1"] * 4 + ["S2"] * 4,
            "centroid_x": np.arange(8, dtype=float),
            "centroid_y": np.arange(8, dtype=float),
            "relative_prompt_axis__necrosis_vs_ductal": [0.1, 0.2, 0.8, 0.9, 0.0, 0.1, 0.7, 0.8],
            "program__hypoxia__mean": [0.0, 0.1, 1.0, 1.1, 0.0, 0.2, 1.2, 1.3],
            "morphology_entropy__top_image_label": [0.1, 0.3, 0.5, 0.8, 0.2, 0.4, 0.6, 0.9],
            "cell_type_diversity__cell_type__shannon": [0.0, 0.2, 0.4, 0.7, 0.1, 0.3, 0.5, 0.8],
        }
    )
    benchmark = pd.DataFrame(
        {
            "target_feature": ["program__hypoxia__mean"],
            "r2_structure_only": [0.05],
            "r2_image_only": [0.65],
            "r2_structure_image": [0.72],
            "delta_r2_combined_over_structure": [0.67],
        }
    )
    associations = pd.DataFrame(
        {
            "image_feature": [
                "relative_prompt_axis__necrosis_vs_ductal",
                "morphology_entropy__top_image_label",
            ],
            "molecular_feature": [
                "program__hypoxia__mean",
                "cell_type_diversity__cell_type__shannon",
            ],
            "partial_spearman_rho": [0.92, 0.81],
            "abs_partial_spearman_rho": [0.92, 0.81],
            "fdr": [0.01, 0.03],
            "n_contours": [8, 8],
        }
    )

    evidence = summarize_morphomolecular_evidence(
        contour_summary,
        prediction_benchmark=benchmark,
        associations=associations,
    )

    assert evidence["hero_targets"].iloc[0]["target_feature"] == "program__hypoxia__mean"
    assert evidence["hero_contours"].iloc[0]["target_feature"] == "program__hypoxia__mean"
    assert evidence["hero_contours"].iloc[0]["hidden_program_score"] > 0
    assert "morphology_entropy_vs_ecology_complexity" in set(evidence["concept_tests"]["concept"])


def test_wta_pathway_partial_correlation_leaderboard_uses_best_axis_per_pathway():
    associations = pd.DataFrame(
        {
            "image_feature": [
                "embedding__103__mean",
                "relative_prompt_axis__necrosis_vs_ductal",
                "embedding__201__mean",
            ],
            "molecular_feature": [
                "program__wta_hypoxia_glycolysis",
                "program__wta_hypoxia_glycolysis",
                "program__wta_t_cell_cytotoxicity",
            ],
            "partial_spearman_rho": [-0.7, 0.4, 0.6],
            "abs_partial_spearman_rho": [0.7, 0.4, 0.6],
            "fdr": [0.001, 0.02, 0.005],
            "n_contours": [100, 100, 95],
            "controls": ["assigned_structure"] * 3,
        }
    )

    leaderboard = summarize_wta_pathway_partial_correlations(associations, max_pathways=2)

    assert list(leaderboard["pathway"]) == ["hypoxia_glycolysis", "t_cell_cytotoxicity"]
    assert leaderboard.iloc[0]["best_image_feature"] == "embedding__103__mean"
    assert leaderboard.iloc[0]["image_axis_family"] == "foundation_embedding_axis"


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
    assert result["model_status"]["text_similarity"]["status"] == "computed"
    assert len(result["tile_features"]) == 2


def test_lazyslide_vision_only_foundation_model_skips_prompt_scoring(monkeypatch):
    slide = _toy_slide()
    contour_table = _prepare_contours(sdata=slide, contour_key="histoseg", contour_query=None)
    image_table = histoseg_contours_to_image_table(contour_table, he_image=slide.images["he"])

    class _FakeWSI:
        def __init__(self):
            self._tables: dict[str, ad.AnnData] = {}
            self.shapes: dict[str, pd.DataFrame] = {}

        def __getitem__(self, key):
            return self._tables[key]

    fake_wsi = _FakeWSI()
    fake_wsi._tables["uni_histoseg_tiles"] = ad.AnnData(np.asarray([[0.1, 0.9]]))
    fake_wsi._tables["uni_histoseg_tiles"].obs_names = pd.Index(["t1"])
    fake_wsi._tables["uni_histoseg_tiles"].var_names = pd.Index(["z0", "z1"])

    monkeypatch.setattr(slide, "inspect_wsi", lambda image_key="he": {"wsi_ready": True, "issues": []})
    monkeypatch.setattr(slide, "to_wsidata", lambda image_key="he": fake_wsi)

    text_calls: list[str] = []

    def _text_embedding(*args, **kwargs):
        text_calls.append("called")
        return []

    fake_geopandas = types.ModuleType("geopandas")
    fake_geopandas.GeoDataFrame = lambda data, geometry, crs=None: pd.DataFrame(data).assign(geometry=list(geometry))

    fake_lazyslide = types.ModuleType("lazyslide")
    fake_lazyslide.__version__ = "test"
    fake_lazyslide.io = types.SimpleNamespace(
        load_annotations=lambda wsi, annotations=None, join_with=None, key_added=None: wsi.shapes.__setitem__(key_added, annotations)
    )
    fake_lazyslide.pp = types.SimpleNamespace(
        tile_tissues=lambda wsi, tile_px, **kwargs: wsi.shapes.__setitem__(
            "histoseg_tiles",
            pd.DataFrame({"tile_id": ["t1"], "x": [2.0], "y": [2.0]}),
        ),
        tile_graph=lambda *args, **kwargs: None,
    )
    fake_lazyslide.tl = types.SimpleNamespace(
        feature_extraction=lambda *args, **kwargs: None,
        text_embedding=_text_embedding,
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
        config=HistoSegLazySlideConfig(model="uni", include_boundary_programs=False),
    )

    assert text_calls == []
    assert result["model_status"]["embedding_model"] == "uni"
    assert result["model_status"]["text_similarity"]["status"] == "skipped"
    assert "vision-only foundation model" in result["model_status"]["text_similarity"]["skipped_reason"]


def test_explicit_mismatched_text_model_is_skipped():
    text_model, reason = histoseg_lazyslide_module._resolve_tile_text_model(
        embedding_model="uni",
        text_model="plip",
        text_terms=["invasive carcinoma"],
    )

    assert text_model is None
    assert "share one latent space" in reason
