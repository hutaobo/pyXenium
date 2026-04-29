from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from pyXenium._topology_core import compute_weighted_searcher_findee_distance_matrix_from_df
from pyXenium.io import XeniumImage, XeniumSData
import pyXenium.validation.atera_wta_cervical_end_to_end as cervical_workflow
import pyXenium.validation.sfplot_tbc_bridge as sfplot_bridge


def _write_transcripts_zip(path: Path) -> None:
    store = zarr.storage.ZipStore(str(path), mode="w")
    try:
        root = zarr.open_group(store=store, mode="w")
        root.attrs["gene_names"] = ["Gene1", "Gene2"]
        chunk = root.require_group("grids").require_group("0").require_group("0,0")
        chunk.create_array(
            "location",
            data=np.asarray(
                [
                    [10.0, 20.0],
                    [15.0, 25.0],
                    [30.0, 40.0],
                    [35.0, 45.0],
                ],
                dtype=float,
            ),
        )
        chunk.create_array("gene_identity", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
        chunk.create_array("quality_score", data=np.asarray([30.0, 25.0, 18.0, 35.0], dtype=float))
        chunk.create_array("valid", data=np.asarray([1, 1, 1, 1], dtype=np.uint8))
        chunk.create_array(
            "cell_id",
            data=np.asarray(["cell_1", "cell_2", "cell_3", "cell_1"], dtype=np.str_),
        )
    finally:
        store.close()


def _tiny_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_id": ["cell_1", "cell_2", "cell_3"],
            "cluster": ["Tumor", "Immune", "Tumor"],
        },
        index=pd.Index(["cell_1", "cell_2", "cell_3"], name="cell_id"),
    )
    adata = ad.AnnData(
        X=np.asarray([[1.0, 0.0], [0.0, 2.0], [4.0, 1.0]], dtype=float),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(["Gene1", "Gene2"], name="feature")),
    )
    adata.obsm["spatial"] = np.asarray([[10.0, 20.0], [20.0, 30.0], [40.0, 50.0]], dtype=float)
    return adata


def _tiny_group_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["cell_1", "cell_2", "cell_3"],
            "group": ["Tumor", "Immune", "Tumor"],
        }
    )


def _mock_sdata() -> XeniumSData:
    image = XeniumImage(
        levels=[np.zeros((24, 24, 3), dtype=np.uint8)],
        axes="yxc",
        dtype="uint8",
        source_path="mock_he.png",
        image_to_xenium_affine=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pixel_size_um=0.2125,
    )
    return XeniumSData(
        table=_tiny_adata(),
        images={"he": image},
        metadata={"sample_id": "mock_cervical", "units": "micron"},
    )


def test_run_sfplot_tbc_table_bundle_writes_expected_outputs(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _write_transcripts_zip(dataset_root / "transcripts.zarr.zip")
    adata = _tiny_adata()

    def fake_load_sfplot_public_api(*, sfplot_root=None):
        del sfplot_root

        def fake_distance_matrix(df, *, x_col="x", y_col="y", celltype_col="celltype"):
            return compute_weighted_searcher_findee_distance_matrix_from_df(
                df,
                x_col=x_col,
                y_col=y_col,
                group_col=celltype_col,
                weight_col=None,
            )

        return {
            "load_xenium_table_bundle": lambda folder, normalize=False: adata.copy(),
            "compute_searcher_findee_distance_matrix_from_df": fake_distance_matrix,
            "plot_cophenetic_heatmap": lambda matrix, matrix_name, output_dir, output_filename, sample: Path(
                output_dir, output_filename
            ).write_bytes(b"%PDF-1.4\n% smoke\n"),
        }

    monkeypatch.setattr(sfplot_bridge, "_load_sfplot_public_api", fake_load_sfplot_public_api)

    result = sfplot_bridge.run_sfplot_tbc_table_bundle(
        dataset_root,
        sample_name="tiny_cervical",
        output_folder=tmp_path / "sfplot_results",
        df=_tiny_group_df(),
        n_jobs=1,
        gene_batch_size=1,
    )

    structure_map_path = Path(result["structure_map_table"])
    t_and_c_path = Path(result["t_and_c_result"])
    pdf_path = Path(result["structure_map_pdf"])

    assert structure_map_path.exists()
    assert t_and_c_path.exists()
    assert pdf_path.exists()

    structure_map = pd.read_csv(structure_map_path, index_col=0)
    t_and_c = pd.read_csv(t_and_c_path, index_col=0)

    assert list(structure_map.columns) == ["Immune", "Tumor"]
    assert list(t_and_c.index) == ["Gene1", "Gene2"]
    assert set(t_and_c.columns) == {"Immune", "Tumor"}


def test_build_atera_wta_cervical_bio6_structures_returns_fixed_six_panels():
    structures = cervical_workflow.build_atera_wta_cervical_bio6_structures()

    assert [entry["structure_name"] for entry in structures] == [
        "Tumor",
        "T-cell",
        "Myeloid",
        "B/Plasma",
        "Stromal/Fibro/Muscle",
        "Vascular/Endocervical",
    ]
    assert len(structures) == 6
    assert "OR4F17+ Cells" in structures[0]["cluster_ids"]
    assert all("structure_color" not in entry for entry in structures)


def test_run_atera_wta_cervical_end_to_end_orchestrates_fixed_outputs(tmp_path, monkeypatch):
    dataset_root = tmp_path / "cervical_dataset"
    dataset_root.mkdir()
    output_root = dataset_root / "pyxenium_cervical_end_to_end"

    all_groups = [
        group_name
        for structure in cervical_workflow.build_atera_wta_cervical_bio6_structures()
        for group_name in structure["cluster_ids"]
    ]
    pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(len(all_groups))],
            "group": all_groups,
            "cluster": all_groups,
            "color": ["#cccccc"] * len(all_groups),
        }
    ).to_csv(dataset_root / cervical_workflow.DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS, index=False)

    captured_structures: list[dict[str, object]] = []

    def fake_run_sfplot_tbc_table_bundle(
        folder,
        *,
        sample_name=None,
        output_folder=None,
        coph_method="average",
        n_jobs=8,
        maxtasks=50,
        df=None,
        gene_batch_size=128,
        sfplot_root=None,
    ):
        del folder, sample_name, coph_method, n_jobs, maxtasks, df, gene_batch_size, sfplot_root
        result_dir = Path(output_folder)
        result_dir.mkdir(parents=True, exist_ok=True)
        structure_map_path = result_dir / "StructureMap_table_mock.csv"
        t_and_c_path = result_dir / "t_and_c_result_mock.csv"
        pdf_path = result_dir / "StructureMap_of_mock.pdf"
        pd.DataFrame([[0.0]], index=["Tumor"], columns=["Tumor"]).to_csv(structure_map_path)
        pd.DataFrame([[0.1]], index=["SPP1"], columns=["Tumor"]).to_csv(t_and_c_path)
        pdf_path.write_bytes(b"%PDF-1.4\n% mock\n")
        return {
            "output_dir": str(result_dir),
            "structure_map_pdf": str(pdf_path),
            "structure_map_table": str(structure_map_path),
            "t_and_c_result": str(t_and_c_path),
        }

    def fake_read_xenium(path, *, as_, **kwargs):
        del path, kwargs
        if as_ == "anndata":
            return _tiny_adata()
        if as_ == "sdata":
            return _mock_sdata()
        raise AssertionError(f"Unexpected read_xenium mode: {as_}")

    def fake_cci(*, output_dir=None, **kwargs):
        del kwargs
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_path = out_dir / "cci_sender_receiver_scores.csv"
        scores = pd.DataFrame(
            [
                {
                    "ligand": "SPP1",
                    "receptor": "CD44",
                    "sender_celltype": "Tumor",
                    "receiver_celltype": "Myeloid",
                    "CCI_score": 0.85,
                }
            ]
        )
        scores.to_csv(scores_path, index=False)
        return {"scores": scores, "files": {"cci_sender_receiver_scores": str(scores_path)}}

    def fake_pathway(*, output_dir=None, **kwargs):
        del kwargs
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pathway_path = out_dir / "pathway_to_cell.csv"
        pathway_table = pd.DataFrame([[0.1]], index=["immune_activation"], columns=["Tumor"])
        pathway_table.to_csv(pathway_path)
        return {
            "pathway_to_cell": pathway_table,
            "pathway_activity_to_cell": pathway_table.copy(),
            "files": {"pathway_to_cell": str(pathway_path)},
        }

    def fake_generate_xenium_explorer_annotations(
        dataset_root_arg,
        *,
        structures,
        output_relpath,
        clusters_relpath,
        histoseg_root,
        barcode_col,
        cluster_col,
        **kwargs,
    ):
        del dataset_root_arg, clusters_relpath, histoseg_root, barcode_col, cluster_col, kwargs
        captured_structures[:] = list(structures)
        out_dir = Path(output_relpath)
        out_dir.mkdir(parents=True, exist_ok=True)
        geojson_path = out_dir / "xenium_explorer_annotations.geojson"
        geojson_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [],
                }
            ),
            encoding="utf-8",
        )
        csv_path = out_dir / "xenium_explorer_annotations.csv"
        summary_path = out_dir / "xenium_explorer_annotations_summary.csv"
        partition_path = out_dir / "partition_table.csv"
        structure_count_path = out_dir / "structure_count.csv"
        metrics_path = out_dir / "metrics.json"
        preview_path = out_dir / "preview.png"
        for path in (csv_path, summary_path, partition_path, structure_count_path):
            path.write_text("placeholder\n", encoding="utf-8")
        metrics_path.write_text("{}\n", encoding="utf-8")
        preview_path.write_bytes(b"png")
        return {
            "out_dir": str(out_dir),
            "geojson": str(geojson_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
            "preview_png": str(preview_path),
            "partition_table": str(partition_path),
            "structure_count_csv": str(structure_count_path),
            "metrics_json": str(metrics_path),
        }

    def fake_add_contours_from_geojson(
        sdata,
        geojson_path,
        *,
        key,
        id_key="polygon_id",
        coordinate_space="xenium_pixel",
        pixel_size_um=None,
        extract_he_patches=False,
        he_image_key="he",
        copy=False,
    ):
        del geojson_path, id_key, coordinate_space, pixel_size_um, extract_he_patches, he_image_key, copy
        sdata.shapes[key] = pd.DataFrame(
            {
                "contour_id": ["S1 Tumor", "S2 T-cell"],
                "assigned_structure": ["Tumor", "T-cell"],
            }
        )

    def fake_expand_contours(
        sdata,
        *,
        contour_key,
        distance,
        mode="overlap",
        output_key=None,
        copy=False,
        voronoi_sample_step=None,
    ):
        del distance, mode, copy, voronoi_sample_step
        sdata.shapes[output_key] = sdata.shapes[contour_key].copy()

    def fake_ring_density(
        sdata,
        *,
        contour_key,
        target="transcripts",
        contour_query=None,
        target_query=None,
        feature_key="gene_name",
        feature_values=None,
        inward=0.0,
        outward=0.0,
        ring_width=1.0,
    ):
        del sdata, contour_key, target, contour_query, target_query, feature_key, feature_values, inward, outward, ring_width
        return pd.DataFrame(
            {
                "contour_id": ["S1 Tumor"],
                "feature_values": ["SPP1"],
                "ring_start_um": [-20.0],
                "ring_end_um": [-15.0],
                "density": [1.25],
            }
        )

    def fake_smooth_density_by_distance(
        sdata,
        *,
        contour_key,
        target="transcripts",
        contour_query=None,
        target_query=None,
        feature_key="gene_name",
        feature_values=None,
        inward=0.0,
        outward=0.0,
        bandwidth=1.0,
        grid_step=None,
    ):
        del sdata, contour_key, target, contour_query, target_query, feature_key, feature_values, inward, outward, bandwidth, grid_step
        return pd.DataFrame(
            {
                "contour_id": ["S1 Tumor"],
                "feature_values": ["SPP1"],
                "signed_distance_um": [0.0],
                "density": [0.75],
            }
        )

    def fake_run_contour_boundary_ecology_pilot(sdata_or_path, *, contour_key, output_dir=None, embedding_backend=None, neighbor_k=6):
        del sdata_or_path, contour_key, embedding_backend, neighbor_k
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        (out_dir / "report.md").write_text("# mock\n", encoding="utf-8")
        (out_dir / "exemplar_montage.png").write_bytes(b"png")
        (out_dir / "contour_features.csv").write_text("placeholder\n", encoding="utf-8")
        (out_dir / "program_scores.csv").write_text("placeholder\n", encoding="utf-8")
        return {
            "sample_summary": {
                "sample_id": "mock_cervical",
                "contour_key": cervical_workflow.DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
                "n_contours": 2,
                "n_ecotypes": 2,
            },
            "artifact_dir": str(out_dir),
        }

    def fake_write_xenium(obj, path, *, format="h5ad", overwrite=False):
        del obj, format, overwrite
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"output_path": str(target), "format": "sdata"}

    monkeypatch.setattr(cervical_workflow, "run_sfplot_tbc_table_bundle", fake_run_sfplot_tbc_table_bundle)
    monkeypatch.setattr(cervical_workflow, "read_xenium", fake_read_xenium)
    monkeypatch.setattr(cervical_workflow, "cci_topology_analysis", fake_cci)
    monkeypatch.setattr(cervical_workflow, "pathway_topology_analysis", fake_pathway)
    monkeypatch.setattr(
        cervical_workflow,
        "generate_xenium_explorer_annotations",
        fake_generate_xenium_explorer_annotations,
    )
    monkeypatch.setattr(cervical_workflow, "add_contours_from_geojson", fake_add_contours_from_geojson)
    monkeypatch.setattr(cervical_workflow, "expand_contours", fake_expand_contours)
    monkeypatch.setattr(cervical_workflow, "ring_density", fake_ring_density)
    monkeypatch.setattr(cervical_workflow, "smooth_density_by_distance", fake_smooth_density_by_distance)
    monkeypatch.setattr(cervical_workflow, "run_contour_boundary_ecology_pilot", fake_run_contour_boundary_ecology_pilot)
    monkeypatch.setattr(cervical_workflow, "write_xenium", fake_write_xenium)

    result = cervical_workflow.run_atera_wta_cervical_end_to_end(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        export_figures=False,
    )

    assert len(captured_structures) == 6
    assert [entry["structure_name"] for entry in captured_structures] == [
        "Tumor",
        "T-cell",
        "Myeloid",
        "B/Plasma",
        "Stromal/Fibro/Muscle",
        "Vascular/Endocervical",
    ]
    assert all("structure_color" not in entry for entry in captured_structures)

    assert result["payload"]["contour_structure_count"] == 6
    assert result["payload"]["contour_key"] == cervical_workflow.DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY
    assert (
        result["payload"]["expanded_contour_key"]
        == cervical_workflow.DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY
    )
    assert result["payload"]["ring_density_summary"]["features"] == ["SPP1"]

    assert Path(result["files"]["summary_json"]).exists()
    assert Path(result["files"]["report_md"]).exists()
    assert Path(result["files"]["ring_density_csv"]).exists()
    assert Path(result["files"]["smooth_density_csv"]).exists()
    assert Path(result["files"]["multimodal_report_md"]).exists()
    assert Path(result["files"]["contour_enriched_sdata"]).exists()

    assert cervical_workflow.DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY in result["sdata"].shapes
    assert cervical_workflow.DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY in result["sdata"].shapes
