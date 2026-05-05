from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from pyXenium.io import XeniumSlide
from pyXenium.multimodal import (
    build_spatho_manifest,
    compare_programs_with_embeddings,
    export_for_stgpt,
    import_stgpt_embeddings,
)


def _toy_sdata() -> XeniumSlide:
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            [
                [1, 0, 3],
                [0, 2, 1],
                [4, 0, 0],
                [0, 1, 5],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame(
            {
                "cell_type": ["tumor", "immune", "tumor", "stromal"],
                "sample_id": ["raw"] * 4,
                "cell_id": ["raw_c1", "raw_c2", "raw_c3", "raw_c4"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame({"feature_id": ["raw_epcam", "raw_cd3d", "raw_vim"]}, index=["EPCAM", "CD3D", "VIM"]),
    )
    adata.obsm["spatial"] = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    adata.uns["sample_id"] = "toy_slide"
    contour_frame = pd.DataFrame(
        {
            "contour_id": ["roi_1", "roi_1", "roi_2", "roi_2"],
            "vertex_id": [0, 1, 0, 1],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "classification_name": ["tumor", "tumor", "stroma", "stroma"],
        }
    )
    return XeniumSlide(table=adata, shapes={"roi": contour_frame}, metadata={"sample_id": "toy_slide"})


def test_export_for_stgpt_writes_contract_bundle(tmp_path):
    sdata = _toy_sdata()
    feature_table = {
        "contour_features": pd.DataFrame(
            {
                "sample_id": ["toy_slide", "toy_slide"],
                "contour_key": ["roi", "roi"],
                "contour_id": ["roi_1", "roi_2"],
                "geometry__area_um2": [10.0, 12.0],
            }
        ),
        "rna_pseudobulk": pd.DataFrame(
            {
                "contour_id": ["roi_1", "roi_2"],
                "EPCAM": [1.5, 0.0],
                "VIM": [0.0, 2.5],
            }
        ),
    }

    manifest = export_for_stgpt(
        sdata,
        tmp_path,
        contour_key="roi",
        feature_table=feature_table,
        neighbor_k=2,
    )

    assert manifest["kind"] == "pyxenium_to_stgpt_handoff"
    assert manifest["sample_id"] == "toy_slide"
    assert manifest["table_format"] == "csv"
    for key in (
        "cell_table",
        "features",
        "coordinates",
        "spatial_edges",
        "rna_matrix",
        "contours",
        "feature_table_contour_features",
        "feature_table_rna_pseudobulk",
    ):
        assert key in manifest["files"]
        assert (tmp_path / manifest["files"][key]).exists()

    saved = json.loads((tmp_path / "stgpt_handoff_manifest.json").read_text(encoding="utf-8"))
    assert saved["package_boundaries"]["stGPT"].startswith("RNA/H&E")
    assert saved["table_format"] == "csv"
    cells = pd.read_csv(tmp_path / manifest["files"]["cell_table"])
    assert list(cells["cell_id"]) == ["c1", "c2", "c3", "c4"]
    assert set(cells["sample_id"]) == {"toy_slide"}
    features = pd.read_csv(tmp_path / manifest["files"]["features"])
    assert list(features["feature_id"]) == ["EPCAM", "CD3D", "VIM"]


def test_export_for_stgpt_parquet_format(tmp_path):
    sdata = _toy_sdata()
    feature_table = {
        "rna_pseudobulk": pd.DataFrame(
            {
                "contour_id": ["roi_1", "roi_2"],
                "EPCAM": [1.5, 0.0],
            }
        ),
    }

    manifest = export_for_stgpt(
        sdata,
        tmp_path,
        contour_key="roi",
        feature_table=feature_table,
        neighbor_k=2,
        table_format="parquet",
    )

    assert manifest["table_format"] == "parquet"
    for key in ("cell_table", "features", "coordinates", "contours"):
        assert key in manifest["files"]
        fname = manifest["files"][key]
        assert fname.endswith(".parquet"), f"{key} should be a parquet file, got {fname}"
        assert (tmp_path / fname).exists()

    saved = json.loads((tmp_path / "stgpt_handoff_manifest.json").read_text(encoding="utf-8"))
    assert saved["table_format"] == "parquet"
    cells = pd.read_parquet(tmp_path / manifest["files"]["cell_table"])
    assert list(cells["cell_id"]) == ["c1", "c2", "c3", "c4"]
    features = pd.read_parquet(tmp_path / manifest["files"]["features"])
    assert list(features["feature_id"]) == ["EPCAM", "CD3D", "VIM"]


def test_import_stgpt_embeddings_attaches_cell_obsm(tmp_path):
    sdata = _toy_sdata()
    path = tmp_path / "cell_embeddings.csv"
    pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "z0": [0.1, 0.2, 0.3, 0.4],
            "z1": [1.0, 0.8, 0.2, 0.1],
            "niche": ["a", "a", "b", "b"],
            "uncertainty": [0.01, 0.02, 0.03, 0.04],
        }
    ).to_csv(path, index=False)

    result = import_stgpt_embeddings(path, target=sdata, obsm_key="X_stgpt")

    assert result["attached_to_anndata"] is True
    assert result["embedding_columns"] == ["z0", "z1"]
    assert "X_stgpt" in sdata.table.obsm
    assert sdata.table.obsm["X_stgpt"].shape == (4, 2)
    assert list(sdata.table.obs["stgpt_niche"]) == ["a", "a", "b", "b"]
    assert list(sdata.table.obs["stgpt_uncertainty"]) == [0.01, 0.02, 0.03, 0.04]


def test_compare_programs_with_embeddings_returns_concordance():
    program_scores = pd.DataFrame(
        {
            "contour_id": ["roi_1", "roi_2", "roi_3", "roi_4"],
            "immune_exclusion": [0.9, 0.8, 0.1, 0.0],
            "emt_invasive_front": [0.0, 0.2, 0.7, 0.9],
            "top_program": [
                "immune_exclusion",
                "immune_exclusion",
                "emt_invasive_front",
                "emt_invasive_front",
            ],
        }
    )
    embeddings = pd.DataFrame(
        {
            "contour_id": ["roi_1", "roi_2", "roi_3", "roi_4"],
            "z0": [1.0, 0.8, 0.2, 0.0],
            "z1": [0.0, 0.2, 0.8, 1.0],
            "niche": ["immune", "immune", "emt", "emt"],
        }
    )

    result = compare_programs_with_embeddings(program_scores, embeddings)

    assert result["summary"]["n_overlap"] == 4
    assert not result["correlations"].empty
    assert set(result["label_summary"]["label"]) == {"immune", "emt"}
    assert set(result["concordance"]["dominant_program"]) == {"immune_exclusion", "emt_invasive_front"}


def test_build_spatho_manifest_writes_package_boundaries(tmp_path):
    output = tmp_path / "spatho_manifest.json"

    manifest = build_spatho_manifest(
        output,
        sample_id="toy_slide",
        xenium_path=tmp_path / "xenium",
        histoseg_artifacts={"contours": tmp_path / "contours.geojson"},
        pyxenium_artifacts={"stgpt_handoff": tmp_path / "stgpt_handoff_manifest.json"},
        stgpt_artifacts={"contour_embeddings": tmp_path / "contour_embeddings.csv"},
        review_targets=[{"kind": "contour", "id": "roi_1"}],
    )

    assert manifest["kind"] == "pyxenium_to_spatho_manifest"
    assert manifest["package_boundaries"]["SPatho"].startswith("Owns pathology")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["artifacts"]["histoseg"]["contours"].endswith("contours.geojson")
