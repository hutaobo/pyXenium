"""End-to-end smoke test for the four-package integration pipeline.

Simulates the full handoff chain without requiring any real Xenium data or
trained models:

  HistoSeg  → (synthetic GeoJSON + QC JSON)
            ↓ add_contours_from_geojson + import_histoseg_segmentation_qc
  pyXenium  → (import contours, build feature table, export stGPT bundle)
            ↓ export_for_stgpt
  stGPT     → (synthetic cell + contour embeddings)
            ↓ import_stgpt_embeddings
  pyXenium  → (attach embeddings, compare with interpretable programs)
            ↓ compare_programs_with_embeddings + build_spatho_manifest
  SPatho    → (reads case manifest)
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from pyXenium.contour import add_contours_from_geojson, import_histoseg_segmentation_qc
from pyXenium.io import XeniumSData
from pyXenium.multimodal import (
    build_spatho_manifest,
    compare_programs_with_embeddings,
    export_for_stgpt,
    import_stgpt_embeddings,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_sdata() -> XeniumSData:
    n_cells = 8
    n_genes = 5
    rng = np.random.default_rng(42)
    X = sparse.csr_matrix(rng.integers(0, 10, (n_cells, n_genes)).astype(float))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "cell_type": ["tumor"] * 4 + ["immune"] * 4,
                "sample_id": ["smoke_slide"] * n_cells,
                "cell_id": [f"smoke_slide_c{i}" for i in range(1, n_cells + 1)],
            },
            index=[f"c{i}" for i in range(1, n_cells + 1)],
        ),
        var=pd.DataFrame(
            {"feature_id": [f"f_{g}" for g in ("EPCAM", "CD3D", "VIM", "MKI67", "PDCD1")]},
            index=["EPCAM", "CD3D", "VIM", "MKI67", "PDCD1"],
        ),
    )
    adata.obsm["spatial"] = np.array(
        [[i * 10.0, j * 10.0] for i in range(2) for j in range(4)], dtype=float
    )
    adata.uns["sample_id"] = "smoke_slide"
    return XeniumSData(table=adata, shapes={}, metadata={"sample_id": "smoke_slide"})


def _write_histoseg_geojson(path: Path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "polygon_id": f"roi_{i}",
                    "segmentation_source": "synthetic_model",
                    "metadata": {
                        "assigned_structure": "TumorCore" if i % 2 == 0 else "Stroma",
                        "classification_name": "tumor_core" if i % 2 == 0 else "stroma",
                        "annotation_source": "auto",
                        "structure_id": str(i),
                        "name": f"Region {i}",
                        "object_type": "polygon_unit",
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [i * 20.0, 0.0],
                            [(i + 1) * 20.0, 0.0],
                            [(i + 1) * 20.0, 20.0],
                            [i * 20.0, 20.0],
                            [i * 20.0, 0.0],
                        ]
                    ],
                },
            }
            for i in range(4)
        ],
    }
    path.write_text(json.dumps(geojson), encoding="utf-8")


def _write_histoseg_qc(path: Path) -> None:
    qc = {
        "n_structures": 4,
        "segmentation_source": "synthetic_model",
        "model_name": "synthetic_histoseg",
        "mean_iou": 0.87,
        "qc_flags": [],
    }
    path.write_text(json.dumps(qc), encoding="utf-8")


def _write_synthetic_cell_embeddings(path: Path, cell_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(cell_ids)
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "z0": rng.uniform(0, 1, n),
            "z1": rng.uniform(0, 1, n),
            "niche": ["niche_A" if i < n // 2 else "niche_B" for i in range(n)],
            "uncertainty": rng.uniform(0.01, 0.1, n),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _make_program_scores(contour_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(contour_ids)
    scores = rng.dirichlet([1, 1], n)
    top_program = ["immune_exclusion" if s[0] >= s[1] else "emt_invasive_front" for s in scores]
    return pd.DataFrame(
        {
            "contour_id": contour_ids,
            "immune_exclusion": scores[:, 0],
            "emt_invasive_front": scores[:, 1],
            "top_program": top_program,
        }
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_full_pipeline_smoke(tmp_path):
    """Smoke-test the complete HistoSeg → pyXenium → stGPT → pyXenium → SPatho chain."""

    # ── Step 1: HistoSeg produces GeoJSON + QC JSON ──────────────────────
    geojson_path = tmp_path / "histoseg_contours.geojson"
    qc_json_path = tmp_path / "histoseg_qc.json"
    _write_histoseg_geojson(geojson_path)
    _write_histoseg_qc(qc_json_path)

    # ── Step 2: pyXenium imports HistoSeg contours ───────────────────────
    sdata = _make_sdata()
    add_contours_from_geojson(
        sdata,
        geojson_path,
        key="histoseg",
        coordinate_space="xenium_um",
    )
    qc_payload = import_histoseg_segmentation_qc(sdata, qc_json_path, shape_key="histoseg")

    assert "histoseg" in sdata.shapes
    assert qc_payload["n_structures"] == 4
    assert sdata.metadata["contours"]["histoseg"]["histoseg_qc"]["model_name"] == "synthetic_histoseg"

    # ── Step 3: pyXenium exports stGPT handoff bundle ────────────────────
    stgpt_input_dir = tmp_path / "stgpt_input"
    contour_ids = sdata.shapes["histoseg"]["contour_id"].unique().tolist()
    feature_table = {
        "rna_pseudobulk": pd.DataFrame(
            {
                "contour_id": contour_ids,
                "EPCAM": [1.0, 0.5, 0.1, 0.0],
                "CD3D": [0.0, 0.2, 0.8, 1.0],
            }
        ),
    }
    handoff_manifest = export_for_stgpt(
        sdata,
        stgpt_input_dir,
        contour_key="histoseg",
        feature_table=feature_table,
        neighbor_k=3,
    )

    assert handoff_manifest["kind"] == "pyxenium_to_stgpt_handoff"
    assert (stgpt_input_dir / "stgpt_handoff_manifest.json").exists()
    assert (stgpt_input_dir / handoff_manifest["files"]["cell_table"]).exists()
    assert (stgpt_input_dir / handoff_manifest["files"]["contours"]).exists()

    # ── Step 4: stGPT returns cell embeddings (synthetic) ────────────────
    stgpt_output_dir = tmp_path / "stgpt_output"
    stgpt_output_dir.mkdir()
    cell_ids = sdata.table.obs_names.tolist()
    emb_path = stgpt_output_dir / "cell_embeddings.csv"
    _write_synthetic_cell_embeddings(emb_path, cell_ids)

    # ── Step 5: pyXenium imports embeddings ──────────────────────────────
    import_result = import_stgpt_embeddings(emb_path, target=sdata, obsm_key="X_stgpt")

    assert import_result["attached_to_anndata"] is True
    assert sdata.table.obsm["X_stgpt"].shape == (len(cell_ids), 2)
    assert "stgpt_niche" in sdata.table.obs.columns

    # ── Step 6: pyXenium compares interpretable programs with embeddings ─
    program_scores = _make_program_scores(contour_ids)
    embedding_frame = pd.DataFrame(
        {
            "contour_id": contour_ids,
            "z0": [0.9, 0.1, 0.8, 0.2],
            "z1": [0.1, 0.9, 0.2, 0.8],
            "niche": ["niche_A", "niche_B", "niche_A", "niche_B"],
        }
    )
    concordance_result = compare_programs_with_embeddings(program_scores, embedding_frame)

    assert concordance_result["summary"]["n_overlap"] == len(contour_ids)
    assert not concordance_result["correlations"].empty

    # ── Step 7: pyXenium builds SPatho case manifest ─────────────────────
    spatho_manifest_path = tmp_path / "spatho_manifest.json"
    spatho_manifest = build_spatho_manifest(
        spatho_manifest_path,
        sample_id="smoke_slide",
        xenium_path=tmp_path / "xenium",
        histoseg_artifacts={
            "contours": geojson_path,
            "qc_report": qc_json_path,
        },
        pyxenium_artifacts={
            "stgpt_handoff": stgpt_input_dir / "stgpt_handoff_manifest.json",
        },
        stgpt_artifacts={
            "cell_embeddings": emb_path,
        },
        review_targets=[{"kind": "contour", "id": contour_ids[0]}],
    )

    assert spatho_manifest["kind"] == "pyxenium_to_spatho_manifest"
    assert spatho_manifest["sample_id"] == "smoke_slide"
    assert spatho_manifest_path.exists()

    # Verify the saved JSON is a valid SPatho case manifest
    saved = json.loads(spatho_manifest_path.read_text(encoding="utf-8"))
    assert saved["artifacts"]["histoseg"]["contours"].endswith("histoseg_contours.geojson")
    assert saved["artifacts"]["stgpt"]["cell_embeddings"].endswith("cell_embeddings.csv")
    assert saved["review_targets"][0]["kind"] == "contour"
