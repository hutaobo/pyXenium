from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from pyXenium.contour import build_contour_feature_table
from pyXenium.contour._geometry import geometry_table_to_contour_frame
from pyXenium.io import XeniumImage, XeniumSData, read_sdata, write_xenium
from pyXenium.multimodal import run_contour_boundary_ecology_pilot, score_contour_boundary_programs


def _square(x0: float, y0: float, size: float):
    return [
        (x0, y0),
        (x0 + size, y0),
        (x0 + size, y0 + size),
        (x0, y0 + size),
    ]


def _paint_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    image[y0:y1, x0:x1, :] = np.asarray(color, dtype=np.uint8)


def _make_boundary_ecology_sdata() -> XeniumSData:
    full_image = np.full((220, 220, 3), 245, dtype=np.uint8)
    contour_specs = {
        "immune_exclusion": {"origin": (0, 0), "label": "Immune exclusion"},
        "myeloid_vascular": {"origin": (120, 0), "label": "Myeloid vascular"},
        "emt_front": {"origin": (0, 120), "label": "EMT front"},
        "tls_adjacent": {"origin": (120, 120), "label": "TLS adjacent"},
    }

    geometry_rows = []
    contour_images: dict[str, XeniumImage] = {}
    all_points: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    protein_rows: list[dict[str, float]] = []
    expression_rows: list[list[float]] = []

    genes = [
        "EPCAM",
        "VIM",
        "ACTA2",
        "CD68",
        "PECAM1",
        "CD3D",
        "CXCL13",
        "MS4A1",
        "SPP1",
        "CD44",
        "VEGFA",
        "KDR",
        "CA9",
        "COL1A1",
        "HLA-DRA",
        "TRAC",
        "MMP11",
        "TGFB1",
        "TGFBR2",
    ]
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    protein_columns = ["PanCK", "Vimentin", "alphaSMA", "CD68", "CD31", "CD3E", "CD20", "PD-1", "PD-L1"]

    def add_cell(
        *,
        cell_id: str,
        x: float,
        y: float,
        state: str,
        niche: str,
        expression_updates: dict[str, float],
        protein_updates: dict[str, float],
    ) -> None:
        cell_rows.append(
            {
                "barcode": cell_id,
                "x_centroid": x,
                "y_centroid": y,
                "joint_cell_state": state,
                "spatial_niche": niche,
            }
        )
        expr = [0.0] * len(genes)
        for gene_name, value in expression_updates.items():
            expr[gene_index[gene_name]] = float(value)
            all_points.append(
                {
                    "x": x,
                    "y": y,
                    "gene_identity": gene_index[gene_name],
                    "gene_name": gene_name,
                    "quality_score": 35.0,
                    "valid": True,
                    "cell_id": cell_id,
                }
            )
        expression_rows.append(expr)
        protein_row = {column: 0.0 for column in protein_columns}
        protein_row.update({key: float(value) for key, value in protein_updates.items()})
        protein_rows.append(protein_row)

    for contour_id, spec in contour_specs.items():
        x0, y0 = spec["origin"]
        size = 60
        contour_color = {
            "immune_exclusion": (225, 165, 180),
            "myeloid_vascular": (220, 175, 170),
            "emt_front": (228, 165, 160),
            "tls_adjacent": (220, 175, 185),
        }[contour_id]
        rim_color = {
            "immune_exclusion": (95, 110, 170),
            "myeloid_vascular": (110, 95, 165),
            "emt_front": (210, 140, 150),
            "tls_adjacent": (90, 100, 180),
        }[contour_id]

        _paint_rect(full_image, x0, y0, x0 + size, y0 + size, contour_color)
        _paint_rect(full_image, max(x0 - 12, 0), y0, x0, y0 + size, rim_color)
        _paint_rect(full_image, x0 + size, y0, min(x0 + size + 12, full_image.shape[1]), y0 + size, rim_color)
        _paint_rect(full_image, x0, max(y0 - 12, 0), x0 + size, y0, rim_color)
        _paint_rect(full_image, x0, y0 + size, x0 + size, min(y0 + size + 12, full_image.shape[0]), rim_color)

        geometry_rows.append(
            {
                "contour_id": contour_id,
                "geometry": Polygon(_square(x0, y0, size)),
                "assigned_structure": spec["label"],
                "classification_name": spec["label"],
                "annotation_source": "unit-test",
                "structure_id": contour_id,
            }
        )

        patch = full_image[y0 : y0 + size, x0 : x0 + size, :].copy()
        contour_images[contour_id] = XeniumImage(
            levels=[patch],
            axes="yxc",
            dtype="uint8",
            source_path=f"{contour_id}.png",
            image_to_xenium_affine=[[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]],
            pixel_size_um=1.0,
            metadata={"contour_id": contour_id},
        )

    add_cell(
        cell_id="immune_inner_1",
        x=20.0,
        y=20.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 8.0, "CD44": 2.0},
        protein_updates={"PanCK": 8.0, "PD-L1": 1.0},
    )
    add_cell(
        cell_id="immune_inner_2",
        x=35.0,
        y=35.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 7.0, "CD44": 1.5},
        protein_updates={"PanCK": 7.0, "PD-L1": 1.0},
    )
    add_cell(
        cell_id="immune_outer_1",
        x=70.0,
        y=18.0,
        state="t_cell_exhausted_cytotoxic",
        niche="immune_rich",
        expression_updates={"CD3D": 8.0, "TRAC": 7.0, "CXCL13": 2.0},
        protein_updates={"CD3E": 6.0, "PD-1": 3.0},
    )
    add_cell(
        cell_id="immune_outer_2",
        x=72.0,
        y=40.0,
        state="b_plasma_like",
        niche="immune_rich",
        expression_updates={"MS4A1": 8.0, "CXCL13": 7.0, "TRAC": 2.0},
        protein_updates={"CD20": 6.0},
    )

    add_cell(
        cell_id="myeloid_inner_1",
        x=140.0,
        y=20.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 8.0, "CD44": 2.0},
        protein_updates={"PanCK": 7.0},
    )
    add_cell(
        cell_id="myeloid_inner_2",
        x=155.0,
        y=35.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 8.0, "TGFB1": 1.0},
        protein_updates={"PanCK": 6.0, "PD-L1": 1.0},
    )
    add_cell(
        cell_id="myeloid_outer_1",
        x=190.0,
        y=18.0,
        state="macrophage_like",
        niche="myeloid_vascular",
        expression_updates={"CD68": 8.0, "SPP1": 7.0, "HLA-DRA": 5.0},
        protein_updates={"CD68": 7.0},
    )
    add_cell(
        cell_id="myeloid_outer_2",
        x=192.0,
        y=40.0,
        state="endothelial_perivascular",
        niche="myeloid_vascular",
        expression_updates={"PECAM1": 8.0, "VEGFA": 7.0, "KDR": 6.0},
        protein_updates={"CD31": 7.0, "alphaSMA": 2.0},
    )

    add_cell(
        cell_id="emt_inner_1",
        x=20.0,
        y=140.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 7.0, "CD44": 2.0},
        protein_updates={"PanCK": 7.0},
    )
    add_cell(
        cell_id="emt_inner_2",
        x=35.0,
        y=155.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 7.0, "TGFB1": 2.0},
        protein_updates={"PanCK": 6.0, "PD-L1": 1.0},
    )
    add_cell(
        cell_id="emt_outer_1",
        x=70.0,
        y=138.0,
        state="emt_like_tumor",
        niche="epithelial_emt_front",
        expression_updates={"VIM": 8.0, "ACTA2": 7.0, "MMP11": 6.0},
        protein_updates={"Vimentin": 7.0, "alphaSMA": 6.0},
    )
    add_cell(
        cell_id="emt_outer_2",
        x=72.0,
        y=160.0,
        state="endothelial_perivascular",
        niche="epithelial_emt_front",
        expression_updates={"COL1A1": 8.0, "ACTA2": 7.0, "TGFBR2": 5.0},
        protein_updates={"alphaSMA": 6.0, "Vimentin": 4.0},
    )

    add_cell(
        cell_id="tls_inner_1",
        x=140.0,
        y=140.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 8.0},
        protein_updates={"PanCK": 7.0},
    )
    add_cell(
        cell_id="tls_inner_2",
        x=155.0,
        y=155.0,
        state="tumor_epithelial",
        niche="mixed_low_signal",
        expression_updates={"EPCAM": 7.0},
        protein_updates={"PanCK": 6.0},
    )
    add_cell(
        cell_id="tls_outer_1",
        x=190.0,
        y=138.0,
        state="b_plasma_like",
        niche="immune_rich",
        expression_updates={"MS4A1": 9.0, "CXCL13": 8.0},
        protein_updates={"CD20": 7.0},
    )
    add_cell(
        cell_id="tls_outer_2",
        x=192.0,
        y=160.0,
        state="t_cell_exhausted_cytotoxic",
        niche="immune_rich",
        expression_updates={"CD3D": 8.0, "TRAC": 7.0, "CXCL13": 5.0},
        protein_updates={"CD3E": 7.0, "PD-1": 4.0},
    )

    obs = pd.DataFrame(cell_rows).set_index("barcode")
    adata = ad.AnnData(
        X=np.asarray(expression_rows, dtype=float),
        obs=obs.copy(),
        var=pd.DataFrame({"name": genes}, index=pd.Index(genes, name="feature")),
    )
    adata.obsm["spatial"] = obs.loc[:, ["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    adata.obsm["protein"] = pd.DataFrame(protein_rows, index=obs.index)

    image = XeniumImage(
        levels=[full_image],
        axes="yxc",
        dtype="uint8",
        source_path="whole_he.png",
        image_to_xenium_affine=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pixel_size_um=1.0,
    )
    shapes = {"tumor_boundary_contours": geometry_table_to_contour_frame(pd.DataFrame(geometry_rows))}
    points = {"transcripts": pd.DataFrame(all_points)}

    return XeniumSData(
        table=adata,
        points=points,
        shapes=shapes,
        images={"he": image},
        contour_images={"tumor_boundary_contours": contour_images},
        metadata={
            "sample_id": "synthetic_boundary_sample",
            "units": "micron",
        },
    )


def test_build_contour_feature_table_zone_semantics_and_roundtrip(tmp_path: Path):
    sdata = _make_boundary_ecology_sdata()
    result = build_contour_feature_table(
        sdata,
        contour_key="tumor_boundary_contours",
        inner_rim_um=20.0,
        outer_rim_um=30.0,
    )

    features = result["contour_features"]
    assert len(features) == 4
    assert {"immune_exclusion", "myeloid_vascular", "emt_front", "tls_adjacent"} == set(features["contour_id"])
    assert "pathomics__whole__stain_blue_ratio" in features.columns
    assert "pathomics__outer_rim__stain_blue_ratio" in features.columns
    assert "edge_contrast__pathomics__stain_blue_ratio" in features.columns
    assert "gradient__immune__outer_minus_inner" in features.columns
    assert (result["zone_summary"]["zone"].isin({"whole", "core", "inner_rim", "outer_rim"})).all()

    payload = write_xenium(sdata, tmp_path / "boundary_workflow.zarr", format="sdata")
    reloaded = read_sdata(payload["output_path"])
    reloaded_result = build_contour_feature_table(
        reloaded,
        contour_key="tumor_boundary_contours",
        inner_rim_um=20.0,
        outer_rim_um=30.0,
    )
    left = features.sort_values("contour_id", kind="stable").reset_index(drop=True)
    right = reloaded_result["contour_features"].sort_values("contour_id", kind="stable").reset_index(drop=True)
    pd.testing.assert_series_equal(left["contour_id"], right["contour_id"])
    np.testing.assert_allclose(
        left["pathomics__whole__mean_r"].to_numpy(dtype=float),
        right["pathomics__whole__mean_r"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        left["gradient__immune__outer_minus_inner"].to_numpy(dtype=float),
        right["gradient__immune__outer_minus_inner"].to_numpy(dtype=float),
    )


def test_score_contour_boundary_programs_identifies_expected_boundary_states():
    sdata = _make_boundary_ecology_sdata()
    feature_table = build_contour_feature_table(
        sdata,
        contour_key="tumor_boundary_contours",
        inner_rim_um=20.0,
        outer_rim_um=30.0,
    )
    scored = score_contour_boundary_programs(
        sdata,
        contour_key="tumor_boundary_contours",
        feature_table=feature_table,
    )["program_scores"].set_index("contour_id")

    assert scored.loc["immune_exclusion", "immune_exclusion"] > scored.loc["immune_exclusion", "emt_invasive_front"]
    assert scored.loc["myeloid_vascular", "myeloid_vascular_belt"] > scored.loc["myeloid_vascular", "immune_exclusion"]
    assert scored.loc["emt_front", "emt_invasive_front"] > scored.loc["emt_front", "myeloid_vascular_belt"]


def test_run_contour_boundary_ecology_pilot_pathomics_only_outputs_discovery_package(tmp_path: Path):
    sdata = _make_boundary_ecology_sdata()
    result = run_contour_boundary_ecology_pilot(
        sdata,
        contour_key="tumor_boundary_contours",
        output_dir=tmp_path / "discovery_package",
    )

    assert set(result) >= {
        "contour_features",
        "program_scores",
        "ecotype_assignments",
        "association_summary",
        "matched_exemplars",
        "edge_gradients",
        "sample_summary",
    }
    assert len(result["contour_features"]) == 4
    assert len(result["ecotype_assignments"]) == 4
    assert "morphology_omics" in result["association_summary"]
    assert (tmp_path / "discovery_package" / "report.md").exists()
    assert (tmp_path / "discovery_package" / "summary.json").exists()
    assert (tmp_path / "discovery_package" / "exemplar_montage.png").exists()
