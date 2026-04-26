from __future__ import annotations

import json
from pathlib import Path
import gzip

import anndata as ad
import numpy as np
import pandas as pd
from click.testing import CliRunner
from scipy.io import mmwrite
from scipy import sparse

import pyXenium as px
from pyXenium.__main__ import app
from pyXenium.io.sdata_model import XeniumSData
from pyXenium.mechanostress import (
    MechanostressConfig,
    TumorStromaGrowthConfig,
    classify_tumor_stroma_growth,
    compute_ane_density,
    compute_cell_polarity,
    compute_distance_expression_coupling,
    estimate_cell_axes,
    run_mechanostress_cohort,
    run_mechanostress_workflow,
    summarize_axial_orientation,
    validate_suzuki_luad_mechanostress_outputs,
)


def _ellipse_boundary(cell_id: str, center: tuple[float, float], *, angle_deg: float, major: float = 4.0, minor: float = 1.0, n: int = 48) -> pd.DataFrame:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([major * np.cos(theta), minor * np.sin(theta)])
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = points @ rotation.T + np.asarray(center)
    return pd.DataFrame({"cell_id": cell_id, "vertex_id": np.arange(n), "x": xy[:, 0], "y": xy[:, 1]})


def _square_boundary(cell_id: str, center: tuple[float, float], *, size: float = 4.0) -> pd.DataFrame:
    cx, cy = center
    half = size / 2.0
    points = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]
    )
    return pd.DataFrame({"cell_id": cell_id, "vertex_id": np.arange(len(points)), "x": points[:, 0], "y": points[:, 1]})


def _toy_sdata() -> XeniumSData:
    cells = ["s1", "s2", "t1", "t2", "t3"]
    coords = np.array([[0.0, 0.0], [0.0, 2.0], [1.0, 0.5], [2.0, 0.8], [3.0, 0.5]])
    obs = pd.DataFrame(
        {
            "Annotation": ["Stromal", "Stromal", "Tumor", "Tumor", "Tumor"],
            "cluster": ["fibro", "fibro", "tumor", "tumor", "tumor"],
            "x_centroid": coords[:, 0],
            "y_centroid": coords[:, 1],
        },
        index=pd.Index(cells, name="cell_id"),
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix([[0, 0], [0, 1], [5, 0], [3, 0], [1, 0]]),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.obsm["spatial"] = coords
    cell_boundaries = pd.concat([_square_boundary(cell_id, tuple(xy), size=4.0) for cell_id, xy in zip(cells, coords)], ignore_index=True)
    nucleus_boundaries = pd.concat(
        [
            _ellipse_boundary(cell_id, tuple(xy + np.array([0.6, 0.0])), angle_deg=30.0)
            for cell_id, xy in zip(cells, coords)
        ],
        ignore_index=True,
    )
    return XeniumSData(
        table=adata,
        shapes={"cell_boundaries": cell_boundaries, "nucleus_boundaries": nucleus_boundaries},
        metadata={"sample_id": "toy_mechanostress"},
    )


def _write_mechanostress_xenium_sample(sample_dir: Path, sample_id: str = "S1") -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    cells = ["s1", "s2", "t1", "t2", "t3"]
    matrix = sparse.csr_matrix([[0, 0], [0, 1], [5, 0], [3, 0], [1, 0]])
    mex_dir = sample_dir / "cell_feature_matrix"
    mex_dir.mkdir()
    with gzip.open(mex_dir / "matrix.mtx.gz", "wb") as stream:
        mmwrite(stream, matrix.T)
    with gzip.open(mex_dir / "features.tsv.gz", "wt", encoding="utf-8") as stream:
        stream.write("G1\tG1\tGene Expression\nG2\tG2\tGene Expression\n")
    with gzip.open(mex_dir / "barcodes.tsv.gz", "wt", encoding="utf-8") as stream:
        stream.write("\n".join(cells) + "\n")

    coords = np.array([[0.0, 0.0], [0.0, 2.0], [1.0, 0.5], [2.0, 0.8], [3.0, 0.5]])
    pd.DataFrame({"cell_id": cells, "x_centroid": coords[:, 0], "y_centroid": coords[:, 1]}).to_csv(
        sample_dir / "cells.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat([_square_boundary(cell_id, tuple(xy), size=4.0) for cell_id, xy in zip(cells, coords)], ignore_index=True).to_csv(
        sample_dir / "cell_boundaries.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(
        [_ellipse_boundary(cell_id, tuple(xy + np.array([0.6, 0.0])), angle_deg=30.0) for cell_id, xy in zip(cells, coords)],
        ignore_index=True,
    ).to_csv(sample_dir / "nucleus_boundaries.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {
            "cell_id": cells,
            "Annotation": ["Stromal", "Stromal", "Tumor", "Tumor", "Tumor"],
            "cluster": ["fibro", "fibro", "tumor", "tumor", "tumor"],
            "x_centroid": coords[:, 0],
            "y_centroid": coords[:, 1],
        }
    ).to_csv(sample_dir / f"{sample_id}_cell_clusters_with_annotation_and_coord.csv", index=False)
    return sample_dir


def test_mechanostress_public_exports_are_importable():
    assert px.mechanostress.MechanostressConfig is MechanostressConfig
    assert callable(px.run_mechanostress_cohort)
    assert px.MechanostressConfig is MechanostressConfig
    assert callable(px.estimate_cell_axes)
    assert callable(px.compute_ane_density)
    assert callable(px.classify_tumor_stroma_growth)
    assert callable(px.compute_cell_polarity)


def test_estimate_cell_axes_recovers_ellipse_orientation_and_er():
    boundaries = _ellipse_boundary("c1", (0.0, 0.0), angle_deg=30.0, major=5.0, minor=1.0)
    axes = estimate_cell_axes(boundaries)

    assert len(axes) == 1
    assert axes.loc[0, "valid_axis"] is True or bool(axes.loc[0, "valid_axis"])
    assert axes.loc[0, "elongation_ratio"] > 4.5
    assert abs(float(axes.loc[0, "axis_angle_degrees"]) - 30.0) < 1e-6


def test_axial_summary_treats_opposite_directions_as_aligned():
    axes = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "axis_angle_radians": [0.0, np.pi, 0.0],
            "centroid_x": [0.0, 1.0, 2.0],
            "centroid_y": [0.0, 0.0, 0.0],
            "valid_axis": [True, True, True],
        }
    )
    summary = summarize_axial_orientation(axes, local_k=1)

    assert summary.loc[0, "n_axes"] == 3
    assert summary.loc[0, "axial_rbar"] == 1.0
    assert summary.loc[0, "local_rbar_median"] == 1.0


def test_ane_density_excludes_self_and_uses_area_normalization():
    axes = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "axis_angle_radians": [0.0, np.pi],
            "centroid_x": [0.0, 1.0],
            "centroid_y": [0.0, 0.0],
            "valid_axis": [True, True],
        }
    )
    summary, cells = compute_ane_density(axes, radii_um=[2.0], return_cell_metrics=True)

    assert summary.loc[0, "neigh_median"] == 1.0
    assert summary.loc[0, "coh_median"] == 1.0
    assert np.isclose(summary.loc[0, "ANE_density_median"], 1.0 / (np.pi * 4.0))
    assert set(cells["neighbor_count"]) == {1.0}


def test_tumor_stroma_growth_supports_delaunay_and_nearest_ratio():
    frame = pd.DataFrame(
        {
            "cell_id": ["s1", "s2", "t1", "t2", "t3"],
            "Annotation": ["Stromal", "Stromal", "Tumor", "Tumor", "Tumor"],
            "x_centroid": [0.0, 0.0, 1.0, 2.0, 3.0],
            "y_centroid": [0.0, 2.0, 0.5, 0.8, 0.5],
        }
    )
    hop = classify_tumor_stroma_growth(frame, method="delaunay_hop")
    ratio = classify_tumor_stroma_growth(frame, method="nearest_distance_ratio")

    assert "infiltrative" in set(hop["tumor_growth_pattern"])
    assert set(ratio.loc[ratio["Annotation"] == "Tumor", "tumor_growth_pattern"]).issubset({"infiltrative", "expanding"})
    assert ratio["dist_to_nearest_stromal"].notna().sum() == 3


def test_distance_expression_coupling_uses_tumor_enriched_genes_from_adata():
    growth = pd.DataFrame(
        {
            "cell_id": ["t1", "t2", "t3"],
            "Annotation": ["Tumor", "Tumor", "Tumor"],
            "dist_to_nearest_stromal": [1.0, 2.0, 3.0],
        }
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix([[0.0, 5.0], [2.0, 0.0], [4.0, 0.0]]),
        obs=pd.DataFrame(index=["t1", "t2", "t3"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    coupling = compute_distance_expression_coupling(
        growth_table=growth,
        adata=adata,
        tumor_enriched_genes=["G1"],
        min_nonzero_cells=2,
    )

    assert coupling["gene"].tolist() == ["G1"]
    assert bool(coupling.loc[0, "tumor_enriched"]) is True
    assert coupling.loc[0, "spearman_rho"] > 0.99


def test_cell_polarity_threshold_from_boundary_centroids():
    polarity = compute_cell_polarity(
        cell_boundaries=_square_boundary("c1", (0.0, 0.0), size=4.0),
        nucleus_boundaries=_square_boundary("c1", (1.0, 0.0), size=1.0),
        offset_norm_threshold=0.30,
    )

    assert len(polarity) == 1
    assert polarity.loc[0, "offset_distance_um"] == 1.0
    assert bool(polarity.loc[0, "polarized"]) is True


def test_mechanostress_workflow_writes_fixed_artifacts(tmp_path):
    result = run_mechanostress_workflow(
        _toy_sdata(),
        config=MechanostressConfig(coupling_genes=("G1",)),
        output_dir=tmp_path,
    )

    expected = {
        "cell_axes.csv",
        "axis_strength_by_radius.csv",
        "orientation_summary.csv",
        "tumor_growth_cells.csv",
        "tumor_growth_summary.csv",
        "distance_expression_coupling.csv",
        "cell_polarity.csv",
        "polarity_summary.csv",
        "summary.json",
        "report.md",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert (tmp_path / "figures").exists()
    assert result.summary["n_axes"] == 5
    assert not result.distance_expression_coupling.empty


def test_mechanostress_workflow_merges_external_cell_table():
    sdata = _toy_sdata()
    sdata.table.obs = sdata.table.obs.drop(columns=["Annotation", "cluster"])
    cell_table = pd.DataFrame(
        {
            "cell_id": sdata.table.obs_names.astype(str),
            "Annotation": ["Stromal", "Stromal", "Tumor", "Tumor", "Tumor"],
            "cluster": ["fibro", "fibro", "tumor", "tumor", "tumor"],
            "x_centroid": sdata.table.obsm["spatial"][:, 0],
            "y_centroid": sdata.table.obsm["spatial"][:, 1],
        }
    )

    result = run_mechanostress_workflow(
        sdata,
        cell_table=cell_table,
        config=MechanostressConfig(coupling_genes=("G1",)),
    )

    assert not result.tumor_growth_cells.empty
    assert result.tumor_growth_summary.loc[0, "n_tumor"] == 3
    assert not result.distance_expression_coupling.empty


def test_mechanostress_workflow_resolves_coupling_gene_symbols_from_var_name():
    sdata = _toy_sdata()
    sdata.table.var["name"] = ["SYMBOL1", "SYMBOL2"]

    result = run_mechanostress_workflow(
        sdata,
        config=MechanostressConfig(coupling_genes=("SYMBOL1", "MISSING")),
    )

    assert result.distance_expression_coupling["gene"].tolist() == ["SYMBOL1"]


def test_run_mechanostress_cohort_writes_cohort_artifacts(tmp_path):
    cohort_root = tmp_path / "cohort"
    _write_mechanostress_xenium_sample(cohort_root / "S1", sample_id="S1")

    result = run_mechanostress_cohort(
        cohort_root,
        output_dir=tmp_path / "out",
        config=MechanostressConfig(coupling_genes=("G1",)),
        prefer="mex",
    )

    assert result.summary.loc[0, "sample_id"] == "S1"
    assert result.errors.empty
    assert Path(result.files["cohort_summary_csv"]).exists()
    assert Path(result.files["cohort_errors_csv"]).exists()
    assert (tmp_path / "out" / "S1" / "tumor_growth_summary.csv").exists()


def test_mechanostress_tumor_stroma_cli(tmp_path):
    table = tmp_path / "cells.csv"
    pd.DataFrame(
        {
            "cell_id": ["s1", "s2", "t1"],
            "Annotation": ["Stromal", "Stromal", "Tumor"],
            "x_centroid": [0.0, 0.0, 1.0],
            "y_centroid": [0.0, 2.0, 0.5],
        }
    ).to_csv(table, index=False)

    result = CliRunner().invoke(
        app,
        [
            "mechanostress",
            "tumor-stroma-growth",
            str(table),
            "--output-dir",
            str(tmp_path / "out"),
            "--method",
            "nearest_distance_ratio",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert Path(payload["tumor_growth_cells"]).exists()
    assert Path(payload["tumor_growth_summary"]).exists()


def test_mechanostress_run_cohort_cli_uses_public_api(tmp_path):
    cohort_root = tmp_path / "cohort"
    _write_mechanostress_xenium_sample(cohort_root / "S1", sample_id="S1")

    result = CliRunner().invoke(
        app,
        [
            "mechanostress",
            "run-cohort",
            str(cohort_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--prefer",
            "mex",
            "--sample-limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["n_completed"] == 1
    assert payload["n_failed"] == 0
    assert Path(payload["files"]["cohort_summary_csv"]).exists()


def test_suzuki_validation_reads_per_sample_recomputed_outputs(tmp_path):
    strength_root = tmp_path / "strength"
    strength_root.mkdir()
    pd.DataFrame(
        {
            "sample": ["S1"],
            "radius_um": [100.0],
            "ANE_density_median": [0.01],
            "ANE_median": [3.0],
            "coh_median": [0.75],
            "neigh_median": [4.0],
        }
    ).to_csv(strength_root / "ALL_samples_fibro_strength_vs_radius.csv", index=False)
    recomputed = tmp_path / "recomputed" / "S1"
    recomputed.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["S1"],
            "radius_um": [100.0],
            "ANE_density_median": [0.01],
            "ANE_median": [3.0],
            "coh_median": [0.75],
            "neigh_median": [4.0],
        }
    ).to_csv(recomputed / "axis_strength_by_radius.csv", index=False)

    payload = validate_suzuki_luad_mechanostress_outputs(
        xenium_root=tmp_path,
        er_results_root=tmp_path / "er",
        strength_root=strength_root,
        recomputed_dir=tmp_path / "recomputed",
    )

    assert payload["comparisons"]["axis_strength_by_radius"]["within_tolerance"] is True
