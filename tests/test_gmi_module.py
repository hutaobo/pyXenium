from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyXenium as px
from click.testing import CliRunner
from scipy import sparse
from shapely.geometry import Polygon

from pyXenium.__main__ import app
from pyXenium.contour._geometry import geometry_table_to_contour_frame
from pyXenium.gmi import (
    ContourGmiConfig,
    SpatialGmiConfig,
    assert_vendored_gmi_complete,
    build_contour_gmi_dataset,
    build_gmi_a100_plan,
    build_gmi_install_command,
    compute_contour_heterogeneity,
    get_vendored_gmi_metadata,
    get_vendored_gmi_path,
    run_contour_gmi,
    validate_a100_gmi_path_policy,
)
from pyXenium.io.sdata_model import XeniumSData


def _toy_sdata(*, contours_per_label: int = 2, cells_per_contour: int = 24) -> XeniumSData:
    rng = np.random.default_rng(7)
    contour_rows: list[dict[str, object]] = []
    obs_rows: list[dict[str, object]] = []
    coords: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    labels = ["S1", "S5"]
    cell_index = 0
    for label_index, label in enumerate(labels):
        for contour_index in range(contours_per_label):
            x0 = contour_index * 120.0
            y0 = label_index * 160.0
            contour_id = f"{label}_{contour_index + 1}"
            contour_rows.append(
                {
                    "contour_id": contour_id,
                    "geometry": Polygon([(x0, y0), (x0 + 80, y0), (x0 + 80, y0 + 80), (x0, y0 + 80)]),
                    "assigned_structure": label,
                    "classification_name": label,
                }
            )
            local_xy = np.column_stack(
                [
                    rng.uniform(x0 + 10, x0 + 70, cells_per_contour),
                    rng.uniform(y0 + 10, y0 + 70, cells_per_contour),
                ]
            )
            coords.append(local_xy)
            if label == "S1":
                base = np.tile([14, 2, 8, 1, 3, 2], (cells_per_contour, 1))
                cluster = "CAFs Invasive Associated"
            else:
                base = np.tile([2, 14, 1, 8, 2, 3], (cells_per_contour, 1))
                cluster = "Luminal-like Amorphous DCIS Cells"
            # Add a contour-specific perturbation so within-label heterogeneity is nonzero.
            base[:, 4] += contour_index * 2
            matrices.append(base)
            for _ in range(cells_per_contour):
                obs_rows.append({"cluster": cluster, "joint_cell_state": cluster})
                cell_index += 1

    obs = pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(cell_index)])
    matrix = sparse.csr_matrix(np.vstack(matrices))
    gene_names = ["INV", "DCIS", "ECM", "LUM", "HET", "HOUSE"]
    var = pd.DataFrame({"gene_symbol": gene_names}, index=gene_names)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["spatial"] = np.vstack(coords)
    contour_frame = geometry_table_to_contour_frame(pd.DataFrame(contour_rows))
    return XeniumSData(table=adata, shapes={"s1_s5_contours": contour_frame}, metadata={"sample_id": "toy_breast"})


def test_gmi_is_top_level_canonical_surface():
    from pyXenium import ContourGmiConfig as TopLevelContourGmiConfig
    from pyXenium import build_contour_gmi_dataset as top_level_builder
    from pyXenium import render_contour_gmi_report as top_level_reporter
    from pyXenium import run_atera_breast_contour_gmi as top_level_atera_runner
    from pyXenium import run_contour_gmi as top_level_runner

    assert px.gmi.ContourGmiConfig is ContourGmiConfig
    assert TopLevelContourGmiConfig is ContourGmiConfig
    assert top_level_builder is build_contour_gmi_dataset
    assert top_level_runner is run_contour_gmi
    assert top_level_atera_runner is px.gmi.run_atera_breast_contour_gmi
    assert top_level_reporter is px.gmi.render_contour_gmi_report
    assert SpatialGmiConfig is ContourGmiConfig


def test_vendored_gmi_snapshot_is_local_and_pinned():
    assert_vendored_gmi_complete()
    root = get_vendored_gmi_path()
    metadata = get_vendored_gmi_metadata()

    assert (root / "R" / "Gmi.R").exists()
    assert (root / "src" / "RcppExports.cpp").exists()
    assert metadata["upstream_commit"] == "9df6626494fd2acb2062960496bcbfd062df9752"

    command = " ".join(build_gmi_install_command(vendor_path=root))
    assert "install.packages" in command
    assert "repos=NULL" in command
    assert "https://github.com" not in command.lower()
    assert "install_github" not in command.lower()


def test_gmi_docs_and_package_data_are_canonical():
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    api_index = (repo_root / "docs" / "api" / "index.md").read_text(encoding="utf-8")
    api_gmi = (repo_root / "docs" / "api" / "gmi.md").read_text(encoding="utf-8")
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert "seven canonical public surfaces" in readme
    assert "gmi" in api_index
    assert "pyXenium.gmi" in api_gmi
    assert "_vendor/Gmi/R/*" in pyproject
    assert "_vendor/Gmi/src/*" in pyproject


def test_build_contour_gmi_dataset_labels_filters_and_combines_feature_blocks():
    config = ContourGmiConfig(feature_count=4, spatial_feature_count=6, min_cells_per_contour=10)
    dataset = build_contour_gmi_dataset(_toy_sdata(), config=config)

    assert dataset.X.shape[0] == 4
    assert set(dataset.sample_metadata.loc[dataset.sample_metadata["retained"], "label"]) == {"S1", "S5"}
    assert set(dataset.y.tolist()) == {0, 1}
    assert "tile_id" not in dataset.sample_metadata.columns
    assert dataset.sample_metadata["sample_id"].str.startswith("contour_").all()
    assert "geometry_wkt" in dataset.sample_metadata.columns
    assert {"rna", "spatial"}.issubset(set(dataset.feature_metadata["feature_block"]))
    assert dataset.feature_metadata["feature_index"].tolist() == list(range(1, len(dataset.feature_metadata) + 1))


def test_no_coordinate_spatial_features_excludes_position_context():
    config = ContourGmiConfig(
        feature_count=4,
        spatial_feature_count=12,
        min_cells_per_contour=10,
        exclude_coordinate_spatial_features=True,
    )
    dataset = build_contour_gmi_dataset(_toy_sdata(contours_per_label=4, cells_per_contour=12), config=config)
    spatial_features = dataset.feature_metadata.loc[
        dataset.feature_metadata["feature_block"] == "spatial", "feature"
    ].astype(str)

    forbidden = ("centroid_x", "centroid_y", "slide_x", "slide_y", "x_fraction", "y_fraction")
    assert not any(any(term in feature.lower() for term in forbidden) for feature in spatial_features)


def test_compute_contour_heterogeneity_marks_high_and_low_contours():
    config = ContourGmiConfig(
        feature_count=4,
        spatial_feature_count=4,
        min_cells_per_contour=8,
        heterogeneity_min_contours=6,
    )
    dataset = build_contour_gmi_dataset(_toy_sdata(contours_per_label=6, cells_per_contour=10), config=config)
    heterogeneity = compute_contour_heterogeneity(dataset)

    assert {"heterogeneity_score", "heterogeneity_pc1", "heterogeneity_class"}.issubset(heterogeneity.columns)
    for label in ("S1", "S5"):
        classes = set(heterogeneity.loc[heterogeneity["label"] == label, "heterogeneity_class"])
        assert {"low", "mid", "high"}.issubset(classes)


def test_run_contour_gmi_parses_mocked_r_outputs_and_writes_heterogeneity(monkeypatch, tmp_path):
    dataset = build_contour_gmi_dataset(
        _toy_sdata(contours_per_label=6, cells_per_contour=10),
        config=ContourGmiConfig(
            feature_count=4,
            spatial_feature_count=4,
            min_cells_per_contour=8,
            install_gmi=False,
            heterogeneity_min_contours=6,
        ),
    )

    def fake_run_gmi_fit(*, design_matrix_path, sample_metadata_path, output_dir, config, prediction_matrix_path=None):
        out = Path(output_dir)
        pd.DataFrame({"feature_index": [1], "feature": ["INV"], "coefficient": [1.25]}).to_csv(
            out / "main_effects.tsv", sep="\t", index=False
        )
        pd.DataFrame(
            {
                "interaction": ["X1X2"],
                "feature_index_a": [1],
                "feature_index_b": [2],
                "feature_a": ["INV"],
                "feature_b": ["DCIS"],
                "coefficient": [-0.5],
            }
        ).to_csv(out / "interaction_effects.tsv", sep="\t", index=False)
        meta = pd.read_csv(sample_metadata_path, sep="\t")
        meta = meta.loc[meta["sample_id"].isin(pd.read_csv(design_matrix_path, sep="\t", index_col=0).index)]
        pd.DataFrame({"sample_id": meta["sample_id"], "prediction": meta["y"], "split": "train"}).to_csv(
            out / "predictions.tsv", sep="\t", index=False
        )
        if prediction_matrix_path is not None:
            pred_ids = pd.read_csv(prediction_matrix_path, sep="\t", index_col=0).index.astype(str)
            pd.DataFrame({"sample_id": pred_ids, "prediction": [0.5] * len(pred_ids), "split": "test"}).to_csv(
                out / "predictions_test.tsv", sep="\t", index=False
            )
        (out / "gmi_diagnostics.tsv").write_text("cri_loc\tselected_main\tselected_interactions\n1\t1\t1\n", encoding="utf-8")
        return {"main_effects": str(out / "main_effects.tsv"), "interaction_effects": str(out / "interaction_effects.tsv")}

    monkeypatch.setattr("pyXenium.gmi._workflow.run_gmi_fit", fake_run_gmi_fit)
    result = run_contour_gmi(dataset, output_dir=tmp_path, config=ContourGmiConfig(feature_count=4, spatial_feature_count=4, min_cells_per_contour=8, install_gmi=False))

    assert result.summary["selected_main_effects"] == 1
    assert result.summary["selected_interactions"] == 1
    assert result.summary["within_label_runs_completed"] == 2
    assert result.summary["n_total_endpoint_contours"] == 12
    assert result.summary["n_dropped_contours"] == 0
    assert (tmp_path / "groups.tsv").exists()
    assert (tmp_path / "heterogeneity.tsv").exists()
    assert (tmp_path / "figures" / "s1_s5_contour_overlay.png").exists()
    assert (tmp_path / "figures" / "qc_retained_vs_dropped.png").exists()
    assert (tmp_path / "within_label" / "S1" / "summary.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()


def test_spatial_cv_is_stratified_so_each_fold_has_both_labels(monkeypatch, tmp_path):
    dataset = build_contour_gmi_dataset(
        _toy_sdata(contours_per_label=6, cells_per_contour=10),
        config=ContourGmiConfig(
            feature_count=4,
            spatial_feature_count=4,
            min_cells_per_contour=8,
            install_gmi=False,
        ),
    )

    def fake_run_gmi_fit(*, design_matrix_path, sample_metadata_path, output_dir, config, prediction_matrix_path=None):
        out = Path(output_dir)
        pd.DataFrame(columns=["feature_index", "feature", "coefficient"]).to_csv(
            out / "main_effects.tsv", sep="\t", index=False
        )
        pd.DataFrame(
            columns=["interaction", "feature_index_a", "feature_index_b", "feature_a", "feature_b", "coefficient"]
        ).to_csv(out / "interaction_effects.tsv", sep="\t", index=False)
        train_ids = pd.read_csv(design_matrix_path, sep="\t", index_col=0).index.astype(str)
        pd.DataFrame({"sample_id": train_ids, "prediction": dataset.y.loc[train_ids].to_numpy(), "split": "train"}).to_csv(
            out / "predictions.tsv", sep="\t", index=False
        )
        if prediction_matrix_path is not None:
            pred_ids = pd.read_csv(prediction_matrix_path, sep="\t", index_col=0).index.astype(str)
            pd.DataFrame({"sample_id": pred_ids, "prediction": dataset.y.loc[pred_ids].to_numpy(), "split": "test"}).to_csv(
                out / "predictions_test.tsv", sep="\t", index=False
            )
        (out / "gmi_diagnostics.tsv").write_text("cri_loc\tselected_main\tselected_interactions\n1\t0\t0\n", encoding="utf-8")
        return {"main_effects": str(out / "main_effects.tsv"), "interaction_effects": str(out / "interaction_effects.tsv")}

    monkeypatch.setattr("pyXenium.gmi._workflow.run_gmi_fit", fake_run_gmi_fit)
    result = run_contour_gmi(
        dataset,
        output_dir=tmp_path,
        config=ContourGmiConfig(
            feature_count=4,
            spatial_feature_count=4,
            min_cells_per_contour=8,
            install_gmi=False,
            spatial_cv_folds=5,
            run_within_label_heterogeneity=False,
        ),
    )

    assert len(result.cv_metrics) == 5
    assert result.cv_metrics["n_test_positive"].min() >= 1
    assert result.cv_metrics["n_test_negative"].min() >= 1
    assert result.cv_metrics["auc"].notna().all()


def test_gmi_cli_rejects_removed_tile_option():
    result = CliRunner().invoke(app, ["gmi", "run", "--tile-size", "100", "--output-dir", "unused"])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_gmi_cli_help_uses_canonical_not_experimental_language():
    result = CliRunner().invoke(app, ["gmi", "--help"])

    assert result.exit_code == 0
    assert "Contour-native GMI" in result.output
    assert "Experimental" not in result.output
    assert "experimental" not in result.output


def test_gmi_a100_path_policy_blocks_mnt_output():
    invalid = validate_a100_gmi_path_policy(remote_xenium_root="/mnt/source", remote_root="/mnt/output")
    valid = validate_a100_gmi_path_policy(remote_xenium_root="/mnt/source", remote_root="/data/output")

    assert invalid["valid"] is False
    assert any("must not be under /mnt" in issue for issue in invalid["issues"])
    assert valid["valid"] is True


def test_gmi_a100_plan_cli_writes_json(tmp_path):
    output = tmp_path / "plan.json"
    result = CliRunner().invoke(
        app,
        [
            "gmi",
            "a100-plan",
            "--remote-xenium-root",
            "/mnt/taobo.hu/long/10X_datasets/Xenium/Atera/WTA_Preview_FFPE_Breast_Cancer_outs",
            "--remote-root",
            "/data/taobo.hu/pyxenium_gmi_contour_2026-04",
            "--output-json",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["path_policy"]["valid"] is True
    assert len(payload["stages"]) == 8
    assert all("/mnt/taobo.hu/long" not in stage["output_dir"] for stage in payload["stages"])
    assert "pyxenium_gmi_contour_2026-04" in payload["remote_root"]


def test_build_gmi_a100_plan_contains_contour_stability_stage():
    payload = build_gmi_a100_plan(remote_xenium_root="/mnt/source", remote_root="/data/output")
    commands = "\n".join(stage["command"] for stage in payload["stages"])

    assert "--rna-feature-count 500" in commands
    assert "--spatial-feature-count 100" in commands
    assert "--spatial-cv-folds 5" in commands
    assert "--bootstrap-repeats 10" in commands
    assert "--coordinate-shuffle-control" in commands
    assert "--spatial-feature-shuffle-control" in commands
    assert "--spatial-feature-count 0" in commands
    assert "--rna-feature-count 0" in commands
    assert "--exclude-coordinate-spatial-features" in commands
    assert "--min-cells-per-contour 1" in commands
    assert "--tile-size" not in commands
