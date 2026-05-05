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
    ContourGmiDataset,
    GmiModuleConfig,
    SpatialGmiConfig,
    assert_vendored_gmi_complete,
    build_contour_gmi_dataset,
    build_gmi_effect_graph,
    build_gmi_a100_plan,
    build_gmi_install_command,
    compute_contour_heterogeneity,
    discover_gmi_modules,
    get_vendored_gmi_metadata,
    get_vendored_gmi_path,
    run_contour_gmi,
    validate_a100_gmi_path_policy,
    write_contour_gmi_dataset,
)
from pyXenium.io.slide_model import XeniumSlide


def _toy_sdata(*, contours_per_label: int = 2, cells_per_contour: int = 24) -> XeniumSlide:
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
    return XeniumSlide(table=adata, shapes={"s1_s5_contours": contour_frame}, metadata={"sample_id": "toy_breast"})


def _toy_module_dataset() -> ContourGmiDataset:
    sample_ids = [f"contour_S1_{i}" for i in range(6)] + [f"contour_S5_{i}" for i in range(6)]
    y = pd.Series([1] * 6 + [0] * 6, index=sample_ids, name="y")
    rng = np.random.default_rng(11)
    nib = np.r_[np.linspace(-1.8, -1.1, 6), np.linspace(1.1, 1.9, 6)]
    sorl = nib * 0.95 + rng.normal(0, 0.04, 12)
    ccnd1 = np.r_[np.linspace(1.0, 1.5, 6), np.linspace(-0.6, -0.2, 6)]
    ecm = np.r_[np.linspace(0.8, 1.2, 6), np.linspace(-0.5, -0.1, 6)]
    lum_comp = nib * 0.8 + rng.normal(0, 0.05, 12)
    rim = np.r_[np.linspace(0.6, 1.0, 6), np.linspace(-0.2, 0.2, 6)]
    X = pd.DataFrame(
        {
            "NIBAN1": nib,
            "SORL1": sorl,
            "CCND1": ccnd1,
            "COL1A1": ecm,
            "omics__Luminal_like_Amorphous_DCIS_fraction": lum_comp,
            "edge_contrast__outer_inner_luminal": rim,
        },
        index=sample_ids,
    )
    sample_metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "contour_id": [item.replace("contour_", "") for item in sample_ids],
            "label": ["S1"] * 6 + ["S5"] * 6,
            "y": y.to_numpy(),
            "n_cells": [40] * 12,
            "library_size": [1000.0] * 12,
            "x_centroid": np.r_[np.arange(6) * 20.0, np.arange(6) * 20.0],
            "y_centroid": np.r_[np.zeros(6), np.ones(6) * 200.0],
            "retained": [True] * 12,
            "drop_reason": ["retained"] * 12,
        }
    )
    feature_metadata = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_block": ["rna", "rna", "rna", "rna", "spatial", "spatial"],
            "feature_group": ["rna", "rna", "rna", "rna", "composition", "edge_contrast"],
        }
    )
    return ContourGmiDataset(
        X=X,
        y=y,
        sample_metadata=sample_metadata,
        feature_metadata=feature_metadata,
        config={},
        provenance={"sample_id": "toy_modules"},
    )


def _write_toy_module_gmi_run(output_dir: Path) -> ContourGmiDataset:
    dataset = _toy_module_dataset()
    write_contour_gmi_dataset(dataset, output_dir)
    pd.DataFrame(
        {
            "feature_index": [1, 2],
            "feature": ["NIBAN1", "SORL1"],
            "coefficient": [-1.4, -1.1],
        }
    ).to_csv(output_dir / "main_effects.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "interaction": ["X1X2"],
            "feature_index_a": [1],
            "feature_index_b": [2],
            "feature_a": ["NIBAN1"],
            "feature_b": ["SORL1"],
            "coefficient": [0.4],
        }
    ).to_csv(output_dir / "interaction_effects.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "effect_type": ["main", "main"],
            "member": ["NIBAN1", "SORL1"],
            "selection_count": [7, 5],
            "selection_frequency": [0.7, 0.5],
        }
    ).to_csv(output_dir / "stability.tsv", sep="\t", index=False)
    return dataset


def test_gmi_is_top_level_canonical_surface():
    from pyXenium import ContourGmiConfig as TopLevelContourGmiConfig
    from pyXenium import GmiModuleConfig as TopLevelGmiModuleConfig
    from pyXenium import build_contour_gmi_dataset as top_level_builder
    from pyXenium import discover_gmi_modules as top_level_module_discovery
    from pyXenium import render_contour_gmi_report as top_level_reporter
    from pyXenium import run_atera_breast_contour_gmi as top_level_atera_runner
    from pyXenium import run_contour_gmi as top_level_runner

    assert px.gmi.ContourGmiConfig is ContourGmiConfig
    assert TopLevelContourGmiConfig is ContourGmiConfig
    assert TopLevelGmiModuleConfig is GmiModuleConfig
    assert top_level_builder is build_contour_gmi_dataset
    assert top_level_module_discovery is discover_gmi_modules
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

    assert "nine feature areas" in readme
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


def test_discover_gmi_modules_writes_supervised_spatial_module_artifacts(tmp_path):
    gmi_dir = tmp_path / "gmi_run"
    gmi_dir.mkdir()
    _write_toy_module_gmi_run(gmi_dir)

    result = discover_gmi_modules(
        gmi_output_dir=gmi_dir,
        config=GmiModuleConfig(write_figures=False, expansion_correlation=0.5, expansion_spatial_lag_correlation=0.4),
    )

    assert result.summary["n_modules"] == 1
    assert result.module_scores.shape == (12, 1)
    assert {"spatial_modules.tsv", "module_features.tsv", "module_scores.tsv.gz"}.issubset(
        {Path(path).name for path in result.files.values()}
    )
    module = result.spatial_modules.iloc[0]
    assert module["direction_label"] == "S5"
    assert {"NIBAN1", "SORL1"}.issubset(set(result.module_features["feature"]))
    enriched = result.module_enrichment.loc[result.module_enrichment["gene_set"] == "DCIS_apocrine_luminal"]
    assert int(enriched["overlap_count"].max()) >= 2
    assert (result.output_dir / "report.md").exists()


def test_build_gmi_effect_graph_marks_anchors_and_interaction_edges(tmp_path):
    gmi_dir = tmp_path / "gmi_run"
    gmi_dir.mkdir()
    dataset = _write_toy_module_gmi_run(gmi_dir)
    main_effects = pd.read_csv(gmi_dir / "main_effects.tsv", sep="\t")
    interactions = pd.read_csv(gmi_dir / "interaction_effects.tsv", sep="\t")
    stability = pd.read_csv(gmi_dir / "stability.tsv", sep="\t")

    graph = build_gmi_effect_graph(
        dataset,
        main_effects=main_effects,
        interaction_effects=interactions,
        stability=stability,
        config=GmiModuleConfig(expansion_correlation=0.5),
    )

    anchors = graph["nodes"].loc[graph["nodes"]["is_anchor"], "feature"].astype(str).tolist()
    assert anchors == ["NIBAN1", "SORL1"]
    assert "gmi_interaction" in set(graph["edges"]["edge_type"])


def test_gmi_module_spatial_autocorr_drops_after_coordinate_value_shuffle(tmp_path):
    dataset = _toy_module_dataset()
    main_effects = pd.DataFrame({"feature": ["NIBAN1", "SORL1"], "coefficient": [-1.4, -1.1]})
    config = GmiModuleConfig(write_figures=False, expansion_correlation=0.5)
    real = discover_gmi_modules(
        dataset=dataset,
        output_dir=tmp_path / "real",
        main_effects=main_effects,
        interaction_effects=pd.DataFrame(),
        stability=pd.DataFrame(),
        config=config,
    )
    shuffled = ContourGmiDataset(
        X=pd.DataFrame(
            dataset.X.sample(frac=1.0, random_state=4).to_numpy(),
            index=dataset.X.index,
            columns=dataset.X.columns,
        ),
        y=dataset.y.copy(),
        sample_metadata=dataset.sample_metadata.copy(),
        feature_metadata=dataset.feature_metadata.copy(),
        config=dataset.config,
        provenance=dataset.provenance,
    )
    shuffled_result = discover_gmi_modules(
        dataset=shuffled,
        output_dir=tmp_path / "shuffled",
        main_effects=main_effects,
        interaction_effects=pd.DataFrame(),
        stability=pd.DataFrame(),
        config=config,
    )

    real_moran = float(real.module_spatial_autocorr["moran_i"].dropna().iloc[0])
    shuffled_moran = float(shuffled_result.module_spatial_autocorr["moran_i"].dropna().iloc[0])
    assert real_moran > shuffled_moran


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
