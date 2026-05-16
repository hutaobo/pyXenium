from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyXenium import (
    MorphoPathwayConfig,
    aggregate_morphopathway_inputs_to_spatial_blocks,
    build_curated_pathway_panel,
    compute_matched_random_pathway_controls,
    compute_pathway_coverage,
    fit_residual_pathway_morphology_associations,
    prepare_xenium_cell_morphopathway_inputs,
    run_atera_morphopathway_brief,
    run_xenium_cell_morphopathway_smoke,
    summarize_cross_cancer_validation,
)

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


def _toy_morphopathway_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 80
    index = pd.Index([f"contour_{idx}" for idx in range(n)], name="contour_id")
    latent = rng.normal(size=n)
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    structure = np.where(np.arange(n) < n / 2, "S1", "S2")
    structure_effect = np.where(structure == "S1", 0.7, -0.4)

    expression = pd.DataFrame(
        {
            "G1": 4.0 + 2.0 * latent + structure_effect + 0.3 * x,
            "G2": 3.0 + 1.4 * latent + 0.2 * y,
            "G3": rng.normal(size=n),
            "NOISE1": rng.normal(size=n),
        },
        index=index,
    )
    image_features = pd.DataFrame(
        {
            "embedding__signal": latent + rng.normal(scale=0.08, size=n),
            "embedding__noise": rng.normal(size=n),
        },
        index=index,
    )
    metadata = pd.DataFrame(
        {
            "structure": structure,
            "x": x,
            "y": y,
            "boundary_distance": np.abs(x - 0.5),
        },
        index=index,
    )
    panel = build_curated_pathway_panel(
        pd.DataFrame(
            [
                {"pathway": "signal_pathway", "family": "toy_family", "gene": "G1", "weight": 1.0},
                {"pathway": "signal_pathway", "family": "toy_family", "gene": "G2", "weight": 1.0},
                {"pathway": "noise_pathway", "family": "toy_family", "gene": "G3", "weight": 1.0},
            ]
        )
    )
    return expression, image_features, metadata, panel


def _write_minimal_brief_package(package_dir: Path) -> None:
    package_dir.mkdir()
    stable_core = [
        "unfolded_protein_response",
        "immune_exclusion",
        "luminal_estrogen_response",
        "myofibroblast_caf_activation",
        "oxidative_phosphorylation",
        "basal_squamous_state",
        "collagen_ecm_organization",
        "immune_activation",
        "epithelial_identity",
    ]
    (package_dir / "README.md").write_text("# Minimal morphopathway package\n", encoding="utf-8")
    (package_dir / "brief_communication_package_manifest.json").write_text(
        json.dumps(
            {
                "cross_cancer_recovery_min": 9,
                "cross_cancer_recovery_max": 10,
                "axis_masked_recovery_min": 9,
                "axis_masked_recovery_max": 10,
                "cross_cancer_total": 10,
                "stable_9_pathway_core": stable_core,
                "candidate_axis_runs": 2,
                "source_tables": [
                    "main_figure_1_source_breast_discovery_highnull32.csv",
                    "main_figure_2_source_cross_cancer_stability_highnull32.csv",
                    "source_table_cross_cancer_validation_by_run.csv",
                    "supp_table_cervical_validation_best_associations.csv",
                    "supp_table_highnull32_gate_and_axis_masked_summary.csv",
                    "supp_table_plip_axis_diagnostics_by_run.csv",
                    "supp_table_spatial_block_and_seed_summary.csv",
                ],
                "notes": [
                    "README.md",
                    "claim_wording.md",
                    "methods_statistics_notes.md",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (package_dir / "claim_wording.md").write_text(
        "\n".join(
            [
                "Recommended primary claim:",
                "",
                "A pathway-family stress-test recovers a stable H&E-WTA morphopathway core.",
                "",
                "Do not claim:",
                "",
                "Axis-masked sensitivity",
                "",
                "Stable pathway-family stress-test core",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (package_dir / "methods_statistics_notes.md").write_text("Methods notes.\n", encoding="utf-8")
    pd.DataFrame({"pathway": stable_core, "breast_rank": range(1, 10)}).to_csv(
        package_dir / "main_figure_1_source_breast_discovery_highnull32.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "pathway": pathway,
                "stable_9_pathway_core": True,
                "primary_recovery_rate": 1.0,
                "axis_masked_recovery_rate": 1.0,
            }
            for pathway in stable_core
        ]
        + [
            {
                "pathway": "emt_invasive_front",
                "stable_9_pathway_core": False,
                "primary_recovery_rate": 0.33,
                "axis_masked_recovery_rate": 0.33,
            }
        ]
    ).to_csv(package_dir / "main_figure_2_source_cross_cancer_stability_highnull32.csv", index=False)
    pd.DataFrame({"run": ["seed17", "seed29", "seed43"], "recovered": [9, 9, 10]}).to_csv(
        package_dir / "source_table_cross_cancer_validation_by_run.csv", index=False
    )
    pd.DataFrame({"pathway": stable_core, "validation_abs_rho": np.linspace(0.4, 0.7, len(stable_core))}).to_csv(
        package_dir / "supp_table_cervical_validation_best_associations.csv", index=False
    )
    pd.DataFrame(
        {
            "run": ["seed17", "seed29", "seed43"],
            "cross_cancer_recovered": [9, 9, 10],
            "axis_masked_cross_cancer_recovered": [9, 9, 10],
            "breast_negative_control_pass95": [8, 8, 10],
            "cervical_negative_control_pass95": [10, 9, 10],
        }
    ).to_csv(package_dir / "supp_table_highnull32_gate_and_axis_masked_summary.csv", index=False)
    pd.DataFrame({"run": ["seed17", "seed29"], "candidate_generic_axis_count": [1, 2]}).to_csv(
        package_dir / "supp_table_plip_axis_diagnostics_by_run.csv", index=False
    )
    pd.DataFrame({"run": ["seed17", "seed29", "seed43"], "spatial_block_bins": [12, 12, 12]}).to_csv(
        package_dir / "supp_table_spatial_block_and_seed_summary.csv", index=False
    )


def test_morphopathway_public_api_exports_are_stable():
    import pyXenium
    import pyXenium.pathway as pathway_api

    expected = [
        "MorphoPathwayConfig",
        "build_curated_pathway_panel",
        "compute_matched_random_pathway_controls",
        "compute_pathway_coverage",
        "fit_residual_pathway_morphology_associations",
        "prepare_xenium_cell_morphopathway_inputs",
        "run_atera_morphopathway_brief",
        "run_xenium_cell_morphopathway_smoke",
        "sample_clip_image_embeddings_at_cells",
        "sample_he_image_features_at_cells",
        "summarize_cross_cancer_validation",
    ]

    for name in expected:
        assert hasattr(pathway_api, name)
        assert hasattr(pyXenium, name)

    signature = inspect.signature(pathway_api.run_xenium_cell_morphopathway_smoke)
    assert signature.parameters["spatial_block_bins"].default == 12


def _write_toy_xenium_outs(root):
    if h5py is None:
        pytest.skip("h5py is required for Xenium H5 smoke fixtures.")

    root.mkdir()
    cell_ids = [f"cell_{idx}" for idx in range(8)]
    cells = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_centroid": np.linspace(0.0, 70.0, len(cell_ids)),
            "y_centroid": np.linspace(5.0, 75.0, len(cell_ids)),
            "transcript_counts": [105, 117, 129, 141, 153, 165, 177, 189],
            "control_probe_counts": [0, 1, 0, 1, 0, 1, 0, 1],
            "genomic_control_counts": [0] * len(cell_ids),
            "total_counts": [105, 118, 129, 142, 153, 166, 177, 190],
            "cell_area": [40.0, 44.0, 50.0, 56.0, 62.0, 68.0, 76.0, 84.0],
            "nucleus_area": [20.0, 22.0, 25.0, 29.0, 32.0, 36.0, 40.0, 45.0],
            "nucleus_count": [1] * len(cell_ids),
            "segmentation_method": ["toy"] * len(cell_ids),
        }
    )
    cells.to_parquet(root / "cells.parquet")

    genes = ["G1", "G2", "G3", "BG1", "BG2"]
    cell_by_gene = np.asarray(
        [
            [20, 18, 1, 5, 2],
            [24, 21, 2, 3, 4],
            [29, 26, 1, 6, 1],
            [35, 31, 3, 3, 4],
            [41, 37, 1, 5, 2],
            [47, 43, 2, 3, 4],
            [54, 50, 1, 6, 1],
            [61, 57, 2, 4, 3],
        ],
        dtype=np.int32,
    )
    data: list[int] = []
    indices: list[int] = []
    indptr = [0]
    for row in cell_by_gene:
        nonzero = np.flatnonzero(row)
        indices.extend(int(value) for value in nonzero)
        data.extend(int(value) for value in row[nonzero])
        indptr.append(len(indices))

    with h5py.File(root / "cell_feature_matrix.h5", "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray(cell_ids, dtype="S"))
        matrix.create_dataset("data", data=np.asarray(data, dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
        matrix.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        matrix.create_dataset("shape", data=np.asarray([len(genes), len(cell_ids)], dtype=np.int32))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray(genes, dtype="S"))
        features.create_dataset("name", data=np.asarray(genes, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(["Gene Expression"] * len(genes), dtype="S"))
        features.create_dataset("genome", data=np.asarray(["toy"] * len(genes), dtype="S"))
    if tifffile is not None:
        yy, xx = np.mgrid[0:100, 0:100]
        he = np.stack(
            [
                np.clip(xx * 2, 0, 255),
                np.clip(255 - yy * 2, 0, 255),
                np.clip(80 + yy + xx, 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)
        tifffile.imwrite(root / "toy_he_image.ome.tif", he, photometric="rgb")
        np.savetxt(root / "toy_he_alignment.csv", np.eye(3), delimiter=",")
    return root


def test_morphopathway_bundle_recovers_residual_signal_and_writes_outputs(tmp_path):
    expression, image_features, metadata, panel = _toy_morphopathway_inputs()
    config = MorphoPathwayConfig(
        min_pathway_genes=1,
        min_pathway_coverage=0.5,
        n_permutations=16,
        permutation_top_n=1,
        n_negative_controls=8,
        negative_control_top_n=1,
        spatial_strata_cols=("structure",),
        random_state=11,
    )

    result = run_atera_morphopathway_brief(
        expression_df=expression,
        image_features_df=image_features,
        metadata_df=metadata,
        pathway_panel=panel,
        output_dir=tmp_path,
        config=config,
        sample_name="toy",
    )

    top = result["associations"].iloc[0]
    assert top["pathway"] == "signal_pathway"
    assert top["image_feature"] == "embedding__signal"
    assert float(top["abs_partial_spearman_rho"]) > 0.85
    assert 0.0 <= float(top["permutation_empirical_p"]) <= 1.0
    assert "negative_control_empirical_p" in result["associations"].columns
    assert not result["negative_controls"].empty
    assert result["negative_controls"].iloc[0]["pathway"] == "signal_pathway"
    assert (tmp_path / "pathway_coverage.csv").exists()
    assert (tmp_path / "association_table.csv").exists()
    assert (tmp_path / "spatial_nulls.csv").exists()
    assert (tmp_path / "negative_controls.csv").exists()
    assert (tmp_path / "figure_source_table.csv").exists()


def test_morphopathway_coverage_and_association_api_are_importable():
    expression, image_features, metadata, panel = _toy_morphopathway_inputs()
    coverage = compute_pathway_coverage(expression, panel, config=MorphoPathwayConfig(min_pathway_genes=1))
    assert set(coverage["pathway"]) == {"signal_pathway", "noise_pathway"}
    assert coverage.set_index("pathway").loc["signal_pathway", "passes"]

    activity = pd.DataFrame(
        {
            "signal_pathway": expression[["G1", "G2"]].mean(axis=1),
            "noise_pathway": expression["G3"],
        },
        index=expression.index,
    )
    fit = fit_residual_pathway_morphology_associations(
        activity,
        image_features,
        metadata,
        pathway_panel=panel,
        config=MorphoPathwayConfig(n_permutations=0),
    )
    assert not fit["associations"].empty
    assert {"partial_spearman_rho", "fdr", "rank"}.issubset(fit["associations"].columns)

    controls = compute_matched_random_pathway_controls(
        expression,
        image_features,
        metadata,
        fit["associations"],
        panel,
        config=MorphoPathwayConfig(n_negative_controls=4, negative_control_top_n=1, random_state=3),
    )
    assert {"negative_control_empirical_p", "negative_control_abs_rho_q95"}.issubset(controls.columns)
    assert len(controls) == 1


def test_cross_cancer_validation_reports_pathway_or_family_recovery():
    discovery = pd.DataFrame(
        [
            {
                "pathway": "signal_pathway",
                "family": "toy_family",
                "image_feature": "embedding__signal",
                "abs_partial_spearman_rho": 0.72,
            },
            {
                "pathway": "noise_pathway",
                "family": "toy_family",
                "image_feature": "embedding__noise",
                "abs_partial_spearman_rho": 0.12,
            },
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "pathway": "noise_pathway",
                "family": "toy_family",
                "image_feature": "embedding__signal",
                "abs_partial_spearman_rho": 0.43,
            }
        ]
    )

    summary = summarize_cross_cancer_validation(
        discovery,
        validation,
        config=MorphoPathwayConfig(validation_abs_rho_threshold=0.35),
    )

    row = summary.loc[summary["pathway"] == "signal_pathway"].iloc[0]
    assert row["validation_family_best_pathway"] == "noise_pathway"
    assert bool(row["recovered_in_validation"])
    assert row["validation_call"] == "pathway_or_family_recovered"


def test_spatial_block_aggregation_preserves_inputs_at_tissue_scale():
    expression, image_features, metadata, _panel = _toy_morphopathway_inputs()

    aggregated = aggregate_morphopathway_inputs_to_spatial_blocks(
        expression,
        image_features,
        metadata,
        bins=4,
        min_cells_per_block=4,
    )

    assert 1 < len(aggregated["expression"]) < len(expression)
    assert aggregated["expression"].index.equals(aggregated["image_features"].index)
    assert aggregated["metadata"].index.equals(aggregated["expression"].index)
    assert "n_cells_in_block" in aggregated["metadata"].columns
    assert int(aggregated["metadata"]["n_cells_in_block"].min()) >= 4
    assert not aggregated["block_manifest"].empty


def test_xenium_cell_morphopathway_smoke_reads_h5_and_writes_bundle(tmp_path):
    if tifffile is None:
        pytest.skip("tifffile is required for H&E feature smoke fixtures.")
    xenium_root = _write_toy_xenium_outs(tmp_path / "xenium")
    panel = build_curated_pathway_panel(
        pd.DataFrame(
            [
                {"pathway": "signal_pathway", "family": "toy_family", "gene": "G1", "weight": 1.0},
                {"pathway": "signal_pathway", "family": "toy_family", "gene": "G2", "weight": 1.0},
                {"pathway": "noise_pathway", "family": "toy_family", "gene": "G3", "weight": 1.0},
            ]
        )
    )

    prepared = prepare_xenium_cell_morphopathway_inputs(
        xenium_root,
        pathway_panel=panel,
        max_cells=None,
        extra_background_genes=2,
        include_patch_projection=True,
        patch_projection_dim=6,
        random_state=5,
    )
    assert prepared["expression"].shape == (8, 5)
    assert prepared["image_features"].filter(like="morphology__").shape[1] >= 4
    assert "image__he_luma_mean" in prepared["image_features"].columns
    assert "image__he_texture_entropy" in prepared["image_features"].columns
    assert "image__he_edge_mean" in prepared["image_features"].columns
    assert "embedding__he_patch_projection_000" in prepared["image_features"].columns
    assert "log_total_counts" in prepared["metadata"].columns
    assert bool(prepared["input_manifest"].loc[0, "he_features_available"])
    assert prepared["input_manifest"].loc[0, "he_embedding_backend_status"] == "deterministic_patch_projection_fallback"

    result = run_xenium_cell_morphopathway_smoke(
        xenium_root,
        output_dir=tmp_path / "bundle",
        sample_name="toy_xenium",
        pathway_panel=panel,
        max_cells=None,
        extra_background_genes=2,
        include_patch_projection=True,
        patch_projection_dim=6,
        random_state=5,
        config=MorphoPathwayConfig(
            min_pathway_genes=1,
            min_pathway_coverage=0.5,
            image_feature_prefixes=("embedding__he_patch_projection_", "image__he_"),
            covariates=("structure", "x", "y", "boundary_distance", "log_total_counts"),
            n_permutations=0,
            n_negative_controls=2,
            negative_control_top_n=1,
            random_state=5,
        ),
    )
    assert not result["associations"].empty
    assert (tmp_path / "bundle" / "input_manifest.csv").exists()
    assert (tmp_path / "bundle" / "he_feature_manifest.csv").exists()
    assert (tmp_path / "bundle" / "negative_controls.csv").exists()


def test_brief_package_qc_and_archive_are_idempotent(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = tmp_path / "brief_package"
    _write_minimal_brief_package(package_dir)

    validate_script = repo_root / "benchmarking" / "morphopathway_atera" / "scripts" / "validate_brief_communication_package.py"
    archive_script = repo_root / "benchmarking" / "morphopathway_atera" / "scripts" / "archive_brief_communication_package.py"
    subprocess.run([sys.executable, str(validate_script), str(package_dir)], check=True, capture_output=True, text=True)

    report_paths = [package_dir / "package_qc_report.json", package_dir / "package_qc_report.md"]
    before_report_mtimes = [path.stat().st_mtime_ns for path in report_paths]
    subprocess.run([sys.executable, str(validate_script), str(package_dir)], check=True, capture_output=True, text=True)
    after_report_mtimes = [path.stat().st_mtime_ns for path in report_paths]
    assert before_report_mtimes == after_report_mtimes

    subprocess.run([sys.executable, str(archive_script), str(package_dir)], check=True, capture_output=True, text=True)
    archive_path = package_dir.parent / f"{package_dir.name}.zip"
    before_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    before_archive_mtime = archive_path.stat().st_mtime_ns
    readme_path = package_dir / "README.md"
    readme_mtime = readme_path.stat().st_mtime
    os.utime(readme_path, (readme_mtime + 10.0, readme_mtime + 10.0))
    second_archive = subprocess.run(
        [sys.executable, str(archive_script), str(package_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    after_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    after_archive_mtime = archive_path.stat().st_mtime_ns
    assert before_hash == after_hash
    assert before_archive_mtime == after_archive_mtime

    archive_manifest = json.loads(second_archive.stdout)
    assert archive_manifest["archive_sha256"] == after_hash

    archive_path.write_bytes(b"corrupt archive")
    corrupt_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    rebuilt_archive = subprocess.run(
        [sys.executable, str(archive_script), str(package_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    rebuilt_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    rebuilt_manifest = json.loads(rebuilt_archive.stdout)
    assert rebuilt_hash != corrupt_hash
    assert rebuilt_manifest["archive_sha256"] == rebuilt_hash


def test_reviewer_evidence_audit_is_idempotent(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = tmp_path / "brief_package"
    _write_minimal_brief_package(package_dir)

    validate_script = repo_root / "benchmarking" / "morphopathway_atera" / "scripts" / "validate_brief_communication_package.py"
    archive_script = repo_root / "benchmarking" / "morphopathway_atera" / "scripts" / "archive_brief_communication_package.py"
    audit_script = repo_root / "benchmarking" / "morphopathway_atera" / "scripts" / "make_reviewer_evidence_audit.py"
    subprocess.run([sys.executable, str(validate_script), str(package_dir)], check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(archive_script), str(package_dir)], check=True, capture_output=True, text=True)

    first = subprocess.run(
        [sys.executable, str(audit_script), str(package_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    first_payload = json.loads(first.stdout)
    assert first_payload["status"] == "pass"

    audit_paths = [
        package_dir / "reviewer_evidence_audit.json",
        package_dir / "reviewer_evidence_audit.csv",
        package_dir / "reviewer_evidence_audit.md",
    ]
    assert all(path.exists() for path in audit_paths)
    audit_json = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    assert audit_json["metrics"]["cross_cancer_recovery_min"] == 9
    assert audit_json["metrics"]["breast_negative_control_pass95_min"] == 8
    assert {row["evidence_item"] for row in audit_json["audit_rows"]}.issuperset(
        {"Primary claim boundary", "Matched negative-control limitation", "Archive reproducibility"}
    )
    required_audit_qc = subprocess.run(
        [sys.executable, str(validate_script), str(package_dir), "--require-reviewer-audit"],
        check=True,
        capture_output=True,
        text=True,
    )
    required_audit_qc_payload = json.loads(required_audit_qc.stdout)
    assert required_audit_qc_payload["status"] == "pass"
    assert required_audit_qc_payload["reviewer_audit_checked"]

    subprocess.run([sys.executable, str(archive_script), str(package_dir)], check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(audit_script), str(package_dir)], check=True, capture_output=True, text=True)
    before_mtimes = [path.stat().st_mtime_ns for path in audit_paths]
    second = subprocess.run(
        [sys.executable, str(audit_script), str(package_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    after_mtimes = [path.stat().st_mtime_ns for path in audit_paths]
    assert second_payload["status"] == "pass"
    assert before_mtimes == after_mtimes
