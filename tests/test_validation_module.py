import pandas as pd

from pyXenium.validation.renal_ffpe_protein import render_markdown_report
from pyXenium.validation.atera_wta_breast_topology import render_atera_wta_breast_topology_report
from pyXenium.validation.atera_wta_cervical_end_to_end import (
    render_atera_wta_cervical_end_to_end_report,
)
from pyXenium.validation.renal_immune_resistance import (
    build_cohort_handoff_spec,
    build_panel_gap_table,
    build_top_hypotheses_table,
    render_renal_immune_resistance_report,
)


def test_render_markdown_report_mentions_core_sections():
    payload = {
        "summary": {
            "dataset_title": "Example dataset",
            "dataset_url": "https://example.org",
            "base_path": "/tmp/example",
            "prefer": "auto",
            "n_cells": 1,
            "n_rna_features": 2,
            "n_protein_markers": 3,
            "x_nnz": 4,
            "has_spatial": True,
            "has_cluster": True,
            "metrics_summary_num_cells_detected": 1,
            "largest_clusters": [{"cluster": "0", "n_cells": 1}],
            "top_rna_features_by_total_counts": [{"feature": "VIM", "total_counts": 5.0, "detected_cells": 1}],
            "top_protein_markers_by_mean_signal": [
                {"marker": "Vimentin", "mean_signal": 2.5, "positive_cells": 1}
            ],
        },
        "validated_reference": {
            "expected_cells": 1,
            "expected_rna_features": 2,
            "expected_protein_markers": 3,
        },
        "issues": [],
    }

    report = render_markdown_report(payload)

    assert "# pyXenium Smoke Test Report" in report
    assert "## Core Results" in report
    assert "## Top RNA Features by Total Counts" in report
    assert "## Top Protein Markers by Mean Signal" in report
    assert "No issues detected." in report


def test_renal_immune_resistance_helpers_render_expected_sections():
    study = {
        "immune_resistance": {
            "branch_summary": pd.DataFrame(
                [
                    {
                        "branch": "myeloid_vascular",
                        "best_model": "myeloid_vascular_branch",
                        "benchmark_score": 0.7,
                        "held_out_auc": 0.8,
                        "spatial_coherence": 0.4,
                        "roi_reproducibility": 0.6,
                    }
                ]
            ),
            "marker_neighborhood_enrichment": pd.DataFrame(
                [
                    {
                        "branch": "myeloid_vascular",
                        "label": "alpha_sma",
                        "protein": "alphaSMA",
                        "gene": "ACTA2",
                        "abs_target_correlation": 0.5,
                        "abs_rank_score": 0.5,
                    }
                ]
            ),
        }
    }

    hypotheses = build_top_hypotheses_table(study)
    assert not hypotheses.empty
    assert "suggested_validation" in hypotheses.columns

    payload = {
        "dataset_title": "Example dataset",
        "dataset_url": "https://example.org",
        "base_path": "/tmp/example",
        "prefer": "auto",
        "n_cells": 10,
        "n_rna_features": 5,
        "n_protein_markers": 2,
        "resistant_niches": ["myeloid_vascular"],
        "claim_positioning": "Joint analysis localizes orthogonal biology.",
        "joint_cell_classes": [{"class": "immune", "n_cells": 5}],
        "joint_cell_states": [{"state": "macrophage_like", "n_cells": 5}],
        "top_marker_discordance": [{"label": "alpha_sma", "protein": "alphaSMA", "gene": "ACTA2", "mean_abs_discordance": 1.2, "mean_signed_discordance": 0.8}],
        "top_pathway_discordance": [{"pathway": "vascular_stromal", "mean_abs_discordance": 0.9, "mean_signed_discordance": 0.4}],
        "branch_summary": hypotheses[["branch", "best_model", "benchmark_score", "held_out_auc"]].to_dict(orient="records"),
        "top_hypotheses": hypotheses.to_dict(orient="records"),
    }
    report = render_renal_immune_resistance_report(payload)

    assert "# Renal Immune Resistance Discovery Package" in report
    assert "## Branch Summary" in report
    assert "## Top Hypotheses" in report

    assert not build_cohort_handoff_spec().empty
    assert not build_panel_gap_table().empty


def test_atera_breast_topology_report_mentions_cci_and_pathway_sections():
    payload = {
        "sample_id": "atera_test",
        "dataset_root": "/tmp/atera",
        "tbc_results": "/tmp/atera/results",
        "n_cells": 170057,
        "n_rna_features": 18028,
        "cluster_count": 19,
        "topology_celltype_count": 19,
        "unassigned_cells": 0,
        "experiment_metadata": {"panel_num_targets_predesigned": 18028},
        "metrics_summary": {"median_transcripts_per_cell": 2116},
        "runtime_seconds": 1.23,
        "cci_pair_summaries": [{"ligand": "CSF1", "receptor": "CSF1R", "best_sender_celltype": "CAFs", "best_receiver_celltype": "Macrophages", "best_score": 0.42}],
        "cci_acceptance": [{"check": "CSF1-CSF1R top sender should not be Mast Cells", "pass": True}],
        "pathway_primary_best": [{"pathway": "MacrophageProgram", "best_celltype": "Macrophages", "best_distance": 0.02}],
        "pathway_acceptance": [{"pathway": "MacrophageProgram", "expected_best_celltypes": ["Macrophages"], "observed_best_celltype": "Macrophages", "pass": True}],
        "files": {"summary_json": "/tmp/atera/summary.json"},
    }

    report = render_atera_wta_breast_topology_report(payload)

    assert "# Atera WTA Breast Topology Reproducibility Bundle" in report
    assert "## CCI Smoke Panel" in report
    assert "## Pathway Primary Results" in report


def test_atera_cervical_end_to_end_report_mentions_contour_and_multimodal_sections():
    payload = {
        "sample_id": "atera_cervical_test",
        "dataset_root": "/tmp/atera_cervical",
        "output_root": "/tmp/atera_cervical/pyxenium_cervical_end_to_end",
        "tbc_results": "/tmp/atera_cervical/sfplot_tbc_formal_wta/results",
        "n_cells": 1234,
        "n_rna_features": 5678,
        "cluster_count": 26,
        "contour_structure_count": 6,
        "contour_polygon_count": 12,
        "expanded_contour_polygon_count": 12,
        "metrics_summary": {"median_transcripts_per_cell": 2116},
        "runtime_seconds": 3.21,
        "ring_density_summary": {"row_count": 24, "features": ["SPP1", "CA9"]},
        "smooth_density_summary": {"row_count": 36, "features": ["SPP1", "CA9"]},
        "multimodal_sample_summary": {"contour_key": "atera_cervical_bio6", "n_contours": 12, "n_ecotypes": 4},
        "contour_structures": [
            {"structure_name": "Tumor", "structure_id": 1, "cluster_ids": ["Hypoxic Tumor Cells"]},
            {"structure_name": "T-cell", "structure_id": 2, "cluster_ids": ["Cytotoxic T Cells"]},
        ],
        "files": {"summary_json": "/tmp/atera_cervical/summary.json"},
    }

    report = render_atera_wta_cervical_end_to_end_report(payload)

    assert "# Atera WTA Cervical End-to-End Reproducibility Bundle" in report
    assert "## Contour Bio6 Structures" in report
    assert "## Density Profiling" in report
    assert "## Multimodal Contour Ecology" in report
