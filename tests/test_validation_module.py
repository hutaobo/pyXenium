import pandas as pd

from pyXenium.validation.renal_ffpe_protein import render_markdown_report
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
