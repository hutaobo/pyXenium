from pyXenium.validation.renal_ffpe_protein import render_markdown_report


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
