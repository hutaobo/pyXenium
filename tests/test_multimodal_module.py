from __future__ import annotations

import pytest

import pyXenium.multimodal as multimodal
import pyXenium.multimodal.analysis as multimodal_analysis
from pyXenium.validation import render_markdown_report


def test_multimodal_public_imports_smoke():
    assert callable(multimodal.load_rna_protein_anndata)
    assert callable(multimodal.rna_protein_cluster_analysis)
    assert callable(multimodal.protein_gene_correlation)
    assert callable(multimodal.run_validated_renal_ffpe_smoke)
    assert callable(multimodal.run_renal_immune_resistance_pilot)
    assert hasattr(multimodal, "analysis")
    assert hasattr(multimodal_analysis, "differential")
    assert hasattr(multimodal_analysis, "microenv_analysis")
    assert "tabnet_pipeline" in dir(multimodal_analysis)


def test_analysis_compat_module_for_moved_multimodal_symbol_warns():
    with pytest.warns(DeprecationWarning):
        from pyXenium.analysis.differential import get_rna_expr_df

    assert callable(get_rna_expr_df)


def test_validation_compat_wrapper_emits_deprecation_warning():
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
            "largest_clusters": [],
            "top_rna_features_by_total_counts": [],
            "top_protein_markers_by_mean_signal": [],
            "obsm_keys": ["protein", "spatial"],
        },
        "validated_reference": {
            "expected_cells": 1,
            "expected_rna_features": 2,
            "expected_protein_markers": 3,
        },
        "issues": [],
    }

    with pytest.warns(DeprecationWarning):
        report = render_markdown_report(payload)

    assert "pyXenium Smoke Test Report" in report
