import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from pyXenium.analysis.spatial_immune_resistance import (
    aggregate_multi_sample_study,
    annotate_joint_cell_states,
    build_spatial_niches,
    compute_rna_protein_discordance,
    score_immune_resistance_program,
)


def _make_test_adata() -> AnnData:
    obs_names = [f"cell_{i}" for i in range(12)]
    genes = [
        "EPCAM",
        "CDH1",
        "VIM",
        "ACTA2",
        "PECAM1",
        "CD68",
        "CD163",
        "HLA-DRA",
        "CD3E",
        "CD8A",
        "PDCD1",
        "LAG3",
        "GZMB",
        "MS4A1",
        "SDC1",
        "CCL5",
        "CD274",
    ]

    matrix = np.array(
        [
            [10, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [8, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 2, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 9, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 8, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 5, 9, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 9, 8, 6, 5, 7, 0, 0, 6, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 5, 4, 6, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 6, 5, 8, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0],
        ],
        dtype=float,
    )

    protein = pd.DataFrame(
        {
            "PanCK": [8, 8, 7, 3, 2, 0, 0, 0, 0, 0, 0, 0],
            "E-Cadherin": [7, 7, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            "Vimentin": [0, 0, 1, 8, 7, 1, 1, 1, 0, 0, 0, 0],
            "alphaSMA": [0, 0, 0, 6, 6, 0, 0, 5, 0, 0, 0, 0],
            "CD31": [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
            "CD68": [0, 0, 0, 0, 0, 9, 8, 1, 0, 0, 0, 0],
            "CD163": [0, 0, 0, 0, 0, 8, 9, 1, 0, 0, 0, 0],
            "HLA-DR": [0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0],
            "PD-1": [0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 8, 0],
            "LAG-3": [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 6, 0],
            "GranzymeB": [0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 8, 0],
            "PD-L1": [0, 0, 0, 0, 0, 8, 7, 1, 0, 0, 0, 0],
            "CD3E": [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 7, 0],
            "CD8A": [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 8, 0],
            "CD45RO": [0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 0],
            "CD20": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
            "CD138": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
            "CD45": [0, 0, 0, 0, 0, 4, 4, 1, 4, 4, 4, 0],
        },
        index=obs_names,
    )

    coords = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
            [10.0, 10.0],
            [10.5, 10.0],
            [11.0, 10.0],
            [20.0, 20.0],
            [20.5, 20.0],
            [21.0, 20.0],
            [20.2, 20.5],
        ],
        dtype=float,
    )

    adata = AnnData(X=sparse.csr_matrix(matrix))
    adata.layers["rna"] = sparse.csr_matrix(matrix)
    adata.obs_names = obs_names
    adata.var_names = genes
    adata.var["name"] = genes
    adata.obsm["protein"] = protein
    adata.obsm["spatial"] = coords
    return adata


def test_joint_state_annotation_assigns_expected_states():
    adata = _make_test_adata()
    result = annotate_joint_cell_states(adata)

    assert "joint_cell_state" in adata.obs.columns
    assert "joint_cell_class" in adata.obs.columns
    assert "tumor_epithelial" in set(adata.obs["joint_cell_state"].astype(str))
    assert "macrophage_like" in set(adata.obs["joint_cell_state"].astype(str))
    assert "t_cell_exhausted_cytotoxic" in set(adata.obs["joint_cell_state"].astype(str))
    assert "immune" in set(adata.obs["joint_cell_class"].astype(str))
    assert not result["class_summary"].empty
    assert not result["state_summary"].empty


def test_discordance_niches_and_program_scores_work_together():
    adata = _make_test_adata()
    annotate_joint_cell_states(adata)
    discordance = compute_rna_protein_discordance(adata, n_neighbors=2, region_bins=4)
    niches = build_spatial_niches(adata, n_neighbors=2, min_score=0.2)
    immune = score_immune_resistance_program(
        adata,
        discordance_result=discordance,
        niche_result=niches,
        region_bins=4,
    )

    assert not discordance["marker_summary"].empty
    assert "discordance__marker__pd1__signed" in discordance["cell_scores"].columns
    assert "myeloid_vascular" in set(adata.obs["spatial_niche"].astype(str))
    assert "immune_resistance__model__joint_activity" in immune["cell_scores"].columns
    assert not immune["model_comparison"].empty
    assert not immune["branch_summary"].empty
    assert not immune["marker_neighborhood_enrichment"].empty
    assert not immune["roi_scores"].empty


def test_aggregate_multi_sample_study_combines_outputs():
    adata_a = _make_test_adata()
    adata_b = _make_test_adata()

    study_a = {
        "sample_id": "sample_a",
        "discordance": compute_rna_protein_discordance(adata_a, n_neighbors=2, region_bins=4),
        "niches": build_spatial_niches(adata_a, n_neighbors=2, min_score=0.2),
    }
    study_a["immune_resistance"] = score_immune_resistance_program(
        adata_a,
        discordance_result=study_a["discordance"],
        niche_result=study_a["niches"],
        region_bins=4,
    )
    study_a["sample_summary"] = study_a["immune_resistance"]["sample_summary"]

    study_b = {
        "sample_id": "sample_b",
        "discordance": compute_rna_protein_discordance(adata_b, n_neighbors=2, region_bins=4),
        "niches": build_spatial_niches(adata_b, n_neighbors=2, min_score=0.2),
    }
    study_b["immune_resistance"] = score_immune_resistance_program(
        adata_b,
        discordance_result=study_b["discordance"],
        niche_result=study_b["niches"],
        region_bins=4,
    )
    study_b["sample_summary"] = study_b["immune_resistance"]["sample_summary"]

    aggregate = aggregate_multi_sample_study([study_a, study_b])

    assert set(aggregate["sample_summary"]["sample_id"]) == {"sample_a", "sample_b"}
    assert not aggregate["cohort_summary"].empty
    assert not aggregate["model_summary"].empty
    assert not aggregate["branch_summary"].empty
