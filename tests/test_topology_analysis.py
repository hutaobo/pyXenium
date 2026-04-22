from __future__ import annotations

import pandas as pd

from pyXenium import (
    compute_pathway_activity_matrix,
    ligand_receptor_topology_analysis,
    pathway_topology_analysis,
)


def _toy_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "x": [0.0, 1.0, 5.0, 6.0, 10.0, 11.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "celltype": ["A", "A", "B", "B", "C", "C"],
        }
    )


def _toy_expression(reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "L1": [5.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            "R1": [0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
            "L2": [0.0, 0.0, 0.0, 0.0, 5.0, 4.0],
            "R2": [4.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            "G1": [5.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            "G2": [0.0, 0.0, 5.0, 4.0, 0.0, 0.0],
            "G3": [0.0, 0.0, 0.0, 0.0, 5.0, 4.0],
        },
        index=reference["cell_id"],
    )


def _toy_t_and_c() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            "B": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "C": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        },
        index=["L1", "R1", "L2", "R2", "G1", "G2", "G3"],
    )


def _toy_structure_map() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.0, 0.3, 0.8], [0.3, 0.0, 0.5], [0.8, 0.5, 0.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


def test_public_topology_exports_are_importable():
    assert callable(ligand_receptor_topology_analysis)
    assert callable(pathway_topology_analysis)
    assert callable(compute_pathway_activity_matrix)


def test_ligand_receptor_precomputed_anchors_match_supplied_topology():
    reference = _toy_reference()
    expression = _toy_expression(reference)
    t_and_c = _toy_t_and_c()
    structure_map = _toy_structure_map()
    lr_pairs = pd.DataFrame([{"ligand": "L1", "receptor": "R1", "evidence_weight": 1.0}])

    result = ligand_receptor_topology_analysis(
        reference_df=reference,
        expression_df=expression,
        lr_pairs=lr_pairs,
        t_and_c_df=t_and_c,
        structure_map_df=structure_map,
        export_figures=False,
    )

    assert result["ligand_to_cell"].loc["L1", "A"] == 0.0
    assert result["receptor_to_cell"].loc["R1", "B"] == 0.0
    assert set(result["scores"]["anchor_source_ligand"]) == {"precomputed"}
    assert set(result["scores"]["anchor_source_receptor"]) == {"precomputed"}


def test_ligand_receptor_hybrid_marks_fallback_sources_and_zeroes_sparse_contact():
    reference = _toy_reference()
    expression = _toy_expression(reference)
    t_and_c = _toy_t_and_c().drop(index=["R2"])
    structure_map = _toy_structure_map()
    lr_pairs = pd.DataFrame([{"ligand": "L2", "receptor": "R2", "evidence_weight": 1.0}])

    result = ligand_receptor_topology_analysis(
        reference_df=reference,
        expression_df=expression,
        lr_pairs=lr_pairs,
        t_and_c_df=t_and_c,
        structure_map_df=structure_map,
        anchor_mode="hybrid",
        min_cross_edges=999,
        export_figures=False,
    )

    assert "precomputed" in set(result["scores"]["anchor_source_ligand"])
    assert "recompute" in set(result["scores"]["anchor_source_receptor"])
    assert float(result["scores"]["local_contact"].fillna(0.0).max()) == 0.0
    assert not result["scores"]["local_contact"].isna().any()


def test_pathway_topology_writes_dual_outputs_and_diagnostics(tmp_path):
    reference = _toy_reference()
    expression = _toy_expression(reference)
    t_and_c = _toy_t_and_c()
    structure_map = _toy_structure_map()

    result = pathway_topology_analysis(
        reference_df=reference,
        expression_df=expression,
        pathway_definitions={
            "ProgA": ["G1"],
            "ProgB": ["G2", "G3"],
        },
        t_and_c_df=t_and_c,
        structure_map_df=structure_map,
        output_dir=tmp_path,
        export_figures=False,
    )

    assert (tmp_path / "pathway_to_cell.csv").exists()
    assert (tmp_path / "pathway_structuremap.csv").exists()
    assert (tmp_path / "pathway_activity_to_cell.csv").exists()
    assert (tmp_path / "pathway_activity_structuremap.csv").exists()
    assert (tmp_path / "pathway_mode_comparison.csv").exists()
    assert {"retained_cell_count", "retained_quantile", "activity_mode"}.issubset(
        result["pathway_mode_comparison"].columns
    )
    assert result["pathway_to_cell"].loc["ProgA"].astype(float).idxmin() == "A"
