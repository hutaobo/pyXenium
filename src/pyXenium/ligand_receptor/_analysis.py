from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pyXenium._topology_core import (
    build_neighbor_index,
    coerce_expression_df,
    coerce_reference_df,
    compute_local_contact_matrix,
    ensure_figures_dir,
    ensure_output_dir,
    geometric_mean,
    normalize_lr_prior,
    prepare_hotspot_table,
    resolve_gene_topology_anchors,
    save_hotspot_overlay,
    save_matrix_heatmap,
    safe_to_parquet,
    summarize_expression_by_celltype,
    winsorized_normalize_series,
)


def ligand_receptor_topology_analysis(
    *,
    reference_df: Optional[pd.DataFrame] = None,
    expression_df: Optional[pd.DataFrame] = None,
    lr_pairs: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    adata: Any = None,
    entity_points_df: Optional[pd.DataFrame] = None,
    tbc_results: Optional[str | Path] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
    prior_col: str = "evidence_weight",
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
    anchor_mode: str = "precomputed",
    expression_support_mode: str = "pseudobulk_detection",
    contact_mode: str = "strength_coverage",
    entity_min_weight: float = 0.0,
    detection_threshold: float = 0.0,
    k_neighbors: int = 8,
    radius: Optional[float] = None,
    topology_method: str = "average",
    top_n_pairs: int = 12,
    hotspot_quantile: float = 0.9,
    min_cross_edges: int = 50,
    contact_expr_threshold: str | float = "q75_nonzero",
    export_figures: bool = True,
    use_raw: bool = False,
) -> dict[str, Any]:
    """
    Score ligand-receptor hypotheses using topology anchors, sender/receiver
    expression support, and de-saturated local contact structure.
    """

    if expression_support_mode != "pseudobulk_detection":
        raise ValueError("expression_support_mode currently supports only 'pseudobulk_detection'.")
    if contact_mode != "strength_coverage":
        raise ValueError("contact_mode currently supports only 'strength_coverage'.")
    if ligand_col not in lr_pairs.columns or receptor_col not in lr_pairs.columns:
        raise ValueError(f"lr_pairs must contain {ligand_col!r} and {receptor_col!r}.")

    reference = coerce_reference_df(
        reference_df,
        adata=adata,
        cluster_col=cluster_col,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        celltype_col=celltype_col,
    )
    reference.index = reference[cell_id_col].astype(str)
    reference["cell_id"] = reference[cell_id_col].astype(str)

    pairs = lr_pairs.copy()
    pairs[ligand_col] = pairs[ligand_col].astype(str)
    pairs[receptor_col] = pairs[receptor_col].astype(str)
    pairs["prior_confidence"] = normalize_lr_prior(pairs, prior_col)

    genes = list(dict.fromkeys(pairs[ligand_col].tolist() + pairs[receptor_col].tolist()))
    expression = coerce_expression_df(
        reference,
        expression_df,
        adata=adata,
        genes=genes,
        cell_id_col=cell_id_col,
        use_raw=use_raw,
    )
    expression.index = reference.index

    topology, anchor_sources, resolved_structure_map, structure_map_source = resolve_gene_topology_anchors(
        reference,
        expression,
        genes,
        tbc_results=tbc_results,
        t_and_c_df=t_and_c_df,
        structure_map=structure_map,
        structure_map_df=structure_map_df,
        anchor_mode=anchor_mode,
        entity_points_df=entity_points_df,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        entity_min_weight=entity_min_weight,
        topology_method=topology_method,
    )
    ligand_to_cell = topology.reindex(pairs[ligand_col].drop_duplicates())
    receptor_to_cell = topology.reindex(pairs[receptor_col].drop_duplicates())
    structure_map_resolved = resolved_structure_map

    _, expression_support = summarize_expression_by_celltype(
        expression,
        reference["celltype"],
        detection_threshold=detection_threshold,
    )
    neighbor_index = build_neighbor_index(reference, x_col=x_col, y_col=y_col, k_neighbors=k_neighbors, radius=radius)
    celltypes = list(dict.fromkeys(reference["celltype"].astype(str).tolist()))
    score_rows: list[dict[str, Any]] = []

    for _, pair in pairs.iterrows():
        ligand = str(pair[ligand_col])
        receptor = str(pair[receptor_col])
        prior = float(pair["prior_confidence"])
        if ligand not in topology.index or receptor not in topology.index:
            continue

        sender_anchor = (1.0 - topology.loc[ligand].reindex(celltypes).fillna(1.0)).clip(lower=0.0, upper=1.0)
        receiver_anchor = (1.0 - topology.loc[receptor].reindex(celltypes).fillna(1.0)).clip(lower=0.0, upper=1.0)
        sender_expr = (
            expression_support.loc[ligand].reindex(celltypes).fillna(0.0)
            if ligand in expression_support.index
            else pd.Series(0.0, index=celltypes, dtype=float)
        )
        receiver_expr = (
            expression_support.loc[receptor].reindex(celltypes).fillna(0.0)
            if receptor in expression_support.index
            else pd.Series(0.0, index=celltypes, dtype=float)
        )

        contact_parts = compute_local_contact_matrix(
            reference,
            expression[ligand] if ligand in expression.columns else pd.Series(0.0, index=reference.index),
            expression[receptor] if receptor in expression.columns else pd.Series(0.0, index=reference.index),
            neighbor_index,
            celltype_col="celltype",
            min_cross_edges=min_cross_edges,
            contact_expr_threshold=contact_expr_threshold,
        )
        local_contact = contact_parts["local_contact"].reindex(index=celltypes, columns=celltypes, fill_value=0.0)
        contact_strength_raw = contact_parts["contact_strength_raw"].reindex(index=celltypes, columns=celltypes, fill_value=0.0)
        contact_strength_normalized = contact_parts["contact_strength_normalized"].reindex(
            index=celltypes,
            columns=celltypes,
            fill_value=0.0,
        )
        contact_coverage = contact_parts["contact_coverage"].reindex(index=celltypes, columns=celltypes, fill_value=0.0)
        cross_edge_count = contact_parts["cross_edge_count"].reindex(index=celltypes, columns=celltypes, fill_value=0)

        for sender in celltypes:
            for receiver in celltypes:
                bridge = 1.0 - float(structure_map_resolved.loc[sender, receiver])
                score_rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_celltype": sender,
                        "receiver_celltype": receiver,
                        "anchor_source_ligand": anchor_sources.get(ligand, "recompute"),
                        "anchor_source_receptor": anchor_sources.get(receptor, "recompute"),
                        "structure_map_source": structure_map_source,
                        "sender_anchor": float(sender_anchor.loc[sender]),
                        "receiver_anchor": float(receiver_anchor.loc[receiver]),
                        "structure_bridge": float(bridge),
                        "sender_expr": float(sender_expr.loc[sender]),
                        "receiver_expr": float(receiver_expr.loc[receiver]),
                        "local_contact": float(local_contact.loc[sender, receiver]),
                        "contact_strength_raw": float(contact_strength_raw.loc[sender, receiver]),
                        "contact_strength_normalized": float(contact_strength_normalized.loc[sender, receiver]),
                        "contact_coverage": float(contact_coverage.loc[sender, receiver]),
                        "cross_edge_count": int(cross_edge_count.loc[sender, receiver]),
                        "prior_confidence": prior,
                        "LR_score": float(
                            geometric_mean(
                                [
                                    float(sender_anchor.loc[sender]),
                                    float(receiver_anchor.loc[receiver]),
                                    float(bridge),
                                    float(sender_expr.loc[sender]),
                                    float(receiver_expr.loc[receiver]),
                                    float(local_contact.loc[sender, receiver]),
                                ]
                            )
                            * prior
                        ),
                    }
                )

    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        scores = scores.sort_values("LR_score", ascending=False).reset_index(drop=True)
    component_diagnostics = scores.copy()

    out_dir = ensure_output_dir(output_dir)
    figures_dir = ensure_figures_dir(output_dir)
    output_files: dict[str, Any] = {}

    if out_dir is not None:
        ligand_path = out_dir / "ligand_to_cell.csv"
        receptor_path = out_dir / "receptor_to_cell.csv"
        scores_path = out_dir / "lr_sender_receiver_scores.csv"
        diagnostics_path = out_dir / "lr_component_diagnostics.csv"
        ligand_to_cell.to_csv(ligand_path)
        receptor_to_cell.to_csv(receptor_path)
        scores.to_csv(scores_path, index=False)
        component_diagnostics.to_csv(diagnostics_path, index=False)
        output_files["ligand_to_cell"] = str(ligand_path)
        output_files["receptor_to_cell"] = str(receptor_path)
        output_files["lr_sender_receiver_scores"] = str(scores_path)
        output_files["lr_component_diagnostics"] = str(diagnostics_path)

        if export_figures and figures_dir is not None and not scores.empty:
            summary = scores.copy()
            summary["lr_pair"] = summary["ligand"] + "→" + summary["receptor"]
            summary["sender_receiver"] = summary["sender_celltype"] + "→" + summary["receiver_celltype"]
            top_pairs = (
                summary.groupby("lr_pair")["LR_score"].max().sort_values(ascending=False).head(int(top_n_pairs)).index.tolist()
            )
            summary_matrix = summary.loc[summary["lr_pair"].isin(top_pairs)].pivot_table(
                index="lr_pair",
                columns="sender_receiver",
                values="LR_score",
                aggfunc="max",
                fill_value=0.0,
            )
            if not summary_matrix.empty:
                output_files["lr_summary_heatmap"] = save_matrix_heatmap(
                    summary_matrix,
                    title="Ligand-receptor topology summary",
                    output_prefix=figures_dir / "lr_summary_heatmap",
                    cmap="magma",
                )

            best = scores.iloc[0]
            best_ligand = str(best["ligand"])
            best_receptor = str(best["receptor"])
            best_sender = str(best["sender_celltype"])
            best_receiver = str(best["receiver_celltype"])
            ligand_cell = (
                winsorized_normalize_series(expression[best_ligand])
                if best_ligand in expression.columns
                else pd.Series(0.0, index=reference.index, dtype=float)
            )
            receptor_cell = (
                winsorized_normalize_series(expression[best_receptor])
                if best_receptor in expression.columns
                else pd.Series(0.0, index=reference.index, dtype=float)
            )
            sender_threshold = float(ligand_cell.quantile(float(hotspot_quantile))) if len(ligand_cell) else 0.0
            receiver_threshold = float(receptor_cell.quantile(float(hotspot_quantile))) if len(receptor_cell) else 0.0
            sender_mask = (reference["celltype"] == best_sender) & (ligand_cell >= sender_threshold)
            receiver_mask = (reference["celltype"] == best_receiver) & (receptor_cell >= receiver_threshold)

            hotspot_table = prepare_hotspot_table(
                reference,
                sender_mask=sender_mask,
                receiver_mask=receiver_mask,
                sender_score=ligand_cell,
                receiver_score=receptor_cell,
                ligand=best_ligand,
                receptor=best_receptor,
                sender_celltype=best_sender,
                receiver_celltype=best_receiver,
                x_col=x_col,
                y_col=y_col,
            )
            hotspot_csv = figures_dir / "lr_hotspot_cells.csv"
            hotspot_table.to_csv(hotspot_csv, index=False)
            output_files["lr_hotspot_cells_csv"] = str(hotspot_csv)
            hotspot_parquet = figures_dir / "lr_hotspot_cells.parquet"
            if safe_to_parquet(hotspot_table, hotspot_parquet):
                output_files["lr_hotspot_cells_parquet"] = str(hotspot_parquet)
            output_files["lr_hotspot_overlay"] = save_hotspot_overlay(
                reference,
                x_col=x_col,
                y_col=y_col,
                sender_mask=sender_mask,
                receiver_mask=receiver_mask,
                sender_score=ligand_cell,
                receiver_score=receptor_cell,
                title=f"{best_ligand}→{best_receptor} hotspot ({best_sender}→{best_receiver})",
                output_prefix=figures_dir / "lr_hotspot_overlay",
            )

    return {
        "ligand_to_cell": ligand_to_cell,
        "receptor_to_cell": receptor_to_cell,
        "structure_map": structure_map_resolved,
        "scores": scores,
        "component_diagnostics": component_diagnostics,
        "anchor_sources": anchor_sources,
        "files": output_files,
    }


__all__ = ["ligand_receptor_topology_analysis"]
