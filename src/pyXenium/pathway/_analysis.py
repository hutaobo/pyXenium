from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pyXenium._topology_core import (
    aggregate_weighted_values,
    build_neighbor_index,
    coerce_expression_df,
    coerce_reference_df,
    compute_entity_structuremap,
    compute_entity_to_cell_topology,
    ensure_figures_dir,
    ensure_output_dir,
    normalize_frame_columns,
    resolve_gene_topology_anchors,
    robust_scale_columns,
    safe_row_cophenetic,
    save_matrix_heatmap,
    smooth_matrix_by_neighbors,
    winsorized_normalize_series,
)


def _standardize_pathway_definitions(pathway_definitions: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(pathway_definitions, pd.DataFrame):
        required = {"pathway", "gene"}
        missing = required.difference(pathway_definitions.columns)
        if missing:
            raise ValueError(f"pathway_definitions DataFrame is missing required columns: {sorted(missing)}")
        out = pathway_definitions.copy()
        if "weight" not in out.columns:
            out["weight"] = 1.0
        out["pathway"] = out["pathway"].astype(str)
        out["gene"] = out["gene"].astype(str)
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0).astype(float)
        return out[["pathway", "gene", "weight"]]

    rows: list[dict[str, Any]] = []
    for pathway, genes in pathway_definitions.items():
        if isinstance(genes, Mapping):
            for gene, weight in genes.items():
                rows.append({"pathway": str(pathway), "gene": str(gene), "weight": float(weight)})
        else:
            for gene in genes:
                rows.append({"pathway": str(pathway), "gene": str(gene), "weight": 1.0})
    return pd.DataFrame(rows, columns=["pathway", "gene", "weight"])


def compute_pathway_activity_matrix(
    expression_df: pd.DataFrame,
    pathway_definitions: Mapping[str, Any] | pd.DataFrame,
    *,
    method: str = "weighted_sum",
    normalize: bool = True,
) -> pd.DataFrame:
    pathway_table = _standardize_pathway_definitions(pathway_definitions)
    expression = expression_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    activity = pd.DataFrame(index=expression.index)

    valid_methods = {"rank_mean", "ucell", "aucell", "weighted_sum", "progeny"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of: {sorted(valid_methods)}")

    if method in {"rank_mean", "ucell", "aucell"}:
        ranked = expression.rank(axis=1, method="average", ascending=True, pct=True)
        for pathway, group in pathway_table.groupby("pathway", sort=False):
            present = [gene for gene in group["gene"] if gene in ranked.columns]
            if not present:
                activity[pathway] = 0.0
                continue
            weights = group.set_index("gene").loc[present, "weight"].astype(float)
            values = ranked[present].to_numpy(dtype=float)
            activity[pathway] = np.average(values, axis=1, weights=np.abs(weights.to_numpy(dtype=float)))
    else:
        for pathway, group in pathway_table.groupby("pathway", sort=False):
            present = [gene for gene in group["gene"] if gene in expression.columns]
            if not present:
                activity[pathway] = 0.0
                continue
            weights = group.set_index("gene").loc[present, "weight"].astype(float)
            denominator = float(np.sum(np.abs(weights.to_numpy(dtype=float)))) or 1.0
            activity[pathway] = expression[present].to_numpy(dtype=float) @ weights.to_numpy(dtype=float) / denominator

    if normalize:
        activity = normalize_frame_columns(activity)
    return activity


def _aggregate_pathway_gene_topology(
    gene_topology: pd.DataFrame,
    pathway_table: pd.DataFrame,
    *,
    aggregate: str = "weighted_median",
) -> pd.DataFrame:
    celltypes = gene_topology.columns.astype(str).tolist()
    rows: list[pd.Series] = []
    for pathway, group in pathway_table.groupby("pathway", sort=False):
        pathway_genes = [gene for gene in group["gene"].astype(str) if gene in gene_topology.index]
        weights = group.set_index("gene").reindex(pathway_genes)["weight"].fillna(1.0).astype(float)
        if not pathway_genes:
            rows.append(pd.Series(np.nan, index=celltypes, name=str(pathway)))
            continue
        aggregated = {
            celltype: aggregate_weighted_values(
                gene_topology.loc[pathway_genes, celltype].to_numpy(dtype=float),
                weights.to_numpy(dtype=float),
                method=aggregate,
            )
            for celltype in celltypes
        }
        rows.append(pd.Series(aggregated, name=str(pathway)))
    out = pd.DataFrame(rows, columns=celltypes)
    out.index.name = "pathway"
    out.columns.name = "celltype"
    return out


def _build_pathway_activity_points(
    reference_df: pd.DataFrame,
    pathway_activity: pd.DataFrame,
    *,
    cell_id_col: str,
    x_col: str,
    y_col: str,
    activity_mode: str,
    activity_threshold_schedule: Sequence[float],
    min_activity_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for pathway in pathway_activity.columns:
        values = pd.to_numeric(pathway_activity[pathway], errors="coerce").fillna(0.0).astype(float)
        retained_quantile: Optional[float] = None
        retained_mask = pd.Series(False, index=values.index)
        positive_values = values.loc[values > 0]
        quantile_pool = positive_values if not positive_values.empty else values

        for quantile in activity_threshold_schedule:
            threshold = float(quantile_pool.quantile(float(quantile)))
            if positive_values.empty:
                mask = values >= threshold
            else:
                mask = (values >= threshold) & (values > 0)
            if int(mask.sum()) >= int(min_activity_cells):
                retained_mask = mask
                retained_quantile = float(quantile)
                break

        if retained_quantile is None:
            top_n = min(int(min_activity_cells), len(values))
            if top_n > 0:
                top_index = values.nlargest(top_n).index
                retained_mask.loc[top_index] = True
            retained_quantile = float(activity_threshold_schedule[-1]) if activity_threshold_schedule else 0.5

        retained_values = values.loc[retained_mask]
        if not retained_values.empty:
            points = reference_df.loc[retained_mask, [cell_id_col, x_col, y_col, "celltype"]].copy()
            points["entity"] = str(pathway)
            points["weight"] = retained_values.to_numpy(dtype=float)
            records.append(points)

        diagnostics.append(
            {
                "pathway": str(pathway),
                "retained_cell_count": int(retained_mask.sum()),
                "retained_quantile": float(retained_quantile),
                "activity_mode": str(activity_mode),
            }
        )

    if records:
        entity_points = pd.concat(records, ignore_index=True)
    else:
        entity_points = pd.DataFrame(columns=[cell_id_col, x_col, y_col, "celltype", "entity", "weight"])
    diagnostics_df = pd.DataFrame(diagnostics)
    return entity_points, diagnostics_df


def _pathway_mode_summary(pathway_to_cell: pd.DataFrame, *, mode_name: str) -> pd.DataFrame:
    if pathway_to_cell.empty:
        return pd.DataFrame(columns=["pathway", f"{mode_name}_best_celltype", f"{mode_name}_best_distance"])
    return pd.DataFrame(
        {
            "pathway": pathway_to_cell.index.astype(str),
            f"{mode_name}_best_celltype": pathway_to_cell.idxmin(axis=1).astype(str).to_numpy(),
            f"{mode_name}_best_distance": pathway_to_cell.min(axis=1).to_numpy(dtype=float),
        }
    )


def pathway_topology_analysis(
    *,
    pathway_definitions: Mapping[str, Any] | pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    expression_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str | Path] = None,
    adata: Any = None,
    tbc_results: Optional[str | Path] = None,
    t_and_c_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "cluster",
    cell_id_col: str = "cell_id",
    x_col: str = "x",
    y_col: str = "y",
    celltype_col: str = "celltype",
    scoring_method: str = "weighted_sum",
    view: str = "intrinsic",
    structure_map: Optional[pd.DataFrame] = None,
    structure_map_df: Optional[pd.DataFrame] = None,
    anchor_mode: str = "precomputed",
    pathway_modes: Sequence[str] = ("gene_topology_aggregate", "activity_point_cloud"),
    primary_pathway_mode: str = "gene_topology_aggregate",
    pathway_aggregate: str = "weighted_median",
    activity_threshold_schedule: Sequence[float] = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50),
    min_activity_cells: int = 50,
    entity_min_weight: float = 0.0,
    k_neighbors: int = 8,
    radius: Optional[float] = None,
    topology_method: str = "average",
    hotspot_quantile: float = 0.9,
    export_figures: bool = True,
    use_raw: bool = False,
) -> dict[str, Any]:
    """
    Compute pathway-level topology in two coordinated views:
    gene-topology aggregate and activity point cloud.
    """

    if view not in {"intrinsic", "niche_smoothed"}:
        raise ValueError("view must be either 'intrinsic' or 'niche_smoothed'.")
    valid_modes = {"gene_topology_aggregate", "activity_point_cloud"}
    if any(mode not in valid_modes for mode in pathway_modes):
        raise ValueError("pathway_modes must contain only 'gene_topology_aggregate' and/or 'activity_point_cloud'.")
    if primary_pathway_mode not in valid_modes:
        raise ValueError("primary_pathway_mode must be 'gene_topology_aggregate' or 'activity_point_cloud'.")
    if primary_pathway_mode not in set(pathway_modes):
        raise ValueError("primary_pathway_mode must also be present in pathway_modes.")

    pathway_table = _standardize_pathway_definitions(pathway_definitions)
    genes = pathway_table["gene"].drop_duplicates().tolist()
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

    expression = coerce_expression_df(
        reference,
        expression_df,
        adata=adata,
        genes=genes,
        cell_id_col=cell_id_col,
        use_raw=use_raw,
    )
    expression.index = reference.index

    pathway_activity = compute_pathway_activity_matrix(expression, pathway_table, method=scoring_method, normalize=False)
    pathway_activity = robust_scale_columns(pathway_activity)
    if view == "niche_smoothed":
        neighbor_index = build_neighbor_index(reference, x_col=x_col, y_col=y_col, k_neighbors=k_neighbors, radius=radius)
        pathway_activity = robust_scale_columns(smooth_matrix_by_neighbors(pathway_activity, neighbor_index, include_self=True))

    gene_topology, anchor_sources, _, structure_map_source = resolve_gene_topology_anchors(
        reference,
        expression,
        genes,
        tbc_results=tbc_results,
        t_and_c_df=t_and_c_df,
        structure_map=structure_map,
        structure_map_df=structure_map_df,
        anchor_mode=anchor_mode,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        entity_min_weight=entity_min_weight,
        topology_method=topology_method,
    )

    gene_topology_aggregate = _aggregate_pathway_gene_topology(
        gene_topology,
        pathway_table,
        aggregate=pathway_aggregate,
    )
    gene_topology_structuremap = safe_row_cophenetic(gene_topology_aggregate, method=topology_method)

    activity_entity_points, activity_diagnostics = _build_pathway_activity_points(
        reference,
        pathway_activity,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        activity_mode=view,
        activity_threshold_schedule=activity_threshold_schedule,
        min_activity_cells=min_activity_cells,
    )
    pathway_activity_to_cell = compute_entity_to_cell_topology(
        reference,
        activity_entity_points,
        x_col=x_col,
        y_col=y_col,
        celltype_col="celltype",
        entity_col="entity",
        weight_col="weight",
        min_weight=entity_min_weight,
        method=topology_method,
    )
    pathway_activity_structuremap = compute_entity_structuremap(
        activity_entity_points,
        x_col=x_col,
        y_col=y_col,
        entity_col="entity",
        weight_col="weight",
        min_weight=entity_min_weight,
        method=topology_method,
    )

    pathway_to_cell = (
        gene_topology_aggregate if primary_pathway_mode == "gene_topology_aggregate" else pathway_activity_to_cell
    )
    pathway_structuremap = (
        gene_topology_structuremap if primary_pathway_mode == "gene_topology_aggregate" else pathway_activity_structuremap
    )

    mode_comparison = _pathway_mode_summary(gene_topology_aggregate, mode_name="gene_topology").merge(
        _pathway_mode_summary(pathway_activity_to_cell, mode_name="activity_point_cloud"),
        on="pathway",
        how="outer",
    )
    mode_comparison = mode_comparison.merge(activity_diagnostics, on="pathway", how="left")

    out_dir = ensure_output_dir(output_dir)
    figures_dir = ensure_figures_dir(output_dir)
    output_files: dict[str, Any] = {}

    if out_dir is not None:
        output_paths = {
            "pathway_to_cell": out_dir / "pathway_to_cell.csv",
            "pathway_structuremap": out_dir / "pathway_structuremap.csv",
            "pathway_activity_to_cell": out_dir / "pathway_activity_to_cell.csv",
            "pathway_activity_structuremap": out_dir / "pathway_activity_structuremap.csv",
            "pathway_mode_comparison": out_dir / "pathway_mode_comparison.csv",
        }
        pathway_to_cell.to_csv(output_paths["pathway_to_cell"])
        pathway_structuremap.to_csv(output_paths["pathway_structuremap"])
        pathway_activity_to_cell.to_csv(output_paths["pathway_activity_to_cell"])
        pathway_activity_structuremap.to_csv(output_paths["pathway_activity_structuremap"])
        mode_comparison.to_csv(output_paths["pathway_mode_comparison"], index=False)
        output_files.update({name: str(path) for name, path in output_paths.items()})

        if export_figures and figures_dir is not None:
            if not pathway_to_cell.empty:
                output_files["pathway_to_cell_heatmap"] = save_matrix_heatmap(
                    pathway_to_cell,
                    title=f"Pathway-to-cell topology ({primary_pathway_mode})",
                    output_prefix=figures_dir / "pathway_to_cell_heatmap",
                    cmap="viridis",
                )
            if not pathway_activity_to_cell.empty:
                output_files["pathway_activity_to_cell_heatmap"] = save_matrix_heatmap(
                    pathway_activity_to_cell,
                    title="Pathway activity-point-cloud topology",
                    output_prefix=figures_dir / "pathway_activity_to_cell_heatmap",
                    cmap="cividis",
                )

            if not mode_comparison.empty and not pathway_activity.empty:
                best_pathway_row = mode_comparison.sort_values(
                    ["gene_topology_best_distance", "activity_point_cloud_best_distance"],
                    ascending=True,
                    na_position="last",
                ).iloc[0]
                best_pathway = str(best_pathway_row["pathway"])
                pathway_scores = (
                    winsorized_normalize_series(pathway_activity[best_pathway])
                    if best_pathway in pathway_activity.columns
                    else pd.Series(0.0, index=reference.index, dtype=float)
                )
                threshold = float(pathway_scores.quantile(float(hotspot_quantile))) if len(pathway_scores) else 0.0
                hotspot_mask = pathway_scores >= threshold
                hotspot_table = reference.loc[hotspot_mask, ["cell_id", x_col, y_col, "celltype"]].copy()
                hotspot_table["pathway"] = best_pathway
                hotspot_table["score"] = pathway_scores.loc[hotspot_mask].to_numpy(dtype=float)
                hotspot_csv = figures_dir / "pathway_hotspot_cells.csv"
                hotspot_table.to_csv(hotspot_csv, index=False)
                output_files["pathway_hotspot_cells_csv"] = str(hotspot_csv)

                overlay_outputs = []
                for ext in ("png", "pdf"):
                    output_path = figures_dir / f"pathway_hotspot_overlay.{ext}"
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7.0, 7.0))
                    ax.scatter(reference[x_col], reference[y_col], s=8, c="#D8D8D8", alpha=0.35, linewidths=0.0)
                    ax.scatter(
                        reference.loc[hotspot_mask, x_col],
                        reference.loc[hotspot_mask, y_col],
                        s=20 + 40 * pathway_scores.loc[hotspot_mask].to_numpy(dtype=float),
                        c=pathway_scores.loc[hotspot_mask].to_numpy(dtype=float),
                        cmap="YlOrRd",
                        alpha=0.9,
                        linewidths=0.0,
                    )
                    ax.set_title(f"Top pathway hotspot: {best_pathway}")
                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.tight_layout()
                    fig.savefig(output_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    overlay_outputs.append(str(output_path))
                output_files["pathway_hotspot_overlay"] = overlay_outputs

    return {
        "pathway_table": pathway_table,
        "gene_topology": gene_topology,
        "gene_topology_aggregate": gene_topology_aggregate,
        "gene_topology_structuremap": gene_topology_structuremap,
        "pathway_activity": pathway_activity,
        "pathway_activity_points": activity_entity_points,
        "pathway_activity_diagnostics": activity_diagnostics,
        "pathway_activity_to_cell": pathway_activity_to_cell,
        "pathway_activity_structuremap": pathway_activity_structuremap,
        "pathway_to_cell": pathway_to_cell,
        "pathway_structuremap": pathway_structuremap,
        "pathway_mode_comparison": mode_comparison,
        "anchor_sources": anchor_sources,
        "structure_map_source": structure_map_source,
        "files": output_files,
    }


__all__ = ["compute_pathway_activity_matrix", "pathway_topology_analysis"]
