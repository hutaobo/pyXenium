from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    normalize_interaction_prior,
    prepare_hotspot_table,
    resolve_gene_topology_anchors,
    save_hotspot_overlay,
    save_matrix_heatmap,
    safe_to_parquet,
    summarize_expression_by_celltype,
    winsorized_normalize_series,
)

_SECRETED_LIGAND_PREFIXES = ("CCL", "CXCL", "IL", "TGFB", "VEGF", "FGF", "PDGF", "CSF", "BMP", "WNT")
_CONTACT_GENES = {
    "CD2",
    "CD48",
    "CDH1",
    "DLL1",
    "DLL4",
    "EFNB2",
    "ICAM1",
    "JAG1",
    "NOTCH1",
    "NOTCH2",
    "NOTCH3",
    "PECAM1",
    "SELP",
    "VCAM1",
}
_ECM_LIGAND_PREFIXES = ("COL", "FN", "LAM", "MMP", "THBS")
_ECM_GENES = {"HSPG2", "MMRN2", "VWF", "CD93", "ITGA9", "ITGB1", "ITGB2", "LRP1"}
_VALID_INTERACTION_MODES = {
    "secreted/paracrine",
    "juxtacrine/contact",
    "ECM-adhesion",
    "scavenger/receptor",
    "shared activation state",
    "non-classic resource axis",
    "cci_resource_axis",
}


def _coerce_interaction_mode(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() in {"nan", "none", "null"}:
        return "cci_resource_axis"
    normalized = text.replace("_", " ").replace("-", " ").casefold()
    if "secret" in normalized or "paracrine" in normalized or "diffus" in normalized:
        return "secreted/paracrine"
    if "juxta" in normalized or "contact" in normalized or "adhesion" in normalized and "ecm" not in normalized:
        return "juxtacrine/contact"
    if "ecm" in normalized or "matrix" in normalized:
        return "ECM-adhesion"
    if "scavenger" in normalized:
        return "scavenger/receptor"
    if "shared" in normalized or "activation" in normalized:
        return "shared activation state"
    if "non" in normalized and "classic" in normalized:
        return "non-classic resource axis"
    return text if text in _VALID_INTERACTION_MODES else "cci_resource_axis"


def _infer_interaction_mode(ligand: str, receptor: str, pair: pd.Series, interaction_mode_col: str | None) -> str:
    if interaction_mode_col and interaction_mode_col in pair.index:
        mode = _coerce_interaction_mode(pair[interaction_mode_col])
        if mode != "cci_resource_axis":
            return mode
    for column in ("interaction_mode", "mode", "annotation_strategy", "category"):
        if column in pair.index:
            mode = _coerce_interaction_mode(pair[column])
            if mode != "cci_resource_axis":
                return mode

    ligand_upper = ligand.upper()
    receptor_upper = receptor.upper()
    if ligand_upper.startswith(_SECRETED_LIGAND_PREFIXES):
        return "secreted/paracrine"
    if ligand_upper.startswith(_ECM_LIGAND_PREFIXES) or ligand_upper in _ECM_GENES or receptor_upper in _ECM_GENES:
        return "ECM-adhesion"
    if ligand_upper in _CONTACT_GENES or receptor_upper in _CONTACT_GENES:
        return "juxtacrine/contact"
    if receptor_upper in {"LRP1", "LRP2", "CD36", "SCARB1"}:
        return "scavenger/receptor"
    return "cci_resource_axis"


def _resolve_downstream_targets(
    downstream_targets: Mapping[str, Sequence[str]] | pd.DataFrame | None,
    *,
    ligand_col: str,
    receptor_col: str,
) -> dict[tuple[str, str], list[str]]:
    if downstream_targets is None:
        return {}
    if isinstance(downstream_targets, pd.DataFrame):
        target_col = next((col for col in ("target", "target_gene", "gene") if col in downstream_targets.columns), None)
        if target_col is None:
            raise ValueError("downstream_targets DataFrame must include a target, target_gene, or gene column.")
        if ligand_col not in downstream_targets.columns or receptor_col not in downstream_targets.columns:
            raise ValueError(f"downstream_targets DataFrame must include {ligand_col!r} and {receptor_col!r}.")
        resolved: dict[tuple[str, str], list[str]] = {}
        for _, row in downstream_targets.iterrows():
            key = (str(row[ligand_col]), str(row[receptor_col]))
            resolved.setdefault(key, []).append(str(row[target_col]))
        return {key: list(dict.fromkeys(values)) for key, values in resolved.items()}

    resolved = {}
    for key, values in downstream_targets.items():
        text = str(key)
        if "^" in text:
            ligand, receptor = text.split("^", 1)
        elif "|" in text:
            ligand, receptor = text.split("|", 1)
        elif "-" in text:
            ligand, receptor = text.split("-", 1)
        else:
            continue
        resolved[(ligand.strip(), receptor.strip())] = list(dict.fromkeys(str(value) for value in values))
    return resolved


def _receiver_response_score(
    expression_support: pd.DataFrame,
    *,
    targets: Sequence[str],
    receiver: str,
) -> tuple[float, int]:
    retained = [target for target in targets if target in expression_support.index]
    if not retained:
        return float("nan"), 0
    values = expression_support.loc[retained, receiver] if receiver in expression_support.columns else pd.Series(dtype=float)
    if values.empty:
        return float("nan"), 0
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).mean()), len(retained)


def _distance_component(
    *,
    mode: str,
    local_contact: float,
    structure_bridge: float,
    sender_expr: float,
    receiver_expr: float,
    distance_kernel: str,
) -> float:
    if distance_kernel in {"contact", "local_contact"}:
        return float(local_contact)
    if distance_kernel != "mechanism_aware":
        raise ValueError("distance_kernel must be one of: contact, local_contact, mechanism_aware.")

    expression_pair = float(np.sqrt(max(sender_expr, 0.0) * max(receiver_expr, 0.0)))
    if mode == "secreted/paracrine":
        return float(max(local_contact, 0.50 * structure_bridge * expression_pair))
    if mode in {"ECM-adhesion", "scavenger/receptor"}:
        return float(max(local_contact, 0.65 * structure_bridge * expression_pair))
    if mode == "shared activation state":
        return float(max(local_contact, 0.35 * expression_pair))
    return float(local_contact)


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=pvalues.index, dtype=float)
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for pos in range(n - 1, -1, -1):
        running = min(running, ranked[pos] * n / float(pos + 1))
        adjusted[pos] = running
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _calibrate_cci_scores(
    scores: pd.DataFrame,
    *,
    null_model: str,
    n_permutations: int,
    random_state: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if null_model in {None, "none", "off"} or scores.empty:
        calibrated = scores.copy()
        calibrated["cci_pvalue"] = np.nan
        calibrated["cci_fdr"] = np.nan
        calibrated["null_z"] = np.nan
        return calibrated, pd.DataFrame()
    if null_model != "component_shuffle":
        raise ValueError("null_model must be one of: none, component_shuffle.")
    if int(n_permutations) <= 0:
        raise ValueError("n_permutations must be positive when null_model='component_shuffle'.")

    rng = np.random.default_rng(random_state)
    component_cols = [
        "sender_anchor",
        "receiver_anchor",
        "structure_bridge",
        "sender_expr",
        "receiver_expr",
        "distance_kernel_component",
    ]
    calibrated = scores.copy()
    pvalues = pd.Series(np.nan, index=calibrated.index, dtype=float)
    null_z = pd.Series(np.nan, index=calibrated.index, dtype=float)
    null_rows: list[dict[str, Any]] = []

    for mode, group in calibrated.groupby("interaction_mode", dropna=False):
        if group.empty:
            continue
        component_matrix = group[component_cols].to_numpy(dtype=float)
        priors = group["prior_confidence"].to_numpy(dtype=float)
        observed = group["CCI_score"].to_numpy(dtype=float)
        null_scores = np.empty((int(n_permutations), len(group)), dtype=float)
        for perm_idx in range(int(n_permutations)):
            shuffled = component_matrix.copy()
            for col_idx in range(shuffled.shape[1]):
                shuffled[:, col_idx] = rng.permutation(shuffled[:, col_idx])
            null_scores[perm_idx] = np.exp(np.mean(np.log(np.clip(shuffled, 0.0, None) + 1e-8), axis=1)) * priors
        flattened = null_scores.ravel()
        p = (1.0 + (flattened[None, :] >= observed[:, None]).sum(axis=1)) / (float(flattened.size) + 1.0)
        std = float(np.std(flattened))
        z = (observed - float(np.mean(flattened))) / std if std > 0 else np.full_like(observed, np.nan)
        pvalues.loc[group.index] = p
        null_z.loc[group.index] = z
        null_rows.append(
            {
                "interaction_mode": mode,
                "null_model": null_model,
                "n_permutations": int(n_permutations),
                "n_observed_rows": int(len(group)),
                "null_mean": float(np.mean(flattened)),
                "null_sd": std,
                "null_p95": float(np.quantile(flattened, 0.95)),
                "null_p99": float(np.quantile(flattened, 0.99)),
            }
        )

    calibrated["cci_pvalue"] = pvalues
    calibrated["cci_fdr"] = _benjamini_hochberg(pvalues)
    calibrated["null_z"] = null_z
    return calibrated, pd.DataFrame(null_rows)


def cci_topology_analysis(
    *,
    reference_df: Optional[pd.DataFrame] = None,
    expression_df: Optional[pd.DataFrame] = None,
    interaction_pairs: pd.DataFrame,
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
    null_model: str = "none",
    n_permutations: int = 100,
    random_state: int | None = 0,
    interaction_mode_col: str | None = "interaction_mode",
    distance_kernel: str = "contact",
    downstream_targets: Mapping[str, Sequence[str]] | pd.DataFrame | None = None,
    return_hotspots: bool = True,
    export_figures: bool = True,
    use_raw: bool = False,
) -> dict[str, Any]:
    """
    Score cell-cell interaction hypotheses using topology anchors, sender/receiver
    expression support, and de-saturated local contact structure.
    """

    if expression_support_mode != "pseudobulk_detection":
        raise ValueError("expression_support_mode currently supports only 'pseudobulk_detection'.")
    if contact_mode != "strength_coverage":
        raise ValueError("contact_mode currently supports only 'strength_coverage'.")
    if ligand_col not in interaction_pairs.columns or receptor_col not in interaction_pairs.columns:
        raise ValueError(f"interaction_pairs must contain {ligand_col!r} and {receptor_col!r}.")

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

    pairs = interaction_pairs.copy()
    pairs[ligand_col] = pairs[ligand_col].astype(str)
    pairs[receptor_col] = pairs[receptor_col].astype(str)
    pairs["prior_confidence"] = normalize_interaction_prior(pairs, prior_col)
    target_lookup = _resolve_downstream_targets(downstream_targets, ligand_col=ligand_col, receptor_col=receptor_col)

    downstream_gene_list = [target for targets in target_lookup.values() for target in targets]
    genes = list(dict.fromkeys(pairs[ligand_col].tolist() + pairs[receptor_col].tolist() + downstream_gene_list))
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
        interaction_mode = _infer_interaction_mode(ligand, receptor, pair, interaction_mode_col)
        target_genes = target_lookup.get((ligand, receptor), [])
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
                sender_anchor_value = float(sender_anchor.loc[sender])
                receiver_anchor_value = float(receiver_anchor.loc[receiver])
                sender_expr_value = float(sender_expr.loc[sender])
                receiver_expr_value = float(receiver_expr.loc[receiver])
                local_contact_value = float(local_contact.loc[sender, receiver])
                distance_component = _distance_component(
                    mode=interaction_mode,
                    local_contact=local_contact_value,
                    structure_bridge=float(bridge),
                    sender_expr=sender_expr_value,
                    receiver_expr=receiver_expr_value,
                    distance_kernel=distance_kernel,
                )
                receiver_response, downstream_target_count = _receiver_response_score(
                    expression_support,
                    targets=target_genes,
                    receiver=receiver,
                )
                score_rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_celltype": sender,
                        "receiver_celltype": receiver,
                        "interaction_mode": interaction_mode,
                        "distance_kernel": distance_kernel,
                        "anchor_source_ligand": anchor_sources.get(ligand, "recompute"),
                        "anchor_source_receptor": anchor_sources.get(receptor, "recompute"),
                        "structure_map_source": structure_map_source,
                        "sender_anchor": sender_anchor_value,
                        "receiver_anchor": receiver_anchor_value,
                        "structure_bridge": float(bridge),
                        "sender_expr": sender_expr_value,
                        "receiver_expr": receiver_expr_value,
                        "local_contact": local_contact_value,
                        "distance_kernel_component": distance_component,
                        "contact_strength_raw": float(contact_strength_raw.loc[sender, receiver]),
                        "contact_strength_normalized": float(contact_strength_normalized.loc[sender, receiver]),
                        "contact_coverage": float(contact_coverage.loc[sender, receiver]),
                        "cross_edge_count": int(cross_edge_count.loc[sender, receiver]),
                        "receiver_response_score": receiver_response,
                        "downstream_target_count": int(downstream_target_count),
                        "prior_confidence": prior,
                        "CCI_score": float(
                            geometric_mean(
                                [
                                    sender_anchor_value,
                                    receiver_anchor_value,
                                    float(bridge),
                                    sender_expr_value,
                                    receiver_expr_value,
                                    distance_component,
                                ]
                            )
                            * prior
                        ),
                    }
                )

    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        scores = scores.sort_values("CCI_score", ascending=False).reset_index(drop=True)
        scores, null_summary = _calibrate_cci_scores(
            scores,
            null_model=null_model,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        scores = scores.sort_values("CCI_score", ascending=False).reset_index(drop=True)
    else:
        null_summary = pd.DataFrame()
    component_diagnostics = scores.copy()

    out_dir = ensure_output_dir(output_dir)
    figures_dir = ensure_figures_dir(output_dir)
    output_files: dict[str, Any] = {}

    hotspot_table = pd.DataFrame()
    if return_hotspots and not scores.empty:
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

    if out_dir is not None:
        ligand_path = out_dir / "ligand_to_cell.csv"
        receptor_path = out_dir / "receptor_to_cell.csv"
        scores_path = out_dir / "cci_sender_receiver_scores.csv"
        diagnostics_path = out_dir / "cci_component_diagnostics.csv"
        null_summary_path = out_dir / "cci_null_calibration_summary.csv"
        ligand_to_cell.to_csv(ligand_path)
        receptor_to_cell.to_csv(receptor_path)
        scores.to_csv(scores_path, index=False)
        component_diagnostics.to_csv(diagnostics_path, index=False)
        null_summary.to_csv(null_summary_path, index=False)
        output_files["ligand_to_cell"] = str(ligand_path)
        output_files["receptor_to_cell"] = str(receptor_path)
        output_files["cci_sender_receiver_scores"] = str(scores_path)
        output_files["cci_component_diagnostics"] = str(diagnostics_path)
        output_files["cci_null_calibration_summary"] = str(null_summary_path)

        if export_figures and figures_dir is not None and not scores.empty:
            summary = scores.copy()
            summary["interaction_pair"] = summary["ligand"] + "→" + summary["receptor"]
            summary["sender_receiver"] = summary["sender_celltype"] + "→" + summary["receiver_celltype"]
            top_pairs = (
                summary.groupby("interaction_pair")["CCI_score"].max().sort_values(ascending=False).head(int(top_n_pairs)).index.tolist()
            )
            summary_matrix = summary.loc[summary["interaction_pair"].isin(top_pairs)].pivot_table(
                index="interaction_pair",
                columns="sender_receiver",
                values="CCI_score",
                aggfunc="max",
                fill_value=0.0,
            )
            if not summary_matrix.empty:
                output_files["cci_summary_heatmap"] = save_matrix_heatmap(
                    summary_matrix,
                    title="Cell-cell interaction topology summary",
                    output_prefix=figures_dir / "cci_summary_heatmap",
                    cmap="magma",
                )

            if return_hotspots:
                hotspot_csv = figures_dir / "cci_hotspot_cells.csv"
                hotspot_table.to_csv(hotspot_csv, index=False)
                output_files["cci_hotspot_cells_csv"] = str(hotspot_csv)
                hotspot_parquet = figures_dir / "cci_hotspot_cells.parquet"
                if safe_to_parquet(hotspot_table, hotspot_parquet):
                    output_files["cci_hotspot_cells_parquet"] = str(hotspot_parquet)
                output_files["cci_hotspot_overlay"] = save_hotspot_overlay(
                    reference,
                    x_col=x_col,
                    y_col=y_col,
                    sender_mask=sender_mask,
                    receiver_mask=receiver_mask,
                    sender_score=ligand_cell,
                    receiver_score=receptor_cell,
                    title=f"{best_ligand}→{best_receptor} hotspot ({best_sender}→{best_receiver})",
                    output_prefix=figures_dir / "cci_hotspot_overlay",
                )

    return {
        "ligand_to_cell": ligand_to_cell,
        "receptor_to_cell": receptor_to_cell,
        "structure_map": structure_map_resolved,
        "scores": scores,
        "component_diagnostics": component_diagnostics,
        "null_calibration_summary": null_summary,
        "hotspot_table": hotspot_table,
        "anchor_sources": anchor_sources,
        "files": output_files,
    }


__all__ = ["cci_topology_analysis"]
