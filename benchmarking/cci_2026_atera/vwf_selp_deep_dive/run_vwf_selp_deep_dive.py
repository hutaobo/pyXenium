from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_ROOT = REPO_ROOT / "benchmarking" / "cci_2026_atera" / "vwf_selp_deep_dive"
TABLE_DIR = WORK_ROOT / "tables"
FIGURE_DIR = WORK_ROOT / "figures"
REPORT_DIR = WORK_ROOT / "reports"

PDC_ROOT = REPO_ROOT / "benchmarking" / "cci_2026_atera" / "pdc_collected" / "pdc_20260426_1327"
PYX_RUN = PDC_ROOT / "runs" / "full_common" / "pyxenium"
PYX_SCORES = PYX_RUN / "pyxenium_scores.tsv"
PYX_STANDARDIZED = PYX_RUN / "pyxenium_standardized.tsv"
FULL_H5AD = PYX_RUN / "input_cache" / "adata_full_from_sparse_bundle.h5ad"

CONTOUR_FEATURES = REPO_ROOT / "manuscript" / "atera_contour_boundary_ecology" / "contour_features.csv"
CONTOUR_PROGRAMS = REPO_ROOT / "manuscript" / "atera_contour_boundary_ecology" / "program_scores.csv"
CONTOUR_HYPOTHESES = REPO_ROOT / "manuscript" / "atera_contour_boundary_ecology" / "hypothesis_ranking.csv"
S1S5_DE = REPO_ROOT / "docs" / "_static" / "tutorials" / "contour_s1_s5" / "s1_s5_transcript_de_global.csv"
MECHANOSTRESS_COUPLING = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "tutorials"
    / "mechanostress_atera_pdc"
    / "distance_expression_coupling.csv"
)

TARGET_LIGAND = "VWF"
TARGET_RECEPTOR = "SELP"
TARGET_SENDER = "Endothelial Cells"
TARGET_RECEIVER = "Endothelial Cells"

COMPONENTS = [
    "sender_anchor",
    "receiver_anchor",
    "structure_bridge",
    "sender_expr",
    "receiver_expr",
    "local_contact",
]

VASCULAR_TARGET_PAIRS = [
    ("VWF", "SELP"),
    ("VWF", "LRP1"),
    ("EFNB2", "PECAM1"),
    ("MMRN2", "CLEC14A"),
    ("COL4A2", "CD93"),
    ("MMRN2", "CD93"),
    ("VWF", "ITGA9"),
    ("VEGFC", "FLT1"),
    ("CD34", "SELP"),
    ("HSPG2", "LRP1"),
    ("MMP2", "PECAM1"),
    ("CXCL12", "ITGA5"),
    ("CCN1", "CAV1"),
]

ENDOTHELIAL_MARKERS = [
    "VWF",
    "SELP",
    "PECAM1",
    "EMCN",
    "CDH5",
    "KDR",
    "FLT1",
    "MMRN2",
    "CLEC14A",
    "EGFL7",
    "COL4A1",
    "COL4A2",
]

CONTAMINATION_MARKERS = ["PPBP", "PF4", "ITGA2B", "GP1BA", "HBB", "HBA1", "HBA2"]

PATHWAY_PANELS = {
    "WPB_EndothelialActivation": ["VWF", "SELP", "CD63", "ANGPT2", "IL6", "CXCL8", "RAB27A", "STXBP5", "PLAT"],
    "VascularIdentity": ["PECAM1", "EMCN", "CDH5", "KDR", "FLT1", "MMRN2", "CLEC14A", "EGFL7"],
    "Hemostasis_Thromboinflammation": ["VWF", "SELP", "THBD", "PLAT", "SERPINE1", "ADAMTS13"],
}


def ensure_dirs() -> None:
    for path in (WORK_ROOT, TABLE_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def geometric_mean(values: Iterable[float], eps: float = 1e-8) -> float:
    array = np.asarray(list(values), dtype=float)
    array = np.clip(array, 0.0, None)
    return float(np.exp(np.mean(np.log(array + eps))))


def normalize01(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    min_value = float(np.nanmin(arr[finite]))
    max_value = float(np.nanmax(arr[finite]))
    if math.isclose(max_value, min_value):
        out = np.zeros_like(arr, dtype=float)
        out[finite] = 1.0 if max_value > 0 else 0.0
        return out
    return np.clip((arr - min_value) / (max_value - min_value), 0.0, 1.0)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix != ".csv" else ",")


def load_scores() -> pd.DataFrame:
    if not PYX_SCORES.exists():
        raise FileNotFoundError(f"Missing pyXenium score table: {PYX_SCORES}")
    scores = pd.read_csv(PYX_SCORES, sep="\t")
    scores["lr_pair"] = scores["ligand"].astype(str) + "-" + scores["receptor"].astype(str)
    return scores


def target_row(scores: pd.DataFrame) -> pd.Series:
    mask = (
        (scores["ligand"] == TARGET_LIGAND)
        & (scores["receptor"] == TARGET_RECEPTOR)
        & (scores["sender_celltype"] == TARGET_SENDER)
        & (scores["receiver_celltype"] == TARGET_RECEIVER)
    )
    matched = scores.loc[mask].copy()
    if matched.empty:
        raise ValueError("Could not find the expected VWF-SELP Endothelial -> Endothelial top hit.")
    return matched.sort_values("CCI_score", ascending=False).iloc[0]


def write_component_tables(scores: pd.DataFrame, row: pd.Series) -> None:
    component_values = {name: float(row[name]) for name in COMPONENTS}
    prior = float(row["prior_confidence"])
    recomputed = geometric_mean(component_values.values()) * prior
    rows = []
    for name, value in component_values.items():
        rows.append(
            {
                "component": name,
                "value": value,
                "geomean_penalty": -math.log(max(value, 0.0) + 1e-8),
                "target_lr": f"{TARGET_LIGAND}-{TARGET_RECEPTOR}",
                "target_sender": TARGET_SENDER,
                "target_receiver": TARGET_RECEIVER,
                "prior_confidence": prior,
                "reported_lr_score": float(row["CCI_score"]),
                "recomputed_lr_score": recomputed,
            }
        )
    component_df = pd.DataFrame(rows)
    component_df.to_csv(TABLE_DIR / "component_decomposition.tsv", sep="\t", index=False)

    sensitivity_rows = []
    base_components = scores[COMPONENTS].clip(lower=0.0).fillna(0.0).to_numpy(dtype=float)
    prior_values = pd.to_numeric(scores["prior_confidence"], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    def rank_target(values: np.ndarray, label: str) -> None:
        adjusted_scores = np.exp(np.mean(np.log(values + 1e-8), axis=1)) * prior_values
        target_index = int(row.name)
        target_score = float(adjusted_scores[target_index])
        rank = int(np.sum(adjusted_scores > target_score) + 1)
        top_index = int(np.argmax(adjusted_scores))
        top = scores.iloc[top_index]
        sensitivity_rows.append(
            {
                "scenario": label,
                "target_score": target_score,
                "target_rank": rank,
                "top_ligand": top["ligand"],
                "top_receptor": top["receptor"],
                "top_sender": top["sender_celltype"],
                "top_receiver": top["receiver_celltype"],
                "top_score": float(adjusted_scores[top_index]),
            }
        )

    rank_target(base_components, "original")
    for idx, component in enumerate(COMPONENTS):
        without = np.delete(base_components, idx, axis=1)
        adjusted_scores = np.exp(np.mean(np.log(without + 1e-8), axis=1)) * prior_values
        target_score = float(adjusted_scores[int(row.name)])
        rank = int(np.sum(adjusted_scores > target_score) + 1)
        top_index = int(np.argmax(adjusted_scores))
        top = scores.iloc[top_index]
        sensitivity_rows.append(
            {
                "scenario": f"remove_{component}",
                "target_score": target_score,
                "target_rank": rank,
                "top_ligand": top["ligand"],
                "top_receptor": top["receptor"],
                "top_sender": top["sender_celltype"],
                "top_receiver": top["receiver_celltype"],
                "top_score": float(adjusted_scores[top_index]),
            }
        )
        replaced = base_components.copy()
        replaced[:, idx] = 1.0
        rank_target(replaced, f"set_{component}_to_1")

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLE_DIR / "rank_sensitivity.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(component_df["component"], component_df["value"], color="#356c8f")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Component value")
    ax.set_title("VWF-SELP pyXenium CCI_score components")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "component_decomposition.png", dpi=200)
    plt.close(fig)


def write_vascular_tables(scores: pd.DataFrame) -> None:
    target_pairs = pd.DataFrame(VASCULAR_TARGET_PAIRS, columns=["ligand", "receptor"])
    exact = scores.merge(target_pairs, on=["ligand", "receptor"], how="inner")
    exact = exact.sort_values("CCI_score", ascending=False)
    exact.to_csv(TABLE_DIR / "vascular_target_pair_hits.tsv", sep="\t", index=False)

    vascular_genes = set(ENDOTHELIAL_MARKERS + ["CD93", "ITGA9", "HSPG2", "MMP2", "CXCL12", "CCN1", "CAV1", "LRP1"])
    vascular = scores.loc[
        scores["ligand"].isin(vascular_genes)
        | scores["receptor"].isin(vascular_genes)
        | scores["sender_celltype"].str.contains("Endothelial", regex=False)
        | scores["receiver_celltype"].str.contains("Endothelial", regex=False)
    ].copy()
    vascular = vascular.sort_values("CCI_score", ascending=False)
    vascular.head(250).to_csv(TABLE_DIR / "vascular_top_hits.tsv", sep="\t", index=False)
    vascular_compact = vascular.head(40).loc[
        :,
        [
            "ligand",
            "receptor",
            "sender_celltype",
            "receiver_celltype",
            "CCI_score",
            "sender_anchor",
            "receiver_anchor",
            "structure_bridge",
            "sender_expr",
            "receiver_expr",
            "local_contact",
            "contact_coverage",
            "cross_edge_count",
        ],
    ]
    vascular_compact.to_csv(TABLE_DIR / "vascular_top_hits_compact.tsv", sep="\t", index=False)

    plot_df = vascular.head(15).copy()
    plot_df["label"] = (
        plot_df["ligand"]
        + "-"
        + plot_df["receptor"]
        + "\n"
        + plot_df["sender_celltype"]
        + " -> "
        + plot_df["receiver_celltype"]
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.barh(plot_df["label"][::-1], plot_df["CCI_score"][::-1], color="#2f6f73")
    ax.set_xlabel("pyXenium CCI_score")
    ax.set_title("Top vascular/WPB-related pyXenium hits")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vascular_top_hits.png", dpi=200)
    plt.close(fig)


def load_selected_expression(genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FULL_H5AD.exists():
        raise FileNotFoundError(f"Missing full h5ad: {FULL_H5AD}")
    adata = ad.read_h5ad(FULL_H5AD, backed="r")
    try:
        var_names = pd.Index(adata.var_names.astype(str))
        present = [gene for gene in genes if gene in var_names]
        missing = sorted(set(genes).difference(present))
        obs = adata.obs[["cell_type", "x", "y", "cell_id"]].copy()
        obs.index = obs.index.astype(str)
        X = adata[:, present].X
        if sparse.issparse(X):
            values = X.toarray()
        else:
            values = np.asarray(X)
        expr = pd.DataFrame(values, index=obs.index, columns=present).astype(float)
        pd.DataFrame({"gene": genes, "available": [gene in present for gene in genes]}).to_csv(
            TABLE_DIR / "selected_gene_availability.tsv",
            sep="\t",
            index=False,
        )
        return obs, expr
    finally:
        adata.file.close()


def write_expression_tables(obs: pd.DataFrame, expr: pd.DataFrame) -> None:
    rows = []
    for cell_type, index in obs.groupby("cell_type").groups.items():
        sub = expr.loc[index]
        for gene in expr.columns:
            values = sub[gene].astype(float)
            rows.append(
                {
                    "gene": gene,
                    "cell_type": cell_type,
                    "n_cells": int(len(values)),
                    "mean_raw": float(values.mean()),
                    "mean_log1p": float(np.log1p(values).mean()),
                    "detection_fraction": float((values > 0).mean()),
                }
            )
    expression = pd.DataFrame(rows)
    expression["mean_log1p_norm_by_gene"] = expression.groupby("gene")["mean_log1p"].transform(normalize01)
    expression["detection_norm_by_gene"] = expression.groupby("gene")["detection_fraction"].transform(normalize01)
    expression.to_csv(TABLE_DIR / "expression_specificity_full_wta.tsv", sep="\t", index=False)

    target_rows = expression.loc[expression["gene"].isin(ENDOTHELIAL_MARKERS + CONTAMINATION_MARKERS)].copy()
    target_rows.to_csv(TABLE_DIR / "marker_expression_specificity_full_wta.tsv", sep="\t", index=False)
    top_celltypes = (
        target_rows.sort_values(["gene", "mean_log1p"], ascending=[True, False])
        .groupby("gene", as_index=False, group_keys=False)
        .head(6)
    )
    top_celltypes.to_csv(TABLE_DIR / "marker_top_celltypes_full_wta.tsv", sep="\t", index=False)

    heatmap_genes = [gene for gene in ENDOTHELIAL_MARKERS + CONTAMINATION_MARKERS if gene in expr.columns]
    heat = (
        expression.loc[expression["gene"].isin(heatmap_genes)]
        .pivot(index="gene", columns="cell_type", values="mean_log1p_norm_by_gene")
        .reindex(heatmap_genes)
    )
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(heatmap_genes))))
    im = ax.imshow(heat.fillna(0).to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=75, ha="right", fontsize=8)
    ax.set_title("Full WTA marker expression specificity (mean log1p, gene-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "expression_specificity_heatmap.png", dpi=220)
    plt.close(fig)

    contamination = expression.loc[expression["gene"].isin(CONTAMINATION_MARKERS)].copy()
    endothelial_mask = contamination["cell_type"].eq(TARGET_SENDER)
    contamination_summary = pd.concat(
        [
            contamination.loc[endothelial_mask].assign(group="Endothelial Cells"),
            contamination.loc[~endothelial_mask].groupby("gene", as_index=False).agg(
                {
                    "mean_raw": "mean",
                    "mean_log1p": "mean",
                    "detection_fraction": "mean",
                    "n_cells": "sum",
                    "mean_log1p_norm_by_gene": "mean",
                    "detection_norm_by_gene": "mean",
                }
            ).assign(cell_type="Other cell types mean", group="Other cell types mean"),
        ],
        ignore_index=True,
    )
    contamination_summary.to_csv(TABLE_DIR / "contamination_marker_control.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    plot = contamination_summary.pivot_table(index="gene", columns="group", values="detection_fraction", aggfunc="mean")
    plot = plot.reindex([gene for gene in CONTAMINATION_MARKERS if gene in plot.index])
    x = np.arange(len(plot.index))
    width = 0.36
    ax.bar(x - width / 2, plot.get("Endothelial Cells", pd.Series(0, index=plot.index)), width, label="Endothelial")
    ax.bar(x + width / 2, plot.get("Other cell types mean", pd.Series(0, index=plot.index)), width, label="Other mean")
    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, rotation=35, ha="right")
    ax.set_ylabel("Detection fraction")
    ax.set_title("Blood/platelet contamination marker check")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "contamination_control.png", dpi=200)
    plt.close(fig)


def write_hotspot_tables(obs: pd.DataFrame, expr: pd.DataFrame) -> None:
    if TARGET_LIGAND not in expr.columns or TARGET_RECEPTOR not in expr.columns:
        return
    coords = obs[["x", "y"]].to_numpy(dtype=float)
    nbrs = NearestNeighbors(n_neighbors=9, algorithm="kd_tree")
    nbrs.fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)[:, 1:]

    celltypes = obs["cell_type"].astype(str).to_numpy()
    endothelial = celltypes == TARGET_SENDER
    vwf = expr[TARGET_LIGAND].to_numpy(dtype=float)
    selp = expr[TARGET_RECEPTOR].to_numpy(dtype=float)
    vwf_norm = normalize01(np.log1p(vwf))
    selp_norm = normalize01(np.log1p(selp))

    vwf_positive = vwf[vwf > 0]
    selp_positive = selp[selp > 0]
    vwf_threshold = float(np.quantile(vwf_positive, 0.75)) if vwf_positive.size else float("inf")
    selp_threshold = float(np.quantile(selp_positive, 0.75)) if selp_positive.size else float("inf")

    ee_neighbor_count = np.zeros(len(obs), dtype=int)
    active_sender_edges = np.zeros(len(obs), dtype=int)
    active_receiver_edges = np.zeros(len(obs), dtype=int)
    mean_neighbor_selp = np.zeros(len(obs), dtype=float)
    mean_neighbor_vwf = np.zeros(len(obs), dtype=float)

    for idx, neigh in enumerate(indices):
        ee_neigh = neigh[endothelial[neigh]]
        if not endothelial[idx] or ee_neigh.size == 0:
            continue
        ee_neighbor_count[idx] = int(ee_neigh.size)
        mean_neighbor_selp[idx] = float(np.mean(selp_norm[ee_neigh]))
        mean_neighbor_vwf[idx] = float(np.mean(vwf_norm[ee_neigh]))
        if vwf[idx] > vwf_threshold:
            active_sender_edges[idx] = int(np.sum(selp[ee_neigh] > selp_threshold))
        if selp[idx] > selp_threshold:
            active_receiver_edges[idx] = int(np.sum(vwf[ee_neigh] > vwf_threshold))

    hotspot_score = np.sqrt(vwf_norm * mean_neighbor_selp) + np.sqrt(selp_norm * mean_neighbor_vwf)
    hotspot_score = normalize01(hotspot_score)
    hotspot_quantile = float(np.quantile(hotspot_score[endothelial], 0.95)) if endothelial.any() else float("inf")
    hotspot = endothelial & (hotspot_score >= hotspot_quantile)

    hotspot_cells = obs.loc[hotspot, ["cell_id", "cell_type", "x", "y"]].copy()
    hotspot_cells["VWF"] = vwf[hotspot]
    hotspot_cells["SELP"] = selp[hotspot]
    hotspot_cells["vwf_norm"] = vwf_norm[hotspot]
    hotspot_cells["selp_norm"] = selp_norm[hotspot]
    hotspot_cells["ee_neighbor_count"] = ee_neighbor_count[hotspot]
    hotspot_cells["active_sender_edges"] = active_sender_edges[hotspot]
    hotspot_cells["active_receiver_edges"] = active_receiver_edges[hotspot]
    hotspot_cells["hotspot_score"] = hotspot_score[hotspot]
    hotspot_cells.sort_values("hotspot_score", ascending=False).to_csv(
        TABLE_DIR / "vwf_selp_hotspot_endothelial_cells.tsv",
        sep="\t",
        index=False,
    )

    context_types = [
        "11q13 Invasive Tumor Cells",
        "CAFs, DCIS Associated",
        "CAFs, Invasive Associated",
        "Pericytes",
        "Macrophages",
        "T Lymphocytes",
    ]
    context_rows = []
    for group_name, mask in [("hotspot_endothelial", hotspot), ("all_endothelial", endothelial)]:
        selected_idx = np.flatnonzero(mask)
        if selected_idx.size == 0:
            continue
        neigh = indices[selected_idx].reshape(-1)
        neighbor_types = pd.Series(celltypes[neigh]).value_counts(normalize=True)
        for cell_type in context_types:
            context_rows.append(
                {
                    "group": group_name,
                    "neighbor_cell_type": cell_type,
                    "neighbor_fraction": float(neighbor_types.get(cell_type, 0.0)),
                    "n_query_cells": int(selected_idx.size),
                }
            )
    pd.DataFrame(context_rows).to_csv(TABLE_DIR / "hotspot_neighbor_context.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_cells_total": int(len(obs)),
                "n_endothelial_cells": int(endothelial.sum()),
                "n_hotspot_endothelial_cells": int(hotspot.sum()),
                "vwf_q75_nonzero_raw": vwf_threshold,
                "selp_q75_nonzero_raw": selp_threshold,
                "mean_hotspot_score_endothelial": float(np.mean(hotspot_score[endothelial])),
                "mean_hotspot_score_hotspot": float(np.mean(hotspot_score[hotspot])) if hotspot.any() else np.nan,
            }
        ]
    )
    summary.to_csv(TABLE_DIR / "hotspot_summary.tsv", sep="\t", index=False)

    rng = np.random.default_rng(7)
    all_idx = np.arange(len(obs))
    background_idx = rng.choice(all_idx, size=min(45000, len(all_idx)), replace=False)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.scatter(obs["x"].to_numpy()[background_idx], obs["y"].to_numpy()[background_idx], s=0.4, c="#d3d3d3", alpha=0.35, linewidths=0)
    endothelial_idx = np.flatnonzero(endothelial)
    ax.scatter(obs["x"].to_numpy()[endothelial_idx], obs["y"].to_numpy()[endothelial_idx], s=1.2, c="#5c88a6", alpha=0.35, linewidths=0)
    hotspot_idx = np.flatnonzero(hotspot)
    sc = ax.scatter(
        obs["x"].to_numpy()[hotspot_idx],
        obs["y"].to_numpy()[hotspot_idx],
        s=5,
        c=hotspot_score[hotspot_idx],
        cmap="magma",
        alpha=0.95,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("VWF-SELP endothelial hotspot map")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="Hotspot score")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vwf_selp_hotspot_spatial_overlay.png", dpi=220)
    plt.close(fig)


def write_pathway_tables(obs: pd.DataFrame, expr: pd.DataFrame) -> None:
    rows = []
    for pathway, genes in PATHWAY_PANELS.items():
        present = [gene for gene in genes if gene in expr.columns]
        if present:
            gene_values = np.log1p(expr[present].astype(float))
            normalized = gene_values.apply(normalize01, axis=0, result_type="broadcast")
            scores = normalized.mean(axis=1)
        else:
            scores = pd.Series(0.0, index=expr.index)
        for cell_type, index in obs.groupby("cell_type").groups.items():
            values = scores.loc[index]
            rows.append(
                {
                    "pathway": pathway,
                    "cell_type": cell_type,
                    "present_genes": ",".join(present),
                    "n_present_genes": len(present),
                    "mean_activity": float(values.mean()),
                    "q95_activity": float(values.quantile(0.95)),
                    "detection_fraction_activity_positive": float((values > 0).mean()),
                }
            )
    pathway_summary = pd.DataFrame(rows)
    pathway_summary.to_csv(TABLE_DIR / "targeted_pathway_celltype_summary.tsv", sep="\t", index=False)
    pathway_top = (
        pathway_summary.sort_values(["pathway", "mean_activity"], ascending=[True, False])
        .groupby("pathway", as_index=False, group_keys=False)
        .head(8)
    )
    pathway_top.to_csv(TABLE_DIR / "targeted_pathway_top_celltypes.tsv", sep="\t", index=False)

    heat = pathway_summary.pivot(index="pathway", columns="cell_type", values="mean_activity")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(heat.fillna(0).to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=75, ha="right", fontsize=8)
    ax.set_title("Targeted vascular/WPB pathway-style activity")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "targeted_pathway_celltype_heatmap.png", dpi=220)
    plt.close(fig)

    if {"VWF", "SELP"}.issubset(expr.columns):
        joint = normalize01(np.log1p(expr["VWF"].to_numpy(dtype=float))) * normalize01(np.log1p(expr["SELP"].to_numpy(dtype=float)))
        corr_rows = []
        for pathway, genes in PATHWAY_PANELS.items():
            present = [gene for gene in genes if gene in expr.columns]
            if not present:
                continue
            values = np.log1p(expr[present].astype(float)).apply(normalize01, axis=0, result_type="broadcast").mean(axis=1)
            rho, pvalue = stats.spearmanr(joint, values)
            corr_rows.append(
                {
                    "pathway": pathway,
                    "present_genes": ",".join(present),
                    "spearman_rho_vs_vwf_selp_joint": float(rho),
                    "p_value": float(pvalue),
                }
            )
        pd.DataFrame(corr_rows).to_csv(TABLE_DIR / "vwf_selp_pathway_correlations.tsv", sep="\t", index=False)


def write_contour_tables() -> None:
    contour_rows = []
    if CONTOUR_FEATURES.exists():
        features = pd.read_csv(CONTOUR_FEATURES)
        cols = [
            column
            for column in features.columns
            if any(token in column.lower() for token in ["pecam1", "emcn", "kdr", "vascular_stromal"])
        ]
        long_rows = []
        for column in cols:
            tmp = features[["contour_id", "classification_name", "assigned_structure", column]].copy()
            tmp = tmp.rename(columns={column: "value"})
            tmp["feature"] = column
            long_rows.append(tmp)
        if long_rows:
            long = pd.concat(long_rows, ignore_index=True)
            long.to_csv(TABLE_DIR / "contour_vascular_features_long.tsv", sep="\t", index=False)
            summary = (
                long.groupby(["feature", "classification_name"], as_index=False)
                .agg(mean_value=("value", "mean"), median_value=("value", "median"), n_contours=("value", "count"))
                .sort_values(["feature", "mean_value"], ascending=[True, False])
            )
            summary.to_csv(TABLE_DIR / "contour_vascular_feature_summary.tsv", sep="\t", index=False)
            contour_rows.append({"source": str(CONTOUR_FEATURES), "n_features": len(cols), "n_rows": len(long)})

            key_features = [c for c in cols if c in {"pathway__whole__vascular_stromal", "pathway__inner_rim__vascular_stromal", "pathway__outer_rim__vascular_stromal", "edge_contrast__pathway__vascular_stromal"}]
            if key_features:
                plot = long.loc[long["feature"].isin(key_features)].copy()
                fig, axes = plt.subplots(len(key_features), 1, figsize=(9, max(3, 2.2 * len(key_features))), sharex=False)
                if len(key_features) == 1:
                    axes = [axes]
                for ax, feature in zip(axes, key_features):
                    data = [group["value"].dropna().to_numpy(dtype=float) for _, group in plot.loc[plot["feature"] == feature].groupby("classification_name")]
                    labels = [name for name, _ in plot.loc[plot["feature"] == feature].groupby("classification_name")]
                    ax.boxplot(data, labels=labels, showfliers=False)
                    ax.set_title(feature)
                    ax.tick_params(axis="x", rotation=35)
                fig.tight_layout()
                fig.savefig(FIGURE_DIR / "contour_vascular_summary.png", dpi=220)
                plt.close(fig)

    if CONTOUR_PROGRAMS.exists():
        programs = pd.read_csv(CONTOUR_PROGRAMS)
        program_cols = [c for c in ["myeloid_vascular_belt", "stromal_encapsulation", "emt_invasive_front"] if c in programs.columns]
        if program_cols:
            programs[["contour_id", *program_cols, "top_program", "top_program_score"]].to_csv(
                TABLE_DIR / "contour_boundary_program_scores.tsv",
                sep="\t",
                index=False,
            )
    if CONTOUR_HYPOTHESES.exists():
        pd.read_csv(CONTOUR_HYPOTHESES).to_csv(TABLE_DIR / "contour_hypothesis_ranking.tsv", sep="\t", index=False)
    if S1S5_DE.exists():
        de = pd.read_csv(S1S5_DE)
        vascular_genes = [gene for gene in ENDOTHELIAL_MARKERS if gene in set(de["gene"].astype(str).str.upper())]
        de_vascular = de.loc[de["gene"].astype(str).str.upper().isin(vascular_genes)].copy()
        de_vascular.to_csv(TABLE_DIR / "s1_s5_vascular_gene_de.tsv", sep="\t", index=False)

    pd.DataFrame(contour_rows).to_csv(TABLE_DIR / "contour_input_summary.tsv", sep="\t", index=False)


def write_mechanostress_tables(obs: pd.DataFrame, expr: pd.DataFrame) -> None:
    rows = []
    if MECHANOSTRESS_COUPLING.exists():
        coupling = pd.read_csv(MECHANOSTRESS_COUPLING)
        coupling.to_csv(TABLE_DIR / "mechanostress_distance_expression_coupling_available.tsv", sep="\t", index=False)
        for gene in ["VWF", "SELP", "PECAM1", "EMCN", "KDR", "COL4A1", "COL4A2"]:
            rows.append({"gene": gene, "present_in_existing_mechanostress_coupling": bool((coupling["gene"].astype(str).str.upper() == gene).any())})

    if {"VWF", "SELP"}.issubset(expr.columns):
        endothelial = obs["cell_type"].astype(str).eq(TARGET_SENDER)
        tumor = obs["cell_type"].astype(str).str.contains("Tumor", regex=False)
        caf = obs["cell_type"].astype(str).str.contains("CAF", regex=False)
        coords = obs[["x", "y"]].to_numpy(dtype=float)
        for label, mask in [("nearest_tumor", tumor.to_numpy()), ("nearest_caf", caf.to_numpy())]:
            if mask.sum() == 0:
                continue
            nbr = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coords[mask])
            distances = nbr.kneighbors(coords[endothelial.to_numpy()], return_distance=True)[0].reshape(-1)
            joint = normalize01(np.log1p(expr.loc[endothelial, "VWF"].to_numpy(dtype=float))) * normalize01(
                np.log1p(expr.loc[endothelial, "SELP"].to_numpy(dtype=float))
            )
            rho, pvalue = stats.spearmanr(joint, -distances)
            rows.append(
                {
                    "gene": f"VWF_SELP_joint_vs_{label}",
                    "present_in_existing_mechanostress_coupling": np.nan,
                    "n_endothelial_cells": int(endothelial.sum()),
                    "spearman_rho_vs_inverse_distance": float(rho),
                    "p_value": float(pvalue),
                    "median_distance_um": float(np.median(distances)),
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "mechanostress_context_summary.tsv", sep="\t", index=False)


def write_cross_method_tables() -> None:
    method_paths = {
        "cellphonedb": PDC_ROOT / "runs" / "full_common" / "cellphonedb" / "cellphonedb_standardized.tsv.gz",
        "laris": PDC_ROOT / "runs" / "full_common" / "laris" / "laris_standardized.tsv.gz",
        "liana": PDC_ROOT / "runs" / "full_common" / "liana" / "liana_standardized.tsv.gz",
        "spatialdm": PDC_ROOT / "runs" / "full_common" / "spatialdm" / "spatialdm_standardized.tsv.gz",
        "stlearn": PDC_ROOT / "runs" / "full_common" / "stlearn" / "stlearn_standardized.tsv.gz",
        "pyxenium": PYX_STANDARDIZED,
    }
    vascular_genes = set(ENDOTHELIAL_MARKERS + ["LRP1", "CD93", "ITGA9", "HSPG2", "MMP2", "CXCL12", "CCN1", "CAV1"])
    rows = []
    exact_rows = []
    for method, path in method_paths.items():
        if not path.exists():
            continue
        table = pd.read_csv(path, sep="\t")
        table["lr_pair"] = table["ligand"].astype(str) + "-" + table["receptor"].astype(str)
        exact = table.loc[(table["ligand"] == TARGET_LIGAND) & (table["receptor"] == TARGET_RECEPTOR)].copy()
        if not exact.empty:
            exact_rows.append(exact.sort_values("rank_within_method").head(10).assign(method_source=method))
        vascular = table.loc[
            table["ligand"].isin(vascular_genes)
            | table["receptor"].isin(vascular_genes)
            | table["sender"].astype(str).str.contains("Endothelial", regex=False)
            | table["receiver"].astype(str).str.contains("Endothelial", regex=False)
        ].copy()
        vascular = vascular.sort_values("rank_within_method").head(20)
        rows.append(vascular.assign(method_source=method))

    if rows:
        consensus = pd.concat(rows, ignore_index=True)
        consensus.to_csv(TABLE_DIR / "cross_method_vascular_consensus.tsv", sep="\t", index=False)
        compact_cols = [
            "method_source",
            "ligand",
            "receptor",
            "sender",
            "receiver",
            "score_raw",
            "score_std",
            "rank_within_method",
            "spatial_support_type",
        ]
        available_cols = [column for column in compact_cols if column in consensus.columns]
        consensus.loc[:, available_cols].to_csv(TABLE_DIR / "cross_method_vascular_consensus_compact.tsv", sep="\t", index=False)
    if exact_rows:
        pd.concat(exact_rows, ignore_index=True).to_csv(TABLE_DIR / "cross_method_vwf_selp_hits.tsv", sep="\t", index=False)


def write_report(row: pd.Series) -> None:
    def maybe_table(path: str, n: int = 8) -> str:
        table_path = TABLE_DIR / path
        if not table_path.exists():
            return "_Not available._\n"
        df = pd.read_csv(table_path, sep="\t")
        return df.head(n).to_markdown(index=False) + "\n"

    expression = pd.read_csv(TABLE_DIR / "expression_specificity_full_wta.tsv", sep="\t")
    vwf_endothelial = expression.loc[(expression["gene"] == "VWF") & (expression["cell_type"] == TARGET_SENDER)].iloc[0]
    selp_endothelial = expression.loc[(expression["gene"] == "SELP") & (expression["cell_type"] == TARGET_SENDER)].iloc[0]
    hotspot_summary = pd.read_csv(TABLE_DIR / "hotspot_summary.tsv", sep="\t").iloc[0]
    contamination = pd.read_csv(TABLE_DIR / "contamination_marker_control.tsv", sep="\t")
    hbb_endothelial = contamination.loc[(contamination["gene"] == "HBB") & (contamination["group"] == "Endothelial Cells")]
    hbb_note = ""
    if not hbb_endothelial.empty:
        hbb_note = (
            f" `HBB` is the strongest blood-marker caveat in endothelial cells "
            f"(detection fraction `{float(hbb_endothelial.iloc[0]['detection_fraction']):.4f}`), "
            "so the report treats blood carryover as a caution rather than ignoring it."
        )

    report = f"""# VWF-SELP endothelial-endothelial deep dive

## Working conclusion

The top pyXenium full-common hit, `VWF-SELP / Endothelial Cells -> Endothelial Cells`, is best interpreted as an endothelial vascular activation and adhesion state rather than a simple classical paracrine cell-cell interaction axis. The score is high because topology, expression, endothelial-endothelial structural proximity, and local contact all support the same vascular compartment.

## Direct pyXenium evidence

- `CCI_score`: `{float(row['CCI_score']):.12f}`
- `sender_anchor`: `{float(row['sender_anchor']):.6f}`
- `receiver_anchor`: `{float(row['receiver_anchor']):.6f}`
- `structure_bridge`: `{float(row['structure_bridge']):.6f}`
- `sender_expr`: `{float(row['sender_expr']):.6f}`
- `receiver_expr`: `{float(row['receiver_expr']):.6f}`
- `local_contact`: `{float(row['local_contact']):.6f}`
- `contact_coverage`: `{float(row['contact_coverage']):.6f}`
- `cross_edge_count`: `{int(row['cross_edge_count'])}`

### Component decomposition

{maybe_table('component_decomposition.tsv', 10)}

### Rank sensitivity

{maybe_table('rank_sensitivity.tsv', 14)}

## Full WTA expression specificity

In the full WTA object, `Endothelial Cells` contain `{int(hotspot_summary['n_endothelial_cells'])}` cells. `VWF` and `SELP` are detectable in the WTA matrix, and the endothelial compartment shows direct expression support:

- `VWF` endothelial mean log1p: `{float(vwf_endothelial['mean_log1p']):.4f}`, detection fraction: `{float(vwf_endothelial['detection_fraction']):.4f}`.
- `SELP` endothelial mean log1p: `{float(selp_endothelial['mean_log1p']):.4f}`, detection fraction: `{float(selp_endothelial['detection_fraction']):.4f}`.

The contamination-control table checks platelet and erythroid markers (`PPBP`, `PF4`, `ITGA2B`, `GP1BA`, `HBB`, `HBA1`, `HBA2`) so the interpretation is not reduced to blood carryover.
{hbb_note}

### Marker top cell types

{maybe_table('marker_top_celltypes_full_wta.tsv', 18)}

## Spatial hotspot evidence

The hotspot analysis uses full WTA coordinates and a k-nearest-neighbor endothelial neighborhood summary. It identifies `{int(hotspot_summary['n_hotspot_endothelial_cells'])}` high-scoring endothelial hotspot cells at the 95th percentile of endothelial VWF-SELP neighborhood support.

### Hotspot neighbor context

{maybe_table('hotspot_neighbor_context.tsv', 12)}

## Vascular and pathway context

The top vascular table places `VWF-SELP` in a broader endothelial program that includes VWF/LRP1, EFNB2/PECAM1, MMRN2/CLEC14A, COL4A2/CD93, VEGFC/FLT1, and other vascular matrix/adhesion axes.

### Top vascular pyXenium hits

{maybe_table('vascular_top_hits_compact.tsv', 14)}

### Targeted pathway-style summary

{maybe_table('targeted_pathway_top_celltypes.tsv', 16)}

## Contour and boundary ecology context

Existing contour outputs do not include direct `SELP` contour features, but they do include vascular markers and `vascular_stromal` pathway scores. The S1-S5 contour DE table supports vascular enrichment in S3 for several endothelial markers, including `VWF`, `PECAM1`, `EMCN`, `EGFL7`, `MMRN2`, `CLEC14A`, `KDR`, and `FLT1`.

### S1-S5 vascular gene DE

{maybe_table('s1_s5_vascular_gene_de.tsv', 12)}

### Boundary program scores

{maybe_table('contour_boundary_program_scores.tsv', 8)}

## Mechanostress and tumor-stroma context

Existing mechanostress coupling outputs do not directly include `VWF` or `SELP`, so the current analysis reports available overlap and uses full WTA geometry to relate endothelial VWF-SELP joint support to tumor/CAF proximity.

{maybe_table('mechanostress_context_summary.tsv', 12)}

The simple nearest-distance test is negative for both inverse tumor distance and inverse CAF distance, so this first-pass result supports a vascular/endothelial self-state more than a signal concentrated immediately at tumor-stroma contact fronts. That does not rule out boundary-associated vascular niches, but it argues they should be tested with contour and vessel-structure annotations rather than inferred from nearest tumor/CAF distance alone.

## Cross-method triangulation

Other full-common benchmark methods do not need to recover the exact `VWF-SELP` pair for this signal to be meaningful. The useful test is whether they recover the broader vascular/stromal theme, such as `VWF-LRP1` in CellPhoneDB and endothelial/stromal vascular axes in LARIS or spatial methods.

{maybe_table('cross_method_vascular_consensus_compact.tsv', 18)}

## Literature-supported interpretation

- NCBI Bookshelf describes Weibel-Palade bodies as endothelial granules that store VWF and P-selectin, linking them to hemostasis, inflammation, platelet aggregation, leukocyte trafficking, and angiogenesis: https://www.ncbi.nlm.nih.gov/books/NBK535353/
- Reactome places P-selectin binding biology under cell-surface interactions at the vascular wall: https://reactome.org/content/detail/R-HSA-202724
- A Frontiers review describes regulated Weibel-Palade body exocytosis as a mechanism that presents highly multimeric VWF and P-selectin at the endothelial surface after vascular injury or inflammatory stimulation: https://www.frontiersin.org/articles/10.3389/fcell.2021.813995/full
- UniProt VWF entry: https://rest.uniprot.org/uniprotkb/P04275.txt

## Caveats

- `VWF-SELP` in transcriptomics is a state-level inference; it does not prove protein-level VWF/P-selectin co-storage or exocytosis.
- P-selectin is also platelet-associated, so contamination controls are required for interpretation.
- pyXenium's `Endothelial Cells -> Endothelial Cells` direction should be read as an endothelial neighborhood/self-state axis, not necessarily directional ligand secretion.
- Direct validation would require protein staining, vascular morphology, or thromboinflammatory markers beyond WTA RNA.
"""
    (REPORT_DIR / "vwf_selp_deep_dive.md").write_text(report, encoding="utf-8")


def write_summary_json(row: pd.Series) -> None:
    summary = {
        "target": {
            "ligand": TARGET_LIGAND,
            "receptor": TARGET_RECEPTOR,
            "sender": TARGET_SENDER,
            "receiver": TARGET_RECEIVER,
            "CCI_score": float(row["CCI_score"]),
            "local_contact": float(row["local_contact"]),
            "cross_edge_count": int(row["cross_edge_count"]),
        },
        "outputs": {
            "tables": str(TABLE_DIR),
            "figures": str(FIGURE_DIR),
            "report": str(REPORT_DIR / "vwf_selp_deep_dive.md"),
        },
    }
    (WORK_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    scores = load_scores()
    row = target_row(scores)
    if not math.isclose(float(row["CCI_score"]), 0.7912892368005828, rel_tol=1e-6):
        raise AssertionError(f"Unexpected VWF-SELP CCI_score: {row['CCI_score']}")
    if int(row["cross_edge_count"]) != 12779:
        raise AssertionError(f"Unexpected VWF-SELP cross_edge_count: {row['cross_edge_count']}")

    write_component_tables(scores, row)
    write_vascular_tables(scores)

    selected_genes = sorted(set(ENDOTHELIAL_MARKERS + CONTAMINATION_MARKERS + sum(PATHWAY_PANELS.values(), [])))
    obs, expr = load_selected_expression(selected_genes)
    write_expression_tables(obs, expr)
    write_hotspot_tables(obs, expr)
    write_pathway_tables(obs, expr)
    write_contour_tables()
    write_mechanostress_tables(obs, expr)
    write_cross_method_tables()
    write_report(row)
    write_summary_json(row)
    print(f"Wrote VWF-SELP deep dive outputs under {WORK_ROOT}")


if __name__ == "__main__":
    main()
