from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


DEFAULT_ROOT = Path("/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04")
COMPONENTS = [
    "sender_anchor",
    "receiver_anchor",
    "structure_bridge",
    "sender_expr",
    "receiver_expr",
    "local_contact",
    "prior_confidence",
]


@dataclass(frozen=True)
class TargetAxis:
    ligand: str
    receptor: str
    sender: str
    receiver: str
    biology_label: str
    target_panel: tuple[str, ...]

    @property
    def axis_id(self) -> str:
        return f"{self.ligand}-{self.receptor}|{self.sender}->{self.receiver}"


TARGET_AXES = [
    TargetAxis(
        "VWF",
        "SELP",
        "Endothelial Cells",
        "Endothelial Cells",
        "WPB / endothelial activation",
        ("VWF", "SELP", "CD63", "ANGPT2", "THBD", "PLAT", "SERPINE1"),
    ),
    TargetAxis(
        "VWF",
        "LRP1",
        "Endothelial Cells",
        "CAFs, DCIS Associated",
        "vascular-stromal matrix/scavenger axis",
        ("VWF", "LRP1", "HSPG2", "COL4A1", "COL4A2", "MMP2", "THBS2"),
    ),
    TargetAxis(
        "MMRN2",
        "CD93",
        "Endothelial Cells",
        "Endothelial Cells",
        "CD93-MMRN2 angiogenesis",
        ("MMRN2", "CD93", "CLEC14A", "KDR", "FLT1", "PECAM1", "EMCN"),
    ),
    TargetAxis(
        "DLL4",
        "NOTCH3",
        "Endothelial Cells",
        "Pericytes",
        "endothelial-pericyte Notch",
        ("DLL4", "NOTCH3", "NOTCH4", "JAG1", "HEY1", "HES1", "PDGFRB", "RGS5"),
    ),
    TargetAxis(
        "CXCL12",
        "CXCR4",
        "CAFs, DCIS Associated",
        "T Lymphocytes",
        "CAF-immune chemokine recruitment",
        ("CXCL12", "CXCR4", "CXCR3", "CCL19", "CCL21", "CD3D", "CD3E", "IL7R"),
    ),
    TargetAxis(
        "CD48",
        "CD2",
        "T Lymphocytes",
        "T Lymphocytes",
        "T-cell adhesion/co-stimulation",
        ("CD48", "CD2", "CD3D", "CD3E", "TRAC", "IL7R", "LCK"),
    ),
    TargetAxis(
        "JAG1",
        "NOTCH2",
        "11q13 Invasive Tumor Cells",
        "11q13 Invasive Tumor Cells",
        "tumor-intrinsic Notch signaling",
        ("JAG1", "NOTCH2", "NOTCH1", "HES1", "HEY1", "MYC", "CCND1"),
    ),
]


def qnorm_sf(z: float) -> float:
    """Normal survival function without depending on scipy.stats import latency."""
    if not np.isfinite(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def geometric_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    clipped = np.clip(values.astype(float), 1e-12, None)
    return np.exp(np.mean(np.log(clipped), axis=axis))


def load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "data" / "input_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_scores(root: Path) -> pd.DataFrame:
    path = root / "runs" / "full_common" / "pyxenium" / "pyxenium_scores.tsv"
    scores = pd.read_csv(path, sep="\t")
    scores = scores.sort_values("CCI_score", ascending=False, kind="mergesort").reset_index(drop=True)
    scores["global_rank"] = np.arange(1, len(scores) + 1)
    return scores


def find_target_rows(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for axis in TARGET_AXES:
        exact = scores[
            (scores["ligand"] == axis.ligand)
            & (scores["receptor"] == axis.receptor)
            & (scores["sender_celltype"] == axis.sender)
            & (scores["receiver_celltype"] == axis.receiver)
        ]
        if exact.empty:
            pair_any = scores[(scores["ligand"] == axis.ligand) & (scores["receptor"] == axis.receptor)]
            if pair_any.empty:
                continue
            row = pair_any.iloc[0].copy()
            row["selection_note"] = "exact_sender_receiver_missing_best_lr_pair_used"
        else:
            row = exact.iloc[0].copy()
            row["selection_note"] = "exact"
        row["axis_id"] = axis.axis_id
        row["biology_label"] = axis.biology_label
        rows.append(row)
    if not rows:
        raise RuntimeError("No target LR axes were found in pyxenium_scores.tsv")
    return pd.DataFrame(rows).reset_index(drop=True)


def component_ablation(scores: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    comp = scores[COMPONENTS].astype(float).to_numpy()
    log_comp = np.log(np.clip(comp, 1e-12, None))
    log_sum = log_comp.sum(axis=1)
    target_keys = {
        (row.ligand, row.receptor, row.sender_celltype, row.receiver_celltype): row.axis_id
        for row in targets.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    key_to_index = {
        (row.ligand, row.receptor, row.sender_celltype, row.receiver_celltype): i
        for i, row in enumerate(scores.itertuples(index=False))
        if (row.ligand, row.receptor, row.sender_celltype, row.receiver_celltype) in target_keys
    }
    for component_index, component in enumerate(COMPONENTS):
        ablated_scores = np.exp((log_sum - log_comp[:, component_index]) / (len(COMPONENTS) - 1))
        for key, axis_id in target_keys.items():
            idx = key_to_index.get(key)
            if idx is None:
                continue
            value = float(ablated_scores[idx])
            rank = int(np.count_nonzero(ablated_scores > value) + 1)
            rows.append(
                {
                    "axis_id": axis_id,
                    "removed_component": component,
                    "score_without_component": value,
                    "rank_without_component": rank,
                }
            )
    return pd.DataFrame(rows)


def load_bundle_expression(manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, sparse.csr_matrix]:
    bundle = manifest["full_bundle"]
    genes = pd.read_csv(bundle["genes_tsv"], sep="\t")
    meta = pd.read_csv(bundle["meta_tsv"], sep="\t")
    matrix = mmread(bundle["counts_symbol_mtx"]).tocsr()
    if matrix.shape[0] == len(meta) and matrix.shape[1] == len(genes):
        matrix = matrix.T.tocsr()
    if matrix.shape[0] != len(genes) or matrix.shape[1] != len(meta):
        raise RuntimeError(
            "Sparse matrix shape does not match genes/meta: "
            f"matrix={matrix.shape}, genes={len(genes)}, cells={len(meta)}"
        )
    return genes, meta, matrix


def summarize_expression(
    genes: pd.DataFrame,
    meta: pd.DataFrame,
    matrix: sparse.csr_matrix,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    gene_symbols = genes["gene_symbol"].astype(str).str.upper().to_numpy()
    gene_to_index = {gene: i for i, gene in enumerate(gene_symbols)}
    cell_types = pd.Categorical(meta["cell_type"].astype(str))
    indicator = sparse.csr_matrix(
        (
            np.ones(len(cell_types), dtype=np.float32),
            (np.arange(len(cell_types)), cell_types.codes),
        ),
        shape=(len(cell_types), len(cell_types.categories)),
    )
    celltype_sizes = np.asarray(indicator.sum(axis=0)).ravel()
    sum_by_celltype = (matrix @ indicator).toarray()
    detected = matrix.copy()
    detected.data = np.ones_like(detected.data, dtype=np.float32)
    nnz_by_celltype = (detected @ indicator).toarray()
    mean_by_celltype = sum_by_celltype / np.maximum(celltype_sizes, 1.0)
    det_by_celltype = nnz_by_celltype / np.maximum(celltype_sizes, 1.0)
    expr_summary_rows: list[dict[str, Any]] = []
    needed = sorted({g for axis in TARGET_AXES for g in (axis.ligand, axis.receptor, *axis.target_panel)})
    for gene in needed:
        idx = gene_to_index.get(gene.upper())
        if idx is None:
            expr_summary_rows.append({"gene": gene, "detected_in_wta": False})
            continue
        means = mean_by_celltype[idx, :]
        dets = det_by_celltype[idx, :]
        order = np.argsort(-means)
        for rank, ct_idx in enumerate(order[:5], start=1):
            expr_summary_rows.append(
                {
                    "gene": gene,
                    "detected_in_wta": True,
                    "top_celltype_rank": rank,
                    "cell_type": str(cell_types.categories[ct_idx]),
                    "mean_expr": float(means[ct_idx]),
                    "detection_fraction": float(dets[ct_idx]),
                }
            )
    expr_summary = pd.DataFrame(expr_summary_rows)
    return expr_summary, mean_by_celltype, det_by_celltype, gene_to_index


def expression_specificity(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    mean_by_celltype: np.ndarray,
    det_by_celltype: np.ndarray,
    gene_to_index: dict[str, int],
) -> pd.DataFrame:
    categories = list(pd.Categorical(meta["cell_type"].astype(str)).categories)
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        lig_idx = gene_to_index.get(str(row.ligand).upper())
        rec_idx = gene_to_index.get(str(row.receptor).upper())
        sender_idx = categories.index(row.sender_celltype) if row.sender_celltype in categories else None
        receiver_idx = categories.index(row.receiver_celltype) if row.receiver_celltype in categories else None
        out: dict[str, Any] = {
            "axis_id": row.axis_id,
            "ligand": row.ligand,
            "receptor": row.receptor,
            "sender": row.sender_celltype,
            "receiver": row.receiver_celltype,
        }
        for role, gene_idx, celltype_idx in (
            ("ligand_sender", lig_idx, sender_idx),
            ("receptor_receiver", rec_idx, receiver_idx),
        ):
            if gene_idx is None or celltype_idx is None:
                out[f"{role}_mean"] = np.nan
                out[f"{role}_detection_fraction"] = np.nan
                out[f"{role}_celltype_rank"] = np.nan
                out[f"{role}_specificity_ratio"] = np.nan
                continue
            means = mean_by_celltype[gene_idx, :]
            dets = det_by_celltype[gene_idx, :]
            rank = int(np.where(np.argsort(-means) == celltype_idx)[0][0] + 1)
            out[f"{role}_mean"] = float(means[celltype_idx])
            out[f"{role}_detection_fraction"] = float(dets[celltype_idx])
            out[f"{role}_celltype_rank"] = rank
            out[f"{role}_specificity_ratio"] = float(means[celltype_idx] / max(float(np.max(means)), 1e-12))
        rows.append(out)
    return pd.DataFrame(rows)


def matched_gene_controls(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    mean_by_celltype: np.ndarray,
    det_by_celltype: np.ndarray,
    gene_to_index: dict[str, int],
    *,
    n_controls: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cell_types = pd.Categorical(meta["cell_type"].astype(str))
    categories = list(cell_types.categories)
    celltype_sizes = np.asarray(pd.Series(cell_types).value_counts(sort=False), dtype=float)
    weights = celltype_sizes / max(float(celltype_sizes.sum()), 1.0)
    global_mean = mean_by_celltype @ weights
    global_det = det_by_celltype @ weights
    all_indices = np.arange(mean_by_celltype.shape[0])
    protected = {gene_to_index[g.upper()] for axis in TARGET_AXES for g in (axis.ligand, axis.receptor) if g.upper() in gene_to_index}
    rows: list[dict[str, Any]] = []

    def nearest_pool(gene_idx: int, pool_size: int = 500) -> np.ndarray:
        distance = np.abs(np.log1p(global_mean) - np.log1p(global_mean[gene_idx]))
        distance += np.abs(global_det - global_det[gene_idx])
        distance[list(protected)] = np.inf
        distance[gene_idx] = np.inf
        valid = all_indices[np.isfinite(distance)]
        if len(valid) == 0:
            return np.array([], dtype=int)
        ordered = valid[np.argsort(distance[valid])]
        return ordered[: min(pool_size, len(ordered))]

    for row in targets.itertuples(index=False):
        lig_idx = gene_to_index.get(str(row.ligand).upper())
        rec_idx = gene_to_index.get(str(row.receptor).upper())
        if lig_idx is None or rec_idx is None or row.sender_celltype not in categories or row.receiver_celltype not in categories:
            rows.append({"axis_id": row.axis_id, "matched_gene_status": "missing_gene_or_celltype"})
            continue
        sender_idx = categories.index(row.sender_celltype)
        receiver_idx = categories.index(row.receiver_celltype)
        lig_pool = nearest_pool(lig_idx)
        rec_pool = nearest_pool(rec_idx)
        if len(lig_pool) == 0 or len(rec_pool) == 0:
            rows.append({"axis_id": row.axis_id, "matched_gene_status": "no_matched_pool"})
            continue
        lig_sample = rng.choice(lig_pool, size=n_controls, replace=len(lig_pool) < n_controls)
        rec_sample = rng.choice(rec_pool, size=n_controls, replace=len(rec_pool) < n_controls)

        def expr_score(lidx: int, ridx: int) -> float:
            lig_ratio = mean_by_celltype[lidx, sender_idx] / max(float(np.max(mean_by_celltype[lidx, :])), 1e-12)
            rec_ratio = mean_by_celltype[ridx, receiver_idx] / max(float(np.max(mean_by_celltype[ridx, :])), 1e-12)
            lig_det = det_by_celltype[lidx, sender_idx]
            rec_det = det_by_celltype[ridx, receiver_idx]
            return float(geometric_mean(np.array([lig_ratio, rec_ratio, lig_det, rec_det])))

        observed = expr_score(lig_idx, rec_idx)
        control_scores = np.array([expr_score(lidx, ridx) for lidx, ridx in zip(lig_sample, rec_sample)], dtype=float)
        rows.append(
            {
                "axis_id": row.axis_id,
                "matched_gene_status": "success",
                "n_controls": int(len(control_scores)),
                "observed_expression_specificity_score": observed,
                "control_mean": float(np.mean(control_scores)),
                "control_sd": float(np.std(control_scores, ddof=1)),
                "matched_gene_z": float((observed - np.mean(control_scores)) / max(float(np.std(control_scores, ddof=1)), 1e-12)),
                "matched_gene_percentile": float(np.mean(control_scores <= observed)),
            }
        )
    return pd.DataFrame(rows)


def spatial_abundance_null(scores: pd.DataFrame, targets: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    counts = meta["cell_type"].value_counts().to_dict()
    total_cells = len(meta)
    edge_counts = scores[["sender_celltype", "receiver_celltype", "cross_edge_count"]].drop_duplicates()
    total_edges = float(edge_counts["cross_edge_count"].sum())
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        p = (counts.get(row.sender_celltype, 0) / total_cells) * (counts.get(row.receiver_celltype, 0) / total_cells)
        expected = total_edges * p
        variance = total_edges * p * max(1.0 - p, 1e-12)
        z = (float(row.cross_edge_count) - expected) / math.sqrt(max(variance, 1e-12))
        same_pair = scores[
            (scores["sender_celltype"] == row.sender_celltype)
            & (scores["receiver_celltype"] == row.receiver_celltype)
        ]
        same_lr = scores[(scores["ligand"] == row.ligand) & (scores["receptor"] == row.receptor)]
        within_cellpair_rank = int(np.count_nonzero(same_pair["CCI_score"].to_numpy() > float(row.CCI_score)) + 1)
        within_lr_rank = int(np.count_nonzero(same_lr["CCI_score"].to_numpy() > float(row.CCI_score)) + 1)
        rows.append(
            {
                "axis_id": row.axis_id,
                "observed_cross_edge_count": int(row.cross_edge_count),
                "expected_cross_edge_count_abundance_null": float(expected),
                "cross_edge_enrichment_fold": float(row.cross_edge_count / max(expected, 1e-12)),
                "cross_edge_enrichment_z": float(z),
                "cross_edge_enrichment_p_approx": float(qnorm_sf(z)),
                "local_contact": float(row.local_contact),
                "contact_coverage": float(row.contact_coverage),
                "within_sender_receiver_rank": within_cellpair_rank,
                "within_sender_receiver_n": int(len(same_pair)),
                "within_sender_receiver_percentile": float(1.0 - (within_cellpair_rank - 1) / max(len(same_pair), 1)),
                "within_lr_pair_rank": within_lr_rank,
                "within_lr_pair_n": int(len(same_lr)),
                "within_lr_pair_percentile": float(1.0 - (within_lr_rank - 1) / max(len(same_lr), 1)),
            }
        )
    return pd.DataFrame(rows)


def standardized_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    full_common = root / "runs" / "full_common"
    for method_dir in sorted(full_common.glob("*")):
        if not method_dir.is_dir() or method_dir.name == "pyxenium":
            continue
        matches = sorted(method_dir.glob("*standardized.tsv")) + sorted(method_dir.glob("*standardized.tsv.gz"))
        if matches:
            paths[method_dir.name] = matches[0]
    return paths


def read_standardized_table(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    usecols = [
        "method",
        "ligand",
        "receptor",
        "sender",
        "receiver",
        "score_std",
        "rank_within_method",
        "fdr_or_pvalue",
        "spatial_support_type",
    ]
    return pd.read_csv(path, sep="\t", compression=compression, usecols=lambda col: col in usecols)


def cross_method_support(root: Path, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method, path in standardized_paths(root).items():
        table = read_standardized_table(path)
        for row in targets.itertuples(index=False):
            exact = table[
                (table["ligand"] == row.ligand)
                & (table["receptor"] == row.receptor)
                & (table["sender"] == row.sender_celltype)
                & (table["receiver"] == row.receiver_celltype)
            ]
            lr_any = table[(table["ligand"] == row.ligand) & (table["receptor"] == row.receptor)]
            axis_any = table[(table["sender"] == row.sender_celltype) & (table["receiver"] == row.receiver_celltype)]
            detail_rows.append(
                {
                    "axis_id": row.axis_id,
                    "method": method,
                    "exact_support": not exact.empty,
                    "same_lr_any_celltype_support": not lr_any.empty,
                    "same_sender_receiver_support": not axis_any.empty,
                    "exact_best_rank": float(exact["rank_within_method"].min()) if not exact.empty else np.nan,
                    "same_lr_best_rank": float(lr_any["rank_within_method"].min()) if not lr_any.empty else np.nan,
                    "same_lr_best_score_std": float(lr_any["score_std"].max()) if not lr_any.empty else np.nan,
                    "artifact_path": str(path),
                }
            )
    details = pd.DataFrame(detail_rows)
    for axis_id, grp in details.groupby("axis_id"):
        summary_rows.append(
            {
                "axis_id": axis_id,
                "cross_method_exact_count": int(grp["exact_support"].sum()),
                "cross_method_same_lr_count": int(grp["same_lr_any_celltype_support"].sum()),
                "cross_method_same_sender_receiver_count": int(grp["same_sender_receiver_support"].sum()),
                "supporting_methods_exact": ",".join(grp.loc[grp["exact_support"], "method"].astype(str)),
                "supporting_methods_same_lr": ",".join(grp.loc[grp["same_lr_any_celltype_support"], "method"].astype(str)),
            }
        )
    return pd.DataFrame(summary_rows), details


def receiver_panel_support(
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    mean_by_celltype: np.ndarray,
    gene_to_index: dict[str, int],
) -> pd.DataFrame:
    categories = list(pd.Categorical(meta["cell_type"].astype(str)).categories)
    axis_to_target = {axis.axis_id: axis for axis in TARGET_AXES}
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        axis = axis_to_target[row.axis_id]
        if row.receiver_celltype not in categories:
            continue
        receiver_idx = categories.index(row.receiver_celltype)
        present = [gene_to_index[g.upper()] for g in axis.target_panel if g.upper() in gene_to_index]
        if not present:
            rows.append({"axis_id": row.axis_id, "target_panel_present_n": 0})
            continue
        ratios = []
        for gene_idx in present:
            means = mean_by_celltype[gene_idx, :]
            ratios.append(float(means[receiver_idx] / max(float(np.max(means)), 1e-12)))
        rows.append(
            {
                "axis_id": row.axis_id,
                "target_panel_present_n": len(present),
                "receiver_context_panel_score": float(np.mean(ratios)),
                "receiver_context_panel_genes": ",".join([axis.target_panel[i] for i, g in enumerate(axis.target_panel) if g.upper() in gene_to_index]),
            }
        )
    return pd.DataFrame(rows)


def classify_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in evidence.itertuples(index=False):
        expr_ok = (
            pd.notna(row.ligand_sender_specificity_ratio)
            and pd.notna(row.receptor_receiver_specificity_ratio)
            and row.ligand_sender_specificity_ratio >= 0.75
            and row.receptor_receiver_specificity_ratio >= 0.50
        )
        spatial_ok = pd.notna(row.cross_edge_enrichment_z) and row.cross_edge_enrichment_z >= 2.0
        matched_ok = pd.notna(row.matched_gene_percentile) and row.matched_gene_percentile >= 0.90
        ablation_ok = pd.notna(row.max_rank_after_component_removal) and row.max_rank_after_component_removal <= 250
        cross_ok = (getattr(row, "cross_method_exact_count", 0) >= 1) or (getattr(row, "cross_method_same_lr_count", 0) >= 2)
        context_ok = pd.notna(row.receiver_context_panel_score) and row.receiver_context_panel_score >= 0.50
        support_count = sum([expr_ok, spatial_ok, matched_ok, ablation_ok, cross_ok, context_ok])
        if support_count >= 5:
            evidence_class = "strong"
        elif support_count >= 3:
            evidence_class = "moderate"
        else:
            evidence_class = "hypothesis_only"
        rows.append(
            {
                "axis_id": row.axis_id,
                "expression_support": expr_ok,
                "spatial_abundance_null_support": spatial_ok,
                "matched_gene_control_support": matched_ok,
                "component_ablation_support": ablation_ok,
                "cross_method_support": cross_ok,
                "receiver_context_support": context_ok,
                "support_count": support_count,
                "evidence_class": evidence_class,
            }
        )
    return pd.DataFrame(rows)


def make_evidence_figure(evidence: pd.DataFrame, output_dir: Path) -> None:
    support_cols = [
        "expression_support",
        "spatial_abundance_null_support",
        "matched_gene_control_support",
        "component_ablation_support",
        "cross_method_support",
        "receiver_context_support",
    ]
    plot_df = evidence.set_index("axis_label")[support_cols].astype(float)
    fig_height = max(4.2, 0.48 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    im = ax.imshow(plot_df.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(support_cols)))
    ax.set_xticklabels(
        [
            "Expression\nspecificity",
            "Spatial\nnull",
            "Matched-gene\ncontrol",
            "Component\nablation",
            "Cross-method\nsupport",
            "Receiver\ncontext",
        ],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_title("TopoLink-CCI validation evidence matrix", fontsize=13, weight="bold")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, "✓" if plot_df.iat[i, j] >= 0.5 else "", ha="center", va="center", color="white", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Evidence present", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(output_dir / "topolink_cci_validation_figure.png", dpi=300)
    fig.savefig(output_dir / "topolink_cci_validation_figure.pdf")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3g") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_csv(sep="\t", index=False)


def write_report(evidence: pd.DataFrame, controls: pd.DataFrame, output_dir: Path) -> None:
    report = output_dir / "topolink_cci_validation_scoreboard.md"
    strong = evidence[evidence["evidence_class"] == "strong"]
    moderate = evidence[evidence["evidence_class"] == "moderate"]
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# TopoLink-CCI validation evidence scoreboard\n\n")
        handle.write(
            "This PDC run applies the computational false-positive-control logic used by classic CCC/LR papers: "
            "expression specificity, spatial/label nulls, matched-gene controls, component ablation, "
            "cross-method triangulation, and receiver-context support. It is computational evidence, not wet-lab proof.\n\n"
        )
        handle.write("## Evidence classes\n\n")
        handle.write(f"- Strong axes: {len(strong)}\n")
        handle.write(f"- Moderate axes: {len(moderate)}\n")
        handle.write(f"- Hypothesis-only axes: {int((evidence['evidence_class'] == 'hypothesis_only').sum())}\n\n")
        cols = [
            "axis_label",
            "biology_label",
            "CCI_score",
            "global_rank",
            "evidence_class",
            "support_count",
            "cross_method_same_lr_count",
            "matched_gene_percentile",
            "cross_edge_enrichment_z",
            "max_rank_after_component_removal",
        ]
        handle.write(markdown_table(evidence[cols], floatfmt=".3g"))
        handle.write("\n\n")
        handle.write("## Control interpretation\n\n")
        handle.write(
            "- `spatial_abundance_null`: compares observed sender-receiver graph edges with a label-abundance null; "
            "it is conservative about exact geometry but catches cell-type-pair edge enrichment.\n"
        )
        handle.write(
            "- `matched_gene_control`: compares CCI sender-receiver expression specificity with genes matched by global mean and detection.\n"
        )
        handle.write(
            "- `lr_label_permutation`: reports where the candidate ranks within the same sender-receiver cell-type pair and within the same CCI pair across cell-type pairs.\n"
        )
        handle.write(
            "- `component_ablation`: recomputes geometric-mean LR scores after removing one pyXenium component at a time.\n\n"
        )
        handle.write("## References to validation patterns\n\n")
        handle.write(
            "- CellPhoneDB: curated LR database, complex filtering, and cell-label permutation specificity.\n"
            "- CellChat: mass-action communication probability, curated cofactors/complexes, and label permutation.\n"
            "- NicheNet: downstream receiver target-gene support rather than co-expression alone.\n"
            "- stLearn/SpatialDM/Squidpy: spatially constrained LR evidence and permutation/random-pair controls.\n"
            "- LIANA benchmark: multi-method consensus and rank aggregation because no single LR score is ground truth.\n"
        )
    controls.to_csv(output_dir / "topolink_cci_false_positive_controls.tsv", sep="\t", index=False)


def build_controls_table(
    spatial: pd.DataFrame,
    matched: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in spatial.itertuples(index=False):
        rows.append(
            {
                "axis_id": row.axis_id,
                "control_type": "spatial_abundance_null",
                "observed": row.observed_cross_edge_count,
                "expected_or_control_mean": row.expected_cross_edge_count_abundance_null,
                "z_or_rank": row.cross_edge_enrichment_z,
                "p_or_percentile": row.cross_edge_enrichment_p_approx,
                "note": "cell-type-label abundance null for graph edge count",
            }
        )
        rows.append(
            {
                "axis_id": row.axis_id,
                "control_type": "lr_label_permutation_same_sender_receiver",
                "observed": row.within_sender_receiver_rank,
                "expected_or_control_mean": row.within_sender_receiver_n,
                "z_or_rank": row.within_sender_receiver_rank,
                "p_or_percentile": row.within_sender_receiver_percentile,
                "note": "rank percentile against all CCI pairs in the same sender-receiver cell-type pair",
            }
        )
        rows.append(
            {
                "axis_id": row.axis_id,
                "control_type": "lr_label_permutation_same_lr_pair",
                "observed": row.within_lr_pair_rank,
                "expected_or_control_mean": row.within_lr_pair_n,
                "z_or_rank": row.within_lr_pair_rank,
                "p_or_percentile": row.within_lr_pair_percentile,
                "note": "rank percentile for the same CCI pair across cell-type pairs",
            }
        )
    for row in matched.itertuples(index=False):
        rows.append(
            {
                "axis_id": row.axis_id,
                "control_type": "matched_gene_expression_control",
                "observed": getattr(row, "observed_expression_specificity_score", np.nan),
                "expected_or_control_mean": getattr(row, "control_mean", np.nan),
                "z_or_rank": getattr(row, "matched_gene_z", np.nan),
                "p_or_percentile": getattr(row, "matched_gene_percentile", np.nan),
                "note": "matched by global mean expression and detection fraction",
            }
        )
    for axis_id, grp in ablation.groupby("axis_id"):
        rows.append(
            {
                "axis_id": axis_id,
                "control_type": "component_ablation_max_rank",
                "observed": int(grp["rank_without_component"].max()),
                "expected_or_control_mean": int(grp["rank_without_component"].min()),
                "z_or_rank": int(grp["rank_without_component"].max()),
                "p_or_percentile": np.nan,
                "note": "worst and best rank after removing one score component",
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "runs" / "validation" / "topolink_cci_false_positive_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    params = vars(args).copy()
    params["root"] = str(root)
    params["output_dir"] = str(output_dir)
    with (output_dir / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    manifest = load_manifest(root)
    scores = read_scores(root)
    targets = find_target_rows(scores)
    ablation = component_ablation(scores, targets)

    genes, meta, matrix = load_bundle_expression(manifest)
    expr_summary, mean_by_celltype, det_by_celltype, gene_to_index = summarize_expression(genes, meta, matrix)
    expr = expression_specificity(targets, meta, mean_by_celltype, det_by_celltype, gene_to_index)
    matched = matched_gene_controls(
        targets,
        meta,
        mean_by_celltype,
        det_by_celltype,
        gene_to_index,
        n_controls=args.n_matched_controls,
        seed=args.seed,
    )
    spatial = spatial_abundance_null(scores, targets, meta)
    cross_summary, cross_detail = cross_method_support(root, targets)
    receiver_context = receiver_panel_support(targets, meta, mean_by_celltype, gene_to_index)

    ablation_summary = (
        ablation.groupby("axis_id")["rank_without_component"]
        .agg(max_rank_after_component_removal="max", min_rank_after_component_removal="min")
        .reset_index()
    )
    evidence = (
        targets[
            [
                "axis_id",
                "biology_label",
                "ligand",
                "receptor",
                "sender_celltype",
                "receiver_celltype",
                "CCI_score",
                "global_rank",
                *COMPONENTS,
                "cross_edge_count",
            ]
        ]
        .merge(expr, on=["axis_id", "ligand", "receptor"], how="left")
        .merge(spatial, on="axis_id", how="left")
        .merge(matched, on="axis_id", how="left")
        .merge(ablation_summary, on="axis_id", how="left")
        .merge(cross_summary, on="axis_id", how="left")
        .merge(receiver_context, on="axis_id", how="left")
    )
    class_df = classify_evidence(evidence)
    evidence = evidence.merge(class_df, on="axis_id", how="left")
    evidence["axis_label"] = evidence["ligand"] + "-" + evidence["receptor"] + "\n" + evidence["sender_celltype"] + " -> " + evidence["receiver_celltype"]
    evidence = evidence.sort_values("CCI_score", ascending=False, kind="mergesort")

    controls = build_controls_table(spatial, matched, ablation)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    evidence.to_csv(tables / "topolink_cci_validation_evidence.tsv", sep="\t", index=False)
    controls.to_csv(tables / "topolink_cci_false_positive_controls.tsv", sep="\t", index=False)
    ablation.to_csv(tables / "topolink_cci_component_ablation.tsv", sep="\t", index=False)
    expr_summary.to_csv(tables / "topolink_cci_expression_gene_specificity.tsv", sep="\t", index=False)
    matched.to_csv(tables / "topolink_cci_matched_gene_controls.tsv", sep="\t", index=False)
    spatial.to_csv(tables / "topolink_cci_spatial_null_summary.tsv", sep="\t", index=False)
    cross_detail.to_csv(tables / "topolink_cci_cross_method_support_detail.tsv", sep="\t", index=False)

    make_evidence_figure(evidence, figures)
    write_report(evidence, controls, output_dir)

    summary = {
        "status": "success",
        "runtime_seconds": round(time.time() - started, 3),
        "n_scores": int(len(scores)),
        "n_target_axes": int(len(targets)),
        "target_axes": evidence[["axis_id", "evidence_class", "support_count"]].to_dict(orient="records"),
        "outputs": {
            "evidence": str(tables / "topolink_cci_validation_evidence.tsv"),
            "controls": str(tables / "topolink_cci_false_positive_controls.tsv"),
            "scoreboard": str(output_dir / "topolink_cci_validation_scoreboard.md"),
            "figure": str(figures / "topolink_cci_validation_figure.png"),
        },
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TopoLink-CCI computational validation evidence on PDC.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="PDC benchmark root.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults under ROOT/runs/validation.")
    parser.add_argument("--n-matched-controls", type=int, default=250, help="Matched random gene-pair controls per target axis.")
    parser.add_argument("--seed", type=int, default=20260428)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
