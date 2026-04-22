from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyXenium.datasets import RENAL_FFPE_PROTEIN_10X_DATASET

from ..immune_resistance import (
    DEFAULT_BRANCH_MODELS,
    DEFAULT_RESISTANT_NICHES,
    annotate_joint_cell_states,
    build_spatial_niches,
    compute_rna_protein_discordance,
    score_immune_resistance_program,
)
from ..loading import load_rna_protein_anndata
from .renal_ffpe_protein import DEFAULT_DATASET_PATH

DEFAULT_METADATA_SPEC = [
    ("sample_id", "string", "Stable sample identifier across assays."),
    ("tumor_subtype", "string", "Histologic or molecular renal cancer subtype."),
    ("anatomic_site", "string", "Primary site, metastasis, or thrombus origin."),
    ("grade_stage", "string", "Grade/stage or other malignancy severity descriptor."),
    ("therapy_exposure", "string", "Pretreatment, on-treatment, post-treatment, or therapy-naive."),
    ("ici_response", "string", "Response category for immunotherapy when available."),
    ("pfs_os", "string", "Progression-free or overall survival linkage field."),
    ("block_roi_id", "string", "FFPE block identifier and ROI identity."),
]

DEFAULT_PANEL_GAPS = [
    ("SPP1", "myeloid", "TAM polarization and SPP1-CD44 resistance axis."),
    ("CD44", "tumor_stromal", "Ligand partner for SPP1-linked resistance states."),
    ("TREM2", "myeloid", "Suppressive macrophage state refinement."),
    ("CXCL13", "lymphoid", "TLS-like and B-cell attracting programs."),
    ("FAP", "stromal", "Activated fibroblast and invasive front context."),
    ("FOXP3", "immune", "Regulatory T-cell context."),
    ("TIGIT", "immune", "Checkpoint exhaustion refinement."),
    ("TIM3", "immune", "Checkpoint exhaustion refinement."),
    ("B2M/HLA-I", "tumor_immune", "Antigen-presentation loss."),
    ("VEGFA/KDR/FLT1", "vascular", "Angiogenic signaling specificity."),
    ("COL4A1/COL4A2", "matrix", "Perivascular collagen belt context."),
]

DEFAULT_BRANCH_VALIDATION = {
    "myeloid_vascular": "mIF/IHC for CD68 or CD163 with CD31 and alphaSMA, plus VISTA or HLA-DR at perivascular belts.",
    "epithelial_emt_front": "mIF/RNAscope for PanCK, Vimentin, alphaSMA, and checkpoint-active myeloid markers at EMT transition fronts.",
}

DEFAULT_BRANCH_MARKER_HINTS = {
    "myeloid_vascular": {"CD68", "CD163", "HLA-DR", "CD31", "alphaSMA", "VISTA", "PD-L1"},
    "epithelial_emt_front": {"PanCK", "E-Cadherin", "Vimentin", "alphaSMA", "VISTA", "PD-L1", "LAG-3"},
}


def _resolve_output_dir(output_dir: str | None, manuscript_mode: bool, manuscript_root: str, sample_id: str) -> Path | None:
    if output_dir:
        return Path(output_dir)
    if manuscript_mode:
        return Path(manuscript_root) / "renal_immune_resistance" / sample_id
    return None


def _coords_from_adata(adata) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError("Expected adata.obsm['spatial'] for plotting.")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    return coords[:, :2]


def _scatter_categorical(coords: np.ndarray, values: pd.Series, output_path: Path, title: str) -> None:
    cats = values.astype("category")
    codes = cats.cat.codes.to_numpy()
    mask = codes != -1
    cmap = plt.get_cmap("tab20", max(len(cats.cat.categories), 1))
    plt.figure(figsize=(7, 6))
    if mask.any():
        sc = plt.scatter(coords[mask, 0], coords[mask, 1], c=codes[mask], s=1.0, cmap=cmap, alpha=0.85)
        cb = plt.colorbar(sc, shrink=0.8)
        cb.set_ticks(np.arange(len(cats.cat.categories)))
        cb.set_ticklabels(list(cats.cat.categories))
    if (~mask).any():
        plt.scatter(coords[~mask, 0], coords[~mask, 1], c="lightgrey", s=1.0, alpha=0.6)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _scatter_numeric(coords: np.ndarray, values: np.ndarray, output_path: Path, title: str, cmap: str = "viridis") -> None:
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=values, s=1.0, cmap=cmap, alpha=0.85)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, shrink=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_roi_panel_summary(payload: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    state_df = pd.DataFrame(payload["joint_cell_states"]).head(6)
    axes[0, 0].barh(state_df["state"], state_df["n_cells"], color="#4f81bd")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Joint Cell States")

    niche_df = pd.DataFrame(payload["top_spatial_niches"]).head(6)
    axes[0, 1].barh(niche_df["niche"], niche_df["mean_score"], color="#c0504d")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("Top Spatial Niches")

    branch_df = pd.DataFrame(payload["branch_summary"]).head(6)
    axes[1, 0].barh(branch_df["branch"], branch_df["benchmark_score"], color="#9bbb59")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Best Branch Models")

    marker_df = pd.DataFrame(payload["top_marker_discordance"]).head(6)
    axes[1, 1].barh(marker_df["label"], marker_df["mean_abs_discordance"], color="#8064a2")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Top Marker Discordance")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_cohort_handoff_spec() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_METADATA_SPEC, columns=["field", "type", "description"])


def build_panel_gap_table() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_PANEL_GAPS, columns=["marker_or_axis", "domain", "rationale"])


def extract_ranked_patches(study: dict, *, branch: str, top_n: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_df = study["immune_resistance"]["roi_scores"].copy()
    branch_summary = study["immune_resistance"]["branch_summary"]
    best_row = branch_summary[branch_summary["branch"] == branch]
    if best_row.empty:
        return pd.DataFrame(), pd.DataFrame()

    best_model = str(best_row.iloc[0]["best_model"])
    score_col = f"immune_resistance__model__{best_model}"
    target_col = f"immune_resistance__branch_target__{branch}"
    if score_col not in roi_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    candidate = roi_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    resistant = candidate.head(top_n).copy()
    resistant["branch"] = branch
    resistant["role"] = "resistant_patch"
    resistant["best_model"] = best_model

    control_pool = roi_df.sort_values(score_col, ascending=True).head(max(top_n * 4, top_n)).copy()
    control_rows = []
    used_idx: set[int] = set()
    for _, row in resistant.iterrows():
        subset = control_pool.loc[~control_pool.index.isin(used_idx)].copy()
        if subset.empty:
            break
        subset["n_cell_distance"] = (subset["n_cells"] - row["n_cells"]).abs()
        match = subset.sort_values(["n_cell_distance", score_col], ascending=[True, True]).iloc[0]
        used_idx.add(int(match.name))
        matched = match.to_dict()
        matched["branch"] = branch
        matched["role"] = "matched_control"
        matched["matched_to_region"] = f"{int(row['region_x_bin'])}_{int(row['region_y_bin'])}"
        matched["best_model"] = best_model
        control_rows.append(matched)

    control = pd.DataFrame(control_rows)
    if target_col in resistant.columns and target_col in control.columns:
        resistant = resistant.sort_values(target_col, ascending=False).reset_index(drop=True)
        control = control.sort_values(target_col, ascending=True).reset_index(drop=True)
    return resistant, control


def build_top_hypotheses_table(study: dict) -> pd.DataFrame:
    marker_df = study["immune_resistance"]["marker_neighborhood_enrichment"]
    branch_df = study["immune_resistance"]["branch_summary"]
    rows = []
    for branch in DEFAULT_RESISTANT_NICHES:
        branch_best = branch_df[branch_df["branch"] == branch]
        if branch_best.empty:
            continue
        marker_best = marker_df[marker_df["branch"] == branch]
        if not marker_best.empty:
            hinted = marker_best[marker_best["protein"].isin(DEFAULT_BRANCH_MARKER_HINTS[branch])]
            marker_best = hinted if not hinted.empty else marker_best
            marker_best = marker_best.sort_values("abs_rank_score", ascending=False).head(1)
        if marker_best.empty:
            key_marker = "NA"
            key_gene = "NA"
            marker_corr = np.nan
        else:
            key_marker = str(marker_best.iloc[0]["protein"])
            key_gene = str(marker_best.iloc[0]["gene"])
            marker_corr = float(marker_best.iloc[0]["abs_target_correlation"])

        best = branch_best.iloc[0]
        rows.append(
            {
                "branch": branch,
                "hypothesis": (
                    "Perivascular immune-resistance belt"
                    if branch == "myeloid_vascular"
                    else "Checkpoint-active myeloid enrichment at EMT transition fronts"
                ),
                "best_model": best["best_model"],
                "benchmark_score": float(best["benchmark_score"]),
                "held_out_auc": float(best["held_out_auc"]),
                "spatial_coherence": float(best["spatial_coherence"]),
                "roi_reproducibility": float(best["roi_reproducibility"]),
                "key_marker": key_marker,
                "key_gene": key_gene,
                "marker_target_correlation": marker_corr,
                "suggested_validation": DEFAULT_BRANCH_VALIDATION[branch],
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark_score", ascending=False, na_position="last").reset_index(drop=True)


def build_serializable_pilot_summary(study: dict, *, top_n: int = 10) -> dict:
    state_summary = study["state_annotation"]["state_summary"].head(top_n)
    class_summary = study["state_annotation"]["class_summary"].head(top_n)
    marker_summary = study["discordance"]["marker_summary"].head(top_n)
    pathway_summary = study["discordance"]["pathway_summary"].head(top_n)
    niche_summary = study["niches"]["summary"].head(top_n)
    model_comparison = study["immune_resistance"]["model_comparison"].copy()
    branch_summary = study["immune_resistance"]["branch_summary"].copy()
    hypotheses = build_top_hypotheses_table(study)

    best_models = branch_summary.to_dict(orient="records")
    best_model_names = {str(row["best_model"]) for row in best_models}
    if any(name in {"rna_only", "protein_only", "checkpoint_only", "myeloid_only", "vascular_only", "emt_only"} for name in best_model_names):
        report_positioning = (
            "Branch-specific axis models localize the pilot signal more clearly than pooled summaries; "
            "the current claim should emphasize localization rather than universal superiority."
        )
    elif any(name in {"myeloid_vascular_branch", "epithelial_emt_front_branch", "joint_activity"} for name in best_model_names):
        report_positioning = "Joint analysis provides the strongest branch-level benchmark signal in this pilot sample."
    else:
        report_positioning = "Joint analysis localizes orthogonal biology in this pilot sample; RNA-only is not universally outperformed."

    return {
        "sample_id": study["sample_id"],
        "dataset_title": study["dataset_title"],
        "dataset_url": study["dataset_url"],
        "base_path": study["base_path"],
        "prefer": study["prefer"],
        "n_cells": study["n_cells"],
        "n_rna_features": study["n_rna_features"],
        "n_protein_markers": study["n_protein_markers"],
        "resistant_niches": list(DEFAULT_RESISTANT_NICHES),
        "sample_summary": study["immune_resistance"]["sample_summary"],
        "joint_cell_classes": class_summary.to_dict(orient="records"),
        "joint_cell_states": state_summary.to_dict(orient="records"),
        "top_marker_discordance": marker_summary.to_dict(orient="records"),
        "top_pathway_discordance": pathway_summary.to_dict(orient="records"),
        "top_spatial_niches": niche_summary.to_dict(orient="records"),
        "branch_summary": best_models,
        "top_hypotheses": hypotheses.to_dict(orient="records"),
        "claim_positioning": report_positioning,
    }


def render_renal_immune_resistance_report(payload: dict) -> str:
    lines = [
        "# Renal Immune Resistance Discovery Package",
        "",
        f"Dataset: {payload['dataset_title']}",
        f"Source: {payload['dataset_url']}",
        f"Local path: `{payload['base_path']}`",
        f"Backend preference: `{payload['prefer']}`",
        "",
        "## Core Summary",
        "",
        f"- Cells: `{payload['n_cells']}`",
        f"- RNA features: `{payload['n_rna_features']}`",
        f"- Protein markers: `{payload['n_protein_markers']}`",
        f"- Resistant niches tracked: `{', '.join(payload['resistant_niches'])}`",
        f"- Positioning: {payload['claim_positioning']}",
        "",
        "## Joint Cell Classes",
        "",
    ]
    for row in payload["joint_cell_classes"]:
        lines.append(f"- `{row['class']}`: `{row['n_cells']}` cells")

    lines.extend(["", "## Joint Cell States", ""])
    for row in payload["joint_cell_states"]:
        lines.append(f"- `{row['state']}`: `{row['n_cells']}` cells")

    lines.extend(["", "## Top Marker Discordance", ""])
    for row in payload["top_marker_discordance"]:
        lines.append(
            f"- `{row['label']}` ({row['protein']} / {row['gene']}): "
            f"mean abs discordance `{row['mean_abs_discordance']:.3f}`, "
            f"signed bias `{row['mean_signed_discordance']:.3f}`"
        )

    lines.extend(["", "## Top Pathway Discordance", ""])
    for row in payload["top_pathway_discordance"]:
        lines.append(
            f"- `{row['pathway']}`: mean abs discordance `{row['mean_abs_discordance']:.3f}`, "
            f"signed bias `{row['mean_signed_discordance']:.3f}`"
        )

    lines.extend(["", "## Branch Summary", ""])
    for row in payload["branch_summary"]:
        lines.append(
            f"- `{row['branch']}`: best model `{row['best_model']}`, "
            f"benchmark `{row['benchmark_score']:.3f}`, held-out AUC `{row['held_out_auc']:.3f}`"
        )

    lines.extend(["", "## Top Hypotheses", ""])
    for row in payload["top_hypotheses"]:
        lines.append(
            f"- `{row['branch']}`: {row['hypothesis']} "
            f"(marker `{row['key_marker']}`, benchmark `{row['benchmark_score']:.3f}`, "
            f"validation: {row['suggested_validation']})"
        )
    lines.append("")
    return "\n".join(lines)


def write_renal_immune_resistance_artifacts(
    study: dict,
    output_dir: str | Path | None,
    *,
    top_n: int = 10,
    export_figures: bool = True,
) -> Path | None:
    if output_dir is None:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_serializable_pilot_summary(study, top_n=top_n)

    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_renal_immune_resistance_report(payload), encoding="utf-8")
    study["state_annotation"]["class_summary"].to_csv(out / "joint_cell_classes.csv", index=False)
    study["state_annotation"]["state_summary"].to_csv(out / "joint_cell_states.csv", index=False)
    study["discordance"]["marker_summary"].to_csv(out / "marker_discordance.csv", index=False)
    study["discordance"]["pathway_summary"].to_csv(out / "pathway_discordance.csv", index=False)
    study["niches"]["summary"].to_csv(out / "spatial_niches.csv", index=False)
    study["immune_resistance"]["branch_summary"].to_csv(out / "branch_summary.csv", index=False)
    study["immune_resistance"]["model_comparison"].to_csv(out / "ablation_summary.csv", index=False)
    study["immune_resistance"]["marker_neighborhood_enrichment"].to_csv(out / "marker_neighborhood_enrichment.csv", index=False)
    study["immune_resistance"]["roi_scores"].to_csv(out / "roi_scores.csv", index=False)

    hypotheses = build_top_hypotheses_table(study)
    hypotheses.to_csv(out / "top_hypotheses.csv", index=False)
    build_cohort_handoff_spec().to_csv(out / "cohort_metadata_spec.csv", index=False)
    build_panel_gap_table().to_csv(out / "panel_gap_recommendations.csv", index=False)

    resistant_tables = []
    control_tables = []
    for branch in DEFAULT_RESISTANT_NICHES:
        resistant, control = extract_ranked_patches(study, branch=branch, top_n=top_n)
        if not resistant.empty:
            resistant_tables.append(resistant)
        if not control.empty:
            control_tables.append(control)
    if resistant_tables:
        pd.concat(resistant_tables, ignore_index=True).to_csv(out / "roi_resistant_patches.csv", index=False)
    if control_tables:
        pd.concat(control_tables, ignore_index=True).to_csv(out / "roi_control_patches.csv", index=False)

    if export_figures:
        figures_dir = out / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        coords = _coords_from_adata(study["adata"])
        _scatter_categorical(coords, study["adata"].obs["joint_cell_state"], figures_dir / "state_map.png", "Joint Cell State Map")
        _scatter_categorical(coords, study["adata"].obs["spatial_niche"], figures_dir / "niche_map.png", "Spatial Niche Map")

        top_marker = study["discordance"]["marker_summary"].iloc[0]["label"]
        marker_col = f"discordance__marker__{top_marker}__abs"
        _scatter_numeric(
            coords,
            study["adata"].obs[marker_col].to_numpy(dtype=float),
            figures_dir / "top_marker_discordance_map.png",
            f"Top Marker Discordance: {top_marker}",
        )
        _plot_roi_panel_summary(payload, figures_dir / "roi_panel_summary.png")

    return out


def run_renal_immune_resistance_pilot(
    *,
    base_path: str,
    prefer: str = "auto",
    sample_id: str = "renal_ffpe_public_10x",
    n_neighbors: int = 15,
    region_bins: int = 24,
    output_json: str | None = None,
    output_dir: str | None = None,
    write_h5ad: str | None = None,
    top_n: int = 10,
    manuscript_mode: bool = False,
    manuscript_root: str = "manuscript",
    export_figures: bool = True,
) -> dict:
    adata = load_rna_protein_anndata(base_path=base_path, prefer=prefer)
    state_result = annotate_joint_cell_states(adata)
    discordance_result = compute_rna_protein_discordance(
        adata,
        n_neighbors=n_neighbors,
        region_bins=region_bins,
    )
    niche_result = build_spatial_niches(adata, n_neighbors=n_neighbors)
    immune_resistance_result = score_immune_resistance_program(
        adata,
        discordance_result=discordance_result,
        niche_result=niche_result,
        region_bins=region_bins,
        n_neighbors=n_neighbors,
    )

    study = {
        "sample_id": sample_id,
        "dataset_title": RENAL_FFPE_PROTEIN_10X_DATASET.title,
        "dataset_url": RENAL_FFPE_PROTEIN_10X_DATASET.url,
        "base_path": base_path,
        "prefer": prefer,
        "n_cells": int(adata.n_obs),
        "n_rna_features": int(adata.n_vars),
        "n_protein_markers": int(getattr(adata.obsm["protein"], "shape", (0, 0))[1]),
        "state_annotation": state_result,
        "discordance": discordance_result,
        "niches": niche_result,
        "immune_resistance": immune_resistance_result,
        "adata": adata,
    }

    payload = build_serializable_pilot_summary(study, top_n=top_n)
    study["payload"] = payload

    if output_json:
        Path(output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    resolved_output = _resolve_output_dir(output_dir, manuscript_mode, manuscript_root, sample_id)
    artifact_dir = write_renal_immune_resistance_artifacts(
        study,
        resolved_output,
        top_n=top_n,
        export_figures=export_figures,
    )
    if artifact_dir is not None:
        study["artifact_dir"] = str(artifact_dir)

    if write_h5ad:
        h5ad_path = Path(write_h5ad)
        h5ad_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(h5ad_path)

    return study


__all__ = [
    "DEFAULT_DATASET_PATH",
    "build_cohort_handoff_spec",
    "build_panel_gap_table",
    "build_serializable_pilot_summary",
    "build_top_hypotheses_table",
    "extract_ranked_patches",
    "render_renal_immune_resistance_report",
    "run_renal_immune_resistance_pilot",
    "write_renal_immune_resistance_artifacts",
]
