from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create manuscript source tables and methods notes for a morphopathway smoke run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def _read_sample(run_dir: Path, sample: str) -> dict[str, pd.DataFrame]:
    sample_dir = run_dir / sample
    required = {
        "associations": sample_dir / "association_table.csv",
        "coverage": sample_dir / "pathway_coverage.csv",
        "spatial_nulls": sample_dir / "spatial_nulls.csv",
        "negative_controls": sample_dir / "negative_controls.csv",
        "figure_source": sample_dir / "figure_source_table.csv",
        "input_manifest": sample_dir / "input_manifest.csv",
        "he_manifest": sample_dir / "he_feature_manifest.csv",
        "spatial_block_manifest": sample_dir / "spatial_block_manifest.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run {sample!r} is missing required files: {missing}")
    return {name: pd.read_csv(path) for name, path in required.items()}


def _pass_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].fillna(False)
    if pd.api.types.is_bool_dtype(values):
        return int(values.sum())
    return int(values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum())


def _feature_axis_diagnostics(sample_role: str, sample: dict[str, pd.DataFrame], *, top_k: int = 40) -> pd.DataFrame:
    associations = sample["associations"].copy()
    negative = sample["negative_controls"].copy()
    if associations.empty or "image_feature" not in associations.columns:
        return pd.DataFrame()

    top = associations.head(top_k).copy()
    top_summary = (
        top.groupby("image_feature", as_index=False)
        .agg(
            top_k_count=("pathway", "size"),
            top_k_pathways=("pathway", lambda values: ";".join(sorted(set(map(str, values))))),
            top_k_families=("family", lambda values: ";".join(sorted(set(map(str, values))))),
        )
    )
    assoc_summary = (
        associations.groupby("image_feature", as_index=False)
        .agg(
            n_associations=("pathway", "size"),
            max_abs_partial_spearman_rho=("abs_partial_spearman_rho", "max"),
            mean_abs_partial_spearman_rho=("abs_partial_spearman_rho", "mean"),
        )
    )
    if negative.empty:
        negative_summary = pd.DataFrame(columns=["image_feature", "negative_control_rows", "negative_control_fail95", "negative_control_fail99"])
    else:
        neg = negative.copy()
        neg["fail95"] = ~neg["passes_negative_control_95"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        neg["fail99"] = ~neg["passes_negative_control_99"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        negative_summary = (
            neg.groupby("image_feature", as_index=False)
            .agg(
                negative_control_rows=("pathway", "size"),
                negative_control_fail95=("fail95", "sum"),
                negative_control_fail99=("fail99", "sum"),
                negative_control_pathways=("pathway", lambda values: ";".join(sorted(set(map(str, values))))),
            )
        )

    diagnostics = assoc_summary.merge(top_summary, on="image_feature", how="left").merge(
        negative_summary, on="image_feature", how="left"
    )
    for column in ["top_k_count", "negative_control_rows", "negative_control_fail95", "negative_control_fail99"]:
        diagnostics[column] = pd.to_numeric(diagnostics[column], errors="coerce").fillna(0).astype(int)
    diagnostics["top_k_pathways"] = diagnostics["top_k_pathways"].fillna("")
    diagnostics["top_k_families"] = diagnostics["top_k_families"].fillna("")
    diagnostics["negative_control_pathways"] = diagnostics["negative_control_pathways"].fillna("")
    diagnostics["sample_role"] = sample_role
    diagnostics["top_k"] = int(top_k)
    diagnostics["candidate_generic_axis"] = (diagnostics["top_k_count"] >= 3) & (
        diagnostics["negative_control_fail95"] >= 1
    )
    return diagnostics.sort_values(
        ["candidate_generic_axis", "negative_control_fail95", "top_k_count", "max_abs_partial_spearman_rho"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _summarize_cross_cancer(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    validation_abs_rho_threshold: float,
) -> pd.DataFrame:
    discovery = discovery.copy()
    validation = validation.copy()
    discovery["abs_partial_spearman_rho"] = pd.to_numeric(
        discovery["abs_partial_spearman_rho"], errors="coerce"
    ).fillna(0.0)
    validation["abs_partial_spearman_rho"] = pd.to_numeric(
        validation["abs_partial_spearman_rho"], errors="coerce"
    ).fillna(0.0)
    discovery_best = (
        discovery.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("pathway", keep="first")
        .reset_index(drop=True)
    )
    validation_best_pathway = (
        validation.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("pathway", keep="first")
        .set_index("pathway")
    )
    validation_best_family = (
        validation.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("family", keep="first")
        .set_index("family")
    )

    rows: list[dict[str, object]] = []
    for _, row in discovery_best.iterrows():
        pathway = str(row["pathway"])
        family = str(row["family"])
        pathway_match = validation_best_pathway.loc[pathway] if pathway in validation_best_pathway.index else None
        family_match = validation_best_family.loc[family] if family in validation_best_family.index else None
        pathway_abs = float(pathway_match["abs_partial_spearman_rho"]) if pathway_match is not None else float("nan")
        family_abs = float(family_match["abs_partial_spearman_rho"]) if family_match is not None else float("nan")
        recovered = bool(
            (np.isfinite(pathway_abs) and pathway_abs >= validation_abs_rho_threshold)
            or (np.isfinite(family_abs) and family_abs >= validation_abs_rho_threshold)
        )
        rows.append(
            {
                "pathway": pathway,
                "family": family,
                "discovery_abs_partial_rho": float(row["abs_partial_spearman_rho"]),
                "validation_pathway_abs_partial_rho": pathway_abs,
                "validation_family_best_pathway": str(family_match["pathway"]) if family_match is not None else "",
                "validation_family_abs_partial_rho": family_abs,
                "validation_abs_rho_threshold": float(validation_abs_rho_threshold),
                "recovered_in_validation": recovered,
                "validation_call": "pathway_or_family_recovered" if recovered else "not_recovered",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or (run_dir / "evidence_pack")
    output_dir.mkdir(parents=True, exist_ok=True)

    breast = _read_sample(run_dir, "breast_discovery")
    cervical = _read_sample(run_dir, "cervical_validation")
    validation_path = run_dir / "cross_cancer_validation.csv"
    summary_path = run_dir / "smoke_summary.csv"
    if not validation_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run directory must contain cross_cancer_validation.csv and smoke_summary.csv.")
    validation = pd.read_csv(validation_path)
    summary = pd.read_csv(summary_path)
    workflow_manifest_path = run_dir / "workflow_manifest.json"
    if workflow_manifest_path.exists():
        workflow_manifest = json.loads(workflow_manifest_path.read_text(encoding="utf-8"))
    else:
        workflow_manifest = {}
    validation_threshold = float(workflow_manifest.get("validation_threshold", 0.20))

    top_n = int(args.top_n)
    fig1 = breast["associations"].head(top_n).copy()
    fig1.insert(0, "sample_role", "breast_discovery")
    fig1.to_csv(output_dir / "fig1_breast_discovery_he_pathway_associations.csv", index=False)

    fig2 = validation.sort_values(
        ["recovered_in_validation", "validation_family_abs_partial_rho", "discovery_abs_partial_rho"],
        ascending=[False, False, False],
        kind="stable",
    ).copy()
    fig2.to_csv(output_dir / "fig2_cross_cancer_pathway_family_validation.csv", index=False)

    figure_family = pd.concat(
        [
            breast["figure_source"].assign(sample_role="breast_discovery"),
            cervical["figure_source"].assign(sample_role="cervical_validation"),
        ],
        ignore_index=True,
    )
    figure_family.to_csv(output_dir / "fig_source_top_family_associations.csv", index=False)

    pd.concat(
        [
            breast["coverage"].assign(sample_role="breast_discovery"),
            cervical["coverage"].assign(sample_role="cervical_validation"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "supp_table_pathway_coverage.csv", index=False)
    pd.concat(
        [
            breast["spatial_nulls"].assign(sample_role="breast_discovery"),
            cervical["spatial_nulls"].assign(sample_role="cervical_validation"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "supp_table_spatial_nulls.csv", index=False)
    pd.concat(
        [
            breast["negative_controls"].assign(sample_role="breast_discovery"),
            cervical["negative_controls"].assign(sample_role="cervical_validation"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "supp_table_matched_random_gene_controls.csv", index=False)
    feature_axis = pd.concat(
        [
            _feature_axis_diagnostics("breast_discovery", breast),
            _feature_axis_diagnostics("cervical_validation", cervical),
        ],
        ignore_index=True,
    )
    feature_axis.to_csv(output_dir / "supp_table_plip_feature_axis_diagnostics.csv", index=False)
    candidate_axes = feature_axis.loc[feature_axis["candidate_generic_axis"]].copy()
    flagged_breast = set(
        candidate_axes.loc[candidate_axes["sample_role"] == "breast_discovery", "image_feature"].astype(str)
    )
    flagged_cervical = set(
        candidate_axes.loc[candidate_axes["sample_role"] == "cervical_validation", "image_feature"].astype(str)
    )
    axis_masked_validation = _summarize_cross_cancer(
        breast["associations"].loc[~breast["associations"]["image_feature"].astype(str).isin(flagged_breast)],
        cervical["associations"].loc[~cervical["associations"]["image_feature"].astype(str).isin(flagged_cervical)],
        validation_abs_rho_threshold=validation_threshold,
    )
    axis_masked_validation.to_csv(
        output_dir / "fig2_cross_cancer_validation_without_candidate_plip_axes.csv", index=False
    )
    pd.concat(
        [
            breast["input_manifest"].assign(sample_role="breast_discovery"),
            cervical["input_manifest"].assign(sample_role="cervical_validation"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "supp_table_input_manifests.csv", index=False)

    breast_top = breast["associations"].iloc[0]
    cervical_top = cervical["associations"].iloc[0]
    n_recovered = int(validation["recovered_in_validation"].sum())
    n_total = int(len(validation))
    gate_counts = {
        "breast_spatial_null_pass95": _pass_count(breast["spatial_nulls"], "passes_spatial_null_95"),
        "breast_spatial_null_pass99": _pass_count(breast["spatial_nulls"], "passes_spatial_null_99"),
        "cervical_spatial_null_pass95": _pass_count(cervical["spatial_nulls"], "passes_spatial_null_95"),
        "cervical_spatial_null_pass99": _pass_count(cervical["spatial_nulls"], "passes_spatial_null_99"),
        "breast_negative_control_pass95": _pass_count(breast["negative_controls"], "passes_negative_control_95"),
        "breast_negative_control_pass99": _pass_count(breast["negative_controls"], "passes_negative_control_99"),
        "cervical_negative_control_pass95": _pass_count(cervical["negative_controls"], "passes_negative_control_95"),
        "cervical_negative_control_pass99": _pass_count(cervical["negative_controls"], "passes_negative_control_99"),
    }
    breast_backend = str(breast["input_manifest"].get("he_embedding_backend_status", pd.Series([""])).iloc[0])
    cervical_backend = str(cervical["input_manifest"].get("he_embedding_backend_status", pd.Series([""])).iloc[0])
    backend_note = (
        f"Breast H&E embedding backend status: `{breast_backend or 'none'}`; "
        f"cervical H&E embedding backend status: `{cervical_backend or 'none'}`."
    )
    if candidate_axes.empty:
        candidate_axis_note = "No candidate generic PLIP axes were flagged by repeated top-40 usage plus matched-control failure."
    else:
        candidate_axis_note = "; ".join(
            f"{row.sample_role}/{row.image_feature} "
            f"(top40={int(row.top_k_count)}, fail95={int(row.negative_control_fail95)}, pathways={row.negative_control_pathways})"
            for row in candidate_axes.itertuples(index=False)
        )
    axis_masked_recovered = int(axis_masked_validation["recovered_in_validation"].sum())
    axis_masked_total = int(len(axis_masked_validation))
    notes = f"""# pyXenium.pathway morphopathway evidence notes

Run directory: `{run_dir}`

## Current claim boundary

This evidence package supports a conservative H&E+WTA pathway-family stress test. It does not support a direct cervical replication claim for the top breast pathway because recovery is evaluated at pathway and family level, and breast discovery effect sizes remain modest.

## Inputs and scale

- Breast discovery observations: {int(summary.loc[summary['sample_name'] == 'breast_discovery', 'analysis_observations'].iloc[0])} spatial blocks from {int(summary.loc[summary['sample_name'] == 'breast_discovery', 'n_cells'].iloc[0])} sampled cells.
- Cervical validation observations: {int(summary.loc[summary['sample_name'] == 'cervical_validation', 'analysis_observations'].iloc[0])} spatial blocks from {int(summary.loc[summary['sample_name'] == 'cervical_validation', 'n_cells'].iloc[0])} sampled cells.
- H&E features: aligned low-resolution image pyramid sampled around Xenium cell centroids, then averaged into spatial blocks.
- {backend_note}
- WTA features: curated pathway activity scores from Xenium `cell_feature_matrix.h5`.

## Key results

- Breast top association: `{breast_top['pathway']}` / `{breast_top['image_feature']}`, abs partial Spearman rho = {float(breast_top['abs_partial_spearman_rho']):.4f}.
- Cervical top association: `{cervical_top['pathway']}` / `{cervical_top['image_feature']}`, abs partial Spearman rho = {float(cervical_top['abs_partial_spearman_rho']):.4f}.
- Cross-cancer pathway/family recovery: {n_recovered}/{n_total}.
- Cross-cancer recovery after removing candidate generic PLIP axes from their sample-specific association tables: {axis_masked_recovered}/{axis_masked_total}.
- Pathway coverage: breast {int(breast['coverage']['passes'].sum())}/{len(breast['coverage'])}; cervical {int(cervical['coverage']['passes'].sum())}/{len(cervical['coverage'])}.
- Spatial null gates: breast {gate_counts['breast_spatial_null_pass95']}/{len(breast['spatial_nulls'])} pass 95% and {gate_counts['breast_spatial_null_pass99']}/{len(breast['spatial_nulls'])} pass 99%; cervical {gate_counts['cervical_spatial_null_pass95']}/{len(cervical['spatial_nulls'])} pass 95% and {gate_counts['cervical_spatial_null_pass99']}/{len(cervical['spatial_nulls'])} pass 99%.
- Matched negative-control gates: breast {gate_counts['breast_negative_control_pass95']}/{len(breast['negative_controls'])} pass 95% and {gate_counts['breast_negative_control_pass99']}/{len(breast['negative_controls'])} pass 99%; cervical {gate_counts['cervical_negative_control_pass95']}/{len(cervical['negative_controls'])} pass 95% and {gate_counts['cervical_negative_control_pass99']}/{len(cervical['negative_controls'])} pass 99%.
- Candidate generic PLIP axes: {candidate_axis_note}

## Statistics

Associations are residual partial Spearman correlations after adjustment for coarse spatial structure, x/y coordinate ranks, boundary distance, and log total counts. Spatial nulls permute residual pathway activity within spatial strata. Negative controls use expression-matched random gene sets for the same image feature. Current smoke settings use limited permutations and negative controls, so p-values are gate checks rather than final inferential values.

## Residual risks

- H&E descriptors are deterministic color/texture/projection features in this run, not PLIP/UNI foundation-model embeddings unless a manifest explicitly records a PLIP/UNI backend.
- Breast signal is stable enough for smoke testing but too modest for a direct discovery claim.
- Cervical validation should be described as pathway-family stress testing unless stronger H&E embeddings recover direct pathway signals.
"""
    (output_dir / "methods_statistics_notes.md").write_text(notes, encoding="utf-8")
    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_tables": sorted(path.name for path in output_dir.glob("*.csv")),
        "notes": str(output_dir / "methods_statistics_notes.md"),
        "claim_boundary": "pathway-family stress test; no direct cervical replication claim",
        "breast_embedding_backend_status": breast_backend,
        "cervical_embedding_backend_status": cervical_backend,
        "cross_cancer_recovered": n_recovered,
        "cross_cancer_total": n_total,
        "axis_masked_cross_cancer_recovered": axis_masked_recovered,
        "axis_masked_cross_cancer_total": axis_masked_total,
        "candidate_generic_plip_axes": int(candidate_axes.shape[0]),
        **gate_counts,
    }
    (output_dir / "evidence_pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
