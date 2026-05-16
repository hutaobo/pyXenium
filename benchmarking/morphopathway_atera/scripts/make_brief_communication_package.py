from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble final source tables and claim notes for the Atera morphopathway Brief Communication package."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--stability-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_rate(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _best_by_pathway(associations: pd.DataFrame, run_id: str, sample_role: str) -> pd.DataFrame:
    best = (
        associations.assign(abs_partial_spearman_rho=pd.to_numeric(associations["abs_partial_spearman_rho"], errors="coerce"))
        .sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("pathway", keep="first")
        .reset_index(drop=True)
    )
    keep = [
        "pathway",
        "family",
        "image_feature",
        "partial_spearman_rho",
        "abs_partial_spearman_rho",
        "p_value",
        "permutation_empirical_p",
        "null_abs_rho_q95",
        "passes_spatial_null_95",
        "passes_spatial_null_99",
        "negative_control_empirical_p",
        "negative_control_abs_rho_q95",
        "passes_negative_control_95",
        "passes_negative_control_99",
    ]
    keep = [column for column in keep if column in best.columns]
    best = best.loc[:, keep].copy()
    best.insert(0, "sample_role", sample_role)
    best.insert(0, "run_id", run_id)
    return best


def _pathway_recovery_summary(validation_frames: list[pd.DataFrame], value_prefix: str) -> pd.DataFrame:
    combined = pd.concat(validation_frames, ignore_index=True)
    combined["recovered_in_validation"] = _bool_rate(combined, "recovered_in_validation")
    return (
        combined.groupby(["pathway", "family"], as_index=False)
        .agg(
            **{
                f"{value_prefix}_n_runs": ("run_id", "nunique"),
                f"{value_prefix}_recovered_runs": ("recovered_in_validation", "sum"),
                f"{value_prefix}_recovery_rate": ("recovered_in_validation", "mean"),
                f"{value_prefix}_mean_discovery_abs_partial_rho": ("discovery_abs_partial_rho", "mean"),
                f"{value_prefix}_mean_validation_pathway_abs_partial_rho": (
                    "validation_pathway_abs_partial_rho",
                    "mean",
                ),
                f"{value_prefix}_mean_validation_family_abs_partial_rho": (
                    "validation_family_abs_partial_rho",
                    "mean",
                ),
            }
        )
        .sort_values(
            [f"{value_prefix}_recovery_rate", f"{value_prefix}_mean_discovery_abs_partial_rho"],
            ascending=[False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stability_pathways = _read_csv(args.stability_dir / "stability_pathway_summary.csv")
    stability_runs = _read_csv(args.stability_dir / "stability_run_summary.csv")
    sensitivity_summary = _read_csv(args.sensitivity_dir / "evidence_pack_sensitivity_summary.csv")
    axis_diagnostics = _read_csv(args.sensitivity_dir / "plip_feature_axis_diagnostics_by_run.csv")
    sensitivity_manifest = _read_json(args.sensitivity_dir / "evidence_pack_sensitivity_manifest.json")

    breast_best_frames: list[pd.DataFrame] = []
    cervical_best_frames: list[pd.DataFrame] = []
    primary_validation_frames: list[pd.DataFrame] = []
    masked_validation_frames: list[pd.DataFrame] = []

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        run_id = run_dir.name
        breast_assoc = _read_csv(run_dir / "breast_discovery" / "association_table.csv")
        cervical_assoc = _read_csv(run_dir / "cervical_validation" / "association_table.csv")
        primary = _read_csv(run_dir / "cross_cancer_validation.csv").assign(run_id=run_id, analysis="primary")
        masked = _read_csv(
            run_dir / "evidence_pack" / "fig2_cross_cancer_validation_without_candidate_plip_axes.csv"
        ).assign(run_id=run_id, analysis="axis_masked")
        breast_best_frames.append(_best_by_pathway(breast_assoc, run_id, "breast_discovery"))
        cervical_best_frames.append(_best_by_pathway(cervical_assoc, run_id, "cervical_validation"))
        primary_validation_frames.append(primary)
        masked_validation_frames.append(masked)

    breast_best = pd.concat(breast_best_frames, ignore_index=True)
    cervical_best = pd.concat(cervical_best_frames, ignore_index=True)
    primary_summary = _pathway_recovery_summary(primary_validation_frames, "primary")
    masked_summary = _pathway_recovery_summary(masked_validation_frames, "axis_masked")
    pathway_summary = primary_summary.merge(masked_summary, on=["pathway", "family"], how="outer")
    pathway_summary["stable_9_pathway_core"] = (
        (pd.to_numeric(pathway_summary["primary_recovery_rate"], errors="coerce") >= 1.0)
        & (pd.to_numeric(pathway_summary["axis_masked_recovery_rate"], errors="coerce") >= 1.0)
    )
    pathway_summary = pathway_summary.merge(
        stability_pathways.loc[:, ["pathway", "recovered_runs", "recovery_rate"]],
        on="pathway",
        how="left",
        suffixes=("", "_stability"),
    )
    pathway_summary = pathway_summary.sort_values(
        ["stable_9_pathway_core", "primary_recovery_rate", "primary_mean_discovery_abs_partial_rho"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    stable_core = pathway_summary.loc[pathway_summary["stable_9_pathway_core"], "pathway"].astype(str).tolist()
    breast_best = breast_best.merge(
        pathway_summary.loc[:, ["pathway", "stable_9_pathway_core", "primary_recovery_rate", "axis_masked_recovery_rate"]],
        on="pathway",
        how="left",
    )
    cervical_best = cervical_best.merge(
        pathway_summary.loc[:, ["pathway", "stable_9_pathway_core", "primary_recovery_rate", "axis_masked_recovery_rate"]],
        on="pathway",
        how="left",
    )

    primary_validation = pd.concat(primary_validation_frames, ignore_index=True)
    masked_validation = pd.concat(masked_validation_frames, ignore_index=True)
    validation_by_run = pd.concat([primary_validation, masked_validation], ignore_index=True)

    breast_best.to_csv(output_dir / "main_figure_1_source_breast_discovery_highnull32.csv", index=False)
    pathway_summary.to_csv(output_dir / "main_figure_2_source_cross_cancer_stability_highnull32.csv", index=False)
    validation_by_run.to_csv(output_dir / "source_table_cross_cancer_validation_by_run.csv", index=False)
    cervical_best.to_csv(output_dir / "supp_table_cervical_validation_best_associations.csv", index=False)
    stability_runs.to_csv(output_dir / "supp_table_spatial_block_and_seed_summary.csv", index=False)
    sensitivity_summary.to_csv(output_dir / "supp_table_highnull32_gate_and_axis_masked_summary.csv", index=False)
    axis_diagnostics.to_csv(output_dir / "supp_table_plip_axis_diagnostics_by_run.csv", index=False)

    recovered_values = pd.to_numeric(sensitivity_summary["cross_cancer_recovered"], errors="coerce")
    masked_values = pd.to_numeric(sensitivity_summary["axis_masked_cross_cancer_recovered"], errors="coerce")
    total = int(pd.to_numeric(sensitivity_summary["cross_cancer_total"], errors="coerce").max())
    candidate_axis_runs = int((pd.to_numeric(sensitivity_summary["candidate_generic_plip_axes"], errors="coerce") > 0).sum())
    claim_lines = [
        "# Conservative Claim Wording",
        "",
        "Recommended primary claim:",
        "",
        (
            "Using a newly developed `pyXenium.pathway` H&E+WTA morphopathway workflow on the Atera "
            "Xenium WTA breast and cervical FFPE preview datasets, PLIP-derived H&E embeddings aggregated "
            "to coarse spatial pseudobulk blocks recovered a stable 9-pathway pathway-family stress-test "
            f"core across three high-null seeds ({int(recovered_values.min())}/{total} to "
            f"{int(recovered_values.max())}/{total} recovered)."
        ),
        "",
        "Supported wording:",
        "",
        (
            f"Axis-masked sensitivity, which removes sample-specific candidate generic PLIP axes, remained "
            f"{int(masked_values.min())}/{total} to {int(masked_values.max())}/{total}, arguing that the "
            "cross-cancer pathway-family signal is not dependent on the flagged axes."
        ),
        "",
        "Do not claim:",
        "",
        "- Direct pathway-level cervical replication of the top breast signal.",
        "- Clinical biomarker performance, patient-level generalization, or causal morphology-pathway mechanisms.",
        "- Final inferential p-values from the smoke-scale nulls; report them as gate checks.",
        "",
        "Stable pathway-family stress-test core:",
        "",
        ", ".join(stable_core),
    ]
    (output_dir / "claim_wording.md").write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    methods_lines = [
        "# Methods And Statistics Notes",
        "",
        "Inputs:",
        "- Breast discovery: Atera Xenium WTA FFPE breast preview output.",
        "- Cervical validation: Atera Xenium WTA FFPE cervical preview output.",
        "- H&E backend: `transformers_clip:vinid/plip` for all three high-null runs.",
        "",
        "Workflow:",
        "- Sampled 3,000 cells per dataset and encoded H&E patches with PLIP at 64 output dimensions.",
        "- Aggregated cells into a 12 x 12 spatial pseudobulk grid with at least 6 cells per retained block.",
        "- Scored curated WTA pathway panels and fit residual partial Spearman associations after covariate adjustment.",
        "- Used 32 spatial permutations and 32 expression-matched random gene-set controls in the high-null runs.",
        "",
        "Evidence gates:",
        (
            f"- Cross-cancer pathway/family recovery range: {int(recovered_values.min())}/{total} to "
            f"{int(recovered_values.max())}/{total}."
        ),
        (
            f"- Axis-masked recovery range: {int(masked_values.min())}/{total} to "
            f"{int(masked_values.max())}/{total}."
        ),
        f"- Runs with candidate generic PLIP axes: {candidate_axis_runs}/{len(sensitivity_summary)}.",
        "",
        "Residual risks:",
        "- Matched negative-control pass95 is imperfect in seed17 and seed29, mainly on the breast side.",
        "- The cervical result should remain a pathway-family stress test rather than a direct replication claim.",
        "- This package is based on sampled public preview datasets; full-scale runs and independent cohorts are still needed.",
    ]
    (output_dir / "methods_statistics_notes.md").write_text("\n".join(methods_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# Brief Communication Evidence Package",
        "",
        "Primary source tables:",
        "- `main_figure_1_source_breast_discovery_highnull32.csv`",
        "- `main_figure_2_source_cross_cancer_stability_highnull32.csv`",
        "",
        "Key supplementary tables:",
        "- `supp_table_highnull32_gate_and_axis_masked_summary.csv`",
        "- `supp_table_plip_axis_diagnostics_by_run.csv`",
        "- `supp_table_spatial_block_and_seed_summary.csv`",
        "- `source_table_cross_cancer_validation_by_run.csv`",
        "",
        "Narrative notes:",
        "- `claim_wording.md`",
        "- `methods_statistics_notes.md`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    manifest = {
        "output_dir": str(output_dir.resolve()),
        "run_dirs": [str(path.resolve()) for path in args.run_dirs],
        "stability_dir": str(args.stability_dir.resolve()),
        "sensitivity_dir": str(args.sensitivity_dir.resolve()),
        "n_runs": int(len(args.run_dirs)),
        "cross_cancer_recovery_min": int(recovered_values.min()),
        "cross_cancer_recovery_max": int(recovered_values.max()),
        "axis_masked_recovery_min": int(masked_values.min()),
        "axis_masked_recovery_max": int(masked_values.max()),
        "cross_cancer_total": total,
        "stable_9_pathway_core": stable_core,
        "candidate_axis_runs": candidate_axis_runs,
        "source_tables": sorted(path.name for path in output_dir.glob("*.csv")),
        "notes": ["README.md", "claim_wording.md", "methods_statistics_notes.md"],
        "upstream_sensitivity_manifest": sensitivity_manifest,
    }
    (output_dir / "brief_communication_package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
