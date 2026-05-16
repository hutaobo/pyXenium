from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stability across morphopathway run directories.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _pass_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].fillna(False)
    if pd.api.types.is_bool_dtype(values):
        return int(values.sum())
    return int(values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum())


def _read_manifest_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    pathway_rows: list[pd.DataFrame] = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        run_id = run_dir.name
        summary = pd.read_csv(run_dir / "smoke_summary.csv")
        validation = pd.read_csv(run_dir / "cross_cancer_validation.csv")
        breast_assoc = pd.read_csv(run_dir / "breast_discovery" / "association_table.csv")
        cervical_assoc = pd.read_csv(run_dir / "cervical_validation" / "association_table.csv")
        breast_spatial = pd.read_csv(run_dir / "breast_discovery" / "spatial_nulls.csv")
        cervical_spatial = pd.read_csv(run_dir / "cervical_validation" / "spatial_nulls.csv")
        breast_negative = pd.read_csv(run_dir / "breast_discovery" / "negative_controls.csv")
        cervical_negative = pd.read_csv(run_dir / "cervical_validation" / "negative_controls.csv")
        breast_manifest = pd.read_csv(run_dir / "breast_discovery" / "input_manifest.csv")
        cervical_manifest = pd.read_csv(run_dir / "cervical_validation" / "input_manifest.csv")
        breast_block = _read_manifest_row(run_dir / "breast_discovery" / "spatial_block_manifest.csv")
        cervical_block = _read_manifest_row(run_dir / "cervical_validation" / "spatial_block_manifest.csv")

        breast_top = breast_assoc.iloc[0]
        cervical_top = cervical_assoc.iloc[0]
        run_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "backend": str(breast_manifest.get("he_embedding_backend_status", pd.Series([""])).iloc[0]),
                "breast_n_cells": int(summary.loc[summary["sample_name"] == "breast_discovery", "n_cells"].iloc[0]),
                "breast_analysis_observations": int(
                    summary.loc[summary["sample_name"] == "breast_discovery", "analysis_observations"].iloc[0]
                ),
                "cervical_analysis_observations": int(
                    summary.loc[summary["sample_name"] == "cervical_validation", "analysis_observations"].iloc[0]
                ),
                "breast_top_pathway": str(breast_top["pathway"]),
                "breast_top_family": str(breast_top["family"]),
                "breast_top_feature": str(breast_top["image_feature"]),
                "breast_top_abs_rho": float(breast_top["abs_partial_spearman_rho"]),
                "cervical_top_pathway": str(cervical_top["pathway"]),
                "cervical_top_family": str(cervical_top["family"]),
                "cervical_top_feature": str(cervical_top["image_feature"]),
                "cervical_top_abs_rho": float(cervical_top["abs_partial_spearman_rho"]),
                "cross_cancer_recovered": int(validation["recovered_in_validation"].fillna(False).astype(bool).sum()),
                "cross_cancer_total": int(len(validation)),
                "breast_spatial_pass95": _pass_count(breast_spatial, "passes_spatial_null_95"),
                "cervical_spatial_pass95": _pass_count(cervical_spatial, "passes_spatial_null_95"),
                "breast_negative_pass95": _pass_count(breast_negative, "passes_negative_control_95"),
                "cervical_negative_pass95": _pass_count(cervical_negative, "passes_negative_control_95"),
                "breast_random_state": breast_manifest.get("random_state", pd.Series([""])).iloc[0],
                "cervical_random_state": cervical_manifest.get("random_state", pd.Series([""])).iloc[0],
                "breast_spatial_block_bins": breast_block.get("bins", ""),
                "cervical_spatial_block_bins": cervical_block.get("bins", ""),
                "breast_spatial_blocks": breast_block.get("n_blocks", ""),
                "cervical_spatial_blocks": cervical_block.get("n_blocks", ""),
                "breast_median_cells_per_block": breast_block.get("median_cells_per_block", ""),
                "cervical_median_cells_per_block": cervical_block.get("median_cells_per_block", ""),
            }
        )

        pathway_rows.append(validation.assign(run_id=run_id, run_dir=str(run_dir)))

    run_summary = pd.DataFrame(run_rows)
    all_pathways = pd.concat(pathway_rows, ignore_index=True)
    pathway_summary = (
        all_pathways.groupby(["pathway", "family"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            recovery_rate=("recovered_in_validation", "mean"),
            recovered_runs=("recovered_in_validation", "sum"),
            mean_discovery_abs_partial_rho=("discovery_abs_partial_rho", "mean"),
            mean_validation_pathway_abs_partial_rho=("validation_pathway_abs_partial_rho", "mean"),
            mean_validation_family_abs_partial_rho=("validation_family_abs_partial_rho", "mean"),
        )
        .sort_values(["recovery_rate", "mean_discovery_abs_partial_rho"], ascending=[False, False], kind="stable")
    )
    run_summary.to_csv(output_dir / "stability_run_summary.csv", index=False)
    all_pathways.to_csv(output_dir / "stability_cross_cancer_by_run.csv", index=False)
    pathway_summary.to_csv(output_dir / "stability_pathway_summary.csv", index=False)

    min_recovery = int(run_summary["cross_cancer_recovered"].min()) if not run_summary.empty else 0
    max_recovery = int(run_summary["cross_cancer_recovered"].max()) if not run_summary.empty else 0
    median_recovery = float(run_summary["cross_cancer_recovered"].median()) if not run_summary.empty else 0.0
    manifest = {
        "output_dir": str(output_dir.resolve()),
        "run_dirs": [str(path.resolve()) for path in args.run_dirs],
        "n_runs": int(len(run_summary)),
        "min_cross_cancer_recovered": min_recovery,
        "median_cross_cancer_recovered": median_recovery,
        "max_cross_cancer_recovered": max_recovery,
        "stable_recovered_pathways": pathway_summary.loc[
            pathway_summary["recovery_rate"] >= 1.0, "pathway"
        ].astype(str).tolist(),
    }
    (output_dir / "stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    notes = [
        "# PLIP Morphopathway Stability Notes",
        "",
        f"Compared runs: {len(run_summary)}",
        f"Cross-cancer recovery range: {min_recovery}/{int(run_summary['cross_cancer_total'].max())} to {max_recovery}/{int(run_summary['cross_cancer_total'].max())}.",
        f"Median cross-cancer recovery: {median_recovery:.1f}/{int(run_summary['cross_cancer_total'].max())}.",
        f"Stable recovered pathways in all compared runs: {', '.join(manifest['stable_recovered_pathways']) or 'none'}.",
        f"Partially recovered pathways (>=2 runs): {', '.join(pathway_summary.loc[pathway_summary['recovery_rate'] >= (2 / max(len(run_summary), 1)), 'pathway'].astype(str).tolist()) or 'none'}.",
        "",
        "Spatial block settings:",
    ]
    for row in run_summary.itertuples(index=False):
        notes.append(
            "- "
            f"{row.run_id}: breast bins={row.breast_spatial_block_bins}, blocks={row.breast_spatial_blocks}, "
            f"median cells/block={row.breast_median_cells_per_block}; "
            f"cervical bins={row.cervical_spatial_block_bins}, blocks={row.cervical_spatial_blocks}, "
            f"median cells/block={row.cervical_median_cells_per_block}."
        )
    notes.extend(
        [
            "",
        "Gate interpretation:",
        "- Strongest evidence: pathways recovered in all compared seeds.",
        "- Supportive evidence: pathways recovered in at least two seeds.",
        "- Do not claim direct cervical replication from one sampled run alone.",
        "",
        "Interpretation: this is a stability screen over sampled cells and spatial blocks, not a final exhaustive analysis.",
        ]
    )
    (output_dir / "stability_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
