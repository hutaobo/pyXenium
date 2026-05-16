from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize morphopathway evidence-pack sensitivity manifests across runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    axis_rows: list[pd.DataFrame] = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        evidence_dir = run_dir / "evidence_pack"
        manifest = _read_json(evidence_dir / "evidence_pack_manifest.json")
        run_id = run_dir.name
        summary_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "cross_cancer_recovered": int(manifest.get("cross_cancer_recovered", 0)),
                "cross_cancer_total": int(manifest.get("cross_cancer_total", 0)),
                "axis_masked_cross_cancer_recovered": int(
                    manifest.get("axis_masked_cross_cancer_recovered", manifest.get("cross_cancer_recovered", 0))
                ),
                "axis_masked_cross_cancer_total": int(
                    manifest.get("axis_masked_cross_cancer_total", manifest.get("cross_cancer_total", 0))
                ),
                "candidate_generic_plip_axes": int(manifest.get("candidate_generic_plip_axes", 0)),
                "breast_spatial_null_pass95": int(manifest.get("breast_spatial_null_pass95", 0)),
                "breast_spatial_null_pass99": int(manifest.get("breast_spatial_null_pass99", 0)),
                "cervical_spatial_null_pass95": int(manifest.get("cervical_spatial_null_pass95", 0)),
                "cervical_spatial_null_pass99": int(manifest.get("cervical_spatial_null_pass99", 0)),
                "breast_negative_control_pass95": int(manifest.get("breast_negative_control_pass95", 0)),
                "breast_negative_control_pass99": int(manifest.get("breast_negative_control_pass99", 0)),
                "cervical_negative_control_pass95": int(manifest.get("cervical_negative_control_pass95", 0)),
                "cervical_negative_control_pass99": int(manifest.get("cervical_negative_control_pass99", 0)),
                "claim_boundary": str(manifest.get("claim_boundary", "")),
            }
        )

        axis_path = evidence_dir / "supp_table_plip_feature_axis_diagnostics.csv"
        if axis_path.exists():
            axis = pd.read_csv(axis_path)
            if not axis.empty:
                axis_rows.append(axis.assign(run_id=run_id, run_dir=str(run_dir)))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "evidence_pack_sensitivity_summary.csv", index=False)

    if axis_rows:
        axes = pd.concat(axis_rows, ignore_index=True)
    else:
        axes = pd.DataFrame()
    if not axes.empty:
        axes.to_csv(output_dir / "plip_feature_axis_diagnostics_by_run.csv", index=False)
        candidate_axes = axes.loc[
            axes.get("candidate_generic_axis", pd.Series(False, index=axes.index)).astype(str).str.lower() == "true"
        ].copy()
    else:
        candidate_axes = pd.DataFrame()
        axes.to_csv(output_dir / "plip_feature_axis_diagnostics_by_run.csv", index=False)

    total = int(summary["cross_cancer_total"].max()) if not summary.empty else 0
    manifest_out = {
        "output_dir": str(output_dir.resolve()),
        "run_dirs": [str(path.resolve()) for path in args.run_dirs],
        "n_runs": int(len(summary)),
        "min_cross_cancer_recovered": int(summary["cross_cancer_recovered"].min()) if not summary.empty else 0,
        "min_axis_masked_cross_cancer_recovered": int(summary["axis_masked_cross_cancer_recovered"].min())
        if not summary.empty
        else 0,
        "cross_cancer_total": total,
        "runs_with_candidate_generic_plip_axes": int((summary["candidate_generic_plip_axes"] > 0).sum())
        if not summary.empty
        else 0,
        "candidate_generic_plip_axes": int(len(candidate_axes)),
    }
    (output_dir / "evidence_pack_sensitivity_manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )

    notes = [
        "# Evidence-Pack Sensitivity Summary",
        "",
        f"Compared runs: {manifest_out['n_runs']}",
        f"Cross-cancer recovery range: {manifest_out['min_cross_cancer_recovered']}/{total} to {int(summary['cross_cancer_recovered'].max()) if not summary.empty else 0}/{total}.",
        f"Axis-masked recovery range: {manifest_out['min_axis_masked_cross_cancer_recovered']}/{total} to {int(summary['axis_masked_cross_cancer_recovered'].max()) if not summary.empty else 0}/{total}.",
        f"Runs with candidate generic PLIP axes: {manifest_out['runs_with_candidate_generic_plip_axes']}/{manifest_out['n_runs']}.",
        "",
        "Interpretation:",
        "- Axis-masked recovery removes sample-specific candidate generic PLIP axes before recomputing pathway/family validation.",
        "- Stable recovery after axis masking argues the cross-cancer signal is not dependent on those flagged axes.",
        "- Negative-control pass95/pass99 counts should be reported separately from the number of tested controls.",
    ]
    (output_dir / "evidence_pack_sensitivity_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(manifest_out, indent=2))


if __name__ == "__main__":
    main()
