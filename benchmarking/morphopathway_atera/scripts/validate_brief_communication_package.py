from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "README.md",
    "brief_communication_package_manifest.json",
    "claim_wording.md",
    "methods_statistics_notes.md",
    "main_figure_1_source_breast_discovery_highnull32.csv",
    "main_figure_2_source_cross_cancer_stability_highnull32.csv",
    "source_table_cross_cancer_validation_by_run.csv",
    "supp_table_cervical_validation_best_associations.csv",
    "supp_table_highnull32_gate_and_axis_masked_summary.csv",
    "supp_table_plip_axis_diagnostics_by_run.csv",
    "supp_table_spatial_block_and_seed_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a morphopathway Brief Communication evidence package.")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--min-stable-core", type=int, default=9)
    parser.add_argument("--min-recovery", type=int, default=9)
    parser.add_argument(
        "--require-reviewer-audit",
        action="store_true",
        help="Require reviewer_evidence_audit.{json,csv,md} and validate them against the package manifest.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _bool_series(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _section_after(text: str, heading: str) -> str:
    if heading not in text:
        return ""
    after = text.split(heading, 1)[1]
    paragraphs = [paragraph.strip() for paragraph in after.split("\n\n") if paragraph.strip()]
    return paragraphs[0] if paragraphs else ""


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    missing = [name for name in REQUIRED_FILES if not (package_dir / name).exists()]
    if missing:
        errors.append(f"Missing required files: {missing}")

    manifest: dict[str, object] = {}
    if (package_dir / "brief_communication_package_manifest.json").exists():
        manifest = _read_json(package_dir / "brief_communication_package_manifest.json")
        stable_core = list(manifest.get("stable_9_pathway_core", []))
        if len(stable_core) < int(args.min_stable_core):
            errors.append(f"Manifest stable core has {len(stable_core)} pathways, expected >= {args.min_stable_core}.")
        if int(manifest.get("cross_cancer_recovery_min", 0)) < int(args.min_recovery):
            errors.append("Manifest cross-cancer recovery minimum is below threshold.")
        if int(manifest.get("axis_masked_recovery_min", 0)) < int(args.min_recovery):
            errors.append("Manifest axis-masked recovery minimum is below threshold.")
        if int(manifest.get("cross_cancer_total", 0)) != 10:
            warnings.append("Manifest cross-cancer total is not 10.")

    if (package_dir / "main_figure_2_source_cross_cancer_stability_highnull32.csv").exists():
        fig2 = _read_csv(package_dir / "main_figure_2_source_cross_cancer_stability_highnull32.csv")
        stable = fig2.loc[_bool_series(fig2["stable_9_pathway_core"])]
        if len(stable) < int(args.min_stable_core):
            errors.append(f"Main Figure 2 stable core has {len(stable)} pathways, expected >= {args.min_stable_core}.")
        emt = fig2.loc[fig2["pathway"].astype(str) == "emt_invasive_front"]
        if not emt.empty and bool(_bool_series(emt["stable_9_pathway_core"]).iloc[0]):
            errors.append("emt_invasive_front is marked stable despite prior sensitivity results.")
        if "primary_recovery_rate" in fig2.columns and "axis_masked_recovery_rate" in fig2.columns:
            recovered_stable = fig2.loc[
                _bool_series(fig2["stable_9_pathway_core"])
                & (pd.to_numeric(fig2["primary_recovery_rate"], errors="coerce") >= 1.0)
                & (pd.to_numeric(fig2["axis_masked_recovery_rate"], errors="coerce") >= 1.0)
            ]
            if len(recovered_stable) != len(stable):
                errors.append("Some stable-core rows are not recovered in both primary and axis-masked summaries.")

    if (package_dir / "supp_table_highnull32_gate_and_axis_masked_summary.csv").exists():
        gates = _read_csv(package_dir / "supp_table_highnull32_gate_and_axis_masked_summary.csv")
        if len(gates) < 3:
            errors.append("Gate summary has fewer than three runs.")
        for column in ["cross_cancer_recovered", "axis_masked_cross_cancer_recovered"]:
            min_value = int(pd.to_numeric(gates[column], errors="coerce").min())
            if min_value < int(args.min_recovery):
                errors.append(f"{column} minimum is {min_value}, expected >= {args.min_recovery}.")
        for column in ["breast_negative_control_pass95", "cervical_negative_control_pass95"]:
            min_value = int(pd.to_numeric(gates[column], errors="coerce").min())
            if min_value < 10:
                warnings.append(f"{column} minimum is {min_value}/10; report as limitation.")

    claim_path = package_dir / "claim_wording.md"
    if claim_path.exists():
        claim_text = claim_path.read_text(encoding="utf-8")
        recommended = _section_after(claim_text, "Recommended primary claim:")
        if "pathway-family stress-test" not in recommended:
            errors.append("Recommended primary claim does not include pathway-family stress-test wording.")
        if "direct" in recommended.lower() or "replication" in recommended.lower():
            errors.append("Recommended primary claim appears to use direct replication wording.")
        for required_phrase in ["Do not claim:", "Axis-masked sensitivity", "Stable pathway-family stress-test core"]:
            if required_phrase not in claim_text:
                errors.append(f"Claim wording is missing required phrase: {required_phrase}")

    reviewer_audit_files = [
        "reviewer_evidence_audit.json",
        "reviewer_evidence_audit.csv",
        "reviewer_evidence_audit.md",
    ]
    reviewer_audit_present = all((package_dir / name).exists() for name in reviewer_audit_files)
    if args.require_reviewer_audit and not reviewer_audit_present:
        errors.append(f"Missing required reviewer audit files: {reviewer_audit_files}")
    if reviewer_audit_present:
        audit = _read_json(package_dir / "reviewer_evidence_audit.json")
        if audit.get("status") != "pass":
            errors.append("Reviewer evidence audit status is not pass.")
        failing_items = list(audit.get("failing_items", []))
        if failing_items:
            errors.append(f"Reviewer evidence audit has failing items: {failing_items}")
        audit_rows = list(audit.get("audit_rows", []))
        if len(audit_rows) < 8:
            errors.append(f"Reviewer evidence audit has {len(audit_rows)} rows, expected >= 8.")
        audit_metrics = dict(audit.get("metrics", {}))
        manifest_metric_pairs = [
            ("cross_cancer_recovery_min", "cross_cancer_recovery_min"),
            ("axis_masked_recovery_min", "axis_masked_recovery_min"),
        ]
        for audit_key, manifest_key in manifest_metric_pairs:
            if audit_metrics.get(audit_key) != manifest.get(manifest_key):
                errors.append(
                    f"Reviewer audit {audit_key}={audit_metrics.get(audit_key)} "
                    f"does not match manifest {manifest_key}={manifest.get(manifest_key)}."
                )
        stable_core_count = audit_metrics.get("stable_core_count")
        if stable_core_count != len(list(manifest.get("stable_9_pathway_core", []))):
            errors.append("Reviewer audit stable_core_count does not match manifest stable core.")
        audit_csv = _read_csv(package_dir / "reviewer_evidence_audit.csv")
        if len(audit_csv) != len(audit_rows):
            errors.append("Reviewer evidence audit CSV row count does not match JSON audit_rows.")
        audit_md = (package_dir / "reviewer_evidence_audit.md").read_text(encoding="utf-8")
        for required_phrase in ["Reviewer Evidence Audit", "Cross-cancer recovery", "Matched negative-control"]:
            if required_phrase not in audit_md:
                errors.append(f"Reviewer evidence audit markdown missing phrase: {required_phrase}")

    report = {
        "package_dir": str(package_dir),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "checked_files": REQUIRED_FILES,
        "reviewer_audit_checked": bool(reviewer_audit_present),
        "manifest_summary": {
            "cross_cancer_recovery_min": manifest.get("cross_cancer_recovery_min"),
            "axis_masked_recovery_min": manifest.get("axis_masked_recovery_min"),
            "stable_9_pathway_core_count": len(list(manifest.get("stable_9_pathway_core", []))),
            "candidate_axis_runs": manifest.get("candidate_axis_runs"),
        },
    }
    _write_text_if_changed(package_dir / "package_qc_report.json", json.dumps(report, indent=2))

    md_lines = [
        "# Package QC Report",
        "",
        f"Status: {report['status']}",
        "",
        "Errors:",
        *(f"- {item}" for item in errors),
        *(["- none"] if not errors else []),
        "",
        "Warnings:",
        *(f"- {item}" for item in warnings),
        *(["- none"] if not warnings else []),
    ]
    _write_text_if_changed(package_dir / "package_qc_report.md", "\n".join(md_lines) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
