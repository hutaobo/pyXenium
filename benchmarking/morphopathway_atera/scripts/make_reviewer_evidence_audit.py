from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    return _write_text_if_changed(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv_if_changed(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> bool:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return _write_text_if_changed(path, buffer.getvalue())


def _min_int(rows: list[dict[str, str]], column: str) -> int | None:
    values: list[int] = []
    for row in rows:
        value = row.get(column)
        if value not in (None, ""):
            values.append(int(float(value)))
    return min(values) if values else None


def _row_count(package_dir: Path, filename: str) -> int:
    path = package_dir / filename
    if not path.exists():
        return 0
    return len(_read_csv_rows(path))


def _status(condition: bool) -> str:
    return "pass" if condition else "review"


def _join_files(*names: str) -> str:
    return "; ".join(name for name in names if name)


def build_reviewer_evidence_audit(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "brief_communication_package_manifest.json"
    qc_path = package_dir / "package_qc_report.json"
    claim_path = package_dir / "claim_wording.md"
    methods_path = package_dir / "methods_statistics_notes.md"
    readiness_path = package_dir / "submission_readiness_checklist.md"

    manifest = _load_json(manifest_path)
    qc = _load_json(qc_path) if qc_path.exists() else {"status": "missing", "warnings": []}
    claim_text = claim_path.read_text(encoding="utf-8") if claim_path.exists() else ""
    readiness_text = readiness_path.read_text(encoding="utf-8") if readiness_path.exists() else ""

    fig2_name = "main_figure_2_source_cross_cancer_stability_highnull32.csv"
    gate_name = "supp_table_highnull32_gate_and_axis_masked_summary.csv"
    axis_name = "supp_table_plip_axis_diagnostics_by_run.csv"
    fig2_rows = _read_csv_rows(package_dir / fig2_name) if (package_dir / fig2_name).exists() else []
    gate_rows = _read_csv_rows(package_dir / gate_name) if (package_dir / gate_name).exists() else []

    stable_core = list(manifest.get("stable_9_pathway_core", []))
    fig2_stable = [
        row.get("pathway", "")
        for row in fig2_rows
        if str(row.get("stable_9_pathway_core", "")).strip().lower() in {"true", "1", "yes"}
    ]
    emt_rows = [row for row in fig2_rows if row.get("pathway") == "emt_invasive_front"]
    emt_stable = any(
        str(row.get("stable_9_pathway_core", "")).strip().lower() in {"true", "1", "yes"} for row in emt_rows
    )

    cross_min = int(manifest.get("cross_cancer_recovery_min", 0))
    cross_max = int(manifest.get("cross_cancer_recovery_max", 0))
    axis_min = int(manifest.get("axis_masked_recovery_min", 0))
    axis_max = int(manifest.get("axis_masked_recovery_max", 0))
    cross_total = int(manifest.get("cross_cancer_total", 0))
    breast_neg_min = _min_int(gate_rows, "breast_negative_control_pass95")
    cervical_neg_min = _min_int(gate_rows, "cervical_negative_control_pass95")
    warning_text = " ".join(str(warning) for warning in qc.get("warnings", []))

    archive_path = package_dir.parent / f"{package_dir.name}.zip"
    archive_manifest_path = package_dir.parent / f"{package_dir.name}.manifest.json"
    archive_manifest = _load_json(archive_manifest_path) if archive_manifest_path.exists() else {}
    archive_hash = _sha256(archive_path) if archive_path.exists() else ""
    archive_manifest_hash = archive_manifest.get("archive_sha256", "")
    archive_sidecar_match = bool(archive_hash) and archive_hash == archive_manifest_hash

    source_tables = list(manifest.get("source_tables", []))
    table_row_counts = {name: _row_count(package_dir, name) for name in source_tables}
    required_note_files = ["README.md", "claim_wording.md", "methods_statistics_notes.md"]

    rows: list[dict[str, Any]] = [
        {
            "evidence_item": "Primary claim boundary",
            "status": _status("pathway-family" in claim_text and "Do not claim" in claim_text),
            "key_numbers": f"stable_core={len(stable_core)} pathways",
            "supporting_files": _join_files("claim_wording.md", "submission_readiness_checklist.md"),
            "residual_risk": "Claim must remain a pathway-family stress test, not direct cervical replication.",
            "next_action": "Preserve conservative wording in manuscript title, abstract, and figure captions.",
        },
        {
            "evidence_item": "Cross-cancer recovery gate",
            "status": _status(cross_min >= 9 and cross_total == 10),
            "key_numbers": f"{cross_min}/{cross_total} to {cross_max}/{cross_total}",
            "supporting_files": _join_files("main_figure_2_source_cross_cancer_stability_highnull32.csv", "source_table_cross_cancer_validation_by_run.csv"),
            "residual_risk": "Cervical result supports stress-test recovery rather than independent direct replication.",
            "next_action": "Report range and per-run source table together.",
        },
        {
            "evidence_item": "Axis-masked PLIP sensitivity",
            "status": _status(axis_min >= 9 and axis_max >= axis_min),
            "key_numbers": f"{axis_min}/{cross_total} to {axis_max}/{cross_total}; candidate_axis_runs={manifest.get('candidate_axis_runs', '')}",
            "supporting_files": _join_files("supp_table_highnull32_gate_and_axis_masked_summary.csv", axis_name),
            "residual_risk": "Two runs contain candidate generic PLIP axes; sensitivity is supportive but smoke-scale.",
            "next_action": "Keep axis-masked result adjacent to the primary gate in the figure or supplement.",
        },
        {
            "evidence_item": "Stable 9-pathway core definition",
            "status": _status(len(stable_core) == 9 and set(stable_core) == set(fig2_stable)),
            "key_numbers": f"manifest_core={len(stable_core)}; figure2_core={len(fig2_stable)}",
            "supporting_files": _join_files("brief_communication_package_manifest.json", fig2_name),
            "residual_risk": "The core is a reproducible pathway-family set, not a ranked clinical signature.",
            "next_action": "Use the exact stable-core list from the manifest.",
        },
        {
            "evidence_item": "EMT/invasive-front exclusion",
            "status": _status("emt_invasive_front" not in stable_core and not emt_stable and bool(emt_rows)),
            "key_numbers": f"emt_invasive_front_stable={emt_stable}",
            "supporting_files": _join_files(fig2_name, "claim_wording.md"),
            "residual_risk": "Tempting biological story, but current evidence is not stable enough.",
            "next_action": "Mention as non-core or omit from primary claims.",
        },
        {
            "evidence_item": "Matched negative-control limitation",
            "status": _status(breast_neg_min is not None and cervical_neg_min is not None and "limitation" in warning_text.lower()),
            "key_numbers": f"breast_min={breast_neg_min}/10; cervical_min={cervical_neg_min}/10",
            "supporting_files": _join_files("package_qc_report.json", gate_name),
            "residual_risk": "Breast negative-control pass95 minimum is below 10/10; this must be called a limitation.",
            "next_action": "Surface this in limitations and statistical notes.",
        },
        {
            "evidence_item": "Figure and supplementary source tables",
            "status": _status(all(count > 0 for count in table_row_counts.values()) and len(source_tables) >= 7),
            "key_numbers": "; ".join(f"{name}={count}" for name, count in table_row_counts.items()),
            "supporting_files": _join_files(*source_tables),
            "residual_risk": "Large PLIP axis diagnostic table should stay supplementary.",
            "next_action": "Use source tables directly for figure construction; avoid hand-edited plot data.",
        },
        {
            "evidence_item": "Methods/statistics traceability",
            "status": _status(all((package_dir / name).exists() for name in required_note_files)),
            "key_numbers": f"notes={len(required_note_files)}/3 present",
            "supporting_files": _join_files(*required_note_files),
            "residual_risk": "Final manuscript still needs exact software/hardware details.",
            "next_action": "Copy methods notes into the draft and fill version/hardware blanks.",
        },
        {
            "evidence_item": "Archive reproducibility",
            "status": _status(archive_sidecar_match),
            "key_numbers": f"archive_sidecar_match={str(archive_sidecar_match).lower()}",
            "supporting_files": _join_files(f"{package_dir.name}.zip", f"{package_dir.name}.manifest.json"),
            "residual_risk": "Archive changes when new audit files are intentionally added.",
            "next_action": "Re-run archive generation after any package content change.",
        },
    ]

    failing = [row["evidence_item"] for row in rows if row["status"] != "pass"]
    file_inventory = {
        name: {
            "exists": (package_dir / name).exists(),
            "size_bytes": (package_dir / name).stat().st_size if (package_dir / name).exists() else 0,
            "sha256": _sha256(package_dir / name) if (package_dir / name).exists() else "",
        }
        for name in sorted(set(source_tables + required_note_files + ["brief_communication_package_manifest.json", "package_qc_report.json"]))
    }

    return {
        "package_dir": str(package_dir),
        "status": "pass" if not failing else "review",
        "failing_items": failing,
        "metrics": {
            "cross_cancer_recovery_min": cross_min,
            "cross_cancer_recovery_max": cross_max,
            "axis_masked_recovery_min": axis_min,
            "axis_masked_recovery_max": axis_max,
            "cross_cancer_total": cross_total,
            "stable_core_count": len(stable_core),
            "breast_negative_control_pass95_min": breast_neg_min,
            "cervical_negative_control_pass95_min": cervical_neg_min,
            "source_table_count": len(source_tables),
        },
        "audit_rows": rows,
        "file_inventory": file_inventory,
        "readiness_excerpt_present": "Not Yet Publication-Ready Without Further Work" in readiness_text,
    }


def _render_markdown(audit: dict[str, Any]) -> str:
    metrics = audit["metrics"]
    lines = [
        "# Reviewer Evidence Audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        "## Key Numbers",
        "",
        f"- Cross-cancer recovery: {metrics['cross_cancer_recovery_min']}/{metrics['cross_cancer_total']} to {metrics['cross_cancer_recovery_max']}/{metrics['cross_cancer_total']}",
        f"- Axis-masked recovery: {metrics['axis_masked_recovery_min']}/{metrics['cross_cancer_total']} to {metrics['axis_masked_recovery_max']}/{metrics['cross_cancer_total']}",
        f"- Stable core: {metrics['stable_core_count']} pathways",
        f"- Matched negative-control pass95 minima: breast {metrics['breast_negative_control_pass95_min']}/10; cervical {metrics['cervical_negative_control_pass95_min']}/10",
        "",
        "## Audit Table",
        "",
        "| Evidence item | Status | Key numbers | Residual risk | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in audit["audit_rows"]:
        lines.append(
            "| {evidence_item} | {status} | {key_numbers} | {residual_risk} | {next_action} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in row.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    audit = build_reviewer_evidence_audit(package_dir)
    rows = audit["audit_rows"]
    fields = ["evidence_item", "status", "key_numbers", "supporting_files", "residual_risk", "next_action"]

    changed = {
        "reviewer_evidence_audit.json": _write_json_if_changed(package_dir / "reviewer_evidence_audit.json", audit),
        "reviewer_evidence_audit.csv": _write_csv_if_changed(package_dir / "reviewer_evidence_audit.csv", rows, fields),
        "reviewer_evidence_audit.md": _write_text_if_changed(package_dir / "reviewer_evidence_audit.md", _render_markdown(audit)),
    }
    print(json.dumps({"package_dir": str(package_dir), "status": audit["status"], "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
