from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


IN_SCOPE_PATHS = [
    "src/pyXenium/pathway/_morphopathway.py",
    "src/pyXenium/pathway/__init__.py",
    "src/pyXenium/__init__.py",
    "tests/test_morphopathway.py",
    "benchmarking/morphopathway_atera/",
]

EXPECTED_UNRELATED_DIRTY = [
    "benchmarking/cci_2026_atera/",
    "benchmarking/lazyslide_a100/",
    "benchmarking/lr_2026_atera/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a handoff scope inventory for morphopathway work.")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _git_status(repo_root: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2].strip()
        path = line[3:].strip().replace("\\", "/")
        rows.append({"status": status, "path": path})
    return rows


def _is_in_scope(path: str) -> bool:
    path = path.replace("\\", "/")
    return any(path == scope.rstrip("/") or path.startswith(scope) for scope in IN_SCOPE_PATHS)


def _is_expected_unrelated(path: str) -> bool:
    path = path.replace("\\", "/")
    return any(path.startswith(prefix) for prefix in EXPECTED_UNRELATED_DIRTY)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    package_dir = args.package_dir.resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    status_rows = _git_status(repo_root)
    in_scope = [row for row in status_rows if _is_in_scope(row["path"])]
    unrelated = [row for row in status_rows if not _is_in_scope(row["path"])]
    unexpected_unrelated = [row for row in unrelated if not _is_expected_unrelated(row["path"])]

    package_manifest_path = package_dir / "brief_communication_package_manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8")) if package_manifest_path.exists() else {}

    inventory = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "package_dir": str(package_dir),
        "in_scope_paths": IN_SCOPE_PATHS,
        "expected_unrelated_dirty_prefixes": EXPECTED_UNRELATED_DIRTY,
        "git_status_in_scope": in_scope,
        "git_status_unrelated": unrelated,
        "unexpected_unrelated_dirty": unexpected_unrelated,
        "final_package": {
            "path": str(package_dir),
            "qc_report": str(package_dir / "package_qc_report.md"),
            "archive_manifest": str(package_dir.parent / f"{package_dir.name}.manifest.json"),
            "cross_cancer_recovery_min": package_manifest.get("cross_cancer_recovery_min", ""),
            "axis_masked_recovery_min": package_manifest.get("axis_masked_recovery_min", ""),
            "stable_9_pathway_core": package_manifest.get("stable_9_pathway_core", []),
        },
        "verification": {
            "focused_tests": "PYTHONPATH=src pytest tests/test_morphopathway.py tests/test_topology_analysis.py -q",
            "package_qc": "python benchmarking/morphopathway_atera/scripts/validate_brief_communication_package.py <package_dir>",
            "archive": "python benchmarking/morphopathway_atera/scripts/archive_brief_communication_package.py <package_dir>",
        },
        "staging_guidance": {
            "stage": IN_SCOPE_PATHS,
            "do_not_stage_without_review": [row["path"] for row in unrelated],
        },
    }

    json_path = package_dir / "handoff_scope_inventory.json"
    md_path = package_dir / "handoff_scope_inventory.md"
    json_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    md_lines = [
        "# Handoff Scope Inventory",
        "",
        f"Generated UTC: {inventory['generated_utc']}",
        "",
        "## In-Scope Paths",
        *(f"- `{path}`" for path in IN_SCOPE_PATHS),
        "",
        "## Current In-Scope Git Status",
        *(f"- `{row['status']}` `{row['path']}`" for row in in_scope),
        *(["- none"] if not in_scope else []),
        "",
        "## Unrelated Dirty Paths To Avoid Staging",
        *(f"- `{row['status']}` `{row['path']}`" for row in unrelated),
        *(["- none"] if not unrelated else []),
        "",
        "## Final Package",
        f"- Package: `{package_dir}`",
        f"- Archive manifest: `{package_dir.parent / f'{package_dir.name}.manifest.json'}`",
        f"- Stable core pathways: {', '.join(map(str, package_manifest.get('stable_9_pathway_core', [])))}",
        "",
        "## Verification Commands",
        f"- `{inventory['verification']['focused_tests']}`",
        f"- `{inventory['verification']['package_qc']}`",
        f"- `{inventory['verification']['archive']}`",
    ]
    if unexpected_unrelated:
        md_lines.extend(
            [
                "",
                "## Unexpected Dirty Paths",
                *(f"- `{row['status']}` `{row['path']}`" for row in unexpected_unrelated),
            ]
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
