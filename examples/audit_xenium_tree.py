from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyXenium.io import read_sdata, read_xenium, write_xenium


BACKEND_MARKERS = {
    "zarr": "cell_feature_matrix.zarr",
    "h5": "cell_feature_matrix.h5",
    "mex": "cell_feature_matrix",
}
BACKEND_PRIORITY = ("zarr", "h5", "mex")
AUDIT_INCLUDE_TRANSCRIPTS = False
AUDIT_INCLUDE_BOUNDARIES = False
AUDIT_INCLUDE_IMAGES = False


def discover_candidate_roots(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("experiment.xenium"))


def available_backends(sample_root: Path) -> list[str]:
    available: list[str] = []
    if (sample_root / BACKEND_MARKERS["zarr"]).exists():
        available.append("zarr")
    if (sample_root / BACKEND_MARKERS["h5"]).exists():
        available.append("h5")

    mex_root = sample_root / BACKEND_MARKERS["mex"]
    mex_files = (
        mex_root / "matrix.mtx.gz",
        mex_root / "features.tsv.gz",
        mex_root / "barcodes.tsv.gz",
    )
    if all(path.exists() for path in mex_files):
        available.append("mex")
    return available


def classify_root(sample_root: Path) -> str:
    return "complete" if available_backends(sample_root) else "skipped"


def canonical_backend(backends: list[str]) -> str:
    for backend in BACKEND_PRIORITY:
        if backend in backends:
            return backend
    raise ValueError("No backend available for canonical selection.")


def sanitize_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"sample_{digest}"


def analysis_snapshot(adata) -> dict[str, Any]:
    analysis = dict(adata.uns.get("xenium_analysis", {}))
    return {
        "default_cluster_key": analysis.get("default_cluster_key"),
        "cluster_keys": sorted((analysis.get("cluster_columns") or {}).keys()),
        "projection_methods": sorted((analysis.get("projection_keys") or {}).keys()),
    }


def _write_report_snapshot(report: dict[str, Any], output_json: Path | None) -> None:
    if output_json is None:
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def verify_roundtrip(original, reloaded) -> None:
    if tuple(original.table.shape) != tuple(reloaded.table.shape):
        raise AssertionError(
            f"Table shape changed across roundtrip: {original.table.shape} -> {reloaded.table.shape}"
        )

    original_analysis = original.table.uns.get("xenium_analysis", {})
    reloaded_analysis = reloaded.table.uns.get("xenium_analysis", {})

    for key in ("cluster",):
        if key in original.table.obs.columns and key not in reloaded.table.obs.columns:
            raise AssertionError(f"Missing obs column after roundtrip: {key}")

    for key, column_name in (original_analysis.get("cluster_columns") or {}).items():
        if column_name not in reloaded.table.obs.columns:
            raise AssertionError(f"Missing clustering column after roundtrip: {column_name}")

    for method, obsm_key in (original_analysis.get("projection_keys") or {}).items():
        if obsm_key not in reloaded.table.obsm:
            raise AssertionError(f"Missing projection after roundtrip: {method} -> {obsm_key}")
        if original.table.obsm[obsm_key].shape != reloaded.table.obsm[obsm_key].shape:
            raise AssertionError(
                f"Projection shape changed for {obsm_key}: "
                f"{original.table.obsm[obsm_key].shape} -> {reloaded.table.obsm[obsm_key].shape}"
            )

    if original_analysis.get("default_cluster_key") != reloaded_analysis.get("default_cluster_key"):
        raise AssertionError("Default clustering key changed across roundtrip.")


def audit_complete_root(sample_root: Path, scratch_root: Path) -> dict[str, Any]:
    backends = available_backends(sample_root)
    result: dict[str, Any] = {
        "root": str(sample_root),
        "status": "ok",
        "available_backends": backends,
        "backend_reads": [],
        "analysis": {},
        "artifacts": {},
    }

    for backend in backends:
        adata = read_xenium(
            str(sample_root),
            as_="anndata",
            prefer=backend,
            include_boundaries=AUDIT_INCLUDE_BOUNDARIES,
            include_images=AUDIT_INCLUDE_IMAGES,
        )
        if adata.n_obs <= 0:
            raise AssertionError(f"{sample_root} produced an empty AnnData for backend={backend}.")
        if "spatial" not in adata.obsm:
            raise AssertionError(f"{sample_root} is missing obsm['spatial'] for backend={backend}.")

        snapshot = analysis_snapshot(adata)
        if snapshot["default_cluster_key"] is not None and "cluster" not in adata.obs.columns:
            raise AssertionError(f"{sample_root} reported a default cluster key but no obs['cluster'].")
        for method in snapshot["projection_methods"]:
            obsm_key = f"X_{method}"
            if obsm_key not in adata.obsm:
                raise AssertionError(f"{sample_root} missing {obsm_key} for backend={backend}.")
            if adata.obsm[obsm_key].shape[0] != adata.n_obs:
                raise AssertionError(f"{sample_root} has a row mismatch in {obsm_key}.")

        result["backend_reads"].append(
            {
                "backend": backend,
                "shape": [int(adata.n_obs), int(adata.n_vars)],
                "cluster_keys": snapshot["cluster_keys"],
                "projection_methods": snapshot["projection_methods"],
            }
        )

    chosen_backend = canonical_backend(backends)
    result["canonical_backend"] = chosen_backend

    sdata = read_xenium(
        str(sample_root),
        as_="sdata",
        prefer=chosen_backend,
        include_transcripts=AUDIT_INCLUDE_TRANSCRIPTS,
        include_boundaries=AUDIT_INCLUDE_BOUNDARIES,
        include_images=AUDIT_INCLUDE_IMAGES,
    )
    result["analysis"] = analysis_snapshot(sdata.table)
    result["artifacts"]["shape_keys"] = sorted(sdata.shapes.keys())
    result["artifacts"]["point_keys"] = sorted(sdata.points.keys())

    sample_scratch = scratch_root / sanitize_name(sample_root)
    sample_scratch.mkdir(parents=True, exist_ok=True)
    sdata_store = sample_scratch / "roundtrip.sdata.zarr"
    h5ad_path = sample_scratch / "roundtrip.h5ad"

    sdata_payload = write_xenium(sdata, sdata_store, format="sdata", overwrite=True)
    reloaded = read_sdata(sdata_store)
    verify_roundtrip(sdata, reloaded)

    h5ad_payload = write_xenium(sdata.table, h5ad_path, format="h5ad", overwrite=True)
    result["artifacts"]["sdata_output"] = sdata_payload["output_path"]
    result["artifacts"]["h5ad_output"] = h5ad_payload["output_path"]

    return result


def run_audit(root: Path, *, output_json: Path | None = None) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {
        "root": str(root),
        "started_at_epoch": started,
        "candidates": [],
        "complete": [],
        "skipped": [],
        "failed": [],
    }

    with tempfile.TemporaryDirectory(prefix="pyxenium_audit_") as temp_dir:
        scratch_root = Path(temp_dir)
        candidates = discover_candidate_roots(root)
        report["candidates"] = [str(path) for path in candidates]
        _write_report_snapshot(report, output_json)

        for sample_root in candidates:
            print(f"[audit] {sample_root}", flush=True)
            status = classify_root(sample_root)
            if status == "skipped":
                report["skipped"].append(
                    {
                        "root": str(sample_root),
                        "reason": "experiment.xenium present but no supported matrix backend found",
                    }
                )
                _write_report_snapshot(report, output_json)
                continue

            try:
                result = audit_complete_root(sample_root, scratch_root)
                report["complete"].append(result)
            except Exception as exc:
                report["failed"].append(
                    {
                        "root": str(sample_root),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            _write_report_snapshot(report, output_json)

        shutil.rmtree(scratch_root, ignore_errors=True)

    report["finished_at_epoch"] = time.time()
    report["duration_seconds"] = round(report["finished_at_epoch"] - started, 2)
    report["summary"] = {
        "n_candidates": len(report["candidates"]),
        "n_complete": len(report["complete"]),
        "n_skipped": len(report["skipped"]),
        "n_failed": len(report["failed"]),
    }
    _write_report_snapshot(report, output_json)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit pyXenium read/write compatibility across a Xenium dataset tree."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory that contains one or more Xenium output folders.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file to write the audit report to.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_audit(args.root.expanduser(), output_json=args.output_json)
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
