from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from pyXenium.datasets import RENAL_FFPE_PROTEIN_10X_DATASET
    from pyXenium.io.xenium_gene_protein_loader import load_xenium_gene_protein
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from pyXenium.datasets import RENAL_FFPE_PROTEIN_10X_DATASET
    from pyXenium.io.xenium_gene_protein_loader import load_xenium_gene_protein


DEFAULT_DATASET_PATH = (
    r"Y:\long\10X_datasets\Xenium\Xenium_Renal\Xenium_V1_Human_Kidney_FFPE_Protein"
)
EXPECTED_CELLS = 465545
EXPECTED_RNA_FEATURES = 405
EXPECTED_PROTEIN_MARKERS = 27


def build_summary(base_path: str, prefer: str) -> dict:
    adata = load_xenium_gene_protein(base_path=base_path, prefer=prefer)

    protein = adata.obsm.get("protein")
    protein_shape = getattr(protein, "shape", None)
    protein_markers = int(protein_shape[1]) if protein_shape is not None else 0

    summary = {
        "dataset_title": RENAL_FFPE_PROTEIN_10X_DATASET.title,
        "dataset_url": RENAL_FFPE_PROTEIN_10X_DATASET.url,
        "base_path": base_path,
        "prefer": prefer,
        "n_cells": int(adata.n_obs),
        "n_rna_features": int(adata.n_vars),
        "n_protein_markers": protein_markers,
        "x_nnz": int(getattr(adata.X, "nnz", 0)),
        "has_spatial": "spatial" in adata.obsm,
        "has_cluster": "cluster" in adata.obs.columns,
        "obsm_keys": sorted(adata.obsm.keys()),
        "metrics_summary_num_cells_detected": None,
    }

    metrics_path = Path(base_path) / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "num_cells_detected" in metrics.columns and not metrics.empty:
            summary["metrics_summary_num_cells_detected"] = int(metrics.loc[0, "num_cells_detected"])

    return summary


def validate_summary(summary: dict) -> list[str]:
    issues: list[str] = []

    if summary["n_cells"] != EXPECTED_CELLS:
        issues.append(f"Expected {EXPECTED_CELLS} cells, observed {summary['n_cells']}.")
    if summary["n_rna_features"] != EXPECTED_RNA_FEATURES:
        issues.append(
            f"Expected {EXPECTED_RNA_FEATURES} RNA features, observed {summary['n_rna_features']}."
        )
    if summary["n_protein_markers"] != EXPECTED_PROTEIN_MARKERS:
        issues.append(
            f"Expected {EXPECTED_PROTEIN_MARKERS} protein markers, observed {summary['n_protein_markers']}."
        )
    if not summary["has_spatial"]:
        issues.append("Expected adata.obsm['spatial'] to be present.")
    if not summary["has_cluster"]:
        issues.append("Expected adata.obs['cluster'] to be present.")

    metric_cells = summary["metrics_summary_num_cells_detected"]
    if metric_cells is not None and metric_cells != summary["n_cells"]:
        issues.append(
            "metrics_summary.csv reports "
            f"{metric_cells} detected cells, but pyXenium loaded {summary['n_cells']} cells."
        )

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test pyXenium on the official 10x Genomics FFPE Human Renal Cell Carcinoma "
            "RNA + Protein Xenium dataset."
        )
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        default=os.environ.get("PYXENIUM_DATASET_PATH", DEFAULT_DATASET_PATH),
        help=(
            "Local path to the Xenium dataset directory. Defaults to the "
            "PYXENIUM_DATASET_PATH environment variable or the validated local path."
        ),
    )
    parser.add_argument(
        "--prefer",
        choices=("auto", "zarr", "h5", "mex"),
        default="auto",
        help="Preferred matrix backend passed to load_xenium_gene_protein().",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Print the summary even if the observed values differ from the validated reference.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(base_path=args.base_path, prefer=args.prefer)
    issues = validate_summary(summary)

    payload = {
        "summary": summary,
        "validated_reference": {
            "expected_cells": EXPECTED_CELLS,
            "expected_rna_features": EXPECTED_RNA_FEATURES,
            "expected_protein_markers": EXPECTED_PROTEIN_MARKERS,
        },
        "issues": issues,
    }

    rendered = json.dumps(payload, indent=2)
    print(rendered)

    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")

    if issues and not args.allow_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
