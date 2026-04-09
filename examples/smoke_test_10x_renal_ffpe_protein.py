from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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


def _top_rna_features(adata, top_n: int) -> list[dict]:
    feature_names = (
        adata.var["name"].astype(str).tolist()
        if "name" in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )
    nnz = np.asarray(adata.X.getnnz(axis=0)).ravel()
    total = np.asarray(adata.X.sum(axis=0)).ravel()

    order = np.argsort(-total)[:top_n]
    rows = []
    for idx in order:
        rows.append(
            {
                "feature": feature_names[idx],
                "detected_cells": int(nnz[idx]),
                "total_counts": float(total[idx]),
            }
        )
    return rows


def _top_protein_markers(adata, top_n: int) -> list[dict]:
    protein = adata.obsm["protein"]
    protein_df = protein if isinstance(protein, pd.DataFrame) else pd.DataFrame(protein, index=adata.obs_names)

    mean_signal = protein_df.mean(axis=0).sort_values(ascending=False)
    rows = []
    for marker, value in mean_signal.head(top_n).items():
        rows.append(
            {
                "marker": str(marker),
                "mean_signal": float(value),
                "positive_cells": int((protein_df[marker] > 0).sum()),
            }
        )
    return rows


def _top_clusters(adata, top_n: int) -> list[dict]:
    if "cluster" not in adata.obs.columns:
        return []

    counts = adata.obs["cluster"].astype(str).value_counts().head(top_n)
    return [{"cluster": str(cluster), "n_cells": int(count)} for cluster, count in counts.items()]


def build_summary(base_path: str, prefer: str, top_n: int = 10) -> tuple[dict, object]:
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
        "top_rna_features_by_total_counts": _top_rna_features(adata, top_n=top_n),
        "top_protein_markers_by_mean_signal": _top_protein_markers(adata, top_n=top_n),
        "largest_clusters": _top_clusters(adata, top_n=top_n),
    }

    metrics_path = Path(base_path) / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "num_cells_detected" in metrics.columns and not metrics.empty:
            summary["metrics_summary_num_cells_detected"] = int(metrics.loc[0, "num_cells_detected"])

    return summary, adata


def render_markdown_report(payload: dict) -> str:
    summary = payload["summary"]
    validated = payload["validated_reference"]
    issues = payload["issues"]

    lines = [
        "# pyXenium Smoke Test Report",
        "",
        f"Dataset: {summary['dataset_title']}",
        f"Source: {summary['dataset_url']}",
        f"Local path: `{summary['base_path']}`",
        f"Backend preference: `{summary['prefer']}`",
        "",
        "## Core Results",
        "",
        f"- Cells: `{summary['n_cells']}`",
        f"- RNA features: `{summary['n_rna_features']}`",
        f"- Protein markers: `{summary['n_protein_markers']}`",
        f"- Sparse matrix nnz: `{summary['x_nnz']}`",
        f"- Spatial coordinates present: `{summary['has_spatial']}`",
        f"- Cluster labels present: `{summary['has_cluster']}`",
        f"- metrics_summary.csv detected cells: `{summary['metrics_summary_num_cells_detected']}`",
        "",
        "## Validated Reference",
        "",
        f"- Expected cells: `{validated['expected_cells']}`",
        f"- Expected RNA features: `{validated['expected_rna_features']}`",
        f"- Expected protein markers: `{validated['expected_protein_markers']}`",
        "",
        "## Largest Clusters",
        "",
    ]

    for row in summary["largest_clusters"]:
        lines.append(f"- `{row['cluster']}`: `{row['n_cells']}` cells")

    lines.extend(["", "## Top RNA Features by Total Counts", ""])
    for row in summary["top_rna_features_by_total_counts"]:
        lines.append(
            f"- `{row['feature']}`: total counts `{row['total_counts']:.0f}`, detected cells `{row['detected_cells']}`"
        )

    lines.extend(["", "## Top Protein Markers by Mean Signal", ""])
    for row in summary["top_protein_markers_by_mean_signal"]:
        lines.append(
            f"- `{row['marker']}`: mean signal `{row['mean_signal']:.4f}`, positive cells `{row['positive_cells']}`"
        )

    lines.extend(["", "## Issues", ""])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No issues detected.")

    lines.append("")
    return "\n".join(lines)


def write_output_artifacts(payload: dict, output_dir: str | None) -> None:
    if not output_dir:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown_report(payload), encoding="utf-8")
    pd.DataFrame(payload["summary"]["top_rna_features_by_total_counts"]).to_csv(
        out / "top_rna_features.csv", index=False
    )
    pd.DataFrame(payload["summary"]["top_protein_markers_by_mean_signal"]).to_csv(
        out / "top_protein_markers.csv", index=False
    )
    pd.DataFrame(payload["summary"]["largest_clusters"]).to_csv(out / "largest_clusters.csv", index=False)


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
        "--top-n",
        type=int,
        default=10,
        help="Number of top RNA features, protein markers, and clusters to report.",
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for report.md, summary.json, and CSV summaries.",
    )
    parser.add_argument(
        "--write-h5ad",
        default=None,
        help="Optional path to export the loaded AnnData object as an .h5ad file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, adata = build_summary(base_path=args.base_path, prefer=args.prefer, top_n=args.top_n)
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
    write_output_artifacts(payload, args.output_dir)

    if args.write_h5ad:
        h5ad_path = Path(args.write_h5ad)
        h5ad_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(h5ad_path)

    if issues and not args.allow_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
