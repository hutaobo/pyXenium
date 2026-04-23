from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyXenium.contour import build_contour_feature_table
from pyXenium.io import read_sdata, read_xenium
from pyXenium.io.sdata_model import XeniumSData

from ..contour_boundary_ecology import (
    DEFAULT_BOUNDARY_PROGRAM_LIBRARY,
    associate_contour_image_omics,
    cluster_contour_ecotypes,
    score_contour_boundary_programs,
)

__all__ = [
    "render_contour_boundary_ecology_report",
    "run_contour_boundary_ecology_pilot",
    "write_contour_boundary_ecology_artifacts",
]


def run_contour_boundary_ecology_pilot(
    sdata_or_path: XeniumSData | str | Path,
    *,
    contour_key: str,
    output_dir: str | Path | None = None,
    embedding_backend: Any = None,
    neighbor_k: int = 6,
) -> dict[str, Any]:
    sdata = _resolve_sdata(sdata_or_path)
    feature_table = build_contour_feature_table(
        sdata,
        contour_key=contour_key,
        include_pathomics=True,
        embedding_backend=embedding_backend,
    )
    program_result = score_contour_boundary_programs(
        sdata,
        contour_key=contour_key,
        feature_table=feature_table,
    )
    ecotype_assignments, ecotype_summary = cluster_contour_ecotypes(
        feature_table["contour_features"],
        program_result["program_scores"],
    )
    association_summary = associate_contour_image_omics(
        feature_table,
        program_result["program_scores"],
        ecotype_assignments,
    )

    sample_summary = _build_sample_summary(
        sdata=sdata,
        contour_key=contour_key,
        feature_table=feature_table,
        program_scores=program_result["program_scores"],
        ecotype_summary=ecotype_summary,
        matched_exemplars=association_summary["matched_exemplars"],
        embedding_backend=embedding_backend,
        neighbor_k=neighbor_k,
    )
    result = {
        "contour_features": feature_table["contour_features"],
        "program_scores": program_result["program_scores"],
        "ecotype_assignments": ecotype_assignments,
        "association_summary": association_summary,
        "matched_exemplars": association_summary["matched_exemplars"],
        "edge_gradients": feature_table["edge_gradients"],
        "sample_summary": sample_summary,
    }

    if output_dir is not None:
        artifact_dir = write_contour_boundary_ecology_artifacts(
            sdata=sdata,
            contour_key=contour_key,
            result=result,
            output_dir=output_dir,
        )
        result["artifact_dir"] = str(artifact_dir)
    return result


def write_contour_boundary_ecology_artifacts(
    *,
    sdata: XeniumSData,
    contour_key: str,
    result: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result["contour_features"].to_csv(out / "contour_features.csv", index=False)
    result["program_scores"].to_csv(out / "program_scores.csv", index=False)
    result["ecotype_assignments"].to_csv(out / "ecotype_assignments.csv", index=False)
    result["edge_gradients"].to_csv(out / "edge_gradients.csv", index=False)
    result["matched_exemplars"].to_csv(out / "matched_exemplars.csv", index=False)

    association = result["association_summary"]
    for key, value in association.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f"{key}.csv", index=False)

    summary_json = {
        "sample_summary": result["sample_summary"],
        "top_program_counts": _program_prevalence(result["program_scores"]),
    }
    (out / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(
        render_contour_boundary_ecology_report(result),
        encoding="utf-8",
    )
    montage_path = _render_exemplar_montage(
        sdata=sdata,
        contour_key=contour_key,
        matched_exemplars=result["matched_exemplars"],
        output_path=out / "exemplar_montage.png",
    )
    if montage_path is not None:
        summary_json["exemplar_montage"] = str(montage_path)
        (out / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    return out


def render_contour_boundary_ecology_report(result: dict[str, Any]) -> str:
    summary = result["sample_summary"]
    hypothesis_table = result["association_summary"]["hypothesis_ranking"]
    matched = result["matched_exemplars"].head(6)
    delta_table = result["association_summary"]["program_feature_deltas"].head(12)
    top_program = result["program_scores"]["top_program"].value_counts()

    lines = [
        "# Contour Boundary Ecology Discovery Package",
        "",
        f"Sample ID: `{summary['sample_id']}`",
        f"Contour layer: `{summary['contour_key']}`",
        "",
        "## Sample Summary",
        "",
        f"- Contours analysed: `{summary['n_contours']}`",
        f"- Cells in table: `{summary['n_cells']}`",
        f"- Ecotypes discovered: `{summary['n_ecotypes']}`",
        f"- Bootstrap ecotype stability (ARI): `{summary['bootstrap_mean_ari']:.3f}`" if np.isfinite(summary["bootstrap_mean_ari"]) else "- Bootstrap ecotype stability (ARI): `nan`",
        f"- Embedding backend enabled: `{summary['embedding_backend']}`",
        "",
        "## Boundary Programs",
        "",
    ]
    for program_name, count in top_program.items():
        lines.append(f"- `{program_name}`: top program for `{int(count)}` contours")

    lines.extend(["", "## Ranked Hypotheses", ""])
    for _, row in hypothesis_table.iterrows():
        lines.append(f"- `{row['program']}`: {row['hypothesis']}")

    lines.extend(["", "## Matched Exemplars", ""])
    for _, row in matched.iterrows():
        lines.append(
            f"- `{row['program']}` exemplar `{row['exemplar_id']}` vs control `{row['control_id']}` "
            f"(delta `{row['delta_score']:.3f}`)"
        )

    lines.extend(["", "## Top Differential Signals", ""])
    for _, row in delta_table.iterrows():
        lines.append(
            f"- `{row['program']}` / `{row['feature_kind']}` / `{row['feature_name']}`: "
            f"effect `{row['effect_size']:.3f}`, matched delta `{row['matched_control_delta']:.3f}`, "
            f"FDR `{row['fdr']:.3g}`"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_sdata(sdata_or_path: XeniumSData | str | Path) -> XeniumSData:
    if isinstance(sdata_or_path, XeniumSData):
        return sdata_or_path
    resolved = Path(sdata_or_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    if resolved.suffix.lower() == ".zarr":
        return read_sdata(resolved)
    return read_xenium(
        str(resolved),
        as_="sdata",
        include_transcripts=True,
        include_boundaries=True,
        include_images=True,
    )


def _build_sample_summary(
    *,
    sdata: XeniumSData,
    contour_key: str,
    feature_table: dict[str, Any],
    program_scores: pd.DataFrame,
    ecotype_summary: dict[str, Any],
    matched_exemplars: pd.DataFrame,
    embedding_backend: Any,
    neighbor_k: int,
) -> dict[str, Any]:
    return {
        "sample_id": feature_table["sample_id"],
        "contour_key": contour_key,
        "n_contours": int(len(feature_table["contour_features"])),
        "n_cells": int(sdata.table.n_obs),
        "n_ecotypes": int(ecotype_summary["n_clusters"]),
        "bootstrap_mean_ari": float(ecotype_summary["bootstrap_mean_ari"]),
        "silhouette": float(ecotype_summary["silhouette"]) if np.isfinite(ecotype_summary["silhouette"]) else float("nan"),
        "feature_count": int(ecotype_summary["feature_count"]),
        "embedding_backend": bool(embedding_backend is not None),
        "neighbor_k": int(neighbor_k),
        "matched_exemplar_pairs": int(len(matched_exemplars)),
        "top_program_counts": _program_prevalence(program_scores),
    }


def _program_prevalence(program_scores: pd.DataFrame) -> dict[str, int]:
    counts = program_scores["top_program"].value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _render_exemplar_montage(
    *,
    sdata: XeniumSData,
    contour_key: str,
    matched_exemplars: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    if contour_key not in sdata.contour_images or matched_exemplars.empty:
        return None
    patches = sdata.contour_images[contour_key]
    preview = matched_exemplars.head(3)
    if preview.empty:
        return None

    n_rows = len(preview)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, max(3, 3 * n_rows)))
    if n_rows == 1:
        axes = np.asarray([axes])
    for row_index, (_, row) in enumerate(preview.iterrows()):
        exemplar = np.asarray(patches[str(row["exemplar_id"])].levels[0])
        control = np.asarray(patches[str(row["control_id"])].levels[0])
        axes[row_index, 0].imshow(_safe_display_image(exemplar))
        axes[row_index, 0].set_title(f"{row['program']} exemplar\n{row['exemplar_id']}")
        axes[row_index, 1].imshow(_safe_display_image(control))
        axes[row_index, 1].set_title(f"matched control\n{row['control_id']}")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _safe_display_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3]
    if array.ndim == 3 and array.shape[0] >= 3:
        return np.moveaxis(array[:3, :, :], 0, -1)
    return np.squeeze(array)
