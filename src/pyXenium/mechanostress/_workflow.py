from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyXenium.io.sdata_model import XeniumSData

from ._axis import compute_ane_density, estimate_cell_axes, summarize_axial_orientation
from ._polarity import compute_cell_polarity, summarize_cell_polarity
from ._tumor_stroma import (
    classify_tumor_stroma_growth,
    compute_distance_expression_coupling,
    summarize_tumor_growth,
)
from ._types import MechanostressCohortResult, MechanostressConfig, MechanostressResult


_EMPTY_COUPLING_COLUMNS = [
    "gene",
    "n_cells",
    "n_nonzero_cells",
    "spearman_rho",
    "p_value",
    "mean_expression",
    "tumor_enriched",
]


def _empty_result(config: MechanostressConfig, *, sample_id: str | None = None) -> MechanostressResult:
    summary = {"sample_id": sample_id, "n_axes": 0, "n_tumor_growth_cells": 0, "n_polarity_cells": 0}
    return MechanostressResult(
        cell_axes=pd.DataFrame(),
        axis_strength_by_radius=pd.DataFrame(),
        orientation_summary=pd.DataFrame(),
        tumor_growth_cells=pd.DataFrame(),
        tumor_growth_summary=pd.DataFrame(),
        distance_expression_coupling=pd.DataFrame(columns=_EMPTY_COUPLING_COLUMNS),
        cell_polarity=pd.DataFrame(),
        polarity_summary=pd.DataFrame(),
        summary=summary,
        config=config.to_dict(),
    )


def _load_cell_table(cell_table: pd.DataFrame | str | Path | None, *, cell_id_col: str) -> pd.DataFrame | None:
    if cell_table is None:
        return None
    if isinstance(cell_table, pd.DataFrame):
        frame = cell_table.copy()
    else:
        frame = pd.read_csv(cell_table)
    if cell_id_col != "cell_id" and cell_id_col in frame.columns and "cell_id" not in frame.columns:
        frame = frame.rename(columns={cell_id_col: "cell_id"})
    if "cell_id" not in frame.columns:
        raise ValueError(f"cell_table must contain a 'cell_id' column or configured cell id column {cell_id_col!r}.")
    frame["cell_id"] = frame["cell_id"].astype(str)
    return frame


def _merge_external_cell_table(obs: pd.DataFrame, cell_table: pd.DataFrame | None) -> pd.DataFrame:
    if cell_table is None:
        return obs
    annotation = cell_table.copy()
    drop_columns = [column for column in annotation.columns if column != "cell_id" and column in obs.columns]
    base = obs.drop(columns=drop_columns, errors="ignore")
    merged = base.merge(annotation, on="cell_id", how="outer")
    return merged


def _obs_as_cell_table(
    sdata: XeniumSData,
    *,
    config: MechanostressConfig,
    cell_table: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    obs = sdata.table.obs.copy()
    obs.index = obs.index.astype(str)
    obs = obs.reset_index().rename(columns={obs.index.name or "index": "cell_id"})
    if "cell_id" not in obs.columns:
        obs = obs.rename(columns={obs.columns[0]: "cell_id"})
    obs["cell_id"] = obs["cell_id"].astype(str)
    obs = _merge_external_cell_table(
        obs,
        _load_cell_table(cell_table, cell_id_col=config.tumor_stroma.cell_id_col),
    )

    if config.tumor_stroma.x_col not in obs.columns or config.tumor_stroma.y_col not in obs.columns:
        if "spatial" in sdata.table.obsm:
            coords = np.asarray(sdata.table.obsm["spatial"], dtype=float)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                obs[config.tumor_stroma.x_col] = coords[:, 0]
                obs[config.tumor_stroma.y_col] = coords[:, 1]
    return obs


def _join_axis_metadata(axes: pd.DataFrame, cell_table: pd.DataFrame) -> pd.DataFrame:
    if axes.empty or cell_table.empty or "cell_id" not in cell_table.columns:
        return axes
    metadata_columns = [
        column
        for column in cell_table.columns
        if column not in {"cell_id", "x", "y", "x_centroid", "y_centroid", "cell_centroid_x", "cell_centroid_y"}
    ]
    if not metadata_columns:
        return axes
    metadata = cell_table[["cell_id", *metadata_columns]].copy()
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    return axes.merge(metadata.drop_duplicates("cell_id"), on="cell_id", how="left")


def _expression_frame_for_genes(sdata: XeniumSData, genes: tuple[str, ...]) -> pd.DataFrame | None:
    if not genes:
        return None
    var_names = pd.Index(sdata.table.var_names.astype(str))
    symbol_columns = [
        column
        for column in ("name", "gene_symbol", "symbol", "gene_name")
        if column in sdata.table.var.columns
    ]
    selected_var_names: list[str] = []
    output_names: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        query = str(gene)
        selected: str | None = None
        if query in var_names:
            selected = query
        else:
            for column in symbol_columns:
                matches = var_names[sdata.table.var[column].astype(str).to_numpy() == query]
                if len(matches):
                    selected = str(matches[0])
                    break
        if selected is None or selected in seen:
            continue
        seen.add(selected)
        selected_var_names.append(selected)
        output_names.append(query)
    if not selected_var_names:
        return pd.DataFrame(index=sdata.table.obs_names.astype(str), columns=[])
    subset = sdata.table[:, selected_var_names]
    matrix = subset.X
    values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return pd.DataFrame(values, index=subset.obs_names.astype(str), columns=output_names)


def _apply_axis_filters(axes: pd.DataFrame, *, config: MechanostressConfig) -> pd.DataFrame:
    if axes.empty:
        return axes
    filtered = axes.copy()
    threshold = float(config.axis.er_threshold)
    filtered = filtered.loc[pd.to_numeric(filtered["elongation_ratio"], errors="coerce") >= threshold].copy()
    if config.axis.cell_query:
        filtered = filtered.query(config.axis.cell_query, engine="python").copy()
    return filtered


def run_mechanostress_workflow(
    sdata: XeniumSData | None = None,
    *,
    base_path: str | Path | None = None,
    cell_table: pd.DataFrame | str | Path | None = None,
    config: MechanostressConfig | None = None,
    output_dir: str | Path | None = None,
) -> MechanostressResult:
    """Run the integrated mechanostress workflow on a XeniumSData object or Xenium export path."""

    config = config or MechanostressConfig()
    if sdata is None:
        if base_path is None:
            raise ValueError("Provide either sdata or base_path.")
        from pyXenium.io import read_xenium

        sdata = read_xenium(str(base_path), as_="sdata", include_transcripts=False, include_boundaries=True)
    if not isinstance(sdata, XeniumSData):
        raise TypeError(f"sdata must be a XeniumSData instance, got {type(sdata)!r}.")

    sample_id = config.sample_id or str(sdata.metadata.get("sample_id") or sdata.metadata.get("source_path") or "")
    cell_table = _obs_as_cell_table(sdata, config=config, cell_table=cell_table)
    result = _empty_result(config, sample_id=sample_id)

    if "nucleus_boundaries" in sdata.shapes:
        axes = estimate_cell_axes(sdata.shapes["nucleus_boundaries"])
        axes = _join_axis_metadata(axes, cell_table)
        result.cell_axes = axes
        filtered_axes = _apply_axis_filters(axes, config=config)
        groupby = [column for column in config.axis.groupby if column in filtered_axes.columns]
        result.orientation_summary = summarize_axial_orientation(
            filtered_axes,
            angle_col=config.axis.angle_col,
            x_col=config.axis.x_col,
            y_col=config.axis.y_col,
            groupby=groupby or None,
            local_k=config.axis.local_k,
        )
        result.axis_strength_by_radius = compute_ane_density(
            filtered_axes,
            radii_um=config.axis.radii_um,
            angle_col=config.axis.angle_col,
            x_col=config.axis.x_col,
            y_col=config.axis.y_col,
            groupby=groupby or None,
        )

    tumor_cfg = config.tumor_stroma
    if (
        tumor_cfg.annotation_col in cell_table.columns
        and tumor_cfg.x_col in cell_table.columns
        and tumor_cfg.y_col in cell_table.columns
    ):
        growth = classify_tumor_stroma_growth(
            cell_table,
            annotation_col=tumor_cfg.annotation_col,
            tumor_label=tumor_cfg.tumor_label,
            stroma_label=tumor_cfg.stroma_label,
            x_col=tumor_cfg.x_col,
            y_col=tumor_cfg.y_col,
            method=tumor_cfg.method,
        )
        result.tumor_growth_cells = growth
        result.tumor_growth_summary = summarize_tumor_growth(
            growth,
            annotation_col=tumor_cfg.annotation_col,
            tumor_label=tumor_cfg.tumor_label,
            sample_id=sample_id,
        )
        expression = _expression_frame_for_genes(sdata, config.coupling_genes)
        if expression is not None and expression.shape[1] > 0:
            result.distance_expression_coupling = compute_distance_expression_coupling(
                growth_table=growth,
                expression_df=expression,
                genes=config.coupling_genes,
                annotation_col=tumor_cfg.annotation_col,
                tumor_label=tumor_cfg.tumor_label,
                cell_id_col=tumor_cfg.cell_id_col,
            )

    if "cell_boundaries" in sdata.shapes and "nucleus_boundaries" in sdata.shapes:
        polarity = compute_cell_polarity(
            cell_boundaries=sdata.shapes["cell_boundaries"],
            nucleus_boundaries=sdata.shapes["nucleus_boundaries"],
            cell_id_col=config.polarity.cell_id_col,
            offset_norm_threshold=config.polarity.offset_norm_threshold,
        )
        polarity = _join_axis_metadata(polarity, cell_table)
        result.cell_polarity = polarity
        result.polarity_summary = summarize_cell_polarity(polarity, sample_id=sample_id)

    result.summary = build_mechanostress_summary(result, sample_id=sample_id)
    if output_dir is not None:
        files = write_mechanostress_artifacts(result, output_dir)
        result.files = files
        result.output_dir = Path(output_dir)
    return result


def _find_sample_cell_table(sample_dir: Path, annotation_glob: str | None) -> Path | None:
    if not annotation_glob:
        return None
    matches = sorted(path for path in sample_dir.glob(annotation_glob) if path.is_file())
    if len(matches) > 1:
        joined = "\n".join(str(path) for path in matches)
        raise FileExistsError(
            f"Multiple annotation tables matched {annotation_glob!r} under {sample_dir}. "
            f"Use a narrower --annotation-glob.\n{joined}"
        )
    return matches[0] if matches else None


def run_mechanostress_cohort(
    cohort_root: str | Path,
    *,
    output_dir: str | Path,
    config: MechanostressConfig | None = None,
    sample_glob: str = "*",
    annotation_glob: str | None = "*_cell_clusters_with_annotation_and_coord.csv",
    prefer: str = "auto",
    sample_limit: int | None = None,
    fail_fast: bool = False,
) -> MechanostressCohortResult:
    """Run the mechanostress workflow across sample directories and write cohort summaries."""

    root = Path(cohort_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = config or MechanostressConfig()

    sample_dirs = sorted(path for path in root.glob(sample_glob) if path.is_dir())
    if sample_limit is not None and int(sample_limit) > 0:
        sample_dirs = sample_dirs[: int(sample_limit)]

    from pyXenium.io import read_xenium

    results: dict[str, MechanostressResult] = {}
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        sample_config = config.copy_with(sample_id=sample_id)
        try:
            annotation_path = _find_sample_cell_table(sample_dir, annotation_glob)
            sdata = read_xenium(
                str(sample_dir),
                as_="sdata",
                prefer=prefer,
                include_transcripts=False,
                include_boundaries=True,
            )
            result = run_mechanostress_workflow(
                sdata,
                cell_table=annotation_path,
                config=sample_config,
                output_dir=out / sample_id,
            )
            results[sample_id] = result
            summaries.append(result.summary)
        except Exception as exc:
            error = {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            if fail_fast:
                raise

    summary_df = pd.json_normalize(summaries) if summaries else pd.DataFrame()
    errors_df = pd.DataFrame(errors)
    summary_csv = out / "cohort_summary.csv"
    errors_csv = out / "cohort_errors.csv"
    summary_json = out / "cohort_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    errors_df.to_csv(errors_csv, index=False)
    payload = {
        "cohort_root": str(root),
        "output_dir": str(out),
        "n_samples": int(len(sample_dirs)),
        "n_completed": int(len(results)),
        "n_failed": int(len(errors_df)),
        "summaries": summary_df.to_dict(orient="records"),
        "errors": errors_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    files = {
        "cohort_summary_csv": str(summary_csv),
        "cohort_errors_csv": str(errors_csv),
        "cohort_summary_json": str(summary_json),
    }
    return MechanostressCohortResult(results=results, summary=summary_df, errors=errors_df, files=files, output_dir=out)


def build_mechanostress_summary(result: MechanostressResult, *, sample_id: str | None = None) -> dict[str, Any]:
    tumor_summary: dict[str, Any] = {}
    if not result.tumor_growth_summary.empty:
        tumor_summary = result.tumor_growth_summary.iloc[0].dropna().to_dict()
    return {
        "sample_id": sample_id,
        "n_axes": int(len(result.cell_axes)),
        "n_axis_strength_rows": int(len(result.axis_strength_by_radius)),
        "n_orientation_summary_rows": int(len(result.orientation_summary)),
        "n_tumor_growth_cells": int(len(result.tumor_growth_cells)),
        "n_distance_expression_coupling_rows": int(len(result.distance_expression_coupling)),
        "n_polarity_cells": int(len(result.cell_polarity)),
        "tumor_growth": tumor_summary,
    }


def render_mechanostress_report(result: MechanostressResult | dict[str, Any]) -> str:
    """Render a compact markdown report for mechanostress outputs."""

    if isinstance(result, MechanostressResult):
        summary = result.summary
    else:
        summary = dict(result)
    lines = ["# pyXenium Mechanostress Report", ""]
    sample_id = summary.get("sample_id") or "unknown"
    lines.append(f"- Sample: `{sample_id}`")
    lines.append(f"- Axes computed: {summary.get('n_axes', 0)}")
    lines.append(f"- Axis strength rows: {summary.get('n_axis_strength_rows', 0)}")
    lines.append(f"- Tumor growth cells: {summary.get('n_tumor_growth_cells', 0)}")
    lines.append(f"- Polarity cells: {summary.get('n_polarity_cells', 0)}")
    tumor = summary.get("tumor_growth") or {}
    if tumor:
        infil = tumor.get("infiltrative_proportion")
        lines.append("")
        lines.append("## Tumor-Stroma Growth")
        lines.append(f"- Infiltrative fraction: {float(infil):.4f}" if infil is not None else "- Infiltrative fraction: n/a")
        lines.append(f"- Infiltrative cells: {tumor.get('n_infiltrative', 0)}")
        lines.append(f"- Expanding cells: {tumor.get('n_expanding', 0)}")
    return "\n".join(lines) + "\n"


def write_mechanostress_artifacts(result: MechanostressResult, output_dir: str | Path) -> dict[str, str]:
    """Write the fixed mechanostress artifact set to disk."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    table_paths = {
        "cell_axes": root / "cell_axes.csv",
        "axis_strength_by_radius": root / "axis_strength_by_radius.csv",
        "orientation_summary": root / "orientation_summary.csv",
        "tumor_growth_cells": root / "tumor_growth_cells.csv",
        "tumor_growth_summary": root / "tumor_growth_summary.csv",
        "distance_expression_coupling": root / "distance_expression_coupling.csv",
        "cell_polarity": root / "cell_polarity.csv",
        "polarity_summary": root / "polarity_summary.csv",
    }
    for key, path in table_paths.items():
        result.table_map()[key].to_csv(path, index=False)

    summary_path = root / "summary.json"
    summary_payload = {
        "summary": result.summary,
        "config": dict(result.config),
        "files": {key: str(path) for key, path in table_paths.items()},
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str) + "\n", encoding="utf-8")
    report_path = root / "report.md"
    report_path.write_text(render_mechanostress_report(result), encoding="utf-8")

    files = {key: str(path) for key, path in table_paths.items()}
    files.update({"summary_json": str(summary_path), "report_md": str(report_path), "figures_dir": str(figures)})
    result.files = files
    return files
