from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from shapely import points
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyXenium.contour import build_contour_feature_table
from pyXenium.contour._analysis import _prepare_contours
from pyXenium.contour._feature_table import _build_contour_zones, _geometry_membership_mask
from pyXenium.io.sdata_model import XeniumSData
from pyXenium.mechanostress import compute_cell_polarity, estimate_cell_axes

from .contour_boundary_ecology import score_contour_boundary_programs

__all__ = ["compare_he_vs_xenium_morphology_sources"]

_ID_COLUMNS = ["sample_id", "contour_key", "contour_id"]
_HE_PREFIXES = (
    "pathomics__",
    "embedding__",
    "bmnet__",
    "descriptor__",
    "cellsam__",
    "pathology__",
    "edge_contrast__pathomics__",
    "edge_contrast__bmnet__",
    "edge_contrast__descriptor__",
    "edge_contrast__cellsam__",
    "edge_contrast__pathology__",
)
_NATIVE_PREFIX = "xenium_native__"


def compare_he_vs_xenium_morphology_sources(
    sdata: XeniumSData,
    *,
    contour_key: str,
    feature_table: dict[str, Any] | None = None,
    program_scores: pd.DataFrame | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    random_state: int = 0,
    min_contours: int = 8,
) -> dict[str, Any]:
    """Test whether H&E morphology adds information beyond Xenium-native morphology."""

    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance.")
    feature_table = feature_table or build_contour_feature_table(sdata, contour_key=contour_key)
    contour_features = pd.DataFrame(feature_table["contour_features"]).copy()
    if contour_features.empty:
        raise ValueError("Contour feature table is empty.")

    scores = _coerce_program_scores(program_scores, sdata=sdata, contour_key=contour_key, feature_table=feature_table)
    xenium_native = _build_xenium_native_morphology(sdata, contour_key=contour_key, contour_features=contour_features)
    he_morphology = _select_prefixed_features(contour_features, prefixes=_HE_PREFIXES)
    redundancy = _compute_feature_redundancy(he_morphology, xenium_native)
    incremental = _compute_incremental_prediction(
        contour_features=contour_features,
        program_scores=scores,
        he_morphology=he_morphology,
        xenium_native=xenium_native,
        random_state=random_state,
        min_contours=min_contours,
    )
    partial = _compute_partial_associations(
        feature_table=feature_table,
        contour_features=contour_features,
        program_scores=scores,
        he_morphology=he_morphology,
        xenium_native=xenium_native,
    )
    review = _build_matched_review_table(
        contour_features=contour_features,
        program_scores=scores,
        he_morphology=he_morphology,
    )
    summary = {
        "sample_id": str(feature_table.get("sample_id", contour_features["sample_id"].iloc[0])),
        "contour_key": contour_key,
        "n_contours": int(len(contour_features)),
        "n_he_morphology_features": int(max(0, he_morphology.shape[1] - len(_ID_COLUMNS))),
        "n_xenium_native_morphology_features": int(max(0, xenium_native.shape[1] - len(_ID_COLUMNS))),
        "xenium_native_available": bool(
            "cell_boundaries" in sdata.shapes or "nucleus_boundaries" in sdata.shapes
        ),
        "has_bmnet_features": bool(any(str(column).startswith("bmnet__") for column in he_morphology.columns)),
        "min_contours_for_cv": int(min_contours),
        "evaluation_mode": (
            "cross_validated"
            if len(contour_features) >= int(min_contours)
            else "in_sample_small_n"
        ),
    }
    result = {
        "xenium_native_morphology": xenium_native,
        "he_morphology_features": he_morphology,
        "feature_redundancy": redundancy,
        "incremental_prediction": incremental,
        "partial_associations": partial,
        "matched_review_table": review,
        "summary": summary,
    }
    if output_dir is not None:
        _write_morphology_increment_artifacts(result, output_dir)
    return result


def _coerce_program_scores(
    program_scores: pd.DataFrame | dict[str, Any] | None,
    *,
    sdata: XeniumSData,
    contour_key: str,
    feature_table: dict[str, Any],
) -> pd.DataFrame:
    if program_scores is None:
        return score_contour_boundary_programs(
            sdata,
            contour_key=contour_key,
            feature_table=feature_table,
        )["program_scores"]
    if isinstance(program_scores, Mapping):
        if "program_scores" not in program_scores:
            raise KeyError("program_scores mapping must contain a `program_scores` entry.")
        return pd.DataFrame(program_scores["program_scores"]).copy()
    return pd.DataFrame(program_scores).copy()


def _build_xenium_native_morphology(
    sdata: XeniumSData,
    *,
    contour_key: str,
    contour_features: pd.DataFrame,
) -> pd.DataFrame:
    contour_table = _prepare_contours(sdata=sdata, contour_key=contour_key, contour_query=None)
    contour_table = contour_table.sort_values("contour_id", kind="stable").reset_index(drop=True)
    metrics = _build_cell_morphology_metrics(sdata)
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        for _, contour in contour_table.iterrows():
            rows.append(_identifier_row(contour_features, str(contour["contour_id"]), contour_key=contour_key))
        return pd.DataFrame(rows)

    point_array = np.asarray(points(metrics["x"].to_numpy(dtype=float), metrics["y"].to_numpy(dtype=float)), dtype=object)
    point_xy = metrics[["x", "y"]].to_numpy(dtype=float)
    for _, contour in contour_table.iterrows():
        contour_id = str(contour["contour_id"])
        row = _identifier_row(contour_features, contour_id, contour_key=contour_key)
        zones = _build_contour_zones(contour["geometry"], inner_rim_um=20.0, outer_rim_um=30.0)
        for zone_name, geometry in zones.items():
            area = float(geometry.area) if geometry is not None and not geometry.is_empty else 0.0
            membership = _geometry_membership_mask(geometry, point_array, point_xy=point_xy)
            row.update(_summarize_native_zone(metrics.loc[membership], zone_name=zone_name, area_um2=area))
        row.update(_outer_minus_inner(row, prefix=_NATIVE_PREFIX.rstrip("__")))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("contour_id", kind="stable").reset_index(drop=True)


def _build_cell_morphology_metrics(sdata: XeniumSData) -> pd.DataFrame:
    if "spatial" not in sdata.table.obsm:
        return pd.DataFrame(columns=["cell_id", "x", "y"])
    spatial = np.asarray(sdata.table.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        return pd.DataFrame(columns=["cell_id", "x", "y"])
    metrics = pd.DataFrame(
        {
            "cell_id": sdata.table.obs_names.astype(str),
            "x": spatial[:, 0],
            "y": spatial[:, 1],
        }
    )
    if "cell_boundaries" in sdata.shapes:
        cell_axes = estimate_cell_axes(sdata.shapes["cell_boundaries"]).rename(
            columns={
                "centroid_x": "cell_axis_centroid_x",
                "centroid_y": "cell_axis_centroid_y",
                "elongation_ratio": "cell_elongation_ratio",
                "major_variance": "cell_major_variance",
                "minor_variance": "cell_minor_variance",
                "n_vertices": "cell_boundary_vertices",
            }
        )
        metrics = metrics.merge(cell_axes.drop(columns=["axis_angle_degrees", "axis_angle_radians"], errors="ignore"), on="cell_id", how="left")
    if "nucleus_boundaries" in sdata.shapes:
        nucleus_axes = estimate_cell_axes(sdata.shapes["nucleus_boundaries"]).rename(
            columns={
                "centroid_x": "nucleus_axis_centroid_x",
                "centroid_y": "nucleus_axis_centroid_y",
                "elongation_ratio": "nucleus_elongation_ratio",
                "major_variance": "nucleus_major_variance",
                "minor_variance": "nucleus_minor_variance",
                "n_vertices": "nucleus_boundary_vertices",
            }
        )
        metrics = metrics.merge(
            nucleus_axes.drop(columns=["axis_angle_degrees", "axis_angle_radians"], errors="ignore"),
            on="cell_id",
            how="left",
        )
    if "cell_boundaries" in sdata.shapes and "nucleus_boundaries" in sdata.shapes:
        polarity = compute_cell_polarity(
            cell_boundaries=sdata.shapes["cell_boundaries"],
            nucleus_boundaries=sdata.shapes["nucleus_boundaries"],
        )
        metrics = metrics.merge(polarity, on="cell_id", how="left", suffixes=("", "_polarity"))
        if "cell_area" in metrics.columns and "nucleus_area" in metrics.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                metrics["nucleus_cell_area_ratio"] = metrics["nucleus_area"] / metrics["cell_area"]
    return metrics


def _summarize_native_zone(frame: pd.DataFrame, *, zone_name: str, area_um2: float) -> dict[str, float]:
    prefix = f"{_NATIVE_PREFIX}{zone_name}__"
    n_cells = int(len(frame))
    output = {
        f"{prefix}n_cells": float(n_cells),
        f"{prefix}cell_density": float(n_cells / area_um2) if area_um2 > 0 else 0.0,
    }
    if frame.empty:
        return output
    for column in _numeric_columns_any(frame, exclude={"x", "y"}):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        output[f"{prefix}{column}__mean"] = float(np.mean(values))
        output[f"{prefix}{column}__median"] = float(np.median(values))
        if values.size > 1:
            output[f"{prefix}{column}__std"] = float(np.std(values))
    if "polarized" in frame.columns:
        output[f"{prefix}polarized_fraction"] = float(frame["polarized"].fillna(False).astype(bool).mean())
    return output


def _select_prefixed_features(frame: pd.DataFrame, *, prefixes: tuple[str, ...]) -> pd.DataFrame:
    columns = [column for column in _ID_COLUMNS if column in frame.columns]
    columns.extend(
        column
        for column in frame.columns
        if any(str(column).startswith(prefix) for prefix in prefixes)
        and column not in columns
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )
    return frame.loc[:, columns].copy()


def _compute_feature_redundancy(he: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    he_columns = _numeric_columns(he, exclude=set(_ID_COLUMNS))
    native_columns = _numeric_columns(native, exclude=set(_ID_COLUMNS))
    if not he_columns or not native_columns:
        return pd.DataFrame(columns=["he_feature", "xenium_native_feature", "spearman_rho", "abs_spearman_rho", "p_value"])
    merged = he.merge(native, on=[column for column in _ID_COLUMNS if column in he.columns and column in native.columns], how="inner")
    rows = []
    for he_column in he_columns:
        for native_column in native_columns:
            rho, p_value = _safe_spearman(merged[he_column], merged[native_column])
            if np.isfinite(rho):
                rows.append(
                    {
                        "he_feature": he_column,
                        "xenium_native_feature": native_column,
                        "spearman_rho": rho,
                        "abs_spearman_rho": abs(rho),
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("abs_spearman_rho", ascending=False, kind="stable").reset_index(drop=True)
    return result


def _compute_incremental_prediction(
    *,
    contour_features: pd.DataFrame,
    program_scores: pd.DataFrame,
    he_morphology: pd.DataFrame,
    xenium_native: pd.DataFrame,
    random_state: int,
    min_contours: int,
) -> pd.DataFrame:
    id_cols = [column for column in _ID_COLUMNS if column in contour_features.columns]
    merge_keys = [column for column in id_cols if column in program_scores.columns]
    merged = contour_features.loc[:, id_cols + _baseline_columns(contour_features)].merge(program_scores, on=merge_keys, how="inner")
    merged = merged.merge(he_morphology, on=[column for column in id_cols if column in he_morphology.columns], how="left")
    merged = merged.merge(xenium_native, on=[column for column in id_cols if column in xenium_native.columns], how="left")

    baseline = _baseline_columns(merged)
    he_columns = _numeric_columns(he_morphology, exclude=set(_ID_COLUMNS))
    native_columns = _numeric_columns(xenium_native, exclude=set(_ID_COLUMNS))
    outcome_columns = _program_outcome_columns(program_scores)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for outcome in outcome_columns:
        model_specs = {
            "baseline": baseline,
            "baseline_plus_xenium_native": baseline + native_columns,
            "baseline_plus_he": baseline + he_columns,
            "baseline_plus_both": baseline + native_columns + he_columns,
        }
        baseline_r2 = np.nan
        for model_name, columns in model_specs.items():
            r2, evaluation = _evaluate_ridge_r2(merged, columns=columns, outcome=outcome, random_state=random_state, min_contours=min_contours)
            if model_name == "baseline":
                baseline_r2 = r2
            rows.append(
                {
                    "outcome": outcome,
                    "model": model_name,
                    "n_contours": int(pd.to_numeric(merged[outcome], errors="coerce").notna().sum()),
                    "n_features": int(len(columns)),
                    "r2": r2,
                    "r2_delta_vs_baseline": float(r2 - baseline_r2) if np.isfinite(r2) and np.isfinite(baseline_r2) else np.nan,
                    "evaluation": evaluation,
                    "shuffled": False,
                }
            )
        if he_columns:
            shuffled = merged.copy()
            permutation = rng.permutation(len(shuffled))
            shuffled.loc[:, he_columns] = shuffled.iloc[permutation][he_columns].to_numpy()
            columns = baseline + he_columns
            r2, evaluation = _evaluate_ridge_r2(shuffled, columns=columns, outcome=outcome, random_state=random_state, min_contours=min_contours)
            rows.append(
                {
                    "outcome": outcome,
                    "model": "baseline_plus_he_shuffle",
                    "n_contours": int(pd.to_numeric(shuffled[outcome], errors="coerce").notna().sum()),
                    "n_features": int(len(columns)),
                    "r2": r2,
                    "r2_delta_vs_baseline": float(r2 - baseline_r2) if np.isfinite(r2) and np.isfinite(baseline_r2) else np.nan,
                    "evaluation": evaluation,
                    "shuffled": True,
                }
            )
    return pd.DataFrame(rows)


def _compute_partial_associations(
    *,
    feature_table: dict[str, Any],
    contour_features: pd.DataFrame,
    program_scores: pd.DataFrame,
    he_morphology: pd.DataFrame,
    xenium_native: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = [column for column in _ID_COLUMNS if column in contour_features.columns]
    he_columns = _numeric_columns(he_morphology, exclude=set(_ID_COLUMNS))
    if not he_columns:
        return pd.DataFrame(columns=["he_feature", "outcome", "outcome_kind", "partial_spearman_rho", "p_value", "fdr"])
    outcome_table = _build_outcome_table(feature_table, program_scores)
    merged = contour_features.loc[:, id_cols + _baseline_columns(contour_features)].merge(outcome_table, on="contour_id", how="inner")
    merged = merged.merge(he_morphology, on=[column for column in id_cols if column in he_morphology.columns], how="left")
    merged = merged.merge(xenium_native, on=[column for column in id_cols if column in xenium_native.columns], how="left")
    controls = _baseline_columns(merged) + _numeric_columns(xenium_native, exclude=set(_ID_COLUMNS))

    rows: list[dict[str, Any]] = []
    for he_column in he_columns:
        x_resid = _residualize(merged[he_column], merged.loc[:, [column for column in controls if column in merged.columns]])
        for outcome in [column for column in outcome_table.columns if column != "contour_id"]:
            y_resid = _residualize(merged[outcome], merged.loc[:, [column for column in controls if column in merged.columns]])
            rho, p_value = _safe_spearman(pd.Series(x_resid), pd.Series(y_resid))
            if np.isfinite(rho):
                rows.append(
                    {
                        "he_feature": he_column,
                        "outcome": outcome,
                        "outcome_kind": str(outcome).split("__", 1)[0],
                        "partial_spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = _benjamini_hochberg(result["p_value"])
        result = result.sort_values(["fdr", "partial_spearman_rho"], ascending=[True, False], kind="stable").reset_index(drop=True)
    return result


def _build_matched_review_table(
    *,
    contour_features: pd.DataFrame,
    program_scores: pd.DataFrame,
    he_morphology: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    merged = contour_features.merge(program_scores, on=[column for column in _ID_COLUMNS if column in contour_features.columns and column in program_scores.columns], how="inner")
    he_columns = _numeric_columns(he_morphology, exclude=set(_ID_COLUMNS))
    outcomes = _program_outcome_columns(program_scores)
    baseline = _baseline_columns(merged)
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        ordered = merged.sort_values(outcome, ascending=False, kind="stable").head(min(top_n, len(merged)))
        control_pool = merged.loc[pd.to_numeric(merged[outcome], errors="coerce") <= pd.to_numeric(merged[outcome], errors="coerce").median()].copy()
        if control_pool.empty:
            control_pool = merged.sort_values(outcome, ascending=True, kind="stable").head(max(1, top_n)).copy()
        for _, exemplar in ordered.iterrows():
            candidates = control_pool.loc[control_pool["contour_id"].astype(str) != str(exemplar["contour_id"])].copy()
            if candidates.empty:
                continue
            candidates["match_distance"] = _row_distance(candidates, exemplar, columns=baseline)
            control = candidates.sort_values("match_distance", kind="stable").iloc[0]
            row = {
                "outcome": outcome,
                "exemplar_id": exemplar["contour_id"],
                "control_id": control["contour_id"],
                "exemplar_score": float(exemplar[outcome]),
                "control_score": float(control[outcome]),
                "delta_score": float(exemplar[outcome] - control[outcome]),
                "match_distance": float(control["match_distance"]),
            }
            if he_columns:
                deltas = {
                    column: abs(float(exemplar.get(column, np.nan)) - float(control.get(column, np.nan)))
                    for column in he_columns
                    if np.isfinite(float(exemplar.get(column, np.nan))) and np.isfinite(float(control.get(column, np.nan)))
                }
                if deltas:
                    top_feature = max(deltas, key=deltas.get)
                    row["top_he_delta_feature"] = top_feature
                    row["top_he_delta_abs"] = float(deltas[top_feature])
            rows.append(row)
    return pd.DataFrame(rows)


def _write_morphology_increment_artifacts(result: dict[str, Any], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, filename in {
        "xenium_native_morphology": "xenium_native_morphology.csv",
        "he_morphology_features": "he_morphology_features.csv",
        "feature_redundancy": "feature_redundancy.csv",
        "incremental_prediction": "incremental_prediction.csv",
        "partial_associations": "partial_associations.csv",
        "matched_review_table": "matched_review_table.csv",
    }.items():
        result[key].to_csv(out / filename, index=False)
    (out / "morphology_increment_summary.json").write_text(
        json.dumps(result["summary"], indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def _identifier_row(contour_features: pd.DataFrame, contour_id: str, *, contour_key: str) -> dict[str, Any]:
    match = contour_features.loc[contour_features["contour_id"].astype(str) == str(contour_id)]
    if match.empty:
        return {"sample_id": "sample_0", "contour_key": contour_key, "contour_id": str(contour_id)}
    row = match.iloc[0]
    return {column: row[column] for column in _ID_COLUMNS if column in row.index}


def _outer_minus_inner(row: dict[str, Any], *, prefix: str) -> dict[str, float]:
    output = {}
    for key, value in row.items():
        text = str(key)
        parts = text.split("__")
        if len(parts) < 3 or parts[0] != prefix or parts[1] != "inner_rim":
            continue
        suffix = "__".join(parts[2:])
        outer = f"{prefix}__outer_rim__{suffix}"
        if outer in row:
            output[f"{prefix}__outer_minus_inner__{suffix}"] = float(row[outer]) - float(value)
    return output


def _numeric_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in exclude:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() >= 1 and float(np.nanstd(series.to_numpy(dtype=float))) > 1e-12:
            columns.append(str(column))
    return columns


def _numeric_columns_any(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in exclude:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() >= 1:
            columns.append(str(column))
    return columns


def _baseline_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "geometry__area_um2",
        "geometry__perimeter_um",
        "geometry__compactness",
        "geometry__eccentricity",
        "context__centroid_x_um",
        "context__centroid_y_um",
        "context__neighbor_count",
        "omics__whole__n_cells",
    ]
    return [column for column in candidates if column in frame.columns]


def _program_outcome_columns(program_scores: pd.DataFrame) -> list[str]:
    exclude = set(_ID_COLUMNS) | {"top_program", "top_program_score"}
    return [
        column
        for column in program_scores.columns
        if column not in exclude
        and not str(column).endswith("_evidence")
        and pd.to_numeric(program_scores[column], errors="coerce").notna().sum() >= 2
    ]


def _evaluate_ridge_r2(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    outcome: str,
    random_state: int,
    min_contours: int,
) -> tuple[float, str]:
    columns = [column for column in dict.fromkeys(columns) if column in frame.columns]
    y = pd.to_numeric(frame[outcome], errors="coerce")
    X = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce") if columns else pd.DataFrame(index=frame.index)
    mask = y.notna()
    if not X.empty:
        mask &= X.notna().any(axis=1)
    y = y.loc[mask].to_numpy(dtype=float)
    X = X.loc[mask]
    if y.size < 3 or np.nanstd(y) < 1e-12:
        return float("nan"), "insufficient_variation"
    if X.empty:
        return 0.0, "mean_only"
    X = _fill_frame(X)
    X = X.loc[:, [column for column in X.columns if float(np.nanstd(X[column].to_numpy(dtype=float))) > 1e-12]]
    if X.empty:
        return 0.0, "mean_only"
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if len(y) >= int(min_contours):
        splits = min(5, len(y))
        cv = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X.to_numpy(dtype=float), y, cv=cv, scoring="r2")
        return float(np.nanmean(scores)), "cross_validated"
    model.fit(X.to_numpy(dtype=float), y)
    return float(r2_score(y, model.predict(X.to_numpy(dtype=float)))), "in_sample_small_n"


def _build_outcome_table(feature_table: dict[str, Any], program_scores: pd.DataFrame) -> pd.DataFrame:
    tables = []
    program_columns = ["contour_id", *_program_outcome_columns(program_scores)]
    tables.append(program_scores.loc[:, [column for column in program_columns if column in program_scores.columns]].copy())
    for key, prefix in (("rna_pseudobulk", "rna"), ("pathway_activity", "pathway")):
        frame = pd.DataFrame(feature_table.get(key, pd.DataFrame())).copy()
        if frame.empty or "contour_id" not in frame.columns:
            continue
        renamed = {"contour_id": "contour_id"}
        for column in frame.columns:
            if column in _ID_COLUMNS:
                continue
            renamed[column] = f"{prefix}__{column}"
        tables.append(frame.loc[:, list(renamed)].rename(columns=renamed))
    output = tables[0]
    for table in tables[1:]:
        output = output.merge(table, on="contour_id", how="left")
    return output


def _residualize(values: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y)
    if controls.empty or finite.sum() < 3:
        residual = y - np.nanmean(y[finite]) if finite.any() else np.full_like(y, np.nan)
        return residual
    X = controls.apply(pd.to_numeric, errors="coerce")
    X = _fill_frame(X)
    X = X.loc[:, [column for column in X.columns if float(np.nanstd(X[column].to_numpy(dtype=float))) > 1e-12]]
    if X.empty:
        return y - np.nanmean(y[finite])
    mask = finite & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    residual = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < 3:
        residual[finite] = y[finite] - np.nanmean(y[finite])
        return residual
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X.loc[mask].to_numpy(dtype=float), y[mask])
    residual[mask] = y[mask] - model.predict(X.loc[mask].to_numpy(dtype=float))
    return residual


def _row_distance(frame: pd.DataFrame, target: pd.Series, *, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=frame.index)
    distance = pd.Series(0.0, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        scale = float(np.nanstd(values.to_numpy(dtype=float)))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        distance += (values - float(target[column])).abs() / scale
    return distance


def _fill_frame(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.apply(pd.to_numeric, errors="coerce").copy()
    for column in filled.columns:
        values = filled[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
        filled[column] = np.where(finite, values, fill)
    return filled


def _safe_spearman(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    if np.nanstd(left_values[mask]) < 1e-12 or np.nanstd(right_values[mask]) < 1e-12:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(left_values[mask], right_values[mask])
    return float(rho), float(p_value)


def _benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return output
    finite_values = values[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for reverse_index in range(n - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, ranked[reverse_index] * n / rank)
        adjusted[reverse_index] = running
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    output[finite_mask] = np.clip(restored, 0.0, 1.0)
    return output
