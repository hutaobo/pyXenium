from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from pyXenium.contour import add_contours_from_geojson
from pyXenium.io import read_xenium
from pyXenium.validation.atera_wta_breast_topology import (
    DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS,
    DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
    DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID,
)

from ._dataset import (
    build_contour_gmi_dataset,
    build_within_label_heterogeneity_dataset,
    compute_contour_heterogeneity,
    shuffle_spatial_feature_block,
    write_contour_gmi_dataset,
)
from ._runner import read_gmi_outputs, run_gmi_fit
from ._types import ContourGmiConfig, ContourGmiDataset, ContourGmiResult


DEFAULT_ATERA_CONTOUR_GEOJSON = "xenium_explorer_annotations.s1_s5.generated.geojson"


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    if y_true.nunique() != 2:
        return None
    try:
        return float(roc_auc_score(y_true.astype(int), y_score.astype(float)))
    except Exception:
        return None


def _classification_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, Any]:
    aligned = pd.DataFrame({"y": y_true.astype(int), "score": pd.to_numeric(y_score, errors="coerce")}).dropna()
    if aligned.empty:
        return {"auc": None, "accuracy": None, "sensitivity": None, "specificity": None}
    pred = (aligned["score"] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(aligned["y"], pred, labels=[0, 1]).ravel()
    return {
        "auc": _safe_auc(aligned["y"], aligned["score"]),
        "accuracy": float(accuracy_score(aligned["y"], pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
    }


def _assign_groups(main_effects: pd.DataFrame, interaction_effects: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def emit_groups(table: pd.DataFrame, effect_type: str, label_col: str) -> None:
        if table.empty or "coefficient" not in table.columns:
            return
        ordered = table.copy().sort_values("coefficient", kind="mergesort").reset_index(drop=True)
        group_id = 0
        last_value: float | None = None
        for _, row in ordered.iterrows():
            value = float(row["coefficient"])
            if last_value is None or abs(value - last_value) > threshold:
                group_id += 1
                last_value = value
            rows.append(
                {
                    "effect_type": effect_type,
                    "group_id": f"{effect_type}_{group_id}",
                    "member": str(row[label_col]),
                    "coefficient": value,
                }
            )

    emit_groups(main_effects, "main", "feature")
    if not interaction_effects.empty:
        interaction_effects = interaction_effects.copy()
        interaction_effects["interaction_label"] = (
            interaction_effects["feature_a"].astype(str) + "^" + interaction_effects["feature_b"].astype(str)
        )
        emit_groups(interaction_effects, "interaction", "interaction_label")
    return pd.DataFrame(rows, columns=["effect_type", "group_id", "member", "coefficient"])


def _write_prediction_matrix(X: pd.DataFrame, output_dir: Path, name: str = "prediction_matrix.tsv.gz") -> Path:
    path = output_dir / name
    X.to_csv(path, sep="\t", compression="gzip", index_label="sample_id")
    return path


def _run_single_fit(
    dataset: ContourGmiDataset,
    *,
    output_dir: Path,
    config: ContourGmiConfig,
    prediction_X: pd.DataFrame | None = None,
) -> dict[str, Any]:
    files = write_contour_gmi_dataset(dataset, output_dir)
    prediction_path = None
    if prediction_X is not None:
        prediction_path = _write_prediction_matrix(prediction_X, output_dir)
        files["prediction_matrix"] = str(prediction_path)
    files.update(
        run_gmi_fit(
            design_matrix_path=files["design_matrix"],
            sample_metadata_path=files["sample_metadata"],
            output_dir=output_dir,
            config=config,
            prediction_matrix_path=prediction_path,
        )
    )
    parsed = read_gmi_outputs(output_dir)
    return {"files": files, **parsed}


def _cv_metric_columns() -> list[str]:
    return [
        "fold",
        "n_train",
        "n_test",
        "auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "n_train_positive",
        "n_train_negative",
        "n_test_positive",
        "n_test_negative",
        "requested_folds",
        "effective_folds",
        "skip_reason",
    ]


def _spatial_fold_ids(
    sample_metadata: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
) -> tuple[pd.Series, int, str | None]:
    requested = int(n_folds)
    if requested <= 1:
        return pd.Series(dtype="Int64"), 0, "spatial_cv_disabled"
    if len(sample_metadata) < requested:
        requested = len(sample_metadata)
    y = y.astype(int)
    if y.nunique() != 2:
        return pd.Series(dtype="Int64"), 0, "cv_requires_two_classes"
    class_counts = y.value_counts()
    effective = min(requested, int(class_counts.min()))
    if effective <= 1:
        return pd.Series(dtype="Int64"), 0, "too_few_contours_per_class_for_cv"

    metadata = sample_metadata.set_index("sample_id", drop=False).loc[y.index].copy()
    folds = pd.Series(index=metadata.index.astype(str), data=pd.NA, dtype="Int64", name="fold")
    for class_value in sorted(y.unique()):
        class_ids = y.index[y == class_value].astype(str)
        ordered = metadata.loc[class_ids].sort_values(["x_centroid", "y_centroid"])["sample_id"].astype(str).to_numpy()
        for fold_index, chunk in enumerate(np.array_split(ordered, effective)):
            if len(chunk):
                folds.loc[chunk.astype(str).tolist()] = int(fold_index)

    reason = None if effective == int(n_folds) else f"requested_{int(n_folds)}_reduced_to_{effective}_for_class_balance"
    return folds, effective, reason


def _run_spatial_cv(dataset: ContourGmiDataset, output_dir: Path, config: ContourGmiConfig) -> pd.DataFrame:
    if config.spatial_cv_folds <= 1:
        return pd.DataFrame(columns=_cv_metric_columns())
    metadata = dataset.sample_metadata.set_index("sample_id", drop=False).loc[dataset.X.index]
    folds, effective_folds, fold_reason = _spatial_fold_ids(metadata, dataset.y, config.spatial_cv_folds)
    if folds.empty or effective_folds <= 1:
        return pd.DataFrame(
            [
                {
                    "fold": pd.NA,
                    "n_train": int(dataset.X.shape[0]),
                    "n_test": 0,
                    "auc": None,
                    "accuracy": None,
                    "sensitivity": None,
                    "specificity": None,
                    "n_train_positive": int((dataset.y == 1).sum()),
                    "n_train_negative": int((dataset.y == 0).sum()),
                    "n_test_positive": 0,
                    "n_test_negative": 0,
                    "requested_folds": int(config.spatial_cv_folds),
                    "effective_folds": int(effective_folds),
                    "skip_reason": fold_reason,
                }
            ],
            columns=_cv_metric_columns(),
        )
    rows: list[dict[str, Any]] = []
    fold_array = folds.astype("float").to_numpy()
    for fold in sorted(folds.dropna().unique()):
        test_ids = folds.index[fold_array == int(fold)].astype(str).tolist()
        train_ids = folds.index[fold_array != int(fold)].astype(str).tolist()
        n_train_positive = int((dataset.y.loc[train_ids] == 1).sum())
        n_train_negative = int((dataset.y.loc[train_ids] == 0).sum())
        n_test_positive = int((dataset.y.loc[test_ids] == 1).sum())
        n_test_negative = int((dataset.y.loc[test_ids] == 0).sum())
        skip_reason = None
        if not test_ids:
            skip_reason = "empty_test_fold"
        elif dataset.y.loc[train_ids].nunique() != 2:
            skip_reason = "train_fold_lacks_both_classes"
        elif dataset.y.loc[test_ids].nunique() != 2:
            skip_reason = "test_fold_lacks_both_classes"
        if skip_reason is not None:
            rows.append(
                {
                    "fold": int(fold) + 1,
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                    "auc": None,
                    "accuracy": None,
                    "sensitivity": None,
                    "specificity": None,
                    "n_train_positive": n_train_positive,
                    "n_train_negative": n_train_negative,
                    "n_test_positive": n_test_positive,
                    "n_test_negative": n_test_negative,
                    "requested_folds": int(config.spatial_cv_folds),
                    "effective_folds": int(effective_folds),
                    "skip_reason": skip_reason,
                }
            )
            continue
        fold_dir = output_dir / "cv" / f"fold_{int(fold) + 1:02d}"
        result = _run_single_fit(dataset.subset(train_ids), output_dir=fold_dir, config=config, prediction_X=dataset.X.loc[test_ids])
        preds = result.get("predictions_test", pd.DataFrame())
        scores = preds.set_index("sample_id")["prediction"] if not preds.empty else pd.Series(dtype=float)
        metrics = _classification_metrics(dataset.y.loc[scores.index], scores)
        rows.append(
            {
                "fold": int(fold) + 1,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                **metrics,
                "n_train_positive": n_train_positive,
                "n_train_negative": n_train_negative,
                "n_test_positive": n_test_positive,
                "n_test_negative": n_test_negative,
                "requested_folds": int(config.spatial_cv_folds),
                "effective_folds": int(effective_folds),
                "skip_reason": fold_reason,
            }
        )
    return pd.DataFrame(rows, columns=_cv_metric_columns())


def _run_bootstrap_stability(dataset: ContourGmiDataset, output_dir: Path, config: ContourGmiConfig) -> pd.DataFrame:
    if config.bootstrap_repeats <= 0:
        return pd.DataFrame(columns=["effect_type", "member", "selection_count", "selection_frequency"])
    rng = np.random.default_rng(config.random_seed)
    sample_ids = dataset.X.index.astype(str).to_numpy()
    take = max(2, int(round(len(sample_ids) * float(config.bootstrap_fraction))))
    main_counter: Counter[str] = Counter()
    inter_counter: Counter[str] = Counter()
    completed = 0
    for repeat in range(int(config.bootstrap_repeats)):
        chosen = sorted(rng.choice(sample_ids, size=min(take, len(sample_ids)), replace=False).astype(str).tolist())
        if dataset.y.loc[chosen].nunique() != 2:
            continue
        repeat_dir = output_dir / "bootstrap" / f"repeat_{repeat + 1:03d}"
        result = _run_single_fit(dataset.subset(chosen), output_dir=repeat_dir, config=config)
        main = result.get("main_effects", pd.DataFrame())
        inter = result.get("interaction_effects", pd.DataFrame())
        main_counter.update(main.get("feature", pd.Series(dtype=str)).astype(str).tolist())
        if not inter.empty:
            labels = inter["feature_a"].astype(str) + "^" + inter["feature_b"].astype(str)
            inter_counter.update(labels.tolist())
        completed += 1
    rows = []
    for effect_type, counter in (("main", main_counter), ("interaction", inter_counter)):
        for member, count in counter.most_common():
            rows.append(
                {
                    "effect_type": effect_type,
                    "member": member,
                    "selection_count": int(count),
                    "selection_frequency": float(count / completed) if completed else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=["effect_type", "member", "selection_count", "selection_frequency"])


def render_contour_gmi_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Contour GMI Report",
        "",
        f"Sample: `{summary.get('sample_id', 'unknown')}`",
        f"Endpoint: `{summary.get('endpoint', 'S1_vs_S5')}`",
        f"Output directory: `{summary.get('output_dir')}`",
        "",
        "## Dataset",
        "",
        f"- Endpoint contours in metadata: `{summary.get('n_total_endpoint_contours')}`",
        f"- Contours retained: `{summary.get('n_contours')}`",
        f"- Contours dropped by QC: `{summary.get('n_dropped_contours')}`",
        f"- Features used: `{summary.get('n_features')}`",
        f"- RNA features: `{summary.get('n_rna_features')}`",
        f"- Spatial features: `{summary.get('n_spatial_features')}`",
        f"- Positive contours: `{summary.get('n_positive_contours')}`",
        f"- Negative contours: `{summary.get('n_negative_contours')}`",
        f"- Feature budget rationale: {summary.get('feature_budget_rationale', 'not recorded')}",
        "",
        "## GMI Fit",
        "",
        f"- Selected main effects: `{summary.get('selected_main_effects')}`",
        f"- Selected interactions: `{summary.get('selected_interactions')}`",
        f"- Training AUC: `{summary.get('train_metrics', {}).get('auc')}`",
        "",
        "## Stability And Heterogeneity",
        "",
        f"- Spatial CV folds completed: `{summary.get('cv_folds_completed')}`",
        f"- Spatial CV effective folds: `{summary.get('cv_effective_folds')}`",
        f"- Bootstrap repeats requested: `{summary.get('bootstrap_repeats_requested')}`",
        f"- Within-label GMI runs: `{summary.get('within_label_runs_completed')}`",
        "",
        "## Biological Readout",
        "",
        "Interpret selected main effects and interactions against S1 invasive tumor/CAF biology versus S5 apocrine-luminal DCIS, including CAF/ECM remodeling, angiogenesis/pericyte axes, myeloid-vascular context, Notch, IGF/IGF1R/MAPK, Wnt, TGF-beta, CXCL12/CXCR4, and CSF1/CSF1R programs.",
        "",
    ]
    if summary.get("top_main_effects"):
        lines.append("Top main effects:")
        for item in summary["top_main_effects"][:10]:
            lines.append(f"- `{item.get('feature')}` (`coef={item.get('coefficient')}`)")
        lines.append("")
    if summary.get("top_interactions"):
        lines.append("Top interactions:")
        for item in summary["top_interactions"][:10]:
            lines.append(f"- `{item.get('feature_a')} ^ {item.get('feature_b')}` (`coef={item.get('coefficient')}`)")
        lines.append("")
    if summary.get("within_label_runs"):
        lines.append("Within-label heterogeneity runs:")
        for item in summary["within_label_runs"]:
            lines.append(f"- `{item.get('label')}`: `{item.get('status')}`")
        lines.append("")
    return "\n".join(lines)


def _write_control_failure(
    *,
    output_dir: Path,
    control: str,
    error: Exception,
    config: ContourGmiConfig,
    provenance: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "sample_id": provenance.get("sample_id", "contour_gmi"),
        "endpoint": f"{config.positive_label}_vs_{config.negative_label}",
        "output_dir": str(output_dir),
        "control": control,
        "status": "not_fit",
        "reason": str(error),
        "interpretation": "The control did not yield both endpoint contour classes after filtering.",
        "config": config.to_dict(),
        "provenance": dict(provenance),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_contour_gmi_report(summary), encoding="utf-8")


def _write_within_label_skip(output_dir: Path, label: str, reason: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": str(label), "status": "skipped", "reason": reason, "output_dir": str(output_dir)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join(["# Within-Label Contour GMI", "", f"Label: `{label}`", "Status: `skipped`", "", reason, ""]),
        encoding="utf-8",
    )
    return summary


def _run_within_label_heterogeneity(
    dataset: ContourGmiDataset,
    heterogeneity: pd.DataFrame,
    output_dir: Path,
    config: ContourGmiConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if heterogeneity.empty or not config.run_within_label_heterogeneity:
        return rows
    for label in (config.positive_label, config.negative_label):
        label_dir = output_dir / "within_label" / str(label)
        subdataset = build_within_label_heterogeneity_dataset(dataset, heterogeneity, label=str(label))
        if subdataset is None:
            rows.append(_write_within_label_skip(label_dir, str(label), "Not enough high/low heterogeneity contours."))
            continue
        subconfig = ContourGmiConfig(
            **{k: v for k, v in subdataset.config.items() if k in ContourGmiConfig.__dataclass_fields__}
        )
        result = run_contour_gmi(subdataset, output_dir=label_dir, config=subconfig)
        rows.append(
            {
                "label": str(label),
                "status": "completed",
                "n_contours": int(subdataset.X.shape[0]),
                "selected_main_effects": int(result.summary.get("selected_main_effects", 0)),
                "selected_interactions": int(result.summary.get("selected_interactions", 0)),
                "output_dir": str(label_dir),
            }
        )
    return rows


def _load_geometry(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        from shapely import wkt

        return wkt.loads(value)
    except Exception:
        return None


def _polygon_exteriors(geometry: Any) -> list[np.ndarray]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if geometry.geom_type == "Polygon":
        return [np.asarray(geometry.exterior.coords, dtype=float)]
    if geometry.geom_type == "MultiPolygon":
        return [np.asarray(part.exterior.coords, dtype=float) for part in geometry.geoms]
    return []


def _plot_contour_values(
    metadata: pd.DataFrame,
    values: pd.Series,
    *,
    output_path: Path,
    title: str,
    cmap: str = "viridis",
    categorical: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch as LegendPatch
    from matplotlib.patches import Polygon as MplPolygon

    frame = metadata.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    values = values.reindex(frame["sample_id"].astype(str))
    fig, ax = plt.subplots(figsize=(7, 7))
    has_polygons = "geometry_wkt" in frame.columns and frame["geometry_wkt"].notna().any()

    if has_polygons and categorical:
        palette = {
            "S1": "#2563eb",
            "S5": "#dc2626",
            "retained": "#16a34a",
            "too_few_cells": "#94a3b8",
            "empty_library": "#64748b",
            "non_endpoint_label": "#a855f7",
        }
        categories = pd.Index(values.fillna("missing").astype(str).unique()).sort_values()
        for category in categories:
            patches: list[MplPolygon] = []
            for (_, row), value in zip(frame.iterrows(), values.fillna("missing").astype(str)):
                if value != category:
                    continue
                for exterior in _polygon_exteriors(_load_geometry(row.get("geometry_wkt"))):
                    patches.append(MplPolygon(exterior, closed=True))
            if patches:
                collection = PatchCollection(
                    patches,
                    facecolor=palette.get(str(category), "#f59e0b"),
                    edgecolor="#1f2937",
                    linewidth=0.35,
                    alpha=0.72,
                    label=str(category),
                )
                ax.add_collection(collection)
        handles = [
            LegendPatch(facecolor=palette.get(str(category), "#f59e0b"), edgecolor="#1f2937", label=str(category))
            for category in categories
        ]
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    elif has_polygons:
        finite_patches: list[MplPolygon] = []
        finite_values: list[float] = []
        missing_patches: list[MplPolygon] = []
        numeric_values = pd.to_numeric(values, errors="coerce")
        for (_, row), value in zip(frame.iterrows(), numeric_values):
            polygons = [
                MplPolygon(exterior, closed=True)
                for exterior in _polygon_exteriors(_load_geometry(row.get("geometry_wkt")))
            ]
            if not polygons:
                continue
            if pd.isna(value):
                missing_patches.extend(polygons)
            else:
                finite_patches.extend(polygons)
                finite_values.extend([float(value)] * len(polygons))
        if missing_patches:
            ax.add_collection(
                PatchCollection(
                    missing_patches,
                    facecolor="#e5e7eb",
                    edgecolor="#9ca3af",
                    linewidth=0.25,
                    alpha=0.55,
                )
            )
        if finite_patches:
            collection = PatchCollection(
                finite_patches,
                cmap=cmap,
                edgecolor="#111827",
                linewidth=0.25,
                alpha=0.82,
            )
            collection.set_array(np.asarray(finite_values, dtype=float))
            ax.add_collection(collection)
            fig.colorbar(collection, ax=ax, shrink=0.72)
    elif categorical:
        scatter_frame = frame.copy()
        scatter_frame["_category"] = values.fillna("missing").astype(str).to_numpy()
        for category, group in scatter_frame.groupby("_category", sort=True):
            ax.scatter(group["x_centroid"], group["y_centroid"], s=18, label=str(category), alpha=0.8)
        ax.legend(loc="best", fontsize=8, frameon=False)
    else:
        scatter = ax.scatter(
            frame["x_centroid"],
            frame["y_centroid"],
            c=pd.to_numeric(values, errors="coerce"),
            s=20,
            cmap=cmap,
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.72)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x centroid (um)")
    ax.set_ylabel("y centroid (um)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.margins(0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _rna_logcpm_from_design(dataset: ContourGmiDataset, feature: str) -> pd.Series:
    matches = dataset.feature_metadata.loc[
        (dataset.feature_metadata["feature_block"].astype(str) == "rna")
        & (dataset.feature_metadata["feature"].astype(str) == str(feature))
    ]
    if matches.empty or str(feature) not in dataset.X.columns:
        return pd.Series(dtype=float)
    row = matches.iloc[0]
    mean = float(row.get("standardization_mean", 0.0))
    std = float(row.get("standardization_std", 1.0))
    return dataset.X[str(feature)].astype(float) * std + mean


def _write_spatial_visualizations(
    dataset: ContourGmiDataset,
    output_dir: Path,
    predictions: pd.DataFrame,
    config: ContourGmiConfig,
) -> dict[str, str]:
    if not config.write_spatial_visualizations:
        return {}
    metadata = dataset.sample_metadata.copy()
    if metadata.empty or not {"sample_id", "x_centroid", "y_centroid"}.issubset(metadata.columns):
        return {}
    figure_dir = output_dir / "figures"
    manifest: list[dict[str, str]] = []
    files: dict[str, str] = {}

    def add_figure(key: str, path: Path, description: str) -> None:
        files[f"figure_{key}"] = str(path)
        manifest.append({"name": key, "path": str(path), "description": description})

    overlay_path = figure_dir / "s1_s5_contour_overlay.png"
    _plot_contour_values(
        metadata,
        metadata.set_index("sample_id")["label"].astype(str),
        output_path=overlay_path,
        title="S1/S5 contour labels",
        categorical=True,
    )
    add_figure("s1_s5_overlay", overlay_path, "S1/S5 endpoint contour labels.")

    qc_path = figure_dir / "qc_retained_vs_dropped.png"
    qc_values = metadata.set_index("sample_id").apply(
        lambda row: "retained" if bool(row.get("retained", False)) else str(row.get("drop_reason", "dropped")),
        axis=1,
    )
    _plot_contour_values(
        metadata,
        qc_values,
        output_path=qc_path,
        title="Contour QC retained/dropped",
        categorical=True,
    )
    add_figure("qc_retained_vs_dropped", qc_path, "Retained contours and QC drop reasons.")

    if not predictions.empty and {"sample_id", "prediction"}.issubset(predictions.columns):
        pred_path = figure_dir / "gmi_prediction_score.png"
        prediction_values = predictions.set_index("sample_id")["prediction"].astype(float)
        _plot_contour_values(
            metadata,
            prediction_values,
            output_path=pred_path,
            title="GMI prediction score",
            cmap="magma",
        )
        add_figure("gmi_prediction_score", pred_path, "GMI train-set prediction score by contour.")

    for gene in tuple(config.visualization_genes):
        values = _rna_logcpm_from_design(dataset, str(gene))
        if values.empty:
            continue
        gene_path = figure_dir / f"gene_{str(gene).replace('/', '_')}_logcpm.png"
        _plot_contour_values(
            metadata,
            values,
            output_path=gene_path,
            title=f"{gene} contour logCPM",
            cmap="viridis",
        )
        add_figure(f"gene_{gene}_logcpm", gene_path, f"{gene} contour-level pseudo-bulk logCPM.")

    if manifest:
        manifest_path = figure_dir / "visualization_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files["visualization_manifest"] = str(manifest_path)
    return files


def _build_summary(
    dataset: ContourGmiDataset,
    output_dir: Path,
    main_effects: pd.DataFrame,
    interaction_effects: pd.DataFrame,
    predictions: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    files: dict[str, str],
    within_label_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    y_by_sample = dataset.y
    if predictions.empty:
        train_metrics = _classification_metrics(pd.Series(dtype=int), pd.Series(dtype=float))
    else:
        scores = predictions.set_index("sample_id")["prediction"]
        train_metrics = _classification_metrics(y_by_sample.loc[scores.index], scores)
    feature_blocks = dataset.feature_metadata.get("feature_block", pd.Series(dtype=str)).astype(str)
    config = ContourGmiConfig(**{k: v for k, v in dataset.config.items() if k in ContourGmiConfig.__dataclass_fields__})
    sample_metadata = dataset.sample_metadata.copy()
    total_endpoint_contours = int(len(sample_metadata))
    retained_count = int(sample_metadata.get("retained", pd.Series(dtype=bool)).astype(bool).sum())
    drop_reasons = (
        sample_metadata.get("drop_reason", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        if "drop_reason" in sample_metadata.columns
        else {}
    )
    cv_effective = None
    if not cv_metrics.empty and "effective_folds" in cv_metrics.columns:
        valid_effective = pd.to_numeric(cv_metrics["effective_folds"], errors="coerce").dropna()
        cv_effective = int(valid_effective.max()) if not valid_effective.empty else None
    summary = {
        "sample_id": dataset.provenance.get("sample_id", "contour_gmi"),
        "endpoint": dataset.provenance.get("endpoint", f"{config.positive_label}_vs_{config.negative_label}"),
        "output_dir": str(output_dir),
        "contour_key": config.contour_key,
        "positive_label": config.positive_label,
        "negative_label": config.negative_label,
        "n_total_endpoint_contours": total_endpoint_contours,
        "n_contours": int(dataset.X.shape[0]),
        "n_retained_contours": retained_count,
        "n_dropped_contours": int(total_endpoint_contours - retained_count),
        "drop_reasons": drop_reasons,
        "n_features": int(dataset.X.shape[1]),
        "n_rna_features": int((feature_blocks == "rna").sum()),
        "n_spatial_features": int((feature_blocks == "spatial").sum()),
        "feature_budget_rationale": (
            "GMI runs on contour pseudo-bulk samples; feature counts are capped so p does not greatly exceed "
            "the retained contour count and selected effects remain interpretable."
        ),
        "n_positive_contours": int((dataset.y == 1).sum()),
        "n_negative_contours": int((dataset.y == 0).sum()),
        "selected_main_effects": int(len(main_effects)),
        "selected_interactions": int(len(interaction_effects)),
        "train_metrics": train_metrics,
        "cv_folds_completed": int(len(cv_metrics)),
        "cv_requested_folds": int(config.spatial_cv_folds),
        "cv_effective_folds": cv_effective,
        "bootstrap_repeats_requested": int(dataset.config.get("bootstrap_repeats", 0)),
        "within_label_runs_completed": int(sum(1 for item in within_label_runs if item.get("status") == "completed")),
        "within_label_runs": within_label_runs,
        "top_main_effects": main_effects.head(20).to_dict(orient="records") if not main_effects.empty else [],
        "top_interactions": interaction_effects.head(20).to_dict(orient="records") if not interaction_effects.empty else [],
        "files": dict(files),
    }
    if not stability.empty:
        summary["top_stable_effects"] = stability.head(20).to_dict(orient="records")
    if not heterogeneity.empty:
        summary["heterogeneity_by_label"] = (
            heterogeneity.groupby("label", dropna=False)["heterogeneity_score"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .to_dict(orient="records")
        )
    return summary


def run_contour_gmi(
    dataset: ContourGmiDataset,
    *,
    output_dir: str | Path,
    config: ContourGmiConfig | None = None,
) -> ContourGmiResult:
    config = config or ContourGmiConfig(**{k: v for k, v in dataset.config.items() if k in ContourGmiConfig.__dataclass_fields__})
    out = Path(output_dir)
    primary = _run_single_fit(dataset, output_dir=out, config=config)
    main_effects = primary.get("main_effects", pd.DataFrame())
    interaction_effects = primary.get("interaction_effects", pd.DataFrame())
    predictions = primary.get("predictions", pd.DataFrame())
    groups = _assign_groups(main_effects, interaction_effects)
    groups_path = out / "groups.tsv"
    groups.to_csv(groups_path, sep="\t", index=False)

    cv_metrics = _run_spatial_cv(dataset, out, config)
    cv_path = out / "cv_metrics.tsv"
    cv_metrics.to_csv(cv_path, sep="\t", index=False)

    stability = _run_bootstrap_stability(dataset, out, config)
    stability_path = out / "stability.tsv"
    stability.to_csv(stability_path, sep="\t", index=False)

    heterogeneity = compute_contour_heterogeneity(dataset)
    heterogeneity_path = out / "heterogeneity.tsv"
    heterogeneity.to_csv(heterogeneity_path, sep="\t", index=False)
    within_label_runs = _run_within_label_heterogeneity(dataset, heterogeneity, out, config)
    visualization_files = _write_spatial_visualizations(dataset, out, predictions, config)

    files = dict(primary["files"])
    files.update(
        {
            "groups": str(groups_path),
            "cv_metrics": str(cv_path),
            "stability": str(stability_path),
            "heterogeneity": str(heterogeneity_path),
            **visualization_files,
        }
    )
    summary = _build_summary(
        dataset,
        out,
        main_effects,
        interaction_effects,
        predictions,
        cv_metrics,
        stability,
        heterogeneity,
        files,
        within_label_runs,
    )
    summary_path = out / "summary.json"
    report_path = out / "report.md"
    summary["files"]["summary_json"] = str(summary_path)
    summary["files"]["report_md"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    report_path.write_text(render_contour_gmi_report(summary), encoding="utf-8")

    return ContourGmiResult(
        output_dir=out,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        groups=groups,
        predictions=predictions,
        cv_metrics=cv_metrics,
        stability=stability,
        heterogeneity=heterogeneity,
        summary=summary,
        files=summary["files"],
    )


def load_atera_breast_for_gmi(dataset_root: str | Path = DEFAULT_ATERA_WTA_BREAST_DATASET_PATH):
    return read_xenium(
        str(dataset_root),
        as_="sdata",
        prefer="h5",
        include_transcripts=False,
        include_boundaries=False,
        include_images=False,
        clusters_relpath=DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS,
        cluster_column_name="cluster",
        cells_parquet="cells.parquet",
    )


def _resolve_contour_geojson(dataset_root: str | Path, config: ContourGmiConfig) -> Path:
    if config.contour_geojson:
        candidate = Path(config.contour_geojson).expanduser()
        if not candidate.is_absolute() and not candidate.exists():
            candidate = Path(dataset_root) / candidate
    else:
        candidate = Path(dataset_root) / DEFAULT_ATERA_CONTOUR_GEOJSON
    if not candidate.exists():
        raise FileNotFoundError(
            "Atera contour GeoJSON was not found. Provide `--contour-geojson` or create "
            f"{Path(dataset_root) / DEFAULT_ATERA_CONTOUR_GEOJSON} from the S1-S5 contour tutorial."
        )
    return candidate


def _ensure_atera_contours(sdata: Any, *, dataset_root: str | Path, config: ContourGmiConfig) -> Path:
    geojson = _resolve_contour_geojson(dataset_root, config)
    if config.contour_key not in sdata.shapes:
        add_contours_from_geojson(
            sdata,
            geojson,
            key=config.contour_key,
            id_key=config.contour_id_key,
            pixel_size_um=config.contour_pixel_size_um,
        )
    return geojson


def run_atera_breast_contour_gmi(
    *,
    dataset_root: str | Path = DEFAULT_ATERA_WTA_BREAST_DATASET_PATH,
    output_dir: str | Path,
    config: ContourGmiConfig | None = None,
    sample_id: str = DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID,
) -> ContourGmiResult:
    config = config or ContourGmiConfig()
    sdata = load_atera_breast_for_gmi(dataset_root)
    geojson = _ensure_atera_contours(sdata, dataset_root=dataset_root, config=config)
    provenance = {
        "dataset_root": str(dataset_root),
        "sample_id": sample_id,
        "contour_geojson": str(geojson),
    }
    dataset = build_contour_gmi_dataset(sdata, config=config, provenance=provenance)
    result = run_contour_gmi(dataset, output_dir=output_dir, config=config)

    control_config = config.copy_with(
        spatial_cv_folds=0,
        bootstrap_repeats=0,
        run_label_permutation_control=False,
        run_coordinate_shuffle_control=False,
        run_spatial_feature_shuffle_control=False,
        run_within_label_heterogeneity=False,
    )
    if config.run_label_permutation_control:
        rng = np.random.default_rng(config.random_seed)
        permuted_y = pd.Series(rng.permutation(dataset.y.to_numpy()), index=dataset.y.index, name=dataset.y.name)
        permuted_metadata = dataset.sample_metadata.copy()
        permuted_metadata.loc[permuted_metadata["sample_id"].isin(permuted_y.index), "y"] = (
            permuted_metadata.loc[permuted_metadata["sample_id"].isin(permuted_y.index), "sample_id"].map(permuted_y).astype(int)
        )
        permuted = ContourGmiDataset(
            X=dataset.X.copy(),
            y=permuted_y,
            sample_metadata=permuted_metadata,
            feature_metadata=dataset.feature_metadata.copy(),
            config=control_config.to_dict(),
            provenance={**dict(dataset.provenance), "control": "label_permutation"},
        )
        run_contour_gmi(permuted, output_dir=Path(output_dir) / "controls" / "label_permutation", config=control_config)

    if config.run_coordinate_shuffle_control:
        shuffled_config = control_config.copy_with(coordinate_shuffle=True)
        shuffled_output = Path(output_dir) / "controls" / "coordinate_shuffle"
        shuffled_provenance = {**provenance, "control": "coordinate_shuffle"}
        try:
            shuffled_dataset = build_contour_gmi_dataset(sdata, config=shuffled_config, provenance=shuffled_provenance)
        except ValueError as exc:
            _write_control_failure(
                output_dir=shuffled_output,
                control="coordinate_shuffle",
                error=exc,
                config=shuffled_config,
                provenance=shuffled_provenance,
            )
        else:
            run_contour_gmi(shuffled_dataset, output_dir=shuffled_output, config=shuffled_config)

    if config.run_spatial_feature_shuffle_control:
        shuffled = shuffle_spatial_feature_block(dataset, random_seed=config.random_seed)
        shuffled = ContourGmiDataset(
            X=shuffled.X,
            y=shuffled.y,
            sample_metadata=shuffled.sample_metadata,
            feature_metadata=shuffled.feature_metadata,
            config=control_config.to_dict(),
            provenance={**dict(dataset.provenance), "control": "spatial_feature_shuffle"},
        )
        run_contour_gmi(
            shuffled,
            output_dir=Path(output_dir) / "controls" / "spatial_feature_shuffle",
            config=control_config,
        )

    return result


# Backwards-compatible function names from the original spatial GMI prototype.
render_spatial_gmi_report = render_contour_gmi_report
run_atera_breast_gmi = run_atera_breast_contour_gmi
run_spatial_gmi = run_contour_gmi
