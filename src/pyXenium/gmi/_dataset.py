from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from shapely import intersects, points

from pyXenium.contour import build_contour_feature_table
from pyXenium.contour._analysis import _prepare_contours
from pyXenium.io.sdata_model import XeniumSData

from ._types import ContourGmiConfig, ContourGmiDataset


_SPATIAL_FEATURE_PREFIXES = {
    "geometry__": "geometry",
    "context__": "context",
    "omics__": "composition",
    "pathway__": "pathway",
    "protein__": "protein",
    "edge_contrast__": "edge_contrast",
    "gradient__": "gradient",
    "lr__": "ligand_receptor",
    "pathomics__": "pathomics",
    "embedding__": "embedding",
}

_COORDINATE_SPATIAL_TERMS = (
    "centroid_x",
    "centroid_y",
    "slide_x",
    "slide_y",
    "x_fraction",
    "y_fraction",
    "bbox_min_x",
    "bbox_max_x",
    "bbox_min_y",
    "bbox_max_y",
)


def _make_unique(values: pd.Index | list[str]) -> pd.Index:
    seen: dict[str, int] = {}
    out: list[str] = []
    for value in pd.Index(values).astype(str):
        count = seen.get(value, 0)
        seen[value] = count + 1
        out.append(value if count == 0 else f"{value}.{count}")
    return pd.Index(out, dtype=str)


def _safe_sample_id(contour_id: str) -> str:
    text = str(contour_id).strip().replace(" ", "_")
    return f"contour_{text}" if text else "contour_unknown"


def _feature_names(adata: Any) -> pd.Index:
    var = adata.var
    for column in ("gene_symbol", "name", "gene_name"):
        if column in var.columns:
            return _make_unique(var[column].astype(str))
    return _make_unique(adata.var_names.astype(str))


def _matrix_from_adata(adata: Any, layer: str | None) -> sparse.csr_matrix:
    if layer:
        if layer not in adata.layers:
            raise ValueError(f"AnnData layer {layer!r} is not present.")
        matrix = adata.layers[layer]
    else:
        matrix = adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def _cell_xy(adata: Any) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2].copy()
    for x_col, y_col in (
        ("x_centroid", "y_centroid"),
        ("cell_x_centroid", "cell_y_centroid"),
        ("cell_centroid_x", "cell_centroid_y"),
        ("centroid_x", "centroid_y"),
        ("x", "y"),
    ):
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return adata.obs[[x_col, y_col]].to_numpy(dtype=float)
    raise ValueError("AnnData must contain .obsm['spatial'] or recognized coordinate columns in .obs.")


def _copy_sdata_with_xy(sdata: XeniumSData, xy: np.ndarray) -> XeniumSData:
    adata = sdata.table.copy()
    adata.obsm["spatial"] = np.asarray(xy, dtype=float).copy()
    for x_col, y_col in (
        ("x_centroid", "y_centroid"),
        ("cell_x_centroid", "cell_y_centroid"),
        ("cell_centroid_x", "cell_centroid_y"),
        ("centroid_x", "centroid_y"),
        ("x", "y"),
    ):
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            adata.obs[x_col] = xy[:, 0]
            adata.obs[y_col] = xy[:, 1]
    return XeniumSData(
        table=adata,
        points={key: frame.copy() for key, frame in sdata.points.items()},
        shapes={key: frame.copy() for key, frame in sdata.shapes.items()},
        images=dict(sdata.images),
        contour_images={key: dict(value) for key, value in sdata.contour_images.items()},
        metadata=dict(sdata.metadata),
        point_sources=dict(sdata.point_sources),
    )


def _maybe_coordinate_shuffle(sdata: XeniumSData, config: ContourGmiConfig) -> XeniumSData:
    if not config.coordinate_shuffle:
        return sdata
    rng = np.random.default_rng(config.random_seed)
    xy = _cell_xy(sdata.table)
    shuffled = xy[rng.permutation(xy.shape[0]), :]
    return _copy_sdata_with_xy(sdata, shuffled)


def _copy_sdata_with_endpoint_contours(
    sdata: XeniumSData,
    *,
    contour_table: pd.DataFrame,
    config: ContourGmiConfig,
) -> XeniumSData:
    if config.contour_label_col not in contour_table.columns:
        return sdata
    endpoint_ids = set(
        contour_table.loc[
            contour_table[config.contour_label_col].astype(str).isin([config.positive_label, config.negative_label]),
            "contour_id",
        ].astype(str)
    )
    if not endpoint_ids:
        return sdata
    frame = sdata.shapes[config.contour_key].copy()
    frame = frame.loc[frame["contour_id"].astype(str).isin(endpoint_ids)].copy()
    return XeniumSData(
        table=sdata.table.copy(),
        points={key: value.copy() for key, value in sdata.points.items()},
        shapes={**{key: value.copy() for key, value in sdata.shapes.items()}, config.contour_key: frame},
        images=dict(sdata.images),
        contour_images={
            key: (
                {contour_id: image for contour_id, image in value.items() if contour_id in endpoint_ids}
                if key == config.contour_key
                else dict(value)
            )
            for key, value in sdata.contour_images.items()
        },
        metadata=dict(sdata.metadata),
        point_sources=dict(sdata.point_sources),
    )


def _membership_mask(geometry: Any, point_array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if geometry is None or geometry.is_empty or point_array.size == 0:
        return np.zeros(len(point_array), dtype=bool)
    buffered = geometry.buffer(1e-6)
    minx, miny, maxx, maxy = buffered.bounds
    candidate_mask = (
        (xy[:, 0] >= float(minx))
        & (xy[:, 0] <= float(maxx))
        & (xy[:, 1] >= float(miny))
        & (xy[:, 1] <= float(maxy))
    )
    if not candidate_mask.any():
        return np.zeros(len(point_array), dtype=bool)
    membership = np.zeros(len(point_array), dtype=bool)
    membership[candidate_mask] = np.asarray(intersects(buffered, point_array[candidate_mask]), dtype=bool)
    return membership


def _build_contour_membership(
    *,
    contour_table: pd.DataFrame,
    xy: np.ndarray,
) -> sparse.csr_matrix:
    point_array = np.asarray(points(xy[:, 0], xy[:, 1]), dtype=object)
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    for row_index, geometry in enumerate(contour_table["geometry"]):
        columns = np.flatnonzero(_membership_mask(geometry, point_array, xy))
        if columns.size == 0:
            continue
        row_indices.append(np.full(columns.size, row_index, dtype=int))
        col_indices.append(columns.astype(int))

    if not row_indices:
        return sparse.csr_matrix((len(contour_table), xy.shape[0]), dtype=float)
    rows = np.concatenate(row_indices)
    cols = np.concatenate(col_indices)
    data = np.ones(rows.size, dtype=float)
    return sparse.coo_matrix((data, (rows, cols)), shape=(len(contour_table), xy.shape[0])).tocsr()


def _build_sample_metadata(
    *,
    contour_table: pd.DataFrame,
    membership: sparse.csr_matrix,
    counts: sparse.csr_matrix,
    config: ContourGmiConfig,
) -> pd.DataFrame:
    labels = (
        contour_table[config.contour_label_col].astype(str)
        if config.contour_label_col in contour_table.columns
        else pd.Series("", index=contour_table.index, dtype=str)
    )
    n_cells = np.asarray(membership.sum(axis=1)).ravel().astype(int)
    library_size = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    rows: list[dict[str, Any]] = []
    sample_ids = _make_unique([_safe_sample_id(contour_id) for contour_id in contour_table["contour_id"].astype(str)])
    for row_index, contour_row in contour_table.reset_index(drop=True).iterrows():
        geometry = contour_row["geometry"]
        label = str(labels.iloc[row_index])
        y_value = 1 if label == config.positive_label else 0 if label == config.negative_label else np.nan
        retained = (
            label in {config.positive_label, config.negative_label}
            and n_cells[row_index] >= int(config.min_cells_per_contour)
            and library_size[row_index] >= float(config.min_library_size)
        )
        if label not in {config.positive_label, config.negative_label}:
            drop_reason = "non_endpoint_label"
        elif n_cells[row_index] < int(config.min_cells_per_contour):
            drop_reason = "too_few_cells"
        elif library_size[row_index] < float(config.min_library_size):
            drop_reason = "empty_library"
        else:
            drop_reason = "retained"
        centroid = geometry.centroid
        rows.append(
            {
                "sample_id": str(sample_ids[row_index]),
                "contour_id": str(contour_row["contour_id"]),
                "contour_key": config.contour_key,
                config.contour_label_col: label,
                "label": label,
                "y": y_value,
                "n_cells": int(n_cells[row_index]),
                "library_size": float(library_size[row_index]),
                "x_centroid": float(centroid.x),
                "y_centroid": float(centroid.y),
                "area_um2": float(geometry.area),
                "perimeter_um": float(geometry.length),
                "geometry_wkt": str(geometry.wkt),
                "retained": bool(retained),
                "drop_reason": drop_reason,
            }
        )
    return pd.DataFrame(rows)


def _standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = frame.mean(axis=0)
    stds = frame.std(axis=0, ddof=0)
    stds = stds.mask(stds <= 0, 1.0).fillna(1.0)
    return (frame - means) / stds, means, stds


def _select_rna_features(
    counts: sparse.csr_matrix,
    feature_names: pd.Index,
    y: pd.Series,
    config: ContourGmiConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(config.feature_count) <= 0:
        return pd.DataFrame(index=y.index), pd.DataFrame()

    dense_counts = counts.toarray().astype(float, copy=False)
    library_size = dense_counts.sum(axis=1)
    library_size[library_size <= 0] = 1.0
    log_cpm = np.log1p(dense_counts / library_size[:, None] * 1_000_000.0)

    detected_samples = np.asarray((dense_counts > 0).sum(axis=0)).ravel()
    prevalence = detected_samples / max(1, dense_counts.shape[0])
    variance = np.var(log_cpm, axis=0)
    y_array = y.to_numpy(dtype=int)
    if len(np.unique(y_array)) != 2:
        raise ValueError("GMI requires both S1 and S5 contour labels after filtering.")
    mean_positive = log_cpm[y_array == 1].mean(axis=0)
    mean_negative = log_cpm[y_array == 0].mean(axis=0)
    mean_difference = np.abs(mean_positive - mean_negative)

    valid = prevalence >= float(config.min_feature_prevalence)
    if int(valid.sum()) < min(int(config.feature_count), len(feature_names)):
        valid = detected_samples > 0
    score = (
        pd.Series(variance).rank(pct=True).to_numpy()
        + pd.Series(mean_difference).rank(pct=True).to_numpy()
        + pd.Series(prevalence).rank(pct=True).to_numpy() * 0.25
    )
    score[~valid] = -np.inf
    n_features = min(int(config.feature_count), int(np.isfinite(score).sum()))
    if n_features <= 0:
        raise ValueError("No eligible RNA features remained after GMI feature filtering.")
    selected = np.argsort(score)[::-1][:n_features]

    selected_names = feature_names[selected].astype(str)
    X_raw = pd.DataFrame(log_cpm[:, selected], index=y.index, columns=selected_names)
    X, means, stds = _standardize(X_raw)
    metadata = pd.DataFrame(
        {
            "feature": selected_names.to_numpy(dtype=str),
            "feature_block": "rna",
            "feature_group": "rna",
            "source_feature_index": selected.astype(int) + 1,
            "source_column": selected_names.to_numpy(dtype=str),
            "total_counts": dense_counts[:, selected].sum(axis=0),
            "detected_samples": detected_samples[selected].astype(int),
            "prevalence": prevalence[selected],
            "variance": variance[selected],
            "mean_difference": mean_difference[selected],
            "selection_score": score[selected],
            "standardization_mean": means.to_numpy(dtype=float),
            "standardization_std": stds.to_numpy(dtype=float),
        }
    ).sort_values("selection_score", ascending=False, ignore_index=True)
    metadata.insert(0, "feature_index", np.arange(1, len(metadata) + 1, dtype=int))
    X = X.loc[:, metadata["feature"].astype(str).tolist()]
    return X, metadata


def _spatial_feature_group(column: str) -> str:
    for prefix, group in _SPATIAL_FEATURE_PREFIXES.items():
        if str(column).startswith(prefix):
            return group
    return "spatial"


def _is_coordinate_spatial_feature(column: str) -> bool:
    text = str(column).lower()
    return any(term in text for term in _COORDINATE_SPATIAL_TERMS)


def _select_spatial_features(
    contour_features: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    y: pd.Series,
    config: ContourGmiConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(config.spatial_feature_count) <= 0 or contour_features.empty:
        return pd.DataFrame(index=y.index), pd.DataFrame()
    if "contour_id" not in contour_features.columns:
        return pd.DataFrame(index=y.index), pd.DataFrame()

    retained = sample_metadata.set_index("contour_id", drop=False).loc[
        sample_metadata.set_index("sample_id").loc[y.index, "contour_id"].astype(str)
    ]
    aligned = contour_features.copy()
    aligned["contour_id"] = aligned["contour_id"].astype(str)
    aligned = aligned.set_index("contour_id", drop=False).reindex(retained["contour_id"].astype(str))
    aligned.index = y.index

    candidates = [
        column
        for column in aligned.columns
        if any(str(column).startswith(prefix) for prefix in _SPATIAL_FEATURE_PREFIXES)
    ]
    if config.exclude_coordinate_spatial_features:
        candidates = [column for column in candidates if not _is_coordinate_spatial_feature(str(column))]
    numeric = aligned.loc[:, candidates].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any(axis=0)]
    if numeric.empty:
        return pd.DataFrame(index=y.index), pd.DataFrame()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(axis=0)).fillna(0.0)

    variance = numeric.var(axis=0, ddof=0)
    prevalence = (numeric != 0).sum(axis=0) / max(1, numeric.shape[0])
    y_array = y.to_numpy(dtype=int)
    mean_difference = (numeric.loc[y_array == 1].mean(axis=0) - numeric.loc[y_array == 0].mean(axis=0)).abs()
    valid = variance > 0
    score = variance.rank(pct=True) + mean_difference.rank(pct=True) + prevalence.rank(pct=True) * 0.25
    score = score.mask(~valid, -np.inf)
    n_features = min(int(config.spatial_feature_count), int(np.isfinite(score.to_numpy()).sum()))
    if n_features <= 0:
        return pd.DataFrame(index=y.index), pd.DataFrame()

    selected_columns = score.sort_values(ascending=False).head(n_features).index.astype(str).tolist()
    X_raw = numeric.loc[:, selected_columns]
    X, means, stds = _standardize(X_raw)
    metadata = pd.DataFrame(
        {
            "feature": selected_columns,
            "feature_block": "spatial",
            "feature_group": [_spatial_feature_group(column) for column in selected_columns],
            "source_feature_index": np.arange(1, len(selected_columns) + 1, dtype=int),
            "source_column": selected_columns,
            "total_counts": np.nan,
            "detected_samples": (numeric.loc[:, selected_columns] != 0).sum(axis=0).to_numpy(dtype=int),
            "prevalence": prevalence.loc[selected_columns].to_numpy(dtype=float),
            "variance": variance.loc[selected_columns].to_numpy(dtype=float),
            "mean_difference": mean_difference.loc[selected_columns].to_numpy(dtype=float),
            "selection_score": score.loc[selected_columns].to_numpy(dtype=float),
            "standardization_mean": means.to_numpy(dtype=float),
            "standardization_std": stds.to_numpy(dtype=float),
        }
    )
    return X, metadata


def _combine_feature_blocks(
    rna_X: pd.DataFrame,
    rna_metadata: pd.DataFrame,
    spatial_X: pd.DataFrame,
    spatial_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = [frame for frame in (rna_X, spatial_X) if not frame.empty]
    if not blocks:
        raise ValueError("No eligible RNA or spatial features remained for GMI.")
    X = pd.concat(blocks, axis=1)
    X.columns = _make_unique(X.columns.astype(str))
    metadata_blocks = [frame for frame in (rna_metadata, spatial_metadata) if not frame.empty]
    feature_metadata = pd.concat(metadata_blocks, ignore_index=True)
    feature_metadata["feature"] = X.columns.astype(str)
    feature_metadata["feature_index"] = np.arange(1, len(feature_metadata) + 1, dtype=int)
    return X, feature_metadata


def build_contour_gmi_dataset(
    sdata: XeniumSData,
    *,
    config: ContourGmiConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
    contour_feature_payload: Mapping[str, Any] | None = None,
) -> ContourGmiDataset:
    """Build a contour-level GMI design matrix from a XeniumSData object."""

    config = config or ContourGmiConfig()
    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance for contour GMI.")
    working_sdata = _maybe_coordinate_shuffle(sdata, config)
    contour_table = _prepare_contours(sdata=working_sdata, contour_key=config.contour_key, contour_query=None)
    if config.contour_label_col not in contour_table.columns:
        raise ValueError(
            f"Contour table for {config.contour_key!r} is missing label column {config.contour_label_col!r}."
        )
    contour_table = contour_table.loc[
        contour_table[config.contour_label_col].astype(str).isin([config.positive_label, config.negative_label])
    ].reset_index(drop=True)
    if contour_table.empty:
        raise ValueError(
            f"No endpoint contours with labels {config.positive_label!r} or {config.negative_label!r} were found."
        )
    working_sdata = _copy_sdata_with_endpoint_contours(working_sdata, contour_table=contour_table, config=config)

    xy = _cell_xy(working_sdata.table)
    membership = _build_contour_membership(contour_table=contour_table, xy=xy)
    counts = membership @ _matrix_from_adata(working_sdata.table, config.layer)
    sample_metadata = _build_sample_metadata(
        contour_table=contour_table,
        membership=membership,
        counts=counts.tocsr(),
        config=config,
    )
    retained_metadata = sample_metadata.loc[sample_metadata["retained"]].copy()
    if retained_metadata.empty:
        raise ValueError("No S1/S5 contours passed the contour GMI filters.")
    if retained_metadata["y"].nunique() != 2:
        raise ValueError("Filtered contour GMI samples must contain both S1 and S5 classes.")

    sample_order = retained_metadata["sample_id"].astype(str).tolist()
    retained_positions = sample_metadata.index[sample_metadata["retained"]].to_numpy(dtype=int)
    y = retained_metadata.set_index("sample_id")["y"].astype(int).loc[sample_order]
    retained_counts = counts.tocsr()[retained_positions, :]
    rna_X, rna_metadata = _select_rna_features(retained_counts, _feature_names(working_sdata.table), y, config)

    if contour_feature_payload is None and int(config.spatial_feature_count) > 0:
        contour_feature_payload = build_contour_feature_table(
            working_sdata,
            contour_key=config.contour_key,
            inner_rim_um=config.inner_rim_um,
            outer_rim_um=config.outer_rim_um,
            include_pathomics=config.include_pathomics,
        )
    contour_features = (
        pd.DataFrame(contour_feature_payload.get("contour_features", pd.DataFrame()))
        if contour_feature_payload is not None
        else pd.DataFrame()
    )
    spatial_X, spatial_metadata = _select_spatial_features(contour_features, sample_metadata, y, config)
    X, feature_metadata = _combine_feature_blocks(rna_X, rna_metadata, spatial_X, spatial_metadata)
    X.index = pd.Index(sample_order, name="sample_id")

    sample_metadata = sample_metadata.set_index("sample_id", drop=False)
    sample_metadata.loc[sample_order, "y"] = y.loc[sample_order].astype(int)
    sample_metadata = sample_metadata.reset_index(drop=True)
    return ContourGmiDataset(
        X=X,
        y=y,
        sample_metadata=sample_metadata,
        feature_metadata=feature_metadata,
        config=config.to_dict(),
        provenance={
            **dict(provenance or {}),
            "contour_key": config.contour_key,
            "endpoint": f"{config.positive_label}_vs_{config.negative_label}",
            "coordinate_shuffle": bool(config.coordinate_shuffle),
            "contour_feature_context": (
                dict(contour_feature_payload.get("context", {})) if contour_feature_payload is not None else {}
            ),
        },
    )


def compute_contour_heterogeneity(dataset: ContourGmiDataset) -> pd.DataFrame:
    """Score contour-to-contour heterogeneity within each endpoint label."""

    if dataset.X.empty:
        return pd.DataFrame()
    metadata = dataset.sample_metadata.set_index("sample_id", drop=False).loc[dataset.X.index].copy()
    config = ContourGmiConfig(**{k: v for k, v in dataset.config.items() if k in ContourGmiConfig.__dataclass_fields__})
    rows: list[pd.DataFrame] = []
    X = dataset.X.astype(float)
    for label, group in metadata.groupby("label", sort=False):
        group_ids = group["sample_id"].astype(str).tolist()
        if len(group_ids) < int(config.heterogeneity_min_contours):
            skipped = group.copy()
            skipped["heterogeneity_score"] = np.nan
            skipped["heterogeneity_pc1"] = np.nan
            skipped["heterogeneity_pc2"] = np.nan
            skipped["heterogeneity_class"] = "insufficient_contours"
            rows.append(skipped)
            continue
        values = X.loc[group_ids].to_numpy(dtype=float)
        center = values.mean(axis=0)
        score = np.sqrt(((values - center) ** 2).sum(axis=1))
        pcs = np.zeros((len(group_ids), 2), dtype=float)
        if len(group_ids) >= 3 and values.shape[1] >= 2:
            try:
                from sklearn.decomposition import PCA

                n_components = min(2, values.shape[0], values.shape[1])
                transformed = PCA(n_components=n_components, random_state=config.random_seed).fit_transform(values)
                pcs[:, :n_components] = transformed
            except Exception:
                pass
        low_cut = float(np.nanquantile(score, 1 / 3))
        high_cut = float(np.nanquantile(score, 2 / 3))
        heterogeneity_class = np.full(len(group_ids), "mid", dtype=object)
        heterogeneity_class[score <= low_cut] = "low"
        heterogeneity_class[score >= high_cut] = "high"
        current = group.copy()
        current["heterogeneity_score"] = score
        current["heterogeneity_pc1"] = pcs[:, 0]
        current["heterogeneity_pc2"] = pcs[:, 1]
        current["heterogeneity_class"] = heterogeneity_class
        rows.append(current)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_within_label_heterogeneity_dataset(
    dataset: ContourGmiDataset,
    heterogeneity: pd.DataFrame,
    *,
    label: str,
) -> ContourGmiDataset | None:
    subset = heterogeneity.loc[
        (heterogeneity["label"].astype(str) == str(label))
        & heterogeneity["heterogeneity_class"].isin(["low", "high"])
    ].copy()
    if subset.empty or subset["heterogeneity_class"].nunique() != 2:
        return None
    ids = subset["sample_id"].astype(str).tolist()
    y = pd.Series((subset["heterogeneity_class"].astype(str) == "high").astype(int).to_numpy(), index=ids, name="y")
    metadata = dataset.sample_metadata.set_index("sample_id", drop=False).loc[ids].reset_index(drop=True)
    metadata = metadata.drop(columns=[column for column in metadata.columns if column.startswith("heterogeneity_")], errors="ignore")
    hetero_cols = [
        "sample_id",
        "heterogeneity_score",
        "heterogeneity_pc1",
        "heterogeneity_pc2",
        "heterogeneity_class",
    ]
    metadata = metadata.merge(subset[hetero_cols], on="sample_id", how="left")
    metadata["y"] = metadata["sample_id"].map(y).astype(int)
    config = ContourGmiConfig(**{k: v for k, v in dataset.config.items() if k in ContourGmiConfig.__dataclass_fields__})
    return ContourGmiDataset(
        X=dataset.X.loc[ids].copy(),
        y=y,
        sample_metadata=metadata.reset_index(drop=True),
        feature_metadata=dataset.feature_metadata.copy(),
        config=config.copy_with(
            spatial_cv_folds=0,
            bootstrap_repeats=0,
            run_label_permutation_control=False,
            run_coordinate_shuffle_control=False,
            run_spatial_feature_shuffle_control=False,
            run_within_label_heterogeneity=False,
        ).to_dict(),
        provenance={
            **dict(dataset.provenance),
            "endpoint": f"{label}_within_label_high_vs_low_heterogeneity",
            "within_label": str(label),
        },
    )


def shuffle_spatial_feature_block(dataset: ContourGmiDataset, *, random_seed: int | None = None) -> ContourGmiDataset:
    rng = np.random.default_rng(dataset.config.get("random_seed", 1) if random_seed is None else random_seed)
    feature_metadata = dataset.feature_metadata.copy()
    spatial_features = feature_metadata.loc[
        feature_metadata["feature_block"].astype(str) == "spatial", "feature"
    ].astype(str)
    X = dataset.X.copy()
    for feature in spatial_features:
        if feature in X.columns:
            X[feature] = rng.permutation(X[feature].to_numpy())
    return ContourGmiDataset(
        X=X,
        y=dataset.y.copy(),
        sample_metadata=dataset.sample_metadata.copy(),
        feature_metadata=feature_metadata,
        config=dict(dataset.config),
        provenance={**dict(dataset.provenance), "control": "spatial_feature_shuffle"},
    )


def write_contour_gmi_dataset(dataset: ContourGmiDataset, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / "design_matrix.tsv.gz"
    sample_path = out / "sample_metadata.tsv"
    feature_path = out / "feature_metadata.tsv"
    config_path = out / "dataset_config.json"
    provenance_path = out / "provenance.json"

    dataset.X.to_csv(design_path, sep="\t", compression="gzip", index_label="sample_id")
    dataset.sample_metadata.to_csv(sample_path, sep="\t", index=False)
    dataset.feature_metadata.to_csv(feature_path, sep="\t", index=False)
    config_path.write_text(json.dumps(dict(dataset.config), indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(dict(dataset.provenance), indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "design_matrix": str(design_path),
        "sample_metadata": str(sample_path),
        "feature_metadata": str(feature_path),
        "dataset_config": str(config_path),
        "provenance": str(provenance_path),
    }


# Backwards-compatible function names. These now build/write contour samples.
build_spatial_gmi_dataset = build_contour_gmi_dataset
write_spatial_gmi_dataset = write_contour_gmi_dataset
