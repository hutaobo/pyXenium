from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from pyXenium.contour import build_contour_feature_table
from pyXenium.contour._feature_table import DEFAULT_CONTOUR_LR_PAIRS, DEFAULT_CONTOUR_PATHWAYS
from pyXenium.io.sdata_model import XeniumSData

__all__ = ["DEFAULT_BOUNDARY_PROGRAM_LIBRARY", "score_contour_boundary_programs"]

_IDENTIFIER_COLUMNS = {"sample_id", "contour_key", "contour_id"}

DEFAULT_BOUNDARY_PROGRAM_LIBRARY: dict[str, dict[str, dict[str, float]]] = {
    "immune_exclusion": {
        "image": {
            "edge_contrast__pathomics__stain_blue_ratio": 0.6,
            "edge_contrast__pathomics__nuclear_density_proxy": 0.5,
            "geometry__boundary_roughness": 0.2,
        },
        "omics": {
            "edge_contrast__omics__state_fraction__t_cell_exhausted_cytotoxic": 1.2,
            "edge_contrast__omics__state_fraction__b_plasma_like": 0.8,
            "edge_contrast__omics__niche_fraction__immune_rich": 1.0,
            "edge_contrast__pathway__immune_activation": 1.0,
            "lr__cxcl13_cxcr5__outer_minus_inner": 0.6,
        },
        "spatial": {
            "gradient__immune__outer_minus_inner": 1.2,
            "context__contact_degree": 0.3,
            "context__neighbor_same_label_fraction": 0.2,
        },
    },
    "myeloid_vascular_belt": {
        "image": {
            "pathomics__outer_rim__stain_blue_ratio": 0.7,
            "pathomics__outer_rim__nuclear_density_proxy": 0.7,
            "pathomics__outer_rim__edge_density": 0.3,
        },
        "omics": {
            "omics__outer_rim__state_fraction__macrophage_like": 1.2,
            "omics__outer_rim__state_fraction__endothelial_perivascular": 1.2,
            "pathway__outer_rim__myeloid_activation": 1.0,
            "pathway__outer_rim__vascular_stromal": 1.0,
            "lr__spp1_cd44__cross_zone": 0.7,
            "lr__vegfa_kdr__cross_zone": 0.7,
        },
        "spatial": {
            "gradient__myeloid__outer_minus_inner": 1.0,
            "gradient__vascular__outer_minus_inner": 1.0,
            "context__contact_degree": 0.3,
        },
    },
    "emt_invasive_front": {
        "image": {
            "geometry__eccentricity": 0.4,
            "geometry__boundary_roughness": 0.7,
            "pathomics__outer_rim__stain_pink_ratio": 0.6,
            "edge_contrast__pathomics__texture_std": 0.4,
        },
        "omics": {
            "omics__outer_rim__state_fraction__emt_like_tumor": 1.2,
            "pathway__outer_rim__emt_invasion": 1.1,
            "pathway__outer_rim__stromal_matrix": 0.9,
            "gradient__emt__outer_minus_inner": 1.0,
        },
        "spatial": {
            "context__neighbor_count": 0.2,
            "gradient__emt__boundary_peak": 0.7,
            "edge_contrast__pathway__stromal_matrix": 0.6,
        },
    },
    "stromal_encapsulation": {
        "image": {
            "geometry__compactness": 0.5,
            "pathomics__outer_rim__stain_pink_ratio": 0.8,
            "pathomics__outer_rim__texture_std": 0.4,
        },
        "omics": {
            "omics__outer_rim__state_fraction__endothelial_perivascular": 0.5,
            "pathway__outer_rim__stromal_matrix": 1.2,
            "pathway__outer_rim__vascular_stromal": 0.8,
            "gradient__emt__outer_minus_inner": 0.5,
        },
        "spatial": {
            "context__neighbor_same_label_fraction": 0.5,
            "context__contact_degree": 0.4,
            "edge_contrast__pathway__stromal_matrix": 0.8,
        },
    },
    "tls_adjacent_activation": {
        "image": {
            "pathomics__outer_rim__stain_blue_ratio": 0.8,
            "pathomics__outer_rim__nuclear_density_proxy": 0.5,
            "geometry__hole_burden": 0.2,
        },
        "omics": {
            "omics__outer_rim__state_fraction__b_plasma_like": 1.1,
            "omics__outer_rim__state_fraction__t_cell_exhausted_cytotoxic": 0.9,
            "pathway__outer_rim__tls_activation": 1.1,
            "lr__cxcl13_cxcr5__cross_zone": 0.8,
        },
        "spatial": {
            "gradient__immune__outer_minus_inner": 0.8,
            "edge_contrast__omics__niche_fraction__immune_rich": 0.7,
            "context__contact_degree": 0.2,
        },
    },
    "necrotic_hypoxic_rim": {
        "image": {
            "pathomics__whole__texture_entropy": 0.5,
            "edge_contrast__pathomics__mean_b": -0.2,
            "edge_contrast__pathomics__nuclear_density_proxy": -0.5,
        },
        "omics": {
            "pathway__outer_rim__hypoxia_necrosis": 1.2,
            "gradient__hypoxia__outer_minus_inner": 1.0,
            "edge_contrast__pathway__hypoxia_necrosis": 0.8,
        },
        "spatial": {
            "geometry__hole_burden": 0.4,
            "geometry__boundary_roughness": 0.3,
            "context__contact_degree": 0.2,
        },
    },
}


def score_contour_boundary_programs(
    sdata: XeniumSData,
    *,
    contour_key: str,
    feature_table: dict[str, Any] | None = None,
    program_library: str | Mapping[str, Mapping[str, Mapping[str, float]]] = "tumor_boundary_v1",
) -> dict[str, Any]:
    """
    Score contour-level tumor boundary ecology programs.
    """

    if feature_table is None:
        feature_table = build_contour_feature_table(sdata, contour_key=contour_key)
    contour_features = feature_table["contour_features"].copy()
    if contour_features.empty:
        raise ValueError("Contour feature table is empty; cannot score boundary programs.")

    resolved_library = _resolve_program_library(program_library)
    score_table = contour_features.loc[:, ["sample_id", "contour_key", "contour_id"]].copy()
    score_metadata_rows: list[dict[str, Any]] = []

    for program_name, components in resolved_library.items():
        image_score = _weighted_score(contour_features, components.get("image", {}))
        omics_score = _weighted_score(contour_features, components.get("omics", {}))
        spatial_score = _weighted_score(contour_features, components.get("spatial", {}))
        combined = _combine_component_scores(image_score, omics_score, spatial_score)

        score_table[f"{program_name}__image_evidence"] = image_score
        score_table[f"{program_name}__omics_evidence"] = omics_score
        score_table[f"{program_name}__spatial_evidence"] = spatial_score
        score_table[program_name] = combined

        for component_name, weights in components.items():
            for feature_name, weight in weights.items():
                score_metadata_rows.append(
                    {
                        "program": program_name,
                        "component": component_name,
                        "feature": feature_name,
                        "weight": float(weight),
                        "available": feature_name in contour_features.columns,
                    }
                )

    program_columns = list(resolved_library)
    score_table["top_program"] = score_table[program_columns].idxmax(axis=1)
    score_table["top_program_score"] = score_table[program_columns].max(axis=1)

    return {
        "program_scores": score_table,
        "program_library": resolved_library,
        "program_feature_weights": pd.DataFrame(score_metadata_rows),
        "feature_table": feature_table,
    }


def cluster_contour_ecotypes(
    contour_features: pd.DataFrame,
    program_scores: pd.DataFrame,
    *,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = contour_features.merge(
        program_scores.loc[:, ["contour_id", *list(DEFAULT_BOUNDARY_PROGRAM_LIBRARY)]],
        on="contour_id",
        how="left",
    )
    numeric = _select_model_features(merged)
    if numeric.empty:
        labels = np.zeros(len(merged), dtype=int)
        return (
            pd.DataFrame(
                {
                    "sample_id": merged["sample_id"].to_numpy(),
                    "contour_key": merged["contour_key"].to_numpy(),
                    "contour_id": merged["contour_id"].to_numpy(),
                    "ecotype": ["ecotype_0"] * len(merged),
                    "ecotype_index": labels,
                }
            ),
            {
                "n_clusters": 1,
                "bootstrap_mean_ari": float("nan"),
                "silhouette": float("nan"),
                "feature_count": 0,
            },
        )

    matrix = numeric.to_numpy(dtype=float)
    matrix = _fill_nan_with_median(matrix)
    scaled = StandardScaler().fit_transform(matrix)
    n_samples = scaled.shape[0]

    if n_samples <= 2:
        labels = np.arange(n_samples, dtype=int) if n_samples > 1 else np.zeros(n_samples, dtype=int)
        n_clusters = max(1, n_samples)
        silhouette = float("nan")
        bootstrap = float("nan")
    else:
        candidate_k = range(2, min(6, n_samples - 1) + 1)
        best_k = 2
        best_score = float("-inf")
        best_labels = None
        for k in candidate_k:
            model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            candidate_labels = model.fit_predict(scaled)
            if len(np.unique(candidate_labels)) < 2:
                continue
            try:
                current_score = float(silhouette_score(scaled, candidate_labels))
            except Exception:
                current_score = float("-inf")
            if current_score > best_score:
                best_score = current_score
                best_k = k
                best_labels = candidate_labels
        if best_labels is None:
            best_labels = KMeans(n_clusters=2, n_init=20, random_state=random_state).fit_predict(scaled)
            best_score = float("nan")
            best_k = 2
        labels = np.asarray(best_labels, dtype=int)
        n_clusters = int(best_k)
        silhouette = best_score
        bootstrap = _bootstrap_cluster_stability(scaled, labels=labels, n_clusters=n_clusters, random_state=random_state)

    assignments = pd.DataFrame(
        {
            "sample_id": merged["sample_id"].to_numpy(),
            "contour_key": merged["contour_key"].to_numpy(),
            "contour_id": merged["contour_id"].to_numpy(),
            "ecotype_index": labels,
            "ecotype": [f"ecotype_{label}" for label in labels],
        }
    )
    return assignments, {
        "n_clusters": int(n_clusters),
        "bootstrap_mean_ari": bootstrap,
        "silhouette": silhouette,
        "feature_count": int(numeric.shape[1]),
    }


def associate_contour_image_omics(
    feature_table: dict[str, Any],
    program_scores: pd.DataFrame,
    ecotype_assignments: pd.DataFrame,
) -> dict[str, Any]:
    contour_features = feature_table["contour_features"].copy()
    merged = contour_features.merge(
        program_scores.loc[:, ["contour_id", *list(DEFAULT_BOUNDARY_PROGRAM_LIBRARY), "top_program", "top_program_score"]],
        on="contour_id",
        how="left",
    ).merge(
        ecotype_assignments.loc[:, ["contour_id", "ecotype", "ecotype_index"]],
        on="contour_id",
        how="left",
    )

    image_columns = _select_columns(
        merged,
        prefixes=("pathomics__", "geometry__", "edge_contrast__pathomics__"),
    )
    omics_columns = _select_columns(
        merged,
        prefixes=("omics__", "pathway__", "protein__", "rna__", "gradient__", "lr__", "edge_contrast__omics__", "edge_contrast__pathway__"),
    )
    correlation_rows: list[dict[str, Any]] = []
    for image_column in image_columns:
        for omics_column in omics_columns:
            rho, p_value = _safe_spearman(merged[image_column], merged[omics_column])
            if np.isnan(rho):
                continue
            correlation_rows.append(
                {
                    "image_feature": image_column,
                    "omics_feature": omics_column,
                    "effect_size": float(rho),
                    "p_value": p_value,
                }
            )
    morphology_omics = pd.DataFrame(correlation_rows)
    if not morphology_omics.empty:
        morphology_omics["fdr"] = _benjamini_hochberg(morphology_omics["p_value"])
        morphology_omics = morphology_omics.sort_values(["fdr", "effect_size"], ascending=[True, False], kind="stable").reset_index(drop=True)

    matched_exemplars = _match_program_controls(
        merged,
        program_columns=list(DEFAULT_BOUNDARY_PROGRAM_LIBRARY),
    )
    program_feature_deltas = _program_feature_delta_table(
        feature_table=feature_table,
        merged=merged,
        matched_exemplars=matched_exemplars,
    )
    hypothesis_ranking = _build_hypothesis_ranking(
        merged=merged,
        matched_exemplars=matched_exemplars,
        program_feature_deltas=program_feature_deltas,
    )
    return {
        "morphology_omics": morphology_omics,
        "program_feature_deltas": program_feature_deltas,
        "hypothesis_ranking": hypothesis_ranking,
        "matched_exemplars": matched_exemplars,
    }


def _resolve_program_library(
    program_library: str | Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    if isinstance(program_library, str):
        if program_library != "tumor_boundary_v1":
            raise ValueError("`program_library` currently supports only 'tumor_boundary_v1'.")
        return DEFAULT_BOUNDARY_PROGRAM_LIBRARY
    resolved: dict[str, dict[str, dict[str, float]]] = {}
    for program_name, components in program_library.items():
        resolved[str(program_name)] = {
            str(component_name): {str(feature): float(weight) for feature, weight in feature_weights.items()}
            for component_name, feature_weights in components.items()
        }
    return resolved


def _weighted_score(frame: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    if not weights:
        return np.full(len(frame), np.nan, dtype=float)
    vectors = []
    total_weight = 0.0
    for column, weight in weights.items():
        if column not in frame.columns:
            continue
        values = _robust_zscore(frame[column].to_numpy(dtype=float))
        if not np.any(np.isfinite(values)):
            continue
        vectors.append(values * float(weight))
        total_weight += abs(float(weight))
    if not vectors or total_weight <= 0:
        return np.full(len(frame), np.nan, dtype=float)
    stacked = np.vstack(vectors)
    return np.nanmean(stacked, axis=0)


def _combine_component_scores(*components: np.ndarray) -> np.ndarray:
    stacked = np.vstack(
        [
            np.asarray(component, dtype=float)
            for component in components
            if component is not None and np.asarray(component).size > 0
        ]
    )
    valid = np.isfinite(stacked)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(stacked.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return np.full(array.shape, np.nan, dtype=float)
    center = float(np.nanmedian(array[finite]))
    mad = float(np.nanmedian(np.abs(array[finite] - center)))
    scale = 1.4826 * mad if mad > 0 else float(np.nanstd(array[finite]))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    output = (array - center) / scale
    output[~finite] = np.nan
    return output


def _select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = _select_columns(
        frame,
        prefixes=(
            "geometry__",
            "context__",
            "pathomics__",
            "omics__",
            "pathway__",
            "protein__whole__",
            "rna__whole__",
            "gradient__",
            "lr__",
            "edge_contrast__",
        ),
    ) + list(DEFAULT_BOUNDARY_PROGRAM_LIBRARY)
    deduped = []
    for column in columns:
        if column in frame.columns and column not in deduped:
            series = pd.to_numeric(frame[column], errors="coerce")
            if np.isfinite(series.to_numpy(dtype=float)).sum() < 2:
                continue
            if np.nanstd(series.to_numpy(dtype=float)) < 1e-8:
                continue
            deduped.append(column)
    return frame.loc[:, deduped].apply(pd.to_numeric, errors="coerce")


def _fill_nan_with_median(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float).copy()
    for column_index in range(array.shape[1]):
        column = array[:, column_index]
        mask = np.isfinite(column)
        fill = float(np.nanmedian(column[mask])) if mask.any() else 0.0
        column[~mask] = fill
        array[:, column_index] = column
    return array


def _bootstrap_cluster_stability(
    matrix: np.ndarray,
    *,
    labels: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_iterations: int = 20,
) -> float:
    if matrix.shape[0] < 4 or matrix.shape[1] < 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    aris: list[float] = []
    for _ in range(int(n_iterations)):
        feature_idx = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        sampled = matrix[:, feature_idx]
        try:
            sampled_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(rng.integers(0, 1_000_000))).fit_predict(sampled)
        except Exception:
            continue
        if len(np.unique(sampled_labels)) < 2:
            continue
        aris.append(float(adjusted_rand_score(labels, sampled_labels)))
    return float(np.mean(aris)) if aris else float("nan")


def _safe_spearman(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    if np.nanstd(left_values[mask]) < 1e-8 or np.nanstd(right_values[mask]) < 1e-8:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(left_values[mask], right_values[mask])
    if np.isnan(rho):
        return float("nan"), float("nan")
    return float(rho), float(p_value)


def _match_program_controls(
    merged: pd.DataFrame,
    *,
    program_columns: Sequence[str],
    top_n: int = 3,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for program_name in program_columns:
        if program_name not in merged.columns:
            continue
        ordered = merged.sort_values(program_name, ascending=False, kind="stable")
        exemplars = ordered.head(min(top_n, len(ordered)))
        threshold = float(merged[program_name].median())
        control_pool = merged.loc[merged[program_name] <= threshold].copy()
        if control_pool.empty:
            control_pool = merged.sort_values(program_name, ascending=True, kind="stable").head(max(top_n, 1)).copy()
        used: set[str] = set()
        for _, exemplar in exemplars.iterrows():
            candidates = control_pool.loc[~control_pool["contour_id"].astype(str).isin(used)].copy()
            if candidates.empty:
                continue
            if "classification_name" in candidates.columns and pd.notna(exemplar.get("classification_name")):
                same_class = candidates["classification_name"].astype(str) == str(exemplar["classification_name"])
                if same_class.any():
                    candidates = candidates.loc[same_class].copy()
            candidates["match_distance"] = (
                _standardized_distance(candidates["geometry__area_um2"], exemplar["geometry__area_um2"])
                + _standardized_distance(candidates["geometry__perimeter_um"], exemplar["geometry__perimeter_um"])
                + _standardized_distance(candidates["context__centroid_x_um"], exemplar["context__centroid_x_um"])
                + _standardized_distance(candidates["context__centroid_y_um"], exemplar["context__centroid_y_um"])
            )
            match = candidates.sort_values(["match_distance", program_name], ascending=[True, True], kind="stable").iloc[0]
            used.add(str(match["contour_id"]))
            records.append(
                {
                    "sample_id": exemplar["sample_id"],
                    "contour_key": exemplar["contour_key"],
                    "program": program_name,
                    "exemplar_id": exemplar["contour_id"],
                    "control_id": match["contour_id"],
                    "exemplar_score": float(exemplar[program_name]),
                    "control_score": float(match[program_name]),
                    "delta_score": float(exemplar[program_name] - match[program_name]),
                    "match_distance": float(match["match_distance"]),
                    "classification_name": exemplar.get("classification_name"),
                }
            )
    return pd.DataFrame(records)


def _program_feature_delta_table(
    *,
    feature_table: dict[str, Any],
    merged: pd.DataFrame,
    matched_exemplars: pd.DataFrame,
) -> pd.DataFrame:
    contour_index = merged.set_index("contour_id", drop=False)
    rna = feature_table["rna_pseudobulk"].set_index("contour_id", drop=False)
    pathways = feature_table["pathway_activity"].set_index("contour_id", drop=False)
    lr = feature_table["ligand_receptor_summary"].set_index("contour_id", drop=False)

    tables = [
        ("gene", rna),
        ("pathway", pathways),
        ("marker_pair", lr),
    ]
    rows: list[dict[str, Any]] = []
    for program_name, program_matches in matched_exemplars.groupby("program", sort=False, dropna=False):
        exemplar_ids = program_matches["exemplar_id"].astype(str).tolist()
        control_ids = program_matches["control_id"].astype(str).tolist()
        if not exemplar_ids or not control_ids:
            continue
        for feature_kind, table in tables:
            numeric_columns = [
                column
                for column in table.columns
                if column not in _IDENTIFIER_COLUMNS and np.issubdtype(np.asarray(table[column]).dtype, np.number)
            ]
            for column in numeric_columns:
                exemplar_values = pd.to_numeric(table.loc[table.index.intersection(exemplar_ids), column], errors="coerce").dropna()
                control_values = pd.to_numeric(table.loc[table.index.intersection(control_ids), column], errors="coerce").dropna()
                if exemplar_values.empty or control_values.empty:
                    continue
                p_value = _safe_mannwhitney(exemplar_values.to_numpy(dtype=float), control_values.to_numpy(dtype=float))
                matched_delta = _paired_delta(table, program_matches, column)
                rows.append(
                    {
                        "program": program_name,
                        "feature_kind": feature_kind,
                        "feature_name": column,
                        "effect_size": float(exemplar_values.mean() - control_values.mean()),
                        "p_value": p_value,
                        "matched_control_delta": matched_delta,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = _benjamini_hochberg(result["p_value"])
        result = result.sort_values(["program", "fdr", "effect_size"], ascending=[True, True, False], kind="stable").reset_index(drop=True)
    return result


def _build_hypothesis_ranking(
    *,
    merged: pd.DataFrame,
    matched_exemplars: pd.DataFrame,
    program_feature_deltas: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for program_name in DEFAULT_BOUNDARY_PROGRAM_LIBRARY:
        score_series = pd.to_numeric(merged[program_name], errors="coerce")
        top_match = matched_exemplars.loc[matched_exemplars["program"] == program_name]
        top_features = program_feature_deltas.loc[program_feature_deltas["program"] == program_name].head(3)
        feature_summary = ", ".join(top_features["feature_name"].astype(str).tolist()) or "no strong differential features"
        if program_name == "immune_exclusion":
            sentence = f"Outer-rim immune signals dominate the boundary while the inner rim stays comparatively immune-cold; top evidence: {feature_summary}."
        elif program_name == "myeloid_vascular_belt":
            sentence = f"Myeloid and vascular programs co-accumulate along the contour edge, consistent with a perivascular suppressive belt; top evidence: {feature_summary}."
        elif program_name == "emt_invasive_front":
            sentence = f"The boundary shows an EMT- and stroma-shifted invasive front relative to matched controls; top evidence: {feature_summary}."
        elif program_name == "stromal_encapsulation":
            sentence = f"Stromal matrix features wrap the contour and may form a physical barrier; top evidence: {feature_summary}."
        elif program_name == "tls_adjacent_activation":
            sentence = f"Lymphoid activation is enriched near the outer rim, consistent with TLS-adjacent immune organization; top evidence: {feature_summary}."
        else:
            sentence = f"Hypoxia-like rim features and reduced cellularity suggest a necrotic or stressed boundary ecology; top evidence: {feature_summary}."
        rows.append(
            {
                "program": program_name,
                "mean_score": float(score_series.mean()),
                "max_score": float(score_series.max()),
                "n_matched_exemplars": int(len(top_match)),
                "hypothesis": sentence,
            }
        )
    return pd.DataFrame(rows).sort_values("max_score", ascending=False, kind="stable").reset_index(drop=True)


def _select_columns(frame: pd.DataFrame, *, prefixes: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        text = str(column)
        if any(text.startswith(prefix) for prefix in prefixes):
            columns.append(text)
    return columns


def _standardized_distance(values: pd.Series, target: Any) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    scale = float(np.nanstd(numeric.to_numpy(dtype=float)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (numeric - float(target)).abs() / scale


def _safe_mannwhitney(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    try:
        _, p_value = mannwhitneyu(left, right, alternative="two-sided")
        return float(p_value)
    except Exception:
        return float("nan")


def _paired_delta(table: pd.DataFrame, matches: pd.DataFrame, column: str) -> float:
    deltas: list[float] = []
    for _, row in matches.iterrows():
        exemplar_id = str(row["exemplar_id"])
        control_id = str(row["control_id"])
        if exemplar_id not in table.index or control_id not in table.index:
            continue
        exemplar_value = pd.to_numeric(table.loc[exemplar_id, column], errors="coerce")
        control_value = pd.to_numeric(table.loc[control_id, column], errors="coerce")
        if np.isfinite(exemplar_value) and np.isfinite(control_value):
            deltas.append(float(exemplar_value - control_value))
    return float(np.mean(deltas)) if deltas else float("nan")


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
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
        candidate = ranked[reverse_index] * n / rank
        running = min(running, candidate)
        adjusted[reverse_index] = running
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    output[finite_mask] = np.clip(restored, 0.0, 1.0)
    return output
