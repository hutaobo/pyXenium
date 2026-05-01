from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from ._types import ContourGmiDataset, GmiModuleConfig, GmiModuleResult


CURATED_GMI_GENE_SETS: dict[str, tuple[str, ...]] = {
    "CAF_ECM": (
        "ACTA2",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "COL6A1",
        "DCN",
        "FAP",
        "FN1",
        "LUM",
        "MMP2",
        "MMP11",
        "POSTN",
        "SPARC",
        "TAGLN",
        "THBS2",
    ),
    "angiogenesis_pericyte": (
        "ANGPT2",
        "COL4A1",
        "ENG",
        "ESAM",
        "FLT1",
        "KDR",
        "MCAM",
        "PDGFRB",
        "PECAM1",
        "RGS5",
        "VEGFA",
        "VWF",
    ),
    "myeloid_vascular_context": (
        "AIF1",
        "C1QA",
        "C1QB",
        "C1QC",
        "CD14",
        "CD68",
        "CD163",
        "CSF1R",
        "FCGR3A",
        "HLA-DRA",
        "LST1",
        "MS4A7",
        "TYROBP",
    ),
    "Notch": ("DLL1", "DLL3", "DLL4", "HES1", "HEY1", "HEY2", "JAG1", "JAG2", "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4"),
    "IGF_MAPK": (
        "BRAF",
        "DUSP6",
        "EGR1",
        "FOS",
        "IGF1",
        "IGF1R",
        "IGF2",
        "IRS1",
        "IRS2",
        "JUN",
        "KRAS",
        "MAPK1",
        "MAPK3",
        "NRAS",
    ),
    "Wnt": ("AXIN2", "CTNNB1", "FZD1", "FZD2", "FZD3", "FZD4", "FZD5", "FZD6", "FZD7", "LGR5", "LEF1", "TCF7", "WNT3A", "WNT5A"),
    "TGF_beta": ("CTGF", "SERPINE1", "SMAD2", "SMAD3", "SMAD4", "SMAD7", "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2"),
    "CXCL12_CXCR4": ("ACKR3", "CXCL12", "CXCR4"),
    "CSF1_CSF1R": ("CSF1", "CSF1R", "IL34"),
    "11q13_amplicon": ("ANO1", "CCND1", "CTTN", "EMSY", "FGF3", "FGF4", "FGF19", "ORAOV1", "PAK1"),
    "DCIS_apocrine_luminal": (
        "AGR2",
        "AR",
        "EFHD1",
        "ESR1",
        "FOXA1",
        "GATA3",
        "KRT8",
        "KRT18",
        "KRT19",
        "NIBAN1",
        "PGR",
        "SCGB2A2",
        "SORL1",
        "TFF3",
        "XBP1",
    ),
}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(gmi_output_dir: Path) -> ContourGmiDataset:
    design_path = gmi_output_dir / "design_matrix.tsv.gz"
    sample_path = gmi_output_dir / "sample_metadata.tsv"
    feature_path = gmi_output_dir / "feature_metadata.tsv"
    if not design_path.exists():
        raise FileNotFoundError(f"Missing GMI design matrix: {design_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing GMI sample metadata: {sample_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing GMI feature metadata: {feature_path}")

    X = pd.read_csv(design_path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    sample_metadata = pd.read_csv(sample_path, sep="\t")
    sample_metadata["sample_id"] = sample_metadata["sample_id"].astype(str)
    feature_metadata = pd.read_csv(feature_path, sep="\t")
    if "feature" in feature_metadata.columns:
        feature_metadata["feature"] = feature_metadata["feature"].astype(str)
    y = sample_metadata.set_index("sample_id").loc[X.index, "y"].astype(int)
    return ContourGmiDataset(
        X=X,
        y=y,
        sample_metadata=sample_metadata,
        feature_metadata=feature_metadata,
        config=_read_json(gmi_output_dir / "dataset_config.json"),
        provenance=_read_json(gmi_output_dir / "provenance.json"),
    )


def _as_config(config: GmiModuleConfig | None) -> GmiModuleConfig:
    return config or GmiModuleConfig()


def _feature_meta_map(dataset: ContourGmiDataset) -> pd.DataFrame:
    frame = dataset.feature_metadata.copy()
    if frame.empty:
        frame = pd.DataFrame({"feature": dataset.X.columns.astype(str)})
    if "feature" not in frame.columns:
        frame["feature"] = dataset.X.columns.astype(str)
    frame["feature"] = frame["feature"].astype(str)
    if "feature_block" not in frame.columns:
        frame["feature_block"] = "unknown"
    if "feature_group" not in frame.columns:
        frame["feature_group"] = frame["feature_block"].astype(str)
    return frame.drop_duplicates("feature", keep="first").set_index("feature", drop=False)


def _effect_tables(
    gmi_output_dir: Path | None,
    *,
    main_effects: pd.DataFrame | None,
    interaction_effects: pd.DataFrame | None,
    stability: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if gmi_output_dir is None:
        return (
            main_effects if main_effects is not None else pd.DataFrame(),
            interaction_effects if interaction_effects is not None else pd.DataFrame(),
            stability if stability is not None else pd.DataFrame(),
            predictions if predictions is not None else pd.DataFrame(),
        )
    return (
        main_effects if main_effects is not None else _read_tsv(gmi_output_dir / "main_effects.tsv"),
        interaction_effects if interaction_effects is not None else _read_tsv(gmi_output_dir / "interaction_effects.tsv"),
        stability if stability is not None else _read_tsv(gmi_output_dir / "stability.tsv"),
        predictions if predictions is not None else _read_tsv(gmi_output_dir / "predictions.tsv"),
    )


def _clean_gene_name(feature: str) -> str:
    text = str(feature).split("__")[-1]
    text = re.sub(r"\.\d+$", "", text)
    return text.upper()


def _corr_with_vector(X: pd.DataFrame, vector: pd.Series) -> pd.Series:
    values = X.astype(float)
    aligned = pd.to_numeric(vector.reindex(values.index), errors="coerce").astype(float)
    matrix = values.to_numpy(dtype=float)
    v = aligned.to_numpy(dtype=float)
    v = v - np.nanmean(v)
    matrix = matrix - np.nanmean(matrix, axis=0)
    numerator = np.nansum(matrix * v[:, None], axis=0)
    denom = np.sqrt(np.nansum(matrix * matrix, axis=0) * np.nansum(v * v))
    corr = np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=float), where=denom > 0)
    return pd.Series(corr, index=values.columns.astype(str), dtype=float)


def _spatial_weights(sample_metadata: pd.DataFrame, sample_ids: pd.Index, k: int) -> np.ndarray:
    metadata = sample_metadata.set_index("sample_id", drop=False).reindex(sample_ids.astype(str))
    if not {"x_centroid", "y_centroid"}.issubset(metadata.columns):
        return np.zeros((len(sample_ids), len(sample_ids)), dtype=float)
    coords = metadata[["x_centroid", "y_centroid"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if len(coords) <= 1 or np.isnan(coords).any():
        return np.zeros((len(sample_ids), len(sample_ids)), dtype=float)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    n_neighbors = max(1, min(int(k), len(sample_ids) - 1))
    weights = np.zeros_like(distances, dtype=float)
    for row in range(distances.shape[0]):
        order = np.argsort(distances[row])[:n_neighbors]
        weights[row, order] = 1.0
    return np.maximum(weights, weights.T)


def _spatial_lag(X: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    if weights.size == 0 or weights.sum() <= 0:
        return pd.DataFrame(0.0, index=X.index, columns=X.columns)
    row_sum = weights.sum(axis=1)
    normalized = np.divide(weights, row_sum[:, None], out=np.zeros_like(weights), where=row_sum[:, None] > 0)
    lag = normalized @ X.astype(float).to_numpy(dtype=float)
    return pd.DataFrame(lag, index=X.index, columns=X.columns)


def _spatial_autocorr(values: pd.Series, weights: np.ndarray) -> tuple[float | None, float | None]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    if weights.size == 0 or weights.sum() <= 0 or ok.sum() <= 2:
        return None, None
    x = x.copy()
    x[~ok] = np.nanmean(x[ok])
    centered = x - x.mean()
    denom = float(np.sum(centered * centered))
    if denom <= 0:
        return None, None
    w_sum = float(weights.sum())
    moran = float((len(x) / w_sum) * np.sum(weights * np.outer(centered, centered)) / denom)
    geary = float(((len(x) - 1) / (2.0 * w_sum)) * np.sum(weights * ((x[:, None] - x[None, :]) ** 2)) / denom)
    return moran, geary


def _label_names(dataset: ContourGmiDataset) -> tuple[str, str]:
    metadata = dataset.sample_metadata.set_index("sample_id", drop=False).reindex(dataset.X.index.astype(str))
    if "label" not in metadata.columns:
        return "positive", "negative"
    positive = metadata.loc[dataset.y == 1, "label"].dropna().astype(str)
    negative = metadata.loc[dataset.y == 0, "label"].dropna().astype(str)
    return (
        str(positive.mode().iloc[0]) if not positive.empty else "positive",
        str(negative.mode().iloc[0]) if not negative.empty else "negative",
    )


def _mean_difference(dataset: ContourGmiDataset, feature: str) -> float:
    values = dataset.X[str(feature)].astype(float)
    return float(values.loc[dataset.y == 1].mean() - values.loc[dataset.y == 0].mean())


def _selection_frequency(stability: pd.DataFrame, feature: str) -> float:
    if stability.empty or "member" not in stability.columns or "selection_frequency" not in stability.columns:
        return 0.0
    effect_type = stability["effect_type"].astype(str) if "effect_type" in stability.columns else pd.Series("main", index=stability.index)
    subset = stability.loc[
        (effect_type == "main") & (stability["member"].astype(str) == str(feature))
    ]
    if subset.empty:
        return 0.0
    return float(pd.to_numeric(subset["selection_frequency"], errors="coerce").fillna(0.0).max())


def _anchor_table(
    dataset: ContourGmiDataset,
    main_effects: pd.DataFrame,
    stability: pd.DataFrame,
    config: GmiModuleConfig,
) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    if not main_effects.empty and "feature" in main_effects.columns:
        for _, row in main_effects.iterrows():
            feature = str(row.get("feature", ""))
            if feature not in dataset.X.columns:
                continue
            coefficient = float(pd.to_numeric(pd.Series([row.get("coefficient", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            if abs(coefficient) < float(config.min_abs_anchor_coefficient):
                continue
            rows[feature] = {
                "feature": feature,
                "coefficient": coefficient,
                "selection_frequency": _selection_frequency(stability, feature),
                "anchor_source": "main_effect",
            }
    if not stability.empty and {"member", "selection_frequency"}.issubset(stability.columns):
        for _, row in stability.iterrows():
            if str(row.get("effect_type", "main")) != "main":
                continue
            feature = str(row.get("member", ""))
            if feature not in dataset.X.columns or feature in rows:
                continue
            frequency = float(pd.to_numeric(pd.Series([row.get("selection_frequency", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            if frequency < float(config.min_stability_frequency):
                continue
            rows[feature] = {
                "feature": feature,
                "coefficient": _mean_difference(dataset, feature),
                "selection_frequency": frequency,
                "anchor_source": "stability",
            }
    return pd.DataFrame(rows.values())


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_gmi_effect_graph(
    dataset: ContourGmiDataset,
    *,
    main_effects: pd.DataFrame | None = None,
    interaction_effects: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    config: GmiModuleConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a lightweight GMI effect graph from selected effects and feature-space support."""

    config = _as_config(config)
    main_effects = main_effects if main_effects is not None else pd.DataFrame()
    interaction_effects = interaction_effects if interaction_effects is not None else pd.DataFrame()
    stability = stability if stability is not None else pd.DataFrame()
    feature_meta = _feature_meta_map(dataset)
    anchors = _anchor_table(dataset, main_effects, stability, config)
    anchor_set = set(anchors["feature"].astype(str)) if not anchors.empty else set()

    nodes = feature_meta.reset_index(drop=True).copy()
    nodes["is_anchor"] = nodes["feature"].astype(str).isin(anchor_set)
    coef_map = anchors.set_index("feature")["coefficient"].to_dict() if not anchors.empty else {}
    freq_map = anchors.set_index("feature")["selection_frequency"].to_dict() if not anchors.empty else {}
    source_map = anchors.set_index("feature")["anchor_source"].to_dict() if not anchors.empty else {}
    nodes["coefficient"] = nodes["feature"].map(coef_map).fillna(0.0)
    nodes["selection_frequency"] = nodes["feature"].map(freq_map).fillna(0.0)
    nodes["anchor_source"] = nodes["feature"].map(source_map).fillna("")

    edges: list[dict[str, Any]] = []
    if not interaction_effects.empty and {"feature_a", "feature_b"}.issubset(interaction_effects.columns):
        for _, row in interaction_effects.iterrows():
            feature_a = str(row.get("feature_a", ""))
            feature_b = str(row.get("feature_b", ""))
            if feature_a in dataset.X.columns and feature_b in dataset.X.columns:
                coefficient = float(pd.to_numeric(pd.Series([row.get("coefficient", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                edges.append(
                    {
                        "source": feature_a,
                        "target": feature_b,
                        "edge_type": "gmi_interaction",
                        "weight": abs(coefficient),
                        "coefficient": coefficient,
                    }
                )
    for anchor in sorted(anchor_set):
        corr = _corr_with_vector(dataset.X, dataset.X[anchor])
        selected = corr.loc[corr.abs() >= float(config.expansion_correlation)].drop(labels=[anchor], errors="ignore")
        for target, value in selected.items():
            edges.append(
                {
                    "source": anchor,
                    "target": str(target),
                    "edge_type": "feature_correlation",
                    "weight": abs(float(value)),
                    "coefficient": float(value),
                }
            )
    return {
        "nodes": nodes.sort_values(["is_anchor", "selection_frequency", "feature"], ascending=[False, False, True]).reset_index(drop=True),
        "edges": pd.DataFrame(edges, columns=["source", "target", "edge_type", "weight", "coefficient"]),
    }


def _anchor_groups(
    dataset: ContourGmiDataset,
    anchors: pd.DataFrame,
    interaction_effects: pd.DataFrame,
    config: GmiModuleConfig,
) -> list[list[str]]:
    if anchors.empty:
        return []
    anchor_ids = anchors["feature"].astype(str).tolist()
    uf = _UnionFind(anchor_ids)
    direction = {feature: math.copysign(1.0, _mean_difference(dataset, feature) or 1.0) for feature in anchor_ids}
    for i, anchor_a in enumerate(anchor_ids):
        corr = _corr_with_vector(dataset.X.loc[:, anchor_ids], dataset.X[anchor_a])
        for anchor_b in anchor_ids[i + 1 :]:
            if direction[anchor_a] == direction[anchor_b] and abs(float(corr.loc[anchor_b])) >= float(config.anchor_merge_correlation):
                uf.union(anchor_a, anchor_b)
    if not interaction_effects.empty and {"feature_a", "feature_b"}.issubset(interaction_effects.columns):
        for _, row in interaction_effects.iterrows():
            a = str(row.get("feature_a", ""))
            b = str(row.get("feature_b", ""))
            if a in anchor_ids and b in anchor_ids and direction[a] == direction[b]:
                uf.union(a, b)
    grouped: dict[str, list[str]] = {}
    for anchor in anchor_ids:
        grouped.setdefault(uf.find(anchor), []).append(anchor)
    return [sorted(values) for values in grouped.values()]


def _interaction_partners(interaction_effects: pd.DataFrame, anchors: set[str], columns: pd.Index) -> dict[str, set[str]]:
    partners = {anchor: set() for anchor in anchors}
    if interaction_effects.empty or not {"feature_a", "feature_b"}.issubset(interaction_effects.columns):
        return partners
    available = set(columns.astype(str))
    for _, row in interaction_effects.iterrows():
        a = str(row.get("feature_a", ""))
        b = str(row.get("feature_b", ""))
        if a in anchors and b in available:
            partners[a].add(b)
        if b in anchors and a in available:
            partners[b].add(a)
    return partners


def _module_feature_rows(
    dataset: ContourGmiDataset,
    anchors: pd.DataFrame,
    interaction_effects: pd.DataFrame,
    stability: pd.DataFrame,
    config: GmiModuleConfig,
) -> pd.DataFrame:
    groups = _anchor_groups(dataset, anchors, interaction_effects, config)
    if not groups:
        return pd.DataFrame()
    feature_meta = _feature_meta_map(dataset)
    weights = _spatial_weights(dataset.sample_metadata, dataset.X.index, config.spatial_neighbor_k)
    lag = _spatial_lag(dataset.X, weights)
    rows: list[dict[str, Any]] = []
    anchor_info = anchors.set_index("feature").to_dict(orient="index")
    all_anchors = set(anchors["feature"].astype(str))
    partners_by_anchor = _interaction_partners(interaction_effects, all_anchors, dataset.X.columns)

    for module_index, group in enumerate(groups, start=1):
        module_id = f"{config.module_prefix}_{module_index:03d}"
        corr_by_feature = pd.Series(0.0, index=dataset.X.columns.astype(str))
        lag_corr_by_feature = pd.Series(0.0, index=dataset.X.columns.astype(str))
        signed_direction = pd.Series(1.0, index=dataset.X.columns.astype(str))
        for anchor in group:
            corr = _corr_with_vector(dataset.X, dataset.X[anchor])
            lag_corr = _corr_with_vector(lag, dataset.X[anchor])
            replace = corr.abs() > corr_by_feature.abs()
            corr_by_feature.loc[replace] = corr.loc[replace]
            signed_direction.loc[replace] = np.sign(corr.loc[replace]).replace(0, 1)
            replace_lag = lag_corr.abs() > lag_corr_by_feature.abs()
            lag_corr_by_feature.loc[replace_lag] = lag_corr.loc[replace_lag]

        required = set(group)
        for anchor in group:
            required.update(partners_by_anchor.get(anchor, set()))
        candidates = []
        for feature in dataset.X.columns.astype(str):
            meta = feature_meta.loc[feature]
            if not config.include_spatial_features and str(meta.get("feature_block", "")) != "rna":
                continue
            corr_abs = abs(float(corr_by_feature.loc[feature]))
            lag_abs = abs(float(lag_corr_by_feature.loc[feature]))
            if feature in required or corr_abs >= config.expansion_correlation or lag_abs >= config.expansion_spatial_lag_correlation:
                evidence = []
                if feature in group:
                    evidence.append("anchor")
                if feature in required and feature not in group:
                    evidence.append("gmi_interaction_partner")
                if corr_abs >= config.expansion_correlation and feature not in group:
                    evidence.append("feature_correlation")
                if lag_abs >= config.expansion_spatial_lag_correlation and feature not in group:
                    evidence.append("spatial_lag_correlation")
                rank_score = (
                    corr_abs
                    + lag_abs
                    + (2.0 if feature in group else 0.0)
                    + (1.0 if feature in required and feature not in group else 0.0)
                    + _selection_frequency(stability, feature)
                )
                candidates.append((feature, rank_score, evidence))

        candidates = sorted(candidates, key=lambda item: (item[0] not in required, -item[1], item[0]))
        selected = candidates[: max(int(config.max_features_per_module), len(required))]
        for feature, _, evidence in selected:
            meta = feature_meta.loc[feature]
            coefficient = float(anchor_info.get(feature, {}).get("coefficient", 0.0))
            if not coefficient:
                coefficient = _mean_difference(dataset, feature)
            direction = 1.0 if feature in group else float(np.sign(corr_by_feature.loc[feature]) or 1.0)
            rows.append(
                {
                    "module_id": module_id,
                    "feature": feature,
                    "feature_block": str(meta.get("feature_block", "unknown")),
                    "feature_group": str(meta.get("feature_group", str(meta.get("feature_block", "unknown")))),
                    "role": ";".join(evidence) if evidence else "expanded",
                    "coefficient": coefficient,
                    "selection_frequency": _selection_frequency(stability, feature),
                    "corr_to_anchor": float(corr_by_feature.loc[feature]),
                    "spatial_lag_corr_to_anchor": float(lag_corr_by_feature.loc[feature]),
                    "direction_sign": direction,
                    "weight": np.nan,
                }
            )
    return pd.DataFrame(rows)


def score_gmi_modules(
    dataset: ContourGmiDataset,
    module_features: pd.DataFrame,
    *,
    config: GmiModuleConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score GMI modules per contour and orient high scores toward the enriched endpoint."""

    config = _as_config(config)
    if module_features.empty:
        return pd.DataFrame(index=dataset.X.index), module_features.copy(), pd.DataFrame()
    positive_label, negative_label = _label_names(dataset)
    scores = pd.DataFrame(index=dataset.X.index)
    feature_rows: list[pd.DataFrame] = []
    module_rows: list[dict[str, Any]] = []
    for module_id, group in module_features.groupby("module_id", sort=False):
        features = [feature for feature in group["feature"].astype(str).tolist() if feature in dataset.X.columns]
        if not features:
            continue
        current = group.loc[group["feature"].astype(str).isin(features)].copy()
        weight_abs = np.ones(len(current), dtype=float) / max(1, len(current))
        signed = current["direction_sign"].fillna(1.0).astype(float).to_numpy() * weight_abs
        raw_score = dataset.X.loc[:, features].astype(float).to_numpy(dtype=float) @ signed
        score = pd.Series(raw_score, index=dataset.X.index, dtype=float)
        orientation = 1.0
        current["weight"] = signed
        feature_rows.append(current)
        scores[str(module_id)] = score
        final_positive = float(score.loc[dataset.y == 1].mean())
        final_negative = float(score.loc[dataset.y == 0].mean())
        direction_label = positive_label if final_positive >= final_negative else negative_label
        module_rows.append(
            {
                "module_id": str(module_id),
                "anchor_features": ",".join(current.loc[current["role"].str.contains("anchor", na=False), "feature"].astype(str)),
                "direction_label": direction_label,
                "direction": "positive" if direction_label == positive_label else "negative",
                "mean_positive": final_positive,
                "mean_negative": final_negative,
                "score_delta_positive_minus_negative": final_positive - final_negative,
                "score_high_label": direction_label,
                "score_orientation": orientation,
                "n_features": int(len(current)),
                "n_rna_features": int((current["feature_block"].astype(str) == "rna").sum()),
                "n_spatial_features": int((current["feature_block"].astype(str) == "spatial").sum()),
                "config_module_prefix": config.module_prefix,
            }
        )
    return scores, pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame(), pd.DataFrame(module_rows)


def _bh_qvalues(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    prev = 1.0
    for rank, index in enumerate(order[::-1], start=1):
        original_rank = len(p) - rank + 1
        value = min(prev, p[index] * len(p) / max(1, original_rank))
        ranked[index] = value
        prev = value
    return ranked.tolist()


def _module_enrichment(dataset: ContourGmiDataset, module_features: pd.DataFrame) -> pd.DataFrame:
    if module_features.empty:
        return pd.DataFrame(columns=["module_id", "gene_set", "overlap_count", "overlap_genes", "p_value", "q_value"])
    feature_meta = _feature_meta_map(dataset)
    background = {
        _clean_gene_name(feature)
        for feature in feature_meta.loc[feature_meta["feature_block"].astype(str) == "rna", "feature"].astype(str)
    }
    if not background:
        return pd.DataFrame(columns=["module_id", "gene_set", "overlap_count", "overlap_genes", "p_value", "q_value"])
    rows: list[dict[str, Any]] = []
    for module_id, group in module_features.groupby("module_id", sort=False):
        genes = {
            _clean_gene_name(feature)
            for feature in group.loc[group["feature_block"].astype(str) == "rna", "feature"].astype(str)
        }
        for gene_set, members in CURATED_GMI_GENE_SETS.items():
            member_set = set(members) & background
            overlap = sorted(genes & member_set)
            p_value = 1.0
            if member_set and genes:
                p_value = float(hypergeom.sf(max(0, len(overlap) - 1), len(background), len(member_set), len(genes)))
            rows.append(
                {
                    "module_id": str(module_id),
                    "gene_set": gene_set,
                    "overlap_count": int(len(overlap)),
                    "overlap_genes": ",".join(overlap),
                    "p_value": p_value,
                }
            )
    q_values = _bh_qvalues([row["p_value"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value"] = float(q_value)
    return pd.DataFrame(rows).sort_values(["module_id", "q_value", "overlap_count"], ascending=[True, True, False])


def _module_interactions(module_features: pd.DataFrame, interaction_effects: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "module_id_a",
        "module_id_b",
        "relation",
        "feature_a",
        "feature_b",
        "interaction",
        "coefficient",
    ]
    if module_features.empty or interaction_effects.empty or not {"feature_a", "feature_b"}.issubset(interaction_effects.columns):
        return pd.DataFrame(columns=columns)
    memberships = module_features.groupby("feature")["module_id"].apply(lambda values: sorted(set(values.astype(str)))).to_dict()
    rows: list[dict[str, Any]] = []
    for _, row in interaction_effects.iterrows():
        a = str(row.get("feature_a", ""))
        b = str(row.get("feature_b", ""))
        for module_a in memberships.get(a, []):
            for module_b in memberships.get(b, []):
                rows.append(
                    {
                        "module_id_a": module_a,
                        "module_id_b": module_b,
                        "relation": "within_module" if module_a == module_b else "between_module",
                        "feature_a": a,
                        "feature_b": b,
                        "interaction": str(row.get("interaction", f"{a}^{b}")),
                        "coefficient": float(pd.to_numeric(pd.Series([row.get("coefficient", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _module_spatial_autocorr(dataset: ContourGmiDataset, module_scores: pd.DataFrame, config: GmiModuleConfig) -> pd.DataFrame:
    weights = _spatial_weights(dataset.sample_metadata, dataset.X.index, config.spatial_neighbor_k)
    rows = []
    for module_id in module_scores.columns.astype(str):
        moran, geary = _spatial_autocorr(module_scores[module_id], weights)
        rows.append({"module_id": module_id, "moran_i": moran, "geary_c": geary, "spatial_neighbor_k": int(config.spatial_neighbor_k)})
    return pd.DataFrame(rows)


def _summarize_modules(
    modules: pd.DataFrame,
    enrichment: pd.DataFrame,
    autocorr: pd.DataFrame,
) -> pd.DataFrame:
    if modules.empty:
        return modules
    modules = modules.copy()
    top_gene_sets = {}
    if not enrichment.empty:
        filtered = enrichment.loc[enrichment["overlap_count"] > 0].copy()
        for module_id, group in filtered.groupby("module_id", sort=False):
            top_gene_sets[str(module_id)] = ",".join(group.sort_values(["q_value", "overlap_count"], ascending=[True, False]).head(3)["gene_set"].astype(str))
    modules["top_gene_sets"] = modules["module_id"].astype(str).map(top_gene_sets).fillna("")
    if not autocorr.empty:
        modules = modules.merge(autocorr[["module_id", "moran_i", "geary_c"]], on="module_id", how="left")
    return modules


def _write_module_figures(
    dataset: ContourGmiDataset,
    module_scores: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    if module_scores.empty:
        return {}
    from ._workflow import _plot_contour_values

    files: dict[str, str] = {}
    manifest: list[dict[str, str]] = []
    figure_dir = output_dir / "figures"
    metadata = dataset.sample_metadata.copy()
    for module_id in module_scores.columns.astype(str):
        path = figure_dir / f"{module_id}_score_map.png"
        _plot_contour_values(
            metadata,
            module_scores[module_id],
            output_path=path,
            title=f"{module_id} spatial module score",
            cmap="magma",
        )
        key = f"figure_{module_id}_score_map"
        files[key] = str(path)
        manifest.append({"name": key, "path": str(path), "description": f"Contour-level score for {module_id}."})
    if manifest:
        manifest_path = figure_dir / "module_visualization_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files["module_visualization_manifest"] = str(manifest_path)
    return files


def render_gmi_module_report(summary: Mapping[str, Any]) -> str:
    """Render a concise Markdown report for GMI spatial modules."""

    lines = [
        "# GMI Spatial Gene Modules",
        "",
        f"GMI output: `{summary.get('gmi_output_dir', 'in_memory')}`",
        f"Output directory: `{summary.get('output_dir')}`",
        "",
        "## Summary",
        "",
        f"- Modules discovered: `{summary.get('n_modules', 0)}`",
        f"- Anchors used: `{summary.get('n_anchor_features', 0)}`",
        f"- Module features: `{summary.get('n_module_features', 0)}`",
        "",
    ]
    top_modules = summary.get("top_modules", [])
    if top_modules:
        lines.extend(["## Modules", ""])
        for item in top_modules:
            lines.append(
                f"- `{item.get('module_id')}`: `{item.get('direction_label')}`; "
                f"anchors `{item.get('anchor_features', '')}`; "
                f"top gene sets `{item.get('top_gene_sets', '')}`; "
                f"Moran's I `{item.get('moran_i')}`"
            )
        lines.append("")
    if summary.get("interpretation"):
        lines.extend(["## Interpretation", "", str(summary["interpretation"]), ""])
    return "\n".join(lines)


def discover_gmi_modules(
    gmi_output_dir: str | Path | None = None,
    *,
    dataset: ContourGmiDataset | None = None,
    output_dir: str | Path | None = None,
    config: GmiModuleConfig | None = None,
    main_effects: pd.DataFrame | None = None,
    interaction_effects: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
) -> GmiModuleResult:
    """Discover supervised spatial gene modules from existing contour-GMI artifacts."""

    config = _as_config(config)
    input_dir = Path(gmi_output_dir) if gmi_output_dir is not None else None
    if dataset is None:
        if input_dir is None:
            raise ValueError("Either `gmi_output_dir` or `dataset` must be provided.")
        dataset = _load_dataset(input_dir)
    out = Path(output_dir) if output_dir is not None else (input_dir / "modules" if input_dir is not None else Path("gmi_modules"))
    out.mkdir(parents=True, exist_ok=True)

    main_effects, interaction_effects, stability, predictions = _effect_tables(
        input_dir,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        stability=stability,
        predictions=predictions,
    )
    effect_graph = build_gmi_effect_graph(
        dataset,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        stability=stability,
        config=config,
    )
    anchors = _anchor_table(dataset, main_effects, stability, config)
    module_features = _module_feature_rows(dataset, anchors, interaction_effects, stability, config)
    module_scores, module_features, modules = score_gmi_modules(dataset, module_features, config=config)
    enrichment = _module_enrichment(dataset, module_features)
    module_interactions = _module_interactions(module_features, interaction_effects)
    autocorr = _module_spatial_autocorr(dataset, module_scores, config)
    modules = _summarize_modules(modules, enrichment, autocorr)

    files = {
        "spatial_modules": str(out / "spatial_modules.tsv"),
        "module_features": str(out / "module_features.tsv"),
        "module_scores": str(out / "module_scores.tsv.gz"),
        "module_enrichment": str(out / "module_enrichment.tsv"),
        "module_interactions": str(out / "module_interactions.tsv"),
        "module_spatial_autocorr": str(out / "module_spatial_autocorr.tsv"),
        "effect_graph_nodes": str(out / "effect_graph_nodes.tsv"),
        "effect_graph_edges": str(out / "effect_graph_edges.tsv"),
        "summary_json": str(out / "summary.json"),
        "report_md": str(out / "report.md"),
    }
    if config.write_figures:
        files.update(_write_module_figures(dataset, module_scores, out))

    modules.to_csv(files["spatial_modules"], sep="\t", index=False)
    module_features.to_csv(files["module_features"], sep="\t", index=False)
    module_scores.to_csv(files["module_scores"], sep="\t", compression="gzip", index_label="sample_id")
    enrichment.to_csv(files["module_enrichment"], sep="\t", index=False)
    module_interactions.to_csv(files["module_interactions"], sep="\t", index=False)
    autocorr.to_csv(files["module_spatial_autocorr"], sep="\t", index=False)
    effect_graph["nodes"].to_csv(files["effect_graph_nodes"], sep="\t", index=False)
    effect_graph["edges"].to_csv(files["effect_graph_edges"], sep="\t", index=False)

    interpretation = (
        "Modules are supervised GMI modules: selected or stable GMI effects seed each module, "
        "then correlated, spatial-lag-correlated, and GMI-interacting features expand the module. "
        "Module scores are oriented so higher values mark the endpoint in `score_high_label`."
    )
    summary = {
        "gmi_output_dir": str(input_dir) if input_dir is not None else None,
        "output_dir": str(out),
        "n_modules": int(len(modules)),
        "n_anchor_features": int(len(anchors)),
        "n_module_features": int(len(module_features)),
        "n_module_interactions": int(len(module_interactions)),
        "config": config.to_dict(),
        "top_modules": modules.head(20).to_dict(orient="records") if not modules.empty else [],
        "interpretation": interpretation,
        "files": files,
    }
    Path(files["summary_json"]).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    Path(files["report_md"]).write_text(render_gmi_module_report(summary), encoding="utf-8")

    return GmiModuleResult(
        output_dir=out,
        spatial_modules=modules,
        module_features=module_features,
        module_scores=module_scores,
        module_enrichment=enrichment,
        module_interactions=module_interactions,
        module_spatial_autocorr=autocorr,
        effect_graph_nodes=effect_graph["nodes"],
        effect_graph_edges=effect_graph["edges"],
        summary=summary,
        files=files,
    )
