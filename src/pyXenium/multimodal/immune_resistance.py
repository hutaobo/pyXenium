"""Spatial RNA + protein immune-resistance analysis for Xenium datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from pyXenium.utils.name_resolver import resolve_protein_column


@dataclass(frozen=True)
class MarkerPair:
    protein: str
    gene: str
    label: str


DEFAULT_MARKER_PAIRS: tuple[MarkerPair, ...] = (
    MarkerPair("PD-1", "PDCD1", "pd1"),
    MarkerPair("PD-L1", "CD274", "pd_l1"),
    MarkerPair("VISTA", "VSIR", "vista"),
    MarkerPair("LAG-3", "LAG3", "lag3"),
    MarkerPair("CD68", "CD68", "cd68"),
    MarkerPair("CD163", "CD163", "cd163"),
    MarkerPair("HLA-DR", "HLA-DRA", "hla_dr"),
    MarkerPair("CD31", "PECAM1", "cd31"),
    MarkerPair("alphaSMA", "ACTA2", "alpha_sma"),
    MarkerPair("PanCK", "EPCAM", "panck"),
    MarkerPair("E-Cadherin", "CDH1", "e_cadherin"),
    MarkerPair("Vimentin", "VIM", "vimentin"),
    MarkerPair("GranzymeB", "GZMB", "granzymeb"),
    MarkerPair("CD20", "MS4A1", "cd20"),
    MarkerPair("CD138", "SDC1", "cd138"),
)

DEFAULT_STATE_SIGNATURES: dict[str, dict[str, list[str]]] = {
    "tumor_epithelial": {
        "protein": ["PanCK", "E-Cadherin"],
        "rna": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    },
    "emt_like_tumor": {
        "protein": ["Vimentin", "alphaSMA"],
        "rna": ["VIM", "ACTA2", "TAGLN", "COL1A1", "COL1A2"],
    },
    "endothelial_perivascular": {
        "protein": ["CD31", "alphaSMA"],
        "rna": ["PECAM1", "ACKR1", "EMCN", "ACTA2", "RGS5"],
    },
    "macrophage_like": {
        "protein": ["CD68", "CD163", "HLA-DR", "CD11c", "CD45"],
        "rna": ["CD68", "CD163", "HLA-DRA", "HLA-DRB1", "LST1", "TYROBP"],
    },
    "t_cell_exhausted_cytotoxic": {
        "protein": ["CD3E", "CD8A", "PD-1", "LAG-3", "GranzymeB", "CD45RO"],
        "rna": ["CD3D", "CD3E", "TRAC", "CD8A", "PDCD1", "LAG3", "PRF1", "GZMB", "GZMK", "CCL5"],
    },
    "b_plasma_like": {
        "protein": ["CD20", "CD138"],
        "rna": ["MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "SDC1"],
    },
}

DEFAULT_STATE_HIERARCHY: dict[str, str] = {
    "tumor_epithelial": "tumor",
    "emt_like_tumor": "tumor",
    "endothelial_perivascular": "stromal",
    "macrophage_like": "immune",
    "t_cell_exhausted_cytotoxic": "immune",
    "b_plasma_like": "immune",
}

DEFAULT_PATHWAY_MARKERS: dict[str, list[str]] = {
    "checkpoint": ["pd1", "pd_l1", "vista", "lag3"],
    "myeloid_activation": ["cd68", "cd163", "hla_dr"],
    "vascular_stromal": ["cd31", "alpha_sma"],
    "epithelial_emt": ["panck", "e_cadherin", "vimentin"],
    "lymphoid_effector": ["granzymeb", "cd20", "cd138"],
}

DEFAULT_RESISTANT_NICHES: tuple[str, ...] = ("myeloid_vascular", "epithelial_emt_front")

DEFAULT_BRANCH_MODELS: dict[str, tuple[str, ...]] = {
    "myeloid_vascular": (
        "myeloid_vascular_branch",
        "joint_minus_emt",
        "myeloid_only",
        "vascular_only",
        "joint_activity",
        "rna_only",
        "protein_only",
    ),
    "epithelial_emt_front": (
        "epithelial_emt_front_branch",
        "joint_minus_vascular",
        "emt_only",
        "checkpoint_only",
        "joint_activity",
        "rna_only",
        "protein_only",
    ),
}


def _get_coords(
    adata: AnnData,
    spatial_obsm: str = "spatial",
    obs_xy: tuple[str, str] = ("x_centroid", "y_centroid"),
) -> np.ndarray:
    if spatial_obsm in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_obsm], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"adata.obsm['{spatial_obsm}'] must be of shape (n_cells, >=2).")
        return coords[:, :2]

    x_key, y_key = obs_xy
    if x_key in adata.obs.columns and y_key in adata.obs.columns:
        return adata.obs.loc[:, [x_key, y_key]].to_numpy(dtype=np.float32)

    raise KeyError(f"Could not find coordinates in adata.obsm['{spatial_obsm}'] or adata.obs[{obs_xy!r}].")


def _get_rna_matrix(adata: AnnData):
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def _gene_lookup(adata: AnnData) -> dict[str, list[int]]:
    names = adata.var["name"].astype(str).tolist() if "name" in adata.var.columns else adata.var_names.astype(str).tolist()
    lookup: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        lookup.setdefault(name.casefold(), []).append(idx)
    return lookup


def _protein_frame(adata: AnnData, protein_obsm: str = "protein") -> pd.DataFrame:
    if protein_obsm not in adata.obsm:
        raise KeyError(f"AnnData is missing adata.obsm['{protein_obsm}'].")
    protein = adata.obsm[protein_obsm]
    if isinstance(protein, pd.DataFrame):
        return protein
    return pd.DataFrame(np.asarray(protein), index=adata.obs_names)


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype=np.float32)

    med = np.nanmedian(arr[finite])
    mad = np.nanmedian(np.abs(arr[finite] - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        std = np.nanstd(arr[finite])
        scale = std if np.isfinite(std) and std >= 1e-6 else 1.0
        center = np.nanmean(arr[finite])
    else:
        center = med
    out = (arr - center) / scale
    out[~finite] = np.nan
    return out.astype(np.float32)


def _mean_stack(columns: Sequence[np.ndarray]) -> np.ndarray:
    if not columns:
        return np.array([], dtype=np.float32)
    return np.nanmean(np.vstack(columns), axis=0).astype(np.float32)


def _normalised_gene_vector(
    adata: AnnData,
    gene: str,
    *,
    rna_matrix=None,
    gene_lookup: Mapping[str, Sequence[int]] | None = None,
    target_sum: float = 1e4,
) -> np.ndarray | None:
    if rna_matrix is None:
        rna_matrix = _get_rna_matrix(adata)
    if gene_lookup is None:
        gene_lookup = _gene_lookup(adata)

    idxs = list(gene_lookup.get(gene.casefold(), []))
    if not idxs:
        return None

    sub = rna_matrix[:, idxs]
    expr = np.asarray(sub.mean(axis=1)).ravel().astype(np.float32)
    cell_sums = np.asarray(rna_matrix.sum(axis=1)).ravel().astype(np.float32)
    cell_sums[cell_sums == 0] = 1.0
    return np.log1p((expr / cell_sums) * target_sum).astype(np.float32)


def _protein_vector(adata: AnnData, protein: str, *, protein_df: pd.DataFrame | None = None) -> np.ndarray | None:
    protein_df = _protein_frame(adata) if protein_df is None else protein_df
    try:
        resolved = resolve_protein_column(adata, protein, "protein_norm", "protein")
    except Exception:
        resolved = protein
    if resolved not in protein_df.columns:
        return None
    values = protein_df[resolved].to_numpy(dtype=np.float32)
    return np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32)


def _knn_indices(coords: np.ndarray, n_neighbors: int) -> np.ndarray:
    if coords.shape[0] <= 1:
        raise ValueError("At least two cells are required to build a neighbourhood graph.")
    n_neighbors = max(1, min(int(n_neighbors), coords.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="kd_tree")
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)
    return indices[:, 1:]


def _neighbour_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.nanmean(values[indices], axis=1).astype(np.float32)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float32)
    if y_true.size == 0 or y_true.min() == y_true.max():
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    if np.nanstd(a[mask]) < 1e-6 or np.nanstd(b[mask]) < 1e-6:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _assign_obs_columns(adata: AnnData, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    keep = adata.obs.drop(columns=[col for col in frame.columns if col in adata.obs.columns], errors="ignore")
    adata.obs = pd.concat([keep, frame], axis=1)


def _compute_region_assignments(coords: np.ndarray, region_bins: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = coords[:, 0]
    y = coords[:, 1]
    region_bins = max(2, int(region_bins))
    x_edges = np.linspace(float(x.min()), float(x.max()), num=region_bins + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()), num=region_bins + 1)
    x_bin = np.clip(np.digitize(x, x_edges[1:-1]), 0, region_bins - 1)
    y_bin = np.clip(np.digitize(y, y_edges[1:-1]), 0, region_bins - 1)
    return pd.DataFrame({"region_x_bin": x_bin, "region_y_bin": y_bin}), x_edges, y_edges


def _region_table(coords: np.ndarray, values: pd.DataFrame, region_bins: int) -> pd.DataFrame:
    assignments, x_edges, y_edges = _compute_region_assignments(coords, region_bins)
    table = pd.concat([assignments, values.reset_index(drop=True)], axis=1)
    grouped = table.groupby(["region_x_bin", "region_y_bin"], dropna=False)
    summary = grouped.mean(numeric_only=True)
    summary["n_cells"] = grouped.size()
    summary = summary.reset_index()
    summary["region_x_min"] = summary["region_x_bin"].map(lambda i: float(x_edges[int(i)]))
    summary["region_x_max"] = summary["region_x_bin"].map(lambda i: float(x_edges[int(i) + 1]))
    summary["region_y_min"] = summary["region_y_bin"].map(lambda i: float(y_edges[int(i)]))
    summary["region_y_max"] = summary["region_y_bin"].map(lambda i: float(y_edges[int(i) + 1]))
    summary["region_x_center"] = 0.5 * (summary["region_x_min"] + summary["region_x_max"])
    summary["region_y_center"] = 0.5 * (summary["region_y_min"] + summary["region_y_max"])
    return summary


def _cell_roi_means(coords: np.ndarray, values: np.ndarray, region_bins: int) -> np.ndarray:
    assignments, _, _ = _compute_region_assignments(coords, region_bins)
    table = assignments.copy()
    table["value"] = values
    grouped = table.groupby(["region_x_bin", "region_y_bin"], dropna=False)["value"].mean()
    return grouped.loc[list(zip(assignments["region_x_bin"], assignments["region_y_bin"]))].to_numpy(dtype=np.float32)


def _roi_reproducibility(coords: np.ndarray, values: np.ndarray, region_bins: int, alt_bins: int, top_fraction: float = 0.1) -> float:
    cell_a = _cell_roi_means(coords, values, region_bins=region_bins)
    cell_b = _cell_roi_means(coords, values, region_bins=max(region_bins + 4, alt_bins))
    qa = np.nanquantile(cell_a, max(0.5, 1.0 - top_fraction))
    qb = np.nanquantile(cell_b, max(0.5, 1.0 - top_fraction))
    mask_a = cell_a >= qa
    mask_b = cell_b >= qb
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def _metric_to_unit_interval(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(np.clip(0.5 + 0.5 * value, 0.0, 1.0))


def annotate_joint_cell_states(
    adata: AnnData,
    *,
    state_signatures: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    state_hierarchy: Mapping[str, str] | None = None,
    protein_obsm: str = "protein",
    state_key: str = "joint_cell_state",
    class_key: str = "joint_cell_class",
    hierarchy_key: str = "joint_cell_hierarchy",
    confidence_key: str = "joint_cell_state_confidence",
    class_confidence_key: str = "joint_cell_class_confidence",
    min_score: float = -0.25,
    min_margin: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Annotate joint RNA/protein classes and subtypes using hierarchical rules."""

    signatures = DEFAULT_STATE_SIGNATURES if state_signatures is None else dict(state_signatures)
    hierarchy = DEFAULT_STATE_HIERARCHY if state_hierarchy is None else dict(state_hierarchy)
    protein_df = _protein_frame(adata, protein_obsm=protein_obsm)
    rna_matrix = _get_rna_matrix(adata)
    gene_lookup = _gene_lookup(adata)

    subtype_scores: dict[str, np.ndarray] = {}
    available_rows = []
    for subtype, spec in signatures.items():
        protein_scores = []
        rna_scores = []
        available_proteins = []
        available_genes = []

        for protein in spec.get("protein", []):
            vec = _protein_vector(adata, protein, protein_df=protein_df)
            if vec is None:
                continue
            protein_scores.append(_robust_zscore(vec))
            try:
                available_proteins.append(resolve_protein_column(adata, protein, "protein_norm", protein_obsm))
            except Exception:
                available_proteins.append(protein)

        for gene in spec.get("rna", []):
            vec = _normalised_gene_vector(adata, gene, rna_matrix=rna_matrix, gene_lookup=gene_lookup)
            if vec is None:
                continue
            rna_scores.append(_robust_zscore(vec))
            available_genes.append(gene)

        protein_score = _mean_stack(protein_scores) if protein_scores else np.full(adata.n_obs, np.nan, dtype=np.float32)
        rna_score = _mean_stack(rna_scores) if rna_scores else np.full(adata.n_obs, np.nan, dtype=np.float32)
        joint_score = _mean_stack([arr for arr in (protein_score, rna_score) if arr.size > 0]) if (protein_scores or rna_scores) else np.full(adata.n_obs, np.nan, dtype=np.float32)

        subtype_scores[f"{subtype}__protein_score"] = protein_score
        subtype_scores[f"{subtype}__rna_score"] = rna_score
        subtype_scores[f"{subtype}__joint_score"] = joint_score
        available_rows.append(
            {
                "state": subtype,
                "class": hierarchy.get(subtype, "other"),
                "available_proteins": ",".join(available_proteins),
                "available_genes": ",".join(available_genes),
                "n_available_proteins": len(available_proteins),
                "n_available_genes": len(available_genes),
            }
        )

    score_df = pd.DataFrame(subtype_scores, index=adata.obs_names)
    family_to_states: dict[str, list[str]] = {}
    for subtype, family in hierarchy.items():
        family_to_states.setdefault(family, []).append(subtype)

    class_columns = {
        f"{family}__joint_score": _mean_stack([score_df[f"{subtype}__joint_score"].to_numpy(dtype=np.float32) for subtype in subtypes if f"{subtype}__joint_score" in score_df.columns])
        for family, subtypes in family_to_states.items()
    }
    class_df = pd.DataFrame(class_columns, index=adata.obs_names)
    class_matrix = class_df.to_numpy(dtype=np.float32)
    class_filled = np.where(np.isfinite(class_matrix), class_matrix, -np.inf)
    class_best_idx = class_filled.argmax(axis=1)
    class_best_score = class_filled[np.arange(adata.n_obs), class_best_idx]
    sorted_class_scores = np.sort(class_filled, axis=1)
    class_second = sorted_class_scores[:, -2] if class_filled.shape[1] >= 2 else np.full(adata.n_obs, -np.inf, dtype=np.float32)
    class_margin = (class_best_score - class_second).astype(np.float32)
    class_labels = np.array([class_df.columns[i].replace("__joint_score", "") for i in class_best_idx], dtype=object)
    invalid_class = (~np.isfinite(class_best_score)) | (class_best_score < float(min_score)) | (class_margin < float(min_margin))
    class_labels[invalid_class] = "unassigned"
    class_margin[invalid_class] = np.nan

    subtype_labels = np.full(adata.n_obs, "unassigned", dtype=object)
    subtype_confidence = np.full(adata.n_obs, np.nan, dtype=np.float32)
    for family, subtypes in family_to_states.items():
        family_mask = class_labels == family
        if not family_mask.any():
            continue
        family_cols = [f"{subtype}__joint_score" for subtype in subtypes if f"{subtype}__joint_score" in score_df.columns]
        family_scores = score_df.loc[family_mask, family_cols].to_numpy(dtype=np.float32)
        family_filled = np.where(np.isfinite(family_scores), family_scores, -np.inf)
        subtype_best_idx = family_filled.argmax(axis=1)
        subtype_best_score = family_filled[np.arange(family_filled.shape[0]), subtype_best_idx]
        sorted_subtype_scores = np.sort(family_filled, axis=1)
        subtype_second = sorted_subtype_scores[:, -2] if family_filled.shape[1] >= 2 else np.full(family_filled.shape[0], -np.inf, dtype=np.float32)
        subtype_margin = (subtype_best_score - subtype_second).astype(np.float32)
        chosen = np.array([family_cols[i].replace("__joint_score", "") for i in subtype_best_idx], dtype=object)
        invalid_subtype = (~np.isfinite(subtype_best_score)) | (subtype_best_score < float(min_score)) | (subtype_margin < float(min_margin))
        chosen[invalid_subtype] = "unassigned"
        subtype_labels[family_mask] = chosen
        subtype_confidence[family_mask] = subtype_margin

    hierarchy_labels = np.array([f"{cls}::{sub}" if cls != "unassigned" else "unassigned" for cls, sub in zip(class_labels, subtype_labels)], dtype=object)

    obs_updates = pd.DataFrame(index=adata.obs_names)
    obs_updates[class_key] = pd.Categorical(class_labels)
    obs_updates[state_key] = pd.Categorical(subtype_labels)
    obs_updates[hierarchy_key] = pd.Categorical(hierarchy_labels)
    obs_updates[class_confidence_key] = class_margin
    obs_updates[confidence_key] = subtype_confidence
    obs_updates = pd.concat([obs_updates, class_df.astype(np.float32), score_df.astype(np.float32)], axis=1)
    _assign_obs_columns(adata, obs_updates)

    class_summary = pd.Series(class_labels, name=class_key).value_counts(dropna=False).rename_axis("class").reset_index(name="n_cells")
    state_summary = pd.Series(subtype_labels, name=state_key).value_counts(dropna=False).rename_axis("state").reset_index(name="n_cells")
    available_df = pd.DataFrame(available_rows).sort_values(["class", "state"]).reset_index(drop=True)

    adata.uns.setdefault("spatial_immune_resistance", {})
    adata.uns["spatial_immune_resistance"]["state_signatures"] = available_df.to_dict(orient="records")
    adata.uns["spatial_immune_resistance"]["state_hierarchy"] = hierarchy

    return {
        "cell_scores": pd.concat([class_df, score_df], axis=1),
        "class_summary": class_summary,
        "state_summary": state_summary,
        "available_features": available_df,
    }


def compute_rna_protein_discordance(
    adata: AnnData,
    *,
    marker_pairs: Sequence[MarkerPair] | None = None,
    state_signatures: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    pathway_markers: Mapping[str, Sequence[str]] | None = None,
    protein_obsm: str = "protein",
    spatial_obsm: str = "spatial",
    state_key: str = "joint_cell_state",
    prefix: str = "discordance",
    n_neighbors: int = 15,
    region_bins: int = 24,
) -> dict[str, pd.DataFrame]:
    """Compute marker-, state-, and pathway-level RNA/protein discordance scores."""

    if state_key not in adata.obs.columns:
        annotate_joint_cell_states(adata, state_key=state_key, protein_obsm=protein_obsm)

    protein_df = _protein_frame(adata, protein_obsm=protein_obsm)
    rna_matrix = _get_rna_matrix(adata)
    gene_lookup = _gene_lookup(adata)
    coords = _get_coords(adata, spatial_obsm=spatial_obsm)
    pairs = tuple(DEFAULT_MARKER_PAIRS if marker_pairs is None else marker_pairs)
    pathway_specs = DEFAULT_PATHWAY_MARKERS if pathway_markers is None else dict(pathway_markers)
    signatures = DEFAULT_STATE_SIGNATURES if state_signatures is None else dict(state_signatures)

    cell_columns: dict[str, np.ndarray] = {}
    marker_rows = []
    for pair in pairs:
        protein_vec = _protein_vector(adata, pair.protein, protein_df=protein_df)
        rna_vec = _normalised_gene_vector(adata, pair.gene, rna_matrix=rna_matrix, gene_lookup=gene_lookup)
        if protein_vec is None or rna_vec is None:
            continue
        protein_score = _robust_zscore(protein_vec)
        rna_score = _robust_zscore(rna_vec)
        signed = (protein_score - rna_score).astype(np.float32)
        abs_discord = np.abs(signed).astype(np.float32)
        joint = np.nanmean(np.vstack([protein_score, rna_score]), axis=0).astype(np.float32)
        base = f"{prefix}__marker__{pair.label}"
        cell_columns[f"{base}__protein"] = protein_score
        cell_columns[f"{base}__rna"] = rna_score
        cell_columns[f"{base}__signed"] = signed
        cell_columns[f"{base}__abs"] = abs_discord
        cell_columns[f"{base}__joint_activation"] = joint
        marker_rows.append(
            {
                "label": pair.label,
                "protein": pair.protein,
                "gene": pair.gene,
                "mean_signed_discordance": float(np.nanmean(signed)),
                "mean_abs_discordance": float(np.nanmean(abs_discord)),
                "protein_rna_correlation": _safe_corr(protein_score, rna_score),
            }
        )

    score_df = pd.DataFrame(cell_columns, index=adata.obs_names)

    state_rows = []
    for state in signatures:
        protein_col = f"{state}__protein_score"
        rna_col = f"{state}__rna_score"
        if protein_col not in adata.obs.columns or rna_col not in adata.obs.columns:
            continue
        protein_score = adata.obs[protein_col].to_numpy(dtype=np.float32)
        rna_score = adata.obs[rna_col].to_numpy(dtype=np.float32)
        signed = (protein_score - rna_score).astype(np.float32)
        abs_discord = np.abs(signed).astype(np.float32)
        joint = np.nanmean(np.vstack([protein_score, rna_score]), axis=0).astype(np.float32)
        base = f"{prefix}__state__{state}"
        score_df[f"{base}__protein"] = protein_score
        score_df[f"{base}__rna"] = rna_score
        score_df[f"{base}__signed"] = signed
        score_df[f"{base}__abs"] = abs_discord
        score_df[f"{base}__joint_activation"] = joint
        state_rows.append(
            {
                "state": state,
                "mean_signed_discordance": float(np.nanmean(signed)),
                "mean_abs_discordance": float(np.nanmean(abs_discord)),
            }
        )

    pathway_rows = []
    for pathway, labels in pathway_specs.items():
        protein_components = []
        rna_components = []
        for label in labels:
            base = f"{prefix}__marker__{label}"
            protein_col = f"{base}__protein"
            rna_col = f"{base}__rna"
            if protein_col in score_df.columns and rna_col in score_df.columns:
                protein_components.append(score_df[protein_col].to_numpy(dtype=np.float32))
                rna_components.append(score_df[rna_col].to_numpy(dtype=np.float32))
        if not protein_components:
            continue
        protein_score = _mean_stack(protein_components)
        rna_score = _mean_stack(rna_components)
        signed = (protein_score - rna_score).astype(np.float32)
        abs_discord = np.abs(signed).astype(np.float32)
        joint = np.nanmean(np.vstack([protein_score, rna_score]), axis=0).astype(np.float32)
        base = f"{prefix}__pathway__{pathway}"
        score_df[f"{base}__protein"] = protein_score
        score_df[f"{base}__rna"] = rna_score
        score_df[f"{base}__signed"] = signed
        score_df[f"{base}__abs"] = abs_discord
        score_df[f"{base}__joint_activation"] = joint
        pathway_rows.append(
            {
                "pathway": pathway,
                "n_markers": len(protein_components),
                "mean_signed_discordance": float(np.nanmean(signed)),
                "mean_abs_discordance": float(np.nanmean(abs_discord)),
            }
        )

    indices = _knn_indices(coords, n_neighbors=n_neighbors)
    neighborhood_df = pd.DataFrame(
        {
            f"{column}__neighborhood_mean": _neighbour_mean(score_df[column].to_numpy(dtype=np.float32), indices)
            for column in score_df.columns
        },
        index=adata.obs_names,
    )
    region_df = _region_table(coords, pd.concat([score_df, neighborhood_df], axis=1), region_bins=region_bins)
    _assign_obs_columns(adata, score_df.astype(np.float32))

    marker_summary = pd.DataFrame(marker_rows).sort_values("mean_abs_discordance", ascending=False).reset_index(drop=True)
    state_summary = pd.DataFrame(state_rows).sort_values("mean_abs_discordance", ascending=False).reset_index(drop=True)
    pathway_summary = pd.DataFrame(pathway_rows).sort_values("mean_abs_discordance", ascending=False).reset_index(drop=True)

    adata.uns.setdefault("spatial_immune_resistance", {})
    adata.uns["spatial_immune_resistance"]["marker_pairs"] = [pair.__dict__ for pair in pairs]
    adata.uns["spatial_immune_resistance"]["discordance_config"] = {
        "state_key": state_key,
        "prefix": prefix,
        "n_neighbors": int(n_neighbors),
        "region_bins": int(region_bins),
    }

    return {
        "cell_scores": score_df,
        "neighborhood_scores": neighborhood_df,
        "region_scores": region_df,
        "marker_summary": marker_summary,
        "state_summary": state_summary,
        "pathway_summary": pathway_summary,
    }


def build_spatial_niches(
    adata: AnnData,
    *,
    cell_state_key: str = "joint_cell_state",
    spatial_obsm: str = "spatial",
    niche_key: str = "spatial_niche",
    niche_score_key: str = "spatial_niche_score",
    niche_margin_key: str = "spatial_niche_margin",
    n_neighbors: int = 15,
    min_score: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Build spatial niches from local composition of joint cell states."""

    if cell_state_key not in adata.obs.columns:
        annotate_joint_cell_states(adata, state_key=cell_state_key)

    coords = _get_coords(adata, spatial_obsm=spatial_obsm)
    indices = _knn_indices(coords, n_neighbors=n_neighbors)
    states = adata.obs[cell_state_key].astype(str).to_numpy()
    categories = pd.Index(pd.unique(states))
    state_to_idx = {state: idx for idx, state in enumerate(categories)}
    encoded = np.array([state_to_idx[state] for state in states], dtype=int)
    eye = np.eye(len(categories), dtype=np.float32)
    neighborhood_fractions = eye[encoded[indices]].mean(axis=1)

    composition_df = pd.DataFrame(
        neighborhood_fractions,
        index=adata.obs_names,
        columns=[f"local_frac__{state}" for state in categories],
    )

    def frac(state: str) -> np.ndarray:
        column = f"local_frac__{state}"
        return composition_df[column].to_numpy(dtype=np.float32) if column in composition_df.columns else np.zeros(adata.n_obs, dtype=np.float32)

    niche_scores = pd.DataFrame(index=adata.obs_names)
    niche_scores["immune_rich"] = frac("t_cell_exhausted_cytotoxic") + frac("b_plasma_like") + 0.5 * frac("macrophage_like")
    niche_scores["myeloid_vascular"] = 0.7 * frac("macrophage_like") + 0.3 * frac("endothelial_perivascular")
    niche_scores["epithelial_emt_front"] = 2.0 * np.minimum(frac("tumor_epithelial"), frac("emt_like_tumor"))
    niche_scores["b_plasma_rich"] = frac("b_plasma_like") + 0.25 * frac("t_cell_exhausted_cytotoxic")

    scores = niche_scores.to_numpy(dtype=np.float32)
    best_idx = scores.argmax(axis=1)
    best_score = scores[np.arange(adata.n_obs), best_idx]
    sorted_scores = np.sort(scores, axis=1)
    second_score = sorted_scores[:, -2] if scores.shape[1] >= 2 else np.zeros(adata.n_obs, dtype=np.float32)
    margin = (best_score - second_score).astype(np.float32)
    labels = niche_scores.columns.to_numpy()[best_idx].astype(object)
    labels[best_score < float(min_score)] = "mixed_low_signal"

    held_out_labels = pd.DataFrame(index=adata.obs_names)
    for branch in DEFAULT_RESISTANT_NICHES:
        threshold = float(np.nanquantile(niche_scores[branch].to_numpy(dtype=np.float32), 0.85))
        held_out_labels[f"{branch}__held_out_label"] = (niche_scores[branch].to_numpy(dtype=np.float32) >= threshold).astype(int)

    obs_updates = pd.DataFrame(index=adata.obs_names)
    obs_updates[niche_key] = pd.Categorical(labels)
    obs_updates[niche_score_key] = best_score
    obs_updates[niche_margin_key] = margin
    obs_updates = pd.concat(
        [
            obs_updates,
            composition_df.astype(np.float32),
            niche_scores.add_prefix(f"{niche_key}__").astype(np.float32),
            held_out_labels.astype(np.int8),
        ],
        axis=1,
    )
    _assign_obs_columns(adata, obs_updates)

    summary = (
        pd.DataFrame({"niche": labels, "niche_score": best_score, "niche_margin": margin})
        .groupby("niche", dropna=False)
        .agg(n_cells=("niche", "size"), mean_score=("niche_score", "mean"), mean_margin=("niche_margin", "mean"))
        .reset_index()
        .sort_values(["mean_score", "n_cells"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return {
        "composition": composition_df,
        "niche_scores": niche_scores,
        "held_out_labels": held_out_labels,
        "summary": summary,
    }


def score_immune_resistance_program(
    adata: AnnData,
    *,
    discordance_result: Mapping[str, pd.DataFrame] | None = None,
    niche_result: Mapping[str, pd.DataFrame] | None = None,
    niche_key: str = "spatial_niche",
    state_key: str = "joint_cell_state",
    prefix: str = "immune_resistance",
    region_bins: int = 24,
    n_neighbors: int = 15,
    resistant_niches: Sequence[str] = DEFAULT_RESISTANT_NICHES,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Score decoupled immune-resistance programs and ablations."""

    if discordance_result is None:
        discordance_result = compute_rna_protein_discordance(adata, n_neighbors=n_neighbors, region_bins=region_bins)
    if niche_result is None:
        niche_result = build_spatial_niches(adata, cell_state_key=state_key, n_neighbors=n_neighbors)

    cell_scores = discordance_result["cell_scores"]
    coords = _get_coords(adata)
    indices = _knn_indices(coords, n_neighbors=n_neighbors)

    axis_map = {
        "checkpoint": "checkpoint",
        "myeloid": "myeloid_activation",
        "vascular": "vascular_stromal",
        "emt": "epithelial_emt",
    }

    axis_joint = {}
    axis_rna = {}
    axis_protein = {}
    axis_abs = {}
    for short_name, pathway in axis_map.items():
        axis_joint[short_name] = cell_scores[f"discordance__pathway__{pathway}__joint_activation"].to_numpy(dtype=np.float32)
        axis_rna[short_name] = cell_scores[f"discordance__pathway__{pathway}__rna"].to_numpy(dtype=np.float32)
        axis_protein[short_name] = cell_scores[f"discordance__pathway__{pathway}__protein"].to_numpy(dtype=np.float32)
        axis_abs[short_name] = cell_scores[f"discordance__pathway__{pathway}__abs"].to_numpy(dtype=np.float32)

    model_vectors = {
        "rna_only": _mean_stack(list(axis_rna.values())),
        "protein_only": _mean_stack(list(axis_protein.values())),
        "joint_activity": _mean_stack(list(axis_joint.values())),
        "discordance_burden": _mean_stack(list(axis_abs.values())),
        "checkpoint_only": axis_joint["checkpoint"],
        "myeloid_only": axis_joint["myeloid"],
        "vascular_only": axis_joint["vascular"],
        "emt_only": axis_joint["emt"],
        "joint_minus_checkpoint": _mean_stack([axis_joint["myeloid"], axis_joint["vascular"], axis_joint["emt"]]),
        "joint_minus_myeloid": _mean_stack([axis_joint["checkpoint"], axis_joint["vascular"], axis_joint["emt"]]),
        "joint_minus_vascular": _mean_stack([axis_joint["checkpoint"], axis_joint["myeloid"], axis_joint["emt"]]),
        "joint_minus_emt": _mean_stack([axis_joint["checkpoint"], axis_joint["myeloid"], axis_joint["vascular"]]),
        "myeloid_vascular_branch": _mean_stack([axis_joint["checkpoint"], axis_joint["myeloid"], axis_joint["vascular"]]),
        "epithelial_emt_front_branch": _mean_stack([axis_joint["checkpoint"], axis_joint["myeloid"], axis_joint["emt"]]),
    }

    model_frame = pd.DataFrame(
        {f"{prefix}__model__{name}": values for name, values in model_vectors.items()},
        index=adata.obs_names,
    )
    axis_frame = pd.DataFrame(
        {f"{prefix}__axis__{name}": values for name, values in axis_joint.items()},
        index=adata.obs_names,
    )
    score_df = pd.concat([axis_frame.astype(np.float32), model_frame.astype(np.float32)], axis=1)
    _assign_obs_columns(adata, score_df)

    branch_targets = {
        branch: niche_result["niche_scores"][branch].to_numpy(dtype=np.float32)
        for branch in resistant_niches
        if branch in niche_result["niche_scores"].columns
    }
    held_out_labels = {
        branch: niche_result["held_out_labels"][f"{branch}__held_out_label"].to_numpy(dtype=int)
        for branch in branch_targets
        if f"{branch}__held_out_label" in niche_result["held_out_labels"].columns
    }

    model_rows = []
    for branch, target in branch_targets.items():
        labels = held_out_labels.get(branch, np.zeros(adata.n_obs, dtype=int))
        for model_name, values in model_vectors.items():
            branch_corr = _safe_corr(values, target)
            spatial_coherence = _safe_corr(values, _neighbour_mean(values, indices))
            roi_reproducibility = _roi_reproducibility(coords, values, region_bins=region_bins, alt_bins=region_bins + 4)
            held_out_auc = _safe_auc(labels, values)
            benchmark_parts = [
                held_out_auc,
                _metric_to_unit_interval(branch_corr),
                _metric_to_unit_interval(spatial_coherence),
                roi_reproducibility,
            ]
            benchmark_score = float(np.nanmean([item for item in benchmark_parts if not pd.isna(item)])) if any(not pd.isna(item) for item in benchmark_parts) else float("nan")
            model_rows.append(
                {
                    "branch": branch,
                    "model": model_name,
                    "held_out_auc": held_out_auc,
                    "branch_target_correlation": branch_corr,
                    "spatial_coherence": spatial_coherence,
                    "roi_reproducibility": roi_reproducibility,
                    "mean_in_held_out": float(np.nanmean(values[labels == 1])) if labels.sum() else float("nan"),
                    "mean_outside_held_out": float(np.nanmean(values[labels == 0])) if (labels == 0).any() else float("nan"),
                    "delta_in_minus_out": float(np.nanmean(values[labels == 1]) - np.nanmean(values[labels == 0]))
                    if labels.sum() and (labels == 0).any()
                    else float("nan"),
                    "benchmark_score": benchmark_score,
                }
            )

    model_comparison = (
        pd.DataFrame(model_rows)
        .sort_values(["branch", "benchmark_score", "held_out_auc"], ascending=[True, False, False], na_position="last")
        .reset_index(drop=True)
    )

    branch_summary_rows = []
    for branch, branch_models in DEFAULT_BRANCH_MODELS.items():
        branch_df = model_comparison[model_comparison["branch"] == branch]
        if branch_df.empty:
            continue
        eligible = branch_df[branch_df["model"].isin(branch_models)]
        best = eligible.iloc[0] if not eligible.empty else branch_df.iloc[0]
        branch_summary_rows.append(
            {
                "branch": branch,
                "best_model": best["model"],
                "benchmark_score": float(best["benchmark_score"]),
                "held_out_auc": float(best["held_out_auc"]),
                "branch_target_correlation": float(best["branch_target_correlation"]),
                "spatial_coherence": float(best["spatial_coherence"]),
                "roi_reproducibility": float(best["roi_reproducibility"]),
            }
        )
    branch_summary = pd.DataFrame(branch_summary_rows).sort_values("benchmark_score", ascending=False, na_position="last").reset_index(drop=True)

    marker_rows = []
    for branch, target in branch_targets.items():
        labels = held_out_labels.get(branch, np.zeros(adata.n_obs, dtype=int))
        for pair in DEFAULT_MARKER_PAIRS:
            abs_col = f"discordance__marker__{pair.label}__abs"
            signed_col = f"discordance__marker__{pair.label}__signed"
            if abs_col not in cell_scores.columns or signed_col not in cell_scores.columns:
                continue
            abs_values = cell_scores[abs_col].to_numpy(dtype=np.float32)
            signed_values = cell_scores[signed_col].to_numpy(dtype=np.float32)
            marker_rows.append(
                {
                    "branch": branch,
                    "label": pair.label,
                    "protein": pair.protein,
                    "gene": pair.gene,
                    "abs_target_correlation": _safe_corr(abs_values, target),
                    "signed_target_correlation": _safe_corr(signed_values, target),
                    "mean_abs_in_held_out": float(np.nanmean(abs_values[labels == 1])) if labels.sum() else float("nan"),
                    "mean_abs_outside_held_out": float(np.nanmean(abs_values[labels == 0])) if (labels == 0).any() else float("nan"),
                    "delta_abs_in_minus_out": float(np.nanmean(abs_values[labels == 1]) - np.nanmean(abs_values[labels == 0]))
                    if labels.sum() and (labels == 0).any()
                    else float("nan"),
                }
            )
    marker_enrichment = pd.DataFrame(marker_rows)
    if not marker_enrichment.empty:
        marker_enrichment["abs_rank_score"] = marker_enrichment["abs_target_correlation"].abs()
        marker_enrichment = marker_enrichment.sort_values(["branch", "abs_rank_score"], ascending=[True, False], na_position="last").reset_index(drop=True)

    branch_target_frame = pd.DataFrame(
        {f"{prefix}__branch_target__{branch}": values for branch, values in branch_targets.items()},
        index=adata.obs_names,
    )
    held_out_frame = pd.DataFrame(
        {f"{prefix}__held_out__{branch}": values for branch, values in held_out_labels.items()},
        index=adata.obs_names,
    )
    roi_input = pd.concat([score_df, branch_target_frame.astype(np.float32), held_out_frame.astype(np.float32)], axis=1)
    roi_df = _region_table(coords, roi_input, region_bins=region_bins)

    niche_summary = (
        pd.concat([adata.obs[[niche_key, state_key]], score_df], axis=1)
        .groupby([niche_key, state_key], dropna=False, observed=False)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(f"{prefix}__model__joint_activity", ascending=False)
        .reset_index(drop=True)
    )

    sample_summary = {
        "n_cells": int(adata.n_obs),
        "fraction_resistant_niche_cells": float(adata.obs[niche_key].astype(str).isin(tuple(resistant_niches)).mean()),
        "mean_rna_only": float(model_vectors["rna_only"].mean()),
        "mean_protein_only": float(model_vectors["protein_only"].mean()),
        "mean_joint_activity": float(model_vectors["joint_activity"].mean()),
        "mean_discordance_burden": float(model_vectors["discordance_burden"].mean()),
        "top_niche_by_joint_score": (
            niche_summary.groupby(niche_key, observed=False)[f"{prefix}__model__joint_activity"].mean().sort_values(ascending=False).index[0]
            if not niche_summary.empty
            else None
        ),
    }
    for row in branch_summary_rows:
        sample_summary[f"best_model__{row['branch']}"] = row["best_model"]
        sample_summary[f"benchmark_score__{row['branch']}"] = float(row["benchmark_score"])

    adata.uns.setdefault("spatial_immune_resistance", {})
    adata.uns["spatial_immune_resistance"]["branch_models"] = DEFAULT_BRANCH_MODELS

    return {
        "cell_scores": score_df,
        "roi_scores": roi_df,
        "model_comparison": model_comparison,
        "ablation_summary": model_comparison.copy(),
        "branch_summary": branch_summary,
        "marker_neighborhood_enrichment": marker_enrichment,
        "niche_summary": niche_summary,
        "sample_summary": sample_summary,
    }


def aggregate_multi_sample_study(
    studies: Sequence[Mapping[str, object]] | Mapping[str, Mapping[str, object]],
    *,
    metadata: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate per-sample immune-resistance pilot outputs into cohort tables."""

    if isinstance(studies, Mapping):
        study_items = list(studies.items())
    else:
        study_items = []
        for idx, study in enumerate(studies):
            sample_id = str(study.get("sample_id", f"sample_{idx}"))
            study_items.append((sample_id, study))

    sample_rows = []
    table_names = (
        "roi_scores",
        "model_comparison",
        "ablation_summary",
        "branch_summary",
        "niche_summary",
        "marker_neighborhood_enrichment",
    )
    collected = {name: [] for name in table_names}

    for sample_id, study in study_items:
        sample_summary = dict(study.get("sample_summary", {}))
        sample_summary["sample_id"] = sample_id
        sample_rows.append(sample_summary)

        source = study["immune_resistance"] if "immune_resistance" in study and isinstance(study["immune_resistance"], Mapping) else study
        for name in table_names:
            table = source.get(name)
            if isinstance(table, pd.DataFrame) and not table.empty:
                enriched = table.copy()
                enriched["sample_id"] = sample_id
                collected[name].append(enriched)

    sample_summary_df = pd.DataFrame(sample_rows)
    if metadata is not None and not metadata.empty:
        sample_summary_df = sample_summary_df.merge(metadata, on="sample_id", how="left")

    cohort_summary = pd.DataFrame()
    if not sample_summary_df.empty:
        numeric_cols = sample_summary_df.select_dtypes(include=[np.number]).columns.tolist()
        cohort_summary = pd.DataFrame(
            {
                "metric": numeric_cols,
                "mean": [float(sample_summary_df[col].mean()) for col in numeric_cols],
                "std": [
                    float(sample_summary_df[col].std(ddof=1)) if sample_summary_df.shape[0] > 1 else float("nan")
                    for col in numeric_cols
                ],
            }
        )

    return {
        "sample_summary": sample_summary_df,
        "roi_summary": pd.concat(collected["roi_scores"], ignore_index=True) if collected["roi_scores"] else pd.DataFrame(),
        "model_summary": pd.concat(collected["model_comparison"], ignore_index=True) if collected["model_comparison"] else pd.DataFrame(),
        "ablation_summary": pd.concat(collected["ablation_summary"], ignore_index=True) if collected["ablation_summary"] else pd.DataFrame(),
        "branch_summary": pd.concat(collected["branch_summary"], ignore_index=True) if collected["branch_summary"] else pd.DataFrame(),
        "niche_summary": pd.concat(collected["niche_summary"], ignore_index=True) if collected["niche_summary"] else pd.DataFrame(),
        "marker_summary": pd.concat(collected["marker_neighborhood_enrichment"], ignore_index=True) if collected["marker_neighborhood_enrichment"] else pd.DataFrame(),
        "cohort_summary": cohort_summary,
    }


__all__ = [
    "MarkerPair",
    "DEFAULT_MARKER_PAIRS",
    "DEFAULT_STATE_SIGNATURES",
    "DEFAULT_STATE_HIERARCHY",
    "DEFAULT_PATHWAY_MARKERS",
    "DEFAULT_RESISTANT_NICHES",
    "DEFAULT_BRANCH_MODELS",
    "annotate_joint_cell_states",
    "compute_rna_protein_discordance",
    "build_spatial_niches",
    "score_immune_resistance_program",
    "aggregate_multi_sample_study",
]
