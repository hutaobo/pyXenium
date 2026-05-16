from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._analysis import compute_pathway_activity_matrix


_PATHWAY_PANEL_COLUMNS = ["pathway", "family", "gene", "weight", "direction", "source", "evidence_tag"]


@dataclass(frozen=True)
class MorphoPathwayConfig:
    """Configuration for contour-level morphology-to-pathway residual analysis."""

    scoring_method: str = "weighted_sum"
    normalize_activity: bool = True
    covariates: tuple[str, ...] = ("structure", "x", "y", "boundary_distance")
    image_feature_prefixes: tuple[str, ...] = ("embedding__", "image__", "morphology__")
    min_pathway_genes: int = 2
    min_pathway_coverage: float = 0.35
    n_permutations: int = 0
    permutation_top_n: int = 20
    n_negative_controls: int = 0
    negative_control_top_n: int = 10
    spatial_strata_cols: tuple[str, ...] = ("structure", "x", "y", "boundary_distance")
    stratification_bins: int = 4
    random_state: int | None = 0
    fdr_alpha: float = 0.05
    spatial_null_alpha: float = 0.01
    validation_abs_rho_threshold: float = 0.35


_DEFAULT_CURATED_PATHWAYS: dict[str, dict[str, Any]] = {
    "luminal_estrogen_response": {
        "family": "endocrine_epithelial_identity",
        "genes": ["ESR1", "PGR", "GATA3", "FOXA1", "XBP1", "AGR2", "KRT8", "KRT18", "TFF1", "GREB1"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "ER/luminal epithelial program",
    },
    "epithelial_identity": {
        "family": "endocrine_epithelial_identity",
        "genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "CLDN4", "TACSTD2"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "epithelial adhesion/cytokeratin state",
    },
    "basal_squamous_state": {
        "family": "endocrine_epithelial_identity",
        "genes": ["KRT5", "KRT14", "KRT17", "TP63", "LAMB3", "LAMC2"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "basal/squamous epithelial state",
    },
    "unfolded_protein_response": {
        "family": "metabolic_stress",
        "genes": ["HSPA5", "XBP1", "ATF4", "DDIT3", "HERPUD1", "DNAJB9", "HSP90B1"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "ER stress and UPR response",
    },
    "oxidative_phosphorylation": {
        "family": "metabolic_stress",
        "genes": ["NDUFA1", "NDUFB5", "COX5A", "ATP5F1A", "UQCRC1", "SDHB", "SLC25A5"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "mitochondrial respiratory chain",
    },
    "collagen_ecm_organization": {
        "family": "stromal_remodeling_caf_ecm",
        "genes": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FN1", "SPARC"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "collagen and extracellular matrix organization",
    },
    "myofibroblast_caf_activation": {
        "family": "stromal_remodeling_caf_ecm",
        "genes": ["ACTA2", "TAGLN", "POSTN", "PDGFRB", "COL11A1", "FAP", "MMP11"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "activated fibroblast/myofibroblast state",
    },
    "immune_activation": {
        "family": "immune_ecology",
        "genes": ["CD3D", "CD3E", "CD8A", "GZMB", "PRF1", "CCL5", "CXCL9"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "T cell and interferon-linked activation",
    },
    "immune_exclusion": {
        "family": "immune_ecology",
        "genes": ["TGFB1", "TGFBI", "CXCL12", "POSTN", "COL1A1", "COL3A1", "ACTA2"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "TGF-beta/stromal barrier program",
    },
    "emt_invasive_front": {
        "family": "invasion_boundary_emt",
        "genes": ["VIM", "SNAI2", "TWIST1", "ZEB1", "MMP2", "MMP9", "ITGA5", "FN1"],
        "source": "curated_cancer_pathway_v1",
        "evidence_tag": "EMT/invasive-front remodeling",
    },
}


def build_curated_pathway_panel(
    pathway_specs: Mapping[str, Any] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the curated long-form pathway panel used by morphopathway workflows."""

    if pathway_specs is None:
        pathway_specs = _DEFAULT_CURATED_PATHWAYS
    if isinstance(pathway_specs, pd.DataFrame):
        missing = {"pathway", "gene"}.difference(pathway_specs.columns)
        if missing:
            raise ValueError(f"pathway panel is missing required columns: {sorted(missing)}")
        panel = pathway_specs.copy()
        for column, default in {
            "family": "unspecified",
            "weight": 1.0,
            "direction": 1.0,
            "source": "user_supplied",
            "evidence_tag": "user_supplied",
        }.items():
            if column not in panel.columns:
                panel[column] = default
    else:
        rows: list[dict[str, Any]] = []
        for pathway, spec in pathway_specs.items():
            if isinstance(spec, Mapping):
                genes = spec.get("genes", [])
                family = spec.get("family", "unspecified")
                source = spec.get("source", "user_supplied")
                evidence_tag = spec.get("evidence_tag", "user_supplied")
                if isinstance(genes, Mapping):
                    iterable = genes.items()
                else:
                    iterable = [(gene, 1.0) for gene in genes]
            else:
                family = "unspecified"
                source = "user_supplied"
                evidence_tag = "user_supplied"
                iterable = [(gene, 1.0) for gene in spec]
            for gene, weight in iterable:
                rows.append(
                    {
                        "pathway": str(pathway),
                        "family": str(family),
                        "gene": str(gene),
                        "weight": float(weight),
                        "direction": 1.0,
                        "source": str(source),
                        "evidence_tag": str(evidence_tag),
                    }
                )
        panel = pd.DataFrame(rows, columns=_PATHWAY_PANEL_COLUMNS)

    if panel.empty:
        raise ValueError("pathway panel is empty.")
    panel["pathway"] = panel["pathway"].astype(str)
    panel["family"] = panel["family"].astype(str)
    panel["gene"] = panel["gene"].astype(str).str.strip()
    panel["weight"] = pd.to_numeric(panel["weight"], errors="coerce").fillna(1.0).astype(float)
    panel["direction"] = pd.to_numeric(panel["direction"], errors="coerce").fillna(1.0).astype(float)
    panel["source"] = panel["source"].astype(str)
    panel["evidence_tag"] = panel["evidence_tag"].astype(str)
    panel = panel.loc[panel["gene"] != ""].copy()
    if panel.empty:
        raise ValueError("pathway panel has no non-empty genes.")
    panel["weight"] = panel["weight"] * panel["direction"]
    panel = (
        panel.groupby(["pathway", "family", "gene", "source", "evidence_tag"], as_index=False, sort=False)["weight"]
        .sum()
        .assign(direction=1.0)
    )
    return panel.loc[:, _PATHWAY_PANEL_COLUMNS]


def compute_pathway_coverage(
    expression_df: pd.DataFrame,
    pathway_panel: Mapping[str, Any] | pd.DataFrame | None = None,
    *,
    config: MorphoPathwayConfig | None = None,
) -> pd.DataFrame:
    """Summarize gene coverage for each curated pathway in an expression matrix."""

    cfg = config or MorphoPathwayConfig()
    panel = build_curated_pathway_panel(pathway_panel)
    available = {str(column) for column in expression_df.columns}
    rows: list[dict[str, Any]] = []
    for (pathway, family), group in panel.groupby(["pathway", "family"], sort=False):
        genes = list(dict.fromkeys(group["gene"].astype(str).tolist()))
        present = [gene for gene in genes if gene in available]
        missing = [gene for gene in genes if gene not in available]
        weight_sum = float(group.loc[group["gene"].isin(present), "weight"].abs().sum())
        coverage = float(len(present) / len(genes)) if genes else 0.0
        rows.append(
            {
                "pathway": str(pathway),
                "family": str(family),
                "n_genes": int(len(genes)),
                "n_present": int(len(present)),
                "coverage": coverage,
                "abs_weight_sum_present": weight_sum,
                "present_genes": ",".join(present),
                "missing_genes": ",".join(missing),
                "passes_min_genes": bool(len(present) >= int(cfg.min_pathway_genes)),
                "passes_min_coverage": bool(coverage >= float(cfg.min_pathway_coverage)),
            }
        )
    coverage_df = pd.DataFrame(rows)
    coverage_df["passes"] = coverage_df["passes_min_genes"] & coverage_df["passes_min_coverage"]
    return coverage_df


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=pvalues.index, dtype=float)
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for pos in range(n - 1, -1, -1):
        running = min(running, ranked[pos] * n / float(pos + 1))
        adjusted[pos] = running
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _numeric_image_columns(
    image_features_df: pd.DataFrame,
    *,
    prefixes: Sequence[str],
) -> list[str]:
    numeric_columns = [
        column
        for column in image_features_df.columns
        if pd.api.types.is_numeric_dtype(image_features_df[column])
    ]
    prefixed = [column for column in numeric_columns if any(str(column).startswith(prefix) for prefix in prefixes)]
    return prefixed or numeric_columns


def _count_true(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].fillna(False)
    if pd.api.types.is_bool_dtype(values):
        return int(values.sum())
    return int(values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum())


def _rank_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True).fillna(0.0).astype(float)


def _design_matrix(metadata: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = [pd.DataFrame({"intercept": np.ones(len(metadata), dtype=float)}, index=metadata.index)]
    for covariate in covariates:
        if covariate not in metadata.columns:
            continue
        values = metadata[covariate]
        if pd.api.types.is_numeric_dtype(values):
            pieces.append(pd.DataFrame({covariate: _rank_series(values)}, index=metadata.index))
        else:
            dummies = pd.get_dummies(values.astype(str).fillna("missing"), prefix=covariate, drop_first=True, dtype=float)
            if not dummies.empty:
                dummies.index = metadata.index
                pieces.append(dummies)
    design = pd.concat(pieces, axis=1).astype(float)
    return design.loc[:, ~design.columns.duplicated()]


def _residualize(values: pd.Series, design: pd.DataFrame) -> pd.Series:
    y = _rank_series(values).reindex(design.index).fillna(0.0).to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    if x.size == 0:
        residual = y - float(np.mean(y))
    else:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
    return pd.Series(residual, index=design.index, dtype=float)


def _partial_spearman_from_residuals(x_residual: pd.Series, y_residual: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(x_residual.to_numpy(dtype=float)) & np.isfinite(y_residual.to_numpy(dtype=float))
    if int(valid.sum()) < 3:
        return float("nan"), float("nan")
    x = x_residual.to_numpy(dtype=float)[valid]
    y = y_residual.to_numpy(dtype=float)[valid]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    rho, pvalue = stats.pearsonr(x, y)
    return float(rho), float(pvalue)


def _build_spatial_strata(metadata: pd.DataFrame, *, cols: Sequence[str], bins: int) -> pd.Series:
    labels: list[pd.Series] = []
    for column in cols:
        if column not in metadata.columns:
            continue
        values = metadata[column]
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > int(bins):
            try:
                label = pd.qcut(pd.to_numeric(values, errors="coerce"), q=int(bins), duplicates="drop").astype(str)
            except ValueError:
                label = values.astype(str)
        else:
            label = values.astype(str)
        labels.append(label.fillna("missing"))
    if not labels:
        return pd.Series("global", index=metadata.index, dtype=str)
    out = labels[0].astype(str)
    for label in labels[1:]:
        out = out + "|" + label.astype(str)
    return pd.Series(out.to_numpy(dtype=str), index=metadata.index, dtype=str)


def _gene_expression_bins(expression_df: pd.DataFrame, *, bins: int = 5) -> pd.Series:
    means = expression_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=0)
    if means.nunique(dropna=True) <= 1:
        return pd.Series("global", index=means.index, dtype=str)
    try:
        labels = pd.qcut(means.rank(method="average"), q=int(bins), duplicates="drop").astype(str)
    except ValueError:
        labels = pd.Series("global", index=means.index, dtype=str)
    return pd.Series(labels.to_numpy(dtype=str), index=means.index.astype(str), dtype=str)


def _sample_matched_genes(
    present_genes: Sequence[str],
    *,
    all_genes: Sequence[str],
    gene_bins: pd.Series,
    rng: np.random.Generator,
) -> list[str]:
    all_gene_list = [str(gene) for gene in all_genes]
    present = [str(gene) for gene in present_genes if str(gene) in set(all_gene_list)]
    if not present:
        return []
    excluded = set(present)
    sampled: list[str] = []
    for gene in present:
        gene_bin = gene_bins.get(gene, "global")
        pool = [candidate for candidate in all_gene_list if candidate not in excluded and gene_bins.get(candidate, "global") == gene_bin]
        if not pool:
            pool = [candidate for candidate in all_gene_list if candidate not in excluded]
        if not pool:
            pool = [candidate for candidate in all_gene_list if candidate != gene]
        if not pool:
            continue
        choice = str(rng.choice(pool))
        sampled.append(choice)
        excluded.add(choice)
    return sampled


def _decode_h5_strings(values: Any) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _read_xenium_h5_features(matrix_h5: str | Path) -> pd.DataFrame:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required to read Xenium cell_feature_matrix.h5 files.") from exc

    with h5py.File(matrix_h5, "r") as handle:
        features = handle["matrix/features"]
        names = _decode_h5_strings(features["name"][:])
        feature_type = _decode_h5_strings(features["feature_type"][:]) if "feature_type" in features else [""] * len(names)
        ids = _decode_h5_strings(features["id"][:]) if "id" in features else names
    return pd.DataFrame({"feature_index": np.arange(len(names), dtype=int), "id": ids, "name": names, "feature_type": feature_type})


def _read_xenium_h5_expression_subset(
    matrix_h5: str | Path,
    *,
    genes: Sequence[str],
    cell_ids: Sequence[str],
) -> pd.DataFrame:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required to read Xenium cell_feature_matrix.h5 files.") from exc

    requested_genes = list(dict.fromkeys(str(gene) for gene in genes if str(gene).strip()))
    requested_cells = list(dict.fromkeys(str(cell_id) for cell_id in cell_ids if str(cell_id).strip()))
    if not requested_genes:
        raise ValueError("At least one gene is required to read a Xenium expression subset.")
    if not requested_cells:
        raise ValueError("At least one cell id is required to read a Xenium expression subset.")

    with h5py.File(matrix_h5, "r") as handle:
        matrix = handle["matrix"]
        features = _read_xenium_h5_features(matrix_h5)
        features = features.loc[features["name"].isin(requested_genes)].copy()
        if "feature_type" in features.columns:
            gene_expression = features["feature_type"].astype(str).str.lower().str.contains("gene")
            features = features.loc[gene_expression].copy()
        if features.empty:
            raise ValueError("None of the requested genes were found in the Xenium feature matrix.")

        gene_order = [gene for gene in requested_genes if gene in set(features["name"].astype(str))]
        gene_position = {gene: idx for idx, gene in enumerate(gene_order)}
        feature_to_gene_position: dict[int, int] = {}
        for _, feature in features.iterrows():
            gene = str(feature["name"])
            if gene in gene_position:
                feature_to_gene_position[int(feature["feature_index"])] = int(gene_position[gene])

        barcodes = _decode_h5_strings(matrix["barcodes"][:])
        barcode_to_column = {barcode: idx for idx, barcode in enumerate(barcodes)}
        selected_cells = [cell_id for cell_id in requested_cells if cell_id in barcode_to_column]
        if not selected_cells:
            raise ValueError("None of the requested cell ids were found in the Xenium feature matrix.")

        data = matrix["data"]
        indices = matrix["indices"]
        indptr = matrix["indptr"]
        values = np.zeros((len(selected_cells), len(gene_order)), dtype=np.float32)
        for row_idx, cell_id in enumerate(selected_cells):
            column_idx = int(barcode_to_column[cell_id])
            start = int(indptr[column_idx])
            stop = int(indptr[column_idx + 1])
            if stop <= start:
                continue
            column_indices = indices[start:stop]
            column_values = data[start:stop]
            for feature_idx, value in zip(column_indices, column_values, strict=False):
                gene_idx = feature_to_gene_position.get(int(feature_idx))
                if gene_idx is not None:
                    values[row_idx, gene_idx] += float(value)

    return pd.DataFrame(values, index=pd.Index(selected_cells, name="cell_id"), columns=gene_order)


def _find_xenium_he_artifacts(root: Path) -> tuple[Path | None, Path | None]:
    image_matches = sorted(root.glob("*_he_image.ome.tif"))
    alignment_matches = sorted(root.glob("*_he_alignment.csv"))
    image_path = image_matches[0] if image_matches else None
    alignment_path = alignment_matches[0] if alignment_matches else None
    return image_path, alignment_path


def _as_yxc_image(array: np.ndarray, axes: str) -> np.ndarray:
    data = np.asarray(array)
    axes = str(axes).upper()
    if data.ndim == 2:
        return np.repeat(data[:, :, None], 3, axis=2)
    if data.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D H&E image level, got shape {data.shape}.")
    if axes in {"SYX", "CYX"}:
        data = np.moveaxis(data, 0, -1)
    elif axes in {"YXS", "YXC"}:
        pass
    elif axes in {"XSY", "XCY"}:
        data = np.moveaxis(data, 1, -1)
        data = np.swapaxes(data, 0, 1)
    else:
        if data.shape[0] in {3, 4}:
            data = np.moveaxis(data, 0, -1)
        elif data.shape[-1] not in {3, 4}:
            raise ValueError(f"Could not infer H&E channel axis from shape {data.shape} and axes {axes!r}.")
    if data.shape[-1] > 3:
        data = data[..., :3]
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=2)
    return data


def _load_he_lowres_level(
    he_image_path: str | Path,
    *,
    max_dimension: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover
        raise ImportError("tifffile is required to sample H&E image features.") from exc

    path = Path(he_image_path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", []) or [series])
        chosen_index = len(levels) - 1
        for idx, level in enumerate(levels):
            shape = tuple(int(value) for value in level.shape)
            axes = str(level.axes).upper()
            y_axis = axes.find("Y")
            x_axis = axes.find("X")
            y_size = shape[y_axis] if y_axis >= 0 else shape[-2]
            x_size = shape[x_axis] if x_axis >= 0 else shape[-1]
            if max(y_size, x_size) <= int(max_dimension):
                chosen_index = idx
                break
        level = levels[chosen_index]
        level_image = _as_yxc_image(level.asarray(), str(level.axes))
        base = levels[0]
        base_shape = tuple(int(value) for value in base.shape)
        base_axes = str(base.axes).upper()
        base_y = base_shape[base_axes.find("Y")] if "Y" in base_axes else base_shape[-2]
        base_x = base_shape[base_axes.find("X")] if "X" in base_axes else base_shape[-1]
    info = {
        "he_image_path": str(path),
        "he_pyramid_level": int(chosen_index),
        "he_level_shape": str(tuple(int(value) for value in level_image.shape)),
        "he_level_axes": "YXC",
        "he_base_shape_yx": str((int(base_y), int(base_x))),
        "he_downsample_y": float(base_y / level_image.shape[0]),
        "he_downsample_x": float(base_x / level_image.shape[1]),
    }
    return level_image, info


def _scale_image_to_float(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.size == 0:
        return data
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    max_value = float(finite.max())
    if max_value > 1.5:
        dtype = np.asarray(image).dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data = data / float(info.max)
        else:
            data = data / max_value
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def _empty_he_patch_feature_record() -> dict[str, float]:
    return {
        "image__he_r_mean": 0.0,
        "image__he_g_mean": 0.0,
        "image__he_b_mean": 0.0,
        "image__he_r_std": 0.0,
        "image__he_g_std": 0.0,
        "image__he_b_std": 0.0,
        "image__he_luma_mean": 0.0,
        "image__he_luma_std": 0.0,
        "image__he_luma_q10": 0.0,
        "image__he_luma_q90": 0.0,
        "image__he_saturation_mean": 0.0,
        "image__he_saturation_std": 0.0,
        "image__he_hematoxylin_proxy": 0.0,
        "image__he_hematoxylin_std": 0.0,
        "image__he_eosin_proxy": 0.0,
        "image__he_eosin_std": 0.0,
        "image__he_stain_contrast": 0.0,
        "image__he_dark_fraction": 0.0,
        "image__he_bright_fraction": 0.0,
        "image__he_local_contrast": 0.0,
        "image__he_edge_mean": 0.0,
        "image__he_edge_std": 0.0,
        "image__he_texture_entropy": 0.0,
        "image__he_texture_energy": 0.0,
    }


def _he_patch_feature_record(patch: np.ndarray) -> dict[str, float]:
    if patch.size == 0:
        return _empty_he_patch_feature_record()
    rgb = np.asarray(patch, dtype=np.float32)[..., :3]
    flat = rgb.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0, ddof=0)
    max_rgb = flat.max(axis=1)
    min_rgb = flat.min(axis=1)
    luma_image = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    luma = luma_image.reshape(-1)
    saturation = np.divide(max_rgb - min_rgb, np.maximum(max_rgb, 1e-6))
    hematoxylin = flat[:, 2] - flat[:, 0]
    eosin = (flat[:, 0] + flat[:, 1]) / 2.0 - flat[:, 2]
    dx = np.diff(luma_image, axis=1)
    dy = np.diff(luma_image, axis=0)
    if dx.size and dy.size:
        edge_values = np.concatenate([np.abs(dx).reshape(-1), np.abs(dy).reshape(-1)])
    elif dx.size:
        edge_values = np.abs(dx).reshape(-1)
    elif dy.size:
        edge_values = np.abs(dy).reshape(-1)
    else:
        edge_values = np.asarray([0.0], dtype=np.float32)
    hist, _ = np.histogram(luma, bins=16, range=(0.0, 1.0), density=False)
    probabilities = hist.astype(float) / max(float(hist.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0
    energy = float(np.sum(probabilities * probabilities))
    return {
        "image__he_r_mean": float(mean_rgb[0]),
        "image__he_g_mean": float(mean_rgb[1]),
        "image__he_b_mean": float(mean_rgb[2]),
        "image__he_r_std": float(std_rgb[0]),
        "image__he_g_std": float(std_rgb[1]),
        "image__he_b_std": float(std_rgb[2]),
        "image__he_luma_mean": float(luma.mean()),
        "image__he_luma_std": float(luma.std(ddof=0)),
        "image__he_luma_q10": float(np.quantile(luma, 0.10)),
        "image__he_luma_q90": float(np.quantile(luma, 0.90)),
        "image__he_saturation_mean": float(saturation.mean()),
        "image__he_saturation_std": float(saturation.std(ddof=0)),
        "image__he_hematoxylin_proxy": float(hematoxylin.mean()),
        "image__he_hematoxylin_std": float(hematoxylin.std(ddof=0)),
        "image__he_eosin_proxy": float(eosin.mean()),
        "image__he_eosin_std": float(eosin.std(ddof=0)),
        "image__he_stain_contrast": float((hematoxylin - eosin).std(ddof=0)),
        "image__he_dark_fraction": float((luma < 0.35).mean()),
        "image__he_bright_fraction": float((luma > 0.85).mean()),
        "image__he_local_contrast": float(luma.std(ddof=0)),
        "image__he_edge_mean": float(edge_values.mean()),
        "image__he_edge_std": float(edge_values.std(ddof=0)),
        "image__he_texture_entropy": entropy,
        "image__he_texture_energy": energy,
    }


def _resize_patch_nearest(patch: np.ndarray, *, size: int = 16) -> np.ndarray:
    if patch.size == 0:
        return np.zeros((int(size), int(size), 3), dtype=np.float32)
    image = np.asarray(patch, dtype=np.float32)[..., :3]
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((int(size), int(size), 3), dtype=np.float32)
    y_idx = np.clip(np.rint(np.linspace(0, height - 1, int(size))).astype(int), 0, height - 1)
    x_idx = np.clip(np.rint(np.linspace(0, width - 1, int(size))).astype(int), 0, width - 1)
    return image[np.ix_(y_idx, x_idx)]


def _he_patch_projection_features(
    patch: np.ndarray,
    *,
    output_dim: int = 24,
    projection_size: int = 16,
) -> dict[str, float]:
    """Deterministic patch-projection embedding fallback for environments without PLIP/UNI."""

    resized = _resize_patch_nearest(patch, size=projection_size)
    luma = 0.2126 * resized[..., 0] + 0.7152 * resized[..., 1] + 0.0722 * resized[..., 2]
    saturation = resized.max(axis=2) - resized.min(axis=2)
    hematoxylin = resized[..., 2] - resized[..., 0]
    eosin = (resized[..., 0] + resized[..., 1]) / 2.0 - resized[..., 2]
    channels = [luma, saturation, hematoxylin, eosin]
    yy, xx = np.mgrid[0:projection_size, 0:projection_size]
    yy = (yy + 0.5) / float(projection_size)
    xx = (xx + 0.5) / float(projection_size)
    bases = [
        np.ones_like(xx),
        xx - 0.5,
        yy - 0.5,
        (xx - 0.5) * (yy - 0.5),
        np.sin(np.pi * xx),
        np.sin(np.pi * yy),
        np.cos(2.0 * np.pi * xx),
        np.cos(2.0 * np.pi * yy),
        np.sin(2.0 * np.pi * (xx + yy)),
        np.cos(2.0 * np.pi * (xx - yy)),
        ((xx - 0.5) ** 2) - ((yy - 0.5) ** 2),
        np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2),
    ]
    values: list[float] = []
    for channel in channels:
        centered = channel - float(np.mean(channel))
        scale = float(np.std(centered)) or 1.0
        normalized = centered / scale
        for basis in bases:
            values.append(float(np.mean(normalized * basis)))
    if len(values) < int(output_dim):
        values.extend([0.0] * (int(output_dim) - len(values)))
    values = values[: int(output_dim)]
    return {f"embedding__he_patch_projection_{idx:03d}": value for idx, value in enumerate(values)}


def _clip_output_tensor(output: Any) -> Any:
    if hasattr(output, "shape"):
        return output
    for name in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if value is None:
            continue
        if name == "last_hidden_state" and hasattr(value, "ndim") and value.ndim >= 3:
            return value[:, 0, :]
        return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Could not extract image feature tensor from {type(output)!r}.")


def sample_he_image_features_at_cells(
    cells_df: pd.DataFrame,
    *,
    he_image_path: str | Path,
    he_alignment_path: str | Path,
    max_dimension: int = 2048,
    patch_radius_px: int = 1,
    include_patch_projection: bool = False,
    patch_projection_dim: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample low-resolution aligned H&E color features at Xenium cell centroids."""

    required = {"cell_id", "x_centroid", "y_centroid"}
    missing = required.difference(cells_df.columns)
    if missing:
        raise ValueError(f"cells_df is missing required columns for H&E sampling: {sorted(missing)}")
    image, image_info = _load_he_lowres_level(he_image_path, max_dimension=max_dimension)
    image = _scale_image_to_float(image)
    affine = np.loadtxt(he_alignment_path, delimiter=",")
    if affine.shape != (3, 3):
        raise ValueError(f"H&E alignment matrix must have shape (3, 3), got {affine.shape}.")
    inverse = np.linalg.inv(affine)
    xy = cells_df.loc[:, ["x_centroid", "y_centroid"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xy_h = np.c_[xy, np.ones(len(xy), dtype=float)]
    image_xy = xy_h @ inverse.T
    low_x = image_xy[:, 0] / float(image_info["he_downsample_x"])
    low_y = image_xy[:, 1] / float(image_info["he_downsample_y"])
    height, width = int(image.shape[0]), int(image.shape[1])
    x_idx = np.rint(low_x).astype(int)
    y_idx = np.rint(low_y).astype(int)
    inside = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    radius = max(int(patch_radius_px), 0)

    rows: list[dict[str, Any]] = []
    for row_idx, _cell_id in enumerate(cells_df["cell_id"].astype(str)):
        record: dict[str, Any] = {
            "qc__he_inside_mask": float(bool(inside[row_idx])),
            "qc__he_lowres_x": float(low_x[row_idx]) if np.isfinite(low_x[row_idx]) else np.nan,
            "qc__he_lowres_y": float(low_y[row_idx]) if np.isfinite(low_y[row_idx]) else np.nan,
        }
        if not inside[row_idx]:
            record.update(_empty_he_patch_feature_record())
            if include_patch_projection:
                record.update(
                    {
                        f"embedding__he_patch_projection_{idx:03d}": 0.0
                        for idx in range(int(patch_projection_dim))
                    }
                )
            rows.append(record)
            continue
        y0 = max(int(y_idx[row_idx]) - radius, 0)
        y1 = min(int(y_idx[row_idx]) + radius + 1, height)
        x0 = max(int(x_idx[row_idx]) - radius, 0)
        x1 = min(int(x_idx[row_idx]) + radius + 1, width)
        patch = image[y0:y1, x0:x1, :3]
        record.update(_he_patch_feature_record(patch))
        if include_patch_projection:
            record.update(_he_patch_projection_features(patch, output_dim=patch_projection_dim))
        rows.append(record)
    features = pd.DataFrame(rows, index=pd.Index(cells_df["cell_id"].astype(str), name="cell_id"))
    manifest = pd.DataFrame(
        [
            {
                **image_info,
                "he_alignment_path": str(he_alignment_path),
                "he_patch_radius_px": int(radius),
                "he_patch_projection_included": bool(include_patch_projection),
                "he_patch_projection_dim": int(patch_projection_dim) if include_patch_projection else 0,
                "he_embedding_backend_status": "deterministic_patch_projection_fallback"
                if include_patch_projection
                else "none",
                "he_cells_requested": int(len(cells_df)),
                "he_cells_inside": int(inside.sum()),
                "he_cells_inside_fraction": float(inside.mean()) if len(inside) else 0.0,
            }
        ]
    )
    return features, manifest


def sample_clip_image_embeddings_at_cells(
    cells_df: pd.DataFrame,
    *,
    he_image_path: str | Path,
    he_alignment_path: str | Path,
    model_name: str = "vinid/plip",
    model_label: str = "plip",
    max_dimension: int = 2048,
    patch_radius_px: int = 16,
    batch_size: int = 16,
    device: str = "cpu",
    output_dim: int | None = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample aligned H&E patches at cells and encode them with a CLIP-compatible model."""

    required = {"cell_id", "x_centroid", "y_centroid"}
    missing = required.difference(cells_df.columns)
    if missing:
        raise ValueError(f"cells_df is missing required columns for CLIP sampling: {sorted(missing)}")
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "CLIP/PLIP embeddings require optional dependencies `torch`, `Pillow`, and `transformers`."
        ) from exc

    image, image_info = _load_he_lowres_level(he_image_path, max_dimension=max_dimension)
    image = _scale_image_to_float(image)
    affine = np.loadtxt(he_alignment_path, delimiter=",")
    if affine.shape != (3, 3):
        raise ValueError(f"H&E alignment matrix must have shape (3, 3), got {affine.shape}.")
    inverse = np.linalg.inv(affine)
    xy = cells_df.loc[:, ["x_centroid", "y_centroid"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xy_h = np.c_[xy, np.ones(len(xy), dtype=float)]
    image_xy = xy_h @ inverse.T
    low_x = image_xy[:, 0] / float(image_info["he_downsample_x"])
    low_y = image_xy[:, 1] / float(image_info["he_downsample_y"])
    height, width = int(image.shape[0]), int(image.shape[1])
    x_idx = np.rint(low_x).astype(int)
    y_idx = np.rint(low_y).astype(int)
    inside = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    radius = max(int(patch_radius_px), 1)

    resolved_device = str(device)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(resolved_device)
    model.eval()

    patches: list[Any] = []
    valid_rows: list[int] = []
    for row_idx in range(len(cells_df)):
        if not inside[row_idx]:
            continue
        y0 = max(int(y_idx[row_idx]) - radius, 0)
        y1 = min(int(y_idx[row_idx]) + radius + 1, height)
        x0 = max(int(x_idx[row_idx]) - radius, 0)
        x1 = min(int(x_idx[row_idx]) + radius + 1, width)
        patch = image[y0:y1, x0:x1, :3]
        if patch.size == 0:
            continue
        array = np.clip(patch * 255.0, 0.0, 255.0).astype(np.uint8)
        patches.append(Image.fromarray(array, mode="RGB"))
        valid_rows.append(row_idx)

    raw_dim = None
    encoded = np.zeros((len(cells_df), 0), dtype=np.float32)
    if patches:
        feature_batches: list[np.ndarray] = []
        batch_size = max(int(batch_size), 1)
        with torch.inference_mode():
            for start in range(0, len(patches), batch_size):
                batch = patches[start : start + batch_size]
                inputs = processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
                values = _clip_output_tensor(model.get_image_features(**inputs))
                values = torch.nn.functional.normalize(values.float(), p=2, dim=-1)
                feature_batches.append(values.cpu().numpy().astype(np.float32))
        valid_features = np.vstack(feature_batches)
        raw_dim = int(valid_features.shape[1])
        n_dim = raw_dim if output_dim is None else min(int(output_dim), raw_dim)
        encoded = np.zeros((len(cells_df), n_dim), dtype=np.float32)
        encoded[np.asarray(valid_rows, dtype=int), :] = valid_features[:, :n_dim]
    else:
        n_dim = 0 if output_dim is None else int(output_dim)
        encoded = np.zeros((len(cells_df), n_dim), dtype=np.float32)

    safe_label = "".join(ch if ch.isalnum() else "_" for ch in str(model_label).lower()).strip("_") or "clip"
    columns = [f"embedding__{safe_label}_{idx:03d}" for idx in range(encoded.shape[1])]
    features = pd.DataFrame(encoded, index=pd.Index(cells_df["cell_id"].astype(str), name="cell_id"), columns=columns)
    manifest = pd.DataFrame(
        [
            {
                **image_info,
                "he_alignment_path": str(he_alignment_path),
                "clip_model_name": str(model_name),
                "clip_model_label": str(model_label),
                "clip_backend": "transformers.CLIPModel",
                "clip_device": resolved_device,
                "clip_patch_radius_px": int(radius),
                "clip_batch_size": int(batch_size),
                "clip_raw_dim": "" if raw_dim is None else int(raw_dim),
                "clip_output_dim": int(encoded.shape[1]),
                "clip_cells_requested": int(len(cells_df)),
                "clip_cells_encoded": int(len(valid_rows)),
                "clip_cells_encoded_fraction": float(len(valid_rows) / len(cells_df)) if len(cells_df) else 0.0,
                "clip_embedding_backend_status": f"transformers_clip:{model_name}",
            }
        ]
    )
    return features, manifest


def aggregate_morphopathway_inputs_to_spatial_blocks(
    expression_df: pd.DataFrame,
    image_features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    bins: int = 16,
    min_cells_per_block: int = 5,
    coarse_structure_bins: int = 4,
) -> dict[str, pd.DataFrame]:
    """Aggregate cell-level morphopathway inputs into spatial-block pseudobulk rows."""

    common_index = expression_df.index.astype(str)
    common_index = common_index.intersection(image_features_df.index.astype(str)).intersection(metadata_df.index.astype(str))
    if common_index.empty:
        raise ValueError("expression_df, image_features_df, and metadata_df have no shared index.")

    expression = expression_df.copy()
    expression.index = expression.index.astype(str)
    image_features = image_features_df.copy()
    image_features.index = image_features.index.astype(str)
    metadata = metadata_df.copy()
    metadata.index = metadata.index.astype(str)
    expression = expression.loc[common_index]
    image_features = image_features.loc[common_index]
    metadata = metadata.loc[common_index]

    if "x" not in metadata.columns or "y" not in metadata.columns:
        raise KeyError("metadata_df must contain `x` and `y` columns for spatial-block aggregation.")
    x = pd.to_numeric(metadata["x"], errors="coerce")
    y = pd.to_numeric(metadata["y"], errors="coerce")
    valid = x.notna() & y.notna()
    expression = expression.loc[valid]
    image_features = image_features.loc[valid]
    metadata = metadata.loc[valid]
    x = x.loc[valid]
    y = y.loc[valid]
    if expression.empty:
        raise ValueError("No cells with finite x/y coordinates were available for spatial-block aggregation.")

    n_bins = max(int(bins), 1)
    try:
        x_bin = pd.qcut(x.rank(method="average"), q=n_bins, labels=False, duplicates="drop").astype("Int64")
        y_bin = pd.qcut(y.rank(method="average"), q=n_bins, labels=False, duplicates="drop").astype("Int64")
    except ValueError:
        x_bin = pd.Series(0, index=metadata.index, dtype="Int64")
        y_bin = pd.Series(0, index=metadata.index, dtype="Int64")
    block_id = "block_x" + x_bin.astype(str) + "_y" + y_bin.astype(str)
    sizes = block_id.value_counts()
    keep_blocks = sizes.loc[sizes >= int(min_cells_per_block)].index
    keep = block_id.isin(keep_blocks)
    if not bool(keep.any()):
        raise ValueError("No spatial blocks passed min_cells_per_block.")
    block_id = block_id.loc[keep]
    expression = expression.loc[keep]
    image_features = image_features.loc[keep]
    metadata = metadata.loc[keep]

    numeric_image = image_features.loc[:, [col for col in image_features.columns if pd.api.types.is_numeric_dtype(image_features[col])]]
    expression_block = expression.groupby(block_id, sort=True).mean()
    image_block = numeric_image.groupby(block_id, sort=True).mean()
    meta_numeric_cols = [
        col
        for col in ["x", "y", "boundary_distance", "log_total_counts"]
        if col in metadata.columns and pd.api.types.is_numeric_dtype(metadata[col])
    ]
    metadata_block = metadata.loc[:, meta_numeric_cols].groupby(block_id, sort=True).mean()
    metadata_block["n_cells_in_block"] = block_id.value_counts().reindex(metadata_block.index).astype(int).to_numpy()

    try:
        coarse_x = pd.qcut(
            metadata_block["x"].rank(method="average"),
            q=max(int(coarse_structure_bins), 1),
            labels=False,
            duplicates="drop",
        ).astype("Int64")
        coarse_y = pd.qcut(
            metadata_block["y"].rank(method="average"),
            q=max(int(coarse_structure_bins), 1),
            labels=False,
            duplicates="drop",
        ).astype("Int64")
        metadata_block["structure"] = ("coarse_x" + coarse_x.astype(str) + "_y" + coarse_y.astype(str)).to_numpy()
    except ValueError:
        metadata_block["structure"] = "global"
    metadata_block.index = pd.Index(metadata_block.index.astype(str), name="spatial_block_id")
    expression_block.index = metadata_block.index
    image_block.index = metadata_block.index

    block_manifest = pd.DataFrame(
        [
            {
                "aggregation": "spatial_block",
                "input_cells": int(len(common_index)),
                "retained_cells": int(len(block_id)),
                "n_blocks": int(len(metadata_block)),
                "bins": int(n_bins),
                "min_cells_per_block": int(min_cells_per_block),
                "median_cells_per_block": float(metadata_block["n_cells_in_block"].median()),
            }
        ]
    )
    return {
        "expression": expression_block,
        "image_features": image_block,
        "metadata": metadata_block,
        "block_manifest": block_manifest,
    }


def prepare_xenium_cell_morphopathway_inputs(
    xenium_root: str | Path,
    *,
    pathway_panel: Mapping[str, Any] | pd.DataFrame | None = None,
    max_cells: int | None = 5000,
    extra_background_genes: int = 256,
    include_he_features: bool = True,
    he_image_path: str | Path | None = None,
    he_alignment_path: str | Path | None = None,
    he_max_dimension: int = 2048,
    he_patch_radius_px: int = 1,
    include_patch_projection: bool = False,
    patch_projection_dim: int = 24,
    clip_model_name: str | None = None,
    clip_model_label: str = "plip",
    clip_output_dim: int | None = 64,
    clip_batch_size: int = 16,
    clip_device: str = "cpu",
    clip_patch_radius_px: int = 16,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Prepare a lightweight cell-level morphopathway input bundle from Xenium outs."""

    root = Path(xenium_root)
    matrix_h5 = root / "cell_feature_matrix.h5"
    cells_path = root / "cells.parquet"
    if not matrix_h5.exists():
        raise FileNotFoundError(f"Missing Xenium feature matrix: {matrix_h5}")
    if not cells_path.exists():
        raise FileNotFoundError(f"Missing Xenium cells table: {cells_path}")

    panel = build_curated_pathway_panel(pathway_panel)
    curated_genes = list(dict.fromkeys(panel["gene"].astype(str).tolist()))
    rng = np.random.default_rng(random_state)

    feature_table = _read_xenium_h5_features(matrix_h5)
    gene_features = feature_table.loc[
        feature_table["feature_type"].astype(str).str.lower().str.contains("gene", na=False)
    ].copy()
    available_gene_order = list(dict.fromkeys(gene_features["name"].astype(str).tolist()))
    curated_set = set(curated_genes)
    background_pool = [gene for gene in available_gene_order if gene not in curated_set]
    n_background = min(max(int(extra_background_genes), 0), len(background_pool))
    background_genes = (
        list(rng.choice(background_pool, size=n_background, replace=False)) if n_background > 0 else []
    )
    genes_to_read = curated_genes + [gene for gene in background_genes if gene not in curated_set]

    required_columns = [
        "cell_id",
        "x_centroid",
        "y_centroid",
        "transcript_counts",
        "control_probe_counts",
        "genomic_control_counts",
        "total_counts",
        "cell_area",
        "nucleus_area",
        "nucleus_count",
        "segmentation_method",
    ]
    cells = pd.read_parquet(cells_path)
    available_columns = [column for column in required_columns if column in cells.columns]
    cells = cells.loc[:, available_columns].copy()
    if "cell_id" not in cells.columns:
        raise KeyError("cells.parquet must contain a `cell_id` column.")
    cells["cell_id"] = cells["cell_id"].astype(str)
    for column in ["x_centroid", "y_centroid", "transcript_counts", "total_counts", "cell_area", "nucleus_area"]:
        if column in cells.columns:
            cells[column] = pd.to_numeric(cells[column], errors="coerce")
    cells = cells.dropna(subset=[column for column in ["x_centroid", "y_centroid"] if column in cells.columns])
    if "transcript_counts" in cells.columns:
        cells = cells.loc[cells["transcript_counts"].fillna(0.0) > 0].copy()
    if max_cells is not None and len(cells) > int(max_cells):
        selected_positions = np.sort(rng.choice(len(cells), size=int(max_cells), replace=False))
        cells = cells.iloc[selected_positions].copy()
    cells = cells.drop_duplicates("cell_id", keep="first").reset_index(drop=True)

    expression_counts = _read_xenium_h5_expression_subset(
        matrix_h5,
        genes=genes_to_read,
        cell_ids=cells["cell_id"].astype(str).tolist(),
    )
    cells = cells.set_index("cell_id").loc[expression_counts.index].copy()
    total_counts = (
        pd.to_numeric(cells.get("transcript_counts", expression_counts.sum(axis=1)), errors="coerce")
        .reindex(expression_counts.index)
        .replace(0, np.nan)
    )
    expression = np.log1p(expression_counts.div(total_counts, axis=0).fillna(0.0) * 10000.0).astype(np.float32)

    cell_area = pd.to_numeric(cells.get("cell_area", pd.Series(0.0, index=cells.index)), errors="coerce").fillna(0.0)
    nucleus_area = pd.to_numeric(cells.get("nucleus_area", pd.Series(0.0, index=cells.index)), errors="coerce").fillna(0.0)
    transcript_counts = pd.to_numeric(cells.get("transcript_counts", pd.Series(0.0, index=cells.index)), errors="coerce").fillna(0.0)
    total = pd.to_numeric(cells.get("total_counts", transcript_counts), errors="coerce").fillna(0.0)
    control_probe = pd.to_numeric(cells.get("control_probe_counts", pd.Series(0.0, index=cells.index)), errors="coerce").fillna(0.0)
    genomic_control = pd.to_numeric(cells.get("genomic_control_counts", pd.Series(0.0, index=cells.index)), errors="coerce").fillna(0.0)
    nucleus_count = pd.to_numeric(cells.get("nucleus_count", pd.Series(0.0, index=cells.index)), errors="coerce").fillna(0.0)
    safe_area = cell_area.replace(0, np.nan)
    safe_total = total.replace(0, np.nan)

    image_features = pd.DataFrame(index=expression.index)
    image_features["morphology__cell_area_log1p"] = np.log1p(cell_area).to_numpy(dtype=float)
    image_features["morphology__nucleus_area_log1p"] = np.log1p(nucleus_area).to_numpy(dtype=float)
    image_features["morphology__nucleus_to_cell_area"] = (nucleus_area / safe_area).fillna(0.0).to_numpy(dtype=float)
    image_features["morphology__transcript_density_log1p"] = np.log1p((transcript_counts / safe_area).fillna(0.0)).to_numpy(dtype=float)
    image_features["morphology__nucleus_count_log1p"] = np.log1p(nucleus_count).to_numpy(dtype=float)
    image_features["morphology__control_fraction"] = ((control_probe + genomic_control) / safe_total).fillna(0.0).to_numpy(dtype=float)

    he_manifest = pd.DataFrame()
    clip_manifest = pd.DataFrame()
    he_available = False
    he_source = "not_requested"
    embedding_backend_status = "none"
    if include_he_features:
        resolved_he_image = Path(he_image_path) if he_image_path is not None else None
        resolved_he_alignment = Path(he_alignment_path) if he_alignment_path is not None else None
        if resolved_he_image is None or resolved_he_alignment is None:
            auto_image, auto_alignment = _find_xenium_he_artifacts(root)
            resolved_he_image = resolved_he_image or auto_image
            resolved_he_alignment = resolved_he_alignment or auto_alignment
        if resolved_he_image is not None and resolved_he_alignment is not None:
            he_features, he_manifest = sample_he_image_features_at_cells(
                cells.reset_index(),
                he_image_path=resolved_he_image,
                he_alignment_path=resolved_he_alignment,
                max_dimension=he_max_dimension,
                patch_radius_px=he_patch_radius_px,
                include_patch_projection=include_patch_projection,
                patch_projection_dim=patch_projection_dim,
            )
            image_features = pd.concat([image_features, he_features.reindex(image_features.index).fillna(0.0)], axis=1)
            he_available = True
            he_source = "aligned low-resolution H&E pyramid sampled at cell centroids"
            embedding_backend_status = str(he_manifest.loc[0, "he_embedding_backend_status"])
            if clip_model_name:
                clip_features, clip_manifest = sample_clip_image_embeddings_at_cells(
                    cells.reset_index(),
                    he_image_path=resolved_he_image,
                    he_alignment_path=resolved_he_alignment,
                    model_name=str(clip_model_name),
                    model_label=clip_model_label,
                    max_dimension=he_max_dimension,
                    patch_radius_px=clip_patch_radius_px,
                    batch_size=clip_batch_size,
                    device=clip_device,
                    output_dim=clip_output_dim,
                )
                image_features = pd.concat(
                    [image_features, clip_features.reindex(image_features.index).fillna(0.0)],
                    axis=1,
                )
                embedding_backend_status = str(clip_manifest.loc[0, "clip_embedding_backend_status"])
        else:
            he_source = "requested but no *_he_image.ome.tif/*_he_alignment.csv pair was found"

    x = pd.to_numeric(cells.get("x_centroid"), errors="coerce").fillna(0.0)
    y = pd.to_numeric(cells.get("y_centroid"), errors="coerce").fillna(0.0)
    x_range = float(x.max() - x.min()) or 1.0
    y_range = float(y.max() - y.min()) or 1.0
    x_norm = (x - float(x.min())) / x_range
    y_norm = (y - float(y.min())) / y_range
    boundary_distance = np.minimum.reduce([x_norm, 1.0 - x_norm, y_norm, 1.0 - y_norm])
    try:
        x_bin = pd.qcut(x.rank(method="average"), q=4, labels=False, duplicates="drop").astype("Int64").astype(str)
        y_bin = pd.qcut(y.rank(method="average"), q=4, labels=False, duplicates="drop").astype("Int64").astype(str)
        structure = "xy_" + x_bin.fillna("missing") + "_" + y_bin.fillna("missing")
    except ValueError:
        structure = pd.Series("global", index=cells.index, dtype=str)

    metadata = pd.DataFrame(index=expression.index)
    metadata["structure"] = pd.Series(structure.to_numpy(dtype=str), index=expression.index, dtype=str)
    metadata["x"] = x.to_numpy(dtype=float)
    metadata["y"] = y.to_numpy(dtype=float)
    metadata["boundary_distance"] = np.asarray(boundary_distance, dtype=float)
    metadata["log_total_counts"] = np.log1p(total).to_numpy(dtype=float)
    if "segmentation_method" in cells.columns:
        metadata["segmentation_method"] = cells["segmentation_method"].astype(str).to_numpy()

    present_curated = [gene for gene in curated_genes if gene in expression.columns]
    input_manifest = pd.DataFrame(
        [
            {
                "xenium_root": str(root),
                "n_cells": int(len(expression)),
                "n_expression_genes": int(expression.shape[1]),
                "n_curated_genes_requested": int(len(curated_genes)),
                "n_curated_genes_present": int(len(present_curated)),
                "n_background_genes": int(len(background_genes)),
                "max_cells": "" if max_cells is None else int(max_cells),
                "random_state": "" if random_state is None else int(random_state),
                "expression_normalization": "log1p(counts_per_10k_transcripts_from_cells.parquet)",
                "image_feature_source": f"cells.parquet morphology proxies; {he_source}",
                "he_features_requested": bool(include_he_features),
                "he_features_available": bool(he_available),
                "he_pyramid_level": ""
                if he_manifest.empty
                else int(he_manifest.loc[0, "he_pyramid_level"]),
                "he_cells_inside_fraction": ""
                if he_manifest.empty
                else float(he_manifest.loc[0, "he_cells_inside_fraction"]),
                "he_image_path": "" if he_manifest.empty else str(he_manifest.loc[0, "he_image_path"]),
                "he_alignment_path": "" if he_manifest.empty else str(he_manifest.loc[0, "he_alignment_path"]),
                "he_embedding_backend_status": ""
                if he_manifest.empty
                else embedding_backend_status,
            }
        ]
    )

    return {
        "expression": expression,
        "image_features": image_features,
        "metadata": metadata,
        "pathway_panel": panel,
        "input_manifest": input_manifest,
        "he_manifest": he_manifest,
        "clip_manifest": clip_manifest,
    }


def run_xenium_cell_morphopathway_smoke(
    xenium_root: str | Path,
    *,
    output_dir: str | Path,
    sample_name: str,
    pathway_panel: Mapping[str, Any] | pd.DataFrame | None = None,
    config: MorphoPathwayConfig | None = None,
    max_cells: int | None = 5000,
    extra_background_genes: int = 256,
    include_he_features: bool = True,
    he_image_path: str | Path | None = None,
    he_alignment_path: str | Path | None = None,
    he_max_dimension: int = 2048,
    he_patch_radius_px: int = 1,
    include_patch_projection: bool = False,
    patch_projection_dim: int = 24,
    clip_model_name: str | None = None,
    clip_model_label: str = "plip",
    clip_output_dim: int | None = 64,
    clip_batch_size: int = 16,
    clip_device: str = "cpu",
    clip_patch_radius_px: int = 16,
    aggregation: str = "cell",
    spatial_block_bins: int = 12,
    min_cells_per_block: int = 5,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Run a cell-level Xenium smoke bundle for morphopathway evidence generation."""

    cfg = config or MorphoPathwayConfig(
        covariates=("structure", "x", "y", "boundary_distance", "log_total_counts"),
        spatial_strata_cols=("structure",),
        n_permutations=16,
        permutation_top_n=10,
        n_negative_controls=16,
        negative_control_top_n=10,
        random_state=random_state,
    )
    prepared = prepare_xenium_cell_morphopathway_inputs(
        xenium_root,
        pathway_panel=pathway_panel,
        max_cells=max_cells,
        extra_background_genes=extra_background_genes,
        include_he_features=include_he_features,
        he_image_path=he_image_path,
        he_alignment_path=he_alignment_path,
        he_max_dimension=he_max_dimension,
        he_patch_radius_px=he_patch_radius_px,
        include_patch_projection=include_patch_projection,
        patch_projection_dim=patch_projection_dim,
        clip_model_name=clip_model_name,
        clip_model_label=clip_model_label,
        clip_output_dim=clip_output_dim,
        clip_batch_size=clip_batch_size,
        clip_device=clip_device,
        clip_patch_radius_px=clip_patch_radius_px,
        random_state=random_state,
    )
    block_manifest = pd.DataFrame()
    expression_df = prepared["expression"]
    image_features_df = prepared["image_features"]
    metadata_df = prepared["metadata"]
    if aggregation == "spatial-block":
        aggregated = aggregate_morphopathway_inputs_to_spatial_blocks(
            expression_df,
            image_features_df,
            metadata_df,
            bins=spatial_block_bins,
            min_cells_per_block=min_cells_per_block,
        )
        expression_df = aggregated["expression"]
        image_features_df = aggregated["image_features"]
        metadata_df = aggregated["metadata"]
        block_manifest = aggregated["block_manifest"]
    elif aggregation != "cell":
        raise ValueError("aggregation must be either 'cell' or 'spatial-block'.")
    result = run_atera_morphopathway_brief(
        expression_df=expression_df,
        image_features_df=image_features_df,
        metadata_df=metadata_df,
        pathway_panel=prepared["pathway_panel"],
        output_dir=output_dir,
        config=cfg,
        sample_name=sample_name,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_manifest_path = out / "input_manifest.csv"
    prepared["input_manifest"]["aggregation"] = aggregation
    prepared["input_manifest"]["analysis_observations"] = int(len(expression_df))
    prepared["input_manifest"].to_csv(input_manifest_path, index=False)
    if not prepared["he_manifest"].empty:
        he_manifest_path = out / "he_feature_manifest.csv"
        prepared["he_manifest"].to_csv(he_manifest_path, index=False)
        result.setdefault("files", {})["he_feature_manifest"] = str(he_manifest_path)
    if not prepared["clip_manifest"].empty:
        clip_manifest_path = out / "clip_feature_manifest.csv"
        prepared["clip_manifest"].to_csv(clip_manifest_path, index=False)
        result.setdefault("files", {})["clip_feature_manifest"] = str(clip_manifest_path)
    if not block_manifest.empty:
        block_manifest_path = out / "spatial_block_manifest.csv"
        block_manifest.to_csv(block_manifest_path, index=False)
        result.setdefault("files", {})["spatial_block_manifest"] = str(block_manifest_path)
    result["input_manifest"] = prepared["input_manifest"]
    result["he_manifest"] = prepared["he_manifest"]
    result["clip_manifest"] = prepared["clip_manifest"]
    result["spatial_block_manifest"] = block_manifest
    result["prepared_inputs"] = prepared
    result.setdefault("files", {})["input_manifest"] = str(input_manifest_path)
    return result


def _permute_within_strata(values: pd.Series, strata: pd.Series, rng: np.random.Generator) -> pd.Series:
    shuffled = values.copy()
    for _, indices in strata.groupby(strata, sort=False).groups.items():
        index = list(indices)
        if len(index) <= 1:
            continue
        shuffled.loc[index] = rng.permutation(values.loc[index].to_numpy(dtype=float))
    return shuffled


def compute_matched_random_pathway_controls(
    expression_df: pd.DataFrame,
    image_features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    associations: pd.DataFrame,
    pathway_panel: pd.DataFrame,
    *,
    config: MorphoPathwayConfig | None = None,
) -> pd.DataFrame:
    """Compare top pathway-image associations with expression-matched random gene sets."""

    cfg = config or MorphoPathwayConfig()
    if int(cfg.n_negative_controls) <= 0 or associations.empty:
        return pd.DataFrame(
            columns=[
                "pathway",
                "image_feature",
                "observed_abs_partial_spearman_rho",
                "n_negative_controls",
                "negative_control_empirical_p",
                "negative_control_abs_rho_q95",
                "negative_control_abs_rho_q99",
                "passes_negative_control_95",
                "passes_negative_control_99",
            ]
        )

    rng = np.random.default_rng(cfg.random_state)
    expression = expression_df.copy()
    expression.index = expression.index.astype(str)
    image_features = image_features_df.copy()
    image_features.index = image_features.index.astype(str)
    metadata = metadata_df.copy()
    metadata.index = metadata.index.astype(str)

    common_index = expression.index.intersection(image_features.index).intersection(metadata.index)
    expression = expression.loc[common_index]
    image_features = image_features.loc[common_index]
    metadata = metadata.loc[common_index]

    design = _design_matrix(metadata, cfg.covariates)
    image_residuals = {
        image: _residualize(image_features[image], design)
        for image in _numeric_image_columns(image_features, prefixes=cfg.image_feature_prefixes)
    }
    gene_bins = _gene_expression_bins(expression)
    all_genes = expression.columns.astype(str).tolist()
    panel = build_curated_pathway_panel(pathway_panel)
    rows: list[dict[str, Any]] = []

    for _, row in associations.head(int(cfg.negative_control_top_n)).iterrows():
        pathway = str(row["pathway"])
        image_feature = str(row["image_feature"])
        if image_feature not in image_residuals:
            continue
        group = panel.loc[panel["pathway"].astype(str) == pathway]
        present_genes = [gene for gene in group["gene"].astype(str).tolist() if gene in expression.columns]
        observed_abs = float(row["abs_partial_spearman_rho"])
        null_abs = np.empty(int(cfg.n_negative_controls), dtype=float)
        for control_idx in range(int(cfg.n_negative_controls)):
            sampled = _sample_matched_genes(present_genes, all_genes=all_genes, gene_bins=gene_bins, rng=rng)
            if not sampled:
                null_abs[control_idx] = 0.0
                continue
            control_activity = expression[sampled].mean(axis=1)
            control_residual = _residualize(control_activity, design)
            rho, _ = _partial_spearman_from_residuals(control_residual, image_residuals[image_feature])
            null_abs[control_idx] = abs(rho) if np.isfinite(rho) else 0.0
        empirical_p = (1.0 + float((null_abs >= observed_abs).sum())) / (float(len(null_abs)) + 1.0)
        q95 = float(np.quantile(null_abs, 0.95))
        q99 = float(np.quantile(null_abs, 0.99))
        rows.append(
            {
                "pathway": pathway,
                "image_feature": image_feature,
                "observed_abs_partial_spearman_rho": observed_abs,
                "n_present_genes": int(len(present_genes)),
                "n_negative_controls": int(cfg.n_negative_controls),
                "negative_control_empirical_p": empirical_p,
                "negative_control_abs_rho_q95": q95,
                "negative_control_abs_rho_q99": q99,
                "passes_negative_control_95": bool(observed_abs > q95),
                "passes_negative_control_99": bool(observed_abs > q99),
            }
        )
    return pd.DataFrame(rows)


def fit_residual_pathway_morphology_associations(
    pathway_activity: pd.DataFrame,
    image_features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    pathway_panel: Mapping[str, Any] | pd.DataFrame | None = None,
    config: MorphoPathwayConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit residual pathway-image associations with optional spatial permutation nulls."""

    cfg = config or MorphoPathwayConfig()
    common_index = pathway_activity.index.astype(str)
    common_index = common_index.intersection(image_features_df.index.astype(str)).intersection(metadata_df.index.astype(str))
    if common_index.empty:
        raise ValueError("pathway_activity, image_features_df, and metadata_df have no shared index.")

    activity = pathway_activity.copy()
    activity.index = activity.index.astype(str)
    images = image_features_df.copy()
    images.index = images.index.astype(str)
    metadata = metadata_df.copy()
    metadata.index = metadata.index.astype(str)
    activity = activity.loc[common_index]
    images = images.loc[common_index]
    metadata = metadata.loc[common_index]

    image_columns = _numeric_image_columns(images, prefixes=cfg.image_feature_prefixes)
    if not image_columns:
        raise ValueError("image_features_df has no numeric image feature columns.")
    pathway_columns = [column for column in activity.columns if pd.api.types.is_numeric_dtype(activity[column])]
    if not pathway_columns:
        raise ValueError("pathway_activity has no numeric pathway columns.")

    panel = build_curated_pathway_panel(pathway_panel) if pathway_panel is not None else build_curated_pathway_panel()
    family_by_pathway = panel.drop_duplicates("pathway").set_index("pathway")["family"].to_dict()

    design = _design_matrix(metadata, cfg.covariates)
    pathway_residuals = {pathway: _residualize(activity[pathway], design) for pathway in pathway_columns}
    image_residuals = {image: _residualize(images[image], design) for image in image_columns}

    rows: list[dict[str, Any]] = []
    for pathway in pathway_columns:
        for image_feature in image_columns:
            rho, pvalue = _partial_spearman_from_residuals(pathway_residuals[pathway], image_residuals[image_feature])
            rows.append(
                {
                    "pathway": str(pathway),
                    "family": str(family_by_pathway.get(str(pathway), "unspecified")),
                    "image_feature": str(image_feature),
                    "partial_spearman_rho": rho,
                    "abs_partial_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "p_value": pvalue,
                    "n_observations": int(len(common_index)),
                    "covariates": ",".join(covariate for covariate in cfg.covariates if covariate in metadata.columns),
                }
            )
    associations = pd.DataFrame(rows)
    associations["fdr"] = _benjamini_hochberg(associations["p_value"])
    associations = associations.sort_values(
        ["abs_partial_spearman_rho", "pathway", "image_feature"],
        ascending=[False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    associations["rank"] = np.arange(1, len(associations) + 1, dtype=int)

    spatial_nulls = pd.DataFrame()
    if int(cfg.n_permutations) > 0 and int(cfg.permutation_top_n) > 0 and not associations.empty:
        rng = np.random.default_rng(cfg.random_state)
        strata = _build_spatial_strata(metadata, cols=cfg.spatial_strata_cols, bins=cfg.stratification_bins)
        null_rows: list[dict[str, Any]] = []
        candidates = associations.head(int(cfg.permutation_top_n))
        for _, row in candidates.iterrows():
            pathway = str(row["pathway"])
            image_feature = str(row["image_feature"])
            observed_abs = float(row["abs_partial_spearman_rho"])
            null_abs = np.empty(int(cfg.n_permutations), dtype=float)
            for permutation_idx in range(int(cfg.n_permutations)):
                permuted = _permute_within_strata(pathway_residuals[pathway], strata, rng)
                rho, _ = _partial_spearman_from_residuals(permuted, image_residuals[image_feature])
                null_abs[permutation_idx] = abs(rho) if np.isfinite(rho) else 0.0
            empirical_p = (1.0 + float((null_abs >= observed_abs).sum())) / (float(len(null_abs)) + 1.0)
            null_rows.append(
                {
                    "pathway": pathway,
                    "image_feature": image_feature,
                    "observed_abs_partial_spearman_rho": observed_abs,
                    "permutations": int(cfg.n_permutations),
                    "permutation_empirical_p": empirical_p,
                    "null_abs_rho_q95": float(np.quantile(null_abs, 0.95)),
                    "null_abs_rho_q99": float(np.quantile(null_abs, 0.99)),
                    "passes_spatial_null_95": bool(observed_abs > float(np.quantile(null_abs, 0.95))),
                    "passes_spatial_null_99": bool(observed_abs > float(np.quantile(null_abs, 0.99))),
                }
            )
        spatial_nulls = pd.DataFrame(null_rows)
        associations = associations.merge(
            spatial_nulls[
                [
                    "pathway",
                    "image_feature",
                    "permutation_empirical_p",
                    "null_abs_rho_q95",
                    "null_abs_rho_q99",
                    "passes_spatial_null_95",
                    "passes_spatial_null_99",
                ]
            ],
            on=["pathway", "image_feature"],
            how="left",
        )

    return {
        "associations": associations,
        "spatial_nulls": spatial_nulls,
        "pathway_residuals": pd.DataFrame(pathway_residuals),
        "image_residuals": pd.DataFrame(image_residuals),
    }


def run_atera_morphopathway_brief(
    *,
    expression_df: pd.DataFrame,
    image_features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    pathway_panel: Mapping[str, Any] | pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    config: MorphoPathwayConfig | None = None,
    sample_name: str = "atera_wta_sample",
) -> dict[str, Any]:
    """Run the core morphopathway bundle writer on prepared contour-level inputs."""

    cfg = config or MorphoPathwayConfig()
    panel = build_curated_pathway_panel(pathway_panel)
    coverage = compute_pathway_coverage(expression_df, panel, config=cfg)
    passing_pathways = coverage.loc[coverage["passes"], "pathway"].astype(str).tolist()
    passing_panel = panel.loc[panel["pathway"].isin(passing_pathways)].copy()
    if passing_panel.empty:
        raise ValueError("No curated pathways passed coverage filters.")

    activity = compute_pathway_activity_matrix(
        expression_df,
        passing_panel.loc[:, ["pathway", "gene", "weight"]],
        method=cfg.scoring_method,
        normalize=cfg.normalize_activity,
    )
    fit = fit_residual_pathway_morphology_associations(
        activity,
        image_features_df,
        metadata_df,
        pathway_panel=passing_panel,
        config=cfg,
    )
    associations = fit["associations"]
    negative_controls = compute_matched_random_pathway_controls(
        expression_df,
        image_features_df,
        metadata_df,
        associations,
        passing_panel,
        config=cfg,
    )
    if not negative_controls.empty:
        associations = associations.merge(
            negative_controls[
                [
                    "pathway",
                    "image_feature",
                    "negative_control_empirical_p",
                    "negative_control_abs_rho_q95",
                    "negative_control_abs_rho_q99",
                    "passes_negative_control_95",
                    "passes_negative_control_99",
                ]
            ],
            on=["pathway", "image_feature"],
            how="left",
        )
    figure_source = (
        associations.sort_values(["family", "abs_partial_spearman_rho"], ascending=[True, False], kind="stable")
        .groupby("family", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    manifest = pd.DataFrame(
        [
            {
                "sample_name": sample_name,
                "n_observations": int(len(activity)),
                "n_input_pathways": int(panel["pathway"].nunique()),
                "n_passing_pathways": int(len(passing_pathways)),
                "n_image_features": int(len(_numeric_image_columns(image_features_df, prefixes=cfg.image_feature_prefixes))),
                "n_associations": int(len(associations)),
                "n_spatial_nulls": int(len(fit["spatial_nulls"])),
                "n_spatial_null_pass95": _count_true(fit["spatial_nulls"], "passes_spatial_null_95"),
                "n_spatial_null_pass99": _count_true(fit["spatial_nulls"], "passes_spatial_null_99"),
                "n_negative_controls": int(len(negative_controls)),
                "n_negative_control_pass95": _count_true(negative_controls, "passes_negative_control_95"),
                "n_negative_control_pass99": _count_true(negative_controls, "passes_negative_control_99"),
                "config": str(asdict(cfg)),
            }
        ]
    )

    files: dict[str, str] = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        outputs = {
            "pathway_panel": out / "pathway_panel.csv",
            "pathway_coverage": out / "pathway_coverage.csv",
            "pathway_activity": out / "pathway_activity.csv",
            "association_table": out / "association_table.csv",
            "spatial_nulls": out / "spatial_nulls.csv",
            "negative_controls": out / "negative_controls.csv",
            "figure_source_table": out / "figure_source_table.csv",
            "run_manifest": out / "run_manifest.csv",
        }
        panel.to_csv(outputs["pathway_panel"], index=False)
        coverage.to_csv(outputs["pathway_coverage"], index=False)
        activity.to_csv(outputs["pathway_activity"])
        associations.to_csv(outputs["association_table"], index=False)
        fit["spatial_nulls"].to_csv(outputs["spatial_nulls"], index=False)
        negative_controls.to_csv(outputs["negative_controls"], index=False)
        figure_source.to_csv(outputs["figure_source_table"], index=False)
        manifest.to_csv(outputs["run_manifest"], index=False)
        files = {name: str(path) for name, path in outputs.items()}

    return {
        "config": cfg,
        "pathway_panel": panel,
        "pathway_coverage": coverage,
        "pathway_activity": activity,
        "associations": associations,
        "spatial_nulls": fit["spatial_nulls"],
        "negative_controls": negative_controls,
        "figure_source_table": figure_source,
        "run_manifest": manifest,
        "files": files,
    }


def summarize_cross_cancer_validation(
    discovery_associations: pd.DataFrame,
    validation_associations: pd.DataFrame,
    *,
    config: MorphoPathwayConfig | None = None,
    discovery_label: str = "breast_discovery",
    validation_label: str = "cervical_validation",
) -> pd.DataFrame:
    """Summarize pathway and family recovery from discovery to validation association tables."""

    cfg = config or MorphoPathwayConfig()
    required = {"pathway", "family", "abs_partial_spearman_rho"}
    missing_discovery = required.difference(discovery_associations.columns)
    missing_validation = required.difference(validation_associations.columns)
    if missing_discovery:
        raise ValueError(f"discovery_associations is missing required columns: {sorted(missing_discovery)}")
    if missing_validation:
        raise ValueError(f"validation_associations is missing required columns: {sorted(missing_validation)}")

    discovery = discovery_associations.copy()
    validation = validation_associations.copy()
    discovery["abs_partial_spearman_rho"] = pd.to_numeric(
        discovery["abs_partial_spearman_rho"], errors="coerce"
    ).fillna(0.0)
    validation["abs_partial_spearman_rho"] = pd.to_numeric(
        validation["abs_partial_spearman_rho"], errors="coerce"
    ).fillna(0.0)
    discovery_best = (
        discovery.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("pathway", keep="first")
        .reset_index(drop=True)
    )
    validation_best_pathway = (
        validation.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("pathway", keep="first")
        .set_index("pathway")
    )
    validation_best_family = (
        validation.sort_values("abs_partial_spearman_rho", ascending=False, kind="stable")
        .drop_duplicates("family", keep="first")
        .set_index("family")
    )

    rows: list[dict[str, Any]] = []
    for _, row in discovery_best.iterrows():
        pathway = str(row["pathway"])
        family = str(row["family"])
        pathway_match = validation_best_pathway.loc[pathway] if pathway in validation_best_pathway.index else None
        family_match = validation_best_family.loc[family] if family in validation_best_family.index else None
        pathway_validation_abs = (
            float(pathway_match["abs_partial_spearman_rho"]) if pathway_match is not None else float("nan")
        )
        family_validation_abs = (
            float(family_match["abs_partial_spearman_rho"]) if family_match is not None else float("nan")
        )
        recovered = bool(
            (np.isfinite(pathway_validation_abs) and pathway_validation_abs >= float(cfg.validation_abs_rho_threshold))
            or (np.isfinite(family_validation_abs) and family_validation_abs >= float(cfg.validation_abs_rho_threshold))
        )
        rows.append(
            {
                "discovery_label": discovery_label,
                "validation_label": validation_label,
                "pathway": pathway,
                "family": family,
                "discovery_abs_partial_rho": float(row["abs_partial_spearman_rho"]),
                "validation_pathway_abs_partial_rho": pathway_validation_abs,
                "validation_family_best_pathway": str(family_match["pathway"]) if family_match is not None else "",
                "validation_family_abs_partial_rho": family_validation_abs,
                "validation_abs_rho_threshold": float(cfg.validation_abs_rho_threshold),
                "recovered_in_validation": recovered,
                "validation_call": "pathway_or_family_recovered" if recovered else "not_recovered",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "MorphoPathwayConfig",
    "aggregate_morphopathway_inputs_to_spatial_blocks",
    "build_curated_pathway_panel",
    "compute_pathway_coverage",
    "compute_matched_random_pathway_controls",
    "fit_residual_pathway_morphology_associations",
    "prepare_xenium_cell_morphopathway_inputs",
    "run_atera_morphopathway_brief",
    "run_xenium_cell_morphopathway_smoke",
    "sample_clip_image_embeddings_at_cells",
    "sample_he_image_features_at_cells",
    "summarize_cross_cancer_validation",
]
