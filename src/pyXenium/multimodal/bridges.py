from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from pyXenium.io.sdata_model import XeniumSData

__all__ = [
    "build_spatho_manifest",
    "compare_programs_with_embeddings",
    "export_for_stgpt",
    "import_stgpt_embeddings",
]

_SCHEMA_VERSION = "pyxenium-bridge-v1"
_DEFAULT_ID_COLUMNS = ("cell_id", "contour_id", "patch_id", "barcode", "id")
_NON_EMBEDDING_COLUMNS = {
    "cell_id",
    "contour_id",
    "patch_id",
    "barcode",
    "id",
    "sample_id",
    "niche",
    "cluster",
    "label",
    "leiden",
    "stgpt_niche",
    "uncertainty",
    "stgpt_uncertainty",
    "confidence",
}


def export_for_stgpt(
    data: ad.AnnData | XeniumSData,
    output_dir: str | Path,
    *,
    contour_key: str | None = None,
    feature_table: Mapping[str, Any] | None = None,
    sample_id: str | None = None,
    neighbor_k: int = 6,
    include_expression_matrix: bool = True,
) -> dict[str, Any]:
    """Write a lightweight pyXenium-to-stGPT handoff bundle.

    pyXenium owns Xenium loading and feature preparation; stGPT owns foundation
    model training/inference. This function writes only contract files that an
    external stGPT workflow can consume.
    """

    adata, sdata = _coerce_data(data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_sample_id = str(sample_id or _resolve_sample_id(adata, sdata=sdata))

    files: dict[str, str] = {}
    cell_table = _build_cell_table(adata, sample_id=resolved_sample_id)
    files["cell_table"] = _write_table(cell_table, out / "cells.csv")

    feature_table_frame = _build_feature_table(adata)
    files["features"] = _write_table(feature_table_frame, out / "features.csv")

    coords = _spatial_coordinates(adata)
    if coords is not None:
        coordinates = pd.DataFrame(
            {
                "cell_id": adata.obs_names.astype(str),
                "x": coords[:, 0],
                "y": coords[:, 1],
            }
        )
        files["coordinates"] = _write_table(coordinates, out / "coordinates.csv")
        edges = _build_spatial_edges(adata.obs_names.astype(str), coords, neighbor_k=neighbor_k)
        if not edges.empty:
            files["spatial_edges"] = _write_table(edges, out / "spatial_edges.csv")

    if include_expression_matrix:
        matrix_path = out / "rna_matrix.npz"
        matrix = _rna_matrix(adata)
        save_npz(matrix_path, matrix)
        files["rna_matrix"] = matrix_path.name
        matrix_meta = {
            "format": "scipy_sparse_csr_npz",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "cell_axis": "cells.csv:cell_id",
            "feature_axis": "features.csv:feature_id",
        }
        files["rna_matrix_metadata"] = _write_json(out / "rna_matrix_metadata.json", matrix_meta)

    if sdata is not None and contour_key is not None:
        if contour_key not in sdata.shapes:
            raise KeyError(f"`sdata.shapes[{contour_key!r}]` was not found.")
        files["contours"] = _write_table(sdata.shapes[contour_key], out / "contours.csv")
        patch_manifest = _build_contour_patch_manifest(sdata, contour_key=contour_key)
        if not patch_manifest.empty:
            files["contour_patch_manifest"] = _write_table(patch_manifest, out / "contour_patch_manifest.csv")

    if feature_table:
        feature_files = _write_feature_table_bundle(feature_table, out)
        files.update(feature_files)

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "pyxenium_to_stgpt_handoff",
        "sample_id": resolved_sample_id,
        "package_boundaries": {
            "pyXenium": "Xenium data loading, alignment, contour/cell feature tables, and interpretable analytics.",
            "stGPT": "RNA/H&E representation learning, masked prediction, spatial graph modeling, and uncertainty.",
            "HistoSeg": "Segmentation, tissue-domain detection, structure labels, contour proposals, and segmentation QC.",
            "SPatho": "Pathology-facing orchestration, review, reporting, and human-in-the-loop workflows.",
        },
        "inputs": {
            "n_cells": int(adata.n_obs),
            "n_features": int(adata.n_vars),
            "has_spatial": bool(coords is not None),
            "contour_key": contour_key,
        },
        "files": files,
        "stgpt_expected_outputs": {
            "cell_embeddings": "CSV/Parquet keyed by cell_id.",
            "contour_embeddings": "CSV/Parquet keyed by contour_id.",
            "optional_columns": ["niche", "cluster", "label", "uncertainty"],
        },
        "interpretation_boundary": (
            "pyXenium exports prepared Xenium artifacts and later compares stGPT outputs "
            "with interpretable programs; stGPT owns model inference/training."
        ),
    }
    _write_json(out / "stgpt_handoff_manifest.json", manifest)
    return manifest


def import_stgpt_embeddings(
    embeddings: str | Path | pd.DataFrame,
    *,
    target: ad.AnnData | XeniumSData | None = None,
    entity: str = "cell",
    id_column: str | None = None,
    obsm_key: str = "stgpt",
    embedding_prefix: str = "stgpt_",
) -> dict[str, Any]:
    """Read stGPT embeddings and optionally attach cell embeddings to AnnData.

    The function intentionally treats stGPT output as an external artifact. It
    does not import stGPT or run any model code.
    """

    frame = _read_table(embeddings)
    resolved_id = _resolve_id_column(frame, preferred=id_column, entity=entity)
    embedding_columns = _numeric_embedding_columns(frame, id_column=resolved_id)
    if not embedding_columns:
        raise ValueError("No numeric embedding columns were found in the stGPT output table.")
    frame[resolved_id] = frame[resolved_id].astype(str)

    adata = _target_adata(target)
    attached = False
    missing_ids: list[str] = []
    if adata is not None and entity == "cell":
        indexed = frame.set_index(resolved_id, drop=False)
        obs_ids = adata.obs_names.astype(str)
        missing_ids = [cell_id for cell_id in obs_ids if cell_id not in indexed.index]
        aligned = indexed.reindex(obs_ids)
        matrix = aligned.loc[:, embedding_columns].to_numpy(dtype=float)
        adata.obsm[obsm_key] = matrix
        for column in _annotation_columns(frame, id_column=resolved_id, embedding_columns=embedding_columns):
            adata.obs[f"{embedding_prefix}{column}"] = aligned[column].to_numpy()
        attached = True

    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stgpt_embedding_import",
        "entity": str(entity),
        "id_column": resolved_id,
        "embedding_columns": list(embedding_columns),
        "n_rows": int(len(frame)),
        "attached_to_anndata": attached,
        "obsm_key": obsm_key if attached else None,
        "missing_target_ids": missing_ids,
        "table": frame,
    }


def compare_programs_with_embeddings(
    program_scores: str | Path | pd.DataFrame,
    embeddings: str | Path | pd.DataFrame,
    *,
    id_column: str = "contour_id",
    program_columns: Sequence[str] | None = None,
    embedding_columns: Sequence[str] | None = None,
    label_column: str | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Compare pyXenium interpretable program scores with external embeddings."""

    scores = _read_table(program_scores)
    embedding_frame = _read_table(embeddings)
    if id_column not in scores.columns or id_column not in embedding_frame.columns:
        raise KeyError(f"Both tables must contain id_column={id_column!r}.")
    scores[id_column] = scores[id_column].astype(str)
    embedding_frame[id_column] = embedding_frame[id_column].astype(str)

    merged = scores.merge(embedding_frame, on=id_column, how="inner", suffixes=("", "__embedding"))
    if merged.empty:
        raise ValueError("No rows overlapped between program scores and embeddings.")

    resolved_programs = list(program_columns or _infer_program_columns(scores))
    resolved_embeddings = list(embedding_columns or _numeric_embedding_columns(embedding_frame, id_column=id_column))
    if not resolved_programs:
        raise ValueError("No numeric program score columns were found.")
    if not resolved_embeddings:
        raise ValueError("No numeric embedding columns were found.")

    correlations = _program_embedding_correlations(
        merged,
        program_columns=resolved_programs,
        embedding_columns=resolved_embeddings,
    )
    resolved_label = _resolve_label_column(merged, preferred=label_column)
    label_summary = _program_label_summary(merged, program_columns=resolved_programs, label_column=resolved_label)
    concordance = _program_label_concordance(merged, label_column=resolved_label)

    return {
        "summary": {
            "schema_version": _SCHEMA_VERSION,
            "kind": "program_embedding_comparison",
            "id_column": id_column,
            "n_overlap": int(len(merged)),
            "program_columns": resolved_programs,
            "embedding_columns": resolved_embeddings,
            "label_column": resolved_label,
        },
        "merged": merged,
        "correlations": correlations,
        "label_summary": label_summary,
        "concordance": concordance,
    }


def build_spatho_manifest(
    output_path: str | Path | None = None,
    *,
    sample_id: str | None = None,
    xenium_path: str | Path | None = None,
    pyxenium_artifacts: Mapping[str, Any] | None = None,
    histoseg_artifacts: Mapping[str, Any] | None = None,
    stgpt_artifacts: Mapping[str, Any] | None = None,
    review_targets: Sequence[Mapping[str, Any]] | None = None,
    report_intent: str = "ai_driven_spatial_pathology_review",
) -> dict[str, Any]:
    """Build a SPatho-facing manifest without importing or running SPatho."""

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "pyxenium_to_spatho_manifest",
        "sample_id": sample_id,
        "report_intent": str(report_intent),
        "package_boundaries": {
            "HistoSeg": "Owns segmentation, contour proposals, masks, and segmentation QC.",
            "pyXenium": "Owns Xenium data loading, spatial quantification, and interpretable analysis artifacts.",
            "stGPT": "Owns foundation embeddings, masked prediction, uncertainty, and learned niche labels.",
            "SPatho": "Owns pathology-facing orchestration, review UI/reporting, and human feedback loops.",
        },
        "inputs": {
            "xenium_path": _optional_path_text(xenium_path),
        },
        "artifacts": {
            "histoseg": _json_ready_paths(histoseg_artifacts or {}),
            "pyxenium": _json_ready_paths(pyxenium_artifacts or {}),
            "stgpt": _json_ready_paths(stgpt_artifacts or {}),
        },
        "review_targets": [_json_ready_paths(target) for target in (review_targets or [])],
        "interpretation_boundary": (
            "SPatho may consume these artifacts to render a pathology workflow, but "
            "the upstream packages retain ownership of their respective algorithms."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)
    return manifest


def _coerce_data(data: ad.AnnData | XeniumSData) -> tuple[ad.AnnData, XeniumSData | None]:
    if isinstance(data, XeniumSData):
        return data.table, data
    if isinstance(data, ad.AnnData):
        return data, None
    raise TypeError("data must be an AnnData or XeniumSData instance.")


def _target_adata(target: ad.AnnData | XeniumSData | None) -> ad.AnnData | None:
    if target is None:
        return None
    if isinstance(target, XeniumSData):
        return target.table
    if isinstance(target, ad.AnnData):
        return target
    raise TypeError("target must be an AnnData, XeniumSData, or None.")


def _resolve_sample_id(adata: ad.AnnData, *, sdata: XeniumSData | None) -> str:
    if sdata is not None:
        for key in ("sample_id", "dataset_id", "source_path"):
            value = sdata.metadata.get(key)
            if value:
                return str(value)
    for key in ("sample_id", "dataset_id"):
        value = adata.uns.get(key)
        if value:
            return str(value)
    return "sample_0"


def _build_cell_table(adata: ad.AnnData, *, sample_id: str) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs = obs.drop(columns=[column for column in ("sample_id", "cell_id") if column in obs.columns])
    obs.insert(0, "cell_id", adata.obs_names.astype(str))
    obs.insert(0, "sample_id", sample_id)
    coords = _spatial_coordinates(adata)
    if coords is not None:
        obs["x"] = coords[:, 0]
        obs["y"] = coords[:, 1]
    return obs.reset_index(drop=True)


def _build_feature_table(adata: ad.AnnData) -> pd.DataFrame:
    var = adata.var.copy()
    var = var.drop(columns=[column for column in ("feature_id",) if column in var.columns])
    var.insert(0, "feature_id", adata.var_names.astype(str))
    if "feature_name" not in var.columns:
        var["feature_name"] = var["name"].astype(str) if "name" in var.columns else adata.var_names.astype(str)
    return var.reset_index(drop=True)


def _spatial_coordinates(adata: ad.AnnData) -> np.ndarray | None:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]
    if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        return adata.obs.loc[:, ["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs.loc[:, ["x", "y"]].to_numpy(dtype=float)
    return None


def _build_spatial_edges(cell_ids: Sequence[str], coords: np.ndarray, *, neighbor_k: int) -> pd.DataFrame:
    if len(coords) <= 1:
        return pd.DataFrame(columns=["source_cell_id", "target_cell_id", "distance"])
    k = max(1, min(int(neighbor_k), len(coords) - 1))
    model = NearestNeighbors(n_neighbors=k + 1)
    model.fit(coords)
    distances, indices = model.kneighbors(coords, return_distance=True)
    rows = []
    ids = [str(cell_id) for cell_id in cell_ids]
    for source_index, source_id in enumerate(ids):
        for distance, target_index in zip(distances[source_index, 1:], indices[source_index, 1:]):
            rows.append(
                {
                    "source_cell_id": source_id,
                    "target_cell_id": ids[int(target_index)],
                    "distance": float(distance),
                }
            )
    return pd.DataFrame(rows)


def _rna_matrix(adata: ad.AnnData) -> sparse.csr_matrix:
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))


def _build_contour_patch_manifest(sdata: XeniumSData, *, contour_key: str) -> pd.DataFrame:
    patches = sdata.contour_images.get(contour_key, {})
    rows = []
    for contour_id, image in patches.items():
        metadata = dict(image.metadata or {})
        rows.append(
            {
                "contour_id": str(contour_id),
                "source_path": image.source_path,
                "alignment_csv_path": image.alignment_csv_path,
                "pixel_size_um": image.pixel_size_um,
                "bbox_image_xy": json.dumps(metadata.get("bbox_image_xy")),
                "bbox_xenium_um": json.dumps(metadata.get("bbox_xenium_um")),
                "has_affine": image.image_to_xenium_affine is not None,
            }
        )
    return pd.DataFrame(rows)


def _write_feature_table_bundle(feature_table: Mapping[str, Any], out: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for key in (
        "contour_features",
        "zone_features",
        "rna_pseudobulk",
        "protein_pseudobulk",
        "pathway_activity",
        "cci_summary",
        "edge_gradients",
    ):
        value = feature_table.get(key)
        if isinstance(value, pd.DataFrame):
            files[f"feature_table_{key}"] = _write_table(value, out / f"{key}.csv")
    return files


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = frame.columns.map(str)
        return frame
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    frame.columns = frame.columns.map(str)
    return frame


def _write_table(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path.name


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready_paths(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.name


def _resolve_id_column(frame: pd.DataFrame, *, preferred: str | None, entity: str) -> str:
    if preferred:
        if preferred not in frame.columns:
            raise KeyError(f"Embedding table is missing id column {preferred!r}.")
        return preferred
    entity_default = f"{entity}_id"
    if entity_default in frame.columns:
        return entity_default
    for column in _DEFAULT_ID_COLUMNS:
        if column in frame.columns:
            return column
    raise KeyError("Could not infer an id column from the embedding table.")


def _numeric_embedding_columns(frame: pd.DataFrame, *, id_column: str) -> list[str]:
    columns = []
    for column in frame.columns:
        text = str(column)
        if text == id_column or text in _NON_EMBEDDING_COLUMNS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            columns.append(text)
    return columns


def _annotation_columns(frame: pd.DataFrame, *, id_column: str, embedding_columns: Sequence[str]) -> list[str]:
    excluded = {id_column, *embedding_columns}
    return [str(column) for column in frame.columns if str(column) not in excluded]


def _infer_program_columns(scores: pd.DataFrame) -> list[str]:
    excluded = {"sample_id", "contour_key", "contour_id", "top_program", "top_program_score"}
    return [
        str(column)
        for column in scores.columns
        if str(column) not in excluded
        and not str(column).endswith("_evidence")
        and pd.to_numeric(scores[column], errors="coerce").notna().sum() >= 2
    ]


def _resolve_label_column(frame: pd.DataFrame, *, preferred: str | None) -> str | None:
    if preferred:
        if preferred not in frame.columns:
            raise KeyError(f"Label column {preferred!r} was not found.")
        return preferred
    for column in ("stgpt_niche", "niche", "cluster", "label", "leiden"):
        if column in frame.columns:
            return column
    return None


def _program_embedding_correlations(
    frame: pd.DataFrame,
    *,
    program_columns: Sequence[str],
    embedding_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for program in program_columns:
        for embedding in embedding_columns:
            left = pd.to_numeric(frame[program], errors="coerce")
            right = pd.to_numeric(frame[embedding], errors="coerce")
            mask = left.notna() & right.notna()
            if int(mask.sum()) < 3:
                continue
            rho, p_value = spearmanr(left.loc[mask], right.loc[mask])
            if np.isfinite(rho):
                rows.append(
                    {
                        "program": str(program),
                        "embedding": str(embedding),
                        "spearman_rho": float(rho),
                        "abs_spearman_rho": float(abs(rho)),
                        "p_value": float(p_value),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["abs_spearman_rho", "program"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return result


def _program_label_summary(
    frame: pd.DataFrame,
    *,
    program_columns: Sequence[str],
    label_column: str | None,
) -> pd.DataFrame:
    if label_column is None:
        return pd.DataFrame(columns=["label", "n_items", "top_program", "top_program_mean_score"])
    rows = []
    for label, group in frame.groupby(label_column, dropna=False, sort=True):
        means = group.loc[:, program_columns].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        top_program = str(means.idxmax()) if not means.empty else None
        rows.append(
            {
                "label": str(label),
                "n_items": int(len(group)),
                "top_program": top_program,
                "top_program_mean_score": float(means[top_program]) if top_program is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _program_label_concordance(frame: pd.DataFrame, *, label_column: str | None) -> pd.DataFrame:
    if label_column is None or "top_program" not in frame.columns:
        return pd.DataFrame(columns=["label", "dominant_program", "dominant_fraction", "n_items"])
    rows = []
    for label, group in frame.groupby(label_column, dropna=False, sort=True):
        counts = group["top_program"].astype(str).value_counts()
        dominant = str(counts.index[0]) if not counts.empty else None
        rows.append(
            {
                "label": str(label),
                "dominant_program": dominant,
                "dominant_fraction": float(counts.iloc[0] / len(group)) if len(group) and not counts.empty else np.nan,
                "n_items": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _optional_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value))


def _json_ready_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
