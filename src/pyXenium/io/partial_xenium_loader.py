from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .xenium_artifacts import (
    ensure_local_path,
    is_remote_path,
    join_path,
    read_analysis_cell_groups,
    read_cells_zarr_spatial,
    read_mex_triplet,
    summarize_zarr_artifact,
)


def _resolve_artifact(base_url: str | None, name: str | None) -> str | None:
    if not name:
        return None

    path = name
    if base_url and not is_remote_path(name) and not name.startswith("/"):
        path = join_path(base_url, name)

    if str(path).endswith((".zip", ".zarr.zip")):
        return ensure_local_path(path, suffix=".zip")
    return path


def load_anndata_from_partial(
    base_url: Optional[str] = None,
    analysis_name: Optional[str] = None,
    cells_name: Optional[str] = None,
    transcripts_name: Optional[str] = None,
    mex_dir: Optional[str] = None,
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    build_counts_if_missing: bool = True,
) -> ad.AnnData:
    local_analysis = _resolve_artifact(base_url, analysis_name)
    local_cells = _resolve_artifact(base_url, cells_name)
    local_transcripts = _resolve_artifact(base_url, transcripts_name)

    uns: dict[str, object] = {
        "io": {
            "base_url": base_url,
            "analysis_name": analysis_name,
            "cells_name": cells_name,
            "transcripts_name": transcripts_name,
            "mex_dir": mex_dir or (join_path(base_url, "cell_feature_matrix") if base_url else None),
            "mex_matrix_name": mex_matrix_name,
            "mex_features_name": mex_features_name,
            "mex_barcodes_name": mex_barcodes_name,
        }
    }

    if local_analysis:
        try:
            uns["analysis"] = summarize_zarr_artifact(local_analysis, kind="analysis")
        except Exception as exc:
            uns["analysis_error"] = str(exc)

    if local_cells:
        try:
            uns["cells"] = summarize_zarr_artifact(local_cells, kind="cells")
        except Exception as exc:
            uns["cells_error"] = str(exc)

    effective_mex_dir = mex_dir or (join_path(base_url, "cell_feature_matrix") if base_url else None)

    X = None
    var_df: pd.DataFrame | None = None
    obs_df: pd.DataFrame | None = None
    barcodes = pd.Index([], name="barcode")

    if effective_mex_dir:
        try:
            X, features, barcodes = read_mex_triplet(
                effective_mex_dir,
                matrix_name=mex_matrix_name,
                features_name=mex_features_name,
                barcodes_name=mex_barcodes_name,
            )
            var_df = pd.DataFrame(index=features["id"].astype(str).to_numpy())
            var_df["feature_id"] = features["id"].astype(str).to_numpy()
            var_df["feature_name"] = features["name"].astype(str).to_numpy()
            var_df["feature_type"] = features["feature_type"].astype(str).to_numpy()
            var_df["feature_types"] = var_df["feature_type"].to_numpy()
            obs_df = pd.DataFrame(index=barcodes)
        except Exception as exc:
            uns["mex_error"] = str(exc)

    if X is None:
        if not build_counts_if_missing:
            raise FileNotFoundError("MEX files not found and build_counts_if_missing=False")
        X = sparse.csr_matrix((0, 0), dtype=np.float32)
        var_df = pd.DataFrame(index=pd.Index([], name=None))
        obs_df = pd.DataFrame(index=pd.Index([], name="barcode"))

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.layers["counts"] = adata.X.copy()
    adata.uns.update(uns)

    spatial_frame = None
    if local_cells:
        try:
            spatial_frame, spatial_meta = read_cells_zarr_spatial(local_cells)
            if adata.n_obs > 0:
                aligned = spatial_frame.reindex(adata.obs_names)
                if "x" in aligned.columns and "y" in aligned.columns:
                    adata.obs["x"] = aligned["x"].to_numpy()
                    adata.obs["y"] = aligned["y"].to_numpy()
                    adata.obsm["spatial"] = np.c_[adata.obs["x"].to_numpy(), adata.obs["y"].to_numpy()]
            adata.uns.setdefault("spatial", {})["units"] = spatial_meta.get("spatial_units", "micron")
            adata.uns.setdefault("cells_meta", {})["cell_summary_cols"] = spatial_meta.get(
                "cell_summary_columns"
            )
        except Exception as exc:
            adata.uns["cells_spatial_error"] = str(exc)

    if local_analysis:
        try:
            target_cells = len(spatial_frame) if spatial_frame is not None else adata.n_obs
            cluster_series, groups_meta = read_analysis_cell_groups(
                local_analysis,
                n_cells=target_cells,
                prefer_group="0",
            )
            if spatial_frame is not None:
                label_series = pd.Series(cluster_series.to_numpy(), index=spatial_frame.index)
                aligned = label_series.reindex(adata.obs_names).fillna("Unassigned")
            elif adata.n_obs == len(cluster_series):
                aligned = cluster_series.astype(str)
            else:
                aligned = pd.Series(["Unassigned"] * adata.n_obs, index=adata.obs_names)
            adata.obs["cluster"] = pd.Categorical(np.asarray(aligned).astype(str))
            adata.uns.setdefault("analysis", {}).update(
                {
                    "group_key_used": groups_meta.get("group_key", "0"),
                    "grouping_names": groups_meta.get("grouping_names"),
                    "n_clusters": groups_meta.get("n_clusters"),
                }
            )
        except Exception as exc:
            adata.uns["analysis_cluster_error"] = str(exc)

    if local_transcripts:
        adata.uns.setdefault("io", {})["transcripts_local_path"] = local_transcripts

    if adata.n_obs > 0 and adata.n_vars > 0:
        adata.var["total_counts"] = np.asarray(adata.X.sum(axis=0)).ravel()

    return adata
