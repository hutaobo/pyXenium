"""
pyXenium.analysis.tabnet_tools

Two high-level helpers that integrate a TabNet classifier workflow with AnnData:

1) tabnet_cluster_classifier
   - Supervised model to predict a (precomputed) cluster label from features
     assembled from RNA (and optionally protein) features.
   - Useful to quantify how well the cluster structure can be re-identified
     from molecular features, and to inspect global feature importances.
   - Writes predictions/probabilities back to `adata.obs` and metadata to
     `adata.uns['tabnet_cluster_classifier']`.

2) tabnet_protein_within_cluster_classifier
   - For a given RNA-defined cluster and target protein, performs a binary
     classification task (protein high vs low within that cluster) using RNA
     features (optionally plus other proteins), returning AUC and feature
     importances that explain the within-cluster variation of that protein.
   - Writes scores back to `adata.obs` for the in-cluster cells and metadata
     to `adata.uns['tabnet_within_cluster']`.

Design goals
------------
- Works on CPU by default, supports GPU if available & requested.
- Handles sparse RNA matrices; optionally reduces RNA with TruncatedSVD.
- Careful memory use: you can subsample, use HVGs, or SVD to keep arrays small.
- Records full provenance (params, shapes, timings) in `adata.uns`.

Dependencies: numpy, pandas, scipy, scikit-learn, pytorch_tabnet, anndata

NOTE: TabNet expects dense numpy arrays. When giving it sparse RNA, we first
      reduce with TruncatedSVD (dense output) or explicitly densify *after*
      subsetting/transforming to a manageable size.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

import time
import warnings
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.cluster import MiniBatchKMeans

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pytorch-tabnet is required. Install via `pip install pytorch-tabnet`.\n"
        f"Original import error: {e}"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _as_dense(X) -> np.ndarray:
    """Safely convert to a dense ndarray.
    If X is scipy sparse, use .toarray(); if it's already ndarray, return as-is.
    """
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _maybe_log1p(X, do: bool) -> np.ndarray:
    if not do:
        return X
    # For sparse, log1p on data only
    if sparse.issparse(X):
        X = X.tocoo(copy=False)
        X.data = np.log1p(X.data)
        return X.tocsr()
    return np.log1p(X)


def _standardize_train_val(X_tr: np.ndarray, X_val: np.ndarray, with_mean=True) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler(with_mean=with_mean)
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    return X_tr_s, X_val_s, scaler


def _fit_svd(X_rna, n_components: int, random_state: int = 42) -> Tuple[TruncatedSVD, np.ndarray]:
    """Fit TruncatedSVD on (possibly sparse) RNA and return (svd, X_svd)."""
    if n_components <= 0:
        raise ValueError("n_components must be > 0 for SVD")
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_svd = svd.fit_transform(X_rna)
    return svd, X_svd


def _concat_features(parts: List[np.ndarray]) -> np.ndarray:
    parts = [np.asarray(p) for p in parts if p is not None]
    if not parts:
        raise ValueError("No feature parts provided.")
    return np.hstack(parts)


@dataclass
class ClusterClassifierResult:
    classes_: List[str]
    accuracy: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    feature_names: List[str]
    feature_importances: List[float]
    label_key: str
    proba_keys: List[str]
    n_train: int
    n_val: int
    params: Dict
    timings_sec: Dict[str, float]


@dataclass
class WithinClusterResult:
    cluster: str
    protein: str
    auc: float
    accuracy: float
    macro_f1: float
    feature_names: List[str]
    feature_importances: List[float]
    score_key: str
    n_train: int
    n_val: int
    params: Dict
    timings_sec: Dict[str, float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tabnet_cluster_classifier(
    adata: AnnData,
    layer_key: str = "rna",
    protein_key: str = "protein",
    cluster_key: str = "cluster",
    # Feature engineering
    use_protein_features: bool = True,
    log1p_rna: bool = False,
    n_top_hvgs: Optional[int] = None,
    svd_components: Optional[int] = 0,
    # Sampling & splits
    n_cells: Optional[int] = None,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    # TabNet & training
    device_name: str = "cpu",
    tabnet_params: Optional[Dict] = None,
    max_epochs: int = 200,
    batch_size: int = 1024,
    patience: int = 20,
    # Outputs
    write_back: bool = True,
    out_key: str = "tabnet_cluster_classifier",
) -> ClusterClassifierResult:
    """
    Train a TabNet classifier to predict `cluster_key` using RNA (+/- protein) features.

    Writes to `adata.obs`:
      - f"{out_key}:pred" : predicted class label (str)
      - f"{out_key}:proba:<class>" : probability columns for each class

    Writes to `adata.uns[out_key]`:
      - metrics, params, timings, classes, feature_names, feature_importances

    Writes to `adata.varm[f"{out_key}:importance"]` if features originate from genes only.

    Returns a ClusterClassifierResult.
    """
    t0 = time.time()

    if layer_key not in adata.layers:
        raise KeyError(f"`{layer_key}` not found in adata.layers")
    if use_protein_features and protein_key not in adata.obsm:
        raise KeyError(f"`{protein_key}` not found in adata.obsm")

    # Labels
    if cluster_key not in adata.obs:
        warnings.warn(
            f"`{cluster_key}` not in adata.obs; creating via MiniBatchKMeans(n_clusters=10)."
        )
        X_rna_for_kmeans = adata.layers[layer_key]
        if sparse.issparse(X_rna_for_kmeans):
            X_km = TruncatedSVD(n_components=50, random_state=random_state).fit_transform(X_rna_for_kmeans)
        else:
            X_km = np.asarray(X_rna_for_kmeans)
        adata.obs[cluster_key] = MiniBatchKMeans(n_clusters=10, random_state=random_state).fit_predict(X_km)

    y_labels = adata.obs[cluster_key].astype(str).values

    # Optionally subsample
    n_obs = adata.n_obs
    if n_cells is not None and n_cells < n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_obs, size=n_cells, replace=False)
    else:
        idx = np.arange(n_obs)

    # Assemble RNA features
    X_rna = adata.layers[layer_key][idx]
    X_rna = _maybe_log1p(X_rna, log1p_rna)

    gene_names = adata.var["name"].astype(str).tolist() if "name" in adata.var else adata.var_names.astype(str).tolist()

    # HVGs (simple variance filter on RNA)
    if n_top_hvgs is not None and n_top_hvgs > 0:
        if sparse.issparse(X_rna):
            # approximate var for sparse: use mean of squares - square of means
            mean = X_rna.mean(axis=0).A1
            mean_sq = X_rna.multiply(X_rna).mean(axis=0).A1
            var = mean_sq - mean**2
        else:
            Xr = np.asarray(X_rna)
            var = Xr.var(axis=0)
        top_idx = np.argsort(var)[-n_top_hvgs:]
        X_rna = X_rna[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]

    # SVD reduction on RNA if requested
    svd_model = None
    if svd_components and svd_components > 0:
        svd_model, X_rna_red = _fit_svd(X_rna, n_components=svd_components, random_state=random_state)
        X_rna_block = np.asarray(X_rna_red)
        gene_feat_names = [f"svd_rna_{i}" for i in range(X_rna_block.shape[1])]
    else:
        X_rna_block = _as_dense(X_rna)
        gene_feat_names = gene_names

    parts = [X_rna_block]

    # Concatenate protein features (as-is; you may standardize later)
    prot_feat_names: List[str] = []
    if use_protein_features:
        prot_df = adata.obsm[protein_key].iloc[idx]
        X_prot = prot_df.values
        parts.append(X_prot)
        prot_feat_names = prot_df.columns.astype(str).tolist()

    X_full = _concat_features(parts)
    feature_names = gene_feat_names + prot_feat_names

    y = y_labels[idx]

    # Train/val split
    strat = y if stratify else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y, test_size=val_size, random_state=random_state, stratify=strat
    )

    # Standardize (mean centering is safe because X_tr is dense here)
    X_tr_s, X_val_s, scaler = _standardize_train_val(X_tr, X_val, with_mean=True)

    # TabNet params
    tb_params = dict(n_d=32, n_a=32, n_steps=5, seed=random_state, verbose=1, device_name=device_name)
    if tabnet_params:
        tb_params.update(tabnet_params)

    clf = TabNetClassifier(**tb_params)

    t1 = time.time()
    clf.fit(
        X_tr_s, y_tr,
        eval_set=[(X_val_s, y_val)],
        eval_metric=["accuracy"],
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
    )
    t2 = time.time()

    # Evaluate
    y_pred = clf.predict(X_val_s)
    acc = float(accuracy_score(y_val, y_pred))
    f1 = float(f1_score(y_val, y_pred, average="macro"))
    cm = confusion_matrix(y_val, y_pred)

    # Probabilities on all cells for write-back
    if write_back:
        # Fit scaler on all X_full to transform then predict_proba
        X_all_s = scaler.transform(X_full)
        proba = clf.predict_proba(X_all_s)
        # Map classes
        classes_ = [str(c) for c in clf.classes_]
        # Write per-class probability columns
        proba_keys = []
        for i, cls in enumerate(classes_):
            key = f"{out_key}:proba:{cls}"
            col = np.full(adata.n_obs, np.nan, dtype=float)
            col[idx] = proba[:, i]
            adata.obs[key] = col
            proba_keys.append(key)
        # Pred labels on all cells (argmax)
        pred_all = np.full(adata.n_obs, None, dtype=object)
        pred_all[idx] = np.asarray(classes_)[proba.argmax(axis=1)]
        adata.obs[f"{out_key}:pred"] = pred_all.astype(str)
    else:
        classes_ = [str(c) for c in clf.classes_]
        proba_keys = []

    # Feature importances
    feat_imp = np.asarray(clf.feature_importances_).astype(float).tolist()

    timings = {"total": time.time() - t0, "fit": t2 - t1}

    result = ClusterClassifierResult(
        classes_=classes_,
        accuracy=acc,
        macro_f1=f1,
        confusion_matrix=cm.tolist(),
        feature_names=feature_names,
        feature_importances=feat_imp,
        label_key=f"{out_key}:pred",
        proba_keys=proba_keys,
        n_train=X_tr.shape[0],
        n_val=X_val.shape[0],
        params={
            "layer_key": layer_key,
            "protein_key": protein_key,
            "cluster_key": cluster_key,
            "use_protein_features": use_protein_features,
            "log1p_rna": log1p_rna,
            "n_top_hvgs": n_top_hvgs,
            "svd_components": svd_components,
            "n_cells": n_cells,
            "val_size": val_size,
            "random_state": random_state,
            "tabnet_params": tb_params,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
        timings_sec=timings,
    )

    if write_back:
        adata.uns[out_key] = asdict(result)
        # If features are purely gene-derived (no protein & no SVD), also expose into varm
        only_genes = (not use_protein_features) and (not svd_components)
        if only_genes and len(feature_names) == adata.n_vars:
            adata.varm[f"{out_key}:importance"] = np.asarray(feat_imp)[:, None]

    return result


# ---------------------------------------------------------------------------

def tabnet_protein_within_cluster_classifier(
    adata: AnnData,
    target_cluster: Union[str, int],
    target_protein: str,
    cluster_key: str = "rna_cluster",
    layer_key: str = "rna",
    protein_key: str = "protein",
    # Feature engineering for the *predictors*
    use_protein_features: bool = False,
    extra_proteins: Optional[List[str]] = None,
    log1p_rna: bool = False,
    n_top_hvgs: Optional[int] = None,
    svd_components: Optional[int] = 0,
    # Label creation (high/low split)
    threshold: str = "median",  # or "quantile:0.6" etc.
    # Sampling & splits
    n_cells: Optional[int] = None,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    # TabNet & training
    device_name: str = "cpu",
    tabnet_params: Optional[Dict] = None,
    max_epochs: int = 100,
    batch_size: int = 1024,
    patience: int = 10,
    # Outputs
    write_back: bool = True,
    out_key: str = "tabnet_within_cluster",
) -> WithinClusterResult:
    """
    Within a given cluster, train a TabNet classifier to predict whether
    `target_protein` is high vs low (binary), using RNA features (and optionally
    some proteins). Records AUC and feature importances.

    Writes to `adata.obs` for in-cluster cells:
      - f"{out_key}:score:{cluster}:{protein}" : predicted probability of HIGH

    Writes to `adata.uns[out_key]` (appends/updates list of results).
    """
    if layer_key not in adata.layers:
        raise KeyError(f"`{layer_key}` not found in adata.layers")
    if protein_key not in adata.obsm:
        raise KeyError(f"`{protein_key}` not found in adata.obsm")
    if cluster_key not in adata.obs:
        raise KeyError(f"`{cluster_key}` not found in adata.obs")

    target_cluster = str(target_cluster)

    # Mask cells in target cluster
    mask = adata.obs[cluster_key].astype(str).values == target_cluster
    if not np.any(mask):
        raise ValueError(f"No cells found for cluster '{target_cluster}'.")

    # Subsample if requested (within cluster only)
    idx_all = np.where(mask)[0]
    if n_cells is not None and n_cells < idx_all.size:
        rng = np.random.default_rng(random_state)
        idx_all = np.sort(rng.choice(idx_all, size=n_cells, replace=False))

    # Labels based on target_protein
    prot_df = adata.obsm[protein_key].iloc[idx_all]
    if target_protein not in prot_df.columns:
        raise KeyError(f"Protein '{target_protein}' not found in adata.obsm['{protein_key}'] columns.")
    values = prot_df[target_protein].values.astype(float)

    thr = None
    if threshold == "median":
        thr = float(np.median(values))
    elif threshold.startswith("quantile:"):
        q = float(threshold.split(":", 1)[1])
        thr = float(np.quantile(values, q))
    else:
        try:
            thr = float(threshold)
        except Exception as e:
            raise ValueError("threshold must be 'median', 'quantile:<q>' or a float value")
    y = (values > thr).astype(int)

    # Build predictors
    X_rna = adata.layers[layer_key][idx_all]
    X_rna = _maybe_log1p(X_rna, log1p_rna)
    gene_names = adata.var["name"].astype(str).tolist() if "name" in adata.var else adata.var_names.astype(str).tolist()

    if n_top_hvgs is not None and n_top_hvgs > 0:
        if sparse.issparse(X_rna):
            mean = X_rna.mean(axis=0).A1
            mean_sq = X_rna.multiply(X_rna).mean(axis=0).A1
            var = mean_sq - mean**2
        else:
            Xr = np.asarray(X_rna)
            var = Xr.var(axis=0)
        top_idx = np.argsort(var)[-n_top_hvgs:]
        X_rna = X_rna[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]

    svd_model = None
    if svd_components and svd_components > 0:
        svd_model, X_rna_red = _fit_svd(X_rna, n_components=svd_components, random_state=random_state)
        X_rna_block = np.asarray(X_rna_red)
        gene_feat_names = [f"svd_rna_{i}" for i in range(X_rna_block.shape[1])]
    else:
        X_rna_block = _as_dense(X_rna)
        gene_feat_names = gene_names

    parts = [X_rna_block]
    prot_feat_names: List[str] = []
    if use_protein_features:
        use_cols = None
        if extra_proteins:
            missing = [c for c in extra_proteins if c not in prot_df.columns]
            if missing:
                warnings.warn(f"Some extra_proteins not found and will be ignored: {missing}")
            use_cols = [c for c in extra_proteins or [] if c in prot_df.columns]
        else:
            use_cols = [c for c in prot_df.columns if c != target_protein]
        Xp = prot_df[use_cols].values if use_cols else None
        if Xp is not None and Xp.size > 0:
            parts.append(Xp)
            prot_feat_names = list(use_cols)

    X_full = _concat_features(parts)
    feature_names = gene_feat_names + prot_feat_names

    # Split
    strat = y if stratify else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y, test_size=val_size, random_state=random_state, stratify=strat
    )

    # Standardize
    X_tr_s, X_val_s, scaler = _standardize_train_val(X_tr, X_val, with_mean=True)

    # Train TabNet
    tb_params = dict(n_d=32, n_a=32, n_steps=5, seed=random_state, verbose=1, device_name=device_name)
    if tabnet_params:
        tb_params.update(tabnet_params)
    clf = TabNetClassifier(**tb_params)

    t1 = time.time()
    clf.fit(
        X_tr_s, y_tr,
        eval_set=[(X_val_s, y_val)],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
    )
    t2 = time.time()

    # Evaluate
    y_pred = clf.predict(X_val_s)
    y_proba = clf.predict_proba(X_val_s)[:, 1]
    acc = float(accuracy_score(y_val, y_pred))
    f1 = float(f1_score(y_val, y_pred, average="macro"))
    try:
        auc = float(roc_auc_score(y_val, y_proba))
    except ValueError:
        auc = float('nan')  # if only one class present in y_val

    feat_imp = np.asarray(clf.feature_importances_).astype(float).tolist()

    score_key = f"{out_key}:score:{target_cluster}:{target_protein}"

    if write_back:
        # Write probability (HIGH) for in-cluster cells
        X_all_s = scaler.transform(X_full)
        proba_all = clf.predict_proba(X_all_s)[:, 1]
        col = np.full(adata.n_obs, np.nan, dtype=float)
        col[idx_all] = proba_all
        adata.obs[score_key] = col

        # Append result under uns
        payload = asdict(WithinClusterResult(
            cluster=target_cluster,
            protein=target_protein,
            auc=auc,
            accuracy=acc,
            macro_f1=f1,
            feature_names=feature_names,
            feature_importances=feat_imp,
            score_key=score_key,
            n_train=X_tr.shape[0],
            n_val=X_val.shape[0],
            params={
                "cluster_key": cluster_key,
                "layer_key": layer_key,
                "protein_key": protein_key,
                "use_protein_features": use_protein_features,
                "extra_proteins": extra_proteins,
                "log1p_rna": log1p_rna,
                "n_top_hvgs": n_top_hvgs,
                "svd_components": svd_components,
                "threshold": threshold,
                "n_cells": n_cells,
                "val_size": val_size,
                "random_state": random_state,
                "tabnet_params": tb_params,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "patience": patience,
            },
            timings_sec={"fit": t2 - t1},
        ))
        if out_key not in adata.uns:
            adata.uns[out_key] = {"results": [payload]}
        else:
            bucket = adata.uns[out_key]
            if "results" not in bucket or not isinstance(bucket["results"], list):
                bucket["results"] = []
            bucket["results"].append(payload)

    return WithinClusterResult(
        cluster=target_cluster,
        protein=target_protein,
        auc=auc,
        accuracy=acc,
        macro_f1=f1,
        feature_names=feature_names,
        feature_importances=feat_imp,
        score_key=score_key,
        n_train=X_tr.shape[0],
        n_val=X_val.shape[0],
        params={
            "cluster_key": cluster_key,
            "layer_key": layer_key,
            "protein_key": protein_key,
            "use_protein_features": use_protein_features,
            "extra_proteins": extra_proteins,
            "log1p_rna": log1p_rna,
            "n_top_hvgs": n_top_hvgs,
            "svd_components": svd_components,
            "threshold": threshold,
            "n_cells": n_cells,
            "val_size": val_size,
            "random_state": random_state,
            "tabnet_params": tb_params,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
        timings_sec={"fit": t2 - t1},
    )


# ---------------------------------------------------------------------------
# Convenience: mapping importances to names
# ---------------------------------------------------------------------------

def tabnet_top_features(result: Union[ClusterClassifierResult, WithinClusterResult], top_k: int = 20) -> pd.DataFrame:
    """Return a DataFrame of top-k features sorted by importance (desc)."""
    feat = np.asarray(result.feature_importances)
    names = list(result.feature_names)
    order = np.argsort(feat)[::-1][:top_k]
    return pd.DataFrame({"feature": [names[i] for i in order], "importance": feat[order]})
