"""Utilities for joint RNA + protein analysis on Xenium AnnData objects.

This module provides a high-level pipeline that clusters cells using the RNA
expression matrix and, for each protein marker, trains a small neural network
to explain high-vs-low protein states within every RNA-defined cluster.

The function is intentionally self-contained and only relies on scikit-learn
primitives so that it can operate on large Xenium datasets without requiring
extra dependencies (e.g. Scanpy).  It works directly on the data structures
produced by :func:`pyXenium.io.xenium_gene_protein_loader.load_xenium_gene_protein`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ProteinModelResult:
    """Container for the protein classification model of one cluster."""

    protein: str
    cluster: str
    threshold: float
    n_cells: int
    n_high: int
    n_low: int
    train_accuracy: float
    test_accuracy: float
    test_auc: float
    model: MLPClassifier
    scaler: StandardScaler


def _get_rna_matrix(adata: AnnData):
    """Return the raw RNA matrix from ``adata`` as CSR sparse matrix."""

    if "rna" in adata.layers:
        X = adata.layers["rna"]
    else:
        X = adata.X

    if sparse.issparse(X):
        return X.tocsr()

    # Dense array – convert to CSR to keep operations memory friendly.
    return sparse.csr_matrix(np.asarray(X))


def _normalize_log1p(matrix: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    """Library-size normalisation followed by log1p for sparse matrices."""

    matrix = matrix.astype(np.float32)
    cell_sums = np.array(matrix.sum(axis=1)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    inv = sparse.diags((target_sum / cell_sums).astype(np.float32))
    norm = inv @ matrix
    norm.data = np.log1p(norm.data)
    return norm


def _fit_pcs(
    matrix: sparse.csr_matrix,
    n_components: int,
    random_state: Optional[int],
) -> np.ndarray:
    """Fit TruncatedSVD on the RNA matrix and return dense principal components."""

    n_features = matrix.shape[1]
    if n_features <= 1:
        raise ValueError("RNA matrix must contain at least two genes for SVD.")

    n_components = max(1, min(n_components, n_features - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    pcs = svd.fit_transform(matrix)
    return pcs.astype(np.float32)


def _resolve_protein_frame(adata: AnnData) -> pd.DataFrame:
    """Return the protein matrix stored in ``adata.obsm['protein']`` as DataFrame."""

    if "protein" not in adata.obsm:
        raise KeyError("AnnData is missing 'protein' modality in adata.obsm['protein'].")

    protein_df = adata.obsm["protein"]
    if isinstance(protein_df, pd.DataFrame):
        return protein_df

    return pd.DataFrame(np.asarray(protein_df), index=adata.obs_names)


def rna_protein_cluster_analysis(
    adata: AnnData,
    *,
    n_clusters: int = 12,
    n_pcs: int = 30,
    cluster_key: str = "rna_cluster",
    random_state: Optional[int] = 0,
    target_sum: float = 1e4,
    min_cells_per_cluster: int = 50,
    min_cells_per_group: int = 20,
    protein_split_method: str = "median",
    protein_quantile: float = 0.75,
    test_size: float = 0.2,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 200,
    early_stopping: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, ProteinModelResult]]]:
    """Joint RNA/protein analysis for Xenium AnnData objects.

    The pipeline performs three consecutive steps:

    1. **RNA preprocessing** – library-size normalisation (counts per ``target_sum``)
       followed by ``log1p``.  A :class:`~sklearn.decomposition.TruncatedSVD`
       is fitted to obtain ``n_pcs`` latent dimensions.
    2. **Clustering** – :class:`~sklearn.cluster.KMeans` is applied on the latent
       representation to create ``n_clusters`` RNA-driven cell groups.  Cluster
       assignments are stored in ``adata.obs[cluster_key]`` and the latent space
       in ``adata.obsm['X_rna_pca']``.
    3. **Protein explanation** – for every cluster and every protein marker, the
       cells are divided into "high" vs. "low" groups (median split by default).
       A small neural network (:class:`~sklearn.neural_network.MLPClassifier`)
       is trained to predict the binary labels from the RNA latent features.  The
       training/test accuracies and optional ROC-AUC are reported.

    Parameters
    ----------
    adata:
        AnnData object returned by
        :func:`pyXenium.io.xenium_gene_protein_loader.load_xenium_gene_protein`.
        Requires ``adata.layers['rna']`` (or ``adata.X``) and
        ``adata.obsm['protein']``.
    n_clusters:
        Number of RNA clusters to compute with KMeans.
    n_pcs:
        Number of latent components extracted with TruncatedSVD.  The value is
        automatically capped at ``n_genes - 1``.
    cluster_key:
        Column name added to ``adata.obs`` that stores cluster labels.
    random_state:
        Seed for the SVD, KMeans and neural networks.  Use ``None`` for random
        initialisation.
    target_sum:
        Target library size after normalisation (Counts Per ``target_sum``).
    min_cells_per_cluster:
        Clusters with fewer cells are skipped entirely.
    min_cells_per_group:
        Minimum number of cells required in both "high" and "low" protein
        groups to train a neural network.
    protein_split_method:
        Either ``"median"`` (default) for a median split or ``"quantile"`` to
        keep only the top ``protein_quantile`` and bottom ``1 - protein_quantile``
        fractions of cells (discarding the middle portion).
    protein_quantile:
        Quantile used when ``protein_split_method='quantile'``.
    test_size:
        Fraction of the cluster reserved for the test split when training the
        neural network.
    hidden_layer_sizes:
        Hidden-layer configuration passed to :class:`MLPClassifier`.
    max_iter:
        Maximum number of training iterations for the neural network.
    early_stopping:
        Whether to use early stopping in :class:`MLPClassifier`.

    Returns
    -------
    summary:
        :class:`pandas.DataFrame` summarising the trained models.  Columns are
        ``['cluster', 'protein', 'threshold', 'n_cells', 'n_high', 'n_low',
        'train_accuracy', 'test_accuracy', 'test_auc']``.
    models:
        Nested dictionary ``{cluster -> {protein -> ProteinModelResult}}``
        containing the fitted neural networks and scalers for downstream use.

    Examples
    --------
    >>> from pyXenium.analysis import rna_protein_cluster_analysis
    >>> summary, models = rna_protein_cluster_analysis(adata, n_clusters=8)
    >>> summary.head()
          cluster          protein  threshold  n_cells  ...  test_accuracy  test_auc
    0    cluster_0      EPCAM (µm)   0.563100      512  ...           0.84      0.91
    1    cluster_0  Podocin (µm^2)   0.118775      512  ...           0.79      0.87
    """

    if adata.n_obs == 0:
        raise ValueError("AnnData contains no cells (n_obs == 0).")

    protein_df = _resolve_protein_frame(adata)
    if protein_df.shape[1] == 0:
        raise ValueError("AnnData.obsm['protein'] is empty – nothing to analyse.")

    rna_csr = _get_rna_matrix(adata)
    if rna_csr.shape[1] < 2:
        raise ValueError("RNA modality must have at least two genes for clustering.")

    log_norm = _normalize_log1p(rna_csr, target_sum=target_sum)
    pcs = _fit_pcs(log_norm, n_components=n_pcs, random_state=random_state)
    adata.obsm["X_rna_pca"] = pcs

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(pcs)
    cluster_names = np.array([f"cluster_{i}" for i in cluster_labels])
    adata.obs[cluster_key] = cluster_names

    results: List[Dict[str, float]] = []
    models: Dict[str, Dict[str, ProteinModelResult]] = {}

    unique_clusters = pd.Index(np.unique(cluster_names))
    for cluster in unique_clusters:
        idx = np.where(cluster_names == cluster)[0]
        if idx.size < min_cells_per_cluster:
            continue

        cluster_pcs = pcs[idx]
        cluster_protein = protein_df.iloc[idx]
        models.setdefault(cluster, {})

        for protein in cluster_protein.columns:
            values_all = cluster_protein[protein].to_numpy(dtype=np.float32)
            finite_mask = np.isfinite(values_all)
            if finite_mask.sum() < min_cells_per_group * 2:
                continue

            values = values_all[finite_mask]
            X_cluster = cluster_pcs[finite_mask]

            if protein_split_method == "median":
                threshold = float(np.median(values))
                labels = (values >= threshold).astype(int)
            elif protein_split_method == "quantile":
                q = float(protein_quantile)
                if not 0.5 < q < 1.0:
                    raise ValueError("protein_quantile must be between 0.5 and 1.0 (exclusive).")
                high_thr = np.quantile(values, q)
                low_mask = values <= np.quantile(values, 1.0 - q)
                high_mask = values >= high_thr
                selected_mask = high_mask | low_mask
                if selected_mask.sum() < min_cells_per_group * 2:
                    continue
                values = values[selected_mask]
                X_cluster = X_cluster[selected_mask]
                labels = high_mask[selected_mask].astype(int)
                threshold = float(high_thr)
            else:
                raise ValueError("protein_split_method must be 'median' or 'quantile'.")

            n_selected = labels.size
            n_high = int(labels.sum())
            n_low = int(n_selected - n_high)

            if n_high < min_cells_per_group or n_low < min_cells_per_group:
                continue

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_cluster,
                    labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=labels,
                )
            except ValueError:
                # Not enough samples to stratify.
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=random_state,
                max_iter=max_iter,
                early_stopping=early_stopping,
            )

            try:
                clf.fit(X_train_scaled, y_train)
            except Exception:
                continue

            train_acc = float(accuracy_score(y_train, clf.predict(X_train_scaled)))
            test_pred = clf.predict(X_test_scaled)
            test_acc = float(accuracy_score(y_test, test_pred))

            if hasattr(clf, "predict_proba") and len(np.unique(y_test)) == 2:
                probs = clf.predict_proba(X_test_scaled)[:, 1]
                try:
                    test_auc = float(roc_auc_score(y_test, probs))
                except ValueError:
                    test_auc = float("nan")
            else:
                test_auc = float("nan")

            result = ProteinModelResult(
                protein=protein,
                cluster=cluster,
                threshold=threshold,
                n_cells=n_selected,
                n_high=n_high,
                n_low=n_low,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                test_auc=test_auc,
                model=clf,
                scaler=scaler,
            )

            models[cluster][protein] = result
            results.append(
                {
                    "cluster": cluster,
                    "protein": protein,
                    "threshold": threshold,
                    "n_cells": n_selected,
                    "n_high": n_high,
                    "n_low": n_low,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "test_auc": test_auc,
                }
            )

    summary = pd.DataFrame(results, columns=[
        "cluster",
        "protein",
        "threshold",
        "n_cells",
        "n_high",
        "n_low",
        "train_accuracy",
        "test_accuracy",
        "test_auc",
    ])

    return summary, models


__all__ = ["rna_protein_cluster_analysis", "ProteinModelResult"]

