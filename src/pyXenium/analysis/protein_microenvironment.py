# -*- coding: utf-8 -*-
"""
pyXenium/analysis/protein_microenvironment.py

Protein microenvironment analysis within an RNA cluster for Xenium RNA+Protein data.

Features
--------
- Within a chosen RNA cluster, split cells into protein-high vs protein-low by automatic thresholding (GMM / quantile).
- Build a spatial graph *within the cluster* for Moran's I (spatial autocorrelation of protein).
- Build a *global* KDTree (no full NxN adjacency) and compute neighborhood composition of the chosen cluster cells
  against a grouping key (e.g., obs['cluster'] or obs['cell_type']) to quantify *microenvironment*.
- Permutation-based neighbor enrichment test: compare neighborhood fractions of protein-high vs protein-low cells.
- Differential expression (RNA) between protein-high vs protein-low *within the cluster* (with built-in normalize+log1p).
- Predictability of protein-high from neighborhood composition (logistic regression, AUC + feature coefficients).
- Robust spatial plotting for both numeric vectors and categorical labels (including NaN handling).

Large-scale notes
-----------------
- Global microenvironment is computed with KDTree radius queries (no full CSR NxN matrix), which scales well for
  hundreds of thousands of cells. The intra-cluster adjacency (CSR) is only used for Moran's I on a subset.
- The heuristic radius estimator samples when n_cells is very large to avoid memory spikes.

Author: (c) 2025
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, List

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _get_coords(adata: ad.AnnData,
                prefer_obsm: str = "spatial",
                obs_xy: Tuple[str, str] = ("x_centroid", "y_centroid")) -> np.ndarray:
    """Return (n_cells x 2) spatial coordinates as float array."""
    if prefer_obsm in adata.obsm_keys():
        arr = np.asarray(adata.obsm[prefer_obsm], dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"obsm['{prefer_obsm}'] must be of shape (n_cells, >=2).")
        return arr[:, :2]
    xk, yk = obs_xy
    if xk in adata.obs.columns and yk in adata.obs.columns:
        return adata.obs[[xk, yk]].to_numpy(dtype=float)
    raise KeyError(f"Cannot find spatial coords in obsm['{prefer_obsm}'] nor obs[{xk},{yk}].")


def _clr(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centered log-ratio per row."""
    mat = np.asarray(mat, dtype=float) + eps
    logm = np.log(mat)
    return logm - logm.mean(axis=1, keepdims=True)


def _arcsinh(mat: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    return np.arcsinh(np.asarray(mat, dtype=float) / cofactor)


def _auto_radius(coords: np.ndarray, k: int = 8, factor: float = 1.5, max_n: int = 200_000) -> float:
    """
    Heuristic radius: median distance to k-th nearest neighbor * factor.
    If very large, sample up to max_n cells for robustness and speed.
    """
    n = coords.shape[0]
    if n > max_n:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=max_n, replace=False)
        X = coords[sel]
    else:
        X = coords
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]-1), algorithm="kd_tree").fit(X)
    d, _ = nn.kneighbors(X)
    kth = d[:, min(k, d.shape[1] - 1)]
    return float(np.median(kth) * factor)


def _build_radius_graph(coords: np.ndarray, radius: float) -> sp.csr_matrix:
    """Symmetric binary adjacency within radius on a *given subset* (for Moran's I)."""
    tree = cKDTree(coords)
    neigh = tree.query_ball_point(coords, r=radius)
    rows, cols = [], []
    for i, lst in enumerate(neigh):
        for j in lst:
            if i != j:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0]))
    A = ((A + A.T) > 0).astype(np.float32)
    return A


def _morans_I(values: np.ndarray, W: sp.csr_matrix, permutations: int = 999, random_state: int = 0) -> Dict[str, float]:
    """
    Moran's I with permutation test.
    values: (n,) vector, W: (n x n) sparse adjacency (binary/weighted).
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(values, dtype=float).reshape(-1)
    m = np.isfinite(x)
    x = x[m]
    W = W[m][:, m]
    n = x.size
    if n < 5 or W.nnz == 0:
        return {"I": np.nan, "p_value": np.nan}
    xc = x - x.mean()
    S0 = float(W.sum())
    num = float(xc.T @ (W @ xc))
    den = float((xc ** 2).sum())
    I = (n / S0) * (num / den)

    perm = np.empty(permutations, dtype=float)
    for b in range(permutations):
        xp = rng.permutation(xc)
        perm[b] = (n / S0) * float(xp.T @ (W @ xp)) / den
    p = (np.sum(np.abs(perm) >= np.abs(I)) + 1) / (permutations + 1)
    return {"I": float(I), "p_value": float(p)}


def _gmm_threshold(x: np.ndarray, random_state: int = 0) -> float:
    """Two-component GMM threshold = mean(midpoint) of two component means."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values for thresholding.")
    gmm = GaussianMixture(n_components=2, random_state=random_state)
    gmm.fit(x.reshape(-1, 1))
    m1, m2 = np.sort(gmm.means_.ravel())
    return float((m1 + m2) / 2.0)


def _labels_and_categories(adata: ad.AnnData, group_key: str) -> Tuple[np.ndarray, List[str]]:
    """Return integer labels and category list for obs[group_key]."""
    ser = adata.obs[group_key].astype(str)
    cats = ser.unique().tolist()
    mp = {c: i for i, c in enumerate(cats)}
    lab = ser.map(mp).to_numpy()
    return lab, cats


# ---------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------

@dataclass
class ProteinMicroEnv:
    """
    Microenvironment analysis for protein heterogeneity within an RNA cluster.
    """
    adata: ad.AnnData
    protein_obsm: str = "protein"           # adata.obsm key for raw protein matrix
    protein_norm_obsm: str = "protein_norm" # adata.obsm key for normalized protein
    cluster_key: str = "cluster"            # obs key for RNA clusters
    spatial_obsm: str = "spatial"           # prefer this; fallback to obs[x_centroid,y_centroid]
    obs_xy: Tuple[str, str] = ("x_centroid", "y_centroid")
    random_state: int = 0

    # ---------------- Normalization ----------------

    def normalize_protein(self, method: str = "clr", cofactor: float = 5.0) -> None:
        """Store normalized protein matrix in obsm[protein_norm_obsm]."""
        if self.protein_obsm not in self.adata.obsm_keys():
            raise KeyError(f"Protein matrix not found in obsm['{self.protein_obsm}'].")
        prot = self.adata.obsm[self.protein_obsm]
        if isinstance(prot, pd.DataFrame):
            mat = prot.to_numpy(dtype=float)
            cols = prot.columns
        else:
            mat = np.asarray(prot, dtype=float)
            cols = [f"prot_{i}" for i in range(mat.shape[1])]
        if method == "clr":
            arr = _clr(mat)
        elif method == "arcsinh":
            arr = _arcsinh(mat, cofactor=cofactor)
        else:
            raise ValueError("method must be 'clr' or 'arcsinh'.")
        self.adata.obsm[self.protein_norm_obsm] = pd.DataFrame(arr, index=self.adata.obs_names, columns=cols)

    # ---------------- Protein high/low split ----------------

    def split_high_low(self,
                       cluster_id: Union[str, int],
                       protein: str,
                       method: str = "gmm",
                       quantile: float = 0.5) -> Tuple[pd.DataFrame, str, float]:
        """
        Split within a cluster into protein_high vs protein_low by thresholding.
        Returns (table, status_col, threshold).
        """
        if self.cluster_key not in self.adata.obs.columns:
            raise KeyError(f"'{self.cluster_key}' not in adata.obs.")
        # choose normalized if present
        key = self.protein_norm_obsm if self.protein_norm_obsm in self.adata.obsm_keys() else self.protein_obsm
        if key not in self.adata.obsm_keys():
            raise KeyError(f"Protein matrix not found in obsm['{key}'].")
        dfp = self.adata.obsm[key]
        if protein not in dfp.columns:
            raise KeyError(f"Protein '{protein}' not found in obsm['{key}'].")

        mask = self.adata.obs[self.cluster_key].astype(str) == str(cluster_id)
        idx = self.adata.obs_names[mask]
        vals = dfp.loc[idx, protein].to_numpy()

        if method == "gmm":
            thr = _gmm_threshold(vals, random_state=self.random_state)
        elif method == "quantile":
            thr = float(np.quantile(vals[np.isfinite(vals)], quantile))
        else:
            raise ValueError("method must be 'gmm' or 'quantile'.")

        status = np.where(vals >= thr, "protein_high", "protein_low")
        col = f"{protein}__status_in_cluster_{cluster_id}"
        self.adata.obs.loc[idx, col] = status
        out = pd.DataFrame({"protein_value": vals, "protein_status": status}, index=idx)
        return out, col, thr

    # ---------------- Spatial: cluster-only (for Moran's I) ----------------

    def build_spatial_graph(self,
                            cluster_id: Union[str, int],
                            radius: Optional[float] = None) -> Tuple[sp.csr_matrix, pd.Index, float]:
        """CSR adjacency within the chosen cluster; used for Moran's I."""
        coords = _get_coords(self.adata, self.spatial_obsm, self.obs_xy)
        mask = self.adata.obs[self.cluster_key].astype(str) == str(cluster_id)
        coords_sub = coords[mask]
        names_sub = self.adata.obs_names[mask]
        if coords_sub.shape[0] < 5:
            raise ValueError("Too few cells in the selected cluster.")
        if radius is None:
            radius = _auto_radius(coords_sub, k=8, factor=1.5)
        A = _build_radius_graph(coords_sub, radius=radius)
        # store meta
        self.adata.uns.setdefault("protein_microenv", {})
        self.adata.uns["protein_microenv"][f"cluster_{cluster_id}__radius_cluster"] = float(radius)
        self.adata.uns["protein_microenv"][f"cluster_{cluster_id}__A_shape"] = A.shape
        return A, names_sub, float(radius)

    # ---------------- Global KDTree for microenvironment ----------------

    def build_kdtree_full(self, radius: Optional[float] = None) -> Tuple[cKDTree, np.ndarray, float]:
        """
        Build a KDTree on all cells and decide a heuristic radius if not provided.
        Returns (tree, coords_all, radius_used).
        """
        coords = _get_coords(self.adata, self.spatial_obsm, self.obs_xy)
        r = _auto_radius(coords, k=8, factor=1.5) if radius is None else float(radius)
        tree = cKDTree(coords)
        self.adata.uns.setdefault("protein_microenv", {})
        self.adata.uns["protein_microenv"]["radius_global"] = float(r)
        return tree, coords, float(r)

    def neighbor_composition_global(self,
                                    focus_index: pd.Index,
                                    tree: cKDTree,
                                    coords_all: np.ndarray,
                                    radius: float,
                                    group_key: str) -> pd.DataFrame:
        """
        Compute neighborhood composition for *focus_index* cells against all cells (global KDTree),
        grouped by obs[group_key].
        """
        labels_all, cats = _labels_and_categories(self.adata, group_key)
        k = len(cats)
        # mapping obs_names -> integer row
        pos = pd.Series(np.arange(coords_all.shape[0]), index=self.adata.obs_names)
        rows = pos.loc[focus_index].to_numpy()

        counts = np.zeros((rows.size, k), dtype=float)
        deg = np.zeros(rows.size, dtype=float)

        for i, r in enumerate(rows):
            nbrs = tree.query_ball_point(coords_all[r], r=radius)
            # remove self if present
            if r in nbrs:
                nbrs.remove(r)
            if len(nbrs) == 0:
                continue
            lab = labels_all[nbrs]
            cnt = np.bincount(lab, minlength=k)
            counts[i, :] = cnt
            deg[i] = cnt.sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            fracs = np.divide(counts, deg[:, None], where=deg[:, None] > 0)
            fracs[~np.isfinite(fracs)] = 0.0

        dfc = pd.DataFrame(counts, index=focus_index, columns=[f"nbr_count:{c}" for c in cats])
        dff = pd.DataFrame(fracs,  index=focus_index, columns=[f"nbr_frac:{c}"  for c in cats])
        return pd.concat([dfc, dff], axis=1)

    # ---------------- Neighbor enrichment / prediction (from composition) ----------------

    def neighbor_enrichment_from_comp(self,
                                      focus_index: pd.Index,
                                      comp: pd.DataFrame,
                                      protein_status_col: str,
                                      permutations: int = 999,
                                      random_state: int = 0) -> pd.DataFrame:
        """Permutation test on neighborhood fractions between protein-high vs protein-low (within focus_index)."""
        frac_cols = [c for c in comp.columns if c.startswith("nbr_frac:")]
        y = self.adata.obs.loc[focus_index, protein_status_col].astype(str)
        mask = y.isin(["protein_high", "protein_low"])
        y = y[mask].to_numpy()
        X = comp.loc[focus_index].iloc[mask.to_numpy()][frac_cols]

        is_high = (y == "protein_high")
        if is_high.sum() == 0 or (~is_high).sum() == 0:
            warnings.warn("One of the groups (high/low) is empty after filtering; skipping enrichment.")
            return pd.DataFrame(columns=["neighbor_type", "delta_frac_high_minus_low", "z_score", "p_value", "q_value"])

        diff_obs = X.loc[is_high, :].mean(axis=0) - X.loc[~is_high, :].mean(axis=0)

        rng = np.random.default_rng(random_state)
        perm = np.zeros((permutations, X.shape[1]), dtype=float)
        for b in range(permutations):
            yp = rng.permutation(is_high)
            perm[b, :] = X.loc[yp, :].mean(axis=0).to_numpy() - X.loc[~yp, :].mean(axis=0).to_numpy()

        mu = perm.mean(axis=0); sd = perm.std(axis=0, ddof=1) + 1e-9
        z = (diff_obs.to_numpy() - mu) / sd
        p = np.mean(np.abs(perm - mu) >= np.abs(diff_obs.to_numpy() - mu), axis=0)
        _, q, _, _ = multipletests(p, method="fdr_bh")

        out = pd.DataFrame({
            "neighbor_type": [c.replace("nbr_frac:", "") for c in X.columns],
            "delta_frac_high_minus_low": diff_obs.values,
            "z_score": z,
            "p_value": p,
            "q_value": q
        }).sort_values("q_value")
        return out

    def microenv_predict_from_comp(self,
                                   focus_index: pd.Index,
                                   comp: pd.DataFrame,
                                   protein_status_col: str,
                                   C: float = 1.0,
                                   max_iter: int = 1000) -> Dict[str, Union[float, pd.DataFrame]]:
        """Predict protein-high from neighborhood fractions; report AUC and feature coefficients."""
        frac_cols = [c for c in comp.columns if c.startswith("nbr_frac:")]
        y = self.adata.obs.loc[focus_index, protein_status_col].astype(str)
        mask = y.isin(["protein_high", "protein_low"])
        y = (y[mask].to_numpy() == "protein_high").astype(int)
        X = comp.loc[focus_index].iloc[mask.to_numpy()][frac_cols].to_numpy()

        if X.shape[0] < 10 or X.shape[1] == 0 or y.sum() == 0 or y.sum() == y.shape[0]:
            return {"auc": np.nan, "coef": pd.DataFrame({"feature": frac_cols, "coef": np.nan})}

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xz = scaler.fit_transform(X)
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=self.random_state)
        clf.fit(Xz, y)
        auc = roc_auc_score(y, clf.predict_proba(Xz)[:, 1])
        coef = pd.DataFrame({"feature": frac_cols, "coef": clf.coef_.ravel()}).sort_values("coef", ascending=False)
        return {"auc": float(auc), "coef": coef}

    # ---------------- Moran's I for protein (cluster-only) ----------------

    def protein_moransI(self,
                        names_sub: pd.Index,
                        A_sub: sp.csr_matrix,
                        protein: str,
                        use_norm: bool = True,
                        permutations: int = 999) -> Dict[str, float]:
        """Moran's I of protein expression within the cluster."""
        key = self.protein_norm_obsm if (use_norm and self.protein_norm_obsm in self.adata.obsm_keys()) else self.protein_obsm
        dfp = self.adata.obsm[key]
        if protein not in dfp.columns:
            raise KeyError(f"Protein '{protein}' not found in obsm['{key}'].")
        vals = dfp.loc[names_sub, protein].to_numpy()
        return _morans_I(vals, A_sub, permutations=permutations, random_state=self.random_state)

    # ---------------- Differential expression (within cluster) ----------------

    def de_within_cluster(self,
                          cluster_id: Union[str, int],
                          protein_status_col: str,
                          layer: str = "rna",
                          method: str = "wilcoxon",
                          n_top_hvg: int = 3000) -> pd.DataFrame:
        """
        RNA DE within a cluster between protein-high vs protein-low.
        Auto normalize_total + log1p (avoid raw-count warning), optional HVG selection for speed.
        """
        mask = self.adata.obs[self.cluster_key].astype(str) == str(cluster_id)
        ad_sub = self.adata[mask].copy()
        if layer in ad_sub.layers:
            ad_sub.X = ad_sub.layers[layer].copy()

        # Normalize + log1p for DE stability
        sc.pp.normalize_total(ad_sub, target_sum=1e4)
        sc.pp.log1p(ad_sub)

        # Optional HVG for speed
        try:
            sc.pp.highly_variable_genes(ad_sub, n_top_genes=n_top_hvg, flavor="seurat_v3")
            ad_sub = ad_sub[:, ad_sub.var["highly_variable"]].copy()
        except Exception:
            pass

        if protein_status_col not in ad_sub.obs.columns:
            raise KeyError(f"'{protein_status_col}' not found in obs (cluster subset).")
        sc.tl.rank_genes_groups(ad_sub, groupby=protein_status_col, method=method, pts=True, use_raw=False)

        keys = ["names", "scores", "logfoldchanges", "pvals_adj", "pvals", "pts", "pts_rest"]
        out = []
        for grp in ad_sub.uns["rank_genes_groups"]["names"].dtype.names:
            df = pd.DataFrame({k: ad_sub.uns["rank_genes_groups"][k][grp] for k in keys})
            df["group"] = grp
            out.append(df)
        return pd.concat(out, axis=0, ignore_index=True)

    # ---------------- Plotting ----------------

    def plot_spatial(self,
                     color: Union[str, np.ndarray],
                     title: str = "",
                     s: float = 2.0,
                     alpha: float = 0.9,
                     cmap: str = "viridis") -> None:
        """Robust spatial scatter for numeric or categorical obs/obsm fields."""
        coords = _get_coords(self.adata, self.spatial_obsm, self.obs_xy)

        def _scatter_num(cnum, cm):
            plt.figure(figsize=(6, 6))
            sca = plt.scatter(coords[:, 0], coords[:, 1], c=cnum, s=s, alpha=alpha, cmap=cm)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.colorbar(sca, shrink=0.8)
            plt.title(title); plt.xlabel("x"); plt.ylabel("y")
            plt.tight_layout(); plt.show()

        if isinstance(color, str):
            # obs column?
            if color in self.adata.obs.columns:
                ser = self.adata.obs[color]
                if ser.dtype.kind in "biufc":  # numeric
                    _scatter_num(ser.to_numpy(), cmap)
                    return
                # categorical / string
                cat = ser.astype("category")
                codes = cat.cat.codes.to_numpy()  # NaN -> -1
                mask = codes != -1
                base = plt.get_cmap("tab20", len(cat.cat.categories) if len(cat.cat.categories) > 0 else 1)
                plt.figure(figsize=(6, 6))
                if mask.any():
                    sca = plt.scatter(coords[mask, 0], coords[mask, 1], c=codes[mask],
                                      s=s, alpha=alpha, cmap=base)
                    cb = plt.colorbar(sca, shrink=0.8)
                    cb.set_ticks(np.arange(len(cat.cat.categories)))
                    cb.set_ticklabels(list(cat.cat.categories))
                if (~mask).any():
                    plt.scatter(coords[~mask, 0], coords[~mask, 1], c="lightgrey", s=s, alpha=alpha)
                plt.gca().set_aspect("equal", adjustable="box")
                plt.title(title); plt.xlabel("x"); plt.ylabel("y")
                plt.tight_layout(); plt.show()
                return

            # protein columns?
            for key in (self.protein_norm_obsm, self.protein_obsm):
                if key in self.adata.obsm_keys():
                    df = self.adata.obsm[key]
                    if isinstance(df, pd.DataFrame) and color in df.columns:
                        _scatter_num(df[color].to_numpy(), cmap)
                        return
            raise KeyError(f"color '{color}' not found in obs nor obsm protein columns.")
        else:
            arr = np.asarray(color)
            _scatter_num(arr, cmap)

    # ---------------- One-stop analysis pipeline ----------------

    def analyze(self,
                cluster_id: Union[str, int],
                protein: str,
                group_key: str = "cluster",
                radius: Optional[float] = None,
                status_method: str = "gmm",
                status_quantile: float = 0.5,
                de_layer: str = "rna",
                permutations: int = 999,
                save_dir: Optional[str] = None) -> Dict[str, object]:
        """
        Run the full microenvironment pipeline.

        Returns dict with keys:
            - status_col, threshold
            - radius_cluster, radius_global
            - neighbor_enrichment (DataFrame)
            - moransI (dict)
            - de (DataFrame)
            - predict_auc (float), predict_coef (DataFrame)
        """
        # 1) protein normalize if missing
        if self.protein_norm_obsm not in self.adata.obsm_keys():
            self.normalize_protein(method="clr")

        # 2) split protein high/low (write status only for cluster cells)
        split_df, status_col, thr = self.split_high_low(cluster_id, protein,
                                                        method=status_method, quantile=status_quantile)

        # 3) Moran's I within cluster
        A_sub, names_sub, r_cluster = self.build_spatial_graph(cluster_id, radius=radius)
        mi = self.protein_moransI(names_sub, A_sub, protein, use_norm=True, permutations=permutations)

        # 4) Global KDTree for microenvironment
        tree, coords_all, r_global = self.build_kdtree_full(radius=radius)

        # 5) Neighborhood composition of target cluster cells (against all cells)
        comp = self.neighbor_composition_global(names_sub, tree, coords_all, r_global, group_key=group_key)

        # 6) Enrichment (high vs low) from composition
        enrich = self.neighbor_enrichment_from_comp(names_sub, comp, status_col,
                                                    permutations=permutations, random_state=self.random_state)

        # 7) RNA DE (within cluster)
        de = self.de_within_cluster(cluster_id, status_col, layer=de_layer, method="wilcoxon")

        # 8) Predictability of protein-high from microenvironment
        pred = self.microenv_predict_from_comp(names_sub, comp, status_col)

        root = self.adata.uns.setdefault("protein_microenv", {})
        key = f"cluster_{cluster_id}__protein_{protein}"
        root[key] = {
            "status_col": status_col,
            "threshold": float(thr),
            "radius_cluster": float(r_cluster),
            "radius_global": float(r_global),
            "neighbor_enrichment": enrich,
            "moransI": mi,
            "de": de,
            "predict_auc": pred["auc"],
            "predict_coef": pred["coef"]
        }

        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            enrich.to_csv(os.path.join(save_dir, f"neighbor_enrichment_cluster{cluster_id}_{protein}.csv"), index=False)
            de.to_csv(os.path.join(save_dir, f"DE_cluster{cluster_id}_{protein}.csv"), index=False)
            pred["coef"].to_csv(os.path.join(save_dir, f"microenv_predict_coef_cluster{cluster_id}_{protein}.csv"), index=False)

        return root[key]


# ---------------------------------------------------------------------
# (Optional) CLI for quick batch usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Protein microenvironment analysis within an RNA cluster.")
    parser.add_argument("--h5ad", required=True, help="AnnData file with RNA + obs/obsm prepared.")
    parser.add_argument("--cluster", required=True, help="Target RNA cluster id to analyze.")
    parser.add_argument("--protein", required=True, help="Target protein column in obsm[protein/protein_norm].")
    parser.add_argument("--group-key", default="cluster", help="Grouping key for neighborhood composition (e.g., 'cluster' or 'cell_type').")
    parser.add_argument("--radius", type=float, default=None, help="Neighbor radius (μm). Auto if None.")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    analyzer = ProteinMicroEnv(adata=adata)
    res = analyzer.analyze(cluster_id=args.cluster,
                           protein=args.protein,
                           group_key=args.group_key,
                           radius=args.radius,
                           permutations=args.permutations,
                           save_dir=args.save_dir)
    print("[Moran's I]", res["moransI"])
    print("[Neighbor enrichment] head:")
    print(res["neighbor_enrichment"].head(10))
    print("[AUC]", res["predict_auc"])
