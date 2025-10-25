# pyXenium/analysis/protein_microenvironment.py
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, List

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


# -----------------------
# Utilities
# -----------------------

def _get_coords(adata: ad.AnnData,
                prefer_obsm: str = "spatial",
                obs_xy: Tuple[str, str] = ("x_centroid", "y_centroid")) -> np.ndarray:
    """Return Nx2 spatial coordinates (float)."""
    if prefer_obsm in adata.obsm_keys():
        arr = np.asarray(adata.obsm[prefer_obsm], dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"obsm['{prefer_obsm}'] must be of shape (n_cells, 2+).")
        return arr[:, :2]
    xk, yk = obs_xy
    if xk in adata.obs.columns and yk in adata.obs.columns:
        return adata.obs[[xk, yk]].to_numpy(dtype=float)
    raise KeyError(f"Cannot find spatial coords in obsm['{prefer_obsm}'] or obs[{xk},{yk}].")


def _clr(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mat = np.asarray(mat, dtype=float) + eps
    logm = np.log(mat)
    return logm - logm.mean(axis=1, keepdims=True)


def _arcsinh(mat: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    return np.arcsinh(np.asarray(mat, dtype=float) / cofactor)


def _auto_radius(coords: np.ndarray, k: int = 8, factor: float = 1.5) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coords)
    d, _ = nn.kneighbors(coords)
    kth = d[:, k]
    return float(np.median(kth) * factor)


def _build_radius_graph(coords: np.ndarray, radius: float) -> sp.csr_matrix:
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

    perm = np.empty(permutations)
    for b in range(permutations):
        xp = rng.permutation(xc)
        perm[b] = (n / S0) * float(xp.T @ (W @ xp)) / den
    p = (np.sum(np.abs(perm) >= np.abs(I)) + 1) / (permutations + 1)
    return {"I": float(I), "p_value": float(p)}


def _gmm_threshold(x: np.ndarray, random_state: int = 0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values for thresholding.")
    gmm = GaussianMixture(n_components=2, random_state=random_state)
    gmm.fit(x.reshape(-1, 1))
    m1, m2 = np.sort(gmm.means_.ravel())
    return float((m1 + m2) / 2)


# -----------------------
# Core class
# -----------------------

@dataclass
class ProteinMicroEnv:
    """Microenvironment analysis for protein heterogeneity within an RNA cluster."""
    adata: ad.AnnData
    protein_obsm: str = "protein"          # adata.obsm key for protein matrix
    protein_norm_obsm: str = "protein_norm"
    cluster_key: str = "cluster"           # RNA cluster column in adata.obs
    spatial_obsm: str = "spatial"          # prefer this; fallback to obs[x_centroid,y_centroid]
    obs_xy: Tuple[str, str] = ("x_centroid", "y_centroid")
    random_state: int = 0

    # ---------- Normalization ----------
    def normalize_protein(self, method: str = "clr", cofactor: float = 5.0) -> None:
        if self.protein_obsm not in self.adata.obsm_keys():
            raise KeyError(f"Protein matrix not found in obsm['{self.protein_obsm}'].")
        prot = self.adata.obsm[self.protein_obsm]
        if isinstance(prot, pd.DataFrame):
            mat = prot.to_numpy(dtype=float); cols = prot.columns
        else:
            mat = np.asarray(prot, dtype=float); cols = [f"prot_{i}" for i in range(mat.shape[1])]
        if method == "clr":
            arr = _clr(mat)
        elif method == "arcsinh":
            arr = _arcsinh(mat, cofactor=cofactor)
        else:
            raise ValueError("method must be 'clr' or 'arcsinh'.")
        self.adata.obsm[self.protein_norm_obsm] = pd.DataFrame(arr, index=self.adata.obs_names, columns=cols)

    # ---------- Split protein high/low within a cluster ----------
    def split_high_low(self,
                       cluster_id: Union[str, int],
                       protein: str,
                       method: str = "gmm",
                       quantile: float = 0.5) -> Tuple[pd.DataFrame, str, float]:
        if self.cluster_key not in self.adata.obs.columns:
            raise KeyError(f"'{self.cluster_key}' not in adata.obs.")
        if self.protein_norm_obsm not in self.adata.obsm_keys():
            warnings.warn(f"Normalized protein not found in obsm['{self.protein_norm_obsm}']; using raw '{self.protein_obsm}'.")
            key = self.protein_obsm
        else:
            key = self.protein_norm_obsm

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

    # ---------- Build spatial graph ----------
    def build_spatial_graph(self,
                            cluster_id: Union[str, int],
                            radius: Optional[float] = None) -> Tuple[sp.csr_matrix, pd.Index]:
        coords = _get_coords(self.adata, prefer_obsm=self.spatial_obsm, obs_xy=self.obs_xy)
        mask = self.adata.obs[self.cluster_key].astype(str) == str(cluster_id)
        coords_sub = coords[mask]
        names_sub = self.adata.obs_names[mask]
        if coords_sub.shape[0] < 5:
            raise ValueError("Too few cells in the selected cluster.")
        if radius is None:
            radius = _auto_radius(coords_sub, k=8, factor=1.5)
        A = _build_radius_graph(coords_sub, radius=radius)
        # store
        self.adata.uns.setdefault("protein_microenv", {})
        self.adata.uns["protein_microenv"][f"cluster_{cluster_id}__radius"] = float(radius)
        self.adata.uns["protein_microenv"][f"cluster_{cluster_id}__A_shape"] = A.shape
        return A, names_sub

    # ---------- Neighbor composition ----------
    def neighbor_composition(self,
                             names_sub: pd.Index,
                             A_sub: sp.csr_matrix,
                             group_key: str) -> pd.DataFrame:
        obs = self.adata.obs.loc[names_sub, [group_key]].astype(str)
        cats = obs[group_key].astype(str).unique().tolist()
        lookup = {c: i for i, c in enumerate(cats)}
        labels = obs[group_key].map(lookup).to_numpy()
        n = labels.size; k = len(cats)
        ind = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, k), dtype=float)
        counts = A_sub @ ind  # (n x k)
        counts = counts.toarray()
        deg = np.asarray(A_sub.sum(axis=1)).reshape(-1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fracs = np.divide(counts, deg, where=deg > 0)
            fracs[np.isnan(fracs)] = 0.0
        dfc = pd.DataFrame(counts, index=names_sub, columns=[f"nbr_count:{c}" for c in cats])
        dff = pd.DataFrame(fracs, index=names_sub, columns=[f"nbr_frac:{c}" for c in cats])
        return pd.concat([dfc, dff], axis=1)

    # ---------- Permutation enrichment test (high vs low) ----------
    def neighbor_enrichment(self,
                            names_sub: pd.Index,
                            A_sub: sp.csr_matrix,
                            protein_status_col: str,
                            group_key: str,
                            permutations: int = 999,
                            random_state: int = 0) -> pd.DataFrame:
        comp = self.neighbor_composition(names_sub, A_sub, group_key=group_key)
        frac_cols = [c for c in comp.columns if c.startswith("nbr_frac:")]
        y = self.adata.obs.loc[names_sub, protein_status_col].astype(str).to_numpy()
        if not set(np.unique(y)) >= {"protein_high", "protein_low"}:
            raise ValueError(f"Column '{protein_status_col}' must contain 'protein_high' and 'protein_low'.")
        is_high = (y == "protein_high")

        diff_obs = comp.loc[is_high, frac_cols].mean(axis=0) - comp.loc[~is_high, frac_cols].mean(axis=0)

        rng = np.random.default_rng(random_state)
        perm = np.zeros((permutations, len(frac_cols)))
        for b in range(permutations):
            yp = rng.permutation(is_high)
            perm[b, :] = comp.loc[yp, frac_cols].mean(axis=0).to_numpy() - comp.loc[~yp, frac_cols].mean(axis=0).to_numpy()
        mu = perm.mean(axis=0); sd = perm.std(axis=0, ddof=1) + 1e-9
        z = (diff_obs.to_numpy() - mu) / sd
        p = np.mean(np.abs(perm - mu) >= np.abs(diff_obs.to_numpy() - mu), axis=0)
        _, q, _, _ = multipletests(p, method="fdr_bh")

        out = pd.DataFrame({
            "neighbor_type": [c.replace("nbr_frac:", "") for c in frac_cols],
            "delta_frac_high_minus_low": diff_obs.values,
            "z_score": z,
            "p_value": p,
            "q_value": q
        }).sort_values("q_value")
        return out

    # ---------- Moran's I for protein spatial autocorrelation ----------
    def protein_moransI(self,
                        names_sub: pd.Index,
                        A_sub: sp.csr_matrix,
                        protein: str,
                        use_norm: bool = True,
                        permutations: int = 999) -> Dict[str, float]:
        key = self.protein_norm_obsm if (use_norm and self.protein_norm_obsm in self.adata.obsm_keys()) else self.protein_obsm
        dfp = self.adata.obsm[key]
        if protein not in dfp.columns:
            raise KeyError(f"Protein '{protein}' not found in obsm['{key}'].")
        vals = dfp.loc[names_sub, protein].to_numpy()
        return _morans_I(vals, A_sub, permutations=permutations, random_state=self.random_state)

    # ---------- Differential expression within cluster (RNA) ----------
    def de_within_cluster(self,
                          cluster_id: Union[str, int],
                          protein_status_col: str,
                          layer: str = "rna",
                          method: str = "wilcoxon") -> pd.DataFrame:
        mask = self.adata.obs[self.cluster_key].astype(str) == str(cluster_id)
        ad_sub = self.adata[mask].copy()
        if layer in ad_sub.layers:
            ad_sub.X = ad_sub.layers[layer]
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

    # ---------- Predict protein-high from neighborhood (microenvironment signal) ----------
    def microenv_predict(self,
                         names_sub: pd.Index,
                         A_sub: sp.csr_matrix,
                         protein_status_col: str,
                         group_key: str,
                         C: float = 1.0,
                         max_iter: int = 1000) -> Dict[str, Union[float, pd.DataFrame]]:
        comp = self.neighbor_composition(names_sub, A_sub, group_key=group_key)
        frac_cols = [c for c in comp.columns if c.startswith("nbr_frac:")]
        X = comp[frac_cols].to_numpy()
        y = (self.adata.obs.loc[names_sub, protein_status_col].astype(str).to_numpy() == "protein_high").astype(int)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xz = scaler.fit_transform(X)
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=self.random_state)
        clf.fit(Xz, y)
        auc = roc_auc_score(y, clf.predict_proba(Xz)[:, 1])
        coef = pd.DataFrame({"feature": frac_cols, "coef": clf.coef_.ravel()}).sort_values("coef", ascending=False)
        return {"auc": float(auc), "coef": coef}

    # ---------- Plot ----------
    def plot_spatial(self,
                     color: Union[str, np.ndarray],
                     title: str = "",
                     s: float = 2.0,
                     alpha: float = 0.9,
                     cmap: str = "viridis") -> None:
        coords = _get_coords(self.adata, self.spatial_obsm, self.obs_xy)
        if isinstance(color, str):
            if color in self.adata.obs.columns:
                c = self.adata.obs[color].values
            elif self.protein_norm_obsm in self.adata.obsm_keys() and color in self.adata.obsm[self.protein_norm_obsm].columns:
                c = self.adata.obsm[self.protein_norm_obsm][color].values
            elif color in self.adata.obsm.get(self.protein_obsm, pd.DataFrame()).columns:
                c = self.adata.obsm[self.protein_obsm][color].values
            else:
                raise KeyError(f"color '{color}' not found in obs or protein obsm.")
        else:
            c = np.asarray(color)
        plt.figure(figsize=(6, 6))
        sca = plt.scatter(coords[:, 0], coords[:, 1], c=c, s=s, alpha=alpha, cmap=cmap)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(sca, shrink=0.8)
        plt.title(title); plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout(); plt.show()

    # ---------- One-stop pipeline ----------
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
        """Run the full microenvironment pipeline and store results in adata.uns['protein_microenv'][..]."""
        # 1) protein normalize if needed
        if self.protein_norm_obsm not in self.adata.obsm_keys():
            self.normalize_protein(method="clr")

        # 2) split high/low
        split_df, status_col, thr = self.split_high_low(cluster_id, protein, method=status_method, quantile=status_quantile)

        # 3) spatial graph (within cluster)
        A_sub, names_sub = self.build_spatial_graph(cluster_id, radius=radius)

        # 4) neighbor enrichment (permutation)
        enrich = self.neighbor_enrichment(names_sub, A_sub, status_col, group_key, permutations=permutations, random_state=self.random_state)

        # 5) Moran's I
        mi = self.protein_moransI(names_sub, A_sub, protein, use_norm=True, permutations=permutations)

        # 6) DE (RNA)
        de = self.de_within_cluster(cluster_id, status_col, layer=de_layer, method="wilcoxon")

        # 7) Microenvironment predicting protein-high
        pred = self.microenv_predict(names_sub, A_sub, status_col, group_key)

        # 8) store to uns
        root = self.adata.uns.setdefault("protein_microenv", {})
        key = f"cluster_{cluster_id}__protein_{protein}"
        root[key] = {
            "status_col": status_col,
            "threshold": float(thr),
            "radius": float(self.adata.uns['protein_microenv'].get(f"cluster_{cluster_id}__radius")),
            "neighbor_enrichment": enrich,
            "moransI": mi,
            "de": de,
            "predict_auc": pred["auc"],
            "predict_coef": pred["coef"]
        }

        # 9) save if requested
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            enrich.to_csv(os.path.join(save_dir, f"neighbor_enrichment_cluster{cluster_id}_{protein}.csv"), index=False)
            de.to_csv(os.path.join(save_dir, f"DE_cluster{cluster_id}_{protein}.csv"), index=False)
            pred["coef"].to_csv(os.path.join(save_dir, f"microenv_predict_coef_cluster{cluster_id}_{protein}.csv"), index=False)

        return root[key]
