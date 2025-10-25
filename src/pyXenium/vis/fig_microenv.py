# -*- coding: utf-8 -*-
"""
pyXenium/vis/fig_microenv.py

Build a publication-ready multi-panel figure from ProteinMicroEnv analysis:
A: spatial categorical (protein status), B: spatial numeric (protein level),
C: neighbor enrichment bars, D: microenvironment predictability (coef + AUC),
E: RNA DE volcano (within cluster), F: protein distribution (hist/KDE).
"""

from __future__ import annotations
import os
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scanpy as sc
from anndata import AnnData

# ---------------------- Global style (journal-friendly) ----------------------

def set_paper_rc(font_family: str = "Arial",
                 base_size: float = 8.0,
                 line_width: float = 0.8) -> None:
    """A minimal, journal-friendly rcParams setup."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family],
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "axes.linewidth": line_width,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,    # TrueType (editable in AI)
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })

# ---------------------- Helpers ----------------------

def _get_coords(adata: AnnData,
                prefer_obsm: str = "spatial",
                obs_xy: Tuple[str, str] = ("x_centroid", "y_centroid")) -> np.ndarray:
    if prefer_obsm in adata.obsm.keys():
        arr = np.asarray(adata.obsm[prefer_obsm])
        return arr[:, :2]
    return adata.obs.loc[:, [obs_xy[0], obs_xy[1]]].to_numpy()

def _draw_scale_bar(ax, coords: np.ndarray, length_um: float = 100.0, pad_ratio: float = 0.04) -> None:
    """Add a simple horizontal scale bar (assumes coords in μm)."""
    xmin, ymin = coords[:,0].min(), coords[:,1].min()
    xmax, ymax = coords[:,0].max(), coords[:,1].max()
    L = length_um
    pad = pad_ratio * (xmax - xmin)
    x0 = xmin + pad
    y0 = ymin + pad
    ax.plot([x0, x0 + L], [y0, y0], lw=1.2, color="black")
    ax.text(x0 + L/2, y0 + 0.8*pad, f"{int(L)} μm", ha="center", va="bottom")

def _rasterized_scatter(ax, x, y, c, title: str = "", rasterized: bool = True,
                        vmin=None, vmax=None, cmap="viridis", s=1.0, alpha=0.9):
    sca = ax.scatter(x, y, c=c, s=s, alpha=alpha, cmap=cmap,
                     rasterized=rasterized, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box"); ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    return sca, cb

def _categorical_scatter(ax, coords: np.ndarray, labels: pd.Series,
                         title: str = "", s=1.0, alpha=0.9):
    """Plot categorical labels (incl. NaN->grey)."""
    cat = labels.astype("category")
    codes = cat.cat.codes.to_numpy()  # NaN -> -1
    mask = codes != -1
    # Discrete colormap for categories
    base = plt.get_cmap("tab20", max(len(cat.cat.categories), 1))
    if mask.any():
        sca = ax.scatter(coords[mask,0], coords[mask,1], c=codes[mask],
                         s=s, alpha=alpha, cmap=base, rasterized=True)
        cb = plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
        cb.set_ticks(np.arange(len(cat.cat.categories)))
        cb.set_ticklabels(list(cat.cat.categories))
    if (~mask).any():
        ax.scatter(coords[~mask,0], coords[~mask,1], c="lightgrey", s=s, alpha=alpha, rasterized=True)
    ax.set_aspect("equal", adjustable="box"); ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

def _barh_with_ci(ax, df: pd.DataFrame, value_col: str, label_col: str,
                  top_k: int = 10, title: str = "", invert: bool = True):
    df2 = df.sort_values(value_col, ascending=False).head(top_k)
    ax.barh(df2[label_col], df2[value_col])
    if invert: ax.invert_yaxis()
    ax.set_title(title)

def _volcano(ax, de: pd.DataFrame, title: str = "", max_points: int = 20000):
    """Generic volcano; expects columns: 'logfoldchanges', 'pvals_adj', 'group'."""
    df = de.copy()
    # pick one direction (protein_high vs rest)
    if "group" in df.columns:
        g = sorted(df["group"].unique())
        # prefer the group named 'protein_high'
        grp = "protein_high" if "protein_high" in g else g[0]
        df = df[df["group"] == grp].copy()
    df["neglog10q"] = -np.log10(np.clip(df["pvals_adj"].astype(float), 1e-300, 1.0))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["logfoldchanges", "neglog10q"])
    # downsample for plotting speed
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    ax.scatter(df["logfoldchanges"], df["neglog10q"], s=6, alpha=0.7, rasterized=True)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(q)")
    ax.set_title(title)

def _hist(ax, x: np.ndarray, bins: int = 40, title: str = ""):
    ax.hist(x, bins=bins)
    ax.set_title(title)

# ---------------------- Main figure builder ----------------------

def build_microenv_figure(adata: AnnData,
                          res: Dict,
                          cluster_id: str,
                          protein: str,
                          group_key: str = "cluster",
                          spatial_obsm: str = "spatial",
                          obs_xy: Tuple[str,str] = ("x_centroid","y_centroid"),
                          outdir: str = "./figures",
                          basename: Optional[str] = None,
                          figsize_inches: Tuple[float,float] = (7.0, 5.0),
                          scatter_s: float = 0.5,
                          scale_bar_um: Optional[float] = 100.0) -> str:
    """
    Assemble a 2x3 multi-panel board: A-F. Returns the saved base path (without extension).
    """
    os.makedirs(outdir, exist_ok=True)
    base = basename or f"Fig_microenv_cluster{cluster_id}_{protein}"

    # Panels need:
    # - coords
    coords = _get_coords(adata, spatial_obsm, obs_xy)
    # - status column, protein numeric, enrichment table, coef table, DE table, MI
    status_col = res["status_col"]
    enrich = res["neighbor_enrichment"]
    coef   = res["predict_coef"]
    de     = res["de"]
    mi     = res["moransI"]
    auc    = res["predict_auc"]

    # Prepare values for panel B
    prot_key = "protein_norm" if "protein_norm" in adata.obsm_keys() else "protein"
    prot_vals = adata.obsm[prot_key][protein].to_numpy()

    # Prepare mask for target cluster (避免绘全图过慢可选子采样)
    mask = adata.obs["cluster"].astype(str) == str(cluster_id)
    coords_c = coords[mask]
    status_c = adata.obs.loc[mask, status_col]

    # rc
    set_paper_rc()

    # Figure & GridSpec
    fig = plt.figure(figsize=figsize_inches, constrained_layout=True)
    gs  = GridSpec(2, 3, figure=fig)

    # ------- A: spatial categorical (status in cluster) -------
    axA = fig.add_subplot(gs[0,0])
    _categorical_scatter(axA, coords_c, status_c, title=f"A  {protein} high/low (cluster {cluster_id})", s=scatter_s)
    if scale_bar_um is not None:
        _draw_scale_bar(axA, coords_c, length_um=scale_bar_um)

    # ------- B: spatial numeric (protein level, cluster only) -------
    axB = fig.add_subplot(gs[0,1])
    prot_c = prot_vals[mask]
    vmin, vmax = np.nanpercentile(prot_c, [2, 98])
    sca, _ = _rasterized_scatter(axB, coords_c[:,0], coords_c[:,1], prot_c,
                                 title=f"B  {protein} level", vmin=vmin, vmax=vmax, s=scatter_s)
    if scale_bar_um is not None:
        _draw_scale_bar(axB, coords_c, length_um=scale_bar_um)

    # ------- C: neighbor enrichment (bars) -------
    axC = fig.add_subplot(gs[0,2])
    if isinstance(enrich, pd.DataFrame) and not enrich.empty:
        # 仅显示显著或top10
        dfC = enrich.copy()
        dfC["label"] = dfC["neighbor_type"].astype(str)
        _barh_with_ci(axC, dfC, value_col="delta_frac_high_minus_low", label_col="label",
                      top_k=10, title="C  Neighbor enrichment (Δfrac High-Low)")
        axC.set_xlabel("Δ fraction")
    else:
        axC.text(0.5, 0.5, "No enrichment", ha="center", va="center")
        axC.axis("off")

    # ------- D: microenvironment predictability (coef + AUC) -------
    axD = fig.add_subplot(gs[1,0])
    if isinstance(coef, pd.DataFrame) and not coef.empty:
        dfD = coef.copy()
        dfD["label"] = dfD["feature"].str.replace("nbr_frac:", "", regex=False)
        dfD = dfD.sort_values("coef", ascending=True).tail(12)
        axD.barh(dfD["label"], dfD["coef"])
        axD.set_title("D  Microenvironment coefficients")
        axD.set_xlabel("logistic coef")
        # annotate AUC
        axD.text(0.98, 0.05, f"AUC={auc:.3f}" if np.isfinite(auc) else "AUC=N/A",
                 ha="right", va="bottom", transform=axD.transAxes)
    else:
        axD.text(0.5, 0.5, "No model", ha="center", va="center")
        axD.axis("off")

    # ------- E: volcano (DE within cluster) -------
    axE = fig.add_subplot(gs[1,1])
    if isinstance(de, pd.DataFrame) and not de.empty:
        _volcano(axE, de, title="E  RNA DE: protein-high vs low")
    else:
        axE.text(0.5, 0.5, "No DE", ha="center", va="center")
        axE.axis("off")

    # ------- F: protein distribution (hist) -------
    axF = fig.add_subplot(gs[1,2])
    axF.hist([prot_c[status_c == "protein_low"], prot_c[status_c == "protein_high"]],
             bins=40, label=["low", "high"], alpha=0.7)
    axF.set_title(f"F  {protein} distribution")
    axF.set_xlabel(f"{protein} (normalized)"); axF.set_ylabel("cells")
    axF.legend(frameon=False)

    # Suptitle with Moran's I
    if isinstance(mi, dict) and "I" in mi:
        fig.suptitle(f"Protein microenvironment (cluster {cluster_id}, {protein})  |  Moran's I={mi['I']:.3f}, p={mi['p_value']:.2g}",
                     y=1.02, fontsize=8)

    # save
    outbase = os.path.join(outdir, base)
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".png", bbox_inches="tight", dpi=600)
    # 可选 svg
    fig.savefig(outbase + ".svg", bbox_inches="tight")
    plt.close(fig)
    return outbase
