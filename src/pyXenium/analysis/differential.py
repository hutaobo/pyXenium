# pyXenium/analysis/differential.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, spearmanr
from statsmodels.stats import multitest

def get_rna_expr_df(adata, layer_key="rna"):
    expr = adata.layers.get(layer_key)
    if expr is None:
        raise KeyError(f"adata.layers does not have {layer_key}")
    try:
        arr = expr.toarray()
    except:
        arr = expr
    return pd.DataFrame(arr, index=adata.obs.index, columns=adata.var.index)

def analyze_one_score(adata, rna_expr, cluster, protein, cluster_key="rna_cluster", score_prefix="score", min_cells=3):
    score_col = f"{score_prefix}:{cluster}:{protein}"
    if score_col not in adata.obs.columns:
        return None

    mask = (adata.obs[cluster_key] == cluster)
    cells = adata.obs.index[mask]
    if len(cells) < min_cells:
        return None

    s = adata.obs.loc[cells, score_col].astype(float)
    median = s.median()
    high = s[s >= median].index
    low  = s[s <  median].index
    if len(high) < min_cells or len(low) < min_cells:
        return None

    # 差异表达
    de_res = []
    for gene in rna_expr.columns:
        gh = rna_expr.loc[high, gene].dropna()
        gl = rna_expr.loc[low, gene].dropna()
        if len(gh) < 3 or len(gl) < 3:
            continue
        t, p = ttest_ind(gh, gl, equal_var=False)
        de_res.append((gene, t, p, gh.mean() - gl.mean()))
    de_df = pd.DataFrame(de_res, columns=["gene","tstat","pval","mean_diff"])
    if not de_df.empty:
        de_df["adj_pval"] = multitest.multipletests(de_df["pval"], method="fdr_bh")[1]

    # 相关性
    corr_res = []
    for gene in rna_expr.columns:
        x = rna_expr.loc[cells, gene].fillna(0).values
        y = s.values
        r, p = spearmanr(x, y)
        corr_res.append((gene, r, p))
    corr_df = pd.DataFrame(corr_res, columns=["gene","spearman_r","pval"])
    if not corr_df.empty:
        corr_df["adj_pval"] = multitest.multipletests(corr_df["pval"], method="fdr_bh")[1]

    return {
        "cluster": cluster,
        "protein": protein,
        "n_cells": len(cells),
        "de": de_df,
        "corr": corr_df
    }

def run_all_clusters_proteins(adata, rna_expr, cluster_label, protein_names, score_prefix="score", min_cells=3):
    results = []
    for cl in adata.obs[cluster_label].unique():
        for p in protein_names:
            rec = analyze_one_score(
                adata, rna_expr, cl, p,
                cluster_key=cluster_label, score_prefix=score_prefix, min_cells=min_cells
            )
            if rec is not None:
                results.append(rec)
    return results

def summarize_results(results):
    # 把结果字典列表拆成两个 DataFrame
    de_list = []
    corr_list = []
    for rec in results:
        c = rec["cluster"]; p = rec["protein"]
        df_de = rec["de"].copy()
        df_de["cluster"] = c; df_de["protein"] = p
        de_list.append(df_de)
        df_corr = rec["corr"].copy()
        df_corr["cluster"] = c; df_corr["protein"] = p
        corr_list.append(df_corr)
    all_de = pd.concat(de_list, ignore_index=True) if de_list else pd.DataFrame()
    all_corr = pd.concat(corr_list, ignore_index=True) if corr_list else pd.DataFrame()
    return all_de, all_corr
