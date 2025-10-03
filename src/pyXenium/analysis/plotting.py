# pyXenium/analysis/plotting.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

def plot_auc_heatmap(summary: pd.DataFrame,
                     figsize=(10, 8),
                     metric="euclidean",
                     method="average",
                     min_shared=2):
    """
    对 summary pivot 出的 AUC 矩阵做热图 + 聚类（clustermap），
    在距离计算阶段忽略 NaN（即只用两个行/列共有的那部分非 NaN 值计算距离）；
    在热图阶段则对 NaN 用 mask 使其不可着色，但保留行/列显示顺序。

    参数
    ----
    summary : pd.DataFrame
        包含至少三列 “cluster”, “protein”, “test_auc”，用于 pivot。
    figsize : tuple (宽, 高)
        图像尺寸传给 seaborn.clustermap。
    metric : str
        用于 pdist 的距离度量（默认 “euclidean”）。
    method : str
        用于 sch.linkage 的聚类方法（默认 “average”）。
    min_shared : int
        当两行／列用来计算距离时，要求它们“共同非 NaN 的特征数量” ≥ min_shared；
        若共有特征太少，则认为距离为缺失（会在后续被替换为一个较大距离）。

    返回
    ----
    sns.ClusterGrid 对象（clustermap 画出的结果）。
    """
    # 构造矩阵
    mat = summary.pivot(index="cluster", columns="protein", values="test_auc")
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat = mat.replace([np.inf, -np.inf], np.nan)

    # 内部辅助函数：用来给 pdist 提供“忽略 NaN 的距离函数”
    def _pairwise_dist(u, v):
        # u, v 是一维 numpy 数组
        mask = (~np.isnan(u)) & (~np.isnan(v))
        if mask.sum() < min_shared:
            return np.nan
        uu = u[mask]
        vv = v[mask]
        # 注意：这里用了 pdist 但其实只计算两个向量之间距离
        # 用 numpy 或其它实现也可以
        return pdist(np.vstack([uu, vv]), metric=metric)[0]

    # 计算行 linkage
    # pdist 会把上三角所有行对的距离打包成“压缩距离向量”
    row_dist = pdist(mat.values, metric=_pairwise_dist)
    # 对 row_dist 中的 NaN 距离赋一个较大数（最大有限值 * 1.1）
    finite = row_dist[np.isfinite(row_dist)]
    if finite.size > 0:
        maxd = np.nanmax(finite)
        row_dist = np.where(np.isfinite(row_dist), row_dist, maxd * 1.1)
        row_linkage = sch.linkage(row_dist, method=method)
    else:
        row_linkage = None

    # 计算列 linkage（对转置矩阵做同样操作）
    col_dist = pdist(mat.values.T, metric=_pairwise_dist)
    finite2 = col_dist[np.isfinite(col_dist)]
    if finite2.size > 0:
        maxd2 = np.nanmax(finite2)
        col_dist = np.where(np.isfinite(col_dist), col_dist, maxd2 * 1.1)
        col_linkage = sch.linkage(col_dist, method=method)
    else:
        col_linkage = None

    # mask：在热图可视化阶段遮蔽 NaN
    mask = mat.isna()

    # 调用 clustermap，传入计算好的 linkage，保留行 / 列显示顺序
    g = sns.clustermap(mat,
                       row_linkage=row_linkage,
                       col_linkage=col_linkage,
                       mask=mask,
                       figsize=figsize,
                       cmap="viridis",
                       linewidths=.3)

    g.ax_heatmap.set_xlabel("Protein")
    g.ax_heatmap.set_ylabel("Cluster")
    return g

def plot_topk_per_cluster(summary: pd.DataFrame, k=5, metric="test_auc"):
    topk = (summary.sort_values(["cluster", metric], ascending=[True, False])
                   .groupby("cluster").head(k))
    fig, ax = plt.subplots(figsize=(max(10, k * 1.2), 6))
    labels = []
    vals = []
    for cl, sub in topk.groupby("cluster"):
        for _, r in sub.iterrows():
            labels.append(f"{cl}:{r['protein']}")
            vals.append(r[metric])
    ax.bar(labels, vals)
    ax.set_ylabel(metric)
    ax.set_xticklabels(labels, rotation=90)
    plt.tight_layout()
    return fig

def plot_DE_volcano(de_df: pd.DataFrame, title="DE Volcano",
                    logfc_col="mean_diff", pval_col="pval", adj_col="adj_pval",
                    fdr_thresh=0.05):
    df = de_df.copy()
    df["-log10p"] = -np.log10(df[pval_col])
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x=logfc_col, y="-log10p",
                    hue=df[adj_col] < fdr_thresh,
                    palette={True: "red", False: "gray"}, legend=False)
    plt.axhline(-np.log10(0.05), ls="--", color="black")
    plt.title(title)
    plt.xlabel("Mean difference (High vs Low)")
    plt.ylabel("-log10(p)")
    plt.tight_layout()
    plt.show()

def plot_model_diagnostics(adata, models, cluster, protein, feature_key="X_rna_pca"):
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import calibration_curve

    res = models[cluster][protein]
    clf, scaler = res.model, res.scaler
    thr = getattr(res, "threshold", None)

    mask = (adata.obs["rna_cluster"] == cluster)
    X = scaler.transform(adata.obsm[feature_key][mask, :])
    # y 真值需要你自己定义：可能 adata.obs[f"protein:{protein}"] ≥ thr
    y = (adata.obs.loc[mask, f"protein:{protein}"] >= thr).astype(int).to_numpy()
    y_prob = clf.predict_proba(X)[:, 1]

    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title(f"ROC — {cluster}:{protein}")
    PrecisionRecallDisplay.from_predictions(y, y_prob)
    plt.title(f"PR — {cluster}:{protein}")
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("Predicted prob"); plt.ylabel("Empirical freq")
    plt.title(f"Calibration — {cluster}:{protein}")
    plt.tight_layout()
    plt.show()
