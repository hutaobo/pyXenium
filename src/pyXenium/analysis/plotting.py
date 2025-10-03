# pyXenium/analysis/plotting.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_auc_heatmap(summary: pd.DataFrame, figsize=(10,8)):
    mat = summary.pivot(index="cluster", columns="protein", values="test_auc")
    mat = mat.apply(pd.to_numeric, errors="coerce")
    g = sns.clustermap(mat, cmap="viridis", linewidths=.3, figsize=figsize)
    g.ax_heatmap.set_xlabel("Protein"); g.ax_heatmap.set_ylabel("Cluster")
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
