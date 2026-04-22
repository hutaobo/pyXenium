# pyXenium/analysis/scoring.py

import numpy as np
import pandas as pd

def write_model_scores(adata, models, feature_key="X_rna_pca", cluster_key="rna_cluster"):
    """
    对 models 中每个 (cluster, protein) 模型，在对应簇的细胞上预测概率，
    并把结果写入 adata.obs 作为 score:cluster:protein 列。
    """
    # 为避免 obs 碎片化，建议先收集所有新列数据，再一次性 assign
    new_cols = {}
    for cluster, protodict in models.items():
        mask = (adata.obs[cluster_key] == cluster)
        if mask.sum() == 0:
            continue
        X_all = adata.obsm.get(feature_key)
        if X_all is None:
            raise KeyError(f"Feature key {feature_key} not in adata.obsm.")
        X_sub = X_all[mask, :]
        idx = adata.obs.index[mask]

        for protein, res in protodict.items():
            clf = res.model
            scaler = res.scaler
            X_scaled = scaler.transform(X_sub)
            y_prob = clf.predict_proba(X_scaled)[:, 1]
            col_name = f"score:{cluster}:{protein}"
            # 创建一个全体 NaN 列，然后填入子集
            col_ser = pd.Series(np.nan, index=adata.obs.index)
            col_ser.loc[idx] = y_prob
            new_cols[col_name] = col_ser

    # 批量添加到 adata.obs
    for col_name, col_ser in new_cols.items():
        adata.obs[col_name] = col_ser

    return adata
