# pyXenium/analysis/tabnet_reports.py

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2

from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


# ------------------------------
# 读取预测（优先从 adata.obs 的写回列）
# ------------------------------
def extract_preds_from_res_or_adata(
    res,
    adata,
    out_key: str = "tabnet_cluster_classifier",
    cluster_key: str = "cluster",
    dropna: bool = True,
    return_mask: bool = False
):
    """
    返回 (y_true, y_pred, proba, classes)；若 return_mask=True，额外返回 mask（有效行）
    优先从 adata.obs 中读取写回列：f"{out_key}:proba:<class>" 与 f"{out_key}:pred"
    若不存在，则回退读取 res.proba_val / res.y_val_true / res.(classes_|label_encoder_classes_)
    """
    y_true_all = adata.obs[cluster_key].astype(str).values

    # 先从 adata.obs 读取
    proba_prefix = f"{out_key}:proba:"
    proba_cols = [c for c in adata.obs.columns if c.startswith(proba_prefix)]
    if len(proba_cols) > 0:
        proba_cols_sorted = sorted(proba_cols, key=lambda c: c.split(":")[-1])
        class_names = [c.split(":")[-1] for c in proba_cols_sorted]
        proba = adata.obs[proba_cols_sorted].to_numpy().astype(float)

        # 预测列
        pred_col = f"{out_key}:pred"
        if pred_col in adata.obs:
            y_pred_all = adata.obs[pred_col].astype(str).values
        else:
            # 暂不计算 pred，等 dropna 后再 argmax
            y_pred_all = None

        # 掩码（去除任何 NaN 的行）
        if dropna:
            mask = ~np.isnan(proba).any(axis=1)
            y_true = y_true_all[mask]
            proba = proba[mask]
            if y_pred_all is not None:
                y_pred = y_pred_all[mask]
            else:
                y_pred = np.array(class_names)[proba.argmax(1)]
        else:
            y_true = y_true_all
            if y_pred_all is None:
                # 有 NaN 时 nanargmax 会报错，因此先构造安全版：
                # 对含 NaN 的行，先用 -inf 替换 NaN，再 argmax
                proba_safe = proba.copy()
                proba_safe[np.isnan(proba_safe)] = -np.inf
                y_pred = np.array(class_names)[np.argmax(proba_safe, axis=1)]
            else:
                y_pred = y_pred_all
            mask = ~np.isnan(proba).any(axis=1)

        if return_mask:
            return y_true, y_pred, proba, class_names, mask
        return y_true, y_pred, proba, class_names

    # 回退到 res
    if res is not None and hasattr(res, "proba_val") and hasattr(res, "y_val_true"):
        proba = np.asarray(res.proba_val, dtype=float)
        y_true = np.asarray(res.y_val_true).astype(str)
        if hasattr(res, "classes_"):
            class_names = list(res.classes_)
        elif hasattr(res, "label_encoder_classes_"):
            class_names = list(res.label_encoder_classes_)
        else:
            class_names = sorted(np.unique(y_true))
        y_pred = np.array(class_names)[proba.argmax(1)]
        mask = np.ones(len(y_true), dtype=bool)
        if return_mask:
            return y_true, y_pred, proba, class_names, mask
        return y_true, y_pred, proba, class_names

    raise ValueError(
        f"找不到预测概率：请确保 {out_key} 已将概率列写回 adata.obs，或 res_cls 包含 proba_val/y_val_true。"
    )


# ------------------------------
# 多分类指标（安全版：自动去 NaN、容忍缺类）
# ------------------------------
def compute_multiclass_metrics_safe(y_true, y_pred, proba, class_names):
    """
    - 自动过滤 proba 含 NaN 的行（以确保 sklearn 指标计算不报错）
    - 若某一类在 y_true 中没有正样本或全为正样本，则对该类的 AUC/AP 返回 NaN
    """
    # 过滤 NaN
    mask = ~np.isnan(proba).any(axis=1)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    proba = np.asarray(proba)[mask]

    labels = list(class_names)

    # 分类报告 & 混淆矩阵
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # per-class AUC/AP（one-vs-rest）
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_true_idx = np.array([label_to_idx[y] for y in y_true])
    auc_list, ap_list = [], []
    for i in range(len(labels)):
        y_bin = (y_true_idx == i).astype(int)
        # 若该类在此集合中无正样本或无负样本，则 AUC/AP 不定义，返回 NaN
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auc_i, ap_i = np.nan, np.nan
        else:
            auc_i = roc_auc_score(y_bin, proba[:, i])
            ap_i = average_precision_score(y_bin, proba[:, i])
        auc_list.append(auc_i)
        ap_list.append(ap_i)
    auc_df = pd.DataFrame({"class": labels, "auc_ovr": auc_list, "ap": ap_list})

    # 多分类 Brier（对每行 one-hot 与 proba 的平方误差）
    y_true_bin = np.eye(len(labels))[y_true_idx]
    brier = float(np.mean(np.sum((proba - y_true_bin) ** 2, axis=1)))

    return report_df, cm, auc_df, brier


# ------------------------------
# 可靠性（校准）Top-class ECE
# ------------------------------
def reliability_topclass(y_true, proba, class_names, n_bins=10):
    mask = ~np.isnan(proba).any(axis=1)
    y_true = np.asarray(y_true)[mask]
    proba = np.asarray(proba)[mask]
    class_names = list(class_names)

    y_pred_idx = proba.argmax(1)
    y_pred = np.array(class_names)[y_pred_idx]
    conf = proba.max(1)
    correct = (y_pred == y_true).astype(int)

    frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=n_bins, strategy="uniform")
    ece = float(np.abs(frac_pos - mean_pred).mean())  # 简易 ECE 近似
    reli_df = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})
    return reli_df, ece


# ------------------------------
# McNemar（配对）整体显著性检验
# ------------------------------
def mcnemar_test(y_true, y_pred_A, y_pred_B):
    correct_A = (y_pred_A == y_true).astype(int)
    correct_B = (y_pred_B == y_true).astype(int)
    b = int(((correct_A == 1) & (correct_B == 0)).sum())  # A对B错
    c = int(((correct_A == 0) & (correct_B == 1)).sum())  # A错B对
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


# ------------------------------
# 生成单模型的完整报告（写回 adata.uns）
# ------------------------------
def generate_tabnet_report(
    adata,
    res=None,
    out_key: str = "tabnet_cluster_classifier",
    cluster_key: str = "cluster",
    write_to_uns: bool = True,
    uns_key: str = "tabnet_reports"
):
    y_true, y_pred, proba, classes = extract_preds_from_res_or_adata(
        res, adata, out_key=out_key, cluster_key=cluster_key, dropna=True
    )

    report_df, cm, auc_df, brier = compute_multiclass_metrics_safe(y_true, y_pred, proba, classes)
    cm_df = pd.DataFrame(cm, index=[f"true:{c}" for c in classes], columns=[f"pred:{c}" for c in classes])
    reli_df, ece = reliability_topclass(y_true, proba, classes, n_bins=10)

    out = {
        "classes": classes,
        "report_df": report_df,
        "confusion_matrix": cm_df,
        "auc_ap_df": auc_df,
        "brier": brier,
        "reliability_df": reli_df,
        "ece": ece,
        "n_eval": int(len(y_true))
    }

    if write_to_uns:
        adata.uns.setdefault(uns_key, {})
        adata.uns[uns_key][out_key] = out
    return out


# ------------------------------
# RNA-only vs RNA+Protein 的对比（含每簇显著性）
# ------------------------------
def compare_modalities_rna_vs_rnap(
    adata,
    out_key_rna: str = "tabnet_cluster_classifier:RNA",
    out_key_rnap: str = "tabnet_cluster_classifier:RNA+Protein",
    cluster_key: str = "cluster",
    write_to_uns: bool = True,
    uns_key: str = "tabnet_reports"
):
    # 分别读取（不 dropna，以便做交集）
    y_true_A, y_pred_A, proba_A, classes_A, maskA = extract_preds_from_res_or_adata(
        None, adata, out_key=out_key_rna, cluster_key=cluster_key, dropna=False, return_mask=True
    )
    y_true_B, y_pred_B, proba_B, classes_B, maskB = extract_preds_from_res_or_adata(
        None, adata, out_key=out_key_rnap, cluster_key=cluster_key, dropna=False, return_mask=True
    )
    assert list(classes_A) == list(classes_B), "两套结果的类别集合不一致"
    classes = classes_A

    # 只保留两者都非 NaN 的样本行
    valid = (~np.isnan(proba_A).any(axis=1)) & (~np.isnan(proba_B).any(axis=1))
    y_true = y_true_A[valid]
    y_pred_A = y_pred_A[valid]
    y_pred_B = y_pred_B[valid]
    proba_A = proba_A[valid]
    proba_B = proba_B[valid]

    # 全局指标
    rep_A, _, auc_A, _ = compute_multiclass_metrics_safe(y_true, y_pred_A, proba_A, classes)
    rep_B, _, auc_B, _ = compute_multiclass_metrics_safe(y_true, y_pred_B, proba_B, classes)
    stat, p = mcnemar_test(y_true, y_pred_A, y_pred_B)

    # 每簇 Δ 指标 + McNemar
    le = LabelEncoder().fit(classes)
    y_int = le.transform(y_true)
    pred_A_int = le.transform(y_pred_A)
    pred_B_int = le.transform(y_pred_B)

    rows = []
    for i, c in enumerate(classes):
        y_bin = (y_int == i).astype(int)
        # per-class AUC/AP
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucA = np.nan; aucB = np.nan; apA = np.nan; apB = np.nan
        else:
            aucA = roc_auc_score(y_bin, proba_A[:, i])
            aucB = roc_auc_score(y_bin, proba_B[:, i])
            apA = average_precision_score(y_bin, proba_A[:, i])
            apB = average_precision_score(y_bin, proba_B[:, i])
        # McNemar（该类为“正确类”的正确/错误比较）
        okA = (pred_A_int == i).astype(int)
        okB = (pred_B_int == i).astype(int)
        b = int(((okA == 1) & (okB == 0)).sum())
        c_ct = int(((okA == 0) & (okB == 1)).sum())
        if b + c_ct == 0:
            chi2_val, p_val = 0.0, 1.0
        else:
            chi2_val = (abs(b - c_ct) - 1) ** 2 / (b + c_ct)
            p_val = 1 - chi2.cdf(chi2_val, df=1)
        rows.append({
            "class": c,
            "ΔAUC": aucB - aucA if (not np.isnan(aucA) and not np.isnan(aucB)) else np.nan,
            "ΔAP": apB - apA if (not np.isnan(apA) and not np.isnan(apB)) else np.nan,
            "McNemar_chi2": chi2_val,
            "McNemar_p": p_val
        })
    delta_df = pd.DataFrame(rows).sort_values("ΔAP", ascending=False)

    out = {
        "classes": classes,
        "global": {
            "RNA_only_macroF1": rep_A.loc["macro avg","f1-score"],
            "RNAp_macroF1": rep_B.loc["macro avg","f1-score"],
            "McNemar_overall_chi2": stat,
            "McNemar_overall_p": p,
            "n_eval": int(len(y_true))
        },
        "per_class": delta_df
    }

    if write_to_uns:
        adata.uns.setdefault(uns_key, {})
        adata.uns[uns_key]["RNA_vs_RNA+Protein"] = out
    return out


# ------------------------------
# 基因集层级的重要度加总 + 置换检验
# ------------------------------
def gene_set_importance_with_permutation_from_impdf(
    imp_df: pd.DataFrame,
    gene_sets: dict,
    n_perm: int = 500,
    random_state: int = 42
):
    """
    imp_df: 包含 ['feature','importance'] 的 DataFrame（建议尽量包含更多特征行数）
    gene_sets: dict, 例如 {"MHC_II": ["HLA-DRA","HLA-DRB1",...], ...}
    返回：每个基因集的 score / 命中数 / 置换 p 值
    """
    rng = np.random.default_rng(random_state)
    imp_df = imp_df.groupby("feature", as_index=False)["importance"].sum()
    imp_df = imp_df.sort_values("importance", ascending=False)
    feat_to_imp = dict(zip(imp_df["feature"], imp_df["importance"]))
    all_feats = imp_df["feature"].tolist()

    rows = []
    for gs_name, genes in gene_sets.items():
        genes_in = [g for g in genes if g in feat_to_imp]
        score = float(sum(feat_to_imp[g] for g in genes_in))
        m = len(genes_in)
        if m == 0:
            rows.append({"set": gs_name, "score": 0.0, "n_hits": 0, "p_perm": 1.0})
            continue
        null_scores = []
        for _ in range(n_perm):
            rnd = rng.choice(all_feats, size=m, replace=False)
            null_scores.append(sum(feat_to_imp.get(x, 0.0) for x in rnd))
        null_scores = np.array(null_scores)
        p = float((null_scores >= score).mean())  # 右尾
        rows.append({"set": gs_name, "score": score, "n_hits": m, "p_perm": p})
    return pd.DataFrame(rows).sort_values(["p_perm", "score"], ascending=[True, False])


# ------------------------------
# 空间一致性（Moran's I + 置换）
# ------------------------------
def morans_I_knn(values, coords, k=8, n_perm=0, rng_seed=42):
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n = len(values)
    nn = NearestNeighbors(n_neighbors=k+1).fit(coords)
    d, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    W = sp.coo_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(n, n)).tocsr()
    W_sum = W.sum()

    z = values - values.mean()
    num = float(z @ (W @ z))
    den = float((z ** 2).sum())
    I = (n / W_sum) * (num / den)

    p = None
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(rng_seed)
        cnt = 0
        for _ in range(n_perm):
            z_perm = rng.permutation(z)
            I_perm = (n / W_sum) * float(z_perm @ (W @ z_perm)) / den
            if I_perm >= I:
                cnt += 1
        p = (cnt + 1) / (n_perm + 1)
    return I, p


def spatial_morans_for_out_key(
    adata,
    out_key: str = "tabnet_cluster_classifier",
    cluster_key: str = "cluster",
    coords_key: str = "spatial",
    k: int = 8,
    n_perm: int = 200
):
    """
    对每个类别的预测概率列计算 Moran's I；自动 dropna。
    返回 DataFrame: class / Moran's I / p_perm
    """
    assert coords_key in adata.obsm, f"缺少空间坐标 adata.obsm['{coords_key}']"
    y_true, y_pred, proba, classes, mask = extract_preds_from_res_or_adata(
        None, adata, out_key=out_key, cluster_key=cluster_key, dropna=False, return_mask=True
    )
    valid = ~np.isnan(proba).any(axis=1)
    proba = proba[valid]
    coords = adata.obsm[coords_key][valid]

    rows = []
    for i, c in enumerate(classes):
        I, p = morans_I_knn(proba[:, i], coords, k=k, n_perm=n_perm)
        rows.append({"class": c, "MoransI": I, "p_perm": p})
    return pd.DataFrame(rows).sort_values("MoransI", ascending=False)


# ------------------------------
# 技术稳健性：分层准确率
# ------------------------------
def stratified_accuracy_by_covariates(
    adata,
    true_key: str = "cluster",
    pred_key: str = "tabnet_cluster_classifier:pred",
    strat_keys=("transcript_counts", "cell_area", "segmentation_method"),
    n_bins: int = 5,
    min_count: int = 50
):
    rows = []
    y_true_all = adata.obs[true_key].astype(str)
    y_pred_all = adata.obs[pred_key].astype(str) if pred_key in adata.obs else None
    if y_pred_all is None:
        raise ValueError(f"找不到预测列 {pred_key}")

    for key in strat_keys:
        s = adata.obs[key]
        if np.issubdtype(s.dtype, np.number):
            q = pd.qcut(s, q=n_bins, duplicates="drop")
            groups = q.astype(str)
        else:
            groups = s.astype(str)

        for g in np.unique(groups):
            idx = (groups == g).values
            if idx.sum() < min_count:
                continue
            acc = (y_true_all.iloc[idx].values == y_pred_all.iloc[idx].values).mean()
            rows.append({"strat_key": key, "bin": str(g), "n": int(idx.sum()), "accuracy": float(acc)})
    return pd.DataFrame(rows).sort_values(["strat_key", "bin"])
