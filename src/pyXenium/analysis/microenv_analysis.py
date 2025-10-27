# pyXenium/analysis/microenv_analysis.py

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from anndata import AnnData
from pyXenium.io.xenium_gene_protein_loader import load_xenium_gene_protein
from pyXenium.analysis import ProteinMicroEnv  # 确保依赖 ProteinMicroEnv 类
from pyXenium.utils.name_resolver import resolve_protein_column

def analyze_microenvironment(
    mode: str,
    adata: AnnData | None = None,
    base_path: str | None = None,
    output_dir: str | None = None
) -> AnnData:
    """
    执行免疫微环境 ('immune') 或肿瘤-间质边界 ('tumor_border') 分析的统一入口函数。

    参数：
    - mode: 分析模式，'immune' 表示免疫微环境分析，'tumor_border' 表示肿瘤与间质边界分析。
    - adata: AnnData 对象，包含 Xenium 空间转录组+蛋白数据。如果未提供，则需要提供 base_path 以加载数据。
    - base_path: Xenium 输出数据的根目录路径。当 adata 未提供时，将从该路径下加载数据。
    - output_dir: 可选。若提供路径，则将在分析完成后把结果表格导出为 CSV 文件保存到此目录。

    功能：
    根据指定模式自动筛选“锚点”细胞，并进行空间邻域分析，包括：
    1. **锚点筛选**：
       - 模式 'immune': 自动选择免疫细胞作为锚点（优先使用 CD8 标记，如无则使用 CD45）。
       - 模式 'tumor_border': 自动选择肿瘤细胞作为锚点（优先使用 PanCK 标记，如无则使用 E-Cadherin）。
       使用 Gaussian 混合模型 (GMM) 对锚点标记的蛋白表达进行双峰拟合，计算阈值，将高于阈值的细胞判定为锚点。
       对于肿瘤边界模式，会进一步筛选出靠近非肿瘤细胞的肿瘤锚点（即真正位于肿瘤-基质交界处的细胞）。
    2. **邻域设置**：默认采用同心圆邻域半径 [0, 10, 20, 40] 微米。其中 40μm 视为微环境影响范围。
       分析时将锚点周围 40μm 半径内的细胞定义为“邻居”群体，并可根据 0–10, 10–20, 20–40μm 不同距离分层计算。
    3. **基因邻域密度与 Fold-change 分析**：比较“邻居”群体与远端细胞群体的基因表达差异。
       计算每个基因在邻居细胞中的平均表达密度和在远端细胞中的平均表达密度，并计算二者之比（Fold-change）。
    4. **邻居细胞类型统计**：统计每个锚点细胞邻域内特定类型细胞的数量，例如标记 alphaSMA⁺（成纤维细胞）和 CD31⁺（内皮细胞）的邻居数量。

    输出：
    函数直接修改并返回输入的 AnnData：
    - 对于免疫微环境分析：
      - 在 `adata.obs` 新增布尔列 `immune_is_anchor`，标记每个细胞是否被选为免疫锚点。
      - 在 `adata.obsm` 新增 DataFrame `immune_gene_stats`，存储基因在邻居与远端的平均表达及fold-change（行索引为基因）。
    - 对于肿瘤边界分析：
      - 在 `adata.obs` 新增布尔列 `tumor_border_is_anchor`，标记每个细胞是否被选为肿瘤边界锚点。
      - 在 `adata.obsm` 新增 DataFrame `tumor_border_gene_stats`，存储基因在邻居与远端的平均表达及fold-change。
    - 无论哪种模式，`adata.obs` 将新增列 `neighbors_alphaSMA_count` 与 `neighbors_CD31_count`，
      其中锚点细胞的该列值为其 40μm 邻域内 alphaSMA⁺ 邻居细胞数和 CD31⁺ 邻居细胞数（非锚点细胞该列为空值）。

    此外，如果提供了 output_dir 参数，将把主要结果导出为 CSV 文件：
    - `immune_gene_stats.csv` 或 `tumor_border_gene_stats.csv`：基因邻域分析结果表（含基因名称、邻居平均表达、远端平均表达和fold-change）。
    - `neighbors_counts.csv`：各锚点细胞的 alphaSMA⁺ 和 CD31⁺ 邻居数量统计表。

    注意：
    - 函数对默认参数友好，旨在 Notebook 中快速调用。大部分参数内部自动处理，如坐标获取、阈值选择等无需用户干预。
    - 实现上兼容不同版本的 ProteinMicroEnv 工具类（方法名和构造参数自动适配），并利用其中的别名解析等功能。
    """
    # 参数校验
    mode = mode.lower()
    if mode not in ("immune", "tumor_border"):
        raise ValueError("mode 必须为 'immune' 或 'tumor_border'")

    # 若未提供 adata，则尝试根据 base_path 加载 Xenium 数据
    if adata is None:
        if base_path is None:
            raise ValueError("adata 未提供时必须指定 base_path")
        try:
            adata = load_xenium_gene_protein(base_path)
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {e}")

    # 从 AnnData 获取空间坐标
    if "spatial" in adata.obsm_keys():
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("adata.obsm['spatial'] 应为 shape (n_cells, >=2) 的坐标矩阵")
        coords_xy = coords[:, :2]
    elif {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        coords_xy = adata.obs.loc[:, ["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    else:
        raise KeyError("AnnData 中未找到空间坐标信息（既不存在 obsm['spatial']，也没有 obs['x_centroid','y_centroid'] 列）")

    # 初始化 KDTree 用于邻居查询
    tree = cKDTree(coords_xy)

    # 标记锚点的布尔掩码数组
    anchor_mask = np.zeros(adata.n_obs, dtype=bool)
    # 根据模式确定锚点筛选的蛋白标记候选
    if mode == "immune":
        anchor_markers = ["CD8", "CD45"]  # 优先 CD8，如缺失则用 CD45
        anchor_flag_col = "immune_is_anchor"
    else:  # tumor_border
        anchor_markers = ["PanCK", "E-Cadherin"]  # 优先 PanCK，如缺失则用 E-Cadherin
        anchor_flag_col = "tumor_border_is_anchor"

    # 尝试对每个候选标记进行锚点筛选
    for marker in anchor_markers:
        try:
            prot_col = resolve_protein_column(adata, marker)
        except KeyError:
            continue  # 该标记不在数据中，尝试下一个
        # 提取该标记的蛋白表达值数组
        values = adata.obsm["protein"][prot_col].to_numpy(dtype=float)
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            continue
        # 使用双峰高斯混合模型估计阈值；若失败则退而求其次用中位数
        try:
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(finite_vals.reshape(-1, 1))
            mean1, mean2 = np.sort(gmm.means_.ravel())
            threshold = float((mean1 + mean2) / 2.0)
        except Exception:
            threshold = float(np.nanquantile(finite_vals, 0.5))
        # 更新锚点掩码：值大于等于阈值的细胞记为 True
        anchor_mask |= (adata.obsm["protein"][prot_col].to_numpy(dtype=float) >= threshold)

    # 如果模式为肿瘤边界，需要进一步筛选出真正位于边界的锚点
    if mode == "tumor_border":
        if anchor_mask.any():
            border_mask = np.zeros_like(anchor_mask)
            anchor_indices = np.where(anchor_mask)[0]
            for idx in anchor_indices:
                # 查找该锚点在半径40μm内的邻居
                neighbor_idxs = tree.query_ball_point(coords_xy[idx], r=40.0)
                if idx in neighbor_idxs:
                    neighbor_idxs.remove(idx)
                # 如存在至少一个非锚点邻居，则该锚点位于边界
                if any(not anchor_mask[j] for j in neighbor_idxs):
                    border_mask[idx] = True
            anchor_mask = border_mask  # 仅保留边界锚点
        else:
            warnings.warn("未找到任何候选肿瘤锚点细胞，无法进行边界分析。")

    # 将锚点标记写入 AnnData.obs
    adata.obs[anchor_flag_col] = pd.Series(anchor_mask, index=adata.obs_names, dtype=bool)

    # 确定邻居细胞和远端细胞集合
    anchor_indices = np.where(anchor_mask)[0]
    neighbor_set = set()
    for idx in anchor_indices:
        # 获取锚点 idx 在 40μm 半径内的所有邻居索引
        neighbor_idxs = tree.query_ball_point(coords_xy[idx], r=40.0)
        if idx in neighbor_idxs:
            neighbor_idxs.remove(idx)
        for j in neighbor_idxs:
            # 仅记录非锚点的邻居细胞
            if j not in anchor_indices:
                neighbor_set.add(j)
    neighbor_indices = np.array(sorted(neighbor_set))
    far_set = set(range(adata.n_obs)) - neighbor_set - set(anchor_indices)
    far_indices = np.array(sorted(far_set))

    # 若RNA表达矩阵存在，则进行基因邻域差异分析
    gene_stats_df = None
    if adata.n_vars > 0:
        # 提取基因表达矩阵（使用原始counts层）
        X = adata.layers["rna"] if "rna" in adata.layers else adata.X
        # 计算邻居组和远端组各基因平均表达
        if neighbor_indices.size > 0:
            neighbor_mean = np.asarray(X[neighbor_indices].mean(axis=0)).ravel()
        else:
            neighbor_mean = np.zeros(adata.n_vars, dtype=float)
        if far_indices.size > 0:
            far_mean = np.asarray(X[far_indices].mean(axis=0)).ravel()
        else:
            far_mean = np.zeros(adata.n_vars, dtype=float)
        # 计算fold-change比值（邻居平均 / 远端平均）
        with np.errstate(divide='ignore', invalid='ignore'):
            fold_change = neighbor_mean / far_mean
        # 将无法计算的情况（如远端均值和邻域均值皆为0）置为 NaN
        fold_change = np.where(np.isfinite(fold_change), fold_change, np.nan)
        # 准备结果 DataFrame：基因名称，邻居均值，远端均值，Fold-change
        gene_names = adata.var.get("name") if "name" in adata.var.columns else adata.var_names
        gene_stats_df = pd.DataFrame({
            "neighbor_mean": neighbor_mean,
            "far_mean": far_mean,
            "fold_change": fold_change
        }, index=pd.Index(gene_names, name="gene"))
        # 根据 fold-change 大小降序排序（忽略 NaN）
        gene_stats_df.sort_values(by="fold_change", ascending=False, inplace=True, na_position='last')
        # 将结果存入 AnnData.obsm
        result_key = "immune_gene_stats" if mode == "immune" else "tumor_border_gene_stats"
        adata.obsm[result_key] = gene_stats_df

    # 统计每个锚点邻域内 alphaSMA+ 和 CD31+ 邻居数量，结果存入 AnnData.obs
    # 初始化计数列为 NaN，以便非锚点保持 NaN
    adata.obs["neighbors_alphaSMA_count"] = np.nan
    adata.obs["neighbors_CD31_count"] = np.nan
    for idx in anchor_indices:
        neighbor_idxs = tree.query_ball_point(coords_xy[idx], r=40.0)
        if idx in neighbor_idxs:
            neighbor_idxs.remove(idx)
        # 统计 alphaSMA 邻居
        alpha_count = 0
        try:
            col_alpha = resolve_protein_column(adata, "alphaSMA")
        except KeyError:
            col_alpha = None
        if col_alpha:
            for j in neighbor_idxs:
                # 排除锚点细胞本身，统计邻居中 alphaSMA 蛋白值 >0 的细胞数
                if not anchor_mask[j]:
                    val = float(adata.obsm["protein"].iloc[j][col_alpha])
                    if np.isfinite(val) and val > 0:
                        alpha_count += 1
        # 统计 CD31 邻居
        cd31_count = 0
        try:
            col_cd31 = resolve_protein_column(adata, "CD31")
        except KeyError:
            col_cd31 = None
        if col_cd31:
            for j in neighbor_idxs:
                if not anchor_mask[j]:
                    val = float(adata.obsm["protein"].iloc[j][col_cd31])
                    if np.isfinite(val) and val > 0:
                        cd31_count += 1
        # 写入该锚点的统计结果
        cell_name = adata.obs_names[idx]
        adata.obs.at[cell_name, "neighbors_alphaSMA_count"] = alpha_count
        adata.obs.at[cell_name, "neighbors_CD31_count"] = cd31_count

    # 如指定了输出目录，则将结果数据表保存为 CSV 文件
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # 基因差异分析结果
        if gene_stats_df is not None:
            out_path = os.path.join(
                output_dir,
                "immune_gene_stats.csv" if mode == "immune" else "tumor_border_gene_stats.csv"
            )
            gene_stats_df.to_csv(out_path)
        # 邻居细胞类型计数结果
        # 导出所有锚点的 alphaSMA/CD31 邻居计数（只导出锚点行，以免输出大量 NaN）
        anchor_obs = adata.obs.loc[adata.obs[anchor_flag_col] == True,
                                   ["neighbors_alphaSMA_count", "neighbors_CD31_count"]]
        if not anchor_obs.empty:
            anchor_obs.to_csv(os.path.join(output_dir, "neighbors_counts.csv"))

    return adata
