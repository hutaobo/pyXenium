# pyXenium/analysis/microenv_analysis.py
from __future__ import annotations
from typing import Sequence, Optional, Dict, Any
import os
import inspect
import numpy as np
import pandas as pd
from anndata import AnnData

# 依赖内部 ProteinMicroEnv
try:
    from pyXenium.analysis.protein_microenvironment import ProteinMicroEnv
except Exception as e:
    raise ImportError(
        "未能导入 pyXenium.analysis.protein_microenvironment.ProteinMicroEnv，"
        "请确认该类已包含在包内并可被导入。"
    ) from e


# ---------------------------
# 工具：方法自适应与存储
# ---------------------------

def _subset_kwargs_by_signature(func, **kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def _call_first_available(obj, names: Sequence[str], **kwargs):
    last = None
    for nm in names:
        if hasattr(obj, nm):
            fn = getattr(obj, nm)
            try:
                return fn(**_subset_kwargs_by_signature(fn, **kwargs))
            except Exception as e:
                last = e
    raise RuntimeError(f"尝试的方法均不可用：{names}\n最后错误：{last!r}")

def _protein_df(adata: AnnData) -> pd.DataFrame:
    prot = adata.obsm.get("protein", None)
    if prot is None:
        raise ValueError("当前 AnnData 不包含 obsm['protein']。")
    if isinstance(prot, pd.DataFrame):
        return prot
    # 若不是 DataFrame，则构造列名兜底
    cols = getattr(prot, "columns", None)
    if cols is None:
        cols = [f"p{i}" for i in range(prot.shape[1])]
    return pd.DataFrame(prot, index=adata.obs_names, columns=cols)

def _normalize_name(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")

def _resolve_protein_column(adata: AnnData, preferred: Sequence[str]) -> Optional[str]:
    """
    在 obsm['protein'].columns 中按同义名/大小写不敏感方式查找最佳列。
    """
    prot = _protein_df(adata)
    norm_cols = {_normalize_name(c): c for c in prot.columns}
    # 一些常用同义词
    synonyms = {
        "cd8": ["cd8", "cd8a"],
        "cd45": ["cd45", "ptprc", "cd45ra", "cd45rb", "cd45ro"],
        "cd3": ["cd3", "cd3e", "cd3d"],
        "panck": ["panck", "pancytokeratin", "pan-cytokeratin", "pan-ck"],
        "ecadherin": ["ecadherin", "e-cadherin", "ecad"],
        "epcam": ["epcam"],
        "alphasma": ["alphasma", "αsma", "alpha-sma", "acta2"],
        "cd31": ["cd31", "pecam1"],
    }
    # 把 preferred 展开成同义词列表
    cand_norms: list[str] = []
    for p in preferred:
        key = _normalize_name(p)
        # 同义词集合
        cand_norms.extend(synonyms.get(key, [key]))
    # 逐个匹配
    for c in cand_norms:
        if c in norm_cols:
            return norm_cols[c]
    return None

def _guess_transcripts_path(base_path: str) -> str:
    cands = [
        os.path.join(base_path, "transcripts.zarr.zip"),
        os.path.join(base_path, "transcripts.zarr"),
        os.path.join(base_path, "analysis", "transcripts.zarr.zip"),
        os.path.join(base_path, "analysis", "transcripts.zarr"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    # 没找到就返回首选，交由 PM 本身报错
    return cands[0]

def _pm_init(
    adata: AnnData,
    transcripts_path: Optional[str],
    pixel_size_um: float,
    qv_threshold: int,
    verbose: bool = True,
) -> ProteinMicroEnv:
    ctor = ProteinMicroEnv
    kw = dict(
        adata=adata,
        transcripts_zarr_path=transcripts_path,
        pixel_size_um=pixel_size_um,
        qv_threshold=qv_threshold,
        auto_detect_cell_units=True,
        verbose=verbose,
    )
    kw = _subset_kwargs_by_signature(ctor, **kw)
    return ctor(**kw)

def _set_anchors(env: ProteinMicroEnv, anchor_indices: np.ndarray):
    return _call_first_available(
        env,
        ["set_anchor_cells", "set_anchors_from_cells", "set_anchor_indices", "set_anchors"],
        anchor_indices=np.asarray(anchor_indices, dtype=int),
        indices=np.asarray(anchor_indices, dtype=int),
    )

def _set_rings(env: ProteinMicroEnv, ring_edges_um: Sequence[float]):
    return _call_first_available(
        env,
        ["set_neighborhood", "set_rings", "set_neighborhood_rings"],
        mode="rings",
        ring_edges_um=list(ring_edges_um),
        ring_edges=list(ring_edges_um),
    )

def _compute_gene_stats(
    env: ProteinMicroEnv,
    genes: Sequence[str],
    background: str = "global",
    area_norm: bool = True,
    return_long: bool = False,
):
    """
    返回通常是：index=锚点（或条形码），columns=每个 (gene@ring) 的宽表。
    """
    candidates = [
        ("compute_transcript_stats", dict(genes=genes,
                                          normalize_by_area=area_norm,
                                          background=background,
                                          return_long=return_long)),
        ("compute_gene_density", dict(genes=genes,
                                      background=background,
                                      area_normalized=area_norm)),
        ("compute_transcript_density", dict(genes=genes,
                                            background=background,
                                            area_normalized=area_norm)),
    ]
    last = None
    for name, kw in candidates:
        if hasattr(env, name):
            try:
                fn = getattr(env, name)
                return fn(**_subset_kwargs_by_signature(fn, **kw))
            except Exception as e:
                last = e
    raise RuntimeError(f"ProteinMicroEnv 无可用基因统计接口；最后错误：{last!r}")

def _compute_neighbor_cells(env: ProteinMicroEnv, cell_types: pd.Series, how: str = "fraction"):
    return _call_first_available(env, ["compute_neighbor_cells", "neighbor_cell_stats"],
                                 cell_types=cell_types, how=how)

def _store_anchor_df(
    adata: AnnData,
    df: pd.DataFrame,
    anchor_indices: np.ndarray,
    obsm_key: str,
    uns_key: str,
):
    """
    存储策略：
      1) 完整 DataFrame 放到 uns[uns_key]（保留行索引信息）；
      2) 同步一份到 obsm[obsm_key]：构造 (n_obs × df.shape[1]) 的矩阵，
         非锚点行填 NaN；行匹配优先按名字对齐，否则按传入的 anchor_indices 顺序放置。
    """
    if not isinstance(df, pd.DataFrame):
        # 尝试包装
        df = pd.DataFrame(df)

    # 1) 放 uns
    adata.uns[uns_key] = df.copy()

    # 2) 铺到 obsm（与 obs 对齐）
    mat = np.full((adata.n_obs, df.shape[1]), np.nan, dtype=float)

    if np.all(np.isin(df.index, adata.obs_names)):
        row_idx = adata.obs_names.get_indexer(df.index)
    else:
        # 行数不一定等于锚点数，这里取两者的 min 并按次序放置
        n = min(len(anchor_indices), len(df))
        row_idx = np.asarray(anchor_indices[:n], dtype=int)
        mat[row_idx, :] = df.iloc[:n, :].to_numpy()
        adata.obsm[obsm_key] = mat
        adata.uns[obsm_key + "_cols"] = list(df.columns)
        return

    mat[row_idx, :] = df.to_numpy()
    adata.obsm[obsm_key] = mat
    adata.uns[obsm_key + "_cols"] = list(df.columns)


# ---------------------------
# 免疫微环境
# ---------------------------

def run_immune_microenvironment(
    adata: AnnData,
    base_path: Optional[str] = None,
    transcripts_path: Optional[str] = None,
    ring_edges_um: Sequence[float] = (0, 10, 20, 40),
    pixel_size_um: float = 1.0,
    qv_threshold: int = 20,
    genes: Sequence[str] = ("PDCD1", "CTLA4", "CCL5", "CXCL9", "CXCL10"),
    out_prefix: str = "immune",
) -> Dict[str, Any]:
    prot = _protein_df(adata)

    # 锚点：CD8 & CD45（若有 CD3 也可；不存在的自动忽略）
    col_cd8  = _resolve_protein_column(adata, ["CD8", "CD8A"])
    col_cd45 = _resolve_protein_column(adata, ["CD45", "PTPRC"])
    col_cd3  = _resolve_protein_column(adata, ["CD3", "CD3E", "CD3D"])

    markers = [c for c in [col_cd8, col_cd45, col_cd3] if c is not None]
    if not markers:
        raise ValueError("找不到用于免疫锚点的蛋白列（CD8/CD45[可选CD3]）。")

    mask = np.ones(adata.n_obs, dtype=bool)
    for m in markers:
        mask &= (prot[m].to_numpy(dtype=float) > 0.0)
    anchor_idx = np.where(mask)[0]
    adata.obs[f"{out_prefix}_is_anchor"] = False
    if anchor_idx.size:
        adata.obs.iloc[anchor_idx, adata.obs.columns.get_loc(f"{out_prefix}_is_anchor")] = True

    # 初始化 ProteinMicroEnv
    if transcripts_path is None and base_path is not None:
        transcripts_path = _guess_transcripts_path(base_path)
    env = _pm_init(adata, transcripts_path, pixel_size_um=pixel_size_um, qv_threshold=qv_threshold)

    # 设置锚点与邻域
    _set_anchors(env, anchor_idx)
    _set_rings(env, ring_edges_um)

    # 基因邻域统计（宽表）
    gene_df = _compute_gene_stats(env, list(genes), background="global", area_norm=True, return_long=False)
    if isinstance(gene_df, pd.DataFrame):
        _store_anchor_df(adata, gene_df, anchor_idx,
                         obsm_key=f"{out_prefix}_gene_stats",
                         uns_key=f"{out_prefix}_gene_stats")

    # 邻居细胞组成（alphaSMA=CAF；CD31=Endothelial）
    col_asma = _resolve_protein_column(adata, ["alphaSMA", "ACTA2"])
    col_cd31 = _resolve_protein_column(adata, ["CD31", "PECAM1"])
    cell_type = pd.Series("Other", index=adata.obs_names, dtype=object)
    if col_asma is not None:
        cell_type.loc[prot[col_asma].to_numpy(dtype=float) > 0.0] = "CAF"
    if col_cd31 is not None:
        cell_type.loc[prot[col_cd31].to_numpy(dtype=float) > 0.0] = "Endothelial"

    # how='fraction'：返回各环占比；如果你的 PM 实现支持 how='count'，也可以改成 'count'
    comp_df = _compute_neighbor_cells(env, cell_type, how="fraction")
    if isinstance(comp_df, pd.DataFrame):
        _store_anchor_df(adata, comp_df, anchor_idx,
                         obsm_key=f"{out_prefix}_neighbor_composition",
                         uns_key=f"{out_prefix}_neighbor_composition")

    return {"env": env, "anchors": anchor_idx}


# ---------------------------
# 肿瘤-间质边界
# ---------------------------

def run_tumor_stroma_border(
    adata: AnnData,
    base_path: Optional[str] = None,
    transcripts_path: Optional[str] = None,
    ring_edges_um: Sequence[float] = (0, 10, 20, 40),
    pixel_size_um: float = 1.0,
    qv_threshold: int = 20,
    ecm_genes: Sequence[str] = ("COL1A1", "COL3A1", "FN1"),
    out_prefix: str = "tumor_border",
) -> Dict[str, Any]:
    prot = _protein_df(adata)

    # 锚点：上皮蛋白（优先 PanCK，其次 E-Cadherin/EPCAM）
    col_panck = _resolve_protein_column(adata, ["PanCK"])
    col_ecad  = _resolve_protein_column(adata, ["E-Cadherin", "ECADHERIN", "ECAD"])
    col_epcam = _resolve_protein_column(adata, ["EPCAM"])
    anchor_col = next((c for c in [col_panck, col_ecad, col_epcam] if c is not None), None)
    if anchor_col is None:
        raise ValueError("找不到用于肿瘤锚点的蛋白列（PanCK/E-Cadherin/EPCAM）。")

    mask = prot[anchor_col].to_numpy(dtype=float) > 0.0
    anchor_idx = np.where(mask)[0]
    adata.obs[f"{out_prefix}_is_anchor"] = False
    if anchor_idx.size:
        adata.obs.iloc[anchor_idx, adata.obs.columns.get_loc(f"{out_prefix}_is_anchor")] = True

    if transcripts_path is None and base_path is not None:
        transcripts_path = _guess_transcripts_path(base_path)
    env = _pm_init(adata, transcripts_path, pixel_size_um=pixel_size_um, qv_threshold=qv_threshold)

    _set_anchors(env, anchor_idx)
    _set_rings(env, ring_edges_um)

    # 邻居细胞组成：CAF / Endothelial
    col_asma = _resolve_protein_column(adata, ["alphaSMA", "ACTA2"])
    col_cd31 = _resolve_protein_column(adata, ["CD31", "PECAM1"])
    cell_type = pd.Series("Other", index=adata.obs_names, dtype=object)
    if col_asma is not None:
        cell_type.loc[prot[col_asma].to_numpy(dtype=float) > 0.0] = "CAF"
    if col_cd31 is not None:
        cell_type.loc[prot[col_cd31].to_numpy(dtype=float) > 0.0] = "Endothelial"

    comp_df = _compute_neighbor_cells(env, cell_type, how="fraction")
    if isinstance(comp_df, pd.DataFrame):
        _store_anchor_df(adata, comp_df, anchor_idx,
                         obsm_key=f"{out_prefix}_neighbor_composition",
                         uns_key=f"{out_prefix}_neighbor_composition")

    # ECM 基因邻域统计
    ecm_df = _compute_gene_stats(env, list(ecm_genes), background="global", area_norm=True, return_long=False)
    if isinstance(ecm_df, pd.DataFrame):
        _store_anchor_df(adata, ecm_df, anchor_idx,
                         obsm_key=f"{out_prefix}_ecm_gene_stats",
                         uns_key=f"{out_prefix}_ecm_gene_stats")

    return {"env": env, "anchors": anchor_idx}


# ---------------------------
# 统一入口（Notebook 友好）
# ---------------------------

def analyze_microenvironment(
    mode: str,
    adata: AnnData,
    base_path: Optional[str] = None,
    transcripts_path: Optional[str] = None,
    ring_edges_um: Sequence[float] = (0, 10, 20, 40),
    pixel_size_um: float = 1.0,
    qv_threshold: int = 20,
    output_dir: Optional[str] = None,
) -> AnnData:
    """
    统一入口：执行 'immune' 或 'tumor_border' 分析。
    - 结果表（按锚点×特征）存到 adata.uns[...]，并铺平到 adata.obsm[...]（非锚点=NaN）。
    - 锚点布尔标记写入 adata.obs['<prefix>_is_anchor']。
    - 若 output_dir 指定，会把主要结果另存 CSV。
    """
    mode = mode.lower()
    if mode == "immune":
        res = run_immune_microenvironment(
            adata=adata, base_path=base_path, transcripts_path=transcripts_path,
            ring_edges_um=ring_edges_um, pixel_size_um=pixel_size_um, qv_threshold=qv_threshold,
        )
        prefix = "immune"
        keys_to_dump = [
            f"{prefix}_gene_stats",
            f"{prefix}_neighbor_composition",
        ]
    elif mode == "tumor_border":
        res = run_tumor_stroma_border(
            adata=adata, base_path=base_path, transcripts_path=transcripts_path,
            ring_edges_um=ring_edges_um, pixel_size_um=pixel_size_um, qv_threshold=qv_threshold,
        )
        prefix = "tumor_border"
        keys_to_dump = [
            f"{prefix}_ecm_gene_stats",
            f"{prefix}_neighbor_composition",
        ]
    else:
        raise ValueError("mode 必须是 'immune' 或 'tumor_border'。")

    # 可选把 uns 表格另存 CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for k in keys_to_dump:
            df = adata.uns.get(k, None)
            if isinstance(df, pd.DataFrame):
                df.to_csv(os.path.join(output_dir, f"{k}.csv"))
        # 同时保存锚点名单
        anchors = res.get("anchors", np.array([], dtype=int))
        pd.Series(adata.obs_names[anchors]).to_csv(
            os.path.join(output_dir, f"{prefix}_anchors.csv"), index=False, header=False
        )

    return adata
