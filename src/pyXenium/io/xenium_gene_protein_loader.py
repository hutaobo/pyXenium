# -*- coding: utf-8 -*-
"""
xenium_gene_protein_loader.py

通用读取 10x Xenium In Situ Gene + Protein 输出（XOA ≥ v4.0）为 AnnData 的工具。
- 自动从 cell_feature_matrix（MEX 或 Zarr 或 HDF5）中拆分 RNA / Protein
- RNA 进入 .X（稀疏计数矩阵），Protein 进入 .obsm["protein"]（float 强度，DataFrame）
- 附带 cells.csv.gz（obs）与几何信息（obs['x'], obs['y'] / 多边形到 .obsm/.uns）
- 可选读取 morphology_focus/ OME-TIFF 与 QC mask（默认关闭，仅留钩子）

参考：
- Protein features & outputs（XOA v4.0+）：10x 文档“Protein data outputs”
- morphology_focus 命名示例与位置：10x “Understanding Xenium Outputs”
- Protein 算法 & 背景偏置说明：10x “Xenium Protein Algorithms”
"""
from __future__ import annotations

import io
import os
import gzip
import json
import warnings
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# 可选依赖，按需导入
try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None
try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

# 为远程/本地透明访问
import fsspec


# ---------------------------
# 工具函数
# ---------------------------

def _open_text(path_or_url: str):
    """以文本模式打开本地或远程文件（自动识别 .gz）"""
    f = fsspec.open(path_or_url, mode="rb").open()
    # 自动解压 .gz
    if path_or_url.endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(fileobj=f), encoding="utf-8")
    # 普通文本
    return io.TextIOWrapper(f, encoding="utf-8")


def _exists(path_or_url: str) -> bool:
    try:
        with fsspec.open(path_or_url).open() as _:
            return True
    except Exception:
        return False


def _join(base: str, *names: str) -> str:
    base = base.rstrip("/")

    # 若 base 看起来已经是完整文件名，直接返回
    if names and any(n.startswith("http") or n.startswith("gs://") for n in names):
        # 子路径本身是完整 URL
        return names[-1]

    for n in names:
        if n is None or n == "":
            continue
        if n.startswith("/"):
            base = n  # 允许绝对路径覆盖
        else:
            base = f"{base}/{n}"
    return base


# ---------------------------
# 读 MEX（三件套）
# ---------------------------

def _read_mex_triplet(mex_dir: str,
                      matrix_name: str = "matrix.mtx.gz",
                      features_name: str = "features.tsv.gz",
                      barcodes_name: str = "barcodes.tsv.gz"
                      ) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    """读取 MEX 为 (csr_matrix, features_df, barcodes_index)"""
    from scipy.io import mmread

    mtx_fp = _join(mex_dir, matrix_name)
    feat_fp = _join(mex_dir, features_name)
    bar_fp = _join(mex_dir, barcodes_name)

    if not all(_exists(p) for p in (mtx_fp, feat_fp, bar_fp)):
        raise FileNotFoundError(
            f"MEX 文件不完整：\n{mtx_fp}\n{feat_fp}\n{bar_fp}"
        )

    with fsspec.open(mtx_fp).open() as f:
        if mtx_fp.endswith(".gz"):
            m = mmread(gzip.GzipFile(fileobj=f)).tocsr()
        else:
            m = mmread(f).tocsr()

    with _open_text(feat_fp) as f:
        # 10x features.tsv.gz 通常 3~5 列：id, name, feature_type[, genome, ...]
        features = pd.read_csv(
            f, sep="\t", header=None, comment="#", engine="python"
        )
    # 兜底列名
    cols = ["id", "name", "feature_type"] + [f"col{i}" for i in range(max(0, features.shape[1] - 3))]
    features.columns = cols[:features.shape[1]]

    with _open_text(bar_fp) as f:
        barcodes = pd.read_csv(f, sep="\t", header=None, engine="python")[0].astype(str).values

    return m, features, pd.Index(barcodes, name="barcode")


# ---------------------------
# 读 Zarr / HDF5（cell_feature_matrix）
# ---------------------------

def _read_cell_feature_matrix_zarr(zarr_root: str) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    """读取 10x Zarr 版 cell_feature_matrix."""
    if zarr is None:
        raise ImportError("需要 zarr>=2 才能读取 Zarr 格式的 cell_feature_matrix")

    # 兼容两种布局：
    # 1) <sample>/cell_feature_matrix.zarr/
    # 2) <sample>/cell_feature_matrix/  (直接是 Zarr store)
    cand = []
    for name in ("cell_feature_matrix.zarr", "cell_feature_matrix"):
        p = _join(zarr_root, name)
        if _exists(p):
            cand.append(p)
    if not cand:
        raise FileNotFoundError("未找到 Zarr cell_feature_matrix（*.zarr 或同名目录）")

    store_path = cand[0]
    store = zarr.open_group(fsspec.get_mapper(store_path), mode="r")

    # 官方约定字段（不同版本可能略有差异）
    data = store["X/data"][:]
    indices = store["X/indices"][:]
    indptr = store["X/indptr"][:]
    shape = tuple(store["X/shape"][:])
    X = sparse.csr_matrix((data, indices, indptr), shape=shape)

    feat = pd.DataFrame({
        "id": store["features/id"][:].astype(str),
        "name": store["features/name"][:].astype(str),
        "feature_type": store["features/feature_type"][:].astype(str),
    })

    barcodes = pd.Index(store["barcodes"][:].astype(str), name="barcode")
    return X, feat, barcodes


from typing import Tuple
import pandas as pd
from scipy import sparse
import h5py, fsspec

def _read_cell_feature_matrix_h5(h5_path: str) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.Index]:
    """Read 10x HDF5 cell_feature_matrix (RNA/Protein). Robust to group names, CSR/CSC, and naming diffs."""
    # 兼容本地/远程
    try:
        fb = fsspec.open(h5_path).open()
        fileobj = h5py.File(fb, "r")
        managed = True
    except Exception:
        fileobj = h5py.File(h5_path, "r")
        managed = False

    def _as_str(arr):
        arr = arr[()]
        # h5py 字节串 -> str
        if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
            return arr.astype(str)
        return arr

    try:
        f = fileobj

        # 1) 找到矩阵分组
        grp = f.get("X") or f.get("matrix") or f.get("cell_feature_matrix")
        if grp is None:
            raise KeyError("Neither 'X' nor 'matrix' nor 'cell_feature_matrix' exists in HDF5.")

        data = grp["data"][()]
        indices = grp["indices"][()]
        indptr = grp["indptr"][()]
        shape = tuple(grp["shape"][()])  # (n_features, n_barcodes) in 10x HDF5

        # 2) 识别是 CSR 还是 CSC
        #    CSR: len(indptr) == n_rows + 1 == shape[0] + 1
        #    CSC: len(indptr) == n_cols + 1 == shape[1] + 1  ← 10x HDF5 常见
        if len(indptr) == shape[0] + 1:
            # 已经是 CSR（行压缩），行=features
            mat = sparse.csr_matrix((data, indices, indptr), shape=shape)
            # 通常我们希望得到 cells x features，因此需要转置
            X = mat.T.tocsr()  # (n_barcodes, n_features)
        elif len(indptr) == shape[1] + 1:
            # 是 CSC（列压缩），列=barcodes
            mat = sparse.csc_matrix((data, indices, indptr), shape=shape)
            # 转成 cells x features 的 CSR
            X = mat.T.tocsr()  # (n_barcodes, n_features)
        else:
            raise ValueError(
                f"Cannot infer matrix format: len(indptr)={len(indptr)}, "
                f"shape={shape} (expect {shape[0]+1} for CSR rows or {shape[1]+1} for CSC cols)."
            )

        # 3) 找 features / barcodes（有的在 grp 下，有的在根）
        def _find(node, name):
            if name in node:
                return node[name]
            # 常见 10x HDF5: features/barcodes 挂在同一层（如 grp 或根）
            if hasattr(node, "parent") and node.parent is not None and name in node.parent:
                return node.parent[name]
            if name in f:
                return f[name]
            return None

        feat_grp = _find(grp, "features")
        if feat_grp is None:
            raise KeyError("Cannot find 'features' group.")

        name_ds = feat_grp.get("name") or feat_grp.get("gene_names")
        if name_ds is None:
            raise KeyError("Cannot find 'features/name' (or 'gene_names').")

        feat = pd.DataFrame({
            "id": _as_str(feat_grp["id"]),
            "name": _as_str(name_ds),
            "feature_type": _as_str(feat_grp["feature_type"]),
        })

        bc_ds = _find(grp, "barcodes")
        if bc_ds is None:
            raise KeyError("Cannot find 'barcodes'.")
        barcodes = pd.Index(_as_str(bc_ds), name="barcode")

        # 4) 一致性检查（可帮助早发现问题）
        n_cells, n_features = X.shape
        if len(barcodes) != n_cells:
            raise ValueError(f"Barcodes length {len(barcodes)} != X.shape[0] (cells) {n_cells}.")
        if len(feat) != n_features:
            raise ValueError(f"Features length {len(feat)} != X.shape[1] (features) {n_features}.")

        return X, feat, barcodes
    finally:
        if managed:
            try: fileobj.close()
            except Exception: pass
        else:
            try: fileobj.close()
            except Exception: pass


# ---------------------------
# 主函数：读取 Gene + Protein
# ---------------------------

def load_xenium_gene_protein(
    base_path: str,
    *,
    prefer: str = "auto",            # "auto" | "zarr" | "h5" | "mex"
    mex_dirname: str = "cell_feature_matrix",
    mex_matrix_name: str = "matrix.mtx.gz",
    mex_features_name: str = "features.tsv.gz",
    mex_barcodes_name: str = "barcodes.tsv.gz",
    cells_csv: str = "cells.csv.gz",
    cells_parquet: Optional[str] = None,
    read_morphology: bool = False,   # 预留：是否读 morphology_focus/
    attach_boundaries: bool = True,  # 若可用，将 cell/nucleus 边界附加进 .uns/.obsm
) -> ad.AnnData:
    """
    从 Xenium 输出目录（本地/远程）读取同时包含 RNA + Protein 的 AnnData。
    参数
    ----
    base_path: 目录或 URL（例如 HuggingFace 原始文件链接所在的“目录 URL”）
    prefer:    读取优先级（默认 auto：依次尝试 zarr > h5 > mex）
    其它参数： 自定义 MEX 三件套与 cells 文件名
    返回
    ----
    AnnData
        - .X: RNA（csr，计数）
        - .var: RNA 基因注释（仅 feature_type == "Gene Expression"）
        - .obsm["protein"]: pd.DataFrame（列为蛋白 marker，值为 scaled mean intensity）
        - .obs: 细胞表（cells.csv.gz/parquet）
        - .uns/.obsm: 可选地附加边界/质心等
    """
    # 1) 选择可用的 cell_feature_matrix 源
    # Zarr
    zarr_candidate = None
    for name in ("cell_feature_matrix.zarr", "cell_feature_matrix"):
        p = _join(base_path, name)
        if _exists(p):
            zarr_candidate = p
            break
    # HDF5
    h5_candidate = _join(base_path, "cell_feature_matrix.h5") if _exists(_join(base_path, "cell_feature_matrix.h5")) else None
    # MEX
    mex_candidate = _join(base_path, mex_dirname) if _exists(_join(base_path, mex_dirname)) else None

    # 策略
    order = {
        "zarr": 0, "h5": 1, "mex": 2
    }
    tried = []

    def read_cfm():
        if prefer in ("auto", "zarr") and zarr_candidate:
            tried.append(zarr_candidate)
            return _read_cell_feature_matrix_zarr(base_path)
        if prefer in ("auto", "h5") and h5_candidate:
            tried.append(h5_candidate)
            return _read_cell_feature_matrix_h5(h5_candidate)
        if prefer in ("auto", "mex") and mex_candidate:
            tried.append(mex_candidate)
            return _read_mex_triplet(mex_candidate, mex_matrix_name, mex_features_name, mex_barcodes_name)
        raise FileNotFoundError(f"未发现可用的 cell_feature_matrix（尝试：{tried or '无'}）")

    X_all, feat_all, barcodes = read_cfm()

    # 2) 拆分 feature_type
    if "feature_type" not in feat_all.columns:
        # 旧版兜底（当缺少类型时视为全是 Gene）
        feat_all["feature_type"] = "Gene Expression"

    # 官方：蛋白为“Protein Expression”，RNA 为“Gene Expression”
    mask_rna = feat_all["feature_type"].astype(str).str.lower().str.contains("gene")
    mask_pro = feat_all["feature_type"].astype(str).str.lower().str.contains("protein")

    # 索引映射
    idx_rna = np.where(mask_rna.values)[0]
    idx_pro = np.where(mask_pro.values)[0]

    # 任何一种缺失都允许（用户可能只有 RNA 或只有 Protein）
    X_rna = X_all[:, idx_rna].tocsr() if idx_rna.size else sparse.csr_matrix((X_all.shape[0], 0))
    var_rna = feat_all.loc[mask_rna, ["id", "name", "feature_type"]].copy()
    var_rna.index = var_rna["id"].values

    if idx_pro.size:
        X_pro = X_all[:, idx_pro].astype(np.float32)  # protein 是强度
        var_pro = feat_all.loc[mask_pro, ["id", "name", "feature_type"]].copy()
        var_pro.index = var_pro["id"].values

        # 以 DataFrame 放到 obsm["protein"]（行与 .obs 对齐；列用 protein marker name，若重名则回退 id）
        pro_names = var_pro["name"].astype(str).values
        # 防止重名
        if len(set(pro_names)) != len(pro_names):
            pro_names = [f"{n}_{i}" for i, n in enumerate(pro_names)]
        protein_df = pd.DataFrame(X_pro.toarray(), index=barcodes, columns=pro_names)
    else:
        protein_df = pd.DataFrame(index=barcodes)

    # 3) cells 表
    obs = None
    if cells_parquet and _exists(_join(base_path, cells_parquet)):
        obs = pd.read_parquet(_join(base_path, cells_parquet))
    elif _exists(_join(base_path, cells_csv)):
        with _open_text(_join(base_path, cells_csv)) as f:
            obs = pd.read_csv(f)
    else:
        warnings.warn("未找到 cells 表（cells.csv.gz 或 parquet）; 将仅使用条码作为 obs。")
        obs = pd.DataFrame(index=barcodes)

    # 对齐 obs 顺序，确保与 barcodes 一致
    if "cell_id" in obs.columns:
        obs = obs.set_index("cell_id")
    if obs.index.name is None or obs.index.name != "barcode":
        obs.index.name = "barcode"
    # 可能出现 obs 与 barcodes 不完全一致，按 barcodes 重建并对齐
    obs = obs.reindex(barcodes).copy()

    # 4) 组装 AnnData
    adata = ad.AnnData(X=X_rna, obs=obs, var=var_rna)
    adata.layers["rna"] = adata.X.copy()  # 显式命名
    adata.obsm["protein"] = protein_df

    # 常用元数据
    adata.uns.setdefault("modality", {})
    adata.uns["modality"]["rna"] = {"feature_type": "Gene Expression"}
    if protein_df.shape[1] > 0:
        adata.uns["modality"]["protein"] = {"feature_type": "Protein Expression", "value": "scaled_mean_intensity"}  # 见 10x 文档

    # 5) 可选：附加边界/质心（如可用）
    if attach_boundaries:
        # cells.csv.gz 中常有 centroid_x/centroid_y/… 字段（不同版本字段名略异）
        for cand_x, cand_y in (("x_centroid", "y_centroid"), ("cell_x_centroid", "cell_y_centroid"),
                               ("centroid_x", "centroid_y")):
            if cand_x in adata.obs.columns and cand_y in adata.obs.columns:
                adata.obsm["spatial"] = adata.obs[[cand_x, cand_y]].to_numpy()
                break

        # cell_boundaries.csv.gz / nucleus_boundaries.csv.gz 可存于 .uns/.obsm
        for fname, key in (("cell_boundaries.csv.gz", "cell_boundaries"),
                           ("nucleus_boundaries.csv.gz", "nucleus_boundaries")):
            p = _join(base_path, fname)
            if _exists(p):
                with _open_text(p) as f:
                    bdf = pd.read_csv(f)
                adata.uns[key] = bdf  # 先原样放入；高阶用户可用 shapely/squidpy 转多边形

    # 6) 预留：读 morphology_focus（默认关闭）
    if read_morphology:
        # 仅做占位说明；实际读取/拼接 OME-TIFF 建议在单独的图像模块中完成
        adata.uns.setdefault("morphology_focus", {"path": _join(base_path, "morphology_focus")})

    return adata
