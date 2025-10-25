# -*- coding: utf-8 -*-
# pyXenium/utils/name_resolver.py

from __future__ import annotations
from typing import Dict, Tuple, Optional
import os
import re
import yaml
import numpy as np
import pandas as pd
from anndata import AnnData

_NORM = lambda s: ''.join(ch for ch in s.upper() if ch.isalnum())

# 内置兜底（与 YAML 一致）；如 YAML 存在则以 YAML 为准
_FALLBACK_ALIASES = {
    "PD-1":         ["PD1", "PDCD1"],
    "VISTA":        ["VSIR"],
    "PD-L1":        ["PDL1", "CD274"],
    "LAG-3":        ["LAG3"],
    "CD16":         ["FCGR3A", "FCGR3B"],
    "GranzymeB":    ["GZMB", "GRANZYMEB"],
    "CD163":        ["CD163"],
    "CD4":          ["CD4"],
    "CD20":         ["MS4A1"],
    "CD8A":         ["CD8A"],
    "CD3E":         ["CD3E"],
    "CD138":        ["SDC1"],
    "HLA-DR":       ["HLADR", "HLA-DRA", "HLA-DRB1"],
    "CD11c":        ["ITGAX", "CD11C"],
    "CD68":         ["CD68"],
    "CD45RA":       ["CD45-RA", "PTPRC-RA"],
    "PCNA":         ["PCNA"],
    "CD45RO":       ["CD45-RO", "PTPRC-RO"],
    "Ki-67":        ["KI67", "MKI67", "Ki67"],
    "Beta-catenin": ["BETACATENIN", "CTNNB1", "BETACATENIN", "BETACATENIN"],
    "CD31":         ["PECAM1"],
    "PTEN":         ["PTEN"],
    "PanCK":        ["PANCK", "KRT8", "KRT18", "KRT19", "PAN-CK", "CK-PAN"],
    "Vimentin":     ["VIM"],
    "alphaSMA":     ["ACTA2", "ALPHASMA", "ASMA"],
    "CD45":         ["PTPRC"],
    "E-Cadherin":   ["ECADHERIN", "CDH1", "E-CADHERIN"],
}

def _load_aliases_from_yaml() -> Dict[str, list]:
    here = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(os.path.dirname(here), "config", "protein_aliases.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 统一成 {canonical: [synonyms...]}
        normed = {}
        for k, v in data.items():
            if v is None: v = []
            normed[str(k)] = [str(x) for x in v]
        return normed
    return _FALLBACK_ALIASES

def _build_reverse_map(aliases: Dict[str, list]) -> Dict[str, str]:
    """
    返回：规范名映射以及反向查找
      - norm(canonical) -> canonical
      - norm(synonym)   -> canonical
    """
    rev = {}
    for canon, syns in aliases.items():
        rev[_NORM(canon)] = canon
        for s in syns:
            rev[_NORM(s)] = canon
    return rev

def resolve_protein_column(adata: AnnData,
                           token: str,
                           protein_norm_obsm: str = "protein_norm",
                           protein_raw_obsm: str = "protein") -> str:
    """
    把输入 token（基因/抗体/别名）解析为 adata.obsm 中真实存在的列名。
    优先在 protein_norm_obsm 中找（若存在），否则用 protein_raw_obsm。
    """
    aliases = _load_aliases_from_yaml()
    rev = _build_reverse_map(aliases)

    # 取当前可用的蛋白列集合
    obsm_key = protein_norm_obsm if protein_norm_obsm in adata.obsm_keys() else protein_raw_obsm
    df = adata.obsm[obsm_key]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(np.asarray(df), index=adata.obs_names)
    cols = list(getattr(df, "columns", []))
    norm2canon_in_data = {_NORM(c): c for c in cols}

    t = _NORM(token)

    # 1) 直接匹配现有列
    if t in norm2canon_in_data:
        return norm2canon_in_data[t]

    # 2) 别名 -> 规范名 -> 现有列
    if t in rev:
        canon = rev[t]
        tc = _NORM(canon)
        if tc in norm2canon_in_data:
            return norm2canon_in_data[tc]

    # 3) 宽松匹配（前缀/包含）
    for nc, orig in norm2canon_in_data.items():
        if t in nc or nc in t:
            return orig

    raise KeyError(f"Cannot resolve protein column for token '{token}'. Available columns (head): {cols[:10]}")

_STATUS_RE = re.compile(r"^(?P<tok>.+)__status_in_cluster_(?P<cid>.+)$")

def resolve_status_col_name(adata: AnnData,
                            status_token: str,
                            protein_norm_obsm: str = "protein_norm",
                            protein_raw_obsm: str = "protein") -> Optional[str]:
    """
    支持把类似 'MKI67__status_in_cluster_3' 自动纠正为真实存在的
    'Ki-67__status_in_cluster_3'（若后者存在于 adata.obs）。
    """
    m = _STATUS_RE.match(status_token)
    if not m:
        return None
    tok = m.group("tok")
    cid = m.group("cid")
    try:
        canon = resolve_protein_column(adata, tok, protein_norm_obsm, protein_raw_obsm)
        candidate = f"{canon}__status_in_cluster_{cid}"
        if candidate in adata.obs.columns:
            return candidate
    except Exception:
        return None
    return None
