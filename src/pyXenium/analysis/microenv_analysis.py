# pyXenium/analysis/microenv_analysis.py
from __future__ import annotations
from typing import Sequence, Optional, Dict, Any
import os
import inspect
import numpy as np
import pandas as pd
from anndata import AnnData

# Dependency: internal ProteinMicroEnv
try:
    from pyXenium.analysis.protein_microenvironment import ProteinMicroEnv
except Exception as e:
    raise ImportError(
        "Failed to import pyXenium.analysis.protein_microenvironment.ProteinMicroEnv. "
        "Please ensure that this class is included in the package and can be imported."
    ) from e

# ---------------------------
# Tools: method adaptation and storage
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
    raise RuntimeError(f"All attempted methods are unavailable: {names}\nLast error: {last!r}")

def _protein_df(adata: AnnData) -> pd.DataFrame:
    prot = adata.obsm.get("protein", None)
    if prot is None:
        raise ValueError("The current AnnData does not contain obsm['protein'].")
    if isinstance(prot, pd.DataFrame):
        return prot
    # If not a DataFrame, construct fallback column names
    cols = getattr(prot, "columns", None)
    if cols is None:
        cols = [f"p{i}" for i in range(prot.shape[1])]
    return pd.DataFrame(prot, index=adata.obs_names, columns=cols)

def _normalize_name(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")

def _resolve_protein_column(adata: AnnData, preferred: Sequence[str]) -> Optional[str]:
    """
    Find the best column in obsm['protein'].columns using synonyms and case-insensitive matching.
    """
    prot = _protein_df(adata)
    norm_cols = {_normalize_name(c): c for c in prot.columns}
    # Some common synonyms
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
    # Expand preferred list into synonyms
    cand_norms: list[str] = []
    for p in preferred:
        key = _normalize_name(p)
        # Synonym set
        cand_norms.extend(synonyms.get(key, [key]))
    # Match one by one
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
    # If not found, return the first candidate and let ProteinMicroEnv handle the error.
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
    Return typically a wide DataFrame with index as anchors (or barcodes), columns for each (gene@ring).
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
    raise RuntimeError(f"No available gene statistics interface in ProteinMicroEnv; last error: {last!r}")

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
    Storage strategy:
      1) Put the complete DataFrame into uns[uns_key] (preserve row index information);
      2) Also sync a copy to obsm[obsm_key]: construct an (n_obs × df.shape[1]) matrix,
         fill non-anchor rows with NaN; align rows by name if possible, otherwise place in order of the provided anchor_indices.
    """
    if not isinstance(df, pd.DataFrame):
        # Try to wrap
        df = pd.DataFrame(df)

    # 1) in uns
    adata.uns[uns_key] = df.copy()

    # 2) sync to obsm (aligned with obs)
    mat = np.full((adata.n_obs, df.shape[1]), np.nan, dtype=float)

    if np.all(np.isin(df.index, adata.obs_names)):
        row_idx = adata.obs_names.get_indexer(df.index)
    else:
        # The number of rows may not equal the number of anchors; take the smaller of the two and place in order
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
# Immune microenvironment
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

    # Anchors: CD8 & CD45 (CD3 if available; automatically ignored if not present)
    col_cd8  = _resolve_protein_column(adata, ["CD8", "CD8A"])
    col_cd45 = _resolve_protein_column(adata, ["CD45", "PTPRC"])
    col_cd3  = _resolve_protein_column(adata, ["CD3", "CD3E", "CD3D"])

    markers = [c for c in [col_cd8, col_cd45, col_cd3] if c is not None]
    if not markers:
        raise ValueError("Cannot find a protein column for immune anchors (CD8/CD45 [optional CD3]).")

    mask = np.ones(adata.n_obs, dtype=bool)
    for m in markers:
        mask &= (prot[m].to_numpy(dtype=float) > 0.0)
    anchor_idx = np.where(mask)[0]
    adata.obs[f"{out_prefix}_is_anchor"] = False
    if anchor_idx.size:
        adata.obs.iloc[anchor_idx, adata.obs.columns.get_loc(f"{out_prefix}_is_anchor")] = True

    # Initialize ProteinMicroEnv
    if transcripts_path is None and base_path is not None:
        transcripts_path = _guess_transcripts_path(base_path)
    env = _pm_init(adata, transcripts_path, pixel_size_um=pixel_size_um, qv_threshold=qv_threshold)

    # Set anchors and neighborhood
    _set_anchors(env, anchor_idx)
    _set_rings(env, ring_edges_um)

    # Gene neighborhood statistics (wide table)
    gene_df = _compute_gene_stats(env, list(genes), background="global", area_norm=True, return_long=False)
    if isinstance(gene_df, pd.DataFrame):
        _store_anchor_df(adata, gene_df, anchor_idx,
                         obsm_key=f"{out_prefix}_gene_stats",
                         uns_key=f"{out_prefix}_gene_stats")

    # Neighbor cell composition (alphaSMA = CAF; CD31 = Endothelial)
    col_asma = _resolve_protein_column(adata, ["alphaSMA", "ACTA2"])
    col_cd31 = _resolve_protein_column(adata, ["CD31", "PECAM1"])
    cell_type = pd.Series("Other", index=adata.obs_names, dtype=object)
    if col_asma is not None:
        cell_type.loc[prot[col_asma].to_numpy(dtype=float) > 0.0] = "CAF"
    if col_cd31 is not None:
        cell_type.loc[prot[col_cd31].to_numpy(dtype=float) > 0.0] = "Endothelial"

    # how='fraction': returns fraction of each ring; if your ProteinMicroEnv implementation supports 'count', you can also use 'count'.
    comp_df = _compute_neighbor_cells(env, cell_type, how="fraction")
    if isinstance(comp_df, pd.DataFrame):
        _store_anchor_df(adata, comp_df, anchor_idx,
                         obsm_key=f"{out_prefix}_neighbor_composition",
                         uns_key=f"{out_prefix}_neighbor_composition")

    return {"env": env, "anchors": anchor_idx}

# ---------------------------
# Tumor-Stroma border
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

    # Anchor: epithelial protein (prefer PanCK, then E-Cadherin/EPCAM)
    col_panck = _resolve_protein_column(adata, ["PanCK"])
    col_ecad  = _resolve_protein_column(adata, ["E-Cadherin", "ECADHERIN", "ECAD"])
    col_epcam = _resolve_protein_column(adata, ["EPCAM"])
    anchor_col = next((c for c in [col_panck, col_ecad, col_epcam] if c is not None), None)
    if anchor_col is None:
        raise ValueError("Cannot find a protein column for tumor anchor (PanCK/E-Cadherin/EPCAM).")

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

    # Neighbor cell composition: CAF / Endothelial
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

    # ECM gene neighborhood statistics
    ecm_df = _compute_gene_stats(env, list(ecm_genes), background="global", area_norm=True, return_long=False)
    if isinstance(ecm_df, pd.DataFrame):
        _store_anchor_df(adata, ecm_df, anchor_idx,
                         obsm_key=f"{out_prefix}_ecm_gene_stats",
                         uns_key=f"{out_prefix}_ecm_gene_stats")

    return {"env": env, "anchors": anchor_idx}

# ---------------------------
# Unified entry point (Notebook-friendly)
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
    Unified entry point: execute 'immune' or 'tumor_border' analysis.
    - The result tables (anchor x feature) are stored in adata.uns[...] and flattened into adata.obsm[...] (non-anchors = NaN).
    - Anchor boolean labels are written to adata.obs['<prefix>_is_anchor'].
    - If output_dir is specified, the main results will be saved as CSV files.
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
        raise ValueError("mode must be 'immune' or 'tumor_border'.")

    # Optionally save uns tables as CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for k in keys_to_dump:
            df = adata.uns.get(k, None)
            if isinstance(df, pd.DataFrame):
                df.to_csv(os.path.join(output_dir, f"{k}.csv"))
        # Also save anchor list
        anchors = res.get("anchors", np.array([], dtype=int))
        pd.Series(adata.obs_names[anchors]).to_csv(
            os.path.join(output_dir, f"{prefix}_anchors.csv"), index=False, header=False
        )

    return adata
