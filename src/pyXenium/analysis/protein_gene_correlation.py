import os
import numpy as np
import pandas as pd
import fsspec
import zarr
import warnings
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def _safe_get_array(group, key):
    """
    容错地从 zarr group 取数组：
      - 如果 key 不存在或 dtype 元数据非法，返回 None；
      - 避免使用 `'key' in group` 触发严格解析。
    """
    try:
        arr = group[key]  # 直接尝试取
        return arr
    except Exception as e:
        # zarr 3 在遇到非法 dtype（如 'u1'）时会报 ValueError: No Zarr data type found...
        if "No Zarr data type found" in str(e):
            warnings.warn(f"[protein_gene_correlation] Skip '{key}' due to non-canonical dtype in Zarr metadata.")
            return None
        # key 不存在或其它读取错误
        return None


def _safe_read_all(arr, key_name=""):
    """
    容错地把 zarr array 读成 numpy 数组。
    """
    if arr is None:
        return None
    try:
        return arr[...]
    except Exception as e:
        warnings.warn(f"[protein_gene_correlation] Failed to read '{key_name}': {e}. It will be ignored.")
        return None


def protein_gene_correlation(
    adata,
    transcripts_zarr_path,
    pairs,
    output_dir,
    grid_size=(50, 50),
    grid_counts=(50, 50),
    pixel_size_um=0.2125,
    qv_threshold=20,
    overwrite=False
):
    """
    Compute spatial correlation between protein average intensity and gene transcript density for Xenium multi-modal data.

    This function divides the spatial region into a grid and calculates, for each (protein, gene) pair,
    the Pearson correlation between:
      - **Protein average intensity per grid cell** (mean protein expression of all cells in that spatial bin)
      - **Gene transcript density per grid cell** (number of transcripts of the gene in that bin, normalized by bin area).

    It uses the Xenium output data provided in an AnnData object and the transcripts Zarr file to perform the analysis.
    The results include scatter plots and CSV files for each pair, as well as a summary CSV of correlation values.

    **Parameters:**
    - `adata` : AnnData
        AnnData object from Xenium output. It must contain protein expression data in `adata.obsm['protein']`
        (with shape [cells, n_proteins]) and per-cell spatial coordinates (either in `adata.obsm['spatial']` or in `adata.obs` as `cell_centroid_x`/`y`).
        Optionally, `adata.uns['protein_qc_mask']` can be provided (as a 2D array or image path) to mask out regions with low protein data quality).
    - `transcripts_zarr_path` : str
        Path (local or remote) to the Xenium `transcripts.zarr` (or `transcripts.zarr.zip`) file containing transcript locations.
        This path is opened with fsspec, so it can be a local file system path or a remote URL (e.g., S3 URI).
    - `pairs` : list of tuple(str, str)
        List of (protein_name, gene_name) pairs to analyze.
    - `output_dir` : str
        Directory path (local or remote) where output files will be saved.
    - `grid_size` : tuple(int, int), default=(50, 50)
        **每个格子的物理大小（单位：μm），格式为 (y_size_um, x_size_um)**。
        当 `grid_counts` 为 None 时生效：程序会按此大小对区域进行等大小分箱，数量为 `ceil(范围/大小)`。
    - `grid_counts` : tuple(int, int) or None, default=(50, 50)
        **格子数量**（y 方向、x 方向）。若提供（非 None），**优先于** `grid_size`，表示将 y、x 轴分别均分为指定数量的格子。
    - `pixel_size_um` : float, default=0.2125
        Microns per pixel for the spatial coordinates. If spatial coordinates in `adata` are pixel-based, they will be multiplied by this factor.
    - `qv_threshold` : int, default=20
        Transcript quality score cutoff (Q-score)。仅保留 `>= qv_threshold` 的转录本（若该列存在）。
    - `overwrite` : bool, default=False
        是否覆盖已有输出。

    **Returns:**
    - `summary_df` : pandas.DataFrame
        Columns: `Protein`, `Gene`, `Pearson_r`, `p_value`.

    **Outputs:**
    - For each (protein, gene) pair:
        - `"<Protein>_<Gene>_correlation.csv"`：**包含全部 `ny×nx` 个网格**的记录（不再仅限于有细胞的 bin）。
          列：`bin_y`, `bin_x`, `n_cells`, `transcript_count`, `transcript_density`, `protein_avg_intensity`。
        - `"<Protein>_<Gene>_scatter.png"`：散点图（仅以 `n_cells>0` 的 bin 参与相关性与绘图）。
    - `"protein_gene_correlation_summary.csv"`：汇总表。
    """
    # Ensure output directory path ends with a separator for consistent path joining
    output_dir = output_dir.rstrip('/')

    # Prepare output filesystem and directory
    fs_out, out_dir_path = fsspec.core.url_to_fs(output_dir + '/')
    try:
        fs_out.makedirs(out_dir_path, exist_ok=True)
    except Exception:
        pass

    # Extract cell spatial coordinates from AnnData
    if 'spatial' in adata.obsm:
        cell_xy = np.array(adata.obsm['spatial'])
    elif ('cell_centroid_x' in adata.obs) and ('cell_centroid_y' in adata.obs):
        cell_xy = np.vstack([adata.obs['cell_centroid_x'].values, adata.obs['cell_centroid_y'].values]).T
    else:
        raise ValueError("Spatial coordinates not found in adata. Please provide adata.obsm['spatial'] or .obs['cell_centroid_x' and 'cell_centroid_y'].")

    # Convert coordinates from pixels to microns if needed
    if pixel_size_um not in (None, 1, 1.0):
        cell_xy = cell_xy * float(pixel_size_um)

    # Determine unique proteins and genes from the pairs list
    unique_proteins = [p for p, _ in set(pairs)]
    unique_genes = [g for _, g in set(pairs)]

    # Prepare mapping for protein names to data index/column
    protein_data = adata.obsm['protein']
    protein_names = list(protein_data.columns)

    def get_protein_vector(name):
        """Retrieve the per-cell intensity vector for the given protein name."""
        if hasattr(protein_data, "columns"):
            if name in protein_data.columns:
                return protein_data[name].values
            else:
                for col in protein_data.columns:
                    if col == name or col.startswith(name + " "):
                        return protein_data[col].values
            raise KeyError(f"Protein '{name}' not found in adata.obsm['protein'] columns.")
        else:
            # 备用分支：若不是 DataFrame，可在此根据 protein_names 定位
            idx = protein_names.index(name)
            return np.asarray(protein_data)[:, idx]

    # Handle optional protein QC mask
    mask = None
    mask_origin = (0.0, 0.0)
    if 'protein_qc_mask' in adata.uns:
        mask_data = adata.uns['protein_qc_mask']
        if isinstance(mask_data, str):
            try:
                from PIL import Image
                mask_img = Image.open(mask_data)
                mask = np.array(mask_img)
            except ImportError:
                import imageio
                mask = imageio.imread(mask_data)
        elif hasattr(mask_data, "compute"):
            mask = mask_data.compute()
        else:
            mask = np.array(mask_data)
        if mask.dtype != bool:
            mask = mask != 0
        if 'mask_origin' in adata.uns:
            mask_origin = tuple(adata.uns['mask_origin'])
        elif 'origin' in adata.uns:
            try:
                ox, oy = adata.uns['origin']
                mask_origin = (float(ox), float(oy))
            except Exception:
                pass

    # Open transcripts Zarr (allow both .zarr directory or .zarr.zip)
    if str(transcripts_zarr_path).endswith((".zarr.zip", ".zip")):
        store = zarr.storage.ZipStore(transcripts_zarr_path, mode='r')
    else:
        store = fsspec.get_mapper(transcripts_zarr_path)
    root = zarr.open(store, mode='r')

    # Determine gene index mapping for transcripts
    gene_index_map = {}
    gene_names_list = None
    if 'gene_names' in root.attrs:
        try:
            gene_names_list = list(root.attrs['gene_names'])
        except Exception:
            gene_names_list = [str(name) for name in root.attrs['gene_names']]
    if gene_names_list is None:
        raise KeyError("Gene names not found in transcripts.zarr attributes.")
    for gene in unique_genes:
        if gene not in gene_names_list:
            raise KeyError(f"Gene '{gene}' not found in transcripts data.")
        gene_index_map[gene] = gene_names_list.index(gene)

    # Determine spatial bounds of the region (min and max coordinates)
    region_min_x = float('inf')
    region_min_y = float('inf')
    region_max_x = float('-inf')
    region_max_y = float('-inf')
    if 'origin' in root.attrs:
        orig_attr = root.attrs['origin']
        if isinstance(orig_attr, dict):
            region_min_x = float(orig_attr.get('x', region_min_x))
            region_min_y = float(orig_attr.get('y', region_min_y))
    if mask is not None:
        ox, oy = mask_origin
        region_min_x = min(region_min_x, ox)
        region_min_y = min(region_min_y, oy)
        try:
            m_h, m_w = mask.shape[:2]
            region_max_x = max(region_max_x, mask_origin[0] + m_w * float(pixel_size_um))
            region_max_y = max(region_max_y, mask_origin[1] + m_h * float(pixel_size_um))
        except Exception:
            pass

    # Traverse transcript chunks to update bounds (and later count transcripts)
    level0 = root['grids']['0'] if 'grids' in root and '0' in root['grids'] else root
    chunk_keys = list(level0.keys())
    chunk_keys = [k for k in chunk_keys if ',' in k]
    for ck in chunk_keys:
        loc = level0[ck]['location']
        if loc.shape[0] == 0:
            continue
        coords = loc[...]
        if coords.size == 0:
            continue
        x_vals = coords[:, 0]
        y_vals = coords[:, 1]
        region_min_x = min(region_min_x, float(x_vals.min()))
        region_max_x = max(region_max_x, float(x_vals.max()))
        region_min_y = min(region_min_y, float(y_vals.min()))
        region_max_y = max(region_max_y, float(y_vals.max()))
    if cell_xy.size > 0:
        region_min_x = min(region_min_x, float(cell_xy[:, 0].min()))
        region_max_x = max(region_max_x, float(cell_xy[:, 0].max()))
        region_min_y = min(region_min_y, float(cell_xy[:, 1].min()))
        region_max_y = max(region_max_y, float(cell_xy[:, 1].max()))

    # ---------------------- Grid definition (grid_counts vs grid_size) ----------------------
    if region_max_x <= region_min_x or region_max_y <= region_min_y:
        raise ValueError("Invalid region bounds for spatial data. Please check the coordinates in adata and transcripts.zarr.")
    region_width = region_max_x - region_min_x
    region_height = region_max_y - region_min_y

    if grid_counts is not None:
        ny, nx = int(grid_counts[0]), int(grid_counts[1])
        if ny <= 0 or nx <= 0:
            raise ValueError("`grid_counts` must be positive integers, e.g., (50, 50).")
        x_bin_size = region_width / nx
        y_bin_size = region_height / ny
    else:
        if grid_size is None:
            raise ValueError("Both `grid_counts` and `grid_size` are None. Provide at least one.")
        gy_um, gx_um = float(grid_size[0]), float(grid_size[1])
        if gy_um <= 0 or gx_um <= 0:
            raise ValueError("`grid_size` must be positive (μm), e.g., (50, 50).")
        ny = int(np.ceil(region_height / gy_um))
        nx = int(np.ceil(region_width / gx_um))
        ny = max(ny, 1)
        nx = max(nx, 1)
        x_bin_size = region_width / nx
        y_bin_size = region_height / ny
    # ---------------------------------------------------------------------------------------

    # Prepare arrays for transcript counts per bin for each gene
    transcripts_count = {gene: np.zeros((ny, nx), dtype=float) for gene in unique_genes}

    # -------------------- BEGIN PATCH: robust qv/valid handling per chunk --------------------
    def _ensure_1d(name, arr, target_len):
        """
        保证数组是一维且长度与 target_len 一致；若为 2D（常见误读 NxN），警告并压平。
        返回：1D ndarray 或 None。
        """
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim > 1:
            if arr.ndim == 2 and arr.shape[0] == arr.shape[-1]:
                print(f"[WARN] `{name}` appears to be {arr.shape}, flattening to 1D. "
                      f"Check transcripts Zarr key selection.")
            arr = arr.reshape(-1)
        if arr.shape[0] != target_len:
            raise ValueError(
                f"`{name}` length ({arr.shape[0]}) does not match number of transcripts ({target_len}). "
                f"Please verify transcripts Zarr schema/keys."
            )
        return arr
    # --------------------- END PATCH: helper -------------------------------------------------

    # Process transcripts in chunks to fill transcript counts
    for ck in chunk_keys:
        group = level0[ck]
        coords = group['location'][...]
        if coords.shape[0] == 0:
            continue
        gene_ids = group['gene_identity'][...]
        gene_ids = gene_ids.flatten()

        # Quality and valid filters
        qv_arr = _safe_get_array(group, 'quality_score')
        qvs = _safe_read_all(qv_arr, 'quality_score')
        valid_arr = _safe_get_array(group, 'valid')
        valid_flags = _safe_read_all(valid_arr, 'valid')

        N = coords.shape[0]
        qvs = _ensure_1d("quality_score", qvs, N) if qvs is not None else None
        valid_flags = _ensure_1d("valid", valid_flags, N) if valid_flags is not None else None

        mask_all = np.ones(N, dtype=bool)
        if (qvs is not None) and (qv_threshold is not None):
            mask_all &= (qvs >= qv_threshold)
        if valid_flags is not None:
            mask_all &= (valid_flags == 1)

        if qvs is None and valid_flags is None and qv_threshold is not None:
            warnings.warn("[protein_gene_correlation] qv/valid unavailable; proceed without quality filtering.")

        if mask is not None:
            rel_x = coords[:, 0] - mask_origin[0]
            rel_y = coords[:, 1] - mask_origin[1]
            col_idx = np.floor(rel_x / float(pixel_size_um)).astype(int)
            row_idx = np.floor(rel_y / float(pixel_size_um)).astype(int)
            valid_idx = (row_idx >= 0) & (row_idx < mask.shape[0]) & (col_idx >= 0) & (col_idx < mask.shape[1])
            mask_all &= valid_idx
            if mask_all.any():
                inside_idx = np.nonzero(mask_all)[0]
                if inside_idx.size > 0:
                    rr = row_idx[inside_idx]
                    cc = col_idx[inside_idx]
                    mask_vals = mask[rr, cc]
                    inside_mask = np.zeros_like(mask_all)
                    inside_mask[inside_idx] = mask_vals
                    mask_all &= inside_mask.astype(bool)
            else:
                continue

        if not mask_all.any():
            continue
        coords_filt = coords[mask_all]
        gene_ids_filt = gene_ids[mask_all]

        for gene, gene_idx in gene_index_map.items():
            sel = (gene_ids_filt == gene_idx)
            if not np.any(sel):
                continue
            xs = coords_filt[sel, 0]
            ys = coords_filt[sel, 1]
            xi = np.floor((xs - region_min_x) / x_bin_size).astype(int)
            yi = np.floor((ys - region_min_y) / y_bin_size).astype(int)
            xi[xi < 0] = 0
            yi[yi < 0] = 0
            xi[xi >= nx] = nx - 1
            yi[yi >= ny] = ny - 1
            np.add.at(transcripts_count[gene], (yi, xi), 1)

    # Close the transcripts store if applicable to free resources
    try:
        store.close()
    except Exception:
        pass

    # -------------------- Cells -> grid indices (single source of truth) --------------------
    if cell_xy.shape[0] > 0:
        cell_x = cell_xy[:, 0]
        cell_y = cell_xy[:, 1]
        cell_x_idx = np.floor((cell_x - region_min_x) / x_bin_size).astype(int)
        cell_y_idx = np.floor((cell_y - region_min_y) / y_bin_size).astype(int)
        cell_x_idx[cell_x_idx < 0] = 0
        cell_x_idx[cell_x_idx >= nx] = nx - 1
        cell_y_idx[cell_y_idx < 0] = 0
        cell_y_idx[cell_y_idx >= ny] = ny - 1
    else:
        cell_x_idx = np.array([], dtype=int)
        cell_y_idx = np.array([], dtype=int)

    # ---- NEW: 构造“全局有效细胞掩膜”，并与索引绑定（避免蛋白循环里二次掩膜） ----
    n_cells_total = cell_xy.shape[0]
    if n_cells_total > 0:
        valid_cell_mask = np.ones(n_cells_total, dtype=bool)
        if mask is not None:
            rel_cx = cell_xy[:, 0] - mask_origin[0]
            rel_cy = cell_xy[:, 1] - mask_origin[1]
            c_col = np.floor(rel_cx / float(pixel_size_um)).astype(int)
            c_row = np.floor(rel_cy / float(pixel_size_um)).astype(int)
            inside_mask_cells = (c_row >= 0) & (c_row < mask.shape[0]) & (c_col >= 0) & (c_col < mask.shape[1])
            if inside_mask_cells.any():
                mask_vals = np.zeros(n_cells_total, dtype=bool)
                valid_cells_idx = np.where(inside_mask_cells)[0]
                if valid_cells_idx.size > 0:
                    rr = c_row[valid_cells_idx]
                    cc = c_col[valid_cells_idx]
                    mask_vals_seg = mask[rr, cc]
                    mask_vals[valid_cells_idx] = mask_vals_seg
                inside_mask_cells &= mask_vals
            valid_cell_mask &= inside_mask_cells
        # 用同一掩膜过滤网格索引（之后蛋白也用同一掩膜来过滤强度向量）
        if not valid_cell_mask.all():
            kept = np.where(valid_cell_mask)[0]
            if kept.size == 0:
                cell_x_idx = np.array([], dtype=int)
                cell_y_idx = np.array([], dtype=int)
            else:
                cell_x_idx = cell_x_idx[valid_cell_mask]
                cell_y_idx = cell_y_idx[valid_cell_mask]
    else:
        valid_cell_mask = np.array([], dtype=bool)
    # ---------------------------------------------------------------------------------------

    # Prepare arrays for protein intensity sums and cell counts per bin
    protein_sums = {prot: np.zeros((ny, nx), dtype=float) for prot in unique_proteins}
    cell_count = np.zeros((ny, nx), dtype=float)

    # Accumulate cell counts
    if cell_x_idx.size > 0:
        np.add.at(cell_count, (cell_y_idx, cell_x_idx), 1)
        # Accumulate protein intensity sums for each protein
        for prot in unique_proteins:
            intensities_all = get_protein_vector(prot)
            # 用同一掩膜过滤，使其与 cell_x_idx/cell_y_idx 一一对应
            if intensities_all.size == n_cells_total and n_cells_total > 0:
                intensities = intensities_all[valid_cell_mask]
            else:
                # 极端情况下长度不匹配，回退到与索引同长度的零向量
                intensities = np.zeros(cell_x_idx.shape[0], dtype=float)
            if intensities.size != cell_x_idx.size:
                # 再保险：若仍不一致，截断到相同最小长度
                m = min(intensities.size, cell_x_idx.size)
                intensities = intensities[:m]
                cx = cell_x_idx[:m]
                cy = cell_y_idx[:m]
            else:
                cx = cell_x_idx
                cy = cell_y_idx
            np.add.at(protein_sums[prot], (cy, cx), intensities.astype(float))

    # Compute average protein intensity per bin for each protein
    protein_avg = {}
    for prot in unique_proteins:
        avg = np.zeros((ny, nx), dtype=float)
        nonzero = cell_count > 0
        if np.any(nonzero):
            avg[nonzero] = protein_sums[prot][nonzero] / cell_count[nonzero]
        protein_avg[prot] = avg

    # Prepare to generate outputs
    summary_records = []
    skip_pairs = []
    old_summary_df = None
    summary_path = f"{output_dir}/protein_gene_correlation_summary.csv"
    if not overwrite:
        try:
            with fs_out.open(f"{out_dir_path}/protein_gene_correlation_summary.csv", 'r') as f:
                old_summary_df = pd.read_csv(f)
        except FileNotFoundError:
            old_summary_df = None

    # Iterate over each requested pair in the original order
    for protein_name, gene_name in pairs:
        safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
        safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
        pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
        pair_png_name = f"{safe_prot}_{safe_gene}_scatter.png"

        # Check if outputs exist and skip if not overwrite
        if not overwrite:
            try:
                file_exists_csv = fs_out.exists(f"{out_dir_path}/{pair_csv_name}")
            except Exception:
                try:
                    ftest = fs_out.open(f"{out_dir_path}/{pair_csv_name}", 'rb')
                    ftest.close()
                    file_exists_csv = True
                except Exception:
                    file_exists_csv = False
            try:
                file_exists_png = fs_out.exists(f"{out_dir_path}/{pair_png_name}")
            except Exception:
                try:
                    ftest = fs_out.open(f"{out_dir_path}/{pair_png_name}", 'rb')
                    ftest.close()
                    file_exists_png = True
                except Exception:
                    file_exists_png = False
            if file_exists_csv and file_exists_png:
                skip_pairs.append((protein_name, gene_name))
                if old_summary_df is not None:
                    prev = old_summary_df[(old_summary_df["Protein"] == protein_name) & (old_summary_df["Gene"] == gene_name)]
                    if len(prev) > 0:
                        prev_r = float(prev["Pearson_r"].iloc[0])
                        prev_p = float(prev["p_value"].iloc[0])
                        summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": prev_r, "p_value": prev_p})
                continue

        # Compute correlation for this pair using the binned data
        if gene_name not in transcripts_count:
            raise KeyError(f"Gene '{gene_name}' not processed in transcripts (not in unique_genes set).")
        gene_counts_mat = transcripts_count[gene_name]
        bin_area = x_bin_size * y_bin_size
        transcripts_density = gene_counts_mat.copy() if bin_area <= 0 else (gene_counts_mat / bin_area)

        # Get protein average intensity matrix for this protein
        if protein_name not in protein_avg:
            intensities = get_protein_vector(protein_name)
            protein_sums_temp = np.zeros((ny, nx), dtype=float)
            if cell_x_idx.size > 0 and intensities.size == cell_x_idx.size:
                np.add.at(protein_sums_temp, (cell_y_idx, cell_x_idx), intensities.astype(float))
            protein_avg_mat = np.zeros((ny, nx), dtype=float)
            nonzero_mask = cell_count > 0
            if np.any(nonzero_mask):
                protein_avg_mat[nonzero_mask] = protein_sums_temp[nonzero_mask] / cell_count[nonzero_mask]
        else:
            protein_avg_mat = protein_avg[protein_name]

        # Only bins with at least one cell are valid for protein avg intensity correlation
        valid_bins = cell_count > 0
        if not np.any(valid_bins):
            pearson_r = np.nan
            p_val = np.nan
            gene_vals = np.array([])
            prot_vals = np.array([])
        else:
            gene_vals = transcripts_density[valid_bins]
            prot_vals = protein_avg_mat[valid_bins]
            if gene_vals.size < 2 or np.all(gene_vals == gene_vals[0]) or np.all(prot_vals == prot_vals[0]):
                pearson_r = np.nan
                p_val = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    pearson_r, p_val = pearsonr(gene_vals, prot_vals)
        summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": pearson_r, "p_value": p_val})

        # Save scatter plot
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(gene_vals, prot_vals, s=30, alpha=0.7, edgecolors='none')
            ax.set_xlabel(f"{gene_name} transcript density (count/μm²)")
            ax.set_ylabel(f"{protein_name} mean intensity")
            title_str = f"{protein_name} vs {gene_name}"
            if not np.isnan(pearson_r):
                title_str += f"\nPearson r = {pearson_r:.3f}, p = {p_val:.3g}"
            ax.set_title(title_str)
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            with fs_out.open(f"{out_dir_path}/{pair_png_name}", 'wb') as f:
                fig.savefig(f, format='png', dpi=300)
        finally:
            plt.close(fig)

        # Save pair CSV with binned data —— 输出全部 ny×nx 个 bin（含无细胞 bin）
        y_idx, x_idx = np.indices((ny, nx))
        y_idx = y_idx.ravel()
        x_idx = x_idx.ravel()
        records = []
        for yi, xi in zip(y_idx, x_idx):
            records.append({
                "bin_y": int(yi),
                "bin_x": int(xi),
                "n_cells": int(cell_count[yi, xi]),
                "transcript_count": int(transcripts_count[gene_name][yi, xi]),
                "transcript_density": transcripts_density[yi, xi],
                "protein_avg_intensity": protein_avg_mat[yi, xi]
            })
        pair_df = pd.DataFrame(records, columns=["bin_y", "bin_x", "n_cells", "transcript_count", "transcript_density", "protein_avg_intensity"])
        with fs_out.open(f"{out_dir_path}/{pair_csv_name}", 'w') as f:
            pair_df.to_csv(f, index=False)

    # If any pairs were skipped and not added to summary_records (due to missing old summary data), handle them:
    if skip_pairs:
        if old_summary_df is None:
            for protein_name, gene_name in skip_pairs:
                safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
                safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
                pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
                try:
                    with fs_out.open(f"{out_dir_path}/{pair_csv_name}", 'r') as f:
                        df = pd.read_csv(f)
                        if "transcript_density" in df.columns and "protein_avg_intensity" in df.columns and "n_cells" in df.columns:
                            df_valid = df[df["n_cells"] > 0]
                            if df_valid.empty:
                                r_val = np.nan
                                p_val = np.nan
                            else:
                                x = df_valid["transcript_density"].values
                                y = df_valid["protein_avg_intensity"].values
                                if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
                                    r_val = np.nan
                                    p_val = np.nan
                                else:
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                                        r_val, p_val = pearsonr(x, y)
                            summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": r_val, "p_value": p_val})
                except FileNotFoundError:
                    summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": np.nan, "p_value": np.nan})

    # Create summary DataFrame in the original input order
    summary_df = pd.DataFrame(columns=["Protein", "Gene", "Pearson_r", "p_value"])
    for protein_name, gene_name in pairs:
        rec = next((rec for rec in summary_records if rec["Protein"] == protein_name and rec["Gene"] == gene_name), None)
        if rec is None:
            rec = {"Protein": protein_name, "Gene": gene_name, "Pearson_r": np.nan, "p_value": np.nan}
        summary_df = pd.concat([summary_df, pd.DataFrame([rec])], ignore_index=True)

    # Save summary CSV
    with fs_out.open(f"{out_dir_path}/protein_gene_correlation_summary.csv", 'w') as f:
        summary_df.to_csv(f, index=False)

    return summary_df
