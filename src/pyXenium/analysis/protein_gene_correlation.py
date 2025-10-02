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
    overwrite=False,
    auto_detect_cell_units=True,   # <--- 新增：自动识别 cell 坐标单位（像素/微米）
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
        and per-cell spatial coordinates (either in `adata.obsm['spatial']` or `adata.obs['cell_centroid_x'/'y']`).
    - `transcripts_zarr_path` : str
        Path to `transcripts.zarr` (or `.zarr.zip`) opened via fsspec.
    - `pairs` : list[tuple(str, str)]
        (protein_name, gene_name).
    - `output_dir` : str
        Directory for outputs.
    - `grid_size` : tuple(int, int), default=(50, 50)
        **每格的物理大小（μm）**，格式 (y_size_um, x_size_um)。仅在 `grid_counts=None` 时使用。
    - `grid_counts` : tuple(int, int) or None, default=(50, 50)
        **网格数量**（y 方向、x 方向）。若提供（非 None），**优先于** `grid_size`。
    - `pixel_size_um` : float, default=0.2125
        掩膜/形态学图像的像素大小（μm/px）。映射坐标到掩膜像素索引时会用到。
    - `qv_threshold` : int, default=20
        转录本质量阈值（Q-score）。
    - `overwrite` : bool, default=False
        是否覆盖已有输出。
    - `auto_detect_cell_units` : bool, default=True
        是否**自动判断** `adata` 中细胞坐标（obsm['spatial'] 或 obs 中）是否已经是**微米**。
        若判断为微米，则**不做**像素→微米缩放；若判断为像素，则按 `pixel_size_um` 缩放。
        若不确定，将默认不缩放并给出 warning。你也可以设置为 False 回到旧逻辑：
        只要 `pixel_size_um not in (None, 1, 1.0)` 就进行缩放。

    **Returns:**
    - `summary_df` : pandas.DataFrame with columns: `Protein`, `Gene`, `Pearson_r`, `p_value`.

    **Outputs:**
    - For each (protein, gene) pair:
        - `"<Protein>_<Gene>_correlation.csv"`：**包含全部 `ny×nx`** 个网格记录（含无细胞 bin）。
        - `"<Protein>_<Gene>_scatter.png"`：散点图（仅 `n_cells>0` 的 bin 参与相关性与绘图）。
    - `"protein_gene_correlation_summary.csv"`：汇总表。

    **Notes:**
    - 10x 文档说明：`cells.csv.gz` 与 `transcripts.csv.gz` 的坐标是**微米**；像素到微米约为 **0.2125 µm/px**（全分辨率层）。
      若提供 `protein_qc_mask`，请确保 `pixel_size_um` 与该掩膜图层的像素大小一致。
    """
    # ------------------ helpers ------------------
    def _maybe_convert_cell_coords_auto(cell_xy_in_um_or_px, pixel_size, tx_width, tx_height):
        """
        基于与转录本范围（必为微米）对比的启发式，判断 cell_xy 是否已是微米。
        返回：scaled_cell_xy, note(str)
        """
        if cell_xy_in_um_or_px.size == 0:
            return cell_xy_in_um_or_px, "empty"

        if not auto_detect_cell_units:
            # 旧逻辑：只要 pixel_size 不为 1，就乘
            if pixel_size not in (None, 1, 1.0):
                return cell_xy_in_um_or_px * float(pixel_size), "manual_scaled_by_pixel_size"
            else:
                return cell_xy_in_um_or_px, "manual_no_scale"

        # auto 模式：对比跨度
        cx = cell_xy_in_um_or_px[:, 0]
        cy = cell_xy_in_um_or_px[:, 1]
        cw = float(np.max(cx) - np.min(cx)) if cx.size else 0.0
        ch = float(np.max(cy) - np.min(cy)) if cy.size else 0.0
        rw = float(tx_width)
        rh = float(tx_height)

        # 若任何跨度为 0，无法判断，按“不过缩放”处理（更安全）
        if cw <= 0 or ch <= 0 or rw <= 0 or rh <= 0:
            return cell_xy_in_um_or_px, "auto_fallback_no_scale"

        ratio_w = cw / rw
        ratio_h = ch / rh
        # “已是微米”的目标带宽：~1x（允许一定浮动）
        if 0.6 <= ratio_w <= 1.4 and 0.6 <= ratio_h <= 1.4:
            return cell_xy_in_um_or_px, "auto_detect_microns_no_scale"

        # “像素”的目标带宽：~ 1 / pixel_size
        if pixel_size not in (None, 0, 1, 1.0):
            target = 1.0 / float(pixel_size)
            if 0.6 * target <= ratio_w <= 1.4 * target and 0.6 * target <= ratio_h <= 1.4 * target:
                return cell_xy_in_um_or_px * float(pixel_size), "auto_detect_pixels_scale"

        warnings.warn("[protein_gene_correlation] Unable to confidently infer cell coordinate units; "
                      "assume microns (no scaling). Set auto_detect_cell_units=False to force scaling by pixel_size_um.")
        return cell_xy_in_um_or_px, "auto_uncertain_assume_microns"

    # Ensure output directory path ends with a separator for consistent path joining
    output_dir = output_dir.rstrip('/')

    # Prepare output filesystem and directory
    fs_out, out_dir_path = fsspec.core.url_to_fs(output_dir + '/')
    try:
        fs_out.makedirs(out_dir_path, exist_ok=True)
    except Exception:
        pass

    # ------------------ load cell coords (raw, 不先缩放) ------------------
    if 'spatial' in adata.obsm:
        cell_xy = np.array(adata.obsm['spatial'])
    elif ('cell_centroid_x' in adata.obs) and ('cell_centroid_y' in adata.obs):
        cell_xy = np.vstack([adata.obs['cell_centroid_x'].values, adata.obs['cell_centroid_y'].values]).T
    else:
        raise ValueError("Spatial coordinates not found in adata. Please provide adata.obsm['spatial'] or .obs['cell_centroid_x' and 'cell_centroid_y'].")

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

    # ------------------ open transcripts zarr & get bounds (transcripts在微米空间) ------------------
    if str(transcripts_zarr_path).endswith((".zarr.zip", ".zip")):
        store = zarr.storage.ZipStore(transcripts_zarr_path, mode='r')
    else:
        store = fsspec.get_mapper(transcripts_zarr_path)
    root = zarr.open(store, mode='r')

    # gene index map
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

    # bounds from transcripts (microns)
    region_min_x = float('inf')
    region_min_y = float('inf')
    region_max_x = float('-inf')
    region_max_y = float('-inf')
    if 'origin' in root.attrs:
        orig_attr = root.attrs['origin']
        if isinstance(orig_attr, dict):
            region_min_x = float(orig_attr.get('x', region_min_x))
            region_min_y = float(orig_attr.get('y', region_min_y))
    # if mask given, roughly include mask area extent
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

    # transcripts-only width/height (microns)，用于单位判定
    tx_region_width = region_max_x - region_min_x if (region_max_x > region_min_x) else 0.0
    tx_region_height = region_max_y - region_min_y if (region_max_y > region_min_y) else 0.0

    # 关闭 zarr store 的 close 放在后面（读 counts 时还会用到）

    # ------------------ auto detect & scale cell coords to microns (if needed) ------------------
    # 注意：此处只用于单位判定，不先把 cell 参与到边界中，避免“像素坐标”把范围拉过大
    cell_xy_scaled, scale_note = _maybe_convert_cell_coords_auto(
        cell_xy_in_um_or_px=cell_xy,
        pixel_size=pixel_size_um,
        tx_width=tx_region_width,
        tx_height=tx_region_height
    )
    # 现在可以把“已在微米空间”的 cell 坐标加入到最终边界中
    if cell_xy_scaled.size > 0:
        region_min_x = min(region_min_x, float(cell_xy_scaled[:, 0].min()))
        region_max_x = max(region_max_x, float(cell_xy_scaled[:, 0].max()))
        region_min_y = min(region_min_y, float(cell_xy_scaled[:, 1].min()))
        region_max_y = max(region_max_y, float(cell_xy_scaled[:, 1].max()))

    # ---------------------- Grid definition (grid_counts vs grid_size) ----------------------
    if region_max_x <= region_min_x or region_max_y <= region_min_y:
        # close store before raising
        try:
            store.close()
        except Exception:
            pass
        raise ValueError("Invalid region bounds for spatial data. Please check the coordinates in adata and transcripts.zarr.")
    region_width = region_max_x - region_min_x
    region_height = region_max_y - region_min_y

    if grid_counts is not None:
        ny, nx = int(grid_counts[0]), int(grid_counts[1])
        if ny <= 0 or nx <= 0:
            try:
                store.close()
            except Exception:
                pass
            raise ValueError("`grid_counts` must be positive integers, e.g., (50, 50).")
        x_bin_size = region_width / nx
        y_bin_size = region_height / ny
    else:
        if grid_size is None:
            try:
                store.close()
            except Exception:
                pass
            raise ValueError("Both `grid_counts` and `grid_size` are None. Provide at least one.")
        gy_um, gx_um = float(grid_size[0]), float(grid_size[1])
        if gy_um <= 0 or gx_um <= 0:
            try:
                store.close()
            except Exception:
                pass
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

    # -------------------- BEGIN: robust qv/valid handling per chunk --------------------
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
    # --------------------- END helper -------------------------------------------------

    # -------------------- fill transcript counts --------------------
    for ck in chunk_keys:
        group = level0[ck]
        coords = group['location'][...]
        if coords.shape[0] == 0:
            continue
        gene_ids = group['gene_identity'][...]
        gene_ids = gene_ids.flatten()

        # quality & valid
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
    # 注意：此时 cell_xy_scaled 已在微米空间（根据 auto 判定）
    if cell_xy_scaled.shape[0] > 0:
        cell_x = cell_xy_scaled[:, 0]
        cell_y = cell_xy_scaled[:, 1]
        cell_x_idx = np.floor((cell_x - region_min_x) / x_bin_size).astype(int)
        cell_y_idx = np.floor((cell_y - region_min_y) / y_bin_size).astype(int)
        cell_x_idx[cell_x_idx < 0] = 0
        cell_x_idx[cell_x_idx >= nx] = nx - 1
        cell_y_idx[cell_y_idx < 0] = 0
        cell_y_idx[cell_y_idx >= ny] = ny - 1
    else:
        cell_x_idx = np.array([], dtype=int)
        cell_y_idx = np.array([], dtype=int)

    # ---- 全局有效细胞掩膜（统一用于索引与蛋白强度） ----
    n_cells_total = cell_xy_scaled.shape[0]
    if n_cells_total > 0:
        valid_cell_mask = np.ones(n_cells_total, dtype=bool)
        if mask is not None:
            rel_cx = cell_xy_scaled[:, 0] - mask_origin[0]
            rel_cy = cell_xy_scaled[:, 1] - mask_origin[1]
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

    # -------------------- accumulate proteins & cells --------------------
    protein_sums = {prot: np.zeros((ny, nx), dtype=float) for prot in unique_proteins}
    cell_count = np.zeros((ny, nx), dtype=float)

    if cell_x_idx.size > 0:
        np.add.at(cell_count, (cell_y_idx, cell_x_idx), 1)
        for prot in unique_proteins:
            intensities_all = get_protein_vector(prot)
            if intensities_all.size == n_cells_total and n_cells_total > 0:
                intensities = intensities_all[valid_cell_mask]
            else:
                intensities = np.zeros(cell_x_idx.shape[0], dtype=float)
            if intensities.size != cell_x_idx.size:
                m = min(intensities.size, cell_x_idx.size)
                intensities = intensities[:m]
                cx = cell_x_idx[:m]
                cy = cell_y_idx[:m]
            else:
                cx = cell_x_idx
                cy = cell_y_idx
            np.add.at(protein_sums[prot], (cy, cx), intensities.astype(float))

    # mean protein per bin
    protein_avg = {}
    nonzero = cell_count > 0
    for prot in unique_proteins:
        avg = np.zeros((ny, nx), dtype=float)
        if np.any(nonzero):
            avg[nonzero] = protein_sums[prot][nonzero] / cell_count[nonzero]
        protein_avg[prot] = avg

    # -------------------- prepare outputs --------------------
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

    # iterate pairs
    for protein_name, gene_name in pairs:
        safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
        safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
        pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
        pair_png_name = f"{safe_prot}_{safe_gene}_scatter.png"

        if not overwrite:
            try:
                file_exists_csv = fs_out.exists(f"{out_dir_path}/{pair_csv_name}")
            except Exception:
                try:
                    ftest = fs_out.open(f"{out_dir_path}/{pair_csv_name}", 'rb'); ftest.close()
                    file_exists_csv = True
                except Exception:
                    file_exists_csv = False
            try:
                file_exists_png = fs_out.exists(f"{out_dir_path}/{pair_png_name}")
            except Exception:
                try:
                    ftest = fs_out.open(f"{out_dir_path}/{pair_png_name}", 'rb'); ftest.close()
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

        # compute correlation for this pair
        if gene_name not in transcripts_count:
            raise KeyError(f"Gene '{gene_name}' not processed in transcripts (not in unique_genes set).")
        gene_counts_mat = transcripts_count[gene_name]
        bin_area = x_bin_size * y_bin_size
        transcripts_density = gene_counts_mat.copy() if bin_area <= 0 else (gene_counts_mat / bin_area)

        protein_avg_mat = protein_avg[protein_name] if protein_name in protein_avg else np.zeros((ny, nx), dtype=float)

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

        # plot
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

        # CSV —— 输出全部 ny×nx 个 bin（含无细胞 bin）
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

    # if skipped pairs without old summary, backfill from CSV
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

    # summary df (keep input order)
    summary_df = pd.DataFrame(columns=["Protein", "Gene", "Pearson_r", "p_value"])
    for protein_name, gene_name in pairs:
        rec = next((rec for rec in summary_records if rec["Protein"] == protein_name and rec["Gene"] == gene_name), None)
        if rec is None:
            rec = {"Protein": protein_name, "Gene": gene_name, "Pearson_r": np.nan, "p_value": np.nan}
        summary_df = pd.concat([summary_df, pd.DataFrame([rec])], ignore_index=True)

    with fs_out.open(f"{out_dir_path}/protein_gene_correlation_summary.csv", 'w') as f:
        summary_df.to_csv(f, index=False)

    return summary_df
