from __future__ import annotations

import os
import warnings

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from pyXenium.io.xenium_artifacts import read_transcripts_table


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
    auto_detect_cell_units=True,
):
    """
    Compute spatial correlation between protein intensity and transcript density.

    This implementation now reuses pyXenium's shared Xenium transcript reader so
    transcript schema handling stays aligned with the main I/O layer.
    """

    def _maybe_convert_cell_coords_auto(cell_xy_in_um_or_px, pixel_size, tx_width, tx_height):
        if cell_xy_in_um_or_px.size == 0:
            return cell_xy_in_um_or_px, "empty"

        if not auto_detect_cell_units:
            if pixel_size not in (None, 1, 1.0):
                return cell_xy_in_um_or_px * float(pixel_size), "manual_scaled_by_pixel_size"
            return cell_xy_in_um_or_px, "manual_no_scale"

        cx = cell_xy_in_um_or_px[:, 0]
        cy = cell_xy_in_um_or_px[:, 1]
        cw = float(np.max(cx) - np.min(cx)) if cx.size else 0.0
        ch = float(np.max(cy) - np.min(cy)) if cy.size else 0.0
        rw = float(tx_width)
        rh = float(tx_height)

        if cw <= 0 or ch <= 0 or rw <= 0 or rh <= 0:
            return cell_xy_in_um_or_px, "auto_fallback_no_scale"

        ratio_w = cw / rw
        ratio_h = ch / rh
        if 0.6 <= ratio_w <= 1.4 and 0.6 <= ratio_h <= 1.4:
            return cell_xy_in_um_or_px, "auto_detect_microns_no_scale"

        if pixel_size not in (None, 0, 1, 1.0):
            target = 1.0 / float(pixel_size)
            if 0.6 * target <= ratio_w <= 1.4 * target and 0.6 * target <= ratio_h <= 1.4 * target:
                return cell_xy_in_um_or_px * float(pixel_size), "auto_detect_pixels_scale"

        warnings.warn(
            "[protein_gene_correlation] Unable to confidently infer cell coordinate units; "
            "assume microns (no scaling). Set auto_detect_cell_units=False to force scaling by pixel_size_um."
        )
        return cell_xy_in_um_or_px, "auto_uncertain_assume_microns"

    output_dir = output_dir.rstrip("/")
    fs_out, out_dir_path = fsspec.core.url_to_fs(output_dir + "/")
    try:
        fs_out.makedirs(out_dir_path, exist_ok=True)
    except Exception:
        pass

    if "spatial" in adata.obsm:
        cell_xy = np.asarray(adata.obsm["spatial"])
    elif ("cell_centroid_x" in adata.obs) and ("cell_centroid_y" in adata.obs):
        cell_xy = np.c_[adata.obs["cell_centroid_x"].to_numpy(), adata.obs["cell_centroid_y"].to_numpy()]
    else:
        raise ValueError(
            "Spatial coordinates not found in adata. Please provide adata.obsm['spatial'] "
            "or obs['cell_centroid_x'/'cell_centroid_y']."
        )

    unique_proteins = [protein for protein, _ in dict.fromkeys(pairs)]
    unique_genes = [gene for _, gene in dict.fromkeys(pairs)]

    protein_data = adata.obsm["protein"]
    protein_names = list(protein_data.columns) if hasattr(protein_data, "columns") else []

    def get_protein_vector(name):
        if hasattr(protein_data, "columns"):
            if name in protein_data.columns:
                return protein_data[name].to_numpy()
            for column in protein_data.columns:
                if column == name or column.startswith(name + " "):
                    return protein_data[column].to_numpy()
            raise KeyError(f"Protein '{name}' not found in adata.obsm['protein'].")
        return np.asarray(protein_data)[:, protein_names.index(name)]

    mask = None
    mask_origin = (0.0, 0.0)
    if "protein_qc_mask" in adata.uns:
        mask_data = adata.uns["protein_qc_mask"]
        if isinstance(mask_data, str):
            try:
                from PIL import Image

                mask = np.asarray(Image.open(mask_data))
            except ImportError:
                import imageio

                mask = imageio.imread(mask_data)
        elif hasattr(mask_data, "compute"):
            mask = mask_data.compute()
        else:
            mask = np.asarray(mask_data)
        if mask.dtype != bool:
            mask = mask != 0
        if "mask_origin" in adata.uns:
            mask_origin = tuple(adata.uns["mask_origin"])
        elif "origin" in adata.uns:
            try:
                ox, oy = adata.uns["origin"]
                mask_origin = (float(ox), float(oy))
            except Exception:
                pass

    transcripts = read_transcripts_table(transcripts_zarr_path, genes=unique_genes)
    if transcripts.empty:
        raise ValueError("No transcripts were loaded from the provided Xenium transcript artifact.")

    available_genes = set(transcripts["gene_name"].astype(str))
    missing_genes = [gene for gene in unique_genes if gene not in available_genes]
    if missing_genes:
        raise KeyError(f"Genes not found in transcripts data: {missing_genes}")

    if "quality_score" in transcripts.columns and qv_threshold is not None:
        transcripts = transcripts[transcripts["quality_score"].fillna(-np.inf) >= qv_threshold].copy()
    elif qv_threshold is not None:
        warnings.warn(
            "[protein_gene_correlation] quality_score unavailable; proceeding without quality filtering."
        )
    if "valid" in transcripts.columns:
        transcripts = transcripts[transcripts["valid"].fillna(False).astype(bool)].copy()

    region_min_x = float(transcripts["x"].min())
    region_min_y = float(transcripts["y"].min())
    region_max_x = float(transcripts["x"].max())
    region_max_y = float(transcripts["y"].max())

    if mask is not None:
        ox, oy = mask_origin
        region_min_x = min(region_min_x, ox)
        region_min_y = min(region_min_y, oy)
        try:
            m_h, m_w = mask.shape[:2]
            region_max_x = max(region_max_x, ox + m_w * float(pixel_size_um))
            region_max_y = max(region_max_y, oy + m_h * float(pixel_size_um))
        except Exception:
            pass

    tx_region_width = region_max_x - region_min_x if region_max_x > region_min_x else 0.0
    tx_region_height = region_max_y - region_min_y if region_max_y > region_min_y else 0.0

    cell_xy_scaled, _ = _maybe_convert_cell_coords_auto(
        cell_xy_in_um_or_px=cell_xy,
        pixel_size=pixel_size_um,
        tx_width=tx_region_width,
        tx_height=tx_region_height,
    )
    if cell_xy_scaled.size > 0:
        region_min_x = min(region_min_x, float(cell_xy_scaled[:, 0].min()))
        region_max_x = max(region_max_x, float(cell_xy_scaled[:, 0].max()))
        region_min_y = min(region_min_y, float(cell_xy_scaled[:, 1].min()))
        region_max_y = max(region_max_y, float(cell_xy_scaled[:, 1].max()))

    if region_max_x <= region_min_x or region_max_y <= region_min_y:
        raise ValueError("Invalid region bounds for spatial data.")

    region_width = region_max_x - region_min_x
    region_height = region_max_y - region_min_y
    if grid_counts is not None:
        ny, nx = int(grid_counts[0]), int(grid_counts[1])
        if ny <= 0 or nx <= 0:
            raise ValueError("`grid_counts` must contain positive integers.")
        x_bin_size = region_width / nx
        y_bin_size = region_height / ny
    else:
        if grid_size is None:
            raise ValueError("Provide either `grid_counts` or `grid_size`.")
        gy_um, gx_um = float(grid_size[0]), float(grid_size[1])
        if gy_um <= 0 or gx_um <= 0:
            raise ValueError("`grid_size` values must be positive.")
        ny = max(int(np.ceil(region_height / gy_um)), 1)
        nx = max(int(np.ceil(region_width / gx_um)), 1)
        x_bin_size = region_width / nx
        y_bin_size = region_height / ny

    if mask is not None and not transcripts.empty:
        rel_x = transcripts["x"].to_numpy() - mask_origin[0]
        rel_y = transcripts["y"].to_numpy() - mask_origin[1]
        col_idx = np.floor(rel_x / float(pixel_size_um)).astype(int)
        row_idx = np.floor(rel_y / float(pixel_size_um)).astype(int)
        inside = (
            (row_idx >= 0)
            & (row_idx < mask.shape[0])
            & (col_idx >= 0)
            & (col_idx < mask.shape[1])
        )
        mask_values = np.zeros(len(transcripts), dtype=bool)
        valid_rows = np.where(inside)[0]
        if len(valid_rows):
            mask_values[valid_rows] = mask[row_idx[valid_rows], col_idx[valid_rows]]
        transcripts = transcripts[inside & mask_values].copy()

    transcripts_count = {gene: np.zeros((ny, nx), dtype=float) for gene in unique_genes}
    if not transcripts.empty:
        xi = np.floor((transcripts["x"].to_numpy() - region_min_x) / x_bin_size).astype(int)
        yi = np.floor((transcripts["y"].to_numpy() - region_min_y) / y_bin_size).astype(int)
        xi = np.clip(xi, 0, nx - 1)
        yi = np.clip(yi, 0, ny - 1)
        for gene in unique_genes:
            gene_mask = transcripts["gene_name"].to_numpy() == gene
            if np.any(gene_mask):
                np.add.at(transcripts_count[gene], (yi[gene_mask], xi[gene_mask]), 1)

    if cell_xy_scaled.shape[0] > 0:
        cell_x_idx = np.floor((cell_xy_scaled[:, 0] - region_min_x) / x_bin_size).astype(int)
        cell_y_idx = np.floor((cell_xy_scaled[:, 1] - region_min_y) / y_bin_size).astype(int)
        cell_x_idx = np.clip(cell_x_idx, 0, nx - 1)
        cell_y_idx = np.clip(cell_y_idx, 0, ny - 1)
    else:
        cell_x_idx = np.array([], dtype=int)
        cell_y_idx = np.array([], dtype=int)

    n_cells_total = cell_xy_scaled.shape[0]
    valid_cell_mask = np.ones(n_cells_total, dtype=bool)
    if mask is not None and n_cells_total > 0:
        rel_x = cell_xy_scaled[:, 0] - mask_origin[0]
        rel_y = cell_xy_scaled[:, 1] - mask_origin[1]
        col_idx = np.floor(rel_x / float(pixel_size_um)).astype(int)
        row_idx = np.floor(rel_y / float(pixel_size_um)).astype(int)
        inside = (
            (row_idx >= 0)
            & (row_idx < mask.shape[0])
            & (col_idx >= 0)
            & (col_idx < mask.shape[1])
        )
        values = np.zeros(n_cells_total, dtype=bool)
        valid_rows = np.where(inside)[0]
        if len(valid_rows):
            values[valid_rows] = mask[row_idx[valid_rows], col_idx[valid_rows]]
        valid_cell_mask &= inside & values

    if n_cells_total > 0 and not valid_cell_mask.all():
        cell_x_idx = cell_x_idx[valid_cell_mask]
        cell_y_idx = cell_y_idx[valid_cell_mask]

    protein_sums = {protein: np.zeros((ny, nx), dtype=float) for protein in unique_proteins}
    cell_count = np.zeros((ny, nx), dtype=float)
    if cell_x_idx.size > 0:
        np.add.at(cell_count, (cell_y_idx, cell_x_idx), 1)
        for protein in unique_proteins:
            intensities_all = np.asarray(get_protein_vector(protein), dtype=float)
            intensities = intensities_all[valid_cell_mask] if intensities_all.size == n_cells_total else intensities_all
            if intensities.size != cell_x_idx.size:
                limit = min(intensities.size, cell_x_idx.size)
                np.add.at(
                    protein_sums[protein],
                    (cell_y_idx[:limit], cell_x_idx[:limit]),
                    intensities[:limit],
                )
            else:
                np.add.at(protein_sums[protein], (cell_y_idx, cell_x_idx), intensities)

    protein_avg = {}
    nonzero = cell_count > 0
    for protein in unique_proteins:
        avg = np.zeros((ny, nx), dtype=float)
        if np.any(nonzero):
            avg[nonzero] = protein_sums[protein][nonzero] / cell_count[nonzero]
        protein_avg[protein] = avg

    summary_records = []
    skip_pairs = []
    old_summary_df = None
    if not overwrite:
        try:
            with fs_out.open(f"{out_dir_path}/protein_gene_correlation_summary.csv", "r") as stream:
                old_summary_df = pd.read_csv(stream)
        except FileNotFoundError:
            old_summary_df = None

    for protein_name, gene_name in pairs:
        safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
        safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
        pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
        pair_png_name = f"{safe_prot}_{safe_gene}_scatter.png"

        if not overwrite:
            file_exists_csv = bool(fs_out.exists(f"{out_dir_path}/{pair_csv_name}"))
            file_exists_png = bool(fs_out.exists(f"{out_dir_path}/{pair_png_name}"))
            if file_exists_csv and file_exists_png:
                skip_pairs.append((protein_name, gene_name))
                if old_summary_df is not None:
                    previous = old_summary_df[
                        (old_summary_df["Protein"] == protein_name)
                        & (old_summary_df["Gene"] == gene_name)
                    ]
                    if len(previous) > 0:
                        summary_records.append(
                            {
                                "Protein": protein_name,
                                "Gene": gene_name,
                                "Pearson_r": float(previous["Pearson_r"].iloc[0]),
                                "p_value": float(previous["p_value"].iloc[0]),
                            }
                        )
                continue

        gene_counts_mat = transcripts_count[gene_name]
        bin_area = x_bin_size * y_bin_size
        transcripts_density = gene_counts_mat.copy() if bin_area <= 0 else gene_counts_mat / bin_area
        protein_avg_mat = protein_avg.get(protein_name, np.zeros((ny, nx), dtype=float))

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

        summary_records.append(
            {
                "Protein": protein_name,
                "Gene": gene_name,
                "Pearson_r": pearson_r,
                "p_value": p_val,
            }
        )

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(gene_vals, prot_vals, s=30, alpha=0.7, edgecolors="none")
            ax.set_xlabel(f"{gene_name} transcript density (count/μm²)")
            ax.set_ylabel(f"{protein_name} mean intensity")
            title = f"{protein_name} vs {gene_name}"
            if not np.isnan(pearson_r):
                title += f"\nPearson r = {pearson_r:.3f}, p = {p_val:.3g}"
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            with fs_out.open(f"{out_dir_path}/{pair_png_name}", "wb") as stream:
                fig.savefig(stream, format="png", dpi=300)
        finally:
            plt.close(fig)

        y_idx, x_idx = np.indices((ny, nx))
        records = []
        for yi, xi in zip(y_idx.ravel(), x_idx.ravel()):
            records.append(
                {
                    "bin_y": int(yi),
                    "bin_x": int(xi),
                    "n_cells": int(cell_count[yi, xi]),
                    "transcript_count": int(transcripts_count[gene_name][yi, xi]),
                    "transcript_density": transcripts_density[yi, xi],
                    "protein_avg_intensity": protein_avg_mat[yi, xi],
                }
            )
        pair_df = pd.DataFrame(
            records,
            columns=[
                "bin_y",
                "bin_x",
                "n_cells",
                "transcript_count",
                "transcript_density",
                "protein_avg_intensity",
            ],
        )
        with fs_out.open(f"{out_dir_path}/{pair_csv_name}", "w") as stream:
            pair_df.to_csv(stream, index=False)

    if skip_pairs and old_summary_df is None:
        for protein_name, gene_name in skip_pairs:
            safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
            safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
            pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
            try:
                with fs_out.open(f"{out_dir_path}/{pair_csv_name}", "r") as stream:
                    pair_df = pd.read_csv(stream)
            except FileNotFoundError:
                pair_df = pd.DataFrame()

            if pair_df.empty:
                summary_records.append(
                    {
                        "Protein": protein_name,
                        "Gene": gene_name,
                        "Pearson_r": np.nan,
                        "p_value": np.nan,
                    }
                )
                continue

            valid_df = pair_df[pair_df["n_cells"] > 0]
            if valid_df.empty:
                r_val = np.nan
                p_val = np.nan
            else:
                x = valid_df["transcript_density"].to_numpy()
                y = valid_df["protein_avg_intensity"].to_numpy()
                if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
                    r_val = np.nan
                    p_val = np.nan
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        r_val, p_val = pearsonr(x, y)
            summary_records.append(
                {
                    "Protein": protein_name,
                    "Gene": gene_name,
                    "Pearson_r": r_val,
                    "p_value": p_val,
                }
            )

    summary_df = pd.DataFrame(columns=["Protein", "Gene", "Pearson_r", "p_value"])
    for protein_name, gene_name in pairs:
        record = next(
            (
                item
                for item in summary_records
                if item["Protein"] == protein_name and item["Gene"] == gene_name
            ),
            None,
        )
        if record is None:
            record = {
                "Protein": protein_name,
                "Gene": gene_name,
                "Pearson_r": np.nan,
                "p_value": np.nan,
            }
        summary_df = pd.concat([summary_df, pd.DataFrame([record])], ignore_index=True)

    with fs_out.open(f"{out_dir_path}/protein_gene_correlation_summary.csv", "w") as stream:
        summary_df.to_csv(stream, index=False)

    return summary_df
