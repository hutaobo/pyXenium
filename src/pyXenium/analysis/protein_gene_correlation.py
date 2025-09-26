import numpy as np
import pandas as pd
import fsspec
import zarr
import warnings
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def protein_gene_correlation(
    adata,
    transcripts_zarr_path,
    pairs,
    output_dir,
    grid_size=(50, 50),
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
        Optionally, `adata.uns['protein_qc_mask']` can be provided (as a 2D array or image path) to mask out regions with low protein data quality.
    - `transcripts_zarr_path` : str
        Path (local or remote) to the Xenium `transcripts.zarr` (or `transcripts.zarr.zip`) file containing transcript locations.
        This path is opened with fsspec, so it can be a local file system path or a remote URL (e.g., S3 URI).
    - `pairs` : list of tuple(str, str)
        List of (protein_name, gene_name) pairs to analyze.
        The `protein_name` should match one of the protein markers in `adata.obsm['protein']` (or be a substring without the " Protein Expression" suffix),
        and `gene_name` should be a gene present in the transcripts data.
    - `output_dir` : str
        Directory path (local or remote) where output files will be saved.
        Each pair will produce a scatter plot PNG and a CSV of binned data, and a summary CSV of all correlations will be saved in this directory.
    - `grid_size` : tuple(int, int), default=(50, 50)
        Number of grid divisions in the y (rows) and x (columns) dimensions. The tissue region is divided into `grid_size[0]` by `grid_size[1]` spatial bins.
    - `pixel_size_um` : float, default=0.2125
        Microns per pixel for the spatial coordinates. Xenium images typically have a resolution of 0.2125 µm/pixel.
        If the coordinates in `adata.obsm['spatial']` are in pixel units, they will be multiplied by this factor to convert to microns (physical space).
        If the coordinates are already in microns, set `pixel_size_um=1.0` to leave them unchanged.
    - `qv_threshold` : int, default=20
        Transcript quality score cutoff. Only transcripts with quality value >= `qv_threshold` are counted.
        (Xenium default Q-score threshold is 20, corresponding to Q20.)
    - `overwrite` : bool, default=False
        Whether to overwrite existing output files. If `False`, the function will skip re-computation for any (protein, gene) pair whose output files
        (both scatter PNG and CSV) already exist in `output_dir`, and will reuse the previous results for the summary.
        If `True`, all pairs are processed and any existing files will be replaced.

    **Returns:**
    - `summary_df` : pandas.DataFrame
        A DataFrame summarizing the Pearson correlation results for each pair.
        It has columns: `Protein`, `Gene`, `Pearson_r`, `p_value`.

    **Outputs:**
    - For each (protein, gene) pair:
        - A CSV file named `"<Protein>_<Gene>_correlation.csv"` containing the binned data for that pair.
          This CSV includes columns: `bin_y`, `bin_x` (grid indices), `n_cells` (number of cells in the bin),
          `transcript_count` (number of transcripts of the gene in the bin),
          `transcript_density` (transcripts per µm² in the bin), and `protein_avg_intensity` (mean protein intensity in the bin).
        - A scatter plot image `"<Protein>_<Gene>_scatter.png"` visualizing the correlation.
          Each point represents one spatial bin, plotting transcript density (x-axis) vs. mean protein intensity (y-axis).
          The plot title and/or caption includes the Pearson r and p-value.
    - A summary CSV file `"protein_gene_correlation_summary.csv"` in `output_dir` listing the correlation results for all pairs analyzed in this function call.

    **Notes:**
    - This function uses **fsspec** for file I/O, so `transcripts_zarr_path` and `output_dir` can refer to remote locations (e.g., S3 buckets) as well as local paths.
    - It processes the transcripts data in chunks (as stored in the Zarr) to avoid loading everything into memory at once.
      Similarly, if a protein QC mask image is very large, it attempts to use dask or partial loading for memory efficiency.
    - If `adata.uns['protein_qc_mask']` is provided (as a NumPy array, a dask array, or a file path to an image),
      the analysis will be restricted to regions where this mask is True. Cells and transcripts outside the mask are ignored in the calculations.
    - Ensure that the `protein_name` in each pair matches the naming in `adata.obsm['protein']`.
      For example, if the AnnData was loaded from a 10x Genomics cell feature matrix, protein features might be named like "PD-1" or "PD-1 Protein Expression".
      This function will try to match the provided name to the data (it will ignore the " Protein Expression" suffix if necessary).
    - The AnnData's `adata.obsm['spatial']` should contain spatial coordinates for each cell (e.g., cell centroids). If these are in pixel units, use the appropriate `pixel_size_um` to convert to microns.
    - The returned DataFrame `summary_df` can be used directly for further analysis. It is also written to the summary CSV file for convenience.

    **Usage Example:**

    ```python
    # Suppose adata is an AnnData object with Xenium multi-modal data already loaded.
    pairs_to_analyze = [("CD3E", "CD3E"), ("PD-L1", "CD274")]  # (Protein, Gene) pairs
    result_df = protein_gene_correlation(
        adata,
        "path/to/transcripts.zarr.zip",
        pairs=pairs_to_analyze,
        output_dir="results/correlation_analysis",
        grid_size=(50, 50),
        pixel_size_um=0.2125,
        qv_threshold=20,
        overwrite=False
    )
    print(result_df)
    # This will print a DataFrame with Pearson correlation and p-values for the specified pairs.
    # Output files (CSV and PNG for each pair, and a summary CSV) will be saved in the specified output directory.
    ```
    """
    # Ensure output directory path ends with a separator for consistent path joining
    output_dir = output_dir.rstrip('/')

    # Prepare output filesystem and directory
    fs_out, out_dir_path = fsspec.core.url_to_fs(output_dir + '/')
    try:
        fs_out.makedirs(out_dir_path, exist_ok=True)
    except Exception:
        # Some filesystems (like object stores) may not need an explicit directory creation
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
    protein_names = None
    if hasattr(protein_data, "columns"):
        # If .obsm['protein'] is a DataFrame, get its column names
        protein_names = list(protein_data.columns)
    else:
        # Try to retrieve protein names from adata.uns or adata.var (if integrated)
        if isinstance(adata.uns, dict):
            if 'protein_names' in adata.uns:
                protein_names = list(adata.uns['protein_names'])
            elif 'protein_markers' in adata.uns:
                protein_names = list(adata.uns['protein_markers'])
            elif 'protein_channels' in adata.uns:
                protein_names = list(adata.uns['protein_channels'])
        if protein_names is None and 'feature_types' in adata.var:
            # If gene and protein features are combined in var with a feature_types column (e.g., Seurat conversions)
            prot_mask = (adata.var['feature_types'] == 'Protein') if 'Protein' in adata.var['feature_types'].values else (adata.var['feature_types'] == 'Antibody')
            if prot_mask.any():
                protein_names = list(adata.var_names[prot_mask])
        if protein_names is None:
            # If still not found, try to infer from obsm shape (not ideal because we need names)
            protein_names = None  # leave as None, we'll handle missing name in lookup function

    def get_protein_vector(name):
        """Retrieve the per-cell intensity vector for the given protein name."""
        if hasattr(protein_data, "columns"):
            # DataFrame case
            if name in protein_data.columns:
                return protein_data[name].values
            else:
                # Try matching with suffix or partial name
                for col in protein_data.columns:
                    if col == name or col.startswith(name + " "):
                        return protein_data[col].values
            raise KeyError(f"Protein '{name}' not found in adata.obsm['protein'] columns.")
        else:
            if protein_names is None:
                raise KeyError(f"Protein '{name}' not found (no protein names available).")
            # Find index in known protein names
            try:
                idx = protein_names.index(name)
            except ValueError:
                # Try matching if the stored names have " Protein Expression" or other suffix
                matches = [i for i, pname in enumerate(protein_names) if pname == name or pname.startswith(name + " ")]
                if matches:
                    idx = matches[0]
                else:
                    raise KeyError(f"Protein '{name}' not found in protein data.")
            return np.asarray(protein_data)[:, idx]

    # Handle optional protein QC mask
    mask = None
    mask_origin = (0.0, 0.0)
    if 'protein_qc_mask' in adata.uns:
        mask_data = adata.uns['protein_qc_mask']
        # Load mask if it's a path or an array
        if isinstance(mask_data, str):
            # Open image via PIL or imageio
            try:
                from PIL import Image
                mask_img = Image.open(mask_data)
                mask = np.array(mask_img)
            except ImportError:
                import imageio
                mask = imageio.imread(mask_data)
        elif hasattr(mask_data, "compute"):
            # If it's a dask array
            mask = mask_data.compute()
        else:
            mask = np.array(mask_data)
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask != 0
        # If mask origin (offset in physical coordinates) is provided in adata.uns, use it
        if 'mask_origin' in adata.uns:
            mask_origin = tuple(adata.uns['mask_origin'])
        elif 'origin' in adata.uns:
            # Some data might store origin in uns (though typically in transcripts file attributes)
            try:
                ox, oy = adata.uns['origin']
                mask_origin = (float(ox), float(oy))
            except Exception:
                pass

    # Open transcripts Zarr (allow both .zarr directory or .zarr.zip)
    if str(transcripts_zarr_path).endswith((".zarr.zip", ".zip")):
        store = zarr.ZipStore(transcripts_zarr_path, mode='r')
    else:
        store = fsspec.get_mapper(transcripts_zarr_path)
    root = zarr.open(store, mode='r')

    # Determine gene index mapping for transcripts
    gene_index_map = {}
    gene_names_list = None
    if 'gene_names' in root.attrs:
        # gene_names might be stored as a list/array in attributes
        try:
            gene_names_list = list(root.attrs['gene_names'])
        except Exception:
            # Zarr might store as array of bytes, ensure conversion to str
            gene_names_list = [str(name) for name in root.attrs['gene_names']]
    if gene_names_list is None:
        raise KeyError("Gene names not found in transcripts.zarr attributes.")
    for gene in unique_genes:
        if gene not in gene_names_list:
            raise KeyError(f"Gene '{gene}' not found in transcripts data.")
        gene_index_map[gene] = gene_names_list.index(gene)

    # Determine spatial bounds of the region (min and max coordinates)
    # We will use both transcripts and cell coordinates to ensure the full extent is covered.
    region_min_x = float('inf')
    region_min_y = float('inf')
    region_max_x = float('-inf')
    region_max_y = float('-inf')
    # If transcripts file provides an origin attribute (min coords) and perhaps grid size:
    if 'origin' in root.attrs:
        orig_attr = root.attrs['origin']
        if isinstance(orig_attr, dict):
            region_min_x = float(orig_attr.get('x', region_min_x))
            region_min_y = float(orig_attr.get('y', region_min_y))
    # If mask origin is given and differs (in case mask covers a superset), possibly update region_min
    if mask is not None:
        ox, oy = mask_origin
        region_min_x = min(region_min_x, ox)
        region_min_y = min(region_min_y, oy)
        # If we know mask dimensions, we can estimate max from it as well
        try:
            # mask shape: (rows, cols), each pixel size in micron = pixel_size_um
            m_h, m_w = mask.shape[:2]
            region_max_x = max(region_max_x, mask_origin[0] + m_w * float(pixel_size_um))
            region_max_y = max(region_max_y, mask_origin[1] + m_h * float(pixel_size_um))
        except Exception:
            pass

    # Traverse transcript chunks to update bounds (and later count transcripts)
    level0 = root['grids']['0'] if 'grids' in root and '0' in root['grids'] else root
    chunk_keys = list(level0.keys())
    # Only consider keys that look like chunk coordinates (e.g., "0,0"); filter out any non-chunk arrays like 'gene_names' if present
    chunk_keys = [k for k in chunk_keys if ',' in k]
    for ck in chunk_keys:
        loc = level0[ck]['location']
        if loc.shape[0] == 0:
            continue  # no transcripts in this chunk
        coords_chunk = loc[...]
        if coords_chunk.size == 0:
            continue
        x_vals = coords_chunk[:, 0]
        y_vals = coords_chunk[:, 1]
        # Update min/max from this chunk
        region_min_x = min(region_min_x, float(x_vals.min()))
        region_max_x = max(region_max_x, float(x_vals.max()))
        region_min_y = min(region_min_y, float(y_vals.min()))
        region_max_y = max(region_max_y, float(y_vals.max()))
    # Also incorporate cell coordinates into region bounds
    if cell_xy.size > 0:
        region_min_x = min(region_min_x, float(cell_xy[:, 0].min()))
        region_max_x = max(region_max_x, float(cell_xy[:, 0].max()))
        region_min_y = min(region_min_y, float(cell_xy[:, 1].min()))
        region_max_y = max(region_max_y, float(cell_xy[:, 1].max()))

    # Define grid edges and bin sizes
    ny, nx = grid_size
    # If region extent is zero (e.g., only one cell?), avoid zero division
    if region_max_x <= region_min_x or region_max_y <= region_min_y:
        raise ValueError("Invalid region bounds for spatial data. Please check the coordinates in adata and transcripts.zarr.")
    region_width = region_max_x - region_min_x
    region_height = region_max_y - region_min_y
    x_bin_size = region_width / nx
    y_bin_size = region_height / ny

    # Prepare arrays for transcript counts per bin for each gene
    transcripts_count = {gene: np.zeros((ny, nx), dtype=float) for gene in unique_genes}

    # Process transcripts in chunks to fill transcript counts
    for ck in chunk_keys:
        group = level0[ck]
        # Load necessary arrays from this chunk
        coords = group['location'][...]
        if coords.shape[0] == 0:
            continue
        gene_ids = group['gene_identity'][...]
        # Flatten gene_ids if it's two-dimensional (it should be 1D per transcript)
        gene_ids = gene_ids.flatten()
        # Quality and valid filters
        if 'quality_score' in group:
            qvs = group['quality_score'][...]
        else:
            qvs = None
        if 'valid' in group:
            valid_flags = group['valid'][...]
        else:
            valid_flags = None
        mask_all = np.ones(coords.shape[0], dtype=bool)
        if qvs is not None:
            mask_all &= (qvs >= qv_threshold)
        if valid_flags is not None:
            # According to Xenium documentation, final output transcripts have valid==1
            mask_all &= (valid_flags == 1)
        # If a protein QC mask is provided, filter transcripts to those inside the mask region
        if mask is not None:
            # Convert transcript coordinates (micron) to mask pixel indices
            # Subtract mask origin to get coordinate relative to mask image
            rel_x = coords[:, 0] - mask_origin[0]
            rel_y = coords[:, 1] - mask_origin[1]
            col_idx = np.floor(rel_x / float(pixel_size_um)).astype(int)
            row_idx = np.floor(rel_y / float(pixel_size_um)).astype(int)
            valid_idx = (row_idx >= 0) & (row_idx < mask.shape[0]) & (col_idx >= 0) & (col_idx < mask.shape[1])
            mask_all &= valid_idx
            if mask_all.any():
                # Only check mask values for those indices within bounds
                inside_idx = np.nonzero(mask_all)[0]
                if inside_idx.size > 0:
                    rr = row_idx[inside_idx]
                    cc = col_idx[inside_idx]
                    mask_vals = mask[rr, cc]
                    inside_mask = np.zeros_like(mask_all)
                    inside_mask[inside_idx] = mask_vals
                    mask_all &= inside_mask.astype(bool)
            else:
                # no transcript from this chunk passes inside mask
                continue
        # Apply combined filter
        if not mask_all.any():
            continue
        coords_filt = coords[mask_all]
        gene_ids_filt = gene_ids[mask_all]
        # For each gene of interest in this chunk, accumulate counts
        for gene, gene_idx in gene_index_map.items():
            # Select transcripts of this gene
            sel = (gene_ids_filt == gene_idx)
            if not np.any(sel):
                continue
            xs = coords_filt[sel, 0]
            ys = coords_filt[sel, 1]
            # Compute bin indices for each transcript
            # Note: using floor division of continuous coordinates by bin size
            xi = np.floor((xs - region_min_x) / x_bin_size).astype(int)
            yi = np.floor((ys - region_min_y) / y_bin_size).astype(int)
            # Clamp indices at boundaries (e.g., if a point falls exactly on max, floor can give index == nx)
            xi[xi < 0] = 0
            yi[yi < 0] = 0
            xi[xi >= nx] = nx - 1
            yi[yi >= ny] = ny - 1
            # Accumulate counts into the corresponding bins
            np.add.at(transcripts_count[gene], (yi, xi), 1)

    # Close the transcripts store if applicable to free resources
    try:
        store.close()
    except Exception:
        pass

    # Compute cell bin indices once for all cells (for protein and cell count accumulation)
    if cell_xy.shape[0] > 0:
        # Compute each cell's bin
        cell_x = cell_xy[:, 0]
        cell_y = cell_xy[:, 1]
        cell_x_idx = np.floor((cell_x - region_min_x) / x_bin_size).astype(int)
        cell_y_idx = np.floor((cell_y - region_min_y) / y_bin_size).astype(int)
        # Clamp indices to valid range
        cell_x_idx[cell_x_idx < 0] = 0
        cell_x_idx[cell_x_idx >= nx] = nx - 1
        cell_y_idx[cell_y_idx < 0] = 0
        cell_y_idx[cell_y_idx >= ny] = ny - 1
    else:
        # No cells (should not usually happen unless data is empty)
        cell_x_idx = np.array([], dtype=int)
        cell_y_idx = np.array([], dtype=int)

    # If mask is provided, also filter out cells outside mask (e.g., cells segmented outside protein-covered area)
    if mask is not None and cell_xy.shape[0] > 0:
        rel_cx = cell_xy[:, 0] - mask_origin[0]
        rel_cy = cell_xy[:, 1] - mask_origin[1]
        c_col = np.floor(rel_cx / float(pixel_size_um)).astype(int)
        c_row = np.floor(rel_cy / float(pixel_size_um)).astype(int)
        inside_mask_cells = (c_row >= 0) & (c_row < mask.shape[0]) & (c_col >= 0) & (c_col < mask.shape[1])
        if inside_mask_cells.any():
            mask_vals = np.zeros(cell_xy.shape[0], dtype=bool)
            valid_cells_idx = np.where(inside_mask_cells)[0]
            if valid_cells_idx.size > 0:
                rr = c_row[valid_cells_idx]
                cc = c_col[valid_cells_idx]
                mask_vals_seg = mask[rr, cc]
                mask_vals[valid_cells_idx] = mask_vals_seg
            inside_mask_cells &= mask_vals
        # Filter out cells not in mask
        if not inside_mask_cells.all():
            if not inside_mask_cells.any():
                cell_x_idx = np.array([], dtype=int)
                cell_y_idx = np.array([], dtype=int)
            else:
                cell_x_idx = cell_x_idx[inside_mask_cells]
                cell_y_idx = cell_y_idx[inside_mask_cells]

    # Prepare arrays for protein intensity sums and cell counts per bin
    protein_sums = {prot: np.zeros((ny, nx), dtype=float) for prot in unique_proteins}
    cell_count = np.zeros((ny, nx), dtype=float)
    # Accumulate cell counts
    if cell_x_idx.size > 0:
        np.add.at(cell_count, (cell_y_idx, cell_x_idx), 1)
        # Accumulate protein intensity sums for each protein
        for prot in unique_proteins:
            intensities = get_protein_vector(prot)
            # If we filtered cells by mask above, we need to apply same filter to intensities
            if mask is not None and cell_xy.shape[0] > 0 and len(intensities) == cell_xy.shape[0]:
                # inside_mask_cells from above context is out of scope here, so recompute or store earlier.
                # We recompute mask filter for cells:
                if cell_xy.shape[0] > 0:
                    rel_cx = cell_xy[:, 0] - mask_origin[0]
                    rel_cy = cell_xy[:, 1] - mask_origin[1]
                    c_col = np.floor(rel_cx / float(pixel_size_um)).astype(int)
                    c_row = np.floor(rel_cy / float(pixel_size_um)).astype(int)
                    inside_mask_cells = (c_row >= 0) & (c_row < mask.shape[0]) & (c_col >= 0) & (c_col < mask.shape[1])
                    if inside_mask_cells.any():
                        mask_vals = np.zeros(cell_xy.shape[0], dtype=bool)
                        valid_cells_idx = np.where(inside_mask_cells)[0]
                        if valid_cells_idx.size > 0:
                            rr = c_row[valid_cells_idx]
                            cc = c_col[valid_cells_idx]
                            mask_vals_seg = mask[rr, cc]
                            mask_vals[valid_cells_idx] = mask_vals_seg
                        inside_mask_cells &= mask_vals
                    if not inside_mask_cells.all():
                        intensities = intensities[inside_mask_cells]
                        # Note: cell_x_idx, cell_y_idx were also filtered above accordingly, so lengths should match now.
            np.add.at(protein_sums[prot], (cell_y_idx, cell_x_idx), intensities.astype(float))

    # Compute average protein intensity per bin for each protein
    protein_avg = {}
    for prot in unique_proteins:
        avg = np.zeros((ny, nx), dtype=float)
        # Avoid division by zero
        nonzero = cell_count > 0
        if np.any(nonzero):
            avg[nonzero] = protein_sums[prot][nonzero] / cell_count[nonzero]
        protein_avg[prot] = avg

    # Prepare to generate outputs
    summary_records = []
    # We will check existing output files if not overwrite
    skip_pairs = []
    # If not overwrite, load existing summary if present (to retrieve previous results for skipped pairs)
    old_summary_df = None
    summary_path = f"{output_dir}/protein_gene_correlation_summary.csv"
    if not overwrite:
        try:
            with fs_out.open(f"{out_dir_path}protein_gene_correlation_summary.csv", 'r') as f:
                old_summary_df = pd.read_csv(f)
        except FileNotFoundError:
            old_summary_df = None

    # Iterate over each requested pair in the original order
    for protein_name, gene_name in pairs:
        # Construct file paths for outputs
        safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
        safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
        pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
        pair_png_name = f"{safe_prot}_{safe_gene}_scatter.png"
        pair_csv_path = f"{output_dir}/{pair_csv_name}"
        pair_png_path = f"{output_dir}/{pair_png_name}"

        # Check if outputs exist and skip if not overwrite
        if not overwrite:
            try:
                file_exists_csv = fs_out.exists(f"{out_dir_path}{pair_csv_name}")
            except Exception:
                # If fs doesn't support exists, attempt open
                try:
                    ftest = fs_out.open(f"{out_dir_path}{pair_csv_name}", 'rb')
                    ftest.close()
                    file_exists_csv = True
                except Exception:
                    file_exists_csv = False
            try:
                file_exists_png = fs_out.exists(f"{out_dir_path}{pair_png_name}")
            except Exception:
                try:
                    ftest = fs_out.open(f"{out_dir_path}{pair_png_name}", 'rb')
                    ftest.close()
                    file_exists_png = True
                except Exception:
                    file_exists_png = False
            if file_exists_csv and file_exists_png:
                # If both outputs already exist, skip recomputation
                skip_pairs.append((protein_name, gene_name))
                # If we have old summary, try to get previous results for this pair
                if old_summary_df is not None:
                    prev = old_summary_df[(old_summary_df["Protein"] == protein_name) & (old_summary_df["Gene"] == gene_name)]
                    if len(prev) > 0:
                        prev_r = float(prev["Pearson_r"].iloc[0])
                        prev_p = float(prev["p_value"].iloc[0])
                        summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": prev_r, "p_value": prev_p})
                # If no summary info is available, we'll compute correlation by reading the CSV later
                continue

        # Compute correlation for this pair using the binned data
        # Get the transcripts count matrix for the gene and convert to density
        if gene_name not in transcripts_count:
            # If gene was not requested originally, transcripts_count may not have it
            raise KeyError(f"Gene '{gene_name}' not processed in transcripts (not in unique_genes set).")
        gene_counts_mat = transcripts_count[gene_name]
        # Calculate transcript density (transcripts per square micron)
        bin_area = x_bin_size * y_bin_size  # area of one grid bin in square microns
        # If bin_area is zero (shouldn't happen unless numeric issues), skip density scaling
        if bin_area <= 0:
            transcripts_density = gene_counts_mat.copy()
        else:
            transcripts_density = gene_counts_mat / bin_area

        # Get protein average intensity matrix for this protein
        if protein_name not in protein_avg:
            # If protein was not precomputed (should not happen, as we did for unique_proteins)
            intensities = get_protein_vector(protein_name)
            protein_sums_temp = np.zeros((ny, nx), dtype=float)
            # If we have filtered cell indices above, reuse them
            if cell_x_idx.size > 0 and intensities.size == cell_x_idx.size:
                np.add.at(protein_sums_temp, (cell_y_idx, cell_x_idx), intensities.astype(float))
            protein_avg_mat = np.zeros((ny, nx), dtype=float)
            nonzero_mask = cell_count > 0
            if np.any(nonzero_mask):
                protein_avg_mat[nonzero_mask] = protein_sums_temp[nonzero_mask] / cell_count[nonzero_mask]
        else:
            protein_avg_mat = protein_avg[protein_name]

        # Consider only bins that have at least one cell for correlation (protein intensity defined)
        valid_bins = cell_count > 0
        if not np.any(valid_bins):
            # If no bin has cells, skip correlation
            pearson_r = np.nan
            p_val = np.nan
        else:
            gene_vals = transcripts_density[valid_bins]
            prot_vals = protein_avg_mat[valid_bins]
            # If either variable has no variation, Pearson correlation is undefined
            if gene_vals.size < 2 or np.all(gene_vals == gene_vals[0]) or np.all(prot_vals == prot_vals[0]):
                pearson_r = np.nan
                p_val = np.nan
            else:
                # Compute Pearson correlation and p-value
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore constant input warning if any
                    pearson_r, p_val = pearsonr(gene_vals, prot_vals)
        # Record summary result
        summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": pearson_r, "p_value": p_val})

        # Save scatter plot
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(gene_vals, prot_vals, s=30, alpha=0.7, edgecolors='none')
            ax.set_xlabel(f"{gene_name} transcript density (count/μm²)")
            ax.set_ylabel(f"{protein_name} mean intensity")
            title_str = f"{protein_name} vs {gene_name}"
            # Annotate correlation in title or as text
            if not np.isnan(pearson_r):
                title_str += f"\nPearson r = {pearson_r:.3f}, p = {p_val:.3g}"
            ax.set_title(title_str)
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            # Write image to file (using fsspec)
            with fs_out.open(f"{out_dir_path}{pair_png_name}", 'wb') as f:
                fig.savefig(f, format='png', dpi=300)
        finally:
            plt.close(fig)

        # Save pair CSV with binned data
        # Prepare DataFrame of binned values (only include bins with at least one cell to avoid overwhelming zeros outside tissue)
        y_idx, x_idx = np.nonzero(cell_count)  # indices of bins that contain cells
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
        # Save to CSV
        with fs_out.open(f"{out_dir_path}{pair_csv_name}", 'w') as f:
            pair_df.to_csv(f, index=False)

    # If any pairs were skipped and not added to summary_records (due to missing old summary data), handle them:
    if skip_pairs:
        # If old_summary_df is available, we already added those to summary_records above.
        # If not, we need to read each skipped pair's CSV to compute correlation now.
        if old_summary_df is None:
            for protein_name, gene_name in skip_pairs:
                safe_prot = protein_name.replace(os.sep, "_").replace(" ", "_")
                safe_gene = gene_name.replace(os.sep, "_").replace(" ", "_")
                pair_csv_name = f"{safe_prot}_{safe_gene}_correlation.csv"
                try:
                    with fs_out.open(f"{out_dir_path}{pair_csv_name}", 'r') as f:
                        df = pd.read_csv(f)
                        # We assume the CSV was produced by this function in a previous run and contains needed columns
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
                    # If file is somehow missing, skip
                    summary_records.append({"Protein": protein_name, "Gene": gene_name, "Pearson_r": np.nan, "p_value": np.nan})

    # Create summary DataFrame in the original input order
    summary_df = pd.DataFrame(columns=["Protein", "Gene", "Pearson_r", "p_value"])
    for protein_name, gene_name in pairs:
        # find the record in summary_records
        rec = next((rec for rec in summary_records if rec["Protein"] == protein_name and rec["Gene"] == gene_name), None)
        if rec is None:
            rec = {"Protein": protein_name, "Gene": gene_name, "Pearson_r": np.nan, "p_value": np.nan}
        summary_df = pd.concat([summary_df, pd.DataFrame([rec])], ignore_index=True)

    # Save summary CSV
    with fs_out.open(f"{out_dir_path}protein_gene_correlation_summary.csv", 'w') as f:
        summary_df.to_csv(f, index=False)

    return summary_df
