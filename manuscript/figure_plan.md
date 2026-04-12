# Figure Plan

## Main figure

**Figure 1: Evidence-backed summary of pyXenium loading and validation**

### Panel A: Loader workflow
- Show the two main entry points:
  - `load_xenium_gene_protein`
  - `load_anndata_from_partial`
- Inputs:
  - `cell_feature_matrix.zarr`
  - `cell_feature_matrix.h5`
  - `cell_feature_matrix/` (MEX)
  - `cells.csv.gz`
  - `analysis/.../clusters.csv`
  - optional `cell_boundaries.csv.gz` and `nucleus_boundaries.csv.gz`
- Outputs:
  - `adata.X` and `adata.layers["rna"]`
  - `adata.obsm["protein"]`
  - `adata.obsm["spatial"]`
  - `adata.obs["cluster"]`
  - `adata.uns["cell_boundaries"]`, `adata.uns["nucleus_boundaries"]`

### Panel B: Real-data smoke test
- Data source: `manuscript/evidence/smoke_auto/summary.json`
- Data source: `manuscript/evidence/smoke_h5/summary.json`
- Display:
  - cells: `465545`
  - RNA features: `405`
  - protein markers: `27`
  - `x_nnz`: `16454170`
  - `has_spatial = true`
  - `has_cluster = true`
  - `issues = []`
  - backend agreement between `auto` and `h5`

### Panel C: Object structure and partial loading evidence
- Data source: `manuscript/evidence/loader_auto_structure.json`
- Data source: `manuscript/evidence/partial_loader_mex_only.json`
- Display:
  - validated multimodal object structure
  - partial-loader MEX-only output shape and feature-type breakdown

### Panel D: Repository reproducibility
- Data source: `manuscript/evidence/pytest_q.txt`
- Display:
  - `pytest -q`: `6 passed`
  - smoke-test script present
  - validation CLI present
  - toy dataset bundled

## Design constraints

- 2 x 2 composite layout
- publication-style muted palette
- no text overlap
- legends or callouts outside data marks
- panel labels `A-D`
- export both PNG and PDF
