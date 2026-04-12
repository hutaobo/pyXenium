# pyXenium: robust loading and multimodal analysis of 10x Xenium outputs

[Author 1]^1, [Author 2]^1,* and [Author 3]^2

^1[Affiliation placeholder]

^2[Affiliation placeholder]

*To whom correspondence should be addressed.

## Abstract

**Summary:** 10x Genomics Xenium outputs combine sparse count matrices, cell tables, clustering results and optional boundary files, and practical analyses often begin with exports that are incomplete or stored in different matrix backends. pyXenium is a Python package for loading these outputs into `AnnData` while preserving modality separation and spatial annotations. In a validated smoke test on the public Xenium FFPE human renal cell carcinoma RNA+Protein dataset, pyXenium recovered 465,545 cells, 405 RNA features and 27 protein markers, reproduced the reported detected-cell count, and attached spatial coordinates and cluster labels under both automatic and explicit HDF5 loading. A separate partial loader supports counts-first recovery from incomplete exports and degrades to structured metadata rather than immediate failure when key artifacts are absent.

**Availability and implementation:** Implemented in Python. Source code: `https://github.com/hutaobo/pyXenium`. Package index: [PyPI URL placeholder if published]. Documentation: [documentation URL placeholder]. Current repository version: `0.1.0`. License: `LicenseRef-Proprietary-NonCommercial`.

**Contact:** [corresponding.author@institution.edu]

**Supplementary information:** Figure-generation code and validation outputs for this draft are stored under `manuscript/` in the repository. Additional supplementary information: [supplementary materials placeholder].

## 1 Introduction

10x Genomics Xenium experiments produce multiple output components rather than a single analysis-ready table. A typical run may include a cell-feature matrix in Zarr, HDF5 or MEX form, a per-cell table, clustering results, spatial centroid information and optional boundary files. Downstream single-cell and spatial analysis workflows, however, usually begin from a single `AnnData`-like object. In practice, loading Xenium data is therefore not only a file-parsing step but also an object-reconstruction step.

Two practical problems motivated pyXenium. First, Xenium RNA+Protein experiments need explicit separation of RNA counts from protein measurements while keeping both modalities aligned at the cell level. Second, users often work with incomplete exports, copied subsets of a run, or archives in which only part of the expected directory structure is available. Under those conditions, a loader that assumes one exact file layout can fail before any analysis begins.

pyXenium addresses these problems as an engineering and reproducibility contribution rather than as a new statistical method. The package provides a multimodal Xenium loader, a second loader for partial exports, a small command-line interface, a bundled toy dataset and optional downstream modules for protein-gene spatial correlation and RNA/protein joint analysis. This application note focuses on the loading and validation layers of the repository, because those are the components directly supported by real-data smoke testing and by the current test suite (Fig. 1).

## 2 Implementation

The public API exposed in `src/pyXenium` centers on two loader functions. `load_xenium_gene_protein` is designed for Xenium RNA+Protein outputs. It searches for a usable `cell_feature_matrix` in Zarr, HDF5 or MEX format, with `prefer="auto"` trying Zarr first, then HDF5, then MEX. The loader reads the matrix, inspects the `feature_type` annotation and splits the resulting features into RNA and protein modalities. RNA counts are stored in `adata.X` and mirrored in `adata.layers["rna"]`; protein measurements are stored as a per-cell `DataFrame` in `adata.obsm["protein"]`.

The same loader then enriches the object with run-level metadata. It reindexes the cell table to Xenium barcodes, reads clustering assignments from `analysis/clustering/gene_expression_graphclust/clusters.csv` when available, stores them in `adata.obs["cluster"]`, and adds centroid coordinates to `adata.obsm["spatial"]` when centroid columns are present in the cell table. If `cell_boundaries.csv.gz` or `nucleus_boundaries.csv.gz` exist, the raw boundary tables are attached in `adata.uns`. The loader also records modality metadata in `adata.uns["modality"]`, including the protein value type `"scaled_mean_intensity"`.

`load_anndata_from_partial` addresses incomplete exports. This entry point can combine any available MEX triplet with optional `analysis.zarr[.zip]`, `cells.zarr[.zip]` and `transcripts.zarr[.zip]` inputs. The implementation includes a ZIP-aware Zarr opener that detects nested Zarr roots inside archives, supports both local and remote paths, and assembles an `AnnData` object even when optional attachments are missing. When a MEX triplet is available, the function returns a counts-bearing object with feature metadata. When the MEX triplet is absent and `build_counts_if_missing=True`, it returns an empty `AnnData` together with parsed `analysis` and `cells` summaries in `adata.uns`, allowing callers to inspect what was available rather than receiving an immediate hard failure.

The repository includes additional reproducibility infrastructure around these loaders. `pyXenium.validation.renal_ffpe_protein` implements a smoke-test workflow for a public 10x Genomics Xenium FFPE human renal cell carcinoma RNA+Protein dataset. The smoke test can be called directly from `examples/smoke_test_10x_renal_ffpe_protein.py` or through the CLI command `pyxenium validate-renal-ffpe-protein` (equivalently `python -m pyXenium validate-renal-ffpe-protein`). Both routes produce machine-readable JSON and optional Markdown/CSV summaries. The package also ships a tiny bundled `toy_slide` dataset and CLI commands for demonstration and dataset copying.

## 3 Validation and use case

We validated the main multimodal loader on the public 10x Genomics dataset `Xenium In Situ Gene and Protein Expression data for FFPE Human Renal Cell Carcinoma`, using the repository smoke-test workflow. The smoke test was executed locally against a downloaded copy of the dataset with `prefer="auto"` and again with `prefer="h5"`. Both runs produced the same summary: 465,545 cells, 405 RNA features, 27 protein markers and 16,454,170 non-zero RNA matrix entries. In both runs, `adata.obsm["spatial"]` and `adata.obs["cluster"]` were present, `metrics_summary.csv` reported `num_cells_detected=465545`, and the validation payload contained no issues. This agreement matters because the default path and the explicit HDF5 path exercise different loader branches while arriving at the same cell, feature and metadata totals.

Direct inspection of the loaded object confirmed the structure expected by downstream workflows. On the validated renal dataset, `load_xenium_gene_protein` returned an `AnnData` object of shape 465,545 x 405, with `adata.layers["rna"]`, `adata.obsm["protein"]` and `adata.obsm["spatial"]` present. The spatial matrix had shape 465,545 x 2. `adata.obs["cluster"]` was categorical, and both `cell_boundaries` and `nucleus_boundaries` were attached under `adata.uns`. The loader therefore preserved not only the RNA matrix but also protein measurements, spatial centroids, clustering assignments and raw boundary tables in one aligned object.

We also evaluated the partial-loading path in two conditions. First, when `load_anndata_from_partial` was given only the real dataset `cell_feature_matrix` MEX directory and no `cells` or `analysis` attachments, it still returned an `AnnData` object of shape 465,545 x 543 with a `counts` layer and feature annotations. The recovered features comprised 405 gene-expression rows, 27 protein-expression rows and 111 control or unassigned rows, showing that a counts-first workflow can proceed even when optional spatial or clustering files are unavailable. Second, on the bundled toy dataset with no MEX triplet, the same function returned an empty `AnnData` but still populated `adata.uns["analysis"]` and `adata.uns["cells"]`, demonstrating the intended metadata-preserving fallback behavior for severely partial exports.

Repository-level checks support these data-level validations. In the current local repository state used for this manuscript draft, `pytest -q` collected and passed six tests. These tests cover the demo CLI, the toy dataset loader, bundled dataset copying, the public dataset catalog, the smoke-report rendering helper and the CLI wrapper for the renal FFPE validation command. Taken together, the real-data smoke test, the partial-loader checks and the passing test suite support a conservative claim: pyXenium provides a reproducible way to turn Xenium outputs into analysis-ready Python objects, with explicit support for RNA+Protein data and for incomplete-export recovery.

## 4 Availability and implementation

pyXenium is implemented in Python and uses `anndata`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `zarr`, `fsspec`, `requests`, `aiohttp` and `click` according to `pyproject.toml`. The current repository version is `0.1.0`, and the declared Python requirement is `>=3.8`. Source code is available at `https://github.com/hutaobo/pyXenium` and may also be distributed through [PyPI URL placeholder]. Documentation source files are present under `docs/`; the deployed documentation URL should be inserted here as [documentation URL placeholder]. The current license is `LicenseRef-Proprietary-NonCommercial`, which permits non-commercial source use; if a different release license is chosen before submission, this sentence should be updated accordingly. Operating-system support should be stated explicitly at submission as [validated platform statement placeholder].

## Figure Legends

**Figure 1. Evidence-backed summary of pyXenium loading and validation.**  
**(A)** Real-data smoke-test summary for the public 10x Genomics FFPE human renal cell carcinoma Xenium RNA+Protein dataset. The automatic loader path and explicit HDF5 path produced identical summaries, recovering 465,545 cells, 405 RNA features and 27 protein markers, while preserving spatial coordinates and cluster labels and returning no validation issues.  
**(B)** Top five RNA features by total counts in the validated `prefer="auto"` smoke test. Bars show total counts in millions, and text annotations report the number of cells with non-zero counts for each feature.  
**(C)** Top five protein markers by mean signal in the validated `prefer="auto"` smoke test. Bars show mean protein signal, and text annotations report the number of positive cells for each marker.  
**(D)** Additional evidence from the same repository validation workflow. Left: sizes of the five largest graph-based clusters recovered from the validated renal dataset. Right: feature-type composition of the real MEX-only partial load, which returned a 465,545 x 543 counts object containing gene-expression, protein-expression and control features without requiring optional spatial or clustering attachments.

## Funding

This work was supported by [Funding information placeholder].

## Conflict of Interest

Conflict of Interest: [Authors to complete. If none, use "none declared."]

## References

1. 10x Genomics. Xenium In Situ Gene and Protein Expression data for FFPE Human Renal Cell Carcinoma. Available at: https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma
2. [Xenium platform and file-format reference placeholder]
3. [AnnData citation placeholder]
4. [scikit-learn citation placeholder if retained]
5. [Additional spatial transcriptomics context reference placeholder]
