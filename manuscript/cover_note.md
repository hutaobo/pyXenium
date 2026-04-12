# Cover Note

## Evidence used

- Repository inspection:
  - `README.md`
  - `pyproject.toml`
  - `src/pyXenium/io/partial_xenium_loader.py`
  - `src/pyXenium/io/xenium_gene_protein_loader.py`
  - `src/pyXenium/validation/renal_ffpe_protein.py`
  - `src/pyXenium/__main__.py`
  - `src/pyXenium/datasets/catalog.py`
  - `tests/`
  - `docs/`
- Real-data runs performed locally on the public 10x FFPE renal carcinoma Xenium RNA+Protein dataset:
  - smoke test with `prefer="auto"`
  - smoke test with `prefer="h5"`
  - direct object inspection with `load_xenium_gene_protein`
  - `load_anndata_from_partial` on the real `cell_feature_matrix` MEX directory
- Additional code-behavior check:
  - `load_anndata_from_partial` on the bundled toy dataset without MEX
- Reproducibility checks:
  - `pytest --collect-only -q` collected 6 tests
  - `pytest -q` passed locally with 6 tests

## Claims strongly supported

- pyXenium can load the validated renal FFPE Xenium RNA+Protein dataset into `AnnData`.
- The `auto` and explicit `h5` loader paths recovered the same validated summary:
  - `465545` cells
  - `405` RNA features
  - `27` protein markers
  - `16454170` RNA non-zero entries
  - spatial coordinates present
  - cluster labels present
  - no smoke-test issues
- The validated loaded object contains:
  - `adata.layers["rna"]`
  - `adata.obsm["protein"]`
  - `adata.obsm["spatial"]`
  - categorical `adata.obs["cluster"]`
  - `cell_boundaries` and `nucleus_boundaries` in `adata.uns`
- `load_anndata_from_partial` can recover a counts object from the real MEX directory alone:
  - shape `465545 x 543`
  - counts layer present
  - feature metadata preserved
  - optional spatial and clustering attachments not required
- The repository currently includes a bundled toy dataset, a validation module, a validation CLI command and a passing local pytest run with 6 tests.

## Claims intentionally avoided

- I did not claim algorithmic novelty beyond software robustness and data handling, because the strongest direct evidence is for loading, validation and reproducibility rather than for a new statistical method.
- I did not claim runtime or memory advantages, because no benchmark suite for those quantities was generated here.
- I did not claim biological findings from the renal dataset beyond the observed loaded counts, top-feature summaries and presence of metadata, because this manuscript draft is positioned as a software note.
- I did not describe the current license as open source, because the repository metadata states `LicenseRef-Proprietary-NonCommercial`.

## Metadata to verify before submission

- Author list, affiliations and corresponding author email
- Final GitHub, PyPI and documentation URLs
- Whether to archive the release on Zenodo and insert a DOI
- Final package version to cite in the manuscript
- Funding statement
- Conflict-of-interest statement
- Preferred software/data references
- Operating-system support statement for the availability paragraph

## Submission optics to consider

- `pyproject.toml` currently describes pyXenium as `"A toy Python package for analyzing 10x Xenium data."` This wording is not suitable for manuscript or release metadata and should be strengthened before submission.
- The current non-commercial source-available license is compatible with the manuscript's factual description, but it may read less favorably than a standard open-source license in reviewer and editor assessment. If the license changes, update the manuscript availability paragraph accordingly.
