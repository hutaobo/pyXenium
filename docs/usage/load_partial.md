# Load a partial Xenium dataset from Hugging Face

This example uses the **public demo dataset** and the **v2 loader** that supports `base_url`.

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

BASE = "https://huggingface.co/datasets/hutaobo/pyxenium-gsm9116572/resolve/main"

adata = load_anndata_from_partial(
    base_url=BASE,
    analysis_name="analysis.zarr.zip",
    cells_name="cells.zarr.zip",
    transcripts_name="transcripts.zarr.zip",
    # Optional: if you uploaded a 10x MEX triplet under BASE/mex/
    # mex_dir=BASE + "/mex",
    # mex_matrix_name="matrix.mtx.gz",
    # mex_features_name="features.tsv.gz",
    # mex_barcodes_name="barcodes.tsv.gz",
    build_counts_if_missing=True,
)
print(adata)
