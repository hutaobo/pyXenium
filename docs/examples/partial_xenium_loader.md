# Load a partial Xenium dataset (remote)

This example loads a public demo dataset directly from Hugging Face using the v2 loader that supports `base_url`.

> **Requirements**
>
> - `pyXenium >= 0.2.0`
> - Internet access for the first run (files are streamed/downloaded)

```python
# doctest: +ELLIPSIS
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

BASE = "https://huggingface.co/datasets/hutaobo/pyxenium-gsm9116572/resolve/main"

adata = load_anndata_from_partial(
    base_url=BASE,
    analysis_name="analysis.zarr.zip",
    cells_name="cells.zarr.zip",
    transcripts_name="transcripts.zarr.zip",
    # Optional 10x MEX:
    # mex_dir=BASE + "/mex",
    # mex_matrix_name="matrix.mtx.gz",
    # mex_features_name="features.tsv.gz",
    # mex_barcodes_name="barcodes.tsv.gz",
    build_counts_if_missing=True,
)

print(adata)  # doctest: +ELLIPSIS
