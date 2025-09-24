# pyXenium

A toy Python package for analyzing 10x Xenium data.

## Installation

```bash
pip install -U "pyXenium>=0.2.0"
```

## Quickstart

### Load a partial Xenium dataset from Hugging Face

The snippet below uses the **public demo dataset** and the **v2 loader** that supports `base_url`.

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
```

> **Note:** Requires `pyXenium>=0.2.0`.
> The demo dataset is hosted at:
> - Hugging Face Datasets: [hutaobo/pyxenium-gsm9116572](https://huggingface.co/datasets/hutaobo/pyxenium-gsm9116572)

---

## Development

To install with development dependencies (testing, docs, etc.):

```bash
pip install -e ".[dev]"
pytest
```

---

## Links

- 📦 PyPI: [pyXenium](https://pypi.org/project/pyXenium/)
- 📖 Documentation: [Read the Docs](https://pyxenium.readthedocs.io/en/latest/)
- 💻 Source code: [GitHub](https://github.com/hutaobo/pyXenium)

## License

MIT
