# Quickstart

## Install

> Until the package is published on PyPI, install from GitHub:

```bash
pip install -U "git+https://github.com/hutaobo/pyXenium@main"
# or for local development:
# git clone https://github.com/hutaobo/pyXenium
# cd pyXenium
# pip install -e ".[docs]"
```

## Load a partial Xenium export (RNA counts + optional cell/analysis attachments)

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial

BASE = "https://huggingface.co/datasets/<your-dataset>/resolve/main"
adata = load_anndata_from_partial(base_url=BASE)

adata
```
