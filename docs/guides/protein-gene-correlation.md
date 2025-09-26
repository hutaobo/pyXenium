# Protein–Gene correlation

Compute the Pearson correlation between **mean protein intensity** and **gene transcript density** on a user-defined grid.

```python
from pyXenium.io.partial_xenium_loader import load_anndata_from_partial
from pyXenium.analysis import protein_gene_correlation

BASE = "https://huggingface.co/datasets/<your-dataset>/resolve/main"
adata = load_anndata_from_partial(base_url=BASE)

pairs = [("CD3E", "CD3E"), ("E-Cadherin", "CDH1")]
summary = protein_gene_correlation(
    adata=adata,
    transcripts_zarr_path=BASE + "/transcripts.zarr.zip",
    pairs=pairs,
    output_dir="./protein_gene_corr",
    grid_size=(50, 50),
    pixel_size_um=0.2125,
    qv_threshold=20,
    overwrite=False,
)
print(summary)
```
