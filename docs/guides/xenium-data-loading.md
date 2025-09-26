# Xenium data loading

This guide demonstrates how to load **RNA counts** together with the **cells table** and optional **analysis attachments** from a partial Xenium export.
It uses `pyXenium.io.partial_xenium_loader.load_anndata_from_partial` so you can work with remote files seamlessly (e.g., on Hugging Face) via `fsspec`.
