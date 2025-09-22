from importlib import resources
import zarr

def open_zarr_zip(zip_path):
    store = zarr.storage.ZipStore(str(zip_path), mode="r")
    return zarr.group(store=store)

def load_toy():
    base = resources.files("pyxenium.datasets.toy_slide")
    return {
        "cells": open_zarr_zip(base / "cells.zarr.zip"),
        "transcripts": open_zarr_zip(base / "transcripts.zarr.zip"),
        "analysis": open_zarr_zip(base / "analysis.zarr.zip"),
    }
