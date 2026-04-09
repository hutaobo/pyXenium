from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import zarr

_BUNDLED_DATASETS = {
    "toy_slide": ("pyXenium.datasets.toy_slide", ("cells.zarr.zip", "transcripts.zarr.zip", "analysis.zarr.zip")),
}


def _bundled_dataset_resource(name: str):
    try:
        package, _ = _BUNDLED_DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BUNDLED_DATASETS))
        raise FileNotFoundError(f"Unknown bundled dataset '{name}'. Available datasets: {available}.") from exc
    return resources.files(package)


def open_zarr_zip(zip_path):
    store = zarr.storage.ZipStore(str(zip_path), mode="r")
    return zarr.open_group(store=store, mode="r")


def load_toy():
    base = _bundled_dataset_resource("toy_slide")
    return {
        "cells": open_zarr_zip(base / "cells.zarr.zip"),
        "transcripts": open_zarr_zip(base / "transcripts.zarr.zip"),
        "analysis": open_zarr_zip(base / "analysis.zarr.zip"),
    }


def copy_bundled_dataset(name: str, dest: str | Path) -> Path:
    base = _bundled_dataset_resource(name)
    _, filenames = _BUNDLED_DATASETS[name]

    target = Path(dest) / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        with resources.as_file(base / filename) as source_path:
            shutil.copyfile(source_path, target / filename)
    return target
