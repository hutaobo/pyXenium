from __future__ import annotations

import json
from pathlib import Path


def get_vendored_gmi_path() -> Path:
    """Return the local, pinned Gmi R package source directory."""

    path = Path(__file__).resolve().parents[1] / "_vendor" / "Gmi"
    if not path.exists():
        raise FileNotFoundError(f"Vendored Gmi source is missing: {path}")
    return path


def get_vendored_gmi_metadata() -> dict:
    metadata_path = get_vendored_gmi_path() / "VENDOR_METADATA.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Vendored Gmi metadata is missing: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def assert_vendored_gmi_complete() -> None:
    root = get_vendored_gmi_path()
    required = [
        "DESCRIPTION",
        "NAMESPACE",
        "README.md",
        "LICENSE",
        "LICENSE.md",
        "R/Gmi.R",
        "R/predict_Gmi.R",
        "src/RcppExports.cpp",
        "man/Gmi.Rd",
        "VENDOR_METADATA.json",
    ]
    missing = [item for item in required if not (root / item).exists()]
    if missing:
        raise FileNotFoundError(f"Vendored Gmi snapshot is incomplete; missing: {missing}")
