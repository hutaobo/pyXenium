from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import pandas as pd


def _normalize_frame_map(mapping: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for key, value in (mapping or {}).items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{key!r} must be a pandas.DataFrame, got {type(value)!r}.")
        normalized[str(key)] = value.copy()
    return normalized


@dataclass
class XeniumSData:
    table: ad.AnnData
    points: dict[str, pd.DataFrame] = field(default_factory=dict)
    shapes: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.table, ad.AnnData):
            raise TypeError("XeniumSData.table must be an anndata.AnnData instance.")
        self.points = _normalize_frame_map(self.points)
        self.shapes = _normalize_frame_map(self.shapes)
        self.metadata = dict(self.metadata or {})
        self._validate()

    def _validate(self) -> None:
        if "transcripts" in self.points:
            required = {"x", "y", "gene_identity", "gene_name"}
            missing = required.difference(self.points["transcripts"].columns)
            if missing:
                raise ValueError(
                    "XeniumSData.points['transcripts'] is missing required columns: "
                    f"{sorted(missing)}"
                )

        for key in ("cell_boundaries", "nucleus_boundaries"):
            if key not in self.shapes:
                continue
            required = {"cell_id", "vertex_id", "x", "y"}
            missing = required.difference(self.shapes[key].columns)
            if missing:
                raise ValueError(
                    f"XeniumSData.shapes[{key!r}] is missing required columns: {sorted(missing)}"
                )

    def to_anndata(self) -> ad.AnnData:
        return self.table.copy()

    def to_spatialdata(self):
        try:
            from spatialdata import SpatialData
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Optional dependency 'spatialdata' is required for XeniumSData.to_spatialdata()."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.points:
            kwargs["points"] = self.points
        if self.shapes:
            kwargs["shapes"] = self.shapes
        kwargs["tables"] = {"cells": self.table}
        return SpatialData(**kwargs)

    def component_summary(self) -> dict[str, list[str]]:
        return {
            "tables": ["cells"],
            "points": sorted(self.points.keys()),
            "shapes": sorted(self.shapes.keys()),
            "images": sorted(self.metadata.get("images", {}).keys()),
            "labels": sorted(self.metadata.get("labels", {}).keys()),
        }
