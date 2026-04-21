from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


def _normalize_frame_map(mapping: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for key, value in (mapping or {}).items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{key!r} must be a pandas.DataFrame, got {type(value)!r}.")
        normalized[str(key)] = value.copy()
    return normalized


def _normalize_image_map(mapping: dict[str, "XeniumImage"] | None) -> dict[str, "XeniumImage"]:
    normalized: dict[str, XeniumImage] = {}
    for key, value in (mapping or {}).items():
        if not isinstance(value, XeniumImage):
            raise TypeError(f"{key!r} must be a XeniumImage, got {type(value)!r}.")
        normalized[str(key)] = value
    return normalized


def _normalize_affine(matrix: Any) -> list[list[float]] | None:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"image_to_xenium_affine must have shape (3, 3), got {array.shape}.")
    return array.tolist()


def _normalize_axes(axes: str) -> str:
    text = str(axes or "").strip().lower()
    if not text:
        raise ValueError("Image axes must be a non-empty string.")
    for axis in text:
        if axis not in {"c", "x", "y"}:
            raise ValueError(f"Unsupported image axis {axis!r} in axes={axes!r}.")
    if "x" not in text or "y" not in text:
        raise ValueError(f"Image axes must include both 'x' and 'y', got {axes!r}.")
    return text


def _coerce_xy_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected an array with shape (N, 2), got {array.shape}.")
    return array


@dataclass
class XeniumImage:
    levels: list[Any]
    axes: str
    dtype: str
    source_path: str
    transform_kind: str = "affine"
    image_to_xenium_affine: list[list[float]] | None = None
    alignment_csv_path: str | None = None
    pixel_size_um: float | None = None
    keypoints_validation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.levels = list(self.levels or [])
        if not self.levels:
            raise ValueError("XeniumImage.levels must contain at least one multiscale level.")
        self.axes = _normalize_axes(self.axes)
        self.dtype = np.dtype(self.dtype).name
        self.source_path = str(self.source_path)
        self.transform_kind = str(self.transform_kind or "affine")
        self.image_to_xenium_affine = _normalize_affine(self.image_to_xenium_affine)
        if self.alignment_csv_path is not None:
            self.alignment_csv_path = str(self.alignment_csv_path)
        if self.pixel_size_um is not None:
            self.pixel_size_um = float(self.pixel_size_um)
        self.keypoints_validation = dict(self.keypoints_validation or {}) or None
        self.metadata = dict(self.metadata or {})
        self._validate_levels()

    def _validate_levels(self) -> None:
        ndim = None
        for index, level in enumerate(self.levels):
            shape = getattr(level, "shape", None)
            if shape is None:
                shape = np.shape(level)
            if not shape:
                raise ValueError(f"Image level {index} has an invalid empty shape.")
            if ndim is None:
                ndim = len(shape)
            elif len(shape) != ndim:
                raise ValueError("All image pyramid levels must have the same number of dimensions.")
        if ndim != len(self.axes):
            raise ValueError(
                "Image axes length does not match the dimensionality of the first pyramid level: "
                f"axes={self.axes!r}, ndim={ndim}."
            )

    def multiscale_shapes(self) -> list[tuple[int, ...]]:
        shapes: list[tuple[int, ...]] = []
        for level in self.levels:
            shape = getattr(level, "shape", None)
            if shape is None:
                shape = np.shape(level)
            shapes.append(tuple(int(value) for value in shape))
        return shapes

    def to_numpy_levels(self) -> list[np.ndarray]:
        return [np.asarray(level) for level in self.levels]

    def transform_metadata(self) -> dict[str, Any]:
        metadata = {
            "transform_direction": self.metadata.get(
                "transform_direction",
                "image_pixel_xy_to_xenium_pixel_xy",
            ),
            "transform_input_space": self.metadata.get(
                "transform_input_space",
                "image_pixel_xy",
            ),
            "transform_output_space": self.metadata.get(
                "transform_output_space",
                "xenium_pixel_xy",
            ),
            "transform_output_unit": self.metadata.get(
                "transform_output_unit",
                "pixel",
            ),
            "xenium_physical_unit": self.metadata.get(
                "xenium_physical_unit",
                "micron",
            ),
        }
        if self.pixel_size_um is not None:
            metadata["xenium_pixel_size_um"] = self.pixel_size_um
        return metadata

    def _affine_matrix(self) -> np.ndarray:
        if self.image_to_xenium_affine is None:
            raise ValueError("XeniumImage is missing image_to_xenium_affine metadata.")
        return np.asarray(self.image_to_xenium_affine, dtype=float)

    def image_xy_to_xenium_pixel_xy(self, image_xy: Any) -> np.ndarray:
        points = _coerce_xy_array(image_xy)
        affine = self._affine_matrix()
        return (np.c_[points, np.ones(len(points), dtype=float)] @ affine.T)[:, :2]

    def xenium_pixel_xy_to_image_xy(self, xenium_pixel_xy: Any) -> np.ndarray:
        points = _coerce_xy_array(xenium_pixel_xy)
        inverse = np.linalg.inv(self._affine_matrix())
        return (np.c_[points, np.ones(len(points), dtype=float)] @ inverse.T)[:, :2]

    def image_xy_to_xenium_um_xy(self, image_xy: Any) -> np.ndarray:
        if self.pixel_size_um is None:
            raise ValueError("XeniumImage is missing pixel_size_um metadata.")
        return self.image_xy_to_xenium_pixel_xy(image_xy) * self.pixel_size_um

    def xenium_um_to_image_xy(self, xenium_um_xy: Any) -> np.ndarray:
        if self.pixel_size_um is None:
            raise ValueError("XeniumImage is missing pixel_size_um metadata.")
        points_um = _coerce_xy_array(xenium_um_xy)
        return self.xenium_pixel_xy_to_image_xy(points_um / self.pixel_size_um)


@dataclass
class XeniumSData:
    table: ad.AnnData
    points: dict[str, pd.DataFrame] = field(default_factory=dict)
    shapes: dict[str, pd.DataFrame] = field(default_factory=dict)
    images: dict[str, XeniumImage] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.table, ad.AnnData):
            raise TypeError("XeniumSData.table must be an anndata.AnnData instance.")
        self.points = _normalize_frame_map(self.points)
        self.shapes = _normalize_frame_map(self.shapes)
        self.images = _normalize_image_map(self.images)
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
            from spatialdata.models import Image2DModel
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
        if self.images:
            images = {}
            for key, image in self.images.items():
                levels = image.to_numpy_levels()
                images[key] = Image2DModel.parse(
                    levels[0],
                    dims=tuple(image.axes),
                    scale_factors=[2] * max(len(levels) - 1, 0) if len(levels) > 1 else None,
                    transformations=None,
                )
            kwargs["images"] = images
        return SpatialData(**kwargs)

    def component_summary(self) -> dict[str, list[str]]:
        return {
            "tables": ["cells"],
            "points": sorted(self.points.keys()),
            "shapes": sorted(self.shapes.keys()),
            "images": sorted(self.images.keys()),
            "labels": sorted(self.metadata.get("labels", {}).keys()),
        }
