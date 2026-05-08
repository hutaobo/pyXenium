from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anndata as ad
import numpy as np
import pandas as pd

WSI_LEVEL_DOWNSAMPLE_RTOL = 0.02
WSI_LEVEL_DOWNSAMPLE_ATOL = 0.05


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


def _normalize_nested_image_map(
    mapping: dict[str, dict[str, "XeniumImage"]] | None,
) -> dict[str, dict[str, "XeniumImage"]]:
    normalized: dict[str, dict[str, XeniumImage]] = {}
    for key, value in (mapping or {}).items():
        if not isinstance(value, dict):
            raise TypeError(
                f"{key!r} must map to a dict[str, XeniumImage], got {type(value)!r}."
            )
        normalized[str(key)] = _normalize_image_map(value)
    return normalized


def _normalize_chunk_source_map(
    mapping: dict[str, "XeniumFrameChunkSource"] | None,
) -> dict[str, "XeniumFrameChunkSource"]:
    normalized: dict[str, XeniumFrameChunkSource] = {}
    for key, value in (mapping or {}).items():
        if not isinstance(value, XeniumFrameChunkSource):
            raise TypeError(
                f"{key!r} must be a XeniumFrameChunkSource, got {type(value)!r}."
            )
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


def _level_yx_shape(shape: tuple[int, ...], axes: str) -> tuple[int, int]:
    return int(shape[axes.index("y")]), int(shape[axes.index("x")])


def _to_yxc(level: Any, axes: str) -> np.ndarray:
    array = np.asarray(level)
    order = [axes.index("y"), axes.index("x")]
    if "c" in axes:
        order.append(axes.index("c"))
        return np.transpose(array, axes=tuple(order))
    array = np.transpose(array, axes=tuple(order))
    return array[..., np.newaxis]


@dataclass(frozen=True)
class XeniumFrameChunkSource:
    columns: tuple[str, ...]
    column_types: dict[str, str]
    chunk_iter_factory: Callable[[], Iterator[pd.DataFrame]]
    attrs: dict[str, Any] = field(default_factory=dict)
    preserve_extra_columns: bool = False

    def __post_init__(self) -> None:
        columns = tuple(str(column) for column in self.columns)
        if not columns:
            raise ValueError("XeniumFrameChunkSource.columns must be non-empty.")
        column_types = {str(key): str(value) for key, value in dict(self.column_types).items()}
        missing = set(columns).difference(column_types)
        if missing:
            raise ValueError(
                "XeniumFrameChunkSource.column_types is missing definitions for columns: "
                f"{sorted(missing)}"
            )
        unsupported = {
            column: dtype_name
            for column, dtype_name in column_types.items()
            if dtype_name not in {"bool", "int64", "float64", "string"}
        }
        if unsupported:
            raise ValueError(
                "Unsupported chunk source column types: "
                + ", ".join(f"{column}={dtype_name!r}" for column, dtype_name in sorted(unsupported.items()))
            )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "column_types", column_types)
        object.__setattr__(self, "attrs", dict(self.attrs or {}))
        object.__setattr__(self, "preserve_extra_columns", bool(self.preserve_extra_columns))

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        iterator = self.chunk_iter_factory()
        if iterator is None:
            raise TypeError("chunk_iter_factory() must return an iterator, got None.")
        expected = list(self.columns)
        for frame in iterator:
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    "XeniumFrameChunkSource must yield pandas.DataFrame chunks, got "
                    f"{type(frame)!r}."
                )
            missing = set(expected).difference(frame.columns)
            if missing:
                raise ValueError(
                    "Chunk source yielded a frame missing required columns: "
                    f"{sorted(missing)}"
                )
            columns = list(expected)
            if self.preserve_extra_columns:
                extras = [column for column in frame.columns if column not in self.column_types]
                columns.extend(extras)
            yield frame.loc[:, columns].copy()

    def materialize(self) -> pd.DataFrame:
        chunks = list(self.iter_chunks())
        if not chunks:
            return pd.DataFrame(columns=list(self.columns))
        return pd.concat(chunks, ignore_index=True)


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

    def level_chunk_shapes(self) -> list[tuple[int, ...] | None]:
        chunk_shapes: list[tuple[int, ...] | None] = []
        for level in self.levels:
            chunks = getattr(level, "chunks", None)
            if not chunks:
                chunk_shapes.append(None)
                continue
            try:
                chunk_shapes.append(tuple(int(value) for value in chunks))
            except TypeError:
                chunk_shapes.append(None)
        return chunk_shapes

    def infer_level_downsamples(
        self,
        *,
        rtol: float = WSI_LEVEL_DOWNSAMPLE_RTOL,
        atol: float = WSI_LEVEL_DOWNSAMPLE_ATOL,
    ) -> tuple[list[float] | None, list[str]]:
        level_shapes = self.multiscale_shapes()
        if not level_shapes:
            return None, ["Image has no multiscale levels."]

        base_y, base_x = _level_yx_shape(level_shapes[0], self.axes)
        if base_y <= 0 or base_x <= 0:
            return None, ["Level 0 has invalid spatial dimensions."]

        downsamples: list[float] = []
        issues: list[str] = []
        previous: float | None = None
        for level_index, shape in enumerate(level_shapes):
            level_y, level_x = _level_yx_shape(shape, self.axes)
            if level_y <= 0 or level_x <= 0:
                return None, [f"Level {level_index} has invalid spatial dimensions."]

            y_ratio = float(base_y) / float(level_y)
            x_ratio = float(base_x) / float(level_x)
            if not (np.isfinite(y_ratio) and np.isfinite(x_ratio)):
                return None, [f"Level {level_index} has non-finite downsample ratios."]
            if not np.isclose(y_ratio, x_ratio, rtol=rtol, atol=atol):
                issues.append(
                    "Level "
                    f"{level_index} has inconsistent x/y downsample ratios "
                    f"(y={y_ratio:.4f}, x={x_ratio:.4f})."
                )
                return None, issues

            downsample = float((y_ratio + x_ratio) / 2.0)
            if previous is not None and downsample <= previous:
                issues.append(
                    f"Level {level_index} is not lower resolution than the previous level."
                )
                return None, issues
            previous = downsample
            downsamples.append(downsample)

        if downsamples and not np.isclose(downsamples[0], 1.0, rtol=rtol, atol=atol):
            issues.append(f"Level 0 downsample must be 1.0, got {downsamples[0]:.4f}.")
            return None, issues
        return downsamples, issues

    def wsi_diagnostics(
        self,
        *,
        image_key: str = "he",
        canonical_source: str = "internal_slide_store",
    ) -> dict[str, Any]:
        level_shapes = [list(shape) for shape in self.multiscale_shapes()]
        chunk_shapes = [
            list(shape) if shape is not None else None for shape in self.level_chunk_shapes()
        ]
        level_downsamples, downsample_issues = self.infer_level_downsamples()
        image_role = str(self.metadata.get("image_role") or ("wsi_he" if image_key == "he" else "image"))
        native_pyramid = bool(self.metadata.get("native_pyramid", False))
        issues: list[str] = []
        if len(self.levels) < 2:
            issues.append("Expected at least 2 pyramid levels.")
        if self.pixel_size_um is None:
            issues.append("Missing level 0 microns-per-pixel metadata.")
        if self.image_to_xenium_affine is None:
            issues.append("Missing image_to_xenium_affine metadata.")
        if level_downsamples is None:
            issues.extend(downsample_issues or ["Unable to infer level downsample factors."])

        return {
            "image_key": image_key,
            "image_role": image_role,
            "canonical_wsi_source": str(
                self.metadata.get("canonical_wsi_source") or canonical_source
            ),
            "source_path": self.source_path,
            "alignment_csv_path": self.alignment_csv_path,
            "axes": self.axes,
            "dtype": self.dtype,
            "native_pyramid": native_pyramid,
            "n_levels": int(len(self.levels)),
            "level_shapes": level_shapes,
            "chunk_shapes": chunk_shapes,
            "level_downsamples": level_downsamples,
            "level0_mpp_um": float(self.pixel_size_um) if self.pixel_size_um is not None else None,
            "pixel_size_um": float(self.pixel_size_um) if self.pixel_size_um is not None else None,
            "transform_kind": self.transform_kind,
            "transform_direction": self.transform_metadata()["transform_direction"],
            "transform_input_space": self.transform_metadata()["transform_input_space"],
            "transform_output_space": self.transform_metadata()["transform_output_space"],
            "transform_output_unit": self.transform_metadata()["transform_output_unit"],
            "xenium_physical_unit": self.transform_metadata()["xenium_physical_unit"],
            "image_to_xenium_affine": self.image_to_xenium_affine,
            "wsi_ready": not issues,
            "issues": issues,
        }

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
class XeniumSlide:
    table: ad.AnnData
    points: dict[str, pd.DataFrame] = field(default_factory=dict)
    shapes: dict[str, pd.DataFrame] = field(default_factory=dict)
    images: dict[str, XeniumImage] = field(default_factory=dict)
    contour_images: dict[str, dict[str, XeniumImage]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    point_sources: dict[str, XeniumFrameChunkSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.table, ad.AnnData):
            raise TypeError("XeniumSlide.table must be an anndata.AnnData instance.")
        self.points = _normalize_frame_map(self.points)
        self.shapes = _normalize_frame_map(self.shapes)
        self.images = _normalize_image_map(self.images)
        self.contour_images = _normalize_nested_image_map(self.contour_images)
        self.metadata = dict(self.metadata or {})
        self.point_sources = _normalize_chunk_source_map(self.point_sources)
        self._validate()

    def _validate(self) -> None:
        overlap = set(self.points).intersection(self.point_sources)
        if overlap:
            raise ValueError(
                "XeniumSlide.points and XeniumSlide.point_sources cannot share keys: "
                f"{sorted(overlap)}"
            )
        if "transcripts" in self.points:
            required = {"x", "y", "gene_identity", "gene_name"}
            missing = required.difference(self.points["transcripts"].columns)
            if missing:
                raise ValueError(
                    "XeniumSlide.points['transcripts'] is missing required columns: "
                    f"{sorted(missing)}"
                )
        if "transcripts" in self.point_sources:
            required = {"x", "y", "gene_identity", "gene_name"}
            missing = required.difference(self.point_sources["transcripts"].columns)
            if missing:
                raise ValueError(
                    "XeniumSlide.point_sources['transcripts'] is missing required columns: "
                    f"{sorted(missing)}"
                )

        for key in ("cell_boundaries", "nucleus_boundaries"):
            if key not in self.shapes:
                continue
            required = {"cell_id", "vertex_id", "x", "y"}
            missing = required.difference(self.shapes[key].columns)
            if missing:
                raise ValueError(
                    f"XeniumSlide.shapes[{key!r}] is missing required columns: {sorted(missing)}"
                )

    def to_anndata(self) -> ad.AnnData:
        return self.table.copy()

    def inspect_wsi(self, image_key: str = "he") -> dict[str, Any]:
        if image_key not in self.images:
            raise KeyError(f"`slide.images[{image_key!r}]` was not found.")
        registry = dict(self.metadata.get("wsi", {}))
        diagnostics = dict(
            self.images[image_key].wsi_diagnostics(
                image_key=image_key,
                canonical_source=str(
                    registry.get("canonical_source", "internal_slide_store")
                ),
            )
        )
        diagnostics.update(
            {
                "format": self.metadata.get("format", "pyxenium.slide"),
                "version": self.metadata.get("store_version"),
                "default_image_key": registry.get("default_image_key", image_key),
            }
        )
        if "slide_store_path" in self.metadata:
            diagnostics["slide_store_path"] = self.metadata["slide_store_path"]
        return diagnostics

    def to_wsidata(self, image_key: str = "he"):
        diagnostics = self.inspect_wsi(image_key=image_key)
        if not diagnostics["wsi_ready"]:
            issues = "; ".join(str(item) for item in diagnostics.get("issues", []) if item)
            raise ValueError(
                f"`slide.images[{image_key!r}]` is not WSI-ready. {issues}"
            )

        try:
            from wsidata import SlideProperties, WSIData
            from wsidata.reader import ReaderBase
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "XeniumSlide.to_wsidata() requires the optional `wsidata` dependency. "
                "Install `pyXenium[lazyslide]` or install `wsidata` separately."
            ) from exc

        image = self.images[image_key]
        level_shapes = [tuple(shape) for shape in image.multiscale_shapes()]
        level_downsamples = diagnostics["level_downsamples"] or [1.0]
        bounds_y, bounds_x = _level_yx_shape(level_shapes[0], image.axes)
        properties_payload = {
            "shape": [int(bounds_y), int(bounds_x)],
            "n_level": int(len(level_shapes)),
            "level_shape": [
                [int(_level_yx_shape(shape, image.axes)[0]), int(_level_yx_shape(shape, image.axes)[1])]
                for shape in level_shapes
            ],
            "level_downsample": [float(value) for value in level_downsamples],
            "mpp": float(diagnostics["level0_mpp_um"]),
            "bounds": [0, 0, int(bounds_x), int(bounds_y)],
            "raw": {
                "pyxenium_image_key": image_key,
                "pyxenium_source_path": diagnostics["source_path"],
            },
        }
        if hasattr(SlideProperties, "from_mapping"):
            properties = SlideProperties.from_mapping(properties_payload)
        else:  # pragma: no cover
            properties = SlideProperties(**properties_payload)

        class _PyXeniumInternalReader(ReaderBase):
            def __init__(self, *, image: XeniumImage, file_path: str, slide_properties: Any):
                try:
                    super().__init__(file=file_path)
                except TypeError:
                    try:
                        super().__init__(file_path)
                    except TypeError:
                        try:
                            super().__init__()
                        except TypeError:
                            pass
                self.file = file_path
                self._reader = image
                self._properties = slide_properties
                self._name = "pyxenium_internal"

            @property
            def name(self) -> str:
                return self._name

            @property
            def reader(self) -> Any:
                return self._reader

            @property
            def properties(self) -> Any:
                return self._properties

            @property
            def raw_properties(self) -> dict[str, Any]:
                return properties_payload

            @property
            def associated_images(self) -> dict[str, Any]:
                return {}

            def create_reader(self) -> Any:
                return self._reader

            def detach_reader(self) -> None:
                self._reader = None

            def translate_level(self, level: int) -> int:
                if level < 0:
                    level = len(image.levels) + int(level)
                level = int(level)
                if level < 0 or level >= len(image.levels):
                    raise IndexError(f"Invalid pyramid level: {level}")
                return level

            def get_level(self, level: int, in_bounds: bool = False) -> np.ndarray:
                del in_bounds
                level_index = self.translate_level(level)
                return _to_yxc(image.levels[level_index], image.axes)

            def get_region(
                self,
                x: int,
                y: int,
                width: int,
                height: int,
                level: int = 0,
                **kwargs,
            ) -> np.ndarray:
                del kwargs
                level_index = self.translate_level(level)
                downsample = float(level_downsamples[level_index])
                x0 = int(np.floor(float(x) / downsample))
                y0 = int(np.floor(float(y) / downsample))
                x1 = max(x0 + int(width), x0)
                y1 = max(y0 + int(height), y0)
                level_data = image.levels[level_index]
                slices = [slice(None)] * len(image.axes)
                slices[image.axes.index("y")] = slice(y0, y1)
                slices[image.axes.index("x")] = slice(x0, x1)
                region = np.asarray(level_data[tuple(slices)])
                return _to_yxc(region, image.axes)

            def get_thumbnail(self, size: int, **kwargs) -> np.ndarray:
                del kwargs
                thumbnail = self.get_level(-1)
                if max(thumbnail.shape[0], thumbnail.shape[1]) <= int(size):
                    return thumbnail
                step = max(1, int(np.ceil(max(thumbnail.shape[0], thumbnail.shape[1]) / float(size))))
                return thumbnail[::step, ::step, ...]

        reader = _PyXeniumInternalReader(
            image=image,
            file_path=str(
                self.metadata.get("slide_store_path")
                or image.source_path
                or f"pyxenium://{image_key}"
            ),
            slide_properties=properties,
        )
        wsi = WSIData(
            reader=reader,
            attrs={
                "slide_properties": properties_payload,
                "pyxenium": {
                    "image_key": image_key,
                    "canonical_source": diagnostics["canonical_wsi_source"],
                },
            },
            slide_properties_source="slide",
        )
        if "slide_store_path" in self.metadata and hasattr(wsi, "set_wsi_store"):
            wsi.set_wsi_store(self.metadata["slide_store_path"])
        return wsi

    def to_wsi_spatialdata(self, image_key: str = "he"):
        return self.to_wsidata(image_key=image_key)

    def to_spatialdata(self):
        try:
            from spatialdata import SpatialData
            from spatialdata.models import Image2DModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "XeniumSlide.to_spatialdata() is an optional bridge that requires the "
                "'spatialdata' package to be installed separately. pyXenium core slide "
                "I/O does not depend on spatialdata."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.points or self.point_sources:
            points = dict(self.points)
            for key, source in self.point_sources.items():
                points[key] = source.materialize()
            kwargs["points"] = points
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
            "points": sorted(set(self.points).union(self.point_sources)),
            "shapes": sorted(self.shapes.keys()),
            "images": sorted(self.images.keys()),
            "contour_images": sorted(self.contour_images.keys()),
            "labels": sorted(self.metadata.get("labels", {}).keys()),
        }


def enrich_slide_wsi_metadata(
    slide: XeniumSlide,
    *,
    image_key: str = "he",
    store_version: int | None = None,
    format_name: str | None = None,
    slide_store_path: str | None = None,
) -> XeniumSlide:
    metadata = dict(slide.metadata or {})
    if store_version is not None:
        metadata["store_version"] = int(store_version)
    if format_name is not None:
        metadata["format"] = str(format_name)
    if slide_store_path is not None:
        metadata["slide_store_path"] = str(slide_store_path)

    if image_key in slide.images:
        diagnostics = slide.images[image_key].wsi_diagnostics(image_key=image_key)
        slide.images[image_key].metadata.update(
            {
                "image_role": diagnostics["image_role"],
                "canonical_wsi_source": diagnostics["canonical_wsi_source"],
                "native_pyramid": diagnostics["native_pyramid"],
                "wsi_ready": diagnostics["wsi_ready"],
                "level_downsamples": diagnostics["level_downsamples"],
                "chunk_shapes": diagnostics["chunk_shapes"],
                "level0_mpp_um": diagnostics["level0_mpp_um"],
                "n_levels": diagnostics["n_levels"],
                "wsi_issues": diagnostics["issues"],
            }
        )
        metadata["wsi"] = {
            "default_image_key": image_key,
            "canonical_source": diagnostics["canonical_wsi_source"],
            "store_version": metadata.get("store_version"),
            "n_levels": diagnostics["n_levels"],
            "level_shapes": diagnostics["level_shapes"],
            "level_downsamples": diagnostics["level_downsamples"],
            "chunk_shapes": diagnostics["chunk_shapes"],
            "level0_mpp_um": diagnostics["level0_mpp_um"],
            "wsi_ready": diagnostics["wsi_ready"],
            "native_pyramid": diagnostics["native_pyramid"],
            "image_role": diagnostics["image_role"],
            "source_path": diagnostics["source_path"],
            "alignment_csv_path": diagnostics["alignment_csv_path"],
            "issues": diagnostics["issues"],
        }
    else:
        metadata.pop("wsi", None)

    slide.metadata = metadata
    return slide
