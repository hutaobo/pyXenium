from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAMPLE_ROLES = ("breast_discovery", "cervical_validation")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_yxc(array: np.ndarray, axes: str) -> np.ndarray:
    data = np.asarray(array)
    axes = str(axes).upper()
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image level, got {data.shape}.")
    elif axes in {"SYX", "CYX"}:
        data = np.moveaxis(data, 0, -1)
    elif axes in {"YXS", "YXC"}:
        pass
    elif axes in {"XSY", "XCY"}:
        data = np.moveaxis(data, 1, -1)
        data = np.swapaxes(data, 0, 1)
    elif data.shape[0] in {3, 4}:
        data = np.moveaxis(data, 0, -1)
    elif data.shape[-1] not in {3, 4}:
        raise ValueError(f"Could not infer channel axis from shape {data.shape} and axes {axes!r}.")
    if data.shape[-1] > 3:
        data = data[..., :3]
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=2)
    return data


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.dtype == np.uint8:
        return data[..., :3]
    if np.issubdtype(data.dtype, np.integer):
        max_value = float(np.iinfo(data.dtype).max)
        scaled = np.clip(np.asarray(data, dtype=np.float32) / max_value, 0.0, 1.0)
    else:
        scaled = np.asarray(data, dtype=np.float32)
        finite = scaled[np.isfinite(scaled)]
        if finite.size and float(finite.max()) > 1.5:
            scaled = scaled / float(finite.max())
        scaled = np.clip(scaled, 0.0, 1.0)
    return np.clip(np.rint(scaled[..., :3] * 255.0), 0, 255).astype(np.uint8)


def _load_he_level(path: Path, level_index: int) -> tuple[np.ndarray, tuple[int, int], str]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", []) or [series])
        if level_index < 0 or level_index >= len(levels):
            raise IndexError(f"Requested H&E level {level_index}, but {path} exposes {len(levels)} levels.")
        level = levels[level_index]
        base = levels[0]
        image = _as_yxc(level.asarray(), str(level.axes))
        base_axes = str(base.axes).upper()
        base_shape = tuple(int(value) for value in base.shape)
        base_y = base_shape[base_axes.find("Y")] if "Y" in base_axes else base_shape[-2]
        base_x = base_shape[base_axes.find("X")] if "X" in base_axes else base_shape[-1]
        return _to_uint8_rgb(image), (int(base_y), int(base_x)), str(level.axes)


def _prepare_cells(sample_dir: Path, input_manifest: pd.Series, block_manifest: pd.Series) -> pd.DataFrame:
    root = Path(str(input_manifest["xenium_root"]))
    cells_path = root / "cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"Missing Xenium cells table: {cells_path}")
    cells = pd.read_parquet(cells_path)
    required = ["cell_id", "x_centroid", "y_centroid", "transcript_counts"]
    missing = [column for column in required if column not in cells.columns]
    if missing:
        raise KeyError(f"{cells_path} is missing required columns: {missing}")
    cells = cells.loc[:, [column for column in required if column in cells.columns]].copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    for column in ("x_centroid", "y_centroid", "transcript_counts"):
        cells[column] = pd.to_numeric(cells[column], errors="coerce")
    cells = cells.dropna(subset=["x_centroid", "y_centroid"])
    cells = cells.loc[cells["transcript_counts"].fillna(0.0) > 0].copy()

    max_cells_raw = input_manifest.get("max_cells", "")
    max_cells = None if pd.isna(max_cells_raw) or str(max_cells_raw) == "" else int(float(max_cells_raw))
    seed_raw = input_manifest.get("random_state", "")
    seed = None if pd.isna(seed_raw) or str(seed_raw) == "" else int(float(seed_raw))
    if max_cells is not None and len(cells) > max_cells:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(cells), size=max_cells, replace=False))
        cells = cells.iloc[selected].copy()
    cells = cells.drop_duplicates("cell_id", keep="first").reset_index(drop=True)

    bins = int(float(block_manifest["bins"]))
    min_cells = int(float(block_manifest["min_cells_per_block"]))
    try:
        x_bin = pd.qcut(cells["x_centroid"].rank(method="average"), q=bins, labels=False, duplicates="drop")
        y_bin = pd.qcut(cells["y_centroid"].rank(method="average"), q=bins, labels=False, duplicates="drop")
    except ValueError:
        x_bin = pd.Series(0, index=cells.index)
        y_bin = pd.Series(0, index=cells.index)
    block_id = "block_x" + x_bin.astype("Int64").astype(str) + "_y" + y_bin.astype("Int64").astype(str)
    sizes = block_id.value_counts()
    keep_blocks = set(sizes.loc[sizes >= min_cells].index)
    cells["spatial_block_id"] = block_id
    cells = cells.loc[cells["spatial_block_id"].isin(keep_blocks)].copy()
    return cells


def _write_pixels(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.int32)
    flat = pd.DataFrame(
        {
            "x": xx.reshape(-1),
            "y": yy.reshape(-1),
            "r": image[:, :, 0].reshape(-1),
            "g": image[:, :, 1].reshape(-1),
            "b": image[:, :, 2].reshape(-1),
        }
    )
    flat.to_csv(path, index=False, compression="gzip")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_sample(
    *,
    sample_role: str,
    run_dir: Path,
    output_dir: Path,
    max_raster_dim: int,
) -> dict[str, Any]:
    sample_dir = run_dir / sample_role
    input_manifest = _read_csv(sample_dir / "input_manifest.csv").iloc[0]
    he_manifest = _read_csv(sample_dir / "he_feature_manifest.csv").iloc[0]
    block_manifest = _read_csv(sample_dir / "spatial_block_manifest.csv").iloc[0]
    pathway_activity = _read_csv(sample_dir / "pathway_activity.csv")

    he_image_path = Path(str(he_manifest["he_image_path"]))
    alignment_path = Path(str(he_manifest["he_alignment_path"]))
    level_index = int(float(he_manifest["he_pyramid_level"]))
    he_downsample_y = float(he_manifest["he_downsample_y"])
    he_downsample_x = float(he_manifest["he_downsample_x"])

    level_image, base_shape_yx, level_axes = _load_he_level(he_image_path, level_index)
    stride = max(1, int(np.ceil(max(level_image.shape[:2]) / float(max_raster_dim))))
    display_image_full = level_image[::stride, ::stride, :3]

    affine = np.loadtxt(alignment_path, delimiter=",")
    if affine.shape != (3, 3):
        raise ValueError(f"H&E alignment matrix must be 3x3, got {affine.shape} at {alignment_path}")
    inverse = np.linalg.inv(affine)
    cells = _prepare_cells(sample_dir, input_manifest, block_manifest)
    xy = cells.loc[:, ["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    image_xy = np.c_[xy, np.ones(len(xy), dtype=float)] @ inverse.T
    cells["he_level_x"] = image_xy[:, 0] / he_downsample_x
    cells["he_level_y"] = image_xy[:, 1] / he_downsample_y
    cells["display_x"] = cells["he_level_x"] / stride
    cells["display_y"] = cells["he_level_y"] / stride
    height, width = display_image_full.shape[:2]
    cells = cells.loc[
        cells["display_x"].between(0, width - 1) & cells["display_y"].between(0, height - 1)
    ].copy()

    crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, width, height
    if not cells.empty:
        min_crop = int(min(max(width, height), 160))
        min_x = float(cells["display_x"].min())
        max_x = float(cells["display_x"].max())
        min_y = float(cells["display_y"].min())
        max_y = float(cells["display_y"].max())
        span = max(max_x - min_x, max_y - min_y, 1.0)
        margin = max(40.0, span * 1.1)
        crop_x0 = int(np.floor(min_x - margin))
        crop_x1 = int(np.ceil(max_x + margin))
        crop_y0 = int(np.floor(min_y - margin))
        crop_y1 = int(np.ceil(max_y + margin))

        def expand_to_min(start: int, end: int, limit: int) -> tuple[int, int]:
            size = end - start
            if size >= min_crop:
                return start, end
            center = (start + end) / 2.0
            start = int(np.floor(center - min_crop / 2.0))
            end = start + min_crop
            if start < 0:
                end -= start
                start = 0
            if end > limit:
                start -= end - limit
                end = limit
            return max(start, 0), min(end, limit)

        crop_x0, crop_x1 = expand_to_min(crop_x0, crop_x1, width)
        crop_y0, crop_y1 = expand_to_min(crop_y0, crop_y1, height)
        crop_x0 = max(crop_x0, 0)
        crop_y0 = max(crop_y0, 0)
        crop_x1 = min(crop_x1, width)
        crop_y1 = min(crop_y1, height)

    display_image = display_image_full[crop_y0:crop_y1, crop_x0:crop_x1, :]
    cells["display_x"] = cells["display_x"] - crop_x0
    cells["display_y"] = cells["display_y"] - crop_y0
    pixel_path = output_dir / f"he_overview_pixels_{sample_role}.csv.gz"
    _write_pixels(pixel_path, display_image)

    block_centroids = (
        cells.groupby("spatial_block_id", as_index=False)
        .agg(
            display_x=("display_x", "mean"),
            display_y=("display_y", "mean"),
            xenium_x=("x_centroid", "mean"),
            xenium_y=("y_centroid", "mean"),
            n_cells=("cell_id", "size"),
        )
        .merge(pathway_activity, on="spatial_block_id", how="left")
    )
    block_path = output_dir / f"he_block_overlay_{sample_role}.csv"
    block_centroids.to_csv(block_path, index=False)

    x_um_per_base_px = float(np.linalg.norm(affine[:2, 0]))
    y_um_per_base_px = float(np.linalg.norm(affine[:2, 1]))
    display_downsample_x = he_downsample_x * stride
    display_downsample_y = he_downsample_y * stride
    return {
        "sample_role": sample_role,
        "raw_he_image_path": str(he_image_path),
        "he_alignment_path": str(alignment_path),
        "he_pyramid_level": level_index,
        "he_level_axes": level_axes,
        "base_shape_yx": str(tuple(base_shape_yx)),
        "level_shape_yx": str(tuple(int(v) for v in level_image.shape[:2])),
        "display_shape_yx": str(tuple(int(v) for v in display_image.shape[:2])),
        "display_crop_xyxy": str((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1))),
        "display_stride_from_level": stride,
        "he_downsample_x": he_downsample_x,
        "he_downsample_y": he_downsample_y,
        "display_downsample_x": display_downsample_x,
        "display_downsample_y": display_downsample_y,
        "um_per_base_pixel_x": x_um_per_base_px,
        "um_per_base_pixel_y": y_um_per_base_px,
        "um_per_display_pixel_mean": float(
            (x_um_per_base_px * display_downsample_x + y_um_per_base_px * display_downsample_y) / 2.0
        ),
        "pixel_source_data": str(pixel_path),
        "block_overlay_source_data": str(block_path),
        "extraction_method": "python_tifffile_numeric_source_data_only; plotting_and_export_in_R",
        "image_adjustment": "no local adjustment; uint8 values read from selected low-resolution OME-TIF pyramid level",
        "scale_calibration": "image_to_xenium affine norm multiplied by emitted display downsample",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare numeric H&E source data for the R-only morphopathway Nature figure renderer."
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--max-raster-dim", type=int, default=900)
    args = parser.parse_args()

    manifest = _read_json(args.package_dir / "brief_communication_package_manifest.json")
    run_dirs = [Path(value) for value in manifest.get("run_dirs", [])]
    if not run_dirs:
        raise ValueError("brief_communication_package_manifest.json has no run_dirs.")
    run_dir = run_dirs[int(args.run_index)]
    rows = [
        prepare_sample(
            sample_role=sample_role,
            run_dir=run_dir,
            output_dir=args.output_dir,
            max_raster_dim=int(args.max_raster_dim),
        )
        for sample_role in SAMPLE_ROLES
    ]
    _write_manifest(args.output_dir / "he_image_integrity_manifest.csv", rows)
    print(json.dumps({"status": "ok", "run_dir": str(run_dir), "n_samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
