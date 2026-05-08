from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a planar OME-TIFF into a tiffslide-readable tiled pyramid.",
    )
    parser.add_argument("--input", required=True, help="Input OME-TIFF.")
    parser.add_argument("--output", required=True, help="Output tiled pyramidal BigTIFF.")
    parser.add_argument("--tile-px", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-levels", type=int, default=None)
    parser.add_argument("--mpp", type=float, default=None)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Open the output with wsidata/tiffslide after writing.",
    )
    return parser.parse_args()


def _resolution_per_centimeter(mpp: float, downsample: float = 1.0) -> tuple[float, float]:
    pixels_per_cm = 10000.0 / (mpp * downsample)
    return pixels_per_cm, pixels_per_cm


def _infer_mpp(series: tifffile.TiffPageSeries) -> float | None:
    page = series.pages[0]
    try:
        xres = page.tags["XResolution"].value
        yres = page.tags["YResolution"].value
        unit = page.tags["ResolutionUnit"].value
    except KeyError:
        return None
    if unit.name == "CENTIMETER":
        x_pp_cm = xres[0] / xres[1]
        y_pp_cm = yres[0] / yres[1]
        return 10000.0 / float(np.mean([x_pp_cm, y_pp_cm]))
    return None


def _as_yxs(array: np.ndarray, axes: str) -> np.ndarray:
    if axes == "YXS":
        return np.ascontiguousarray(array)
    if axes == "SYX":
        return np.ascontiguousarray(np.moveaxis(array, 0, -1))
    if axes == "YXC":
        return np.ascontiguousarray(array)
    if axes == "CYX":
        return np.ascontiguousarray(np.moveaxis(array, 0, -1))
    raise ValueError(f"Unsupported TIFF axes for RGB conversion: {axes!r}")


def main() -> None:
    args = _parse_args()
    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    started = time.time()
    manifest: dict[str, object] = {
        "input": str(src),
        "output": str(dst),
        "tile_px": args.tile_px,
        "jpeg_quality": args.jpeg_quality,
        "started_at_unix": started,
        "status": "started",
    }

    with tifffile.TiffFile(src) as tif:
        series = tif.series[0]
        source_levels = list(series.levels)
        if args.max_levels is not None:
            source_levels = source_levels[: args.max_levels]
        if not source_levels:
            raise ValueError(f"No TIFF levels found in {src}")
        mpp = args.mpp if args.mpp is not None else _infer_mpp(series)
        if mpp is None:
            raise ValueError("Could not infer MPP; pass --mpp explicitly.")

        base_shape = source_levels[0].shape
        base_x = int(base_shape[2] if source_levels[0].axes == "SYX" else base_shape[1])

        written_levels: list[dict[str, object]] = []
        with tifffile.TiffWriter(tmp, bigtiff=True) as writer:
            for idx, level in enumerate(source_levels):
                array = _as_yxs(level.asarray(), level.axes)
                y, x = array.shape[:2]
                downsample = float(base_x / x)
                common = {
                    "photometric": "rgb",
                    "planarconfig": "CONTIG",
                    "tile": (args.tile_px, args.tile_px),
                    "compression": "jpeg",
                    "compressionargs": {"level": args.jpeg_quality},
                    "metadata": {"axes": "YXS"},
                    "resolution": _resolution_per_centimeter(mpp, downsample),
                    "resolutionunit": "CENTIMETER",
                    "maxworkers": 4,
                }
                if idx == 0:
                    writer.write(array, subifds=len(source_levels) - 1, **common)
                else:
                    writer.write(array, subfiletype=1, **common)
                written_levels.append(
                    {
                        "level": idx,
                        "shape_yxs": [int(y), int(x), int(array.shape[2])],
                        "downsample": downsample,
                    }
                )
                print(f"wrote level {idx}: {y}x{x}", flush=True)

    tmp.replace(dst)
    manifest.update(
        {
            "status": "completed",
            "runtime_seconds": time.time() - started,
            "mpp": mpp,
            "levels": written_levels,
            "n_levels": len(written_levels),
        }
    )

    if args.verify:
        from wsidata import open_wsi

        wsi = open_wsi(str(dst), reader="tiffslide", attach_thumbnail=False)
        manifest["verification"] = wsi.attrs.get("slide_properties", {})
        print("verification", json.dumps(manifest["verification"], sort_keys=True), flush=True)

    manifest_path = dst.with_suffix(dst.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {dst}", flush=True)
    print(f"manifest {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
