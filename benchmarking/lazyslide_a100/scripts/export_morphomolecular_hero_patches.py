from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export H&E patches for mTM hero contours selected from a direct LazySlide run."
    )
    parser.add_argument("--run-dir", required=True, help="Workflow output directory.")
    parser.add_argument("--wsi-path", required=True, help="Prepared tiled pyramidal WSI.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <run-dir>/hero_patches.")
    parser.add_argument(
        "--hero-stem",
        default="morphomolecular_hero_contours",
        help="Hero contour table stem in run-dir. Defaults to morphomolecular_hero_contours.",
    )
    parser.add_argument("--max-contours", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--pad", type=int, default=192)
    parser.add_argument("--montage-tile-size", type=int, default=256)
    parser.add_argument(
        "--allow-duplicate-contours",
        action="store_true",
        help="By default, keep only the highest-ranked row for each contour_id.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_table(run_dir: Path, stem: str) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        path = run_dir / f"{stem}{suffix}"
        if path.exists():
            return _read_table(path)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {run_dir}")


def _open_slide(path: Path):
    try:
        import tiffslide

        return tiffslide.TiffSlide(str(path))
    except ImportError as exc:
        raise ImportError("Install tiffslide to export hero patches from pyramidal WSIs.") from exc


def _geometry_from_row(row: pd.Series) -> BaseGeometry | None:
    if "geometry_wkt" in row and pd.notna(row["geometry_wkt"]):
        return wkt.loads(str(row["geometry_wkt"]))
    if "geometry" in row and pd.notna(row["geometry"]):
        value = row["geometry"]
        if isinstance(value, BaseGeometry):
            return value
        return wkt.loads(str(value))
    return None


def _crop_bounds(geometry: BaseGeometry, *, patch_size: int, pad: int) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = geometry.bounds
    width = max(int(math.ceil(maxx - minx)) + 2 * pad, int(patch_size))
    height = max(int(math.ceil(maxy - miny)) + 2 * pad, int(patch_size))
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    x0 = max(int(round(center_x - width / 2.0)), 0)
    y0 = max(int(round(center_y - height / 2.0)), 0)
    return x0, y0, width, height


def _draw_geometry(image: Image.Image, geometry: BaseGeometry, *, x0: int, y0: int) -> None:
    draw = ImageDraw.Draw(image)
    for polygon in _iter_polygons(geometry):
        coords = [(float(x) - x0, float(y) - y0) for x, y in polygon.exterior.coords]
        if len(coords) >= 2:
            draw.line(coords, fill=(255, 64, 64), width=5, joint="curve")
        for interior in polygon.interiors:
            hole = [(float(x) - x0, float(y) - y0) for x, y in interior.coords]
            if len(hole) >= 2:
                draw.line(hole, fill=(255, 160, 64), width=3, joint="curve")


def _iter_polygons(geometry: BaseGeometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def _label(row: pd.Series) -> str:
    target = str(row.get("target_feature", "target")).replace("program__", "").replace("__mean", "")
    structure = str(row.get("assigned_structure", "structure"))
    delta = row.get("hidden_program_score", row.get("concordance_score", ""))
    if isinstance(delta, float):
        delta_text = f"{delta:.2f}"
    else:
        delta_text = str(delta)
    return f"{structure}\n{target}\nscore {delta_text}"


def _make_montage(records: list[dict[str, object]], output_path: Path, *, tile_size: int) -> None:
    if not records:
        return
    cols = min(4, len(records))
    rows = int(math.ceil(len(records) / cols))
    label_h = 72
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        image = Image.open(record["patch_path"]).convert("RGB").resize((tile_size, tile_size))
        x = (index % cols) * tile_size
        y = (index // cols) * (tile_size + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + tile_size + 8), str(record["label"])[:96], fill=(0, 0, 0))
    canvas.save(output_path)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "hero_patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    hero = _resolve_table(run_dir, args.hero_stem)
    if not args.allow_duplicate_contours and "contour_id" in hero.columns:
        hero = hero.drop_duplicates("contour_id", keep="first")
    hero = hero.head(args.max_contours)
    contours = _resolve_table(run_dir, "image_contours")
    contours["contour_id"] = contours["contour_id"].astype(str)
    contour_lookup = contours.set_index("contour_id", drop=False)
    slide = _open_slide(Path(args.wsi_path).expanduser().resolve())

    records: list[dict[str, object]] = []
    for rank, (_, row) in enumerate(hero.iterrows(), start=1):
        contour_id = str(row["contour_id"])
        if contour_id not in contour_lookup.index:
            continue
        geometry = _geometry_from_row(contour_lookup.loc[contour_id])
        if geometry is None or geometry.is_empty:
            continue
        x0, y0, width, height = _crop_bounds(geometry, patch_size=args.patch_size, pad=args.pad)
        patch = slide.read_region((x0, y0), 0, (width, height)).convert("RGB")
        _draw_geometry(patch, geometry, x0=x0, y0=y0)
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in contour_id)[:80]
        patch_path = output_dir / f"hero_{rank:02d}_{safe_id}.png"
        patch.save(patch_path)
        record = {
            "rank": rank,
            "contour_id": contour_id,
            "assigned_structure": row.get("assigned_structure"),
            "target_feature": row.get("target_feature"),
            "image_feature": row.get("image_feature"),
            "hidden_program_score": row.get("hidden_program_score"),
            "target_z_within_structure": row.get("target_z_within_structure"),
            "image_z_within_structure": row.get("image_z_within_structure"),
            "x0": x0,
            "y0": y0,
            "width": width,
            "height": height,
            "patch_path": str(patch_path),
            "label": _label(row),
        }
        records.append(record)

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "hero_patch_manifest.csv", index=False)
    _make_montage(records, output_dir / "hero_patch_montage.png", tile_size=args.montage_tile_size)
    print(f"Wrote {len(records)} hero patches to {output_dir}")


if __name__ == "__main__":
    main()
