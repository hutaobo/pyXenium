#!/usr/bin/env python3
"""Harden breast S3 hero contours and contour-size sensitivity tables.

This script is intentionally read-only with respect to mTM runs. It reads
existing contour-level PLIP/UNI output tables, selects QC-filtered illustrative
breast S3 luminal estrogen-response contours, exports H&E crops, and recomputes
partial Spearman sensitivity after excluding small contours.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


DEFAULT_CONTROLS = (
    "assigned_structure",
    "centroid_x",
    "centroid_y",
    "cell_boundary_distance_um__mean",
    "tile_boundary_distance_px__mean",
)

BREAST_SELECTED_PAIRS = [
    ("S3 S3 #417.1", "S3 S3 #519.1"),
    ("S3 S3 #139.1", "S3 S3 #56.1"),
    ("S3 S3 #351.1", "S3 S3 #188.1"),
    ("S3 S3 #173.1", "S3 S3 #285.1"),
]

SENSITIVITY_FILTERS = [
    ("full_set", None, None),
    ("n_tiles_ge_2", 2, None),
    ("n_tiles_ge_3", 3, None),
    ("n_cells_ge_20", None, 20),
    ("n_cells_ge_50", None, 50),
    ("n_tiles_ge_3_and_n_cells_ge_50", 3, 50),
]


@dataclass(frozen=True)
class Candidate:
    dataset: str
    model: str
    program: str
    molecular_feature: str
    image_feature: str
    assigned_structure_filter: str | None
    reported_partial_rho: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breast-plip-run", type=Path, required=True)
    parser.add_argument("--breast-uni-run", type=Path, required=True)
    parser.add_argument("--cervical-plip-run", type=Path, required=True)
    parser.add_argument("--cervical-uni-run", type=Path, required=True)
    parser.add_argument("--breast-wsi", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--pad", type=int, default=240)
    parser.add_argument("--max-patch-size", type=int, default=2600)
    parser.add_argument("--montage-patch-size", type=int, default=560)
    parser.add_argument("--area-quantiles", type=float, nargs=2, default=(0.10, 0.90))
    parser.add_argument("--max-edge-fraction", type=float, default=0.75)
    parser.add_argument("--min-tiles", type=int, default=3)
    parser.add_argument("--min-cells", type=int, default=50)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table: {path}")


def contour_table(run_dir: Path) -> pd.DataFrame:
    for name in ("contour_multimodal_summary.parquet", "contour_multimodal_summary.csv"):
        path = run_dir / name
        if path.exists():
            return read_table(path)
    raise FileNotFoundError(f"Missing contour_multimodal_summary under {run_dir}")


def image_contours(run_dir: Path) -> pd.DataFrame:
    for name in ("image_contours.parquet", "image_contours.csv"):
        path = run_dir / name
        if path.exists():
            return read_table(path)
    raise FileNotFoundError(f"Missing image_contours under {run_dir}")


def read_candidates(path: Path) -> list[Candidate]:
    source = pd.read_csv(path)
    rows: list[Candidate] = []
    for _, row in source.iterrows():
        rows.append(
            Candidate(
                dataset=str(row.get("dataset", "")).lower(),
                model=str(row.get("model", "")).lower(),
                program=str(row.get("program", "")),
                molecular_feature=str(row.get("molecular_feature", "")),
                image_feature=str(row.get("image_feature", "")),
                assigned_structure_filter=clean_optional(row.get("assigned_structure_filter")),
                reported_partial_rho=to_float_or_none(row.get("reported_partial_rho")),
            )
        )
    return rows


def clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def to_float_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def control_matrix(frame: pd.DataFrame, controls: tuple[str, ...] = DEFAULT_CONTROLS) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if "assigned_structure" in controls and "assigned_structure" in frame.columns:
        dummies = pd.get_dummies(
            frame["assigned_structure"].fillna("unassigned").astype(str),
            prefix="structure",
            dtype=float,
        )
        if dummies.shape[1] > 1:
            dummies = dummies.iloc[:, 1:]
        pieces.append(dummies)
    for control in controls:
        if control == "assigned_structure" or control not in frame.columns:
            continue
        values = pd.to_numeric(frame[control], errors="coerce")
        if values.notna().sum() == 0:
            continue
        pieces.append(pd.DataFrame({control: values.fillna(values.median()).astype(float)}, index=frame.index))
    if not pieces:
        return pd.DataFrame(index=frame.index)
    out = pd.concat(pieces, axis=1)
    out.index = frame.index
    return out


def residualize(values: np.ndarray, controls: pd.DataFrame) -> np.ndarray:
    if controls.empty:
        return values - np.nanmean(values)
    design = np.column_stack([np.ones(values.shape[0], dtype=float), controls.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def partial_spearman(frame: pd.DataFrame, image_feature: str, molecular_feature: str) -> tuple[float, int]:
    work = frame.copy()
    x = pd.to_numeric(work[image_feature], errors="coerce")
    y = pd.to_numeric(work[molecular_feature], errors="coerce")
    mask = x.notna() & y.notna()
    work = work.loc[mask, :].copy()
    if len(work) < 6:
        return np.nan, int(len(work))
    ranked_x = pd.to_numeric(work[image_feature], errors="coerce").rank(method="average").to_numpy(dtype=float)
    ranked_y = pd.to_numeric(work[molecular_feature], errors="coerce").rank(method="average").to_numpy(dtype=float)
    controls = control_matrix(work)
    x_resid = residualize(ranked_x, controls)
    y_resid = residualize(ranked_y, controls)
    if np.nanstd(x_resid) == 0.0 or np.nanstd(y_resid) == 0.0:
        return np.nan, int(len(work))
    return float(np.corrcoef(x_resid, y_resid)[0, 1]), int(len(work))


def candidate_frame(frame: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    missing = [
        column
        for column in (candidate.image_feature, candidate.molecular_feature)
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Missing {candidate.dataset}/{candidate.model}/{candidate.program}: {missing}")
    work = frame.copy()
    if candidate.assigned_structure_filter and "assigned_structure" in work.columns:
        work = work.loc[work["assigned_structure"].astype(str).eq(candidate.assigned_structure_filter)].copy()
    keep = [
        "contour_id",
        "assigned_structure",
        "n_tiles",
        "n_cells",
        candidate.image_feature,
        candidate.molecular_feature,
    ]
    keep += [column for column in DEFAULT_CONTROLS if column in work.columns]
    return work.loc[:, list(dict.fromkeys([column for column in keep if column in work.columns]))].copy()


def apply_size_filter(frame: pd.DataFrame, min_tiles: int | None, min_cells: int | None) -> pd.DataFrame:
    work = frame.copy()
    if min_tiles is not None and "n_tiles" in work.columns:
        work = work.loc[pd.to_numeric(work["n_tiles"], errors="coerce").ge(min_tiles)].copy()
    if min_cells is not None and "n_cells" in work.columns:
        work = work.loc[pd.to_numeric(work["n_cells"], errors="coerce").ge(min_cells)].copy()
    return work


def run_contour_size_sensitivity(
    run_frames: dict[tuple[str, str], pd.DataFrame],
    candidates: list[Candidate],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.dataset, candidate.model)
        if key not in run_frames:
            continue
        unique_key = (
            candidate.dataset,
            candidate.model,
            candidate.program,
            candidate.image_feature,
            candidate.molecular_feature,
        )
        if unique_key in seen:
            continue
        seen.add(unique_key)
        base = candidate_frame(run_frames[key], candidate)
        for filter_name, min_tiles, min_cells in SENSITIVITY_FILTERS:
            filtered = apply_size_filter(base, min_tiles, min_cells)
            rho, n = partial_spearman(filtered, candidate.image_feature, candidate.molecular_feature)
            rows.append(
                {
                    "dataset": candidate.dataset,
                    "model": candidate.model.upper(),
                    "program": candidate.program,
                    "molecular_feature": candidate.molecular_feature,
                    "image_feature": candidate.image_feature,
                    "assigned_structure_filter": candidate.assigned_structure_filter or "",
                    "filter_name": filter_name,
                    "n_tiles_min": min_tiles if min_tiles is not None else "",
                    "n_cells_min": min_cells if min_cells is not None else "",
                    "n_contours": n,
                    "partial_spearman_rho": rho,
                    "abs_partial_spearman_rho": abs(rho) if math.isfinite(rho) else np.nan,
                    "reported_partial_rho": candidate.reported_partial_rho
                    if candidate.reported_partial_rho is not None
                    else "",
                }
            )
    return pd.DataFrame(rows)


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not math.isfinite(float(std)) or float(std) == 0.0:
        return numeric * np.nan
    return (numeric - numeric.mean()) / std


def build_breast_hero_candidates(
    frame: pd.DataFrame,
    *,
    min_tiles: int,
    min_cells: int,
    max_edge_fraction: float,
    area_quantiles: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecular = "program__wta_luminal_estrogen_response"
    image = "embedding__103__mean"
    s3 = frame.loc[frame["assigned_structure"].astype(str).eq("S3")].copy()
    s3["wta_program_z"] = zscore(s3[molecular])
    s3["oriented_he_embedding_z"] = -zscore(s3[image])
    s3["balanced_concordance_score"] = (
        np.minimum(s3["wta_program_z"].abs(), s3["oriented_he_embedding_z"].abs())
        + 0.25 * np.maximum(s3["wta_program_z"].abs(), s3["oriented_he_embedding_z"].abs())
    )
    s3["orientation_concordant"] = np.sign(s3["wta_program_z"]) == np.sign(s3["oriented_he_embedding_z"])
    s3["polarity"] = np.where(s3["wta_program_z"].ge(0), "high", "low")
    s3["qc_min_tiles_pass"] = pd.to_numeric(s3["n_tiles"], errors="coerce").ge(min_tiles)
    s3["qc_min_cells_pass"] = pd.to_numeric(s3["n_cells"], errors="coerce").ge(min_cells)
    if "cell_edge_proximity_fraction__lt_25um" in s3.columns:
        edge = pd.to_numeric(s3["cell_edge_proximity_fraction__lt_25um"], errors="coerce")
        s3["qc_edge_pass"] = edge.le(max_edge_fraction) | edge.isna()
    else:
        s3["qc_edge_pass"] = True
    area = pd.to_numeric(s3["area_image_px2"], errors="coerce")
    basic_pool = s3.loc[
        s3["qc_min_tiles_pass"]
        & s3["qc_min_cells_pass"]
        & s3["qc_edge_pass"]
        & s3["orientation_concordant"],
        "area_image_px2",
    ].pipe(pd.to_numeric, errors="coerce")
    lo_q, hi_q = area_quantiles
    area_low = float(basic_pool.quantile(lo_q))
    area_high = float(basic_pool.quantile(hi_q))
    s3["qc_area_pass"] = area.between(area_low, area_high, inclusive="both")
    s3["qc_filter_pass"] = (
        s3["qc_min_tiles_pass"]
        & s3["qc_min_cells_pass"]
        & s3["qc_edge_pass"]
        & s3["qc_area_pass"]
        & s3["orientation_concordant"]
    )
    candidates = s3.loc[s3["qc_filter_pass"]].copy()
    candidates = candidates.sort_values(["polarity", "balanced_concordance_score"], ascending=[True, False])
    selected_ids = {item for pair in BREAST_SELECTED_PAIRS for item in pair}
    selected = s3.loc[s3["contour_id"].astype(str).isin(selected_ids)].copy()
    if len(selected) != len(selected_ids):
        found = set(selected["contour_id"].astype(str))
        missing = sorted(selected_ids - found)
        raise ValueError(f"Missing selected hero contours: {missing}")
    failed = selected.loc[~selected["qc_filter_pass"], ["contour_id", "n_tiles", "n_cells", "cell_edge_proximity_fraction__lt_25um", "area_image_px2"]]
    if not failed.empty:
        raise ValueError("Selected hero contours failed QC:\n" + failed.to_string(index=False))
    return candidates, selected


def open_slide(path: Path):
    import tiffslide

    return tiffslide.TiffSlide(str(path))


def geometry_from_row(row: pd.Series) -> BaseGeometry:
    if "geometry_wkt" in row and pd.notna(row["geometry_wkt"]):
        return wkt.loads(str(row["geometry_wkt"]))
    if "geometry" in row and pd.notna(row["geometry"]):
        value = row["geometry"]
        if isinstance(value, BaseGeometry):
            return value
        return wkt.loads(str(value))
    raise ValueError(f"No geometry for {row.get('contour_id')}")


def iter_polygons(geometry: BaseGeometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def crop_bounds(
    geometry: BaseGeometry,
    *,
    patch_size: int,
    pad: int,
    max_patch_size: int,
) -> tuple[int, int, int, int, bool]:
    minx, miny, maxx, maxy = geometry.bounds
    width = max(int(math.ceil(maxx - minx)) + 2 * pad, patch_size)
    height = max(int(math.ceil(maxy - miny)) + 2 * pad, patch_size)
    truncated = False
    if max_patch_size > 0:
        new_width = min(width, max_patch_size)
        new_height = min(height, max_patch_size)
        truncated = (new_width != width) or (new_height != height)
        width, height = new_width, new_height
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    x0 = max(int(round(center_x - width / 2.0)), 0)
    y0 = max(int(round(center_y - height / 2.0)), 0)
    return x0, y0, int(width), int(height), truncated


def draw_geometry(image: Image.Image, geometry: BaseGeometry, *, x0: int, y0: int) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for polygon in iter_polygons(geometry):
        coords = [(float(x) - x0, float(y) - y0) for x, y in polygon.exterior.coords]
        if len(coords) >= 3:
            draw.polygon(coords, fill=(255, 35, 35, 34))
            draw.line(coords, fill=(215, 20, 35, 230), width=7, joint="curve")
        for interior in polygon.interiors:
            hole = [(float(x) - x0, float(y) - y0) for x, y in interior.coords]
            if len(hole) >= 2:
                draw.line(hole, fill=(255, 160, 64, 230), width=4, joint="curve")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)[:90]


def export_patches(
    selected: pd.DataFrame,
    contours: pd.DataFrame,
    *,
    wsi_path: Path,
    out_dir: Path,
    patch_size: int,
    pad: int,
    max_patch_size: int,
) -> pd.DataFrame:
    patch_dir = out_dir / "hero_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    contours = contours.copy()
    contours["contour_id"] = contours["contour_id"].astype(str)
    lookup = contours.set_index("contour_id", drop=False)
    slide = open_slide(wsi_path)
    selected_by_id = selected.set_index(selected["contour_id"].astype(str), drop=False)
    rows: list[dict[str, object]] = []
    rank = 0
    for pair_idx, (high_id, low_id) in enumerate(BREAST_SELECTED_PAIRS, start=1):
        for polarity, contour_id in (("high", high_id), ("low", low_id)):
            rank += 1
            record = selected_by_id.loc[contour_id]
            geometry = geometry_from_row(lookup.loc[contour_id])
            x0, y0, width, height, truncated = crop_bounds(
                geometry,
                patch_size=patch_size,
                pad=pad,
                max_patch_size=max_patch_size,
            )
            patch = slide.read_region((x0, y0), 0, (width, height)).convert("RGB")
            draw_geometry(patch, geometry, x0=x0, y0=y0)
            filename = f"hero_qc_{rank:02d}_{polarity}_{safe_name(contour_id)}.png"
            patch_path = patch_dir / filename
            patch.save(patch_path)
            rows.append(
                {
                    "pair": pair_idx,
                    "polarity": polarity,
                    "contour_id": contour_id,
                    "patch_file": str(patch_path),
                    "wta_program_z": float(record["wta_program_z"]),
                    "oriented_he_embedding_z": float(record["oriented_he_embedding_z"]),
                    "balanced_concordance_score": float(record["balanced_concordance_score"]),
                    "hidden_program_score": float(record["balanced_concordance_score"]),
                    "n_cells": int(record["n_cells"]),
                    "n_tiles": int(record["n_tiles"]),
                    "area_image_px2": float(record["area_image_px2"]),
                    "cell_edge_proximity_fraction__lt_25um": float(record.get("cell_edge_proximity_fraction__lt_25um", np.nan)),
                    "qc_filter": "n_tiles>=3;n_cells>=50;cell_edge_proximity_fraction__lt_25um<=0.75;area_q10_q90;orientation_concordant",
                    "truncated_to_max_patch_size": bool(truncated),
                    "crop_x0": x0,
                    "crop_y0": y0,
                    "crop_width": width,
                    "crop_height": height,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "Figure_1b_Hero_Patches_Source_Data.csv", index=False)
    return manifest


def fit_patch(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font_obj: ImageFont.ImageFont) -> None:
    x, y = xy
    lines = text.split("\n")
    line_heights = [draw.textbbox((0, 0), line, font=font_obj)[3] - draw.textbbox((0, 0), line, font=font_obj)[1] for line in lines]
    width = max(draw.textlength(line, font=font_obj) for line in lines)
    height = sum(line_heights) + 8 * (len(lines) - 1)
    draw.rounded_rectangle((x - 10, y - 8, x + int(width) + 10, y + height + 10), radius=8, fill=(0, 0, 0))
    cursor = y
    for line, line_h in zip(lines, line_heights, strict=True):
        draw.text((x, cursor), line, fill="white", font=font_obj)
        cursor += line_h + 8


def make_montage(
    manifest: pd.DataFrame,
    out_base: Path,
    *,
    pairs: int,
    patch_size: int,
    title: str,
    subtitle: str,
) -> None:
    selected = manifest.loc[manifest["pair"].le(pairs)].copy()
    margin = 46
    gutter = 32
    title_h = 118
    label_h = 132
    row_h = patch_size + label_h + 26
    width = margin * 2 + patch_size * 2 + gutter
    height = title_h + margin + row_h * pairs
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(32, bold=True)
    sub_font = font(18)
    header_font = font(22, bold=True)
    label_font = font(19, bold=True)
    small_font = font(17)
    overlay_font = font(16, bold=True)
    high_color = "#B42318"
    low_color = "#175CD3"
    black = "#202020"
    gray = "#4f4f4f"

    draw.text((margin, 24), title, fill=black, font=title_font)
    draw.text((margin, 66), subtitle, fill=gray, font=sub_font)
    draw.text((margin, title_h - 10), "High luminal ER", fill=high_color, font=header_font)
    draw.text((margin + patch_size + gutter, title_h - 10), "Low luminal ER", fill=low_color, font=header_font)

    for pair_idx in range(1, pairs + 1):
        y = title_h + margin + (pair_idx - 1) * row_h
        pair = selected.loc[selected["pair"].eq(pair_idx)].copy()
        for col, polarity in enumerate(("high", "low")):
            row = pair.loc[pair["polarity"].eq(polarity)].iloc[0]
            x = margin + col * (patch_size + gutter)
            patch = fit_patch(Path(str(row["patch_file"])), patch_size)
            canvas.paste(patch, (x, y))
            color = high_color if polarity == "high" else low_color
            draw.rectangle((x, y, x + patch_size - 1, y + patch_size - 1), outline=color, width=6)
            draw_text_box(
                draw,
                (x + 16, y + patch_size - 78),
                f"WTA {float(row['wta_program_z']):+.2f}\nH&E {float(row['oriented_he_embedding_z']):+.2f}",
                overlay_font,
            )
            label_y = y + patch_size + 13
            draw.text((x, label_y), str(row["contour_id"]), fill=black, font=label_font)
            draw.text(
                (x, label_y + 30),
                f"n_tiles = {int(row['n_tiles'])}; n_cells = {int(row['n_cells'])}",
                fill=black,
                font=small_font,
            )
            draw.text(
                (x, label_y + 57),
                f"balanced score = {float(row['balanced_concordance_score']):.2f}",
                fill=gray,
                font=small_font,
            )
        draw.text((12, y + patch_size / 2 - 14), f"{pair_idx}", fill="#666666", font=header_font)

    canvas.save(out_base.with_suffix(".png"))
    canvas.save(out_base.with_suffix(".pdf"), "PDF", resolution=300.0)


def write_report(out_dir: Path, manifest: pd.DataFrame, candidates: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    target = sensitivity[
        (sensitivity["dataset"].eq("breast"))
        & (sensitivity["model"].eq("PLIP"))
        & (sensitivity["program"].eq("luminal_estrogen_response"))
        & (sensitivity["filter_name"].eq("n_tiles_ge_3_and_n_cells_ge_50"))
    ]
    rho_text = "not available"
    if not target.empty:
        row = target.iloc[0]
        rho_text = f"{float(row['partial_spearman_rho']):.6f} (n = {int(row['n_contours'])})"
    lines = [
        "# Contour QC Hero Hardening Report",
        "",
        "This run does not rerun PLIP/UNI. It reads existing contour-level tables and H&E geometry.",
        "",
        f"- Selected hero rows: {len(manifest)}.",
        f"- QC-filtered breast S3 luminal candidates: {len(candidates)}.",
        f"- Breast S3 luminal ER rho after n_tiles>=3 and n_cells>=50: {rho_text}.",
        "",
        "## Selected Figure 1b/Supplementary Figure 2 contours",
        "",
    ]
    for _, row in manifest.iterrows():
        lines.append(
            f"- Pair {int(row['pair'])} {row['polarity']}: {row['contour_id']}; "
            f"n_tiles={int(row['n_tiles'])}, n_cells={int(row['n_cells'])}, "
            f"WTA z={float(row['wta_program_z']):+.2f}, H&E z={float(row['oriented_he_embedding_z']):+.2f}."
        )
    lines += [
        "",
        "## Files",
        "",
        "- `Figure_1b_Hero_Patches_Source_Data.csv`",
        "- `breast_s3_luminal_qc_candidates.csv`",
        "- `Supplementary_Table_9_ContourSizeSensitivity.csv`",
        "- `Figure_1b_QC_HeroPatch_2Pairs.png/pdf`",
        "- `Supplementary_Figure_2_QC_HeroPatch_4Pairs.png/pdf`",
    ]
    (out_dir / "Contour_QC_Hero_Hardening_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        ("breast", "plip"): args.breast_plip_run.expanduser().resolve(),
        ("breast", "uni"): args.breast_uni_run.expanduser().resolve(),
        ("cervical", "plip"): args.cervical_plip_run.expanduser().resolve(),
        ("cervical", "uni"): args.cervical_uni_run.expanduser().resolve(),
    }
    run_frames = {key: contour_table(path) for key, path in run_dirs.items()}
    candidates = read_candidates(args.candidates.expanduser().resolve())
    sensitivity = run_contour_size_sensitivity(run_frames, candidates)
    sensitivity.to_csv(out_dir / "Supplementary_Table_9_ContourSizeSensitivity.csv", index=False)

    breast_candidates, selected = build_breast_hero_candidates(
        run_frames[("breast", "plip")],
        min_tiles=args.min_tiles,
        min_cells=args.min_cells,
        max_edge_fraction=args.max_edge_fraction,
        area_quantiles=tuple(args.area_quantiles),
    )
    breast_candidates.to_csv(out_dir / "breast_s3_luminal_qc_candidates.csv", index=False)
    contour_geometry = image_contours(run_dirs[("breast", "plip")])
    manifest = export_patches(
        selected,
        contour_geometry,
        wsi_path=args.breast_wsi.expanduser().resolve(),
        out_dir=out_dir,
        patch_size=args.patch_size,
        pad=args.pad,
        max_patch_size=args.max_patch_size,
    )
    make_montage(
        manifest,
        out_dir / "Figure_1b_QC_HeroPatch_2Pairs",
        pairs=2,
        patch_size=args.montage_patch_size,
        title="Breast S3 QC-filtered luminal ER examples",
        subtitle="Representative contours; statistical claim uses all paired contours and size-filter sensitivity.",
    )
    make_montage(
        manifest,
        out_dir / "Supplementary_Figure_2_QC_HeroPatch_4Pairs",
        pairs=4,
        patch_size=args.montage_patch_size,
        title="Expanded breast S3 QC-filtered luminal ER examples",
        subtitle="All examples pass n_tiles>=3, n_cells>=50 and edge/area QC filters.",
    )
    write_report(out_dir, manifest, breast_candidates, sensitivity)
    print(f"Wrote contour QC hardening outputs to {out_dir}")
    print(manifest[["pair", "polarity", "contour_id", "n_tiles", "n_cells", "wta_program_z", "oriented_he_embedding_z"]].to_string(index=False))
    target = sensitivity.loc[
        (sensitivity["dataset"].eq("breast"))
        & (sensitivity["model"].eq("PLIP"))
        & (sensitivity["program"].eq("luminal_estrogen_response"))
    ]
    print(target[["filter_name", "n_contours", "partial_spearman_rho"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
