#!/usr/bin/env python3
"""Test candidate mTM associations under small registration perturbations.

The check does not rerun H&E encoders. It jitters tile-to-contour assignment by
small translations, rotations and scale changes, recomputes contour-level
candidate embedding means from existing tile embeddings, and then reruns the
same partial Spearman model used by the manuscript source data.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_spatial_sensitivity import Candidate, find_contour_table, partial_spearman, read_candidates, read_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return 0

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = find_contour_table(run_dir)
    tile_features = read_table(run_dir / "tile_features.parquet")
    tile_assignments = read_table(run_dir / "tile_assignments.parquet")
    contours = read_table(run_dir / "image_contours.parquet")
    candidates = read_candidates(args.candidates, dataset=args.dataset, model=args.model)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise SystemExit("No candidates matched dataset/model.")

    long = run_registration_perturbation(frame, tile_features, tile_assignments, contours, candidates)
    summary = summarize(long)
    long.to_csv(out_dir / "registration_perturbation_long.csv", index=False)
    summary.to_csv(out_dir / "registration_perturbation_summary.csv", index=False)
    write_report(out_dir / "registration_perturbation_report.md", summary)
    print(f"Wrote registration perturbation outputs to {out_dir}")
    print(summary.to_string(index=False))
    return 0


def run_registration_perturbation(
    frame: pd.DataFrame,
    tile_features: pd.DataFrame,
    tile_assignments: pd.DataFrame,
    contours: pd.DataFrame,
    candidates: list[Candidate],
) -> pd.DataFrame:
    tile_table = prepare_tile_table(tile_features, tile_assignments)
    x_col, y_col = infer_tile_xy(tile_table)
    contour_geometries = load_contour_geometries(contours)
    feature_map = {candidate.image_feature: tile_column_for_image_feature(candidate.image_feature, tile_table) for candidate in candidates}
    perturbations = perturbation_grid()
    center_x = float(pd.to_numeric(tile_table[x_col], errors="coerce").median())
    center_y = float(pd.to_numeric(tile_table[y_col], errors="coerce").median())

    base_rows = []
    for candidate in candidates:
        base_rho, base_n = partial_spearman(candidate_base_frame(frame, candidate), candidate.image_feature, candidate.molecular_feature)
        base_rows.append((candidate, base_rho, base_n))

    rows: list[dict[str, object]] = []
    for perturbation in perturbations:
        assigned = assign_tiles_under_perturbation(
            tile_table,
            contour_geometries,
            x_col=x_col,
            y_col=y_col,
            center_x=center_x,
            center_y=center_y,
            dx=float(perturbation["dx_px"]),
            dy=float(perturbation["dy_px"]),
            rotation_deg=float(perturbation["rotation_deg"]),
            scale=float(perturbation["scale"]),
        )
        reassigned_features = aggregate_tile_features(tile_table, assigned, feature_map)
        assignment_fraction = float(pd.Series(assigned).notna().mean()) if len(assigned) else np.nan
        perturbed_frame = frame.drop(columns=list(feature_map.keys()), errors="ignore").merge(
            reassigned_features,
            on="contour_id",
            how="left",
        )
        for candidate, base_rho, base_n in base_rows:
            work = candidate_base_frame(perturbed_frame, candidate)
            rho, n = partial_spearman(work, candidate.image_feature, candidate.molecular_feature)
            rows.append(
                {
                    **asdict(candidate),
                    **perturbation,
                    "base_partial_spearman_rho": base_rho,
                    "base_n_contours": base_n,
                    "perturbed_partial_spearman_rho": rho,
                    "perturbed_n_contours": n,
                    "delta_from_base": rho - base_rho if np.isfinite(rho) and np.isfinite(base_rho) else np.nan,
                    "sign_matches_base": sign_matches(rho, base_rho),
                    "assignment_fraction": assignment_fraction,
                    "n_assigned_tiles": int(pd.Series(assigned).notna().sum()),
                    "n_total_tiles": int(len(assigned)),
                }
            )
    return pd.DataFrame(rows)


def prepare_tile_table(tile_features: pd.DataFrame, tile_assignments: pd.DataFrame) -> pd.DataFrame:
    if {"tile_x", "tile_y"}.issubset(tile_features.columns):
        return tile_features.copy()
    keep = [column for column in ["tile_id", "tile_x", "tile_y"] if column in tile_assignments.columns]
    if len(keep) < 3:
        raise KeyError("tile_assignments must include tile_id, tile_x and tile_y.")
    return tile_features.merge(tile_assignments.loc[:, keep], on="tile_id", how="left")


def infer_tile_xy(frame: pd.DataFrame) -> tuple[str, str]:
    pairs = [
        ("tile_x", "tile_y"),
        ("tile_center_x", "tile_center_y"),
        ("center_x", "center_y"),
        ("x_center", "y_center"),
        ("centroid_x", "centroid_y"),
        ("x", "y"),
    ]
    for x_col, y_col in pairs:
        if x_col in frame.columns and y_col in frame.columns:
            return x_col, y_col
    if {"x_min", "x_max", "y_min", "y_max"}.issubset(frame.columns):
        frame["tile_x"] = (pd.to_numeric(frame["x_min"], errors="coerce") + pd.to_numeric(frame["x_max"], errors="coerce")) / 2.0
        frame["tile_y"] = (pd.to_numeric(frame["y_min"], errors="coerce") + pd.to_numeric(frame["y_max"], errors="coerce")) / 2.0
        return "tile_x", "tile_y"
    raise KeyError("Could not infer tile coordinate columns.")


def tile_column_for_image_feature(image_feature: str, tile_table: pd.DataFrame) -> str:
    candidates = [image_feature]
    if image_feature.endswith("__mean"):
        candidates.append(image_feature[: -len("__mean")])
    for column in candidates:
        if column in tile_table.columns:
            return column
    raise KeyError(f"No tile embedding column found for {image_feature!r}. Tried {candidates}.")


def load_contour_geometries(contours: pd.DataFrame) -> list[tuple[str, object]]:
    from shapely import wkt

    required = {"contour_id", "geometry_wkt"}
    if not required.issubset(contours.columns):
        raise KeyError("image_contours must include contour_id and geometry_wkt.")
    out = []
    for _, row in contours.iterrows():
        text = row.get("geometry_wkt")
        if not isinstance(text, str) or not text:
            continue
        geometry = wkt.loads(text)
        if geometry.is_empty:
            continue
        out.append((str(row["contour_id"]), geometry))
    if not out:
        raise ValueError("No valid contour geometries found.")
    return out


def assign_tiles_under_perturbation(
    tile_table: pd.DataFrame,
    contour_geometries: list[tuple[str, object]],
    *,
    x_col: str,
    y_col: str,
    center_x: float,
    center_y: float,
    dx: float,
    dy: float,
    rotation_deg: float,
    scale: float,
) -> list[str | None]:
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    contour_ids = [item[0] for item in contour_geometries]
    polygons = [item[1] for item in contour_geometries]
    areas = np.asarray([float(poly.area) for poly in polygons], dtype=float)
    tree = STRtree(polygons)
    geom_to_index = {id(geom): idx for idx, geom in enumerate(polygons)}
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xs = pd.to_numeric(tile_table[x_col], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(tile_table[y_col], errors="coerce").to_numpy(dtype=float)
    out: list[str | None] = []
    for x, y in zip(xs, ys, strict=False):
        if not np.isfinite(x) or not np.isfinite(y):
            out.append(None)
            continue
        centered_x = x - center_x
        centered_y = y - center_y
        px = center_x + scale * (cos_t * centered_x - sin_t * centered_y) + dx
        py = center_y + scale * (sin_t * centered_x + cos_t * centered_y) + dy
        point = Point(float(px), float(py))
        matches = []
        for hit in tree.query(point):
            idx = int(hit) if isinstance(hit, (int, np.integer)) else geom_to_index.get(id(hit), -1)
            if idx < 0:
                continue
            polygon = polygons[idx]
            if polygon.covers(point):
                matches.append(idx)
        if not matches:
            out.append(None)
            continue
        best = min(matches, key=lambda idx: areas[idx])
        out.append(contour_ids[best])
    return out


def aggregate_tile_features(
    tile_table: pd.DataFrame,
    assigned_contours: Iterable[str | None],
    feature_map: dict[str, str],
) -> pd.DataFrame:
    work = pd.DataFrame({"contour_id": list(assigned_contours)})
    for output_col, tile_col in feature_map.items():
        work[output_col] = pd.to_numeric(tile_table[tile_col], errors="coerce").to_numpy(dtype=float)
    work = work.dropna(subset=["contour_id"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["contour_id", *feature_map.keys()])
    return work.groupby("contour_id", as_index=False)[list(feature_map.keys())].mean()


def candidate_base_frame(frame: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    work = frame.copy()
    if candidate.assigned_structure_filter and "assigned_structure" in work.columns:
        work = work.loc[work["assigned_structure"].astype(str).eq(candidate.assigned_structure_filter)].copy()
    return work


def perturbation_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"perturbation": "identity", "dx_px": 0.0, "dy_px": 0.0, "rotation_deg": 0.0, "scale": 1.0}
    ]
    for dx in [-32, -16, -8, 8, 16, 32]:
        rows.append({"perturbation": f"shift_x_{dx:+d}px", "dx_px": float(dx), "dy_px": 0.0, "rotation_deg": 0.0, "scale": 1.0})
    for dy in [-32, -16, -8, 8, 16, 32]:
        rows.append({"perturbation": f"shift_y_{dy:+d}px", "dx_px": 0.0, "dy_px": float(dy), "rotation_deg": 0.0, "scale": 1.0})
    for rot in [-0.5, -0.25, 0.25, 0.5]:
        rows.append({"perturbation": f"rotate_{rot:+.2f}deg", "dx_px": 0.0, "dy_px": 0.0, "rotation_deg": rot, "scale": 1.0})
    for scale in [0.995, 1.005]:
        rows.append({"perturbation": f"scale_{scale:.3f}", "dx_px": 0.0, "dy_px": 0.0, "rotation_deg": 0.0, "scale": scale})
    for dx, dy, rot in [(8, 8, 0.25), (-8, 8, -0.25), (8, -8, 0.25), (-8, -8, -0.25)]:
        rows.append(
            {
                "perturbation": f"combo_dx{dx:+d}_dy{dy:+d}_rot{rot:+.2f}",
                "dx_px": float(dx),
                "dy_px": float(dy),
                "rotation_deg": float(rot),
                "scale": 1.0,
            }
        )
    return rows


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in long.groupby(["dataset", "model", "program", "molecular_feature", "image_feature"], sort=False):
        finite_delta = pd.to_numeric(group["delta_from_base"], errors="coerce").dropna()
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "program": keys[2],
                "molecular_feature": keys[3],
                "image_feature": keys[4],
                "perturbations": int(group["perturbation"].nunique()),
                "base_partial_spearman_rho": float(group["base_partial_spearman_rho"].iloc[0]),
                "median_perturbed_partial_spearman_rho": float(pd.to_numeric(group["perturbed_partial_spearman_rho"], errors="coerce").median()),
                "min_perturbed_partial_spearman_rho": float(pd.to_numeric(group["perturbed_partial_spearman_rho"], errors="coerce").min()),
                "max_perturbed_partial_spearman_rho": float(pd.to_numeric(group["perturbed_partial_spearman_rho"], errors="coerce").max()),
                "median_abs_delta_from_base": float(finite_delta.abs().median()) if not finite_delta.empty else np.nan,
                "max_abs_delta_from_base": float(finite_delta.abs().max()) if not finite_delta.empty else np.nan,
                "sign_stability_fraction": float(pd.Series(group["sign_matches_base"]).mean()),
                "min_assignment_fraction": float(pd.to_numeric(group["assignment_fraction"], errors="coerce").min()),
                "median_perturbed_n_contours": float(pd.to_numeric(group["perturbed_n_contours"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def sign_matches(value: float, reference: float) -> bool:
    if not np.isfinite(value) or not np.isfinite(reference) or value == 0.0 or reference == 0.0:
        return False
    return bool(np.sign(value) == np.sign(reference))


def write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Registration Perturbation Report",
        "",
        "Candidate embedding means were recomputed after small tile-coordinate translations, rotations and scale perturbations. The analysis uses existing A100 tile embeddings and contour polygons; it does not rerun H&E encoders.",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "- {dataset} / {model} / {program}: base rho {base:.3f}, median perturbed rho {median:.3f}, max |delta| {delta:.3f}, sign stability {stable:.0%}, min assignment fraction {assign:.1%}.".format(
                dataset=row["dataset"],
                model=str(row["model"]).upper(),
                program=row["program"],
                base=float(row["base_partial_spearman_rho"]),
                median=float(row["median_perturbed_partial_spearman_rho"]),
                delta=float(row["max_abs_delta_from_base"]),
                stable=float(row["sign_stability_fraction"]),
                assign=float(row["min_assignment_fraction"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_self_test() -> None:
    from shapely.geometry import box

    contour_rows = []
    summary_rows = []
    tile_rows = []
    assignment_rows = []
    for idx in range(25):
        x0 = (idx % 5) * 100.0
        y0 = (idx // 5) * 100.0
        contour_id = f"C{idx}"
        contour_rows.append({"contour_id": contour_id, "geometry_wkt": box(x0, y0, x0 + 80, y0 + 80).wkt})
        signal = float(idx)
        summary_rows.append(
            {
                "contour_id": contour_id,
                "assigned_structure": "S",
                "centroid_x": x0 + 40,
                "centroid_y": y0 + 40,
                "embedding__1__mean": signal,
                "program__wta_test": signal,
            }
        )
        for j in range(4):
            tile_id = idx * 10 + j
            tile_rows.append({"tile_id": tile_id, "embedding__1": signal})
            assignment_rows.append({"tile_id": tile_id, "tile_x": x0 + 20 + j * 5, "tile_y": y0 + 20 + j * 5})
    candidate = Candidate("synthetic", "plip", "test", "program__wta_test", "embedding__1__mean", "S", 1.0)
    long = run_registration_perturbation(
        pd.DataFrame(summary_rows),
        pd.DataFrame(tile_rows),
        pd.DataFrame(assignment_rows),
        pd.DataFrame(contour_rows),
        [candidate],
    )
    summary = summarize(long)
    stability = float(summary.iloc[0]["sign_stability_fraction"])
    if stability < 1.0:
        raise SystemExit(f"Self-test failed: sign stability {stability}")
    print("Self-test passed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
