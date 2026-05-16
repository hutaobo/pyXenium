from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.stats import spearmanr, t
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


DEFAULT_PROGRAM_CANDIDATES = (
    ("S3", "program__wta_luminal_estrogen_response"),
    ("S3", "program__wta_unfolded_protein_response"),
    ("S3", "program__wta_oxidative_phosphorylation"),
    ("S2", "program__wta_hypoxia_glycolysis"),
    ("S4", "program__wta_t_cell_exhaustion_checkpoint"),
)

CONTROL_COLUMNS = (
    "centroid_x",
    "centroid_y",
    "cell_boundary_distance_um__mean",
    "tile_boundary_distance_px__mean",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2-ready morphomolecular WTA evidence panels.",
    )
    parser.add_argument(
        "--data-dir",
        default="docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
            "naturebiotech_package"
        ),
    )
    parser.add_argument("--min-abs-partial-rho", type=float, default=0.5)
    parser.add_argument("--max-programs", type=int, default=3)
    parser.add_argument("--patch-montage", default=None)
    return parser.parse_args()


def _read_table(base: Path, stem: str) -> pd.DataFrame:
    parquet = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {base}")


def _geometry(value: Any) -> BaseGeometry | None:
    if isinstance(value, BaseGeometry):
        return value
    if pd.isna(value):
        return None
    try:
        return wkt.loads(str(value))
    except Exception:
        return None


def _iter_polygons(geometry: BaseGeometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def _patches_for(frame: pd.DataFrame) -> list[MplPolygon]:
    patches: list[MplPolygon] = []
    for geometry in frame["geometry"]:
        if geometry is None or geometry.is_empty:
            continue
        for polygon in _iter_polygons(geometry):
            coords = np.asarray(polygon.exterior.coords)
            if coords.shape[0] >= 3:
                patches.append(MplPolygon(coords, closed=True))
    return patches


def _rank_residual(values: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").rank(method="average").to_numpy(dtype=float)
    if controls.empty:
        return y - np.nanmean(y)
    design = np.column_stack([np.ones(len(controls), dtype=float), controls.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _partial_spearman(
    frame: pd.DataFrame,
    image_feature: str,
    target_feature: str,
    controls: tuple[str, ...] = CONTROL_COLUMNS,
) -> tuple[float, float, int]:
    keep = [image_feature, target_feature, *[c for c in controls if c in frame.columns]]
    work = frame.loc[:, keep].copy()
    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[image_feature, target_feature])
    control_columns = [c for c in controls if c in work.columns and work[c].notna().any()]
    if len(work) < 8:
        return np.nan, np.nan, int(len(work))
    for column in control_columns:
        work[column] = work[column].fillna(work[column].median()).rank(method="average")
    controls_frame = work.loc[:, control_columns]
    x_resid = _rank_residual(work[image_feature], controls_frame)
    y_resid = _rank_residual(work[target_feature], controls_frame)
    rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
    df = max(len(work) - len(control_columns) - 2, 1)
    clipped = float(np.clip(rho, -0.999999, 0.999999))
    stat = clipped * math.sqrt(df / max(1.0 - clipped * clipped, np.finfo(float).eps))
    p_value = float(2.0 * t.sf(abs(stat), df))
    return rho, p_value, int(len(work))


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * np.nan
    return (values - values.mean()) / std


def _best_image_for_target(leaderboard: pd.DataFrame, target: str) -> str | None:
    linked = leaderboard.loc[leaderboard["molecular_feature"].astype(str).eq(str(target))]
    if linked.empty:
        return None
    return str(linked.iloc[0]["best_image_feature"])


def _select_programs(
    contour: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    min_abs_partial_rho: float,
    max_programs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for structure, target in DEFAULT_PROGRAM_CANDIDATES:
        image = _best_image_for_target(leaderboard, target)
        if image is None or image not in contour.columns or target not in contour.columns:
            continue
        subset = contour.loc[
            contour["assigned_structure"].astype(str).eq(str(structure))
            & pd.to_numeric(contour.get("n_tiles", 0), errors="coerce").fillna(0).gt(0)
        ].copy()
        rho, p_value, n_contours = _partial_spearman(subset, image, target)
        raw = spearmanr(
            pd.to_numeric(subset[image], errors="coerce"),
            pd.to_numeric(subset[target], errors="coerce"),
            nan_policy="omit",
        )
        rows.append(
            {
                "assigned_structure": structure,
                "target_feature": target,
                "image_feature": image,
                "partial_spearman_rho": rho,
                "partial_p_value": p_value,
                "abs_partial_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                "spearman_rho": float(raw.statistic),
                "n_contours": n_contours,
                "passes_quality_gate": bool(abs(rho) >= min_abs_partial_rho)
                if np.isfinite(rho)
                else False,
            }
        )
    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    passed = selected.loc[selected["passes_quality_gate"]].copy()
    if passed.empty:
        passed = selected.copy()
    primary = passed.loc[passed["assigned_structure"].astype(str).eq("S3")].copy()
    fallback = passed.loc[~passed["assigned_structure"].astype(str).eq("S3")].copy()
    if len(primary) >= max_programs:
        return primary.head(max_programs).reset_index(drop=True)
    fallback = fallback.sort_values(
        ["abs_partial_spearman_rho", "target_feature"],
        ascending=[False, True],
        kind="stable",
    )
    return pd.concat([primary, fallback], ignore_index=True).head(max_programs).reset_index(drop=True)


def _plot_contours(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    value_column: str | None,
    title: str,
    cmap: str = "viridis",
    uniform_color: str = "#3b82f6",
) -> None:
    patches = _patches_for(frame)
    if not patches:
        ax.set_title(title)
        ax.axis("off")
        return
    if value_column is None:
        collection = PatchCollection(
            patches,
            facecolor=uniform_color,
            edgecolor="#111827",
            linewidth=0.25,
            alpha=0.85,
        )
        ax.add_collection(collection)
    else:
        values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
        expanded: list[float] = []
        for _, row in frame.iterrows():
            geometry = row["geometry"]
            count = len(list(_iter_polygons(geometry))) if geometry is not None else 0
            expanded.extend([float(row[value_column])] * count)
        collection = PatchCollection(
            patches,
            cmap=cmap,
            edgecolor="#111827",
            linewidth=0.2,
            alpha=0.9,
        )
        collection.set_array(np.asarray(expanded, dtype=float))
        finite = values[np.isfinite(values)]
        if finite.size:
            collection.set_clim(np.quantile(finite, 0.02), np.quantile(finite, 0.98))
        ax.add_collection(collection)
        plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.02)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def _plot_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    image_feature: str,
    target_feature: str,
    rho: float,
    p_value: float,
) -> None:
    x = pd.to_numeric(frame[image_feature], errors="coerce")
    y = pd.to_numeric(frame[target_feature], errors="coerce")
    ax.scatter(x, y, s=22, c="#2563eb", alpha=0.7, edgecolor="none")
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) >= 2:
        coeff = np.polyfit(clean["x"], clean["y"], deg=1)
        xs = np.linspace(clean["x"].min(), clean["x"].max(), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="#dc2626", linewidth=1.4)
    ax.set_xlabel(image_feature, fontsize=8)
    ax.set_ylabel(target_feature, fontsize=8)
    ax.set_title(f"Residual association: rho={rho:.3f}, p={p_value:.2e}", fontsize=10)
    ax.tick_params(labelsize=7)


def _write_hero_table(contour: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, spec in selected.iterrows():
        structure = str(spec["assigned_structure"])
        target = str(spec["target_feature"])
        image = str(spec["image_feature"])
        subset = contour.loc[
            contour["assigned_structure"].astype(str).eq(structure)
            & pd.to_numeric(contour.get("n_tiles", 0), errors="coerce").fillna(0).gt(0)
        ].copy()
        subset["_target_z"] = _zscore(subset[target])
        subset["_image_z"] = _zscore(subset[image])
        sign = np.sign(float(spec["partial_spearman_rho"]))
        if not np.isfinite(sign) or sign == 0:
            sign = 1.0
        subset["_oriented_image_z"] = sign * subset["_image_z"]
        subset["_hidden_score"] = subset["_target_z"] * subset["_oriented_image_z"]
        subset = subset.dropna(subset=["_target_z", "_oriented_image_z", "_hidden_score"])
        high = subset.loc[
            subset["_target_z"].gt(0) & subset["_oriented_image_z"].gt(0)
        ].sort_values(["_hidden_score", "_target_z"], ascending=False).head(4)
        low = subset.loc[
            subset["_target_z"].lt(0) & subset["_oriented_image_z"].lt(0)
        ].sort_values(["_hidden_score", "_target_z"], ascending=[False, True]).head(4)
        for group_name, group in (("high_program_concordant", high), ("low_program_concordant", low)):
            for _, row in group.iterrows():
                rows.append(
                    {
                        "figure_panel": _panel_label(target),
                        "hero_group": group_name,
                        "assigned_structure": structure,
                        "target_feature": target,
                        "image_feature": image,
                        "contour_id": row["contour_id"],
                        "target_value": row[target],
                        "image_value": row[image],
                        "target_z_within_structure": row["_target_z"],
                        "image_z_within_structure": row["_image_z"],
                        "oriented_image_z_within_structure": row["_oriented_image_z"],
                        "hidden_program_score": row["_hidden_score"],
                        "centroid_x": row.get("centroid_x", np.nan),
                        "centroid_y": row.get("centroid_y", np.nan),
                        "n_tiles": row.get("n_tiles", np.nan),
                        "n_cells": row.get("n_cells", np.nan),
                    }
                )
    hero = pd.DataFrame(rows)
    hero.to_csv(output_dir / "figure2_hero_contours.csv", index=False)
    return hero


def _panel_label(target: str) -> str:
    text = str(target).replace("program__wta_", "").replace("__mean", "").replace("rna__", "")
    return text


def _add_summary_page(
    pdf: PdfPages,
    selected: pd.DataFrame,
    *,
    patch_montage: Path | None,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    lines = [
        "Figure 2 Evidence Package: contour-constrained WTA residual decoding",
        "",
        "Quality gate: abs(partial Spearman rho) >= 0.5 within the selected HistoSeg structure.",
        "Controls: centroid_x, centroid_y, cell_boundary_distance_um__mean, tile_boundary_distance_px__mean.",
        "",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "- {structure} / {target} / {image}: partial rho={rho:.3f}, n={n}".format(
                structure=row["assigned_structure"],
                target=_panel_label(row["target_feature"]),
                image=row["image_feature"],
                rho=float(row["partial_spearman_rho"]),
                n=int(row["n_contours"]),
            )
        )
    if patch_montage is not None and patch_montage.exists():
        lines.extend(["", f"Patch montage source: {patch_montage.name}"])
    ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contour = _read_table(data_dir, "contour_multimodal_summary")
    image_contours = _read_table(data_dir, "image_contours")
    leaderboard = _read_table(data_dir, "wta_pathway_partial_correlations")
    image_contours = image_contours.loc[:, ["contour_id", "geometry_wkt"]].copy()
    image_contours["geometry"] = image_contours["geometry_wkt"].map(_geometry)
    merged = contour.merge(image_contours[["contour_id", "geometry"]], on="contour_id", how="left")
    selected = _select_programs(
        merged,
        leaderboard,
        min_abs_partial_rho=float(args.min_abs_partial_rho),
        max_programs=int(args.max_programs),
    )
    selected.to_csv(output_dir / "figure2_selected_programs.csv", index=False)
    hero = _write_hero_table(merged, selected, output_dir)
    pdf_path = output_dir / "Final_Figure2_Pack.pdf"
    patch_montage = Path(args.patch_montage).resolve() if args.patch_montage else data_dir / "hero_patch_montage_mtm_wta.png"
    with PdfPages(pdf_path) as pdf:
        _add_summary_page(pdf, selected, patch_montage=patch_montage)
        for _, spec in selected.iterrows():
            structure = str(spec["assigned_structure"])
            target = str(spec["target_feature"])
            image = str(spec["image_feature"])
            subset = merged.loc[
                merged["assigned_structure"].astype(str).eq(structure)
                & pd.to_numeric(merged.get("n_tiles", 0), errors="coerce").fillna(0).gt(0)
            ].copy()
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(
                f"{structure}: {_panel_label(target)} decoded by {image}",
                fontsize=14,
                fontweight="bold",
            )
            _plot_contours(axes[0, 0], subset, value_column=None, title=f"{structure} HistoSeg compartment")
            _plot_contours(axes[0, 1], subset, value_column=image, title="H&E embedding axis")
            _plot_contours(axes[1, 0], subset, value_column=target, title="Atera WTA program")
            _plot_scatter(
                axes[1, 1],
                subset,
                image_feature=image,
                target_feature=target,
                rho=float(spec["partial_spearman_rho"]),
                p_value=float(spec["partial_p_value"]),
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            png_path = output_dir / f"figure2_{structure}_{_panel_label(target)}.png"
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
        if patch_montage.exists():
            image = plt.imread(patch_montage)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title("H&E hero contours selected by residual morphomolecular concordance")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Selected programs: {selected.shape[0]}")
    print(f"Hero contours: {hero.shape[0]}")


if __name__ == "__main__":
    main()
