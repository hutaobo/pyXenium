#!/usr/bin/env python3
"""Audit component genes behind candidate mTM WTA programs.

This script uses the Xenium cell-feature matrix plus HistoSeg contour geometry
to recompute contour-level means for the genes inside selected WTA programs.
It then asks whether the same H&E embedding axis reported for a program also
tracks the program's component genes after the usual contour/spatial controls.

The output is a biological sanity check, not an independent wet-lab validation:
the WTA program scores are constructed from these genes, so the useful question
is whether the image axis carries gene-level signal in the expected direction.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONTROLS = (
    "assigned_structure",
    "centroid_x",
    "centroid_y",
    "cell_boundary_distance_um__mean",
    "tile_boundary_distance_px__mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Xenium export root.")
    parser.add_argument("--contour-geojson", required=True, help="HistoSeg contour GeoJSON.")
    parser.add_argument("--contour-multimodal", required=True, help="Full contour_multimodal_summary parquet/csv.")
    parser.add_argument("--candidates", required=True, help="Candidate source-data CSV.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="breast")
    parser.add_argument("--model", default="plip")
    parser.add_argument("--contour-key", default="histoseg_structures")
    parser.add_argument("--contour-id-key", default="name")
    parser.add_argument("--coordinate-space", default="xenium_pixel")
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--repo-src", default=None, help="Optional pyXenium src path for remote runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repo_src:
        sys.path.insert(0, str(Path(args.repo_src).expanduser()))

    from pyXenium.contour import add_contours_from_geojson
    from pyXenium.contour._analysis import _prepare_contours
    from pyXenium.io import read_xenium
    from pyXenium.multimodal.histoseg_lazyslide import (
        DEFAULT_WTA_GENE_PROGRAMS,
        _cell_indices_within_geometry,
        _normalized_expression_frame,
    )
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(Path(args.candidates), dataset=args.dataset, model=args.model)
    if not candidates:
        raise SystemExit("No candidates matched the requested dataset/model.")
    programs = [candidate["program"] for candidate in candidates]
    genes = unique_gene_list(DEFAULT_WTA_GENE_PROGRAMS, programs)

    slide = read_xenium(
        str(Path(args.dataset_root).expanduser()),
        as_="slide",
        prefer="h5",
        include_transcripts=False,
        include_boundaries=True,
        include_images=False,
    )
    add_contours_from_geojson(
        slide,
        args.contour_geojson,
        key=args.contour_key,
        id_key=args.contour_id_key,
        coordinate_space=args.coordinate_space,
        pixel_size_um=args.pixel_size_um,
        extract_he_patches=False,
    )
    contour_table = _prepare_contours(
        sdata=slide,
        contour_key=args.contour_key,
        contour_query=None,
    )

    gene_means = build_contour_gene_means(
        slide=slide,
        contour_table=contour_table,
        genes=genes,
        normalized_expression_frame=_normalized_expression_frame,
        cell_indices_within_geometry=_cell_indices_within_geometry,
        point_cls=Point,
        tree_cls=STRtree,
    )
    multimodal = read_table(Path(args.contour_multimodal))
    merged = multimodal.merge(gene_means, on="contour_id", how="left", suffixes=("", "__gene"))

    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for candidate in candidates:
        program = candidate["program"]
        program_feature = candidate["molecular_feature"]
        image_feature = candidate["image_feature"]
        reported_rho = candidate["reported_partial_rho"]
        program_genes = [gene for gene in DEFAULT_WTA_GENE_PROGRAMS[program] if gene in gene_means.columns]
        work = merged.copy()
        structure_filter = candidate.get("assigned_structure_filter")
        if structure_filter and "assigned_structure" in work.columns:
            work = work.loc[work["assigned_structure"].astype(str).eq(structure_filter)].copy()
        required = [program_feature, image_feature, *program_genes]
        missing = [column for column in required if column not in work.columns]
        if missing:
            raise KeyError(f"Missing columns for {program}: {missing}")

        q25 = pd.to_numeric(work[program_feature], errors="coerce").quantile(0.25)
        q75 = pd.to_numeric(work[program_feature], errors="coerce").quantile(0.75)
        low = work.loc[pd.to_numeric(work[program_feature], errors="coerce").le(q25), :]
        high = work.loc[pd.to_numeric(work[program_feature], errors="coerce").ge(q75), :]

        gene_rhos = []
        gene_ns = []
        image_sign_matches = []
        program_deltas = []
        for gene in program_genes:
            rho, n = partial_spearman(work, image_feature, gene)
            gene_ns.append(n)
            high_mean = float(pd.to_numeric(high[gene], errors="coerce").mean())
            low_mean = float(pd.to_numeric(low[gene], errors="coerce").mean())
            delta = high_mean - low_mean
            gene_rhos.append(rho)
            program_deltas.append(delta)
            sign_match = sign_matches(rho, reported_rho)
            image_sign_matches.append(sign_match)
            long_rows.append(
                {
                    "dataset": candidate["dataset"],
                    "model": candidate["model"],
                    "program": program,
                    "image_feature": image_feature,
                    "gene": gene,
                    "n_contours": n,
                    "image_gene_partial_spearman_rho": rho,
                    "reported_program_image_partial_rho": reported_rho,
                    "image_gene_sign_matches_program_axis": sign_match,
                    "program_top_quartile_gene_mean": high_mean,
                    "program_bottom_quartile_gene_mean": low_mean,
                    "program_top_minus_bottom_gene_mean": delta,
                }
            )

        finite_rhos = [value for value in gene_rhos if np.isfinite(value)]
        summary_rows.append(
            {
                "dataset": candidate["dataset"],
                "model": candidate["model"],
                "program": program,
                "assigned_structure_filter": structure_filter or "",
                "n_contours_in_filter": int(work["contour_id"].nunique()),
                "min_effective_image_gene_n": int(np.nanmin(gene_ns)) if gene_ns else 0,
                "median_effective_image_gene_n": float(np.nanmedian(gene_ns)) if gene_ns else np.nan,
                "n_program_genes_found": len(program_genes),
                "reported_program_image_partial_rho": reported_rho,
                "median_image_gene_partial_spearman_rho": float(np.nanmedian(finite_rhos))
                if finite_rhos
                else np.nan,
                "image_gene_sign_match_fraction": mean_bool(image_sign_matches),
                "program_top_minus_bottom_positive_fraction": mean_bool([delta > 0 for delta in program_deltas]),
                "strongest_component_gene_by_abs_rho": strongest_gene(long_rows, program),
            }
        )

    long = pd.DataFrame(long_rows)
    summary = pd.DataFrame(summary_rows)
    long_path = out_dir / "gene_component_validation_long.csv"
    summary_path = out_dir / "gene_component_validation_summary.csv"
    report_path = out_dir / "gene_component_validation_report.md"
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, summary, long)
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


def read_candidates(path: Path, *, dataset: str, model: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != dataset or row.get("model") != model:
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "program": row["program"],
                    "molecular_feature": row["molecular_feature"],
                    "image_feature": row["image_feature"],
                    "assigned_structure_filter": row.get("assigned_structure_filter") or "",
                    "reported_partial_rho": to_float(row.get("reported_partial_rho")),
                }
            )
    return rows


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path}")


def unique_gene_list(program_library: dict[str, Sequence[str]], programs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for program in programs:
        if program not in program_library:
            raise KeyError(f"Program {program!r} not found in DEFAULT_WTA_GENE_PROGRAMS.")
        for gene in program_library[program]:
            if gene not in seen:
                out.append(gene)
                seen.add(gene)
    return out


def build_contour_gene_means(
    *,
    slide: object,
    contour_table: pd.DataFrame,
    genes: Sequence[str],
    normalized_expression_frame: object,
    cell_indices_within_geometry: object,
    point_cls: object,
    tree_cls: object,
) -> pd.DataFrame:
    adata = slide.table  # pyXenium XeniumSlide
    expression = normalized_expression_frame(adata, genes)
    if expression.empty:
        raise ValueError("No requested program genes were found in the Xenium matrix.")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    cell_points = [point_cls(float(x), float(y)) for x, y in coords[:, :2]]
    point_tree = tree_cls(cell_points) if cell_points else None
    rows: list[dict[str, object]] = []
    for _, contour in contour_table.iterrows():
        indices = cell_indices_within_geometry(
            contour["geometry"],
            cell_points=cell_points,
            point_tree=point_tree,
        )
        row: dict[str, object] = {"contour_id": str(contour["contour_id"]), "n_cells": int(len(indices))}
        if len(indices):
            means = expression.iloc[indices, :].mean(axis=0)
            row.update({str(gene): float(value) for gene, value in means.items()})
        rows.append(row)
    return pd.DataFrame(rows)


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


def partial_spearman(frame: pd.DataFrame, image_feature: str, gene: str) -> tuple[float, int]:
    work = frame.copy()
    x = pd.to_numeric(work[image_feature], errors="coerce")
    y = pd.to_numeric(work[gene], errors="coerce")
    mask = x.notna() & y.notna()
    work = work.loc[mask, :].copy()
    if len(work) < 6:
        return np.nan, int(len(work))
    ranked_x = pd.to_numeric(work[image_feature], errors="coerce").rank(method="average").to_numpy(dtype=float)
    ranked_y = pd.to_numeric(work[gene], errors="coerce").rank(method="average").to_numpy(dtype=float)
    controls = control_matrix(work)
    x_resid = residualize(ranked_x, controls)
    y_resid = residualize(ranked_y, controls)
    if np.nanstd(x_resid) == 0.0 or np.nanstd(y_resid) == 0.0:
        return np.nan, int(len(work))
    return float(np.corrcoef(x_resid, y_resid)[0, 1]), int(len(work))


def to_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def sign_matches(value: float, reference: float) -> bool:
    if not np.isfinite(value) or not np.isfinite(reference) or value == 0.0 or reference == 0.0:
        return False
    return bool(np.sign(value) == np.sign(reference))


def mean_bool(values: Sequence[bool]) -> float:
    if not values:
        return np.nan
    return float(np.mean(np.asarray(values, dtype=float)))


def strongest_gene(long_rows: list[dict[str, object]], program: str) -> str:
    rows = [row for row in long_rows if row["program"] == program]
    finite = [
        row
        for row in rows
        if np.isfinite(float(row["image_gene_partial_spearman_rho"]))
    ]
    if not finite:
        return ""
    best = max(finite, key=lambda row: abs(float(row["image_gene_partial_spearman_rho"])))
    return str(best["gene"])


def write_report(path: Path, summary: pd.DataFrame, long: pd.DataFrame) -> None:
    lines = [
        "# Gene Component Validation Report",
        "",
        "This report audits whether candidate H&E embedding axes track component genes from the WTA programs used in mTM. It is a component-gene sanity check, not independent protein or IHC validation.",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "- {dataset} / {model} / {program}: {n_genes} genes, "
            "median effective image-gene n {n_eff:.0f}, "
            "median image-gene partial rho {median:.3f}, "
            "image-gene sign-match fraction {match:.2f}, "
            "program high-low positive fraction {contrast:.2f}; strongest gene {gene}.".format(
                dataset=row["dataset"],
                model=str(row["model"]).upper(),
                program=row["program"],
                n_genes=int(row["n_program_genes_found"]),
                n_eff=float(row["median_effective_image_gene_n"]),
                median=float(row["median_image_gene_partial_spearman_rho"]),
                match=float(row["image_gene_sign_match_fraction"]),
                contrast=float(row["program_top_minus_bottom_positive_fraction"]),
                gene=row["strongest_component_gene_by_abs_rho"],
            )
        )
    lines.extend(["", "## Component Genes", ""])
    for program, group in long.groupby("program", sort=False):
        gene_bits = []
        for _, row in group.sort_values("image_gene_partial_spearman_rho", key=lambda s: s.abs(), ascending=False).iterrows():
            gene_bits.append(f"{row['gene']} ({float(row['image_gene_partial_spearman_rho']):.3f})")
        lines.append(f"- {program}: " + ", ".join(gene_bits))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
