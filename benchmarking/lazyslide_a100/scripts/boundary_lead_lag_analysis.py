from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BOUNDARY_COLUMNS = (
    "cell_boundary_distance_um__mean",
    "cell_boundary_distance_um__p10",
    "tile_boundary_distance_px__mean",
    "tile_boundary_distance_px__p10",
    "cell_edge_proximity_fraction__lt_25um",
    "tile_edge_proximity_fraction__lt_256px",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute provisional MAZ boundary lead-lag metrics from contour-scale mTM outputs.",
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
    parser.add_argument("--top-programs", type=int, default=5)
    parser.add_argument("--slide-mpp", type=float, default=0.2738)
    return parser.parse_args()


def _read_table(base: Path, stem: str) -> pd.DataFrame:
    parquet = base / f"{stem}.parquet"
    csv = base / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet or {stem}.csv in {base}")


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * np.nan
    return (values - values.mean()) / std


def _corr(frame: pd.DataFrame, left: str, right: str) -> tuple[float, float, int]:
    work = frame.loc[:, [left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 8:
        return np.nan, np.nan, int(len(work))
    result = spearmanr(work[left], work[right], nan_policy="omit")
    return float(result.statistic), float(result.pvalue), int(len(work))


def _quartile_shift_um(
    frame: pd.DataFrame,
    feature: str,
    distance_column: str,
    *,
    px_to_um: float | None = None,
) -> float:
    work = frame.loc[:, [feature, distance_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 12:
        return np.nan
    low_cut = work[feature].quantile(0.25)
    high_cut = work[feature].quantile(0.75)
    low = work.loc[work[feature] <= low_cut, distance_column]
    high = work.loc[work[feature] >= high_cut, distance_column]
    if low.empty or high.empty:
        return np.nan
    shift = float(high.mean() - low.mean())
    if px_to_um is not None:
        shift *= float(px_to_um)
    return shift


def _lead_lag_class(molecular_edge: float, image_edge: float) -> str:
    if not np.isfinite(molecular_edge) or not np.isfinite(image_edge):
        return "insufficient_boundary_signal"
    diff = abs(molecular_edge) - abs(image_edge)
    if diff > 0.15:
        return "molecular_lead"
    if diff < -0.15:
        return "morphology_lead"
    if abs(molecular_edge) > 0.25 or abs(image_edge) > 0.25:
        return "coupled_boundary_zone"
    return "weak_boundary_zone"


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _morphological_lag_count(
    frame: pd.DataFrame,
    *,
    target: str,
    image: str,
    association_sign: float,
) -> int:
    target_z = _zscore(frame[target])
    image_z = association_sign * _zscore(frame[image])
    lag = target_z.gt(1.0) & image_z.lt(0.25)
    return int(lag.sum())


def _program_label(feature: Any) -> str:
    text = str(feature)
    text = text.replace("program__wta_", "")
    if text.endswith("__mean"):
        text = text[: -len("__mean")]
    return text


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contour = _read_table(data_dir, "contour_multimodal_summary")
    leaderboard = _read_table(data_dir, "wta_pathway_partial_correlations")
    candidates = leaderboard.head(int(args.top_programs)).copy()
    rows: list[dict[str, Any]] = []
    tiled = contour.loc[pd.to_numeric(contour.get("n_tiles", 0), errors="coerce").fillna(0).gt(0)].copy()
    molecular_edge_column = _first_existing(
        tiled,
        ("cell_edge_proximity_fraction__lt_25um", "cell_boundary_distance_um__p10"),
    )
    image_edge_column = _first_existing(
        tiled,
        (
            "tile_edge_proximity_fraction__lt_256px",
            "tile_boundary_distance_px__p10",
            "cell_edge_proximity_fraction__lt_25um",
        ),
    )
    molecular_distance_column = _first_existing(
        tiled,
        ("cell_boundary_distance_um__mean", "cell_boundary_distance_um__median"),
    )
    image_distance_column = _first_existing(
        tiled,
        (
            "tile_boundary_distance_px__mean",
            "tile_boundary_distance_px__median",
            "cell_boundary_distance_um__mean",
        ),
    )
    if molecular_edge_column is None or image_edge_column is None:
        raise ValueError("No usable boundary proximity columns found in contour summary.")
    for _, candidate in candidates.iterrows():
        target = str(candidate["molecular_feature"])
        image = str(candidate["best_image_feature"])
        if target not in tiled.columns or image not in tiled.columns:
            continue
        association_sign = float(np.sign(candidate["partial_spearman_rho"]))
        if not np.isfinite(association_sign) or association_sign == 0:
            association_sign = 1.0
        for structure, group in tiled.groupby("assigned_structure", sort=True, dropna=False):
            if len(group) < 12:
                continue
            mol_edge, mol_p, n_mol = _corr(group, target, molecular_edge_column)
            img_edge, img_p, n_img = _corr(group, image, image_edge_column)
            mol_shift = (
                _quartile_shift_um(group, target, molecular_distance_column)
                if molecular_distance_column is not None
                else np.nan
            )
            image_px_to_um = (
                float(args.slide_mpp)
                if image_distance_column is not None and image_distance_column.startswith("tile_")
                else None
            )
            img_shift = (
                _quartile_shift_um(
                    group,
                    image,
                    image_distance_column,
                    px_to_um=image_px_to_um,
                )
                if image_distance_column is not None
                else np.nan
            )
            lag_count = _morphological_lag_count(
                group,
                target=target,
                image=image,
                association_sign=association_sign,
            )
            rows.append(
                {
                    "program": _program_label(target),
                    "molecular_feature": target,
                    "image_feature": image,
                    "assigned_structure": str(structure),
                    "n_contours": int(len(group)),
                    "global_partial_spearman_rho": candidate.get("partial_spearman_rho", np.nan),
                    "molecular_edge_spearman_rho": mol_edge,
                    "molecular_edge_column": molecular_edge_column,
                    "molecular_edge_p_value": mol_p,
                    "image_edge_spearman_rho": img_edge,
                    "image_edge_column": image_edge_column,
                    "image_edge_p_value": img_p,
                    "molecular_high_minus_low_distance_um": mol_shift,
                    "molecular_distance_column": molecular_distance_column,
                    "image_high_minus_low_distance_um_proxy": img_shift,
                    "image_distance_column": image_distance_column,
                    "morphological_lag_candidate_count": lag_count,
                    "lead_lag_class": _lead_lag_class(mol_edge, img_edge),
                    "boundary_method": (
                        "contour-level proxy using cell/tile distance-to-boundary summaries; "
                        "requires ring-level validation before manuscript claim"
                    ),
                    "n_molecular_edge": n_mol,
                    "n_image_edge": n_img,
                }
            )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            [
                "lead_lag_class",
                "morphological_lag_candidate_count",
                "n_contours",
                "program",
            ],
            ascending=[True, False, False, True],
            kind="stable",
        ).reset_index(drop=True)
    report.to_csv(output_dir / "MAZ_LeadLag_Report.csv", index=False)
    print(f"Wrote {output_dir / 'MAZ_LeadLag_Report.csv'}")
    print(f"Rows: {len(report)}")


if __name__ == "__main__":
    main()
