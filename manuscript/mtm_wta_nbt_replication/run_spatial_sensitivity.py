#!/usr/bin/env python3
"""Run spatial sensitivity checks for candidate mTM associations.

This script expects contour-level mTM output tables from a full run directory,
especially `contour_multimodal_summary.parquet` or `.csv`. It strengthens the
main residual-decoding claim by testing whether candidate H&E-WTA associations
are stable to spatial block holdout, local mismatched-pair controls and centroid
covariate jitter.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (
    REPO_ROOT
    / "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
    "naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/"
    "Source_Data/Figure_1c_Spatial_Permutation_Source_Data.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "manuscript/mtm_wta_nbt_replication/spatial_sensitivity_20260516"
)
DEFAULT_CONTROLS = (
    "assigned_structure",
    "centroid_x",
    "centroid_y",
    "cell_boundary_distance_um__mean",
    "tile_boundary_distance_px__mean",
)


@dataclass(frozen=True)
class Candidate:
    dataset: str
    model: str
    program: str
    molecular_feature: str
    image_feature: str
    assigned_structure_filter: str | None
    reported_partial_rho: float | None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path}")


def find_contour_table(run_dir: Path) -> pd.DataFrame:
    for name in ("contour_multimodal_summary.parquet", "contour_multimodal_summary.csv"):
        path = run_dir / name
        if path.exists():
            return read_table(path)
    raise FileNotFoundError(
        f"Missing contour_multimodal_summary.parquet/csv under {run_dir}. "
        "Run the full mTM workflow or export the table from A100 first."
    )


def read_candidates(path: Path, *, dataset: str | None, model: str | None) -> list[Candidate]:
    rows: list[Candidate] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if dataset is not None and row.get("dataset") != dataset:
                continue
            if model is not None and row.get("model") != model:
                continue
            structure = row.get("assigned_structure_filter") or None
            rows.append(
                Candidate(
                    dataset=str(row.get("dataset", "")),
                    model=str(row.get("model", "")),
                    program=str(row.get("program", "")),
                    molecular_feature=str(row.get("molecular_feature", "")),
                    image_feature=str(row.get("image_feature", "")),
                    assigned_structure_filter=str(structure) if structure else None,
                    reported_partial_rho=_to_float_or_none(row.get("reported_partial_rho")),
                )
            )
    return rows


def _to_float_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def candidate_frame(frame: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    required = [candidate.image_feature, candidate.molecular_feature]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing candidate columns for {candidate.program}: {missing}")
    work = frame.copy()
    if candidate.assigned_structure_filter and "assigned_structure" in work.columns:
        work = work.loc[work["assigned_structure"].astype(str).eq(candidate.assigned_structure_filter)].copy()
    keep = [candidate.image_feature, candidate.molecular_feature]
    keep += [column for column in DEFAULT_CONTROLS if column in work.columns]
    return work.loc[:, list(dict.fromkeys(keep))].copy()


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
    columns = [image_feature, molecular_feature]
    work = frame.copy()
    x = pd.to_numeric(work[image_feature], errors="coerce")
    y = pd.to_numeric(work[molecular_feature], errors="coerce")
    mask = x.notna() & y.notna()
    work = work.loc[mask, :].copy()
    if len(work) < 6:
        return np.nan, int(len(work))
    ranked_x = pd.to_numeric(work[columns[0]], errors="coerce").rank(method="average").to_numpy(dtype=float)
    ranked_y = pd.to_numeric(work[columns[1]], errors="coerce").rank(method="average").to_numpy(dtype=float)
    controls = control_matrix(work)
    x_resid = residualize(ranked_x, controls)
    y_resid = residualize(ranked_y, controls)
    if np.nanstd(x_resid) == 0.0 or np.nanstd(y_resid) == 0.0:
        return np.nan, int(len(work))
    rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
    return rho, int(len(work))


def spatial_blocks(frame: pd.DataFrame, *, bins: int) -> pd.Series:
    if not {"centroid_x", "centroid_y"}.issubset(frame.columns):
        return pd.Series(["all"] * len(frame), index=frame.index)
    x = pd.to_numeric(frame["centroid_x"], errors="coerce")
    y = pd.to_numeric(frame["centroid_y"], errors="coerce")
    x_bin = pd.qcut(x.rank(method="first"), q=min(bins, max(1, x.notna().sum())), labels=False, duplicates="drop")
    y_bin = pd.qcut(y.rank(method="first"), q=min(bins, max(1, y.notna().sum())), labels=False, duplicates="drop")
    return x_bin.fillna(-1).astype(int).astype(str) + "_" + y_bin.fillna(-1).astype(int).astype(str)


def leave_one_block_out(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    bins: int,
    min_contours: int,
) -> pd.DataFrame:
    work = candidate_frame(frame, candidate)
    blocks = spatial_blocks(work, bins=bins)
    rows = []
    for block in sorted(blocks.unique()):
        subset = work.loc[~blocks.eq(block), :].copy()
        rho, n = partial_spearman(subset, candidate.image_feature, candidate.molecular_feature)
        if n < min_contours:
            continue
        rows.append(
            {
                "dataset": candidate.dataset,
                "model": candidate.model,
                "program": candidate.program,
                "excluded_block": block,
                "excluded_n": int(blocks.eq(block).sum()),
                "remaining_n": n,
                "partial_spearman_rho": rho,
                "sign_matches_reported": _sign_matches(rho, candidate.reported_partial_rho),
            }
        )
    return pd.DataFrame(rows)


def local_mismatch_controls(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    k_neighbors: int,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    work = candidate_frame(frame, candidate)
    work = work.dropna(subset=[candidate.image_feature, candidate.molecular_feature]).copy()
    observed, n = partial_spearman(work, candidate.image_feature, candidate.molecular_feature)
    if n < 6 or not {"centroid_x", "centroid_y"}.issubset(work.columns):
        return _empty_control_result(candidate, observed, n, reason="insufficient_coordinates")
    coords = work.loc[:, ["centroid_x", "centroid_y"]].apply(pd.to_numeric, errors="coerce")
    coords = coords.fillna(coords.median()).to_numpy(dtype=float)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    k = max(1, min(k_neighbors, len(work) - 1))
    neighbor_index = np.argsort(distances, axis=1)[:, :k]
    rng = np.random.default_rng(seed)
    null_rhos = []
    values = work[candidate.molecular_feature].to_numpy(copy=True)
    for _ in range(n_permutations):
        picked = neighbor_index[np.arange(len(work)), rng.integers(0, k, size=len(work))]
        mismatched = work.copy()
        mismatched[candidate.molecular_feature] = values[picked]
        rho, _ = partial_spearman(mismatched, candidate.image_feature, candidate.molecular_feature)
        if math.isfinite(rho):
            null_rhos.append(rho)
    if not null_rhos:
        return _empty_control_result(candidate, observed, n, reason="no_finite_null")
    arr = np.asarray(null_rhos, dtype=float)
    return {
        "dataset": candidate.dataset,
        "model": candidate.model,
        "program": candidate.program,
        "n_contours": n,
        "observed_partial_spearman_rho": observed,
        "observed_abs_rho": abs(observed),
        "local_mismatch_iterations": int(len(arr)),
        "local_mismatch_abs_rho_median": float(np.median(np.abs(arr))),
        "local_mismatch_abs_rho_q95": float(np.quantile(np.abs(arr), 0.95)),
        "observed_exceeds_local_mismatch_q95": bool(abs(observed) > np.quantile(np.abs(arr), 0.95)),
        "reason": "ok",
    }


def centroid_jitter_sensitivity(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    scales: tuple[float, ...],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    work = candidate_frame(frame, candidate)
    observed, n = partial_spearman(work, candidate.image_feature, candidate.molecular_feature)
    if not {"centroid_x", "centroid_y"}.issubset(work.columns):
        return pd.DataFrame()
    coords = work.loc[:, ["centroid_x", "centroid_y"]].apply(pd.to_numeric, errors="coerce")
    span = np.nanmedian(
        [
            float(coords["centroid_x"].max() - coords["centroid_x"].min()),
            float(coords["centroid_y"].max() - coords["centroid_y"].min()),
        ]
    )
    span = span if math.isfinite(span) and span > 0 else 1.0
    rng = np.random.default_rng(seed)
    rows = []
    for scale in scales:
        rhos = []
        for _ in range(n_permutations):
            jittered = work.copy()
            sigma = span * scale
            jittered["centroid_x"] = pd.to_numeric(jittered["centroid_x"], errors="coerce") + rng.normal(0.0, sigma, len(jittered))
            jittered["centroid_y"] = pd.to_numeric(jittered["centroid_y"], errors="coerce") + rng.normal(0.0, sigma, len(jittered))
            rho, _ = partial_spearman(jittered, candidate.image_feature, candidate.molecular_feature)
            if math.isfinite(rho):
                rhos.append(rho)
        arr = np.asarray(rhos, dtype=float)
        rows.append(
            {
                "dataset": candidate.dataset,
                "model": candidate.model,
                "program": candidate.program,
                "n_contours": n,
                "observed_partial_spearman_rho": observed,
                "centroid_jitter_scale_fraction_of_slide_span": scale,
                "iterations": int(len(arr)),
                "rho_median": float(np.median(arr)) if len(arr) else np.nan,
                "rho_q05": float(np.quantile(arr, 0.05)) if len(arr) else np.nan,
                "rho_q95": float(np.quantile(arr, 0.95)) if len(arr) else np.nan,
                "sign_stability_fraction": float(np.mean(np.sign(arr) == np.sign(observed))) if len(arr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_candidate(frame: pd.DataFrame, candidate: Candidate) -> dict[str, object]:
    work = candidate_frame(frame, candidate)
    rho, n = partial_spearman(work, candidate.image_feature, candidate.molecular_feature)
    return {
        "dataset": candidate.dataset,
        "model": candidate.model,
        "program": candidate.program,
        "molecular_feature": candidate.molecular_feature,
        "image_feature": candidate.image_feature,
        "assigned_structure_filter": candidate.assigned_structure_filter or "",
        "n_contours": n,
        "observed_partial_spearman_rho": rho,
        "reported_partial_spearman_rho": candidate.reported_partial_rho,
        "observed_minus_reported": rho - candidate.reported_partial_rho
        if candidate.reported_partial_rho is not None and math.isfinite(rho)
        else np.nan,
    }


def _empty_control_result(candidate: Candidate, observed: float, n: int, *, reason: str) -> dict[str, object]:
    return {
        "dataset": candidate.dataset,
        "model": candidate.model,
        "program": candidate.program,
        "n_contours": n,
        "observed_partial_spearman_rho": observed,
        "observed_abs_rho": abs(observed) if math.isfinite(observed) else np.nan,
        "local_mismatch_iterations": 0,
        "local_mismatch_abs_rho_median": np.nan,
        "local_mismatch_abs_rho_q95": np.nan,
        "observed_exceeds_local_mismatch_q95": False,
        "reason": reason,
    }


def _sign_matches(value: float, reference: float | None) -> bool:
    if reference is None or not math.isfinite(value):
        return False
    return bool(np.sign(value) == np.sign(reference))


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    loo: pd.DataFrame,
    mismatch: pd.DataFrame,
    jitter: pd.DataFrame,
) -> None:
    lines = [
        "# Spatial sensitivity report",
        "",
        "This report summarizes reviewer-facing sensitivity checks for candidate",
        "mTM residual H&E-WTA associations.",
        "",
        "## Candidate summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['dataset']} / {row['model']} / {row['program']}: "
            f"rho={float(row['observed_partial_spearman_rho']):.3f}, "
            f"n={int(row['n_contours'])}."
        )
    if not loo.empty:
        stable = loo.groupby(["dataset", "model", "program"])["sign_matches_reported"].mean().reset_index()
        lines.extend(["", "## Leave-one-spatial-block-out", ""])
        for _, row in stable.iterrows():
            lines.append(
                f"- {row['dataset']} / {row['model']} / {row['program']}: "
                f"sign-stable in {float(row['sign_matches_reported']):.0%} of held-out block fits."
            )
    if not mismatch.empty:
        lines.extend(["", "## Local mismatch controls", ""])
        for _, row in mismatch.iterrows():
            lines.append(
                f"- {row['dataset']} / {row['model']} / {row['program']}: "
                f"observed |rho|={float(row['observed_abs_rho']):.3f}; "
                f"local mismatch q95={float(row['local_mismatch_abs_rho_q95']):.3f}; "
                f"exceeds q95={row['observed_exceeds_local_mismatch_q95']}."
            )
    if not jitter.empty:
        lines.extend(["", "## Centroid jitter sensitivity", ""])
        grouped = jitter.groupby(["dataset", "model", "program"])["sign_stability_fraction"].min().reset_index()
        for _, row in grouped.iterrows():
            lines.append(
                f"- {row['dataset']} / {row['model']} / {row['program']}: "
                f"minimum sign-stability across jitter scales={float(row['sign_stability_fraction']):.0%}."
            )
    (out_dir / "spatial_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sensitivity(frame: pd.DataFrame, candidates: list[Candidate], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    summary_rows = []
    loo_frames = []
    mismatch_rows = []
    jitter_frames = []
    for candidate in candidates:
        summary_rows.append(summarize_candidate(frame, candidate))
        loo_frames.append(
            leave_one_block_out(
                frame,
                candidate,
                bins=args.spatial_bins,
                min_contours=args.min_contours,
            )
        )
        mismatch_rows.append(
            local_mismatch_controls(
                frame,
                candidate,
                k_neighbors=args.local_neighbors,
                n_permutations=args.local_mismatch_permutations,
                seed=args.seed,
            )
        )
        jitter_frames.append(
            centroid_jitter_sensitivity(
                frame,
                candidate,
                scales=tuple(args.jitter_scales),
                n_permutations=args.jitter_permutations,
                seed=args.seed,
            )
        )
    return {
        "summary": pd.DataFrame(summary_rows),
        "leave_one_block_out": pd.concat(loo_frames, ignore_index=True) if loo_frames else pd.DataFrame(),
        "local_mismatch_controls": pd.DataFrame(mismatch_rows),
        "centroid_jitter_sensitivity": pd.concat(jitter_frames, ignore_index=True) if jitter_frames else pd.DataFrame(),
    }


def write_outputs(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    write_report(
        out_dir,
        tables["summary"],
        tables["leave_one_block_out"],
        tables["local_mismatch_controls"],
        tables["centroid_jitter_sensitivity"],
    )


def synthetic_self_test() -> None:
    rng = np.random.default_rng(17)
    n = 160
    x_coord = rng.uniform(0, 1000, n)
    y_coord = rng.uniform(0, 800, n)
    image = rng.normal(size=n)
    program = 0.65 * image + 0.2 * (x_coord / x_coord.std()) + rng.normal(scale=0.7, size=n)
    frame = pd.DataFrame(
        {
            "contour_id": [f"S3 #{i}" for i in range(n)],
            "assigned_structure": ["S3"] * n,
            "centroid_x": x_coord,
            "centroid_y": y_coord,
            "cell_boundary_distance_um__mean": rng.uniform(0, 100, n),
            "tile_boundary_distance_px__mean": rng.uniform(0, 50, n),
            "embedding__103__mean": image,
            "program__wta_luminal_estrogen_response": program,
        }
    )
    candidate = Candidate(
        dataset="synthetic",
        model="plip",
        program="luminal_estrogen_response",
        molecular_feature="program__wta_luminal_estrogen_response",
        image_feature="embedding__103__mean",
        assigned_structure_filter="S3",
        reported_partial_rho=None,
    )
    args = argparse.Namespace(
        spatial_bins=4,
        min_contours=40,
        local_neighbors=5,
        local_mismatch_permutations=50,
        jitter_scales=[0.0, 0.005],
        jitter_permutations=20,
        seed=17,
    )
    tables = run_sensitivity(frame, [candidate], args)
    observed = float(tables["summary"].iloc[0]["observed_partial_spearman_rho"])
    if observed < 0.4:
        raise SystemExit(f"Self-test failed: observed rho too small ({observed:.3f})")
    print("Self-test passed.")
    print(tables["summary"].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Full mTM run directory.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", default="breast")
    parser.add_argument("--model", default="plip")
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument("--min-contours", type=int, default=40)
    parser.add_argument("--local-neighbors", type=int, default=5)
    parser.add_argument("--local-mismatch-permutations", type=int, default=1000)
    parser.add_argument("--jitter-scales", type=float, nargs="+", default=[0.0, 0.0025, 0.005, 0.01])
    parser.add_argument("--jitter-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return 0
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --self-test is used.")
    frame = find_contour_table(args.run_dir.expanduser().resolve())
    candidates = read_candidates(args.candidates, dataset=args.dataset, model=args.model)
    if not candidates:
        raise SystemExit(f"No candidates selected from {args.candidates}.")
    tables = run_sensitivity(frame, candidates, args)
    write_outputs(args.out_dir.expanduser().resolve(), tables)
    print(f"Wrote spatial sensitivity outputs to: {args.out_dir.expanduser().resolve()}")
    print(tables["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
