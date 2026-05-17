#!/usr/bin/env python3
"""Run nested spatial holdout checks for mTM associations.

For each candidate molecular program, each spatial block is held out. The
embedding dimension is selected only on the remaining blocks, then evaluated on
the held-out block with the same residual partial Spearman model.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_spatial_sensitivity import Candidate, control_matrix, find_contour_table, partial_spearman, read_candidates, spatial_blocks


EMBEDDING_MEAN_RE = re.compile(r"^embedding__\d+__mean$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument("--min-train-contours", type=int, default=60)
    parser.add_argument("--min-test-contours", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return 0

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = find_contour_table(args.run_dir.expanduser().resolve())
    candidates = read_candidates(args.candidates, dataset=args.dataset, model=args.model)
    if not candidates:
        raise SystemExit("No candidates matched dataset/model.")
    long = run_nested_holdout(
        frame,
        candidates,
        spatial_bins=args.spatial_bins,
        min_train_contours=args.min_train_contours,
        min_test_contours=args.min_test_contours,
    )
    summary = summarize(long)
    long.to_csv(out_dir / "nested_spatial_holdout_long.csv", index=False)
    summary.to_csv(out_dir / "nested_spatial_holdout_summary.csv", index=False)
    write_report(out_dir / "nested_spatial_holdout_report.md", summary)
    print(f"Wrote nested holdout outputs to {out_dir}")
    print(summary.to_string(index=False))
    return 0


def run_nested_holdout(
    frame: pd.DataFrame,
    candidates: list[Candidate],
    *,
    spatial_bins: int,
    min_train_contours: int,
    min_test_contours: int,
) -> pd.DataFrame:
    embedding_columns = [column for column in frame.columns if EMBEDDING_MEAN_RE.match(str(column))]
    if not embedding_columns:
        raise KeyError("No embedding__*__mean columns found in contour table.")
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        work = candidate_work_frame(frame, candidate)
        if candidate.molecular_feature not in work.columns:
            raise KeyError(f"Missing molecular feature {candidate.molecular_feature!r}.")
        blocks = spatial_blocks(work, bins=spatial_bins)
        base_rho, base_n = partial_spearman(work, candidate.image_feature, candidate.molecular_feature)
        for block in sorted(blocks.unique()):
            train = work.loc[~blocks.eq(block), :].copy()
            test = work.loc[blocks.eq(block), :].copy()
            if len(train) < min_train_contours or len(test) < min_test_contours:
                continue
            selected_feature, train_rho, train_n = select_feature(train, embedding_columns, candidate.molecular_feature)
            test_rho, test_n = partial_spearman(test, selected_feature, candidate.molecular_feature)
            locked_test_rho, locked_test_n = partial_spearman(test, candidate.image_feature, candidate.molecular_feature)
            rows.append(
                {
                    **asdict(candidate),
                    "heldout_block": block,
                    "train_n_contours": train_n,
                    "test_n_contours": test_n,
                    "base_locked_partial_spearman_rho": base_rho,
                    "base_locked_n_contours": base_n,
                    "selected_image_feature": selected_feature,
                    "train_selected_partial_spearman_rho": train_rho,
                    "test_selected_partial_spearman_rho": test_rho,
                    "locked_image_feature": candidate.image_feature,
                    "test_locked_partial_spearman_rho": locked_test_rho,
                    "test_locked_n_contours": locked_test_n,
                    "selected_feature_matches_locked": selected_feature == candidate.image_feature,
                    "test_selected_sign_matches_train": sign_matches(test_rho, train_rho),
                    "test_selected_sign_matches_reported": sign_matches(test_rho, candidate.reported_partial_rho),
                    "test_locked_sign_matches_reported": sign_matches(locked_test_rho, candidate.reported_partial_rho),
                }
            )
    return pd.DataFrame(rows)


def candidate_work_frame(frame: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    work = frame.copy()
    if candidate.assigned_structure_filter and "assigned_structure" in work.columns:
        work = work.loc[work["assigned_structure"].astype(str).eq(candidate.assigned_structure_filter)].copy()
    return work


def select_feature(frame: pd.DataFrame, embedding_columns: list[str], molecular_feature: str) -> tuple[str, float, int]:
    rhos, n = vectorized_partial_spearman(frame, embedding_columns, molecular_feature)
    finite = rhos.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError(f"No finite feature correlations for {molecular_feature}.")
    selected = str(finite.abs().idxmax())
    return selected, float(rhos[selected]), n


def vectorized_partial_spearman(frame: pd.DataFrame, image_features: list[str], molecular_feature: str) -> tuple[pd.Series, int]:
    y = pd.to_numeric(frame[molecular_feature], errors="coerce")
    x = frame.loc[:, image_features].apply(pd.to_numeric, errors="coerce")
    valid = y.notna() & x.notna().all(axis=1)
    work = frame.loc[valid, :].copy()
    x = x.loc[valid, :]
    y = y.loc[valid]
    if len(work) < 6:
        return pd.Series(np.nan, index=image_features), int(len(work))
    ranked_x = x.rank(method="average").to_numpy(dtype=float)
    ranked_y = y.rank(method="average").to_numpy(dtype=float)
    controls = control_matrix(work)
    x_resid = residualize_matrix(ranked_x, controls)
    y_resid = residualize_vector(ranked_y, controls)
    numer = np.sum(x_resid * y_resid[:, None], axis=0)
    denom = np.sqrt(np.sum(x_resid**2, axis=0) * np.sum(y_resid**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numer / denom
    return pd.Series(values, index=image_features), int(len(work))


def residualize_matrix(values: np.ndarray, controls: pd.DataFrame) -> np.ndarray:
    if controls.empty:
        return values - np.nanmean(values, axis=0, keepdims=True)
    design = np.column_stack([np.ones(values.shape[0], dtype=float), controls.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def residualize_vector(values: np.ndarray, controls: pd.DataFrame) -> np.ndarray:
    if controls.empty:
        return values - np.nanmean(values)
    design = np.column_stack([np.ones(values.shape[0], dtype=float), controls.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in long.groupby(["dataset", "model", "program", "molecular_feature", "locked_image_feature"], sort=False):
        selected = group["selected_image_feature"].value_counts()
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "program": keys[2],
                "molecular_feature": keys[3],
                "locked_image_feature": keys[4],
                "folds": int(len(group)),
                "base_locked_partial_spearman_rho": float(group["base_locked_partial_spearman_rho"].iloc[0]),
                "median_train_selected_partial_spearman_rho": float(pd.to_numeric(group["train_selected_partial_spearman_rho"], errors="coerce").median()),
                "median_test_selected_partial_spearman_rho": float(pd.to_numeric(group["test_selected_partial_spearman_rho"], errors="coerce").median()),
                "median_test_locked_partial_spearman_rho": float(pd.to_numeric(group["test_locked_partial_spearman_rho"], errors="coerce").median()),
                "selected_test_sign_stability_fraction": float(pd.Series(group["test_selected_sign_matches_train"]).mean()),
                "selected_test_sign_matches_reported_fraction": float(pd.Series(group["test_selected_sign_matches_reported"]).mean()),
                "locked_test_sign_matches_reported_fraction": float(pd.Series(group["test_locked_sign_matches_reported"]).mean()),
                "selected_feature_matches_locked_fraction": float(pd.Series(group["selected_feature_matches_locked"]).mean()),
                "top_selected_feature": str(selected.index[0]) if len(selected) else "",
                "top_selected_feature_fold_fraction": float(selected.iloc[0] / len(group)) if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sign_matches(value: float, reference: float | None) -> bool:
    if reference is None:
        return False
    try:
        reference_float = float(reference)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value) or not np.isfinite(reference_float) or value == 0.0 or reference_float == 0.0:
        return False
    return bool(np.sign(value) == np.sign(reference_float))


def write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Nested Spatial Holdout Report",
        "",
        "Each fold selected the embedding dimension using only training spatial blocks, then evaluated the selected feature on the held-out block.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("- No eligible folds were produced.")
    for _, row in summary.iterrows():
        lines.append(
            "- {dataset} / {model} / {program}: {folds} folds, median held-out selected rho {test:.3f}, selected sign stability {stable:.0%}, locked-feature held-out sign match {locked:.0%}, top selected feature {feature} ({feature_frac:.0%} of folds).".format(
                dataset=row["dataset"],
                model=str(row["model"]).upper(),
                program=row["program"],
                folds=int(row["folds"]),
                test=float(row["median_test_selected_partial_spearman_rho"]),
                stable=float(row["selected_test_sign_stability_fraction"]),
                locked=float(row["locked_test_sign_matches_reported_fraction"]),
                feature=row["top_selected_feature"],
                feature_frac=float(row["top_selected_feature_fold_fraction"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_self_test() -> None:
    rng = np.random.default_rng(17)
    n = 240
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "assigned_structure": ["S"] * n,
            "centroid_x": x,
            "centroid_y": y,
            "program__wta_test": signal + rng.normal(scale=0.4, size=n),
            "embedding__10__mean": signal + rng.normal(scale=0.4, size=n),
        }
    )
    for idx in range(32):
        frame[f"embedding__{idx}__mean"] = rng.normal(size=n)
    frame["embedding__10__mean"] = signal + rng.normal(scale=0.4, size=n)
    candidate = Candidate("synthetic", "plip", "test", "program__wta_test", "embedding__10__mean", "S", 0.8)
    long = run_nested_holdout(frame, [candidate], spatial_bins=4, min_train_contours=80, min_test_contours=5)
    summary = summarize(long)
    if summary.empty or float(summary.iloc[0]["selected_test_sign_stability_fraction"]) < 0.75:
        raise SystemExit("Self-test failed: unstable selected signal.")
    print("Self-test passed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
