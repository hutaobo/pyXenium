#!/usr/bin/env python3
"""Build candidate tables for NBT strengthening analyses.

The locked one-figure package already contains PLIP source-data rows for the
main breast and cervical claims. This helper keeps those rows as-is and
reconstructs the Figure 1e UNI/PLIP program-family candidate rows from A100
association tables, including the embedding feature name needed by downstream
robustness scripts.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_DATA = (
    Path(__file__).resolve().parents[2]
    / "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
    "naturebiotech_package/NBT_INITIAL_SUBMISSION_UPLOAD_20260515_ONEFIGURE/Source_Data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--breast-plip-run", type=Path, required=True)
    parser.add_argument("--breast-uni-run", type=Path, required=True)
    parser.add_argument("--cervical-plip-run", type=Path, required=True)
    parser.add_argument("--cervical-uni-run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_1c = args.source_data_dir / "Figure_1c_Spatial_Permutation_Source_Data.csv"
    figure_1e = args.source_data_dir / "Figure_1e_CrossCancer_Signature_Source_Data.csv"
    if not figure_1c.exists() or not figure_1e.exists():
        raise FileNotFoundError(f"Missing source-data candidate tables under {args.source_data_dir}")

    plip_out = out_dir / "nbt_candidates_figure1c_plip.csv"
    shutil.copy2(figure_1c, plip_out)

    run_dirs = {
        ("breast", "plip"): args.breast_plip_run,
        ("breast", "uni"): args.breast_uni_run,
        ("cervical", "plip"): args.cervical_plip_run,
        ("cervical", "uni"): args.cervical_uni_run,
    }
    cross = pd.read_csv(figure_1e)
    rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for _, source_row in cross.iterrows():
        dataset = str(source_row["dataset"]).lower()
        model = str(source_row["model"]).lower()
        run_dir = run_dirs[(dataset, model)]
        assoc = read_associations(run_dir)
        program = str(source_row["top_pathway"])
        reported = _to_float(source_row["top_partial_rho"])
        if program == "not_detected" or not math.isfinite(reported):
            skipped_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "program_family": source_row["program_family"],
                    "top_pathway": program,
                    "top_partial_rho": source_row["top_partial_rho"],
                    "support_call": source_row["support_call"],
                    "skip_reason": "not_detected_or_nonfinite",
                }
            )
            continue
        molecular_feature = f"program__wta_{program}"
        selected = select_association(assoc, molecular_feature, reported)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "program": program,
                "program_family": source_row["program_family"],
                "molecular_feature": molecular_feature,
                "image_feature": selected["image_feature"],
                "assigned_structure_filter": "",
                "n_contours": int(selected["n_contours"]),
                "reported_partial_rho": reported,
                "association_table_partial_rho": float(selected["partial_spearman_rho"]),
                "association_minus_reported": float(selected["partial_spearman_rho"])
                - reported,
                "source": "Figure_1e_CrossCancer_Signature_Source_Data",
            }
        )
    cross_out = out_dir / "nbt_candidates_figure1e_cross_cancer.csv"
    pd.DataFrame(rows).to_csv(cross_out, index=False)
    skipped_out = out_dir / "nbt_candidates_figure1e_skipped.csv"
    pd.DataFrame(skipped_rows).to_csv(skipped_out, index=False)

    combined_out = out_dir / "nbt_candidates_combined.csv"
    combined = pd.concat(
        [
            normalize_candidate_columns(pd.read_csv(plip_out)),
            normalize_candidate_columns(pd.read_csv(cross_out)),
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        subset=["dataset", "model", "program", "molecular_feature", "image_feature"],
        keep="first",
    )
    combined.to_csv(combined_out, index=False)

    print(f"Wrote {plip_out}")
    print(f"Wrote {cross_out}")
    print(f"Wrote {skipped_out}")
    print(f"Wrote {combined_out}")
    print(combined.groupby(["dataset", "model"]).size().to_string())
    return 0


def read_associations(run_dir: Path) -> pd.DataFrame:
    path = run_dir.expanduser().resolve() / "contour_image_molecular_associations.parquet"
    if not path.exists():
        path = path.with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing association table under {run_dir}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def select_association(assoc: pd.DataFrame, molecular_feature: str, reported_rho: float) -> pd.Series:
    sub = assoc.loc[assoc["molecular_feature"].astype(str).eq(molecular_feature)].copy()
    if sub.empty:
        raise KeyError(f"No rows for {molecular_feature!r} in association table.")
    partial = pd.to_numeric(sub["partial_spearman_rho"], errors="coerce")
    if math.isfinite(reported_rho):
        idx = (partial - reported_rho).abs().idxmin()
    else:
        idx = partial.abs().idxmax()
    selected = sub.loc[idx]
    if not np.isfinite(float(selected["partial_spearman_rho"])):
        raise ValueError(f"Selected association for {molecular_feature!r} is not finite.")
    return selected


def _to_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def normalize_candidate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in [
        "dataset",
        "model",
        "program",
        "program_family",
        "molecular_feature",
        "image_feature",
        "assigned_structure_filter",
        "n_contours",
        "reported_partial_rho",
        "source",
    ]:
        if column not in out.columns:
            out[column] = ""
    out["model"] = out["model"].astype(str).str.lower()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    return out[
        [
            "dataset",
            "model",
            "program",
            "program_family",
            "molecular_feature",
            "image_feature",
            "assigned_structure_filter",
            "n_contours",
            "reported_partial_rho",
            "source",
        ]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
