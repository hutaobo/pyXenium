from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HNSCC_ROOT = r"Y:\long\publication_datasets\headandneckSCC"
DEFAULT_SUZUKI_XENIUM_ROOT = r"Y:\long\publication_datasets\Suzuki_Lung_2024\Xenium"
DEFAULT_SUZUKI_ER_RESULTS_ROOT = r"Y:\long\publication_datasets\Suzuki_Lung_2024\Xenium_ER_results"
DEFAULT_SUZUKI_STRENGTH_ROOT = r"Y:\long\publication_datasets\Suzuki_Lung_2024\Xenium_fibro_strength_from_ER"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_recomputed_table(root: Path, filename: str) -> pd.DataFrame:
    direct = root / filename
    if direct.exists():
        return pd.read_csv(direct)

    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"*/{filename}")):
        frame = pd.read_csv(path)
        if "sample_id" not in frame.columns:
            frame.insert(0, "sample_id", path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compare_numeric(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key: str | list[str],
    columns: list[str],
    tolerance: float,
) -> dict[str, Any]:
    if left.empty or right.empty:
        return {"status": "missing", "n_compared": 0, "max_abs_delta": np.nan, "within_tolerance": False}
    keys = [key] if isinstance(key, str) else list(key)
    missing_keys = [column for column in keys if column not in left.columns or column not in right.columns]
    if missing_keys:
        return {
            "status": "missing_key",
            "missing_keys": missing_keys,
            "n_compared": 0,
            "max_abs_delta": np.nan,
            "within_tolerance": False,
        }
    merged = left.merge(right, on=keys, suffixes=("_left", "_right"))
    deltas: list[float] = []
    for column in columns:
        left_col = f"{column}_left"
        right_col = f"{column}_right"
        if left_col in merged.columns and right_col in merged.columns:
            delta = (pd.to_numeric(merged[left_col], errors="coerce") - pd.to_numeric(merged[right_col], errors="coerce")).abs()
            deltas.extend(delta.dropna().tolist())
    max_delta = max(deltas) if deltas else np.nan
    return {
        "status": "compared",
        "n_compared": int(len(merged)),
        "max_abs_delta": float(max_delta) if np.isfinite(max_delta) else np.nan,
        "within_tolerance": bool(np.isfinite(max_delta) and max_delta <= float(tolerance)),
    }


def validate_hnscc_mechanostress_outputs(
    root: str | Path = DEFAULT_HNSCC_ROOT,
    *,
    recomputed_dir: str | Path | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate HNSCC tumor-stroma reference outputs or compare recomputed outputs."""

    root = Path(root)
    reference_hop = _read_csv_if_exists(root / "HNSCC_infiltrative_hop_distance_summary.csv")
    reference_prop = _read_csv_if_exists(root / "infiltrative_proportion_per_sample.csv")
    coupling = _read_csv_if_exists(root / "tumor_enriched_distance_coupling_score_per_sample.csv")
    payload: dict[str, Any] = {
        "root": str(root),
        "reference_files": {
            "hop_distance_summary": str(root / "HNSCC_infiltrative_hop_distance_summary.csv"),
            "infiltrative_proportion": str(root / "infiltrative_proportion_per_sample.csv"),
            "distance_coupling": str(root / "tumor_enriched_distance_coupling_score_per_sample.csv"),
        },
        "available": {
            "hop_distance_summary": bool(not reference_hop.empty),
            "infiltrative_proportion": bool(not reference_prop.empty),
            "distance_coupling": bool(not coupling.empty),
        },
        "n_reference_samples": int(reference_hop["sample"].nunique()) if "sample" in reference_hop.columns else 0,
        "n_coupling_samples": int(coupling["sample_id"].nunique()) if "sample_id" in coupling.columns else 0,
        "status": "reference_only",
        "comparisons": {},
    }
    if recomputed_dir is not None:
        recomputed = _read_recomputed_table(Path(recomputed_dir), "tumor_growth_summary.csv")
        reference = reference_hop.rename(columns={"sample": "sample_id", "infil_prop": "infiltrative_proportion"})
        payload["status"] = "compared"
        payload["comparisons"]["tumor_growth_summary"] = _compare_numeric(
            reference,
            recomputed,
            key="sample_id",
            columns=["infiltrative_proportion", "mean_infil_dist_to_stromal", "mean_expand_dist_to_stromal"],
            tolerance=tolerance,
        )
    return payload


def validate_suzuki_luad_mechanostress_outputs(
    xenium_root: str | Path = DEFAULT_SUZUKI_XENIUM_ROOT,
    *,
    er_results_root: str | Path = DEFAULT_SUZUKI_ER_RESULTS_ROOT,
    strength_root: str | Path = DEFAULT_SUZUKI_STRENGTH_ROOT,
    recomputed_dir: str | Path | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate Suzuki LUAD/TSU ER and ANE reference outputs or compare recomputed outputs."""

    xenium_root = Path(xenium_root)
    er_results_root = Path(er_results_root)
    strength_root = Path(strength_root)
    er_summary = _read_csv_if_exists(er_results_root / "summary_fibro_ER_by_dataset.csv")
    strength = _read_csv_if_exists(strength_root / "ALL_samples_fibro_strength_vs_radius.csv")
    cluster_strength = _read_csv_if_exists(strength_root / "ALL_samples_fibro_strength_vs_radius_by_cluster.csv")
    cluster_100 = _read_csv_if_exists(strength_root / "summary_by_cluster_at_100um.csv")
    payload: dict[str, Any] = {
        "xenium_root": str(xenium_root),
        "er_results_root": str(er_results_root),
        "strength_root": str(strength_root),
        "available": {
            "er_summary": bool(not er_summary.empty),
            "strength": bool(not strength.empty),
            "cluster_strength": bool(not cluster_strength.empty),
            "cluster_100um": bool(not cluster_100.empty),
        },
        "n_er_datasets": int(er_summary["dataset"].nunique()) if "dataset" in er_summary.columns else 0,
        "n_strength_samples": int(strength["sample"].nunique()) if "sample" in strength.columns else 0,
        "n_strength_rows": int(len(strength)),
        "status": "reference_only",
        "comparisons": {},
    }
    if recomputed_dir is not None:
        recomputed = _read_recomputed_table(Path(recomputed_dir), "axis_strength_by_radius.csv")
        reference = strength.copy()
        if "sample" in reference.columns and "sample_id" not in reference.columns:
            reference = reference.rename(columns={"sample": "sample_id"})
        if "sample" in recomputed.columns and "sample_id" not in recomputed.columns:
            recomputed = recomputed.rename(columns={"sample": "sample_id"})
        payload["status"] = "compared"
        payload["comparisons"]["axis_strength_by_radius"] = _compare_numeric(
            reference,
            recomputed,
            key=["sample_id", "radius_um"],
            columns=["ANE_density_median", "ANE_median", "coh_median", "neigh_median"],
            tolerance=tolerance,
        )
    return payload
