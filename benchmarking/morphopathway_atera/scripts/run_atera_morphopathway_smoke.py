from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_BREAST_ROOT = Path(r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs")
DEFAULT_CERVICAL_ROOT = Path(r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Cervical_Cancer_outs")
DEFAULT_SPATIAL_BLOCK_BINS = 12


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "pyXenium").exists():
            return parent
    raise RuntimeError("Could not resolve pyXenium repository root.")


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return _repo_root() / "benchmarking" / "morphopathway_atera" / "results" / f"he_cell_smoke_{stamp}"


def _pass_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].fillna(False)
    if pd.api.types.is_bool_dtype(values):
        return int(values.sum())
    return int(values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pyXenium.pathway Atera breast/cervical morphopathway smoke bundle."
    )
    parser.add_argument("--breast-root", type=Path, default=DEFAULT_BREAST_ROOT)
    parser.add_argument("--cervical-root", type=Path, default=DEFAULT_CERVICAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cells", type=int, default=2500)
    parser.add_argument("--extra-background-genes", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=8)
    parser.add_argument("--negative-controls", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--validation-threshold", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--no-he-features", action="store_true")
    parser.add_argument("--he-max-dimension", type=int, default=2048)
    parser.add_argument("--he-patch-radius-px", type=int, default=1)
    parser.add_argument(
        "--feature-set",
        choices=("he", "he-cell", "he-projection", "plip"),
        default="he",
        help="Use H&E sampled features, H&E plus cell proxies, or deterministic H&E patch-projection embeddings.",
    )
    parser.add_argument("--aggregation", choices=("cell", "spatial-block"), default="spatial-block")
    parser.add_argument(
        "--spatial-block-bins",
        type=int,
        default=DEFAULT_SPATIAL_BLOCK_BINS,
        help="Spatial pseudobulk grid size; bins=12 was stable across the Atera PLIP 3000-cell smoke seeds.",
    )
    parser.add_argument("--min-cells-per-block", type=int, default=5)
    parser.add_argument("--patch-projection-dim", type=int, default=24)
    parser.add_argument("--clip-model-name", default="vinid/plip")
    parser.add_argument("--clip-model-label", default="plip")
    parser.add_argument("--clip-output-dim", type=int, default=64)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--clip-patch-radius-px", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    import sys

    repo = _repo_root()
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from pyXenium import (
        MorphoPathwayConfig,
        run_xenium_cell_morphopathway_smoke,
        summarize_cross_cancer_validation,
    )

    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set == "plip":
        image_feature_prefixes = (f"embedding__{str(args.clip_model_label).lower()}_",)
        include_patch_projection = False
        clip_model_name = str(args.clip_model_name)
    elif args.feature_set == "he-projection":
        image_feature_prefixes = ("embedding__he_patch_projection_", "image__he_")
        include_patch_projection = True
        clip_model_name = None
    elif args.feature_set == "he":
        image_feature_prefixes = ("image__he_",)
        include_patch_projection = False
        clip_model_name = None
    else:
        image_feature_prefixes = ("image__", "morphology__")
        include_patch_projection = False
        clip_model_name = None
    config = MorphoPathwayConfig(
        covariates=("structure", "x", "y", "boundary_distance", "log_total_counts"),
        image_feature_prefixes=image_feature_prefixes,
        spatial_strata_cols=("structure",),
        min_pathway_genes=2,
        min_pathway_coverage=0.35,
        n_permutations=int(args.permutations),
        permutation_top_n=int(args.top_n),
        n_negative_controls=int(args.negative_controls),
        negative_control_top_n=int(args.top_n),
        random_state=int(args.random_state),
    )

    runs = {
        "breast_discovery": args.breast_root,
        "cervical_validation": args.cervical_root,
    }
    results = {}
    summaries = []
    for sample_name, root in runs.items():
        sample_output = output_dir / sample_name
        result = run_xenium_cell_morphopathway_smoke(
            root,
            output_dir=sample_output,
            sample_name=sample_name,
            config=config,
            max_cells=int(args.max_cells),
            extra_background_genes=int(args.extra_background_genes),
            include_he_features=not bool(args.no_he_features),
            he_max_dimension=int(args.he_max_dimension),
            he_patch_radius_px=int(args.he_patch_radius_px),
            include_patch_projection=include_patch_projection,
            patch_projection_dim=int(args.patch_projection_dim),
            clip_model_name=clip_model_name,
            clip_model_label=str(args.clip_model_label),
            clip_output_dim=int(args.clip_output_dim),
            clip_batch_size=int(args.clip_batch_size),
            clip_device=str(args.clip_device),
            clip_patch_radius_px=int(args.clip_patch_radius_px),
            aggregation=str(args.aggregation),
            spatial_block_bins=int(args.spatial_block_bins),
            min_cells_per_block=int(args.min_cells_per_block),
            random_state=int(args.random_state),
        )
        results[sample_name] = result
        associations = result["associations"]
        coverage = result["pathway_coverage"]
        negative_controls = result["negative_controls"]
        spatial_nulls = result["spatial_nulls"]
        top = associations.iloc[0]
        summaries.append(
            {
                "sample_name": sample_name,
                "xenium_root": str(root),
                "output_dir": str(sample_output),
                "n_cells": int(result["input_manifest"].loc[0, "n_cells"]),
                "aggregation": str(result["input_manifest"].loc[0, "aggregation"]),
                "analysis_observations": int(result["input_manifest"].loc[0, "analysis_observations"]),
                "n_passing_pathways": int(coverage["passes"].sum()),
                "n_pathways": int(len(coverage)),
                "n_associations": int(len(associations)),
                "n_spatial_nulls": int(len(spatial_nulls)),
                "n_spatial_null_pass95": _pass_count(spatial_nulls, "passes_spatial_null_95"),
                "n_spatial_null_pass99": _pass_count(spatial_nulls, "passes_spatial_null_99"),
                "n_negative_controls": int(len(negative_controls)),
                "n_negative_control_pass95": _pass_count(negative_controls, "passes_negative_control_95"),
                "n_negative_control_pass99": _pass_count(negative_controls, "passes_negative_control_99"),
                "he_features_available": bool(result["input_manifest"].loc[0, "he_features_available"]),
                "he_cells_inside_fraction": result["input_manifest"].loc[0, "he_cells_inside_fraction"],
                "he_embedding_backend_status": result["input_manifest"].loc[0, "he_embedding_backend_status"],
                "top_pathway": str(top["pathway"]),
                "top_family": str(top["family"]),
                "top_image_feature": str(top["image_feature"]),
                "top_abs_partial_rho": float(top["abs_partial_spearman_rho"]),
            }
        )

    validation = summarize_cross_cancer_validation(
        results["breast_discovery"]["associations"],
        results["cervical_validation"]["associations"],
        config=MorphoPathwayConfig(validation_abs_rho_threshold=float(args.validation_threshold)),
    )
    validation.to_csv(output_dir / "cross_cancer_validation.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary["validation_recovered"] = int(validation["recovered_in_validation"].sum())
    summary["validation_total"] = int(len(validation))
    summary.to_csv(output_dir / "smoke_summary.csv", index=False)
    manifest = {
        "output_dir": str(output_dir),
        "config": config.__dict__,
        "validation_threshold": float(args.validation_threshold),
        "summary_csv": str(output_dir / "smoke_summary.csv"),
        "cross_cancer_validation_csv": str(output_dir / "cross_cancer_validation.csv"),
    }
    (output_dir / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
