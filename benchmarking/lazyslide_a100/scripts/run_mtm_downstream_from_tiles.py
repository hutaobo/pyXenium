from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "pyXenium").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute contour-scale morphomolecular translation artifacts from "
            "precomputed LazySlide tile features."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("PYXENIUM_ATERA_DATASET"),
        required=os.environ.get("PYXENIUM_ATERA_DATASET") is None,
    )
    parser.add_argument("--histoseg-geojson", default=os.environ.get("HISTOSEG_GEOJSON"))
    parser.add_argument("--contour-key", default="histoseg_structures")
    parser.add_argument("--contour-id-key", default="polygon_id")
    parser.add_argument("--coordinate-space", default="xenium_pixel")
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--he-image-key", default="he")
    parser.add_argument("--he-source-path", default=os.environ.get("PYXENIUM_HE_SOURCE_PATH"))
    parser.add_argument("--wsi-reader", default=os.environ.get("PYXENIUM_WSI_READER"))
    parser.add_argument("--model", default=os.environ.get("LAZYSLIDE_MODEL", "plip"))
    parser.add_argument("--text-model", default=os.environ.get("LAZYSLIDE_TEXT_MODEL", "plip"))
    parser.add_argument("--tile-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--table-format", choices=("csv", "parquet"), default="parquet")
    parser.add_argument("--skip-rna", action="store_true")
    parser.add_argument("--skip-wta-programs", action="store_true")
    parser.add_argument("--wta-program-library", default="breast_tme_wta_v1")
    parser.add_argument("--skip-boundary-programs", action="store_true")
    parser.add_argument("--skip-prediction-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    repo = _repo_root()
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from pyXenium.multimodal import run_histoseg_lazyslide_structure_workflow

    args = _parse_args()
    result = run_histoseg_lazyslide_structure_workflow(
        args.dataset_root,
        output_dir=args.output_dir,
        contour_key=args.contour_key,
        contour_geojson=args.histoseg_geojson,
        contour_id_key=args.contour_id_key,
        contour_coordinate_space=args.coordinate_space,
        contour_pixel_size_um=args.pixel_size_um,
        he_image_key=args.he_image_key,
        he_source_path=args.he_source_path,
        wsi_reader=args.wsi_reader,
        model=args.model,
        text_model=args.text_model,
        table_format=args.table_format,
        include_rna=not args.skip_rna,
        include_wta_programs=not args.skip_wta_programs,
        include_boundary_programs=not args.skip_boundary_programs,
        include_prediction_benchmark=not args.skip_prediction_benchmark,
        wta_program_library=args.wta_program_library,
        precomputed_tile_features=args.tile_features,
    )
    print(f"Wrote mTM artifacts to: {Path(args.output_dir).resolve()}")
    for key in [
        "contour_multimodal_summary",
        "wta_pathway_partial_correlations",
        "molecular_prediction_benchmark",
        "morphomolecular_hero_targets",
        "morphomolecular_hero_contours",
        "boundary_coupling_summary",
    ]:
        print(f"{key}: {result[key].shape}")


if __name__ == "__main__":
    main()
