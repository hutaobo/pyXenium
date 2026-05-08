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
        description="Run pyXenium HistoSeg + LazySlide structure image features.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("PYXENIUM_ATERA_DATASET"),
        required=os.environ.get("PYXENIUM_ATERA_DATASET") is None,
        help="Xenium export root. Defaults to PYXENIUM_ATERA_DATASET.",
    )
    parser.add_argument(
        "--histoseg-geojson",
        default=os.environ.get("HISTOSEG_GEOJSON"),
        help="HistoSeg structure GeoJSON, if contour_key is not already present.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("A100_OUTPUT_DIR", "runs/histoseg_lazyslide_breast_wta"),
        help="Output artifact directory. Defaults to A100_OUTPUT_DIR.",
    )
    parser.add_argument("--contour-key", default="histoseg_structures")
    parser.add_argument("--contour-id-key", default="polygon_id")
    parser.add_argument("--coordinate-space", default="xenium_pixel")
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--he-image-key", default="he")
    parser.add_argument(
        "--he-source-path",
        default=os.environ.get("PYXENIUM_HE_SOURCE_PATH"),
        help="Optional H&E WSI path override for slide stores with stale source_path metadata.",
    )
    parser.add_argument(
        "--wsi-reader",
        default=os.environ.get("PYXENIUM_WSI_READER"),
        help="Optional WSIData reader override, for example 'tiffslide'.",
    )
    parser.add_argument(
        "--slide-mpp",
        type=float,
        default=float(os.environ["PYXENIUM_SLIDE_MPP"])
        if os.environ.get("PYXENIUM_SLIDE_MPP")
        else None,
        help="Optional physical pixel size of the H&E source image.",
    )
    parser.add_argument("--model", default=os.environ.get("LAZYSLIDE_MODEL", "plip"))
    parser.add_argument(
        "--text-model",
        default=os.environ.get("LAZYSLIDE_TEXT_MODEL"),
        help=(
            "Optional LazySlide vision-language model for tile prompt scoring. "
            "Use 'none' to disable. If omitted, PLIP/CONCH/OmiCLIP models score "
            "their own prompts and vision-only models skip prompt scoring."
        ),
    )
    parser.add_argument(
        "--text-terms",
        nargs="*",
        default=None,
        help="Optional prompt terms for PLIP/CONCH/OmiCLIP tile text-image similarity.",
    )
    parser.add_argument("--prompt-set-name", default="breast_histology_v1")
    parser.add_argument("--prompt-source", default="manual exploratory prompt set")
    parser.add_argument("--prompt-review-status", default="not pathologist-confirmed")
    parser.add_argument("--tile-px", type=int, default=224)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=os.environ.get("LAZYSLIDE_DEVICE", "cuda"))
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--table-format", choices=("csv", "parquet"), default="parquet")
    parser.add_argument("--skip-rna", action="store_true")
    parser.add_argument("--skip-boundary-programs", action="store_true")
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
        slide_mpp=args.slide_mpp,
        model=args.model,
        text_model=args.text_model,
        text_terms=args.text_terms,
        prompt_set_name=args.prompt_set_name,
        prompt_source=args.prompt_source,
        prompt_review_status=args.prompt_review_status,
        tile_px=args.tile_px,
        mpp=args.mpp,
        batch_size=args.batch_size,
        device=args.device,
        max_tiles=args.max_tiles,
        table_format=args.table_format,
        include_rna=not args.skip_rna,
        include_boundary_programs=not args.skip_boundary_programs,
    )
    manifest = Path(args.output_dir) / "run_manifest.json"
    print(f"Wrote HistoSeg + LazySlide artifacts to: {Path(args.output_dir).resolve()}")
    print(f"Manifest: {manifest.resolve()}")
    print(f"Assigned tiles: {result['run_manifest']['outputs']['n_assigned_tiles']}")


if __name__ == "__main__":
    main()
