#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyXenium.multimodal import run_bmnet_morphology_increment_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BM-Net/H&E morphology increment pilot.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contour-geojson", default=None)
    parser.add_argument("--contour-key", default="s1_s5_contours")
    parser.add_argument("--contour-id-key", default="polygon_id")
    parser.add_argument("--contour-coordinate-space", default="xenium_pixel")
    parser.add_argument("--contour-pixel-size-um", type=float, default=None)
    parser.add_argument("--he-image-key", default="he")
    parser.add_argument("--cells-parquet", default="cells.parquet")
    parser.add_argument("--clusters-relpath", default="WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv")
    parser.add_argument("--cluster-column-name", default="cluster")
    parser.add_argument(
        "--backend",
        default="deterministic-smoke",
        choices=["deterministic-smoke", "bmnet-local", "bmnet-like-trainable", "hf-pathology-backbone"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-model", default="1aurent/vit_small_patch8_224.lunit_dino")
    parser.add_argument("--timm-architecture", default="mobilenetv3_small_100")
    parser.add_argument("--timm-pretrained", action="store_true")
    parser.add_argument("--max-contours", type=int, default=None)
    parser.add_argument("--inner-rim-um", type=float, default=20.0)
    parser.add_argument("--outer-rim-um", type=float, default=30.0)
    parser.add_argument("--skip-pathomics", action="store_true")
    parser.add_argument("--include-transcripts", action="store_true")
    parser.add_argument("--program-library", default="breast_boundary_bmnet_v1")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--min-contours", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_bmnet_morphology_increment_pilot(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        contour_geojson=args.contour_geojson,
        contour_key=args.contour_key,
        contour_id_key=args.contour_id_key,
        contour_coordinate_space=args.contour_coordinate_space,
        contour_pixel_size_um=args.contour_pixel_size_um,
        he_image_key=args.he_image_key,
        cells_parquet=args.cells_parquet,
        clusters_relpath=args.clusters_relpath,
        cluster_column_name=args.cluster_column_name,
        backend=args.backend,
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        timm_architecture=args.timm_architecture,
        timm_pretrained=args.timm_pretrained,
        max_contours=args.max_contours,
        inner_rim_um=args.inner_rim_um,
        outer_rim_um=args.outer_rim_um,
        include_pathomics=not args.skip_pathomics,
        include_transcripts=args.include_transcripts,
        program_library=args.program_library,
        random_state=args.random_state,
        min_contours=args.min_contours,
    )
    summary = result["summary"]
    print(json.dumps({"artifact_dir": result["artifact_dir"], "summary": summary}, indent=2, default=str))
    summary_path = Path(summary["artifact_files"]["run_summary"])
    return 0 if summary_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
