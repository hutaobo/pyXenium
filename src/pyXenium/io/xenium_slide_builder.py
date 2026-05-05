from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform
from shapely.strtree import STRtree

from .api import read_xenium
from .slide_model import XeniumImage, XeniumSlide
from .slide_store import write_xenium_slide
from .xenium_artifacts import discover_image_artifacts, read_he_image, resolve_transcripts_path


ATERA_BREAST_CASE = "WTA_Preview_FFPE_Breast_Cancer_outs"
ATERA_CERVICAL_CASE = "WTA_Preview_FFPE_Cervical_Cancer_outs"
DEFAULT_MAX_CROP_SIDE_PX = 1024


@dataclass(frozen=True)
class XeniumSlideBuildResult:
    case_name: str
    output_dir: Path
    slide_store: Path
    slide_manifest: Path
    qc_report: Path
    cell_to_contour: Path | None
    structure_assignments: Path | None
    contour_patch_manifest: Path | None
    contour_patch_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "output_dir": str(self.output_dir),
            "xenium_slide": str(self.slide_store),
            "slide_manifest": str(self.slide_manifest),
            "qc_report": str(self.qc_report),
            "cell_to_contour": str(self.cell_to_contour) if self.cell_to_contour is not None else None,
            "structure_assignments": str(self.structure_assignments) if self.structure_assignments is not None else None,
            "contour_patches_manifest": (
                str(self.contour_patch_manifest) if self.contour_patch_manifest is not None else None
            ),
            "contour_patch_count": int(self.contour_patch_count),
        }


def build_xenium_slide(
    *,
    xenium_root: str | Path,
    output_dir: str | Path,
    case_name: str,
    organ: str | None = None,
    contour_geojson: str | Path | None = None,
    extract_contour_images: bool = False,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    patient_id: str | None = None,
    batch_id: str | None = None,
    stain: str | None = "H&E",
    scanner: str | None = None,
    prefer: str = "auto",
    overwrite: bool = True,
    source_metadata: dict[str, Any] | None = None,
    contour_source: dict[str, Any] | None = None,
) -> XeniumSlideBuildResult:
    """Build a canonical XeniumSlide store plus auditable contour artifacts."""

    root = Path(xenium_root).expanduser()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    slide = read_xenium(
        str(root),
        as_="slide",
        prefer=prefer,
        include_transcripts=False,
        stream_transcripts=True,
        include_boundaries=True,
        include_images=False,
    )
    _standardize_slide_table(
        slide,
        case_name=case_name,
        organ=organ,
        patient_id=patient_id,
        batch_id=batch_id,
        stain=stain,
        scanner=scanner,
    )

    transcript_path = resolve_transcripts_path(str(root))
    image_artifacts = discover_image_artifacts(str(root))
    slide.metadata.setdefault("schema", {})["name"] = "XeniumSlide"
    slide.metadata.setdefault("schema", {})["version"] = 1
    slide.metadata["case_name"] = case_name
    slide.metadata["organ"] = organ
    slide.metadata["batch"] = {
        "slide_id": case_name,
        "patient_id": patient_id,
        "batch_id": batch_id,
        "organ": organ,
        "stain": stain,
        "scanner": scanner,
    }
    slide.metadata["image_artifacts"] = image_artifacts
    slide.metadata["transcript_source"] = {
        "source_path": str(transcript_path) if transcript_path is not None else None,
        "materialized_in_slide_store": False,
    }
    if source_metadata:
        slide.metadata["tenx_source"] = dict(source_metadata)
    if contour_source:
        slide.metadata["contour_source"] = dict(contour_source)
    slide.table.uns.setdefault("xenium_slide", {}).update(
        {
            "schema_version": 1,
            "case_name": case_name,
            "organ": organ,
            "source_path": str(root),
            "slide_store": str(out / "xenium_slide.zarr"),
            "image_artifacts": image_artifacts,
            "transcript_source": slide.metadata["transcript_source"],
            "batch": slide.metadata["batch"],
        }
    )
    if source_metadata:
        slide.table.uns.setdefault("xenium_slide", {})["tenx_source"] = dict(source_metadata)
    if contour_source:
        slide.table.uns.setdefault("xenium_slide", {})["contour_source"] = dict(contour_source)

    contour_frame = pd.DataFrame()
    cell_to_contour = pd.DataFrame()
    patch_manifest: list[dict[str, Any]] = []
    contour_path = Path(contour_geojson).expanduser() if contour_geojson is not None else None
    if contour_path is not None:
        contour_frame, contour_features = load_contour_geojson(contour_path)
        slide.shapes["contours"] = contour_frame
        cell_to_contour = assign_cells_to_contours(
            slide.table,
            contour_features,
            pixel_size_um=_pixel_size_um_for_slide(slide, image_artifacts),
        )
        _attach_assignments_to_obs(slide.table, cell_to_contour)
        structure_assignments = _structure_assignments_from_contours(contour_frame)
        cell_to_contour_path = out / "cell_to_contour.parquet"
        structure_assignments_path = out / "structure_assignments.csv"
        cell_to_contour.to_parquet(cell_to_contour_path, index=False)
        structure_assignments.to_csv(structure_assignments_path, index=False)
        slide.metadata["contours"] = {
            "source_geojson": str(contour_path),
            "n_contours": int(len(contour_frame)),
            "n_assigned_cells": int((cell_to_contour["assignment_status"] == "assigned").sum()),
            "cell_to_contour": str(cell_to_contour_path),
            "structure_assignments": str(structure_assignments_path),
        }
        slide.table.uns.setdefault("xenium_slide", {})["contours"] = slide.metadata["contours"]

        if extract_contour_images:
            he_image = read_he_image(str(root))
            if he_image is None:
                raise RuntimeError(f"No aligned H&E image artifact was found under {root}.")
            patch_manifest = extract_contour_patches(
                features=contour_features,
                he_image=he_image,
                output_dir=out / "contour_patches",
                contour_geojson=contour_path,
                max_crop_side_px=max_crop_side_px,
            )
        else:
            existing_manifest = out / "contour_patches_manifest.json"
            if existing_manifest.exists():
                existing_payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
                if isinstance(existing_payload, list):
                    patch_manifest = existing_payload
    else:
        cell_to_contour = assign_cells_to_contours(
            slide.table,
            [],
            pixel_size_um=_pixel_size_um_for_slide(slide, image_artifacts),
        )
        _attach_assignments_to_obs(slide.table, cell_to_contour)
        structure_assignments = _structure_assignments_from_contours(contour_frame)
        cell_to_contour_path = out / "cell_to_contour.parquet"
        structure_assignments_path = out / "structure_assignments.csv"
        cell_to_contour.to_parquet(cell_to_contour_path, index=False)
        structure_assignments.to_csv(structure_assignments_path, index=False)
        slide.metadata["contours"] = {
            "source_geojson": None,
            "n_contours": 0,
            "n_assigned_cells": 0,
            "cell_to_contour": str(cell_to_contour_path),
            "structure_assignments": str(structure_assignments_path),
        }
        slide.table.uns.setdefault("xenium_slide", {})["contours"] = slide.metadata["contours"]

    patch_manifest_path = out / "contour_patches_manifest.json"
    _write_json(patch_manifest_path, patch_manifest)
    if patch_manifest:
        slide.metadata.setdefault("contours", {})["contour_patches_manifest"] = str(patch_manifest_path)
        slide.table.uns.setdefault("xenium_slide", {}).setdefault("contours", {})[
            "contour_patches_manifest"
        ] = str(patch_manifest_path)

    slide_store = out / "xenium_slide.zarr"
    write_payload = write_xenium_slide(slide, slide_store, overwrite=overwrite)
    manifest = build_slide_manifest(
        slide=slide,
        write_payload=write_payload,
        xenium_root=root,
        output_dir=out,
        contour_geojson=contour_path,
        contour_frame=contour_frame,
        cell_to_contour=cell_to_contour,
        patch_manifest=patch_manifest,
        max_crop_side_px=max_crop_side_px,
    )
    qc = build_slide_qc_report(manifest)
    manifest_path = out / "slide_manifest.json"
    qc_path = out / "qc_report.json"
    _write_json(manifest_path, manifest)
    _write_json(qc_path, qc)

    return XeniumSlideBuildResult(
        case_name=case_name,
        output_dir=out,
        slide_store=slide_store,
        slide_manifest=manifest_path,
        qc_report=qc_path,
        cell_to_contour=cell_to_contour_path,
        structure_assignments=structure_assignments_path,
        contour_patch_manifest=patch_manifest_path,
        contour_patch_count=len(patch_manifest),
    )


def build_atera_slides(
    *,
    atera_root: str | Path,
    output_root: str | Path,
    extract_contour_images: bool = True,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    overwrite: bool = True,
) -> dict[str, dict[str, Any]]:
    root = Path(atera_root).expanduser()
    output = Path(output_root).expanduser()
    breast = root / ATERA_BREAST_CASE
    cervical = root / ATERA_CERVICAL_CASE
    results = {
        "breast": build_xenium_slide(
            xenium_root=breast,
            output_dir=output / "breast",
            case_name="atera_wta_breast",
            organ="breast",
            contour_geojson=breast / "xenium_explorer_annotations.generated.geojson",
            extract_contour_images=extract_contour_images,
            max_crop_side_px=max_crop_side_px,
            overwrite=overwrite,
        ).to_dict(),
        "cervical": build_xenium_slide(
            xenium_root=cervical,
            output_dir=output / "cervical",
            case_name="atera_wta_cervical",
            organ="cervix",
            contour_geojson=cervical / "pyxenium_cervical_end_to_end" / "contours_bio6" / "xenium_explorer_annotations.geojson",
            extract_contour_images=extract_contour_images,
            max_crop_side_px=max_crop_side_px,
            overwrite=overwrite,
        ).to_dict(),
    }
    _write_json(output / "build_atera_manifest.json", results)
    return results


def load_contour_geojson(path: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raise ValueError(f"GeoJSON features must be a list: {source}")

    rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for index, feature in enumerate(raw_features, start=1):
        geometry = shape(feature.get("geometry"))
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            continue
        props = dict(feature.get("properties", {}) or {})
        contour_id = str(
            props.get("contour_id")
            or props.get("name")
            or props.get("polygon_id")
            or props.get("id")
            or f"contour_{index:05d}"
        )
        structure_label = str(props.get("assigned_structure") or props.get("structure_label") or props.get("label") or "")
        structure_id = props.get("structure_id")
        structure_id = int(structure_id) if structure_id is not None and str(structure_id) != "" else index
        min_x, min_y, max_x, max_y = geometry.bounds
        record = {
            "contour_id": contour_id,
            "structure_id": int(structure_id),
            "structure_label": structure_label or f"structure_{structure_id}",
            "component_index": props.get("component_index"),
            "polygon_index": props.get("polygon_index"),
            "geometry_wkt": geometry.wkt,
            "bbox_x0": float(min_x),
            "bbox_y0": float(min_y),
            "bbox_x1": float(max_x),
            "bbox_y1": float(max_y),
            "area": float(geometry.area),
            "source_geojson": str(source),
        }
        rows.append(record)
        features.append({"geometry": geometry, "properties": record})
    return pd.DataFrame(rows), features


def assign_cells_to_contours(
    adata: Any,
    contour_features: list[dict[str, Any]],
    *,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    spatial = _spatial_frame_from_adata(adata)
    if not contour_features:
        frame = spatial.copy()
        frame["contour_id"] = None
        frame["structure_id"] = None
        frame["structure_label"] = None
        frame["assignment_status"] = "unassigned"
        return frame

    geometries = [feature["geometry"] for feature in contour_features]
    assignment_coords, coordinate_space = _assignment_coordinates_for_contours(
        spatial,
        geometries,
        pixel_size_um=pixel_size_um,
    )
    tree = STRtree(geometries)
    rows: list[dict[str, Any]] = []
    for row, assignment_xy in zip(spatial.itertuples(index=False), assignment_coords, strict=False):
        point = Point(float(assignment_xy[0]), float(assignment_xy[1]))
        match_index = _first_covering_geometry(tree, geometries, point)
        if match_index is None:
            rows.append(
                {
                    "cell_id": row.cell_id,
                    "x": float(row.x),
                    "y": float(row.y),
                    "contour_x": float(assignment_xy[0]),
                    "contour_y": float(assignment_xy[1]),
                    "contour_coordinate_space": coordinate_space,
                    "contour_id": None,
                    "structure_id": None,
                    "structure_label": None,
                    "assignment_status": "unassigned",
                }
            )
            continue
        props = contour_features[match_index]["properties"]
        rows.append(
            {
                "cell_id": row.cell_id,
                "x": float(row.x),
                "y": float(row.y),
                "contour_x": float(assignment_xy[0]),
                "contour_y": float(assignment_xy[1]),
                "contour_coordinate_space": coordinate_space,
                "contour_id": props["contour_id"],
                "structure_id": props["structure_id"],
                "structure_label": props["structure_label"],
                "assignment_status": "assigned",
            }
        )
    return pd.DataFrame(rows)


def extract_contour_patches(
    *,
    features: list[dict[str, Any]],
    he_image: XeniumImage,
    output_dir: str | Path,
    contour_geojson: str | Path,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
) -> list[dict[str, Any]]:
    out = Path(output_dir).expanduser()
    records: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        props = dict(feature["properties"])
        contour_id = str(props.get("contour_id") or f"contour_{index:05d}")
        patch_path = out / f"{index:05d}_{_slug(contour_id)}.png"
        patch_meta = _extract_feature_patch(
            geometry=feature["geometry"],
            he_image=he_image,
            output_path=patch_path,
            max_side_px=max_crop_side_px,
        )
        records.append(
            {
                "contour_id": contour_id,
                "structure_id": props.get("structure_id"),
                "structure_label": props.get("structure_label"),
                "structure_name": props.get("structure_label"),
                "image_path": str(patch_path),
                "bbox": {
                    "x0": props.get("bbox_x0"),
                    "y0": props.get("bbox_y0"),
                    "x1": props.get("bbox_x1"),
                    "y1": props.get("bbox_y1"),
                    "coordinate_space": "xenium",
                },
                "pyramid_level": patch_meta["pyramid_level"],
                "transform": he_image.transform_metadata(),
                "source_geojson": str(Path(contour_geojson).expanduser()),
                "patch": patch_meta,
            }
        )
    return records


def build_slide_manifest(
    *,
    slide: XeniumSlide,
    write_payload: dict[str, Any],
    xenium_root: Path,
    output_dir: Path,
    contour_geojson: Path | None,
    contour_frame: pd.DataFrame,
    cell_to_contour: pd.DataFrame,
    patch_manifest: list[dict[str, Any]],
    max_crop_side_px: int,
) -> dict[str, Any]:
    adata = slide.table
    spatial = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else np.empty((0, 2))
    assigned_count = (
        int((cell_to_contour["assignment_status"] == "assigned").sum())
        if not cell_to_contour.empty and "assignment_status" in cell_to_contour
        else 0
    )
    return {
        "schema": {"name": "XeniumSlide", "version": 1},
        "case_name": slide.metadata.get("case_name"),
        "source_xenium_root": str(xenium_root),
        "output_dir": str(output_dir),
        "slide_store": str(output_dir / "xenium_slide.zarr"),
        "write_payload": write_payload,
        "counts": {
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "matrix_nonzero": int(adata.X.nnz if hasattr(adata.X, "nnz") else np.count_nonzero(adata.X)),
            "cell_boundaries": int(len(slide.shapes.get("cell_boundaries", []))),
            "nucleus_boundaries": int(len(slide.shapes.get("nucleus_boundaries", []))),
            "contours": int(len(contour_frame)),
            "assigned_cells": assigned_count,
            "contour_patches": int(len(patch_manifest)),
        },
        "spatial_bounds": _spatial_bounds(spatial),
        "panel": slide.table.uns.get("xenium_slide", {}).get("panel") or slide.metadata.get("feature_summary", {}),
        "metadata": slide.metadata.get("batch", {}),
        "tenx_source": slide.metadata.get("tenx_source", {}),
        "image_artifacts": slide.metadata.get("image_artifacts", {}),
        "contour_source": slide.metadata.get("contour_source", {}),
        "contours": {
            "source_geojson": str(contour_geojson) if contour_geojson is not None else None,
            "cell_assignment_coverage": float(assigned_count / max(int(adata.n_obs), 1)),
            "max_crop_side_px": int(max_crop_side_px),
        },
        "artifacts": {
            "cell_to_contour": str(output_dir / "cell_to_contour.parquet"),
            "structure_assignments": str(output_dir / "structure_assignments.csv"),
            "contour_patches_manifest": str(output_dir / "contour_patches_manifest.json"),
        },
    }


def build_slide_qc_report(manifest: dict[str, Any]) -> dict[str, Any]:
    fatal_errors: list[str] = []
    warnings: list[str] = []
    counts = manifest["counts"]
    if counts["cells"] <= 0:
        fatal_errors.append("No cells were loaded into the XeniumSlide table.")
    if counts["genes"] <= 0:
        fatal_errors.append("No genes were loaded into the XeniumSlide table.")
    if not manifest["image_artifacts"]:
        warnings.append("No H&E image artifact metadata was found.")
    if counts["contours"] == 0:
        warnings.append("No contour polygons were loaded.")
    elif manifest["contours"]["cell_assignment_coverage"] == 0:
        warnings.append("Contour polygons loaded, but no cells were assigned to contours.")
    if counts["contours"] > 0 and counts["contour_patches"] == 0:
        warnings.append("Contour patch extraction was not run or produced no patches.")
    return {
        "case_name": manifest.get("case_name"),
        "status": "fail" if fatal_errors else "pass",
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "metrics": {
            "n_cells": counts["cells"],
            "n_genes": counts["genes"],
            "n_contours": counts["contours"],
            "cell_assignment_coverage": manifest["contours"]["cell_assignment_coverage"],
            "contour_patch_count": counts["contour_patches"],
            "has_he_transform": bool(
                manifest.get("image_artifacts", {}).get("he", {}).get("image_to_xenium_affine")
            ),
        },
    }


def _standardize_slide_table(
    slide: XeniumSlide,
    *,
    case_name: str,
    organ: str | None,
    patient_id: str | None,
    batch_id: str | None,
    stain: str | None,
    scanner: str | None,
) -> None:
    adata = slide.table
    if "cell_id" not in adata.obs.columns:
        adata.obs["cell_id"] = adata.obs_names.astype(str)
    if "spatial" not in adata.obsm:
        spatial = _spatial_frame_from_adata(adata)
        adata.obsm["spatial"] = spatial[["x", "y"]].to_numpy(dtype=np.float32)
    for column, value in {
        "slide_id": case_name,
        "patient_id": patient_id,
        "organ": organ,
        "batch_id": batch_id or case_name,
        "stain": stain,
        "scanner": scanner,
    }.items():
        if value is not None and column not in adata.obs.columns:
            adata.obs[column] = str(value)
    if "feature_name" not in adata.var.columns:
        if "name" in adata.var.columns:
            adata.var["feature_name"] = adata.var["name"].astype(str)
        else:
            adata.var["feature_name"] = adata.var_names.astype(str)


def _spatial_frame_from_adata(adata: Any) -> pd.DataFrame:
    if "cell_id" in adata.obs.columns:
        cell_ids = adata.obs["cell_id"].astype(str).to_numpy()
    else:
        cell_ids = adata.obs_names.astype(str)
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    else:
        for x_col, y_col in (
            ("x_centroid", "y_centroid"),
            ("cell_x_centroid", "cell_y_centroid"),
            ("cell_centroid_x", "cell_centroid_y"),
            ("x", "y"),
        ):
            if x_col in adata.obs.columns and y_col in adata.obs.columns:
                coords = adata.obs[[x_col, y_col]].to_numpy(dtype=float)
                break
        else:
            raise ValueError("XeniumSlide table is missing spatial coordinates.")
    return pd.DataFrame({"cell_id": cell_ids, "x": coords[:, 0], "y": coords[:, 1]})


def _attach_assignments_to_obs(adata: Any, assignments: pd.DataFrame) -> None:
    if assignments.empty:
        return
    obs = adata.obs.copy()
    cell_ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else pd.Series(adata.obs_names.astype(str), index=obs.index)
    mapping = assignments.set_index("cell_id", drop=False)
    for column in ("contour_id", "structure_id", "structure_label", "assignment_status"):
        obs[column] = cell_ids.map(mapping[column]).to_numpy()
    adata.obs = obs


def _structure_assignments_from_contours(contour_frame: pd.DataFrame) -> pd.DataFrame:
    if contour_frame.empty:
        return pd.DataFrame(columns=["contour_id", "structure_id", "structure_label"])
    return contour_frame[["contour_id", "structure_id", "structure_label"]].drop_duplicates().copy()


def _pixel_size_um_for_slide(slide: XeniumSlide, image_artifacts: dict[str, Any] | None) -> float | None:
    he_artifact = (image_artifacts or {}).get("he", {}) if isinstance(image_artifacts, dict) else {}
    for value in (
        he_artifact.get("pixel_size_um") if isinstance(he_artifact, dict) else None,
        slide.metadata.get("experiment", {}).get("pixel_size") if isinstance(slide.metadata.get("experiment"), dict) else None,
        slide.table.uns.get("xenium_slide", {}).get("experiment", {}).get("pixel_size")
        if isinstance(slide.table.uns.get("xenium_slide"), dict)
        and isinstance(slide.table.uns.get("xenium_slide", {}).get("experiment"), dict)
        else None,
    ):
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return None


def _assignment_coordinates_for_contours(
    spatial: pd.DataFrame,
    geometries: list[Any],
    *,
    pixel_size_um: float | None,
) -> tuple[np.ndarray, str]:
    coords_um = spatial[["x", "y"]].to_numpy(dtype=float)
    if pixel_size_um is None or pixel_size_um <= 0:
        return coords_um, "xenium_um"
    coords_pixel = coords_um / float(pixel_size_um)
    raw_score = _bounds_match_score(coords_um, geometries)
    pixel_score = _bounds_match_score(coords_pixel, geometries)
    if pixel_score < raw_score:
        return coords_pixel, "xenium_pixel"
    return coords_um, "xenium_um"


def _bounds_match_score(coords: np.ndarray, geometries: list[Any]) -> float:
    if coords.size == 0 or not geometries:
        return float("inf")
    coord_min = np.nanmin(coords[:, :2], axis=0)
    coord_max = np.nanmax(coords[:, :2], axis=0)
    geom_bounds = np.asarray([geometry.bounds for geometry in geometries], dtype=float)
    geom_min = np.nanmin(geom_bounds[:, [0, 1]], axis=0)
    geom_max = np.nanmax(geom_bounds[:, [2, 3]], axis=0)
    coord_extent = np.maximum(coord_max - coord_min, 1e-9)
    geom_extent = np.maximum(geom_max - geom_min, 1e-9)
    extent_score = float(np.sum(np.abs(np.log(coord_extent / geom_extent))))
    coord_center = (coord_min + coord_max) / 2.0
    geom_center = (geom_min + geom_max) / 2.0
    center_score = float(np.linalg.norm((coord_center - geom_center) / geom_extent))
    return extent_score + center_score


def _first_covering_geometry(tree: STRtree, geometries: list[Any], point: Point) -> int | None:
    candidates = tree.query(point)
    for candidate in candidates:
        index = int(candidate)
        if geometries[index].covers(point):
            return index
    return None


def _extract_feature_patch(
    *,
    geometry: Polygon | MultiPolygon,
    he_image: XeniumImage,
    output_path: Path,
    max_side_px: int,
) -> dict[str, Any]:
    image_geometry_level0 = _geometry_xenium_pixel_to_image_xy(geometry, he_image)
    min_x, min_y, max_x, max_y = image_geometry_level0.bounds
    bbox_level0 = (
        int(np.floor(min_x)),
        int(np.floor(min_y)),
        int(np.ceil(max_x)),
        int(np.ceil(max_y)),
    )
    level_index, scale_x, scale_y = _select_pyramid_level(
        he_image,
        bbox_level0=bbox_level0,
        max_side_px=max_side_px,
    )
    image_geometry = affinity.scale(
        image_geometry_level0,
        xfact=1.0 / scale_x,
        yfact=1.0 / scale_y,
        origin=(0.0, 0.0),
    )
    level = he_image.levels[level_index]
    shape_at_level = tuple(int(value) for value in getattr(level, "shape", np.shape(level)))
    image_width = shape_at_level[he_image.axes.index("x")]
    image_height = shape_at_level[he_image.axes.index("y")]
    min_x, min_y, max_x, max_y = image_geometry.bounds
    bbox = (
        max(int(np.floor(min_x)), 0),
        max(int(np.floor(min_y)), 0),
        min(int(np.ceil(max_x)), image_width),
        min(int(np.ceil(max_y)), image_height),
    )
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Contour bbox does not intersect H&E image at level {level_index}: {bbox}")
    patch_array = _crop_image_level(level, axes=he_image.axes, bbox=bbox)
    mask = _polygon_mask_for_bbox(image_geometry=image_geometry, bbox=bbox)
    masked_patch = _apply_polygon_mask(patch_array, mask=mask, axes=he_image.axes)
    patch_meta = _save_patch(
        SimpleNamespace(levels=[masked_patch], axes=he_image.axes),
        output_path,
        max_side_px=max_side_px,
    )
    patch_meta.update(
        {
            "pyramid_level": int(level_index),
            "level_downsample_x": float(scale_x),
            "level_downsample_y": float(scale_y),
            "bbox_level_xy": [int(value) for value in bbox],
            "bbox_level0_xy": [int(value) for value in bbox_level0],
        }
    )
    return patch_meta


def _geometry_xenium_pixel_to_image_xy(geometry: Polygon | MultiPolygon, he_image: XeniumImage) -> Polygon | MultiPolygon:
    if he_image.image_to_xenium_affine is None:
        raise ValueError("H&E image is missing image_to_xenium_affine metadata.")
    inverse = np.linalg.inv(np.asarray(he_image.image_to_xenium_affine, dtype=float))

    def transform_xy(x: Any, y: Any, z: Any | None = None) -> tuple[Any, Any] | tuple[Any, Any, Any]:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        flat = np.column_stack([x_array.reshape(-1), y_array.reshape(-1), np.ones(x_array.size)])
        transformed = flat @ inverse.T
        out_x = transformed[:, 0].reshape(x_array.shape)
        out_y = transformed[:, 1].reshape(y_array.shape)
        if z is None:
            return out_x, out_y
        return out_x, out_y, z

    return transform(transform_xy, geometry)


def _select_pyramid_level(he_image: XeniumImage, *, bbox_level0: tuple[int, int, int, int], max_side_px: int) -> tuple[int, float, float]:
    x0, y0, x1, y1 = bbox_level0
    max_side = max(int(x1 - x0), int(y1 - y0), 1)
    target_scale = max(float(max_side) / max(float(max_side_px), 1.0), 1.0)
    shapes = he_image.multiscale_shapes()
    x_index = he_image.axes.index("x")
    y_index = he_image.axes.index("y")
    level0_shape = shapes[0]
    selected = len(shapes) - 1
    selected_scale_x = float(level0_shape[x_index]) / float(shapes[selected][x_index])
    selected_scale_y = float(level0_shape[y_index]) / float(shapes[selected][y_index])
    for level_index, shape_at_level in enumerate(shapes):
        scale_x = float(level0_shape[x_index]) / float(shape_at_level[x_index])
        scale_y = float(level0_shape[y_index]) / float(shape_at_level[y_index])
        if max(scale_x, scale_y) >= target_scale:
            selected = level_index
            selected_scale_x = scale_x
            selected_scale_y = scale_y
            break
    return selected, selected_scale_x, selected_scale_y


def _crop_image_level(level: Any, *, axes: str, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    slices = [slice(None)] * len(level.shape)
    slices[axes.index("x")] = slice(x0, x1)
    slices[axes.index("y")] = slice(y0, y1)
    try:
        return np.asarray(level[tuple(slices)]).copy()
    except TypeError:
        if hasattr(level, "open_zarr_source"):
            store, source = level.open_zarr_source()
            try:
                return np.asarray(source[tuple(slices)]).copy()
            finally:
                if hasattr(store, "close"):
                    store.close()
        if hasattr(level, "asarray"):
            return np.asarray(level.asarray()[tuple(slices)]).copy()
        raise


def _polygon_mask_for_bbox(*, image_geometry: Polygon | MultiPolygon, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    width = int(x1 - x0)
    height = int(y1 - y0)
    local_geometry = affinity.translate(image_geometry, xoff=-float(x0), yoff=-float(y0))
    yy, xx = np.mgrid[0:height, 0:width]
    sample_points = np.column_stack([xx.reshape(-1).astype(float) + 0.5, yy.reshape(-1).astype(float) + 0.5])
    mask = np.zeros(sample_points.shape[0], dtype=bool)
    polygons = list(local_geometry.geoms) if isinstance(local_geometry, MultiPolygon) else [local_geometry]
    for polygon in polygons:
        mask |= _polygon_contains_points(polygon, sample_points)
    return mask.reshape(height, width)


def _polygon_contains_points(polygon: Polygon, sample_points: np.ndarray) -> np.ndarray:
    exterior_path = MplPath(np.asarray(polygon.exterior.coords, dtype=float))
    contained = exterior_path.contains_points(sample_points, radius=1e-9)
    for interior in polygon.interiors:
        hole_path = MplPath(np.asarray(interior.coords, dtype=float))
        contained &= ~hole_path.contains_points(sample_points)
    return contained


def _apply_polygon_mask(patch_array: np.ndarray, *, mask: np.ndarray, axes: str) -> np.ndarray:
    y_index = axes.index("y")
    x_index = axes.index("x")
    broadcast_shape = [1] * patch_array.ndim
    broadcast_shape[y_index] = mask.shape[0]
    broadcast_shape[x_index] = mask.shape[1]
    broadcast_mask = mask.reshape(broadcast_shape)
    fill_value = np.zeros((), dtype=patch_array.dtype)
    return np.where(broadcast_mask, patch_array, fill_value)


def _save_patch(image: Any, output_path: Path, *, max_side_px: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _to_rgb(np.asarray(image.levels[0]), image.axes)
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Saving H&E contour patches requires Pillow.") from exc
    pil = Image.fromarray(rgb)
    original_size = pil.size
    if max(pil.size) > int(max_side_px):
        pil.thumbnail((int(max_side_px), int(max_side_px)), Image.Resampling.LANCZOS)
    pil.save(output_path)
    nonzero = float(np.count_nonzero(rgb.sum(axis=-1) > 3) / max(rgb.shape[0] * rgb.shape[1], 1))
    return {
        "path": str(output_path),
        "original_width": int(original_size[0]),
        "original_height": int(original_size[1]),
        "saved_width": int(pil.size[0]),
        "saved_height": int(pil.size[1]),
        "nonzero_fraction": nonzero,
    }


def _to_rgb(array: np.ndarray, axes: str) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 2:
        rgb = np.stack([data, data, data], axis=-1)
    elif "c" in axes:
        data = np.moveaxis(data, axes.index("c"), -1)
        if data.shape[-1] == 1:
            rgb = np.repeat(data, 3, axis=-1)
        else:
            rgb = data[..., :3]
    else:
        rgb = data[..., :3] if data.shape[-1] >= 3 else np.repeat(data[..., :1], 3, axis=-1)
    if rgb.dtype != np.uint8:
        rgb = np.asarray(rgb, dtype=float)
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _spatial_bounds(spatial: np.ndarray) -> dict[str, float | None]:
    if spatial.size == 0:
        return {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
    coords = np.asarray(spatial[:, :2], dtype=float)
    return {
        "x_min": float(np.nanmin(coords[:, 0])),
        "x_max": float(np.nanmax(coords[:, 0])),
        "y_min": float(np.nanmin(coords[:, 1])),
        "y_max": float(np.nanmax(coords[:, 1])),
    }


def _slug(value: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (slug or "contour")[:max_len]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return path
