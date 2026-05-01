from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon, shape

from pyXenium.contour.generation import generate_xenium_explorer_annotations

from .xenium_artifacts import discover_image_artifacts, read_experiment_metadata
from .xenium_slide_builder import DEFAULT_MAX_CROP_SIDE_PX, build_xenium_slide


TENX_DATASETS_URL = "https://www.10xgenomics.com/datasets"
TENX_DATASET_URL_PREFIX = f"{TENX_DATASETS_URL}/"
DEFAULT_OUTPUT_ROOT = Path(r"D:\GitHub\stGPT\outputs\xenium_slides\10x_public")
DEFAULT_XENIUM_ROOT = Path(r"Y:\long\10X_datasets\Xenium")


@dataclass(frozen=True)
class TenXDatasetRecord:
    xenium_root: str
    relative_path: str
    collection_slug: str
    case_slug: str
    case_name: str
    buildable: bool
    exclusion_reason: str | None
    experiment_uuid: str | None
    analysis_uuid: str | None
    run_name: str | None
    panel_name: str | None
    has_matrix: bool
    matrix_backend: str | None
    has_cells: bool
    has_he: bool
    has_alignment: bool
    selected_geojson: str | None
    geojson_status: str
    geojson_candidate_count: int
    selected_for_build: bool = True
    duplicate_of: str | None = None
    reused_output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_10x_xenium_datasets(
    xenium_root: str | Path = DEFAULT_XENIUM_ROOT,
    *,
    reuse_atera: str | Path | None = None,
    include_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """Discover buildable public 10x Xenium outs under a local download root."""

    root = Path(xenium_root).expanduser()
    records = [_inventory_xenium_root(path.parent, root_base=root) for path in root.rglob("experiment.xenium")]
    selected = _mark_duplicate_records(records)
    if reuse_atera is not None:
        selected = _mark_reused_atera(selected, Path(reuse_atera).expanduser())
    if include_duplicates:
        return [record.to_dict() for record in selected]
    return [record.to_dict() for record in selected if record.selected_for_build]


def build_10x_public_slides(
    *,
    xenium_root: str | Path = DEFAULT_XENIUM_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reuse_atera: str | Path | None = None,
    histoseg_root: str | Path | None = None,
    metadata_cache: Mapping[str, Any] | str | Path | None = None,
    extract_contour_images: bool = True,
    generate_missing_contours: bool = True,
    max_crop_side_px: int = DEFAULT_MAX_CROP_SIDE_PX,
    overwrite: bool = False,
    refresh_metadata: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Build XeniumSlide artifacts for all discovered public 10x Xenium cases."""

    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    cache_payload = _metadata_cache_payload(metadata_cache)

    records = discover_10x_xenium_datasets(
        xenium_root,
        reuse_atera=reuse_atera,
        include_duplicates=True,
    )
    registry_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        row = dict(record)
        if progress:
            print(f"[{index}/{len(records)}] {row.get('relative_path')}...", flush=True)
        if not row.get("selected_for_build"):
            row["build_status"] = "duplicate_skipped"
            registry_rows.append(row)
            _checkpoint_registry(output, registry_rows, failed_rows, metadata_rows)
            continue
        metadata = _resolve_metadata_for_row(
            row,
            cache_payload=cache_payload,
            refresh_metadata=refresh_metadata,
        )
        row.update(_metadata_registry_fields(metadata))
        metadata_rows.append(_metadata_report_row(row, metadata))
        if row.get("reused_output_dir"):
            row["build_status"] = "reused_existing"
            contour_source = _contour_source_from_record(row)
            row["contour_status"] = contour_source.get("status")
            _ensure_case_artifact_contract(
                Path(str(row["reused_output_dir"])),
                metadata=metadata,
                contour_source=contour_source,
            )
            _attach_reused_artifact_status(row)
            registry_rows.append(row)
            _checkpoint_registry(output, registry_rows, failed_rows, metadata_rows)
            continue
        if not row.get("buildable"):
            row["build_status"] = "not_buildable"
            failed_rows.append(row)
            registry_rows.append(row)
            _checkpoint_registry(output, registry_rows, failed_rows, metadata_rows)
            continue

        case_output = output / str(row["collection_slug"]) / str(row["case_slug"])
        existing_manifest = case_output / "slide_manifest.json"
        if existing_manifest.exists() and not overwrite:
            row["build_status"] = "skipped_existing"
            row["output_dir"] = str(case_output)
            row["metadata_10x"] = str(case_output / "metadata_10x.json")
            row["contour_source_manifest"] = str(case_output / "contour_source_manifest.json")
            contour_source = _contour_source_from_record(row)
            row["contour_status"] = contour_source.get("status")
            _ensure_case_artifact_contract(
                case_output,
                metadata=metadata,
                contour_source=contour_source,
            )
            row.update(_summarize_existing_case(case_output))
            registry_rows.append(row)
            _checkpoint_registry(output, registry_rows, failed_rows, metadata_rows)
            continue

        contour_source = _resolve_contour_source_for_build(
            row,
            output_dir=case_output,
            histoseg_root=histoseg_root,
            generate_missing_contours=generate_missing_contours,
        )
        contour_path = contour_source.get("selected_geojson")
        case_output.mkdir(parents=True, exist_ok=True)
        _write_json(case_output / "metadata_10x.json", metadata)
        _write_json(case_output / "contour_source_manifest.json", contour_source)
        row["metadata_10x"] = str(case_output / "metadata_10x.json")
        row["contour_source_manifest"] = str(case_output / "contour_source_manifest.json")
        row["contour_status"] = contour_source.get("status")

        try:
            case_overwrite = bool(overwrite or ((case_output / "xenium_slide.zarr").exists() and not existing_manifest.exists()))
            result = build_xenium_slide(
                xenium_root=row["xenium_root"],
                output_dir=case_output,
                case_name=str(row["case_name"]),
                organ=_metadata_field(metadata, "Anatomical Entity") or _infer_organ_from_name(str(row["case_name"])),
                contour_geojson=contour_path if contour_path else None,
                extract_contour_images=bool(
                    extract_contour_images
                    and contour_path
                    and row.get("has_he")
                    and row.get("has_alignment")
                ),
                max_crop_side_px=max_crop_side_px,
                overwrite=case_overwrite,
                source_metadata=metadata,
                contour_source=contour_source,
            )
            row["build_status"] = "built"
            row["output_dir"] = str(case_output)
            row["slide_manifest"] = str(result.slide_manifest)
            row["qc_report"] = str(result.qc_report)
            row["contour_patch_count"] = int(result.contour_patch_count)
            row.update(_summarize_existing_case(case_output))
            _ensure_case_artifact_contract(case_output)
        except Exception as exc:
            row["build_status"] = "failed"
            row["output_dir"] = str(case_output)
            row["error"] = f"{type(exc).__name__}: {exc}"
            failed_rows.append(row)
        registry_rows.append(row)
        if progress:
            print(f"[{index}/{len(records)}] {row.get('build_status')} {row.get('case_name')}", flush=True)
        _checkpoint_registry(output, registry_rows, failed_rows, metadata_rows)

    _write_registry_outputs(output, registry_rows, failed_rows, metadata_rows)
    return _build_summary(registry_rows, failed_rows, output)


def resolve_10x_dataset_metadata(
    record_or_root: Mapping[str, Any] | str | Path,
    *,
    metadata_cache: Mapping[str, Any] | str | Path | None = None,
    allow_network: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    """Resolve official 10x dataset metadata for a local Xenium output."""

    record = _record_mapping(record_or_root)
    cache_payload = _metadata_cache_payload(metadata_cache)
    cached = _lookup_metadata_cache(record, cache_payload)
    if cached is not None:
        payload = dict(cached)
        payload.setdefault("metadata_status", "resolved")
        payload.setdefault("resolver", {})["method"] = "cache"
        return payload

    candidates = _candidate_10x_urls(record)
    if allow_network:
        session = requests.Session()
        for url in candidates:
            try:
                response = session.get(url, timeout=timeout)
            except Exception:
                continue
            if response.status_code != 200 or "10x Genomics" not in response.text:
                continue
            parsed = _parse_10x_dataset_page(response.text, url)
            if parsed.get("title") and "Page Not Found" not in str(parsed.get("title")):
                parsed["metadata_status"] = "resolved"
                parsed["resolver"] = {
                    "method": "official_10x_candidate_url",
                    "candidate_urls": candidates,
                    "matched_url": url,
                    "retrieved_at": _utc_now(),
                }
                return parsed

    return {
        "metadata_status": "unresolved",
        "source": "10x Genomics datasets",
        "source_url": None,
        "source_index_url": TENX_DATASETS_URL,
        "resolver": {
            "method": "candidate_url",
            "candidate_urls": candidates,
            "retrieved_at": _utc_now(),
            "notes": "No official 10x dataset page matched confidently.",
        },
        "local_experiment": _local_experiment_summary(record),
    }


def select_primary_contour_geojson(
    xenium_root: str | Path,
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    """Select the primary Xenium Explorer contour GeoJSON from a case directory."""

    root = Path(xenium_root).expanduser()
    candidates = _discover_geojson_candidates(root, recursive=recursive)
    valid = [item for item in candidates if item["valid"]]
    if not candidates:
        return {
            "status": "missing",
            "selected_geojson": None,
            "reason": "No GeoJSON files were found.",
            "candidates": [],
        }
    if not valid:
        return {
            "status": "no_valid_geojson",
            "selected_geojson": None,
            "reason": "GeoJSON files were found, but none contained polygon features.",
            "candidates": candidates,
        }
    selected = sorted(
        valid,
        key=lambda item: (
            int(item["priority"]),
            -int(item["polygon_feature_count"]),
            -int(item["size_bytes"]),
            item["relative_path"].lower(),
        ),
    )[0]
    return {
        "status": "selected",
        "selected_geojson": selected["path"],
        "reason": selected["priority_reason"],
        "candidates": candidates,
    }


def generate_missing_contours_with_histoseg(
    xenium_root: str | Path,
    output_dir: str | Path,
    *,
    histoseg_root: str | Path | None = None,
    min_cells: int = 500,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Generate a Xenium Explorer GeoJSON using HistoSeg when no annotations exist."""

    root = Path(xenium_root).expanduser()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    existing_geojson = out / "xenium_explorer_annotations.geojson"
    if existing_geojson.exists():
        return {
            "status": "generated_existing",
            "selected_geojson": str(existing_geojson),
            "reason": "Reused existing HistoSeg-generated contour GeoJSON.",
        }
    try:
        cluster_info = _find_cluster_csv(root)
        cells_path = _ensure_cells_parquet_for_histoseg(root, out / "histoseg_inputs")
        if cluster_info is None or cells_path is None:
            return {
                "status": "missing_inputs",
                "selected_geojson": None,
                "reason": "HistoSeg contour generation requires a cluster CSV and a cells table.",
            }
        structures = _infer_histoseg_structures(cluster_info["path"], cluster_info["cluster_col"], min_cells=min_cells)
        if not structures:
            return {
                "status": "missing_inputs",
                "selected_geojson": None,
                "reason": "No cluster labels had enough cells for HistoSeg contour generation.",
                "cluster_csv": str(cluster_info["path"]),
            }
        generated = generate_xenium_explorer_annotations(
            root,
            structures=structures,
            output_relpath=out,
            clusters_relpath=cluster_info["path"],
            cells_parquet_relpath=cells_path,
            histoseg_root=histoseg_root,
            barcode_col=cluster_info["barcode_col"],
            cluster_col=cluster_info["cluster_col"],
            min_cells=min_cells,
        )
        return {
            "status": "generated",
            "selected_geojson": generated.get("geojson"),
            "reason": "Generated missing contour GeoJSON with HistoSeg.",
            "histoseg_outputs": generated,
            "cluster_csv": str(cluster_info["path"]),
            "cells_parquet": str(cells_path),
            "structures": structures,
        }
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "status": "failed",
            "selected_geojson": None,
            "reason": f"HistoSeg contour generation failed: {type(exc).__name__}: {exc}",
        }


def _inventory_xenium_root(root: Path, *, root_base: Path) -> TenXDatasetRecord:
    rel = _safe_relative(root, root_base)
    experiment = _safe_read_experiment(root)
    matrix_backend = _matrix_backend(root)
    has_cells = _has_cells_table(root)
    image_artifacts = _safe_discover_image_artifacts(root)
    contour = select_primary_contour_geojson(root)
    buildable = bool(matrix_backend and has_cells)
    collection_slug = _slug(rel.parts[0] if rel.parts else "root")
    case_slug = _case_slug(root)
    return TenXDatasetRecord(
        xenium_root=str(root),
        relative_path=str(rel),
        collection_slug=collection_slug,
        case_slug=case_slug,
        case_name=case_slug,
        buildable=buildable,
        exclusion_reason=None if buildable else _exclusion_reason(matrix_backend, has_cells),
        experiment_uuid=_string_or_none(experiment.get("experiment_uuid")),
        analysis_uuid=_string_or_none(experiment.get("analysis_uuid")),
        run_name=_string_or_none(experiment.get("run_name")),
        panel_name=_string_or_none(experiment.get("panel_name")),
        has_matrix=matrix_backend is not None,
        matrix_backend=matrix_backend,
        has_cells=has_cells,
        has_he="he" in image_artifacts,
        has_alignment=bool(image_artifacts.get("he", {}).get("alignment_csv_path")),
        selected_geojson=contour.get("selected_geojson"),
        geojson_status=str(contour.get("status")),
        geojson_candidate_count=len(contour.get("candidates", [])),
    )


def _mark_duplicate_records(records: list[TenXDatasetRecord]) -> list[TenXDatasetRecord]:
    groups: dict[str, list[TenXDatasetRecord]] = {}
    for record in records:
        key = (
            f"uuid-case::{record.experiment_uuid}::{record.case_slug}"
            if record.experiment_uuid
            else f"path::{record.xenium_root}"
        )
        groups.setdefault(key, []).append(record)

    marked: list[TenXDatasetRecord] = []
    for group in groups.values():
        if len(group) == 1:
            marked.append(group[0])
            continue
        selected = sorted(group, key=_record_rank, reverse=True)[0]
        for record in group:
            if record is selected:
                marked.append(record)
            else:
                marked.append(
                    TenXDatasetRecord(
                        **{
                            **record.to_dict(),
                            "selected_for_build": False,
                            "duplicate_of": selected.xenium_root,
                        }
                    )
                )
    return sorted(marked, key=lambda item: item.relative_path.lower())


def _mark_reused_atera(records: list[TenXDatasetRecord], reuse_root: Path) -> list[TenXDatasetRecord]:
    reused: list[TenXDatasetRecord] = []
    for record in records:
        root_name = Path(record.xenium_root).name.lower()
        target = None
        if "atera" in record.relative_path.lower() and "breast" in root_name and "unzipped" not in record.relative_path.lower():
            target = reuse_root / "breast"
        elif "atera" in record.relative_path.lower() and "cervical" in root_name and "unzipped" not in record.relative_path.lower():
            target = reuse_root / "cervical"
        if target is not None and target.exists():
            reused.append(
                TenXDatasetRecord(
                    **{
                        **record.to_dict(),
                        "selected_for_build": True,
                        "reused_output_dir": str(target),
                    }
                )
            )
        else:
            reused.append(record)
    return reused


def _resolve_metadata_for_row(
    row: Mapping[str, Any],
    *,
    cache_payload: Mapping[str, Any],
    refresh_metadata: bool,
) -> dict[str, Any]:
    metadata = resolve_10x_dataset_metadata(
        row,
        metadata_cache=cache_payload,
        allow_network=not refresh_metadata,
    )
    if refresh_metadata or metadata.get("metadata_status") == "unresolved":
        metadata = resolve_10x_dataset_metadata(row, metadata_cache={}, allow_network=True)
    return metadata


def _metadata_registry_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metadata_status": metadata.get("metadata_status"),
        "metadata_source_url": metadata.get("source_url"),
        "metadata_source_index_url": metadata.get("source_index_url"),
        "metadata_title": metadata.get("title"),
        "metadata_species": _metadata_field(metadata, "Species"),
        "metadata_anatomical_entity": _metadata_field(metadata, "Anatomical Entity"),
        "metadata_disease_state": _metadata_field(metadata, "Disease State"),
    }


def _contour_source_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    selected = record.get("selected_geojson")
    if selected:
        return {
            "status": "existing_selected",
            "selected_geojson": selected,
            "reason": "Selected primary existing GeoJSON.",
            "geojson_status": record.get("geojson_status"),
            "geojson_candidate_count": record.get("geojson_candidate_count"),
        }
    return {
        "status": "missing_or_failed",
        "selected_geojson": None,
        "reason": "No contour GeoJSON was available for this case.",
        "geojson_status": record.get("geojson_status"),
        "geojson_candidate_count": record.get("geojson_candidate_count"),
    }


def _resolve_contour_source_for_build(
    record: Mapping[str, Any],
    *,
    output_dir: Path,
    histoseg_root: str | Path | None,
    generate_missing_contours: bool,
) -> dict[str, Any]:
    selected = record.get("selected_geojson")
    if selected:
        return {
            "status": "existing_selected",
            "selected_geojson": selected,
            "reason": "Selected primary existing GeoJSON.",
            "geojson_status": record.get("geojson_status"),
            "geojson_candidate_count": record.get("geojson_candidate_count"),
        }
    if generate_missing_contours and record.get("has_he"):
        generated = generate_missing_contours_with_histoseg(
            record["xenium_root"],
            output_dir / "histoseg_contours",
            histoseg_root=histoseg_root,
        )
        generated.setdefault("geojson_status", record.get("geojson_status"))
        return generated
    return {
        "status": "missing_or_failed",
        "selected_geojson": None,
        "reason": "No existing GeoJSON was selected and HistoSeg generation was skipped or lacks H&E inputs.",
        "geojson_status": record.get("geojson_status"),
        "geojson_candidate_count": record.get("geojson_candidate_count"),
    }


def _discover_geojson_candidates(root: Path, *, recursive: bool) -> list[dict[str, Any]]:
    iterator = root.rglob("*.geojson") if recursive else root.glob("*.geojson")
    candidates: list[dict[str, Any]] = []
    for path in iterator:
        if path.name.startswith("._"):
            continue
        candidates.append(_geojson_candidate(root, path))
    return sorted(candidates, key=lambda item: item["relative_path"].lower())


def _geojson_candidate(root: Path, path: Path) -> dict[str, Any]:
    priority, reason = _geojson_priority(path)
    feature_count = 0
    polygon_count = 0
    valid = False
    error = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features", []) if isinstance(payload, dict) else []
        feature_count = len(features) if isinstance(features, list) else 0
        for feature in features if isinstance(features, list) else []:
            geometry = shape(feature.get("geometry"))
            if isinstance(geometry, (Polygon, MultiPolygon)):
                polygon_count += 1
        valid = polygon_count > 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "relative_path": str(_safe_relative(path, root)),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "feature_count": int(feature_count),
        "polygon_feature_count": int(polygon_count),
        "valid": bool(valid),
        "priority": int(priority),
        "priority_reason": reason,
        "error": error,
    }


def _geojson_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name == "xenium_explorer_annotations.generated.geojson":
        return 0, "Preferred generated Xenium Explorer annotations file."
    if name == "xenium_explorer_annotations.geojson":
        return 1, "Preferred canonical Xenium Explorer annotations file."
    if "annotation" in name:
        return 2, "Preferred annotation-like GeoJSON filename."
    return 3, "Fallback to largest valid polygon GeoJSON."


def _matrix_backend(root: Path) -> str | None:
    if (root / "cell_feature_matrix.zarr").exists() or (root / "cell_feature_matrix").is_dir():
        return "zarr" if (root / "cell_feature_matrix.zarr").exists() else "mex"
    if (root / "cell_feature_matrix.h5").exists() or any(root.glob("*_cell_feature_matrix.h5")):
        return "h5"
    matrix_dir = root / "cell_feature_matrix"
    if matrix_dir.is_dir() and (matrix_dir / "matrix.mtx.gz").exists():
        return "mex"
    return None


def _has_cells_table(root: Path) -> bool:
    direct = ("cells.parquet", "cells.parquet.gz", "cells.csv.gz", "cells.csv", "cells.zarr.zip")
    return any((root / name).exists() for name in direct) or bool(
        list(root.glob("*_cells.parquet")) + list(root.glob("*_cells.parquet.gz"))
    )


def _safe_read_experiment(root: Path) -> dict[str, Any]:
    try:
        return read_experiment_metadata(str(root))
    except Exception:
        return {}


def _safe_discover_image_artifacts(root: Path) -> dict[str, Any]:
    try:
        return discover_image_artifacts(str(root))
    except Exception:
        return {}


def _record_rank(record: TenXDatasetRecord) -> tuple[int, int, int, int, int, int]:
    backend_rank = {"zarr": 3, "h5": 2, "mex": 1}.get(record.matrix_backend or "", 0)
    return (
        int(record.buildable),
        int(record.has_he),
        int(record.has_alignment),
        int(record.selected_geojson is not None),
        backend_rank,
        -len(Path(record.xenium_root).parts),
    )


def _candidate_10x_urls(record: Mapping[str, Any]) -> list[str]:
    texts = [
        str(record.get("run_name") or ""),
        str(record.get("case_name") or ""),
        Path(str(record.get("xenium_root") or "")).name,
    ]
    normalized = " ".join(_humanize_name(text) for text in texts).lower()
    slugs: list[str] = []
    curated = {
        "atera wta ffpe human breast cancer": "atera-wta-ffpe-human-breast-cancer",
        "atera wta ffpe human cervical cancer": "atera-wta-ffpe-human-cervical-cancer",
        "xenium prime human breast cancer ffpe": "xenium-prime-ffpe-human-breast-cancer",
        "xenium prime human cervical cancer ffpe": "xenium-prime-ffpe-human-cervical-cancer",
        "xenium prime human ovarian cancer ffpe": "xenium-prime-ffpe-human-ovarian-cancer",
        "xenium prime human pancreas ffpe": "xenium-prime-ffpe-human-pancreas",
        "xenium prime human lung cancer ffpe": "xenium-prime-ffpe-human-lung-cancer",
        "xenium prime human lymph node reactive ffpe": "xenium-prime-ffpe-human-lymph-node-reactive",
    }
    for phrase, slug in curated.items():
        if phrase in normalized:
            slugs.append(slug)
    if "wta preview" in normalized and "breast" in normalized:
        slugs.append("atera-wta-ffpe-human-breast-cancer")
    if "wta preview" in normalized and "cervical" in normalized:
        slugs.append("atera-wta-ffpe-human-cervical-cancer")
    for text in texts:
        base = _url_slug(text.replace("_outs", "").replace("_output", ""))
        if base:
            slugs.append(base)
            slugs.append(base.replace("xenium-prime-human-", "xenium-prime-ffpe-human-"))
            slugs.append(base.replace("xenium-prime-", "xenium-prime-ffpe-human-"))
            slugs.append(base.replace("human-", "ffpe-human-"))
            if base.startswith("xenium-prime-") and base.endswith("-ffpe"):
                middle = base.removeprefix("xenium-prime-").removesuffix("-ffpe")
                if not middle.startswith("human-") and not middle.startswith("mouse-"):
                    middle = f"human-{middle}"
                slugs.append(f"xenium-prime-ffpe-{middle}")
            slugs.append(f"preview-data-{base}")
    return [f"{TENX_DATASET_URL_PREFIX}{slug}" for slug in _unique(slugs)]


def _parse_10x_dataset_page(text: str, url: str) -> dict[str, Any]:
    title = _first_match(text, r"<title[^>]*>(.*?)</title>") or _first_match(
        text, r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"'
    )
    canonical = _first_match(text, r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"') or url
    fields = _parse_sidebar_fields(text)
    metrics = _parse_metric_table(text)
    return {
        "metadata_status": "resolved",
        "source": "10x Genomics datasets",
        "source_url": html.unescape(canonical),
        "source_index_url": TENX_DATASETS_URL,
        "title": _clean_html(title or ""),
        "fields": fields,
        "metrics": metrics,
        "license": _extract_license(text),
    }


def _parse_sidebar_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    pattern = re.compile(r"<li[^>]*>\s*<p[^>]*>(.*?)</p>\s*<p[^>]*>(.*?)</p>\s*</li>", re.S)
    for key, value in pattern.findall(text):
        clean_key = _clean_html(key)
        clean_value = _clean_html(value)
        if clean_key and clean_value:
            fields[clean_key] = clean_value
    return fields


def _parse_metric_table(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)
        if len(cells) == 2:
            key = _clean_html(cells[0])
            value = _clean_html(cells[1])
            if key and value and key.lower() not in {"metric"}:
                metrics[key] = value
    return metrics


def _extract_license(text: str) -> str | None:
    if "Creative Commons Attribution 4.0 International" in text or "CC BY 4.0" in text:
        return "CC BY 4.0"
    return None


def _metadata_field(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get("fields", {}).get(key) if isinstance(metadata.get("fields"), dict) else None
    return str(value) if value else None


def _metadata_cache_payload(metadata_cache: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if metadata_cache is None:
        return {}
    if isinstance(metadata_cache, (str, Path)):
        return _read_json(Path(metadata_cache).expanduser())
    return metadata_cache


def _lookup_metadata_cache(record: Mapping[str, Any], cache_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    records = cache_payload.get("records", cache_payload) if isinstance(cache_payload, Mapping) else {}
    if not isinstance(records, Mapping):
        return None
    keys = [
        record.get("experiment_uuid"),
        record.get("analysis_uuid"),
        record.get("run_name"),
        record.get("case_name"),
        record.get("case_slug"),
        Path(str(record.get("xenium_root") or "")).name,
    ]
    for key in keys:
        if key is not None and str(key) in records and isinstance(records[str(key)], Mapping):
            return dict(records[str(key)])
    return None


def _record_mapping(record_or_root: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(record_or_root, Mapping):
        return dict(record_or_root)
    root = Path(record_or_root).expanduser()
    return _inventory_xenium_root(root, root=root.parent).to_dict()


def _local_experiment_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_uuid": record.get("experiment_uuid"),
        "analysis_uuid": record.get("analysis_uuid"),
        "run_name": record.get("run_name"),
        "panel_name": record.get("panel_name"),
    }


def _find_cluster_csv(root: Path) -> dict[str, Any] | None:
    candidates = [
        root / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv",
        root / "analysis" / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv",
    ]
    candidates.extend(root.glob("**/gene_expression_graphclust/clusters.csv"))
    for path in _unique_paths(candidates):
        if not path.exists():
            continue
        frame = pd.read_csv(path, nrows=20)
        barcode_col = _first_existing_column(frame, ("Barcode", "barcode", "cell_id", "CellID"))
        cluster_col = _first_existing_column(frame, ("Cluster", "Clusters", "cluster", "group"))
        if barcode_col and cluster_col:
            return {"path": path, "barcode_col": barcode_col, "cluster_col": cluster_col}
    return None


def _infer_histoseg_structures(cluster_csv: Path, cluster_col: str, *, min_cells: int) -> list[dict[str, Any]]:
    frame = pd.read_csv(cluster_csv, usecols=[cluster_col])
    counts = frame[cluster_col].astype(str).value_counts()
    structures: list[dict[str, Any]] = []
    for index, (label, count) in enumerate(counts.items(), start=1):
        if int(count) < int(min_cells):
            continue
        structures.append(
            {
                "structure_id": index,
                "structure_name": f"cluster_{_slug(label, max_len=40)}",
                "cluster_ids": [str(label)],
            }
        )
    return structures


def _ensure_cells_parquet_for_histoseg(root: Path, output_dir: Path) -> Path | None:
    for path in [root / "cells.parquet", *root.glob("*_cells.parquet")]:
        if path.exists():
            return path
    csv_path = root / "cells.csv.gz"
    if not csv_path.exists() and (root / "cells.csv").exists():
        csv_path = root / "cells.csv"
    if not csv_path.exists():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "cells.parquet"
    compression = "gzip" if csv_path.suffix == ".gz" else None
    frame = pd.read_csv(csv_path, compression=compression)
    frame.to_parquet(target, index=False)
    return target


def _first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if str(candidate).lower() in lower:
            return lower[str(candidate).lower()]
    return None


def _write_registry_outputs(
    output: Path,
    registry_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> None:
    _write_json(output / "dataset_registry.json", registry_rows)
    _write_json(output / "failed_cases.json", failed_rows)
    _write_json(output / "metadata_resolution_report.json", metadata_rows)
    _write_json(output / "build_summary.json", _build_summary(registry_rows, failed_rows, output))
    _write_csv(output / "dataset_registry.csv", registry_rows)
    _write_csv(output / "failed_cases.csv", failed_rows)
    _write_csv(output / "metadata_resolution_report.csv", metadata_rows)
    summary_rows = [
        {"metric": key, "value": value}
        for key, value in _build_summary(registry_rows, failed_rows, output).items()
        if key != "output_root"
    ]
    _write_csv(output / "build_summary.csv", summary_rows)


def _checkpoint_registry(
    output: Path,
    registry_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> None:
    _write_registry_outputs(output, registry_rows, failed_rows, metadata_rows)


def _build_summary(registry_rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    statuses = pd.Series([row.get("build_status", "unknown") for row in registry_rows], dtype="object")
    completed = statuses.isin(["built", "reused_existing", "skipped_existing"])
    return {
        "output_root": str(output),
        "records": int(len(registry_rows)),
        "selected_records": int(sum(bool(row.get("selected_for_build")) for row in registry_rows)),
        "completed_outputs": int(completed.sum()),
        "built": int((statuses == "built").sum()),
        "reused_existing": int((statuses == "reused_existing").sum()),
        "skipped_existing": int((statuses == "skipped_existing").sum()),
        "failed": int(len(failed_rows)),
        "duplicates_skipped": int((statuses == "duplicate_skipped").sum()),
        "not_buildable": int((statuses == "not_buildable").sum()),
    }


def _metadata_report_row(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_name": record.get("case_name"),
        "xenium_root": record.get("xenium_root"),
        "metadata_status": metadata.get("metadata_status"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "species": _metadata_field(metadata, "Species"),
        "anatomical_entity": _metadata_field(metadata, "Anatomical Entity"),
        "disease_state": _metadata_field(metadata, "Disease State"),
    }


def _attach_reused_artifact_status(row: dict[str, Any]) -> None:
    output = Path(str(row["reused_output_dir"]))
    row["output_dir"] = str(output)
    row["slide_manifest"] = str(output / "slide_manifest.json") if (output / "slide_manifest.json").exists() else None
    row["qc_report"] = str(output / "qc_report.json") if (output / "qc_report.json").exists() else None
    row["metadata_10x"] = str(output / "metadata_10x.json") if (output / "metadata_10x.json").exists() else None
    row["contour_source_manifest"] = (
        str(output / "contour_source_manifest.json") if (output / "contour_source_manifest.json").exists() else None
    )
    row.update(_summarize_existing_case(output))


def _summarize_existing_case(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "slide_manifest.json")
    qc = _read_json(output / "qc_report.json")
    return {
        "qc_status": qc.get("status"),
        "cells": manifest.get("counts", {}).get("cells"),
        "genes": manifest.get("counts", {}).get("genes"),
        "assigned_cells": manifest.get("counts", {}).get("assigned_cells"),
        "contour_patches": manifest.get("counts", {}).get("contour_patches"),
    }


def _ensure_case_artifact_contract(
    output: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    contour_source: Mapping[str, Any] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        _write_json(output / "metadata_10x.json", metadata)
    if contour_source is not None and not (output / "contour_source_manifest.json").exists():
        _write_json(output / "contour_source_manifest.json", dict(contour_source))
    patch_manifest_path = output / "contour_patches_manifest.json"
    if not patch_manifest_path.exists():
        _write_json(patch_manifest_path, [])
    cell_to_contour_path = output / "cell_to_contour.parquet"
    if not cell_to_contour_path.exists():
        _empty_cell_to_contour_frame().to_parquet(cell_to_contour_path, index=False)
    structure_assignments_path = output / "structure_assignments.csv"
    if not structure_assignments_path.exists():
        _empty_structure_assignments_frame().to_csv(structure_assignments_path, index=False)
    _patch_slide_manifest_artifact_contract(output, metadata=metadata, contour_source=contour_source)


def _patch_slide_manifest_artifact_contract(
    output: Path,
    *,
    metadata: Mapping[str, Any] | None,
    contour_source: Mapping[str, Any] | None,
) -> None:
    manifest_path = output / "slide_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    original = json.dumps(manifest, sort_keys=True, default=str)
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["cell_to_contour"] = str(output / "cell_to_contour.parquet")
    artifacts["structure_assignments"] = str(output / "structure_assignments.csv")
    artifacts["contour_patches_manifest"] = str(output / "contour_patches_manifest.json")
    if metadata is not None and not manifest.get("tenx_source"):
        manifest["tenx_source"] = dict(metadata)
    if contour_source is not None and not manifest.get("contour_source"):
        manifest["contour_source"] = dict(contour_source)
    updated = json.dumps(manifest, sort_keys=True, default=str)
    if updated != original:
        _write_json(manifest_path, manifest)


def _empty_cell_to_contour_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cell_id",
            "x",
            "y",
            "contour_id",
            "structure_id",
            "structure_label",
            "assignment_status",
        ]
    )


def _empty_structure_assignments_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["contour_id", "structure_id", "structure_label"])


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return value


def _exclusion_reason(matrix_backend: str | None, has_cells: bool) -> str:
    missing = []
    if matrix_backend is None:
        missing.append("cell_feature_matrix")
    if not has_cells:
        missing.append("cells table")
    return "Missing " + " and ".join(missing) + "."


def _infer_organ_from_name(name: str) -> str | None:
    text = name.lower()
    organs = {
        "breast": "breast",
        "cervical": "cervix",
        "cervix": "cervix",
        "lung": "lung",
        "lymph": "lymph node",
        "ovarian": "ovary",
        "pancreas": "pancreas",
        "kidney": "kidney",
        "renal": "kidney",
        "liver": "liver",
        "brain": "brain",
        "tonsil": "tonsil",
        "skin": "skin",
        "heart": "heart",
        "bone": "bone",
    }
    for token, organ in organs.items():
        if token in text:
            return organ
    return None


def _case_slug(root: Path) -> str:
    name = root.name
    for suffix in ("_outs", "_output", "_out"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return _slug(name.lower(), max_len=120)


def _slug(value: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return (slug or "xenium_case")[:max_len]


def _url_slug(value: str, *, max_len: int = 160) -> str:
    text = str(value).strip()
    text = re.sub(r"[_\s]+", "-", text)
    text = re.sub(r"[^A-Za-z0-9-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-").lower()
    return text[:max_len]


def _humanize_name(value: str) -> str:
    text = str(value)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\bouts?\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value, flags=re.S)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.S)
    return match.group(1) if match else None


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return Path(path.name)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _unique_paths(values: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
