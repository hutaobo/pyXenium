from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, ttest_ind
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from pyXenium.contour import add_contours_from_geojson, build_contour_feature_table
from pyXenium.contour._analysis import _prepare_contours
from pyXenium.contour.loading import _geometry_um_to_image_xy
from pyXenium.io import read_slide, read_xenium
from pyXenium.io.slide_model import XeniumImage, XeniumSlide

from .contour_boundary_ecology import score_contour_boundary_programs

__all__ = [
    "DEFAULT_LAZYSLIDE_TEXT_TERMS",
    "DEFAULT_STRUCTURE_RNA_MARKERS",
    "HistoSegLazySlideConfig",
    "aggregate_structure_image_features",
    "assign_tiles_to_histoseg_structures",
    "histoseg_contours_to_image_table",
    "run_histoseg_lazyslide_structure_workflow",
]

DEFAULT_LAZYSLIDE_TEXT_TERMS = (
    "ductal epithelium",
    "invasive carcinoma",
    "in situ carcinoma",
    "fibrotic stroma",
    "immune infiltrate",
    "necrosis",
    "adipose tissue",
    "vascular stroma",
    "lumen or secretion",
)

DEFAULT_STRUCTURE_RNA_MARKERS = (
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT19",
    "ERBB2",
    "ESR1",
    "PGR",
    "MKI67",
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "DCN",
    "LUM",
    "ACTA2",
    "FAP",
    "PECAM1",
    "VWF",
    "RGS5",
    "CD3D",
    "CD3E",
    "TRAC",
    "MS4A1",
    "CD79A",
    "CXCL13",
    "CD68",
    "CD163",
    "LST1",
    "SPP1",
    "CA9",
    "SLC2A1",
)

_SCHEMA_VERSION = "pyxenium-histoseg-lazyslide-v1"
_TABLE_FORMATS = ("csv", "parquet")
_ID_COLUMNS = ("contour_id", "assigned_structure", "structure_id")
_TEXT_PREFIX = "text_similarity__"
_EMBEDDING_PREFIX = "embedding__"


@dataclass(frozen=True)
class HistoSegLazySlideConfig:
    """Configuration for HistoSeg-anchored LazySlide image feature analysis."""

    output_dir: str | Path | None = None
    contour_key: str = "histoseg_structures"
    contour_geojson: str | Path | None = None
    contour_id_key: str = "polygon_id"
    contour_coordinate_space: str = "xenium_pixel"
    contour_pixel_size_um: float | None = None
    he_image_key: str = "he"
    he_source_path: str | Path | None = None
    model: str = "plip"
    text_terms: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LAZYSLIDE_TEXT_TERMS)
    tile_px: int = 224
    mpp: float = 0.5
    device: str = "cuda"
    amp: bool = True
    batch_size: int = 64
    max_tiles: int | None = None
    table_format: Literal["csv", "parquet"] = "csv"
    include_rna: bool = True
    include_boundary_programs: bool = True
    program_library: str = "tumor_boundary_v1"
    rna_markers: tuple[str, ...] = field(default_factory=lambda: DEFAULT_STRUCTURE_RNA_MARKERS)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = _optional_path_text(self.output_dir)
        payload["contour_geojson"] = _optional_path_text(self.contour_geojson)
        return payload


def run_histoseg_lazyslide_structure_workflow(
    sdata_or_path: XeniumSlide | str | Path,
    *,
    output_dir: str | Path | None = None,
    contour_key: str = "histoseg_structures",
    contour_geojson: str | Path | None = None,
    contour_id_key: str = "polygon_id",
    contour_coordinate_space: str = "xenium_pixel",
    contour_pixel_size_um: float | None = None,
    he_image_key: str = "he",
    he_source_path: str | Path | None = None,
    model: str = "plip",
    text_terms: Sequence[str] | None = None,
    tile_px: int = 224,
    mpp: float = 0.5,
    device: str = "cuda",
    amp: bool = True,
    batch_size: int = 64,
    max_tiles: int | None = None,
    table_format: Literal["csv", "parquet"] = "csv",
    include_rna: bool = True,
    include_boundary_programs: bool = True,
    program_library: str = "tumor_boundary_v1",
    rna_markers: Sequence[str] | None = None,
    lazy_backend: Any = None,
    precomputed_tile_features: pd.DataFrame | str | Path | None = None,
    precomputed_feature_table: Mapping[str, Any] | None = None,
    precomputed_program_scores: pd.DataFrame | str | Path | None = None,
) -> dict[str, Any]:
    """Run a HistoSeg structure-to-H&E feature workflow with optional LazySlide.

    HistoSeg owns segmentation and structure proposals. LazySlide owns WSI tile
    extraction and image-model inference when the optional backend is used.
    pyXenium owns coordinate alignment, structure-level aggregation, and
    RNA/image interpretation artifacts.
    """

    config = HistoSegLazySlideConfig(
        output_dir=output_dir,
        contour_key=contour_key,
        contour_geojson=contour_geojson,
        contour_id_key=contour_id_key,
        contour_coordinate_space=contour_coordinate_space,
        contour_pixel_size_um=contour_pixel_size_um,
        he_image_key=he_image_key,
        he_source_path=he_source_path,
        model=model,
        text_terms=tuple(text_terms or DEFAULT_LAZYSLIDE_TEXT_TERMS),
        tile_px=int(tile_px),
        mpp=float(mpp),
        device=str(device),
        amp=bool(amp),
        batch_size=int(batch_size),
        max_tiles=max_tiles,
        table_format=table_format,
        include_rna=bool(include_rna),
        include_boundary_programs=bool(include_boundary_programs),
        program_library=str(program_library),
        rna_markers=tuple(rna_markers or DEFAULT_STRUCTURE_RNA_MARKERS),
    )
    if config.table_format not in _TABLE_FORMATS:
        raise ValueError(f"`table_format` must be one of {_TABLE_FORMATS}.")

    started = time.time()
    sdata = _resolve_sdata(sdata_or_path)
    _ensure_histoseg_contours(sdata, config=config)
    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=config.contour_key,
        contour_query=None,
    )
    he_image = _resolve_he_image(sdata, he_image_key=config.he_image_key)
    if config.he_source_path is not None:
        he_image.source_path = str(Path(config.he_source_path).expanduser())
    image_contours = histoseg_contours_to_image_table(contour_table, he_image=he_image)

    model_result = _resolve_tile_features(
        sdata=sdata,
        image_contours=image_contours,
        he_image=he_image,
        config=config,
        lazy_backend=lazy_backend,
        precomputed_tile_features=precomputed_tile_features,
    )
    tile_features = _normalize_tile_feature_table(
        model_result["tile_features"],
        text_terms=config.text_terms,
        max_tiles=config.max_tiles,
    )
    tile_assignments = assign_tiles_to_histoseg_structures(
        tile_features,
        image_contours,
    )
    assigned_tile_features = _merge_tile_features_with_assignments(
        tile_features,
        tile_assignments,
    )

    structure_image_features = aggregate_structure_image_features(
        assigned_tile_features,
        text_terms=config.text_terms,
    )
    differential = _differential_image_features(assigned_tile_features)

    structure_rna = (
        _build_structure_rna_summary(
            sdata=sdata,
            contour_table=contour_table,
            genes=config.rna_markers,
        )
        if config.include_rna
        else _empty_structure_rna_summary()
    )
    rna_image_associations = _associate_structure_tables(
        structure_image_features,
        structure_rna,
        left_prefixes=(_TEXT_PREFIX, _EMBEDDING_PREFIX, "domain_fraction__"),
        right_prefixes=("rna__", "cell_type_fraction__"),
        association_kind="rna_image_association",
    )

    program_scores, program_status = _resolve_program_scores(
        sdata=sdata,
        contour_key=config.contour_key,
        precomputed_feature_table=precomputed_feature_table,
        precomputed_program_scores=precomputed_program_scores,
        program_library=config.program_library,
        include_boundary_programs=config.include_boundary_programs,
    )
    structure_programs = _aggregate_program_scores_by_structure(
        program_scores,
        contour_table=contour_table,
    )
    program_image_associations = _associate_structure_tables(
        structure_image_features,
        structure_programs,
        left_prefixes=(_TEXT_PREFIX, _EMBEDDING_PREFIX, "domain_fraction__"),
        right_prefixes=("program__",),
        association_kind="program_image_association",
    )

    manifest = _build_manifest(
        config=config,
        sdata=sdata,
        model_result=model_result,
        program_status=program_status,
        n_contours=len(contour_table),
        n_tiles=len(assigned_tile_features),
        n_assigned_tiles=int(tile_assignments["assigned"].sum()) if not tile_assignments.empty else 0,
        started=started,
    )
    result = {
        "image_contours": image_contours,
        "tile_features": assigned_tile_features,
        "tile_assignments": tile_assignments,
        "structure_image_features": structure_image_features,
        "structure_differential_features": differential,
        "structure_rna_summary": structure_rna,
        "structure_program_scores": structure_programs,
        "rna_image_associations": rna_image_associations,
        "program_image_associations": program_image_associations,
        "run_manifest": manifest,
    }
    if output_dir is not None:
        _write_workflow_artifacts(result, output_dir, table_format=config.table_format)
    return result


def histoseg_contours_to_image_table(
    contour_table: pd.DataFrame,
    *,
    he_image: XeniumImage,
) -> pd.DataFrame:
    """Convert HistoSeg/pyXenium contour geometries from Xenium um to H&E pixels."""

    if "geometry" not in contour_table.columns:
        raise ValueError("`contour_table` must contain a `geometry` column.")
    rows: list[dict[str, Any]] = []
    for _, row in contour_table.iterrows():
        image_geometry = _geometry_um_to_image_xy(row["geometry"], he_image=he_image)
        payload = {
            column: row[column]
            for column in contour_table.columns
            if column != "geometry"
        }
        payload["geometry"] = image_geometry
        payload["area_image_px2"] = float(image_geometry.area)
        payload["centroid_x"] = float(image_geometry.centroid.x)
        payload["centroid_y"] = float(image_geometry.centroid.y)
        rows.append(payload)
    return pd.DataFrame(rows)


def assign_tiles_to_histoseg_structures(
    tile_features: pd.DataFrame,
    image_contours: pd.DataFrame,
) -> pd.DataFrame:
    """Assign image tiles to HistoSeg structures using tile centroids."""

    if tile_features.empty:
        return pd.DataFrame(
            columns=[
                "tile_id",
                "contour_id",
                "assigned_structure",
                "structure_id",
                "classification_name",
                "tile_x",
                "tile_y",
                "assigned",
            ]
        )
    if "tile_id" not in tile_features.columns:
        raise ValueError("`tile_features` must contain `tile_id`.")
    if "geometry" not in image_contours.columns:
        raise ValueError("`image_contours` must contain `geometry`.")

    contour_records = []
    for _, contour in image_contours.iterrows():
        geometry = contour["geometry"]
        if geometry is None or geometry.is_empty:
            continue
        contour_records.append((contour, geometry, float(geometry.area)))

    rows = []
    for _, tile in tile_features.iterrows():
        point = _tile_centroid(tile)
        best: pd.Series | None = None
        best_area = np.inf
        for contour, geometry, area in contour_records:
            if not (geometry.covers(point) or geometry.intersects(point)):
                continue
            if area < best_area:
                best = contour
                best_area = area
        base = {
            "tile_id": str(tile["tile_id"]),
            "tile_x": float(point.x),
            "tile_y": float(point.y),
            "assigned": best is not None,
        }
        if best is None:
            base.update(
                {
                    "contour_id": None,
                    "assigned_structure": None,
                    "structure_id": None,
                    "classification_name": None,
                }
            )
        else:
            base.update(
                {
                    "contour_id": str(best.get("contour_id", "")),
                    "assigned_structure": _structure_label(best),
                    "structure_id": _optional_text(best.get("structure_id")),
                    "classification_name": _optional_text(best.get("classification_name")),
                }
            )
        rows.append(base)
    return pd.DataFrame(rows)


def aggregate_structure_image_features(
    tile_features: pd.DataFrame,
    *,
    text_terms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate tile-level image features by HistoSeg structure."""

    if tile_features.empty or "assigned_structure" not in tile_features.columns:
        return _empty_structure_image_features()
    assigned = tile_features[tile_features["assigned_structure"].notna()].copy()
    if assigned.empty:
        return _empty_structure_image_features()
    assigned["assigned_structure"] = assigned["assigned_structure"].astype(str)
    numeric_columns = _image_numeric_columns(assigned)
    text_terms = tuple(text_terms or DEFAULT_LAZYSLIDE_TEXT_TERMS)

    rows = []
    for structure, group in assigned.groupby("assigned_structure", sort=True, dropna=False):
        row: dict[str, Any] = {
            "assigned_structure": str(structure),
            "n_tiles": int(len(group)),
            "n_contours": int(group["contour_id"].nunique())
            if "contour_id" in group.columns
            else 0,
            "structure_id": _mode_or_none(group.get("structure_id")),
            "classification_name": _mode_or_none(group.get("classification_name")),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}__std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        if "top_image_label" in group.columns:
            label_counts = group["top_image_label"].dropna().astype(str).value_counts()
            if not label_counts.empty:
                row["top_image_label"] = str(label_counts.index[0])
                row["top_image_label_fraction"] = float(label_counts.iloc[0] / len(group))
        if "spatial_domain" in group.columns:
            for domain, count in group["spatial_domain"].dropna().astype(str).value_counts().items():
                row[f"domain_fraction__{_slug(domain)}"] = float(count / len(group))
        for term in text_terms:
            column = f"{_TEXT_PREFIX}{_slug(term)}"
            if column in group.columns:
                row[f"{column}__rank"] = _mean_rank(group[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("assigned_structure", kind="stable").reset_index(drop=True)


def _resolve_sdata(sdata_or_path: XeniumSlide | str | Path) -> XeniumSlide:
    if isinstance(sdata_or_path, XeniumSlide):
        return sdata_or_path
    path = Path(sdata_or_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.suffix.lower() == ".zarr":
        return read_slide(path)
    return read_xenium(
        str(path),
        as_="slide",
        prefer="h5",
        include_transcripts=False,
        include_boundaries=True,
        include_images=True,
    )


def _ensure_histoseg_contours(sdata: XeniumSlide, *, config: HistoSegLazySlideConfig) -> None:
    if config.contour_key in sdata.shapes:
        return
    if config.contour_geojson is None:
        raise KeyError(
            f"`sdata.shapes[{config.contour_key!r}]` was not found and "
            "`contour_geojson` was not provided."
        )
    add_contours_from_geojson(
        sdata,
        config.contour_geojson,
        key=config.contour_key,
        id_key=config.contour_id_key,
        coordinate_space=config.contour_coordinate_space,
        pixel_size_um=config.contour_pixel_size_um,
        extract_he_patches=False,
        he_image_key=config.he_image_key,
    )


def _resolve_he_image(sdata: XeniumSlide, *, he_image_key: str) -> XeniumImage:
    if he_image_key not in sdata.images:
        raise KeyError(f"`sdata.images[{he_image_key!r}]` was not found.")
    he_image = sdata.images[he_image_key]
    if he_image.image_to_xenium_affine is None:
        raise ValueError(f"`sdata.images[{he_image_key!r}]` is missing image alignment.")
    if he_image.pixel_size_um is None:
        raise ValueError(f"`sdata.images[{he_image_key!r}]` is missing pixel_size_um.")
    return he_image


def _resolve_tile_features(
    *,
    sdata: XeniumSlide,
    image_contours: pd.DataFrame,
    he_image: XeniumImage,
    config: HistoSegLazySlideConfig,
    lazy_backend: Any,
    precomputed_tile_features: pd.DataFrame | str | Path | None,
) -> dict[str, Any]:
    if precomputed_tile_features is not None:
        return {
            "tile_features": _read_table(precomputed_tile_features),
            "model_status": {
                "backend": "precomputed",
                "model": config.model,
                "status": "loaded",
            },
        }
    if lazy_backend is not None:
        return _run_custom_lazy_backend(
            lazy_backend,
            sdata=sdata,
            image_contours=image_contours,
            he_image=he_image,
            config=config,
        )
    return _run_lazyslide_backend(
        image_contours=image_contours,
        he_image=he_image,
        config=config,
    )


def _run_custom_lazy_backend(
    backend: Any,
    *,
    sdata: XeniumSlide,
    image_contours: pd.DataFrame,
    he_image: XeniumImage,
    config: HistoSegLazySlideConfig,
) -> dict[str, Any]:
    if hasattr(backend, "run"):
        result = backend.run(
            sdata=sdata,
            image_contours=image_contours,
            he_image=he_image,
            config=config,
        )
    elif callable(backend):
        result = backend(
            sdata=sdata,
            image_contours=image_contours,
            he_image=he_image,
            config=config,
        )
    else:
        raise TypeError("`lazy_backend` must be callable or expose `run(...)`.")
    if isinstance(result, pd.DataFrame):
        frame = result
        status = {"backend": "custom", "model": config.model, "status": "completed"}
    elif isinstance(result, Mapping):
        if "tile_features" not in result:
            raise KeyError("Custom LazySlide backend result must contain `tile_features`.")
        frame = pd.DataFrame(result["tile_features"]).copy()
        status = dict(
            result.get(
                "model_status",
                {"backend": "custom", "model": config.model, "status": "completed"},
            )
        )
    else:
        raise TypeError("Custom LazySlide backend must return a DataFrame or mapping.")
    return {"tile_features": frame, "model_status": status}


def _run_lazyslide_backend(
    *,
    image_contours: pd.DataFrame,
    he_image: XeniumImage,
    config: HistoSegLazySlideConfig,
) -> dict[str, Any]:
    try:
        import geopandas as gpd
        import lazyslide as zs
        from wsidata import open_wsi
    except ImportError as exc:
        raise ImportError(
            "LazySlide support is optional. Install it with "
            "`pip install 'pyXenium[lazyslide]'` in the A100 environment."
        ) from exc

    if not he_image.source_path:
        raise ValueError("The H&E image has no `source_path` for WSIData/open_wsi.")

    annotation_key = "histoseg_structures"
    tile_key = "histoseg_tiles"
    feature_key = f"{_slug(config.model)}_{tile_key}"
    status: dict[str, Any] = {
        "backend": "lazyslide",
        "model": config.model,
        "status": "started",
        "lazy_module": getattr(zs, "__version__", "unknown"),
    }

    try:
        wsi = open_wsi(str(he_image.source_path))
        annotation_frame = image_contours.drop(columns=["geometry"]).copy()
        annotation_frame["tissue_id"] = annotation_frame["contour_id"].astype(str)
        annotations = gpd.GeoDataFrame(
            annotation_frame,
            geometry=image_contours["geometry"].to_list(),
            crs=None,
        )
        _load_lazyslide_annotations(zs, wsi, annotations, key_added=annotation_key)
        zs.pp.tile_tissues(
            wsi,
            int(config.tile_px),
            mpp=float(config.mpp),
            tissue_key=annotation_key,
            key_added=tile_key,
        )
        if config.max_tiles is not None:
            _limit_lazyslide_tiles(wsi, tile_key=tile_key, max_tiles=int(config.max_tiles))
        zs.tl.feature_extraction(
            wsi,
            config.model,
            amp=bool(config.amp),
            device=config.device,
            batch_size=int(config.batch_size),
            tile_key=tile_key,
            key_added=feature_key,
        )
        _try_lazyslide_text_similarity(
            zs,
            wsi,
            model=config.model,
            feature_key=feature_key,
            text_terms=config.text_terms,
        )
        _try_lazyslide_spatial_domain(zs, wsi, model=config.model, tile_key=tile_key)
        frame = _extract_lazyslide_tile_features(
            wsi,
            feature_key=feature_key,
            tile_key=tile_key,
            text_terms=config.text_terms,
        )
        status["status"] = "completed"
        status["n_tiles"] = int(len(frame))
        return {"tile_features": frame, "model_status": status}
    except Exception as exc:
        if _slug(config.model) == "conch":
            status.update(
                {
                    "status": "skipped",
                    "skipped_reason": str(exc),
                    "n_tiles": 0,
                }
            )
            return {"tile_features": pd.DataFrame(columns=["tile_id"]), "model_status": status}
        raise


def _load_lazyslide_annotations(zs: Any, wsi: Any, annotations: Any, *, key_added: str) -> None:
    try:
        zs.io.load_annotations(
            wsi,
            annotations=annotations,
            join_with=None,
            key_added=key_added,
        )
        return
    except TypeError:
        pass
    try:
        zs.io.load_annotations(wsi, annotations=annotations, key_added=key_added)
        return
    except Exception:
        pass
    if not hasattr(wsi, "shapes"):
        raise RuntimeError("Could not attach HistoSeg annotations to WSIData.")
    wsi.shapes[key_added] = annotations


def _limit_lazyslide_tiles(wsi: Any, *, tile_key: str, max_tiles: int) -> None:
    if max_tiles <= 0 or not hasattr(wsi, "shapes"):
        return
    try:
        frame = wsi.shapes[tile_key]
        wsi.shapes[tile_key] = frame.iloc[:max_tiles].copy()
    except Exception:
        return


def _try_lazyslide_text_similarity(
    zs: Any,
    wsi: Any,
    *,
    model: str,
    feature_key: str,
    text_terms: Sequence[str],
) -> None:
    try:
        text_embeddings = zs.tl.text_embedding(list(text_terms), model=model)
        try:
            zs.tl.text_image_similarity(
                wsi,
                text_embeddings,
                model=model,
                softmax=True,
                feature_key=feature_key,
            )
        except TypeError:
            zs.tl.text_image_similarity(wsi, text_embeddings, model=model, softmax=True)
    except Exception:
        return


def _try_lazyslide_spatial_domain(zs: Any, wsi: Any, *, model: str, tile_key: str) -> None:
    try:
        zs.pp.tile_graph(wsi, tile_key=tile_key)
        feature_key = f"{_slug(model)}_{tile_key}"
        zs.tl.spatial_features(wsi, feature_key, tile_key=tile_key)
        zs.tl.spatial_domain(wsi, feature_key=feature_key, tile_key=tile_key)
    except Exception:
        return


def _extract_lazyslide_tile_features(
    wsi: Any,
    *,
    feature_key: str,
    tile_key: str,
    text_terms: Sequence[str],
) -> pd.DataFrame:
    tile_frame = _fetch_wsi_shape_frame(wsi, tile_key)
    features = _fetch_wsi_table(wsi, feature_key)
    feature_frame = _anndata_to_feature_frame(features)
    text_frame = _fetch_optional_text_similarity(wsi, feature_key, text_terms)
    if not text_frame.empty:
        feature_frame = feature_frame.merge(text_frame, on="tile_id", how="left")
    if not tile_frame.empty:
        feature_frame = feature_frame.merge(tile_frame, on="tile_id", how="left")
    return feature_frame


def _fetch_wsi_shape_frame(wsi: Any, key: str) -> pd.DataFrame:
    try:
        frame = wsi.shapes[key]
    except Exception:
        return pd.DataFrame(columns=["tile_id"])
    output = pd.DataFrame(frame).copy()
    if "tile_id" not in output.columns:
        output.insert(0, "tile_id", output.index.astype(str))
    else:
        output["tile_id"] = output["tile_id"].astype(str)
    return output


def _fetch_wsi_table(wsi: Any, key: str) -> ad.AnnData:
    try:
        value = wsi[key]
        if isinstance(value, ad.AnnData):
            return value
    except Exception:
        pass
    if hasattr(wsi, "fetch") and hasattr(wsi.fetch, "features_anndata"):
        try:
            value = wsi.fetch.features_anndata(key)
            if isinstance(value, ad.AnnData):
                return value
        except Exception:
            pass
    raise KeyError(f"Could not fetch LazySlide feature table {key!r}.")


def _anndata_to_feature_frame(adata: ad.AnnData) -> pd.DataFrame:
    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    columns = [f"{_EMBEDDING_PREFIX}{_slug(name)}" for name in adata.var_names.astype(str)]
    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "tile_id", adata.obs_names.astype(str))
    for column in adata.obs.columns:
        if column not in frame.columns:
            frame[column] = adata.obs[column].to_numpy()
    return frame


def _fetch_optional_text_similarity(
    wsi: Any,
    feature_key: str,
    text_terms: Sequence[str],
) -> pd.DataFrame:
    candidates = [
        f"{feature_key}_text_similarity",
        f"{feature_key}_tiles_text_similarity",
        "text_similarity",
    ]
    for key in candidates:
        try:
            adata = _fetch_wsi_table(wsi, key)
        except Exception:
            continue
        matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        terms = list(text_terms)
        if matrix.shape[1] != len(terms):
            terms = adata.var_names.astype(str).tolist()
        columns = [f"{_TEXT_PREFIX}{_slug(term)}" for term in terms]
        frame = pd.DataFrame(matrix, columns=columns)
        frame.insert(0, "tile_id", adata.obs_names.astype(str))
        return frame
    return pd.DataFrame(columns=["tile_id"])


def _normalize_tile_feature_table(
    frame: pd.DataFrame,
    *,
    text_terms: Sequence[str],
    max_tiles: int | None,
) -> pd.DataFrame:
    output = pd.DataFrame(frame).copy()
    if output.empty:
        return pd.DataFrame(columns=["tile_id"])
    output.columns = output.columns.map(str)
    if "tile_id" not in output.columns:
        output.insert(0, "tile_id", [f"tile_{index}" for index in range(len(output))])
    output["tile_id"] = output["tile_id"].astype(str)
    output = _standardize_text_columns(output, text_terms=text_terms)
    output = _standardize_embedding_columns(output)
    if max_tiles is not None:
        output = output.iloc[: int(max_tiles)].copy()
    text_columns = [column for column in output.columns if column.startswith(_TEXT_PREFIX)]
    if text_columns and "top_image_label" not in output.columns:
        values = output.loc[:, text_columns].apply(pd.to_numeric, errors="coerce")
        top_columns = values.idxmax(axis=1)
        output["top_image_label"] = [
            column.removeprefix(_TEXT_PREFIX).replace("_", " ")
            if isinstance(column, str)
            else None
            for column in top_columns
        ]
        output["top_image_label_score"] = values.max(axis=1).to_numpy(dtype=float)
    if "domain" in output.columns and "spatial_domain" not in output.columns:
        output["spatial_domain"] = output["domain"].astype(str)
    return output.reset_index(drop=True)


def _standardize_text_columns(frame: pd.DataFrame, *, text_terms: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    known = {str(term): f"{_TEXT_PREFIX}{_slug(term)}" for term in text_terms}
    renames = {}
    for column in output.columns:
        text = str(column)
        if text.startswith(_TEXT_PREFIX):
            renames[column] = f"{_TEXT_PREFIX}{_slug(text.removeprefix(_TEXT_PREFIX))}"
        elif text in known:
            renames[column] = known[text]
    return output.rename(columns=renames)


def _standardize_embedding_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    renames = {}
    for column in output.columns:
        text = str(column)
        if text.startswith(_EMBEDDING_PREFIX):
            renames[column] = f"{_EMBEDDING_PREFIX}{_slug(text.removeprefix(_EMBEDDING_PREFIX))}"
        elif text.startswith(("z", "dim_")) and pd.to_numeric(output[column], errors="coerce").notna().any():
            renames[column] = f"{_EMBEDDING_PREFIX}{_slug(text)}"
    return output.rename(columns=renames)


def _tile_centroid(row: pd.Series) -> Point:
    geometry = row.get("geometry")
    if isinstance(geometry, BaseGeometry) and not geometry.is_empty:
        centroid = geometry.centroid
        return Point(float(centroid.x), float(centroid.y))
    for x_name, y_name in (("tile_x", "tile_y"), ("x", "y"), ("centroid_x", "centroid_y")):
        if x_name in row.index and y_name in row.index:
            return Point(float(row[x_name]), float(row[y_name]))
    raise ValueError(
        "Tile assignment requires a geometry column or centroid columns "
        "(`tile_x`/`tile_y`, `x`/`y`, or `centroid_x`/`centroid_y`)."
    )


def _merge_tile_features_with_assignments(
    tile_features: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    if tile_features.empty:
        return assignments.copy()
    drop_columns = [
        column
        for column in ("contour_id", "assigned_structure", "structure_id", "classification_name")
        if column in tile_features.columns
    ]
    clean = tile_features.drop(columns=drop_columns, errors="ignore")
    return clean.merge(assignments, on="tile_id", how="left")


def _build_structure_rna_summary(
    *,
    sdata: XeniumSlide,
    contour_table: pd.DataFrame,
    genes: Sequence[str],
) -> pd.DataFrame:
    adata = sdata.table
    if "spatial" not in adata.obsm or adata.n_obs == 0:
        return _empty_structure_rna_summary()
    present_genes = [gene for gene in genes if gene in adata.var_names]
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return _empty_structure_rna_summary()
    matrix = _expression_frame(adata, present_genes)
    cell_points = [Point(float(x), float(y)) for x, y in coords[:, :2]]

    rows = []
    for _, contour in contour_table.iterrows():
        geometry = contour["geometry"]
        mask = np.asarray([geometry.covers(point) for point in cell_points], dtype=bool)
        row = {
            "contour_id": str(contour["contour_id"]),
            "assigned_structure": _structure_label(contour),
            "structure_id": _optional_text(contour.get("structure_id")),
            "n_cells": int(mask.sum()),
        }
        if mask.any() and not matrix.empty:
            means = matrix.loc[mask, :].mean(axis=0)
            for gene, value in means.items():
                row[f"rna__{gene}__mean"] = float(value)
        for column in ("cell_type", "cluster", "cell_state"):
            if column in adata.obs.columns and mask.any():
                values = adata.obs.loc[mask, column].astype(str).value_counts()
                for label, count in values.items():
                    row[f"cell_type_fraction__{_slug(column)}__{_slug(label)}"] = float(
                        count / mask.sum()
                    )
        rows.append(row)
    contour_rna = pd.DataFrame(rows)
    if contour_rna.empty:
        return _empty_structure_rna_summary()
    numeric_columns = _numeric_columns(contour_rna, exclude=set(_ID_COLUMNS))
    grouped_rows = []
    for structure, group in contour_rna.groupby("assigned_structure", sort=True, dropna=False):
        row = {
            "assigned_structure": str(structure),
            "structure_id": _mode_or_none(group.get("structure_id")),
            "n_contours": int(group["contour_id"].nunique()),
            "n_cells": int(group["n_cells"].sum()),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            weights = pd.to_numeric(group["n_cells"], errors="coerce").fillna(0.0)
            if values.notna().any() and float(weights.sum()) > 0:
                row[column] = float(np.average(values.fillna(0.0), weights=weights))
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows).reset_index(drop=True)


def _expression_frame(adata: ad.AnnData, genes: Sequence[str]) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame(index=adata.obs_names)
    indices = [int(np.where(adata.var_names == gene)[0][0]) for gene in genes]
    values = adata.X[:, indices]
    if sparse.issparse(values):
        values = values.toarray()
    return pd.DataFrame(np.asarray(values, dtype=float), columns=list(genes), index=adata.obs_names)


def _resolve_program_scores(
    *,
    sdata: XeniumSlide,
    contour_key: str,
    precomputed_feature_table: Mapping[str, Any] | None,
    precomputed_program_scores: pd.DataFrame | str | Path | None,
    program_library: str,
    include_boundary_programs: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if precomputed_program_scores is not None:
        return _read_table(precomputed_program_scores), {"status": "loaded_precomputed"}
    if not include_boundary_programs:
        return pd.DataFrame(columns=["contour_id"]), {"status": "disabled"}
    try:
        feature_table = (
            precomputed_feature_table
            if precomputed_feature_table is not None
            else build_contour_feature_table(
                sdata,
                contour_key=contour_key,
                include_pathomics=False,
            )
        )
        result = score_contour_boundary_programs(
            sdata,
            contour_key=contour_key,
            feature_table=feature_table,
            program_library=program_library,
        )
        return pd.DataFrame(result["program_scores"]).copy(), {"status": "computed"}
    except Exception as exc:
        return (
            pd.DataFrame(columns=["contour_id"]),
            {"status": "skipped", "reason": str(exc)},
        )


def _aggregate_program_scores_by_structure(
    program_scores: pd.DataFrame,
    *,
    contour_table: pd.DataFrame,
) -> pd.DataFrame:
    if program_scores.empty or "contour_id" not in program_scores.columns:
        return pd.DataFrame(columns=["assigned_structure"])
    mapping = contour_table.loc[:, [column for column in _ID_COLUMNS if column in contour_table.columns]].copy()
    if "assigned_structure" not in mapping.columns:
        mapping["assigned_structure"] = contour_table.apply(_structure_label, axis=1)
    merged = pd.DataFrame(program_scores).copy()
    merged["contour_id"] = merged["contour_id"].astype(str)
    mapping["contour_id"] = mapping["contour_id"].astype(str)
    merged = merged.merge(mapping, on="contour_id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=["assigned_structure"])
    program_columns = [
        column
        for column in _numeric_columns(merged, exclude=set(_ID_COLUMNS))
        if not str(column).endswith("_rank")
    ]
    rows = []
    for structure, group in merged.groupby("assigned_structure", sort=True, dropna=False):
        row = {
            "assigned_structure": str(structure),
            "n_contours": int(group["contour_id"].nunique()),
        }
        for column in program_columns:
            row[f"program__{column}__mean"] = float(pd.to_numeric(group[column], errors="coerce").mean())
        if "top_program" in group.columns:
            row["top_program"] = _mode_or_none(group["top_program"])
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _associate_structure_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_prefixes: Sequence[str],
    right_prefixes: Sequence[str],
    association_kind: str,
) -> pd.DataFrame:
    columns = [
        "association_kind",
        "left_feature",
        "right_feature",
        "spearman_rho",
        "abs_spearman_rho",
        "p_value",
        "n_structures",
    ]
    if left.empty or right.empty or "assigned_structure" not in left or "assigned_structure" not in right:
        return pd.DataFrame(columns=columns)
    merged = left.merge(right, on="assigned_structure", how="inner", suffixes=("", "__right"))
    if len(merged) < 3:
        return pd.DataFrame(columns=columns)
    left_columns = [
        column
        for column in _numeric_columns(merged, exclude={"n_tiles", "n_contours"})
        if any(str(column).startswith(prefix) for prefix in left_prefixes)
    ]
    right_columns = [
        column
        for column in _numeric_columns(merged, exclude={"n_tiles", "n_contours"})
        if any(str(column).startswith(prefix) for prefix in right_prefixes)
    ]
    rows = []
    for left_column in left_columns:
        left_values = pd.to_numeric(merged[left_column], errors="coerce")
        for right_column in right_columns:
            right_values = pd.to_numeric(merged[right_column], errors="coerce")
            mask = left_values.notna() & right_values.notna()
            if int(mask.sum()) < 3:
                continue
            rho, p_value = spearmanr(left_values.loc[mask], right_values.loc[mask])
            if not np.isfinite(rho):
                continue
            rows.append(
                {
                    "association_kind": association_kind,
                    "left_feature": str(left_column),
                    "right_feature": str(right_column),
                    "spearman_rho": float(rho),
                    "abs_spearman_rho": float(abs(rho)),
                    "p_value": float(p_value),
                    "n_structures": int(mask.sum()),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["abs_spearman_rho", "left_feature"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
    return result


def _differential_image_features(tile_features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "assigned_structure",
        "feature_name",
        "feature_kind",
        "group_mean",
        "rest_mean",
        "effect_size",
        "p_value",
        "fdr",
        "n_group_tiles",
        "n_rest_tiles",
    ]
    if tile_features.empty or "assigned_structure" not in tile_features.columns:
        return pd.DataFrame(columns=columns)
    assigned = tile_features[tile_features["assigned_structure"].notna()].copy()
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    numeric_columns = _image_numeric_columns(assigned)
    rows = []
    for structure, group in assigned.groupby("assigned_structure", sort=True, dropna=False):
        rest = assigned[assigned["assigned_structure"] != structure]
        if rest.empty:
            continue
        for column in numeric_columns:
            left = pd.to_numeric(group[column], errors="coerce").dropna()
            right = pd.to_numeric(rest[column], errors="coerce").dropna()
            if left.empty or right.empty:
                continue
            p_value = np.nan
            if len(left) >= 2 and len(right) >= 2:
                _, p_value = ttest_ind(left, right, equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "assigned_structure": str(structure),
                    "feature_name": str(column),
                    "feature_kind": _feature_kind(column),
                    "group_mean": float(left.mean()),
                    "rest_mean": float(right.mean()),
                    "effect_size": float(left.mean() - right.mean()),
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "n_group_tiles": int(len(left)),
                    "n_rest_tiles": int(len(right)),
                }
            )
    result = pd.DataFrame(rows, columns=[column for column in columns if column != "fdr"])
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["fdr"] = _benjamini_hochberg(result["p_value"])
    result["abs_effect_size"] = result["effect_size"].abs()
    result = result.sort_values(
        ["abs_effect_size", "assigned_structure"],
        ascending=[False, True],
        kind="stable",
    ).drop(columns=["abs_effect_size"])
    return result.loc[:, columns].reset_index(drop=True)


def _build_manifest(
    *,
    config: HistoSegLazySlideConfig,
    sdata: XeniumSlide,
    model_result: Mapping[str, Any],
    program_status: Mapping[str, Any],
    n_contours: int,
    n_tiles: int,
    n_assigned_tiles: int,
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "histoseg_lazyslide_structure_workflow",
        "sample_id": _resolve_sample_id(sdata),
        "created_at_unix": float(time.time()),
        "runtime_seconds": float(time.time() - started),
        "config": config.to_dict(),
        "package_boundaries": {
            "HistoSeg": "Owns tissue structure segmentation and contour generation.",
            "LazySlide": "Owns WSI tiling and PLIP/CONCH image model inference.",
            "pyXenium": (
                "Owns Xenium alignment, structure-level RNA/image summaries, "
                "and interpretable multimodal associations."
            ),
        },
        "inputs": {
            "n_cells": int(sdata.table.n_obs),
            "n_features": int(sdata.table.n_vars),
            "contour_key": config.contour_key,
            "he_image_key": config.he_image_key,
            "he_source_path": sdata.images[config.he_image_key].source_path
            if config.he_image_key in sdata.images
            else None,
        },
        "outputs": {
            "n_contours": int(n_contours),
            "n_tiles": int(n_tiles),
            "n_assigned_tiles": int(n_assigned_tiles),
            "assignment_fraction": float(n_assigned_tiles / n_tiles) if n_tiles else 0.0,
        },
        "model_status": dict(model_result.get("model_status", {})),
        "program_status": dict(program_status),
        "git_commit": _git_commit(),
        "gpu": _gpu_summary(),
    }


def _write_workflow_artifacts(
    result: Mapping[str, Any],
    output_dir: str | Path,
    *,
    table_format: Literal["csv", "parquet"],
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "image_contours": _write_table(
            result["image_contours"],
            out / f"image_contours.{table_format}",
            table_format=table_format,
        ),
        "tile_features": _write_table(
            result["tile_features"],
            out / f"tile_features.{table_format}",
            table_format=table_format,
        ),
        "tile_assignments": _write_table(
            result["tile_assignments"],
            out / f"tile_assignments.{table_format}",
            table_format=table_format,
        ),
        "structure_image_features": _write_table(
            result["structure_image_features"],
            out / f"structure_image_features.{table_format}",
            table_format=table_format,
        ),
        "structure_differential_features": _write_table(
            result["structure_differential_features"],
            out / f"structure_differential_features.{table_format}",
            table_format=table_format,
        ),
        "structure_rna_summary": _write_table(
            result["structure_rna_summary"],
            out / f"structure_rna_summary.{table_format}",
            table_format=table_format,
        ),
        "structure_program_scores": _write_table(
            result["structure_program_scores"],
            out / f"structure_program_scores.{table_format}",
            table_format=table_format,
        ),
        "rna_image_associations": _write_table(
            result["rna_image_associations"],
            out / f"rna_image_associations.{table_format}",
            table_format=table_format,
        ),
        "program_image_associations": _write_table(
            result["program_image_associations"],
            out / f"program_image_associations.{table_format}",
            table_format=table_format,
        ),
    }
    manifest = dict(result["run_manifest"])
    manifest["files"] = files
    (out / "run_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["run_manifest"].update({"files": files})


def _read_table(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        path = Path(value)
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    frame.columns = frame.columns.map(str)
    return frame


def _write_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    table_format: Literal["csv", "parquet"],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _serializable_frame(frame)
    if table_format == "parquet":
        serializable.to_parquet(path, index=False)
    else:
        serializable.to_csv(path, index=False)
    return path.name


def _serializable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(frame).copy()
    for column in list(output.columns):
        if any(isinstance(value, BaseGeometry) for value in output[column].dropna().head(5)):
            output[f"{column}_wkt"] = output[column].map(lambda value: value.wkt if isinstance(value, BaseGeometry) else None)
            output = output.drop(columns=[column])
    return output


def _image_numeric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "tile_id",
        "tile_x",
        "tile_y",
        "assigned",
        "contour_id",
        "structure_id",
    }
    return [
        str(column)
        for column in frame.columns
        if str(column) not in excluded
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
        and (
            str(column).startswith(_EMBEDDING_PREFIX)
            or str(column).startswith(_TEXT_PREFIX)
            or str(column) in {"top_image_label_score"}
        )
    ]


def _numeric_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) not in exclude
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _feature_kind(column: str) -> str:
    text = str(column)
    if text.startswith(_TEXT_PREFIX):
        return "text_similarity"
    if text.startswith(_EMBEDDING_PREFIX):
        return "embedding"
    if text.startswith("domain_fraction__"):
        return "spatial_domain"
    return "image_feature"


def _structure_label(row: pd.Series) -> str:
    for column in ("assigned_structure", "classification_name", "name", "structure_id"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value):
            return str(value)
    return str(row.get("contour_id", "unlabeled_structure"))


def _mode_or_none(values: Any) -> str | None:
    if values is None:
        return None
    series = pd.Series(values).dropna().astype(str)
    if series.empty:
        return None
    return str(series.value_counts().index[0])


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_path_text(value: str | Path | None) -> str | None:
    return None if value is None else str(Path(value))


def _empty_structure_image_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "assigned_structure",
            "structure_id",
            "classification_name",
            "n_tiles",
            "n_contours",
            "top_image_label",
            "top_image_label_fraction",
        ]
    )


def _empty_structure_rna_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["assigned_structure", "structure_id", "n_contours", "n_cells"])


def _mean_rank(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.nan
    ranks = numeric.rank(ascending=False, method="average")
    return float(ranks.mean())


def _benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    output = np.full_like(p_values, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    finite = p_values[finite_mask]
    if finite.size == 0:
        return output
    order = np.argsort(finite)
    ranked = finite[order]
    n = float(len(ranked))
    adjusted = ranked * n / (np.arange(len(ranked), dtype=float) + 1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    finite_output = np.empty_like(adjusted)
    finite_output[order] = adjusted
    output[finite_mask] = finite_output
    return output


def _resolve_sample_id(sdata: XeniumSlide) -> str:
    for key in ("sample_id", "dataset_id", "source_path"):
        value = sdata.metadata.get(key)
        if value:
            return str(value)
    for key in ("sample_id", "dataset_id"):
        value = sdata.table.uns.get(key)
        if value:
            return str(value)
    return "sample_0"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _gpu_summary() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {"available": False}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"available": bool(lines), "devices": lines}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    chars = []
    previous_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_sep = False
        elif not previous_sep:
            chars.append("_")
            previous_sep = True
    return "".join(chars).strip("_") or "value"
