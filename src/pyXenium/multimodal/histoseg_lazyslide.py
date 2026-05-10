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
from scipy.stats import pearsonr, spearmanr, t, ttest_ind
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from pyXenium.contour import add_contours_from_geojson, build_contour_feature_table
from pyXenium.contour._analysis import _prepare_contours
from pyXenium.contour.loading import _geometry_um_to_image_xy
from pyXenium.io import read_slide, read_xenium
from pyXenium.io.slide_model import XeniumImage, XeniumSlide

from .contour_boundary_ecology import score_contour_boundary_programs

__all__ = [
    "DEFAULT_LAZYSLIDE_TEXT_TERMS",
    "DEFAULT_RELATIVE_PROMPT_AXES",
    "DEFAULT_STRUCTURE_RNA_MARKERS",
    "DEFAULT_WTA_GENE_PROGRAMS",
    "HistoSegLazySlideConfig",
    "aggregate_structure_image_features",
    "associate_contour_image_molecular_features",
    "assign_tiles_to_histoseg_structures",
    "benchmark_contour_molecular_prediction",
    "build_contour_multimodal_summary",
    "histoseg_contours_to_image_table",
    "run_histoseg_lazyslide_structure_workflow",
    "summarize_morphomolecular_evidence",
    "summarize_wta_pathway_partial_correlations",
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

DEFAULT_RELATIVE_PROMPT_AXES = (
    ("carcinoma_vs_ductal", "invasive carcinoma", "ductal epithelium"),
    ("fibrotic_vs_immune", "fibrotic stroma", "immune infiltrate"),
    ("necrosis_vs_ductal", "necrosis", "ductal epithelium"),
    ("vascular_vs_adipose", "vascular stroma", "adipose tissue"),
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

DEFAULT_WTA_GENE_PROGRAMS: dict[str, tuple[str, ...]] = {
    "epithelial_identity": (
        "EPCAM",
        "KRT7",
        "KRT8",
        "KRT18",
        "KRT19",
        "MUC1",
        "TACSTD2",
        "CLDN4",
        "CLDN7",
    ),
    "luminal_estrogen_response": (
        "ESR1",
        "PGR",
        "GATA3",
        "FOXA1",
        "XBP1",
        "AGR2",
        "TFF1",
        "TFF3",
        "SCUBE2",
    ),
    "her2_amplicon_signaling": (
        "ERBB2",
        "GRB7",
        "PGAP3",
        "MIEN1",
        "STARD3",
        "TCAP",
        "PSMD3",
    ),
    "basal_squamous_state": (
        "KRT5",
        "KRT14",
        "KRT17",
        "TP63",
        "LAMB3",
        "LAMC2",
        "ITGA6",
    ),
    "cell_cycle_proliferation": (
        "MKI67",
        "TOP2A",
        "UBE2C",
        "CCNB1",
        "CCNB2",
        "CDK1",
        "PCNA",
        "MCM2",
        "MCM5",
        "BIRC5",
        "CENPF",
        "AURKA",
    ),
    "hypoxia_glycolysis": (
        "CA9",
        "SLC2A1",
        "LDHA",
        "ENO1",
        "PGK1",
        "ALDOA",
        "VEGFA",
        "HILPDA",
        "BNIP3",
        "NDRG1",
        "PFKP",
    ),
    "oxidative_phosphorylation": (
        "NDUFA1",
        "NDUFB8",
        "COX5A",
        "COX6C",
        "ATP5F1A",
        "ATP5MC1",
        "UQCRC1",
        "SDHB",
    ),
    "emt_invasion": (
        "VIM",
        "FN1",
        "SNAI2",
        "TWIST1",
        "ZEB1",
        "MMP2",
        "MMP9",
        "MMP11",
        "ITGA5",
        "ITGB1",
        "CDH2",
        "SPARC",
    ),
    "collagen_ecm_organization": (
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "COL5A2",
        "COL6A1",
        "COL6A2",
        "DCN",
        "LUM",
        "POSTN",
        "THBS2",
    ),
    "myofibroblast_caf_activation": (
        "ACTA2",
        "TAGLN",
        "MYL9",
        "COL11A1",
        "FAP",
        "PDPN",
        "PDGFRA",
        "PDGFRB",
        "CTHRC1",
    ),
    "tgf_beta_response": (
        "TGFB1",
        "TGFBR1",
        "TGFBR2",
        "SMAD2",
        "SMAD3",
        "SERPINE1",
        "CTGF",
        "INHBA",
        "THBS1",
        "PMEPA1",
    ),
    "angiogenesis_endothelial": (
        "PECAM1",
        "VWF",
        "KDR",
        "FLT1",
        "ESAM",
        "EMCN",
        "ENG",
        "PLVAP",
        "CDH5",
        "RAMP2",
    ),
    "pericyte_smooth_muscle": (
        "RGS5",
        "MCAM",
        "CSPG4",
        "PDGFRB",
        "ACTA2",
        "MYH11",
        "NOTCH3",
        "DES",
    ),
    "t_cell_cytotoxicity": (
        "CD3D",
        "CD3E",
        "TRAC",
        "CD8A",
        "CD8B",
        "GZMB",
        "GZMK",
        "PRF1",
        "NKG7",
        "GNLY",
        "IFNG",
    ),
    "t_cell_exhaustion_checkpoint": (
        "PDCD1",
        "CTLA4",
        "LAG3",
        "HAVCR2",
        "TIGIT",
        "TOX",
        "CXCL13",
        "ENTPD1",
        "LAYN",
    ),
    "tls_b_cell_plasma": (
        "MS4A1",
        "CD79A",
        "CD79B",
        "CD19",
        "MZB1",
        "JCHAIN",
        "IGHG1",
        "IGKC",
        "CXCL13",
        "LTB",
    ),
    "myeloid_spp1_macrophage": (
        "CD68",
        "CD163",
        "LST1",
        "TYROBP",
        "FCGR3A",
        "SPP1",
        "APOE",
        "C1QA",
        "C1QB",
        "C1QC",
        "LILRB4",
    ),
    "antigen_presentation_interferon": (
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DPA1",
        "HLA-DPB1",
        "B2M",
        "TAP1",
        "STAT1",
        "IRF1",
        "IFIT1",
        "ISG15",
        "CXCL10",
    ),
    "complement_inflammation": (
        "C1QA",
        "C1QB",
        "C1QC",
        "C3",
        "C4A",
        "CFD",
        "CFB",
        "SERPING1",
        "IL1B",
        "CXCL8",
        "CCL2",
    ),
    "chemokine_recruitment": (
        "CXCL9",
        "CXCL10",
        "CXCL11",
        "CXCL13",
        "CCL2",
        "CCL3",
        "CCL4",
        "CCL5",
        "CCL19",
        "CCL21",
    ),
    "interferon_alpha_response": (
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "ISG15",
        "MX1",
        "OAS1",
        "OAS2",
        "IFI6",
        "IFI44L",
    ),
    "p53_apoptosis_stress": (
        "TP53",
        "CDKN1A",
        "MDM2",
        "BAX",
        "PMAIP1",
        "BBC3",
        "GADD45A",
        "FAS",
        "DDB2",
    ),
    "unfolded_protein_response": (
        "XBP1",
        "DDIT3",
        "HSPA5",
        "HERPUD1",
        "ATF3",
        "ATF4",
        "ERN1",
        "PDIA3",
        "HSP90B1",
    ),
    "adipocyte_lipid_state": (
        "ADIPOQ",
        "PLIN1",
        "LEP",
        "FABP4",
        "LPL",
        "PPARG",
        "CEBPA",
        "APOE",
        "LIPE",
    ),
    "ribosomal_translation": (
        "RPLP0",
        "RPSA",
        "RPL13A",
        "RPS3",
        "RPL10",
        "RPS6",
        "EIF4E",
        "EEF1A1",
    ),
}

_SCHEMA_VERSION = "pyxenium-histoseg-lazyslide-v1"
_TABLE_FORMATS = ("csv", "parquet")
_ID_COLUMNS = ("contour_id", "assigned_structure", "structure_id")
_TEXT_PREFIX = "text_similarity__"
_EMBEDDING_PREFIX = "embedding__"
_LAZYSLIDE_TILE_TEXT_MODELS = ("plip", "conch", "omiclip")


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
    wsi_reader: str | None = None
    slide_mpp: float | None = None
    model: str = "plip"
    text_model: str | None = None
    text_terms: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LAZYSLIDE_TEXT_TERMS)
    prompt_set_name: str = "breast_histology_v1"
    prompt_source: str = "manual exploratory prompt set"
    prompt_review_status: str = "not pathologist-confirmed"
    relative_prompt_axes: tuple[tuple[str, str, str], ...] = field(
        default_factory=lambda: DEFAULT_RELATIVE_PROMPT_AXES
    )
    tile_px: int = 224
    mpp: float = 0.5
    device: str = "cuda"
    amp: bool = True
    batch_size: int = 64
    max_tiles: int | None = None
    table_format: Literal["csv", "parquet"] = "csv"
    include_rna: bool = True
    include_wta_programs: bool = True
    include_boundary_programs: bool = True
    include_prediction_benchmark: bool = True
    wta_program_library: str = "breast_tme_wta_v1"
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
    wsi_reader: str | None = None,
    slide_mpp: float | None = None,
    model: str = "plip",
    text_model: str | None = None,
    text_terms: Sequence[str] | None = None,
    prompt_set_name: str = "breast_histology_v1",
    prompt_source: str = "manual exploratory prompt set",
    prompt_review_status: str = "not pathologist-confirmed",
    relative_prompt_axes: Sequence[Sequence[str]] | None = None,
    tile_px: int = 224,
    mpp: float = 0.5,
    device: str = "cuda",
    amp: bool = True,
    batch_size: int = 64,
    max_tiles: int | None = None,
    table_format: Literal["csv", "parquet"] = "csv",
    include_rna: bool = True,
    include_wta_programs: bool = True,
    include_boundary_programs: bool = True,
    include_prediction_benchmark: bool = True,
    wta_program_library: str = "breast_tme_wta_v1",
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
        wsi_reader=wsi_reader,
        slide_mpp=slide_mpp,
        model=model,
        text_model=text_model,
        text_terms=tuple(text_terms or DEFAULT_LAZYSLIDE_TEXT_TERMS),
        prompt_set_name=str(prompt_set_name),
        prompt_source=str(prompt_source),
        prompt_review_status=str(prompt_review_status),
        relative_prompt_axes=_normalize_relative_prompt_axes(relative_prompt_axes),
        tile_px=int(tile_px),
        mpp=float(mpp),
        device=str(device),
        amp=bool(amp),
        batch_size=int(batch_size),
        max_tiles=max_tiles,
        table_format=table_format,
        include_rna=bool(include_rna),
        include_wta_programs=bool(include_wta_programs),
        include_boundary_programs=bool(include_boundary_programs),
        include_prediction_benchmark=bool(include_prediction_benchmark),
        wta_program_library=str(wta_program_library),
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

    contour_rna = (
        _build_contour_rna_summary(
            sdata=sdata,
            contour_table=contour_table,
            genes=config.rna_markers,
        )
        if config.include_rna
        else _empty_contour_rna_summary()
    )
    structure_rna = _aggregate_contour_rna_by_structure(contour_rna)
    rna_image_associations = _associate_structure_tables(
        structure_image_features,
        structure_rna,
        left_prefixes=(_TEXT_PREFIX, _EMBEDDING_PREFIX, "domain_fraction__"),
        right_prefixes=("rna__", "cell_type_fraction__"),
        association_kind="rna_image_association",
    )

    program_scores, program_status = _resolve_program_scores(
        sdata=sdata,
        contour_table=contour_table,
        contour_key=config.contour_key,
        precomputed_feature_table=precomputed_feature_table,
        precomputed_program_scores=precomputed_program_scores,
        wta_program_library=config.wta_program_library,
        program_library=config.program_library,
        include_wta_programs=config.include_wta_programs,
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
    contour_multimodal = build_contour_multimodal_summary(
        image_contours=image_contours,
        tile_features=assigned_tile_features,
        contour_rna_summary=contour_rna,
        program_scores=program_scores,
        text_terms=config.text_terms,
        relative_prompt_axes=config.relative_prompt_axes,
    )
    contour_image_molecular_associations = associate_contour_image_molecular_features(
        contour_multimodal
    )
    wta_pathway_partial_correlations = summarize_wta_pathway_partial_correlations(
        contour_image_molecular_associations
    )
    molecular_prediction_benchmark = (
        benchmark_contour_molecular_prediction(contour_multimodal)
        if config.include_prediction_benchmark
        else _empty_molecular_prediction_benchmark()
    )
    morphomolecular_evidence = summarize_morphomolecular_evidence(
        contour_multimodal,
        prediction_benchmark=molecular_prediction_benchmark,
        associations=contour_image_molecular_associations,
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
        "contour_multimodal_summary": contour_multimodal,
        "structure_image_features": structure_image_features,
        "structure_differential_features": differential,
        "structure_rna_summary": structure_rna,
        "structure_program_scores": structure_programs,
        "contour_image_molecular_associations": contour_image_molecular_associations,
        "wta_pathway_partial_correlations": wta_pathway_partial_correlations,
        "molecular_prediction_benchmark": molecular_prediction_benchmark,
        "morphomolecular_hero_targets": morphomolecular_evidence["hero_targets"],
        "morphomolecular_hero_contours": morphomolecular_evidence["hero_contours"],
        "morphomolecular_concept_tests": morphomolecular_evidence["concept_tests"],
        "boundary_coupling_summary": morphomolecular_evidence["boundary_coupling_summary"],
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
        prompt_column = "top_prompt_term" if "top_prompt_term" in group.columns else "top_image_label"
        if prompt_column in group.columns:
            label_counts = group[prompt_column].dropna().astype(str).value_counts()
            if not label_counts.empty:
                row["top_prompt_term"] = str(label_counts.index[0])
                row["top_prompt_term_fraction"] = float(label_counts.iloc[0] / len(group))
                row["top_image_label"] = row["top_prompt_term"]
                row["top_image_label_fraction"] = row["top_prompt_term_fraction"]
        if "spatial_domain" in group.columns:
            for domain, count in group["spatial_domain"].dropna().astype(str).value_counts().items():
                row[f"domain_fraction__{_slug(domain)}"] = float(count / len(group))
        for term in text_terms:
            column = f"{_TEXT_PREFIX}{_slug(term)}"
            if column in group.columns:
                row[f"{column}__rank"] = _mean_rank(group[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("assigned_structure", kind="stable").reset_index(drop=True)


def build_contour_multimodal_summary(
    *,
    image_contours: pd.DataFrame,
    tile_features: pd.DataFrame,
    contour_rna_summary: pd.DataFrame | None = None,
    program_scores: pd.DataFrame | None = None,
    text_terms: Sequence[str] | None = None,
    relative_prompt_axes: Sequence[Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Build a contour-level H&E/Xenium table for morphomolecular analyses."""

    if image_contours.empty:
        return _empty_contour_multimodal_summary()
    base = _base_contour_summary(image_contours)
    image_summary = _aggregate_contour_image_features(
        tile_features,
        text_terms=text_terms,
        relative_prompt_axes=relative_prompt_axes,
    )
    summary = _merge_contour_summary(base, image_summary)
    if contour_rna_summary is not None and not contour_rna_summary.empty:
        summary = _merge_contour_summary(summary, contour_rna_summary)
    if program_scores is not None and not program_scores.empty:
        summary = _merge_contour_summary(summary, _prefix_contour_program_scores(program_scores))
    summary = _add_tile_boundary_distance_features(summary, image_contours, tile_features)
    summary = _add_density_features(summary)
    return summary.sort_values(["assigned_structure", "contour_id"], kind="stable").reset_index(drop=True)


def associate_contour_image_molecular_features(
    contour_summary: pd.DataFrame,
    *,
    controls: Sequence[str] = (
        "assigned_structure",
        "centroid_x",
        "centroid_y",
        "cell_boundary_distance_um__mean",
        "tile_boundary_distance_px__mean",
    ),
    min_contours: int = 6,
) -> pd.DataFrame:
    """Associate contour image features with molecular features after residualizing controls."""

    columns = [
        "association_kind",
        "image_feature",
        "molecular_feature",
        "spearman_rho",
        "partial_spearman_rho",
        "abs_partial_spearman_rho",
        "p_value",
        "fdr",
        "n_contours",
        "controls",
    ]
    if contour_summary.empty or len(contour_summary) < min_contours:
        return pd.DataFrame(columns=columns)
    image_columns = _contour_image_feature_columns(contour_summary)
    molecular_columns = _contour_molecular_feature_columns(contour_summary)
    if not image_columns or not molecular_columns:
        return pd.DataFrame(columns=columns)
    result = _matrix_partial_spearman_associations(
        contour_summary,
        image_columns=image_columns,
        molecular_columns=molecular_columns,
        controls=controls,
        min_contours=min_contours,
    )
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["fdr"] = _benjamini_hochberg(result["p_value"])
    return result.loc[:, columns].sort_values(
        ["abs_partial_spearman_rho", "image_feature"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def benchmark_contour_molecular_prediction(
    contour_summary: pd.DataFrame,
    *,
    min_contours: int = 12,
    max_targets: int = 50,
    random_state: int = 17,
) -> pd.DataFrame:
    """Benchmark added molecular-program predictivity from contour image features."""

    columns = [
        "target_feature",
        "n_contours",
        "n_image_features",
        "cv_strategy",
        "r2_structure_only",
        "r2_image_only",
        "r2_structure_image",
        "delta_r2_image_over_structure",
        "delta_r2_combined_over_structure",
    ]
    if contour_summary.empty or len(contour_summary) < min_contours:
        return pd.DataFrame(columns=columns)
    image_columns = _contour_image_feature_columns(contour_summary)
    molecular_columns = _contour_molecular_feature_columns(contour_summary)
    if not image_columns or not molecular_columns or "assigned_structure" not in contour_summary.columns:
        return pd.DataFrame(columns=columns)

    image_matrix = _finite_feature_matrix(contour_summary, image_columns)
    structure_matrix = pd.get_dummies(
        contour_summary["assigned_structure"].fillna("unassigned").astype(str),
        prefix="structure",
        dtype=float,
    )
    if image_matrix.empty or structure_matrix.empty:
        return pd.DataFrame(columns=columns)

    targets = _select_prediction_targets(contour_summary, molecular_columns, max_targets=max_targets)
    if not targets:
        return pd.DataFrame(columns=columns)
    image_available = _image_available_mask(contour_summary, image_columns)
    groups, cv_strategy = _spatial_cv_groups(contour_summary)
    rows = []
    for target in targets:
        y = pd.to_numeric(contour_summary[target], errors="coerce")
        mask = y.notna() & image_available
        if int(mask.sum()) < min_contours or float(y.loc[mask].std(ddof=0)) == 0.0:
            continue
        y_values = y.loc[mask].astype(float).to_numpy()
        structure_r2 = _cross_validated_r2(
            structure_matrix.loc[mask, :],
            y_values,
            groups.loc[mask] if groups is not None else None,
            random_state=random_state,
        )
        image_r2 = _cross_validated_r2(
            image_matrix.loc[mask, :],
            y_values,
            groups.loc[mask] if groups is not None else None,
            random_state=random_state,
        )
        combined = pd.concat([structure_matrix, image_matrix], axis=1)
        combined_r2 = _cross_validated_r2(
            combined.loc[mask, :],
            y_values,
            groups.loc[mask] if groups is not None else None,
            random_state=random_state,
        )
        rows.append(
            {
                "target_feature": str(target),
                "n_contours": int(mask.sum()),
                "n_image_features": int(image_matrix.shape[1]),
                "cv_strategy": cv_strategy,
                "r2_structure_only": structure_r2,
                "r2_image_only": image_r2,
                "r2_structure_image": combined_r2,
                "delta_r2_image_over_structure": image_r2 - structure_r2,
                "delta_r2_combined_over_structure": combined_r2 - structure_r2,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["delta_r2_combined_over_structure", "target_feature"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
    return result


def summarize_morphomolecular_evidence(
    contour_summary: pd.DataFrame,
    *,
    prediction_benchmark: pd.DataFrame | None = None,
    associations: pd.DataFrame | None = None,
    max_targets: int = 12,
    max_contours_per_target: int = 6,
    min_tiles_per_hero_contour: int = 2,
    min_cells_per_hero_contour: int = 25,
) -> dict[str, pd.DataFrame]:
    """Create manuscript-facing evidence tables for morphomolecular translation."""

    prediction_benchmark = (
        prediction_benchmark.copy()
        if prediction_benchmark is not None
        else _empty_molecular_prediction_benchmark()
    )
    associations = (
        associations.copy()
        if associations is not None
        else pd.DataFrame(columns=_empty_contour_image_molecular_associations_columns())
    )
    hero_targets = _select_hero_targets(
        prediction_benchmark,
        associations,
        max_targets=max_targets,
    )
    hero_contours = _select_hero_contours(
        contour_summary,
        hero_targets=hero_targets,
        associations=associations,
        max_contours_per_target=max_contours_per_target,
        min_tiles=min_tiles_per_hero_contour,
        min_cells=min_cells_per_hero_contour,
    )
    concept_tests = _build_morphomolecular_concept_tests(associations)
    boundary_coupling = _build_boundary_coupling_summary(associations)
    return {
        "hero_targets": hero_targets,
        "hero_contours": hero_contours,
        "concept_tests": concept_tests,
        "boundary_coupling_summary": boundary_coupling,
    }


def summarize_wta_pathway_partial_correlations(
    associations: pd.DataFrame,
    *,
    max_pathways: int = 20,
    program_prefix: str = "program__wta_",
) -> pd.DataFrame:
    """Rank WTA pathway programs by their strongest residual H&E association."""

    columns = [
        "rank",
        "pathway",
        "molecular_feature",
        "best_image_feature",
        "image_axis_family",
        "partial_spearman_rho",
        "abs_partial_spearman_rho",
        "fdr",
        "n_contours",
        "controls",
        "interpretation",
    ]
    if associations.empty or "molecular_feature" not in associations.columns:
        return pd.DataFrame(columns=columns)
    frame = associations.copy()
    mask = frame["molecular_feature"].astype(str).str.startswith(str(program_prefix))
    frame = frame.loc[mask].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["abs_partial_spearman_rho"] = pd.to_numeric(
        frame["abs_partial_spearman_rho"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["abs_partial_spearman_rho"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.sort_values(
        ["abs_partial_spearman_rho", "molecular_feature", "image_feature"],
        ascending=[False, True, True],
        kind="stable",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        molecular = str(row["molecular_feature"])
        if molecular in seen:
            continue
        seen.add(molecular)
        image_feature = str(row.get("image_feature", ""))
        pathway = _program_feature_label(molecular, prefix=program_prefix)
        rows.append(
            {
                "rank": len(rows) + 1,
                "pathway": pathway,
                "molecular_feature": molecular,
                "best_image_feature": image_feature,
                "image_axis_family": _image_axis_family(image_feature),
                "partial_spearman_rho": row.get("partial_spearman_rho", np.nan),
                "abs_partial_spearman_rho": row.get("abs_partial_spearman_rho", np.nan),
                "fdr": row.get("fdr", np.nan),
                "n_contours": row.get("n_contours", np.nan),
                "controls": row.get("controls", ""),
                "interpretation": _wta_pathway_interpretation(pathway, image_feature),
            }
        )
        if len(rows) >= int(max_pathways):
            break
    return pd.DataFrame(rows, columns=columns)


def _select_hero_targets(
    prediction_benchmark: pd.DataFrame,
    associations: pd.DataFrame,
    *,
    max_targets: int,
) -> pd.DataFrame:
    columns = [
        "target_feature",
        "r2_structure_only",
        "r2_image_only",
        "r2_structure_image",
        "delta_r2_combined_over_structure",
        "best_image_feature",
        "best_partial_spearman_rho",
        "best_association_fdr",
        "hero_score",
        "evidence_mode",
        "storyline",
    ]
    if prediction_benchmark.empty or "target_feature" not in prediction_benchmark.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in prediction_benchmark.iterrows():
        target = str(row["target_feature"])
        linked = pd.DataFrame()
        if not associations.empty and "molecular_feature" in associations.columns:
            linked = associations[associations["molecular_feature"].astype(str) == target].copy()
            if not linked.empty and "abs_partial_spearman_rho" in linked.columns:
                linked = linked.sort_values(
                    ["abs_partial_spearman_rho", "image_feature"],
                    ascending=[False, True],
                    kind="stable",
                )
        best = linked.iloc[0] if not linked.empty else pd.Series(dtype=object)
        delta = float(row.get("delta_r2_combined_over_structure", np.nan))
        partial = float(best.get("abs_partial_spearman_rho", np.nan)) if not best.empty else np.nan
        r2_combined = float(row.get("r2_structure_image", np.nan))
        partial_component = _positive(partial)
        r2_component = max(_positive(r2_combined), 0.1)
        delta_component = 1.0 + _positive(delta)
        hero_score = partial_component * r2_component * delta_component
        evidence_mode = (
            "added_prediction_and_residual_association"
            if _positive(delta) > 0 and partial_component > 0
            else "within_structure_residual_association"
            if partial_component > 0
            else "prediction_benchmark_only"
        )
        rows.append(
            {
                "target_feature": target,
                "r2_structure_only": row.get("r2_structure_only", np.nan),
                "r2_image_only": row.get("r2_image_only", np.nan),
                "r2_structure_image": row.get("r2_structure_image", np.nan),
                "delta_r2_combined_over_structure": delta,
                "best_image_feature": best.get("image_feature") if not best.empty else None,
                "best_partial_spearman_rho": best.get("partial_spearman_rho") if not best.empty else np.nan,
                "best_association_fdr": best.get("fdr") if not best.empty else np.nan,
                "hero_score": float(hero_score),
                "evidence_mode": evidence_mode,
                "storyline": _target_storyline(target),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        ["hero_score", "delta_r2_combined_over_structure", "target_feature"],
        ascending=[False, False, True],
        kind="stable",
    ).head(max_targets).reset_index(drop=True)


def _select_hero_contours(
    contour_summary: pd.DataFrame,
    *,
    hero_targets: pd.DataFrame,
    associations: pd.DataFrame,
    max_contours_per_target: int,
    min_tiles: int,
    min_cells: int,
) -> pd.DataFrame:
    columns = [
        "target_feature",
        "image_feature",
        "contour_id",
        "assigned_structure",
        "target_value",
        "image_value",
        "target_z_within_structure",
        "image_z_within_structure",
        "concordance_score",
        "hidden_program_score",
        "centroid_x",
        "centroid_y",
        "n_tiles",
        "n_cells",
        "recommended_figure_use",
    ]
    if contour_summary.empty or hero_targets.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, target_row in hero_targets.iterrows():
        target = str(target_row["target_feature"])
        image_feature = target_row.get("best_image_feature")
        if image_feature is None or pd.isna(image_feature):
            image_feature = _fallback_image_feature_for_target(target, contour_summary, associations)
        if image_feature is None or image_feature not in contour_summary.columns or target not in contour_summary.columns:
            continue
        target_values = pd.to_numeric(contour_summary[target], errors="coerce")
        image_values = pd.to_numeric(contour_summary[image_feature], errors="coerce")
        mask = target_values.notna() & image_values.notna()
        if "n_tiles" in contour_summary.columns:
            mask &= pd.to_numeric(contour_summary["n_tiles"], errors="coerce").fillna(0) >= min_tiles
        if "n_cells" in contour_summary.columns:
            mask &= pd.to_numeric(contour_summary["n_cells"], errors="coerce").fillna(0) >= min_cells
        if int(mask.sum()) < 3:
            continue
        working = contour_summary.loc[mask, :].copy()
        working["_target_value"] = target_values.loc[mask].to_numpy(dtype=float)
        working["_image_value"] = image_values.loc[mask].to_numpy(dtype=float)
        working["_target_z"] = _within_group_zscore(working["_target_value"], working.get("assigned_structure"))
        working["_image_z"] = _within_group_zscore(working["_image_value"], working.get("assigned_structure"))
        sign = _association_sign(associations, image_feature=str(image_feature), target=target)
        if not np.isfinite(sign) or sign == 0.0:
            sign = 1.0
        working["_concordance"] = sign * working["_target_z"] * working["_image_z"]
        delta = _positive(float(target_row.get("delta_r2_combined_over_structure", np.nan)))
        working["_hidden_score"] = working["_concordance"].clip(lower=0.0) * (1.0 + delta)
        candidates = working.sort_values(
            ["_hidden_score", "_concordance", "contour_id"],
            ascending=[False, False, True],
            kind="stable",
        ).head(max_contours_per_target)
        for _, row in candidates.iterrows():
            rows.append(
                {
                    "target_feature": target,
                    "image_feature": str(image_feature),
                    "contour_id": str(row.get("contour_id")),
                    "assigned_structure": row.get("assigned_structure"),
                    "target_value": float(row["_target_value"]),
                    "image_value": float(row["_image_value"]),
                    "target_z_within_structure": float(row["_target_z"]),
                    "image_z_within_structure": float(row["_image_z"]),
                    "concordance_score": float(row["_concordance"]),
                    "hidden_program_score": float(row["_hidden_score"]),
                    "centroid_x": row.get("centroid_x", np.nan),
                    "centroid_y": row.get("centroid_y", np.nan),
                    "n_tiles": row.get("n_tiles", np.nan),
                    "n_cells": row.get("n_cells", np.nan),
                    "recommended_figure_use": _recommended_figure_use(target, image_feature=str(image_feature)),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["hidden_program_score", "target_feature", "contour_id"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    return result


def _build_morphomolecular_concept_tests(associations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "concept",
        "image_feature",
        "molecular_feature",
        "partial_spearman_rho",
        "abs_partial_spearman_rho",
        "fdr",
        "n_contours",
        "interpretation",
    ]
    if associations.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in associations.iterrows():
        image_feature = str(row.get("image_feature", ""))
        molecular_feature = str(row.get("molecular_feature", ""))
        concept = _concept_label(image_feature, molecular_feature)
        if concept is None:
            continue
        rows.append(
            {
                "concept": concept,
                "image_feature": image_feature,
                "molecular_feature": molecular_feature,
                "partial_spearman_rho": row.get("partial_spearman_rho", np.nan),
                "abs_partial_spearman_rho": row.get("abs_partial_spearman_rho", np.nan),
                "fdr": row.get("fdr", np.nan),
                "n_contours": row.get("n_contours", np.nan),
                "interpretation": _concept_interpretation(concept),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["concept", "abs_partial_spearman_rho", "image_feature"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)
    return result


def _build_boundary_coupling_summary(associations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "image_feature",
        "molecular_feature",
        "c_mm",
        "abs_c_mm",
        "fdr",
        "n_contours",
        "maz_priority_score",
        "boundary_storyline",
    ]
    if associations.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in associations.iterrows():
        image_feature = str(row.get("image_feature", ""))
        molecular_feature = str(row.get("molecular_feature", ""))
        if not (_is_boundary_like(image_feature) or _is_boundary_like(molecular_feature)):
            continue
        c_mm = float(row.get("partial_spearman_rho", np.nan))
        if not np.isfinite(c_mm):
            continue
        fdr = float(row.get("fdr", np.nan))
        rows.append(
            {
                "image_feature": image_feature,
                "molecular_feature": molecular_feature,
                "c_mm": c_mm,
                "abs_c_mm": abs(c_mm),
                "fdr": fdr,
                "n_contours": row.get("n_contours", np.nan),
                "maz_priority_score": abs(c_mm) * (1.0 if not np.isfinite(fdr) else max(0.0, 1.0 - fdr)),
                "boundary_storyline": _boundary_storyline(image_feature, molecular_feature),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["maz_priority_score", "image_feature"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
    return result


def _empty_contour_image_molecular_associations_columns() -> list[str]:
    return [
        "association_kind",
        "image_feature",
        "molecular_feature",
        "spearman_rho",
        "partial_spearman_rho",
        "abs_partial_spearman_rho",
        "p_value",
        "fdr",
        "n_contours",
        "controls",
    ]


def _positive(value: float) -> float:
    return float(value) if np.isfinite(value) and value > 0 else 0.0


def _target_storyline(target: str) -> str:
    text = target.lower()
    if any(
        token in text
        for token in (
            "epithelial",
            "luminal",
            "her2",
            "basal",
            "epcam",
            "krt",
            "erbb2",
            "esr1",
            "pgr",
            "tumor",
        )
    ):
        return "epithelial tumor state"
    if any(token in text for token in ("hypoxia", "necrosis", "ca9", "slc2a1")):
        return "metabolic stress / necrotic-hypoxic niche"
    if any(token in text for token in ("immune", "t_cell", "cd3", "trac", "cxcl13", "exhaust")):
        return "immune infiltration or exclusion"
    if any(token in text for token in ("vascular", "endothelial", "pecam", "vwf", "kdr", "vegf")):
        return "vascular or perivascular niche"
    if any(
        token in text
        for token in (
            "emt",
            "invasion",
            "stromal",
            "matrix",
            "collagen",
            "ecm",
            "caf",
            "tgf",
            "myofibroblast",
            "col1",
            "dcn",
            "rna__lum__mean",
            "fap",
        )
    ):
        return "stromal remodeling / invasive-front biology"
    return "hidden within-structure molecular program"


def _fallback_image_feature_for_target(
    target: str,
    contour_summary: pd.DataFrame,
    associations: pd.DataFrame,
) -> str | None:
    image_columns = _contour_image_feature_columns(contour_summary)
    if not image_columns:
        return None
    if not associations.empty and "image_feature" in associations.columns:
        linked = associations[associations["molecular_feature"].astype(str) == target]
        for image_feature in linked.get("image_feature", pd.Series(dtype=object)).astype(str):
            if image_feature in contour_summary.columns:
                return image_feature
    priority_tokens = []
    story = _target_storyline(target)
    if "hypoxic" in story or "metabolic" in story:
        priority_tokens = ["necrosis", "heterogeneity", "entropy"]
    elif "immune" in story:
        priority_tokens = ["immune", "fibrotic", "entropy"]
    elif "stromal" in story:
        priority_tokens = ["fibrotic", "carcinoma", "heterogeneity"]
    for token in priority_tokens:
        match = [column for column in image_columns if token in column.lower()]
        if match:
            return match[0]
    return image_columns[0]


def _within_group_zscore(values: pd.Series, groups: pd.Series | None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if groups is None:
        return _zscore(numeric)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    for _, index in groups.fillna("unassigned").astype(str).groupby(groups.fillna("unassigned").astype(str)).groups.items():
        output.loc[index] = _zscore(numeric.loc[index])
    return output.fillna(0.0)


def _zscore(values: pd.Series) -> pd.Series:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - mean) / std


def _association_sign(associations: pd.DataFrame, *, image_feature: str, target: str) -> float:
    if associations.empty:
        return np.nan
    mask = (
        associations.get("image_feature", pd.Series(dtype=object)).astype(str).eq(str(image_feature))
        & associations.get("molecular_feature", pd.Series(dtype=object)).astype(str).eq(str(target))
    )
    linked = associations.loc[mask]
    if linked.empty:
        return np.nan
    value = float(linked.iloc[0].get("partial_spearman_rho", np.nan))
    if not np.isfinite(value):
        return np.nan
    return float(np.sign(value))


def _recommended_figure_use(target: str, *, image_feature: str) -> str:
    story = _target_storyline(target)
    if "hypoxic" in story or "metabolic" in story:
        return "Figure 2 hero contour: H&E-predicted metabolic stress"
    if "immune" in story:
        return "Figure 3 boundary/immune-exclusion candidate"
    if "stromal" in story:
        return "Figure 3 invasive-front or stromal remodeling candidate"
    if "heterogeneity" in image_feature or "entropy" in image_feature:
        return "Figure 2 hidden heterogeneity case"
    return "Extended Data morphomolecular validation case"


def _concept_label(image_feature: str, molecular_feature: str) -> str | None:
    image = image_feature.lower()
    molecular = molecular_feature.lower()
    if ("entropy" in image or "heterogeneity" in image) and (
        "cell_type_diversity" in molecular or "fraction" in molecular
    ):
        return "morphology_entropy_vs_ecology_complexity"
    if "necrosis" in image and any(token in molecular for token in ("necrosis", "hypoxia", "ca9", "slc2a1")):
        return "prompt_axis_grounding_necrosis_hypoxia"
    if "immune" in image and any(token in molecular for token in ("immune", "t_cell", "cd3", "trac", "cxcl13")):
        return "prompt_axis_grounding_immune"
    if "fibrotic" in image and any(token in molecular for token in ("stromal", "matrix", "col1", "dcn", "lum", "fap")):
        return "prompt_axis_grounding_stroma"
    if "embedding__" in image and any(token in molecular for token in ("program__", "rna__", "cell_type_fraction__")):
        return "foundation_embedding_hidden_program"
    return None


def _concept_interpretation(concept: str) -> str:
    return {
        "morphology_entropy_vs_ecology_complexity": (
            "Tests whether H&E texture diversity tracks Xenium ecological diversity within contours."
        ),
        "prompt_axis_grounding_necrosis_hypoxia": (
            "Tests whether necrosis-like image-language scores align with hypoxia/necrosis molecular programs."
        ),
        "prompt_axis_grounding_immune": (
            "Tests whether immune-like image-language scores align with immune cell or activation programs."
        ),
        "prompt_axis_grounding_stroma": (
            "Tests whether fibrotic/stromal image-language scores align with ECM or CAF-like molecular programs."
        ),
        "foundation_embedding_hidden_program": (
            "Tests prompt-independent foundation-model axes for hidden molecular programs."
        ),
    }.get(concept, "Morphomolecular validation concept.")


def _program_feature_label(feature: str, *, prefix: str) -> str:
    text = str(feature)
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith("__mean"):
        text = text[: -len("__mean")]
    return text


def _image_axis_family(feature: str) -> str:
    text = str(feature)
    if text.startswith(_EMBEDDING_PREFIX):
        return "foundation_embedding_axis"
    if text.startswith("relative_prompt_axis__"):
        return "relative_prompt_axis"
    if text.startswith(_TEXT_PREFIX):
        return "image_language_prompt_score"
    if text.startswith("image_heterogeneity__") or text.startswith("morphology_entropy__"):
        return "morphology_heterogeneity"
    if text.startswith("domain_fraction__"):
        return "tile_domain_fraction"
    return "image_feature"


def _wta_pathway_interpretation(pathway: str, image_feature: str) -> str:
    text = pathway.lower()
    axis = _image_axis_family(image_feature)
    if any(token in text for token in ("hypoxia", "glycolysis", "oxidative")):
        return f"Residual metabolic-state gradient linked to {axis} after structure and spatial controls."
    if any(token in text for token in ("emt", "collagen", "caf", "tgf", "matrix")):
        return f"Residual stromal-remodeling or invasion program linked to {axis}."
    if any(token in text for token in ("t_cell", "immune", "interferon", "chemokine", "antigen", "tls")):
        return f"Residual immune-ecology program linked to {axis} within matched structures."
    if any(token in text for token in ("epithelial", "luminal", "her2", "basal")):
        return f"Residual epithelial tumor-state program linked to {axis} beyond discrete HistoSeg labels."
    if any(token in text for token in ("proliferation", "p53", "stress", "protein")):
        return f"Residual cell-state stress or proliferation program linked to {axis}."
    return f"Residual WTA pathway activity linked to {axis} after coarse structure is fixed."


def _is_boundary_like(feature: str) -> bool:
    text = feature.lower()
    return any(
        token in text
        for token in (
            "boundary",
            "edge",
            "outer_minus_inner",
            "inner_minus_outer",
            "gradient__",
            "rim",
            "cci__",
            "infiltration",
            "sharpness",
        )
    )


def _boundary_storyline(image_feature: str, molecular_feature: str) -> str:
    molecular = molecular_feature.lower()
    if any(token in molecular for token in ("cci__", "ligand", "receptor")):
        return "molecularly active zone with candidate cell-cell communication"
    if any(token in molecular for token in ("immune", "t_cell", "cxcl13", "trac")):
        return "immune boundary infiltration or exclusion"
    if any(token in molecular for token in ("emt", "invasion", "stromal", "matrix")):
        return "invasive-front molecular transition"
    if any(token in molecular for token in ("hypoxia", "necrosis")):
        return "necrotic-hypoxic boundary transition"
    return "morphology-molecular boundary coupling"


def _base_contour_summary(image_contours: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, contour in image_contours.iterrows():
        geometry = contour.get("geometry")
        area = float(geometry.area) if isinstance(geometry, BaseGeometry) else np.nan
        perimeter = float(geometry.length) if isinstance(geometry, BaseGeometry) else np.nan
        centroid = geometry.centroid if isinstance(geometry, BaseGeometry) and not geometry.is_empty else None
        area_value = contour.get("area_image_px2", area)
        centroid_x = contour.get("centroid_x", centroid.x if centroid is not None else np.nan)
        centroid_y = contour.get("centroid_y", centroid.y if centroid is not None else np.nan)
        row = {
            "contour_id": str(contour.get("contour_id", "")),
            "assigned_structure": _structure_label(contour),
            "structure_id": _optional_text(contour.get("structure_id")),
            "classification_name": _optional_text(contour.get("classification_name")),
            "area_image_px2": float(area if pd.isna(area_value) else area_value),
            "perimeter_image_px": perimeter,
            "centroid_x": float(np.nan if pd.isna(centroid_x) else centroid_x),
            "centroid_y": float(np.nan if pd.isna(centroid_y) else centroid_y),
        }
        if np.isfinite(row["area_image_px2"]) and np.isfinite(perimeter) and perimeter > 0:
            row["shape_compactness"] = float(4.0 * np.pi * row["area_image_px2"] / (perimeter * perimeter))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_contour_image_features(
    tile_features: pd.DataFrame,
    *,
    text_terms: Sequence[str] | None,
    relative_prompt_axes: Sequence[Sequence[str]] | None,
) -> pd.DataFrame:
    columns = [
        "contour_id",
        "n_tiles",
        "top_prompt_term",
        "top_prompt_term_fraction",
        "top_image_label",
        "top_image_label_fraction",
        "morphology_entropy__top_image_label",
        "image_heterogeneity__embedding_cosine_similarity_mean",
        "image_heterogeneity__embedding_cosine_similarity_variance",
        "image_heterogeneity__embedding_cosine_distance_mean",
        "image_heterogeneity__embedding_centroid_distance_mean",
        "image_heterogeneity__embedding_centroid_distance_std",
    ]
    if tile_features.empty or "contour_id" not in tile_features.columns:
        return pd.DataFrame(columns=columns)
    assigned = tile_features[tile_features["contour_id"].notna()].copy()
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    assigned["contour_id"] = assigned["contour_id"].astype(str)
    numeric_columns = _image_numeric_columns(assigned)
    embedding_columns = [column for column in numeric_columns if column.startswith(_EMBEDDING_PREFIX)]
    axes = _normalize_relative_prompt_axes(relative_prompt_axes)
    text_terms = tuple(text_terms or DEFAULT_LAZYSLIDE_TEXT_TERMS)

    rows = []
    for contour_id, group in assigned.groupby("contour_id", sort=True, dropna=False):
        row: dict[str, Any] = {
            "contour_id": str(contour_id),
            "n_tiles": int(len(group)),
            "assigned_structure": _mode_or_none(group.get("assigned_structure")),
            "structure_id": _mode_or_none(group.get("structure_id")),
            "classification_name": _mode_or_none(group.get("classification_name")),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}__std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        prompt_column = "top_prompt_term" if "top_prompt_term" in group.columns else "top_image_label"
        if prompt_column in group.columns:
            labels = group[prompt_column].dropna().astype(str)
            label_counts = labels.value_counts()
            if not label_counts.empty:
                row["top_prompt_term"] = str(label_counts.index[0])
                row["top_prompt_term_fraction"] = float(label_counts.iloc[0] / len(group))
                row["top_image_label"] = row["top_prompt_term"]
                row["top_image_label_fraction"] = row["top_prompt_term_fraction"]
                row["morphology_entropy__top_image_label"] = _shannon_entropy(label_counts)
        if "spatial_domain" in group.columns:
            domain_counts = group["spatial_domain"].dropna().astype(str).value_counts()
            row["morphology_entropy__spatial_domain"] = _shannon_entropy(domain_counts)
            for domain, count in domain_counts.items():
                row[f"domain_fraction__{_slug(domain)}"] = float(count / len(group))
        if embedding_columns:
            row.update(
                _embedding_heterogeneity(
                    group.loc[:, embedding_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                )
            )
        for name, positive, negative in axes:
            pos_col = f"{_TEXT_PREFIX}{_slug(positive)}"
            neg_col = f"{_TEXT_PREFIX}{_slug(negative)}"
            if pos_col in group.columns and neg_col in group.columns:
                pos = pd.to_numeric(group[pos_col], errors="coerce")
                neg = pd.to_numeric(group[neg_col], errors="coerce")
                row[f"relative_prompt_axis__{_slug(name)}"] = float((pos - neg).mean())
        for term in text_terms:
            column = f"{_TEXT_PREFIX}{_slug(term)}"
            if column in group.columns:
                row[f"{column}__rank"] = _mean_rank(group[column])
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _merge_contour_summary(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right.empty or "contour_id" not in right.columns:
        return left
    clean = pd.DataFrame(right).copy()
    clean["contour_id"] = clean["contour_id"].astype(str)
    drop_columns = [
        column
        for column in ("assigned_structure", "structure_id", "classification_name")
        if column in clean.columns and column in left.columns
    ]
    clean = clean.drop(columns=drop_columns, errors="ignore")
    merged = left.copy()
    merged["contour_id"] = merged["contour_id"].astype(str)
    return merged.merge(clean, on="contour_id", how="left")


def _prefix_contour_program_scores(program_scores: pd.DataFrame) -> pd.DataFrame:
    if program_scores.empty or "contour_id" not in program_scores.columns:
        return pd.DataFrame(columns=["contour_id"])
    output = pd.DataFrame(program_scores).copy()
    output["contour_id"] = output["contour_id"].astype(str)
    rename: dict[str, str] = {}
    for column in _numeric_columns(output, exclude={"contour_id"}):
        if not str(column).startswith("program__"):
            rename[column] = f"program__{column}"
    return output.rename(columns=rename)


def _add_density_features(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    if {"n_cells", "area_image_px2"}.issubset(output.columns):
        area = pd.to_numeric(output["area_image_px2"], errors="coerce")
        cells = pd.to_numeric(output["n_cells"], errors="coerce")
        output["cell_density_per_1e6_image_px2"] = np.where(
            area > 0,
            cells / area * 1_000_000.0,
            np.nan,
        )
    if {"n_tiles", "area_image_px2"}.issubset(output.columns):
        area = pd.to_numeric(output["area_image_px2"], errors="coerce")
        tiles = pd.to_numeric(output["n_tiles"], errors="coerce")
        output["tile_density_per_1e6_image_px2"] = np.where(
            area > 0,
            tiles / area * 1_000_000.0,
            np.nan,
        )
    return output


def _add_tile_boundary_distance_features(
    summary: pd.DataFrame,
    image_contours: pd.DataFrame,
    tile_features: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty or image_contours.empty or tile_features.empty or "contour_id" not in tile_features.columns:
        return summary
    assigned = tile_features[tile_features["contour_id"].notna()].copy()
    if assigned.empty:
        return summary
    assigned["contour_id"] = assigned["contour_id"].astype(str)
    rows = []
    for _, contour in image_contours.iterrows():
        geometry = contour.get("geometry")
        contour_id = str(contour.get("contour_id", ""))
        if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
            continue
        group = assigned[assigned["contour_id"].astype(str) == contour_id]
        if group.empty:
            continue
        distances = []
        for _, tile in group.iterrows():
            try:
                point = _tile_centroid(tile)
            except ValueError:
                continue
            distances.append(float(geometry.boundary.distance(point)))
        if not distances:
            continue
        values = np.asarray(distances, dtype=float)
        area = float(geometry.area)
        scale = np.sqrt(area) if area > 0 else np.nan
        row = {
            "contour_id": contour_id,
            "tile_boundary_distance_px__mean": float(np.mean(values)),
            "tile_boundary_distance_px__median": float(np.median(values)),
            "tile_boundary_distance_px__min": float(np.min(values)),
            "tile_boundary_distance_px__p10": float(np.quantile(values, 0.1)),
            "tile_boundary_distance_px__std": float(np.std(values)),
            "tile_edge_proximity_fraction__lt_256px": float(np.mean(values <= 256.0)),
        }
        if np.isfinite(scale) and scale > 0:
            row["tile_boundary_distance_normalized__mean"] = float(np.mean(values / scale))
        rows.append(row)
    if not rows:
        return summary
    return _merge_contour_summary(summary, pd.DataFrame(rows))


def _embedding_heterogeneity(matrix: np.ndarray) -> dict[str, float]:
    output = {
        "image_heterogeneity__embedding_cosine_similarity_mean": np.nan,
        "image_heterogeneity__embedding_cosine_similarity_variance": np.nan,
        "image_heterogeneity__embedding_cosine_distance_mean": np.nan,
        "image_heterogeneity__embedding_centroid_distance_mean": np.nan,
        "image_heterogeneity__embedding_centroid_distance_std": np.nan,
    }
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return output
    finite = matrix[np.isfinite(matrix).all(axis=1)]
    if finite.shape[0] < 2:
        return output
    if finite.shape[0] > 256:
        finite = finite[np.linspace(0, finite.shape[0] - 1, 256).astype(int)]
    norms = np.linalg.norm(finite, axis=1)
    finite = finite[norms > 0]
    norms = norms[norms > 0]
    if finite.shape[0] < 2:
        return output
    normalized = finite / norms[:, None]
    similarity = normalized @ normalized.T
    pairwise = similarity[np.triu_indices(similarity.shape[0], k=1)]
    centroid = normalized.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 0:
        centroid_distances = 1.0 - normalized @ (centroid / centroid_norm)
    else:
        centroid_distances = np.full(normalized.shape[0], np.nan)
    output.update(
        {
            "image_heterogeneity__embedding_cosine_similarity_mean": float(np.mean(pairwise)),
            "image_heterogeneity__embedding_cosine_similarity_variance": float(np.var(pairwise)),
            "image_heterogeneity__embedding_cosine_distance_mean": float(np.mean(1.0 - pairwise)),
            "image_heterogeneity__embedding_centroid_distance_mean": float(np.nanmean(centroid_distances)),
            "image_heterogeneity__embedding_centroid_distance_std": float(np.nanstd(centroid_distances)),
        }
    )
    return output


def _shannon_entropy(counts: pd.Series) -> float:
    values = pd.to_numeric(counts, errors="coerce").dropna().to_numpy(dtype=float)
    total = float(values.sum())
    if total <= 0:
        return np.nan
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


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


def _not_wsi_ready_error(diagnostics: Mapping[str, Any], *, image_key: str) -> ValueError:
    issues = diagnostics.get("issues", []) or []
    issue_text = "; ".join(str(item) for item in issues if item)
    if not issue_text:
        issue_text = "Missing required WSI metadata."
    return ValueError(
        f"`slide.images[{image_key!r}]` is not WSI-ready: {issue_text}. "
        "Either rebuild the slide store with native H&E pyramid metadata or pass "
        "`he_source_path` explicitly to use an external WSI file override."
    )


def _open_lazyslide_wsi(
    sdata: XeniumSlide,
    *,
    he_image: XeniumImage,
    config: HistoSegLazySlideConfig,
    open_wsi_func: Any,
) -> tuple[Any, dict[str, Any]]:
    if config.he_source_path is not None:
        source_path = str(Path(config.he_source_path).expanduser())
        if config.wsi_reader:
            wsi = open_wsi_func(source_path, reader=config.wsi_reader)
        else:
            wsi = open_wsi_func(source_path)
        if config.slide_mpp is not None and hasattr(wsi, "set_mpp"):
            wsi.set_mpp(float(config.slide_mpp))
        return wsi, {"wsi_source": "external_override", "path": source_path}

    diagnostics = sdata.inspect_wsi(config.he_image_key)
    if not diagnostics.get("wsi_ready"):
        raise _not_wsi_ready_error(diagnostics, image_key=config.he_image_key)

    wsi = sdata.to_wsidata(image_key=config.he_image_key)
    if config.slide_mpp is not None and hasattr(wsi, "set_mpp"):
        wsi.set_mpp(float(config.slide_mpp))
    return (
        wsi,
        {
            "wsi_source": "internal_slide_store",
            "path": str(sdata.metadata.get("slide_store_path") or he_image.source_path),
        },
    )


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
        sdata=sdata,
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
    sdata: XeniumSlide,
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

    annotation_key = "histoseg_structures"
    tile_key = "histoseg_tiles"
    feature_key = f"{_slug(config.model)}_{tile_key}"
    status: dict[str, Any] = {
        "backend": "lazyslide",
        "model": config.model,
        "embedding_model": config.model,
        "text_model": config.text_model,
        "status": "started",
        "lazy_module": getattr(zs, "__version__", "unknown"),
    }

    try:
        wsi, source_info = _open_lazyslide_wsi(
            sdata,
            he_image=he_image,
            config=config,
            open_wsi_func=open_wsi,
        )
        status.update(source_info)
        annotation_frame = image_contours.drop(columns=["geometry"]).copy()
        annotation_frame["tissue_id"] = annotation_frame["contour_id"].astype(str)
        annotations = gpd.GeoDataFrame(
            annotation_frame,
            geometry=image_contours["geometry"].to_list(),
            crs=None,
        )
        _load_lazyslide_annotations(zs, wsi, annotations, key_added=annotation_key)
        tile_kwargs = {
            "mpp": float(config.mpp),
            "tissue_key": annotation_key,
            "key_added": tile_key,
        }
        if config.slide_mpp is not None:
            tile_kwargs["slide_mpp"] = float(config.slide_mpp)
        zs.pp.tile_tissues(wsi, int(config.tile_px), **tile_kwargs)
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
        text_status = _try_lazyslide_text_similarity(
            zs,
            wsi,
            embedding_model=config.model,
            text_model=config.text_model,
            feature_key=feature_key,
            tile_key=tile_key,
            text_terms=config.text_terms,
        )
        status["text_similarity"] = text_status
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
    embedding_model: str,
    text_model: str | None,
    feature_key: str,
    tile_key: str,
    text_terms: Sequence[str],
) -> dict[str, Any]:
    resolved_text_model, skipped_reason = _resolve_tile_text_model(
        embedding_model=embedding_model,
        text_model=text_model,
        text_terms=text_terms,
    )
    if resolved_text_model is None:
        return {
            "status": "skipped",
            "embedding_model": embedding_model,
            "text_model": text_model,
            "skipped_reason": skipped_reason,
        }
    try:
        text_embeddings = zs.tl.text_embedding(list(text_terms), model=resolved_text_model)
        try:
            zs.tl.text_image_similarity(
                wsi,
                text_embeddings,
                model=resolved_text_model,
                tile_key=tile_key,
                softmax=True,
                feature_key=feature_key,
            )
        except TypeError:
            zs.tl.text_image_similarity(
                wsi,
                text_embeddings,
                model=resolved_text_model,
                tile_key=tile_key,
                softmax=True,
            )
        return {
            "status": "computed",
            "embedding_model": embedding_model,
            "text_model": resolved_text_model,
            "n_text_terms": int(len(text_terms)),
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "embedding_model": embedding_model,
            "text_model": resolved_text_model,
            "skipped_reason": str(exc),
        }


def _resolve_tile_text_model(
    *,
    embedding_model: str,
    text_model: str | None,
    text_terms: Sequence[str],
) -> tuple[str | None, str | None]:
    if not text_terms:
        return None, "No text_terms were provided."
    embedding_slug = _slug(embedding_model)
    requested = None if text_model is None else str(text_model).strip()
    if requested is not None and (requested == "" or _slug(requested) in {"none", "skip", "false", "0"}):
        return None, "Text similarity was disabled by text_model."
    resolved = requested or (embedding_model if embedding_slug in _LAZYSLIDE_TILE_TEXT_MODELS else None)
    if resolved is None:
        return (
            None,
            (
                f"Embedding model {embedding_model!r} is treated as a vision-only "
                "foundation model for tile-level pyXenium summaries."
            ),
        )
    resolved_slug = _slug(resolved)
    if resolved_slug not in _LAZYSLIDE_TILE_TEXT_MODELS:
        return (
            None,
            (
                f"Text model {resolved!r} is not configured for LazySlide tile-level "
                "text-image similarity."
            ),
        )
    if resolved_slug != embedding_slug:
        return (
            None,
            (
                "Tile-level text-image similarity requires the text model and image "
                f"embedding model to share one latent space; got text_model={resolved!r} "
                f"and model={embedding_model!r}."
            ),
        )
    return resolved, None


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
    if text_columns and "top_prompt_term" not in output.columns:
        values = output.loc[:, text_columns].apply(pd.to_numeric, errors="coerce")
        top_columns = values.idxmax(axis=1)
        output["top_prompt_term"] = [
            column.removeprefix(_TEXT_PREFIX).replace("_", " ")
            if isinstance(column, str)
            else None
            for column in top_columns
        ]
        output["top_prompt_similarity"] = values.max(axis=1).to_numpy(dtype=float)
    if "top_prompt_term" in output.columns and "top_image_label" not in output.columns:
        output["top_image_label"] = output["top_prompt_term"]
    if "top_prompt_similarity" in output.columns and "top_image_label_score" not in output.columns:
        output["top_image_label_score"] = output["top_prompt_similarity"]
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


def _build_contour_rna_summary(
    *,
    sdata: XeniumSlide,
    contour_table: pd.DataFrame,
    genes: Sequence[str],
) -> pd.DataFrame:
    adata = sdata.table
    if "spatial" not in adata.obsm or adata.n_obs == 0:
        return _empty_contour_rna_summary()
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return _empty_contour_rna_summary()
    matrix = _expression_frame(adata, genes)
    cell_points = [Point(float(x), float(y)) for x, y in coords[:, :2]]
    point_tree = STRtree(cell_points) if cell_points else None

    rows = []
    for _, contour in contour_table.iterrows():
        geometry = contour["geometry"]
        cell_indices = _cell_indices_within_geometry(
            geometry,
            cell_points=cell_points,
            point_tree=point_tree,
        )
        row = {
            "contour_id": str(contour["contour_id"]),
            "assigned_structure": _structure_label(contour),
            "structure_id": _optional_text(contour.get("structure_id")),
            "n_cells": int(len(cell_indices)),
        }
        if len(cell_indices):
            distances = np.asarray(
                [float(geometry.boundary.distance(cell_points[int(index)])) for index in cell_indices],
                dtype=float,
            )
            if distances.size:
                row["cell_boundary_distance_um__mean"] = float(np.mean(distances))
                row["cell_boundary_distance_um__median"] = float(np.median(distances))
                row["cell_boundary_distance_um__min"] = float(np.min(distances))
                row["cell_boundary_distance_um__p10"] = float(np.quantile(distances, 0.1))
                row["cell_boundary_distance_um__std"] = float(np.std(distances))
                row["cell_edge_proximity_fraction__lt_25um"] = float(np.mean(distances <= 25.0))
        if len(cell_indices) and not matrix.empty:
            means = matrix.iloc[cell_indices, :].mean(axis=0)
            for gene, value in means.items():
                row[f"rna__{gene}__mean"] = float(value)
        for column in ("cell_type", "cluster", "cell_state"):
            if column in adata.obs.columns and len(cell_indices):
                values = adata.obs.iloc[cell_indices][column].astype(str).value_counts()
                row[f"cell_type_diversity__{_slug(column)}__shannon"] = _shannon_entropy(values)
                proportions = values.to_numpy(dtype=float) / float(values.sum())
                row[f"cell_type_diversity__{_slug(column)}__simpson"] = float(
                    1.0 - np.sum(proportions * proportions)
                )
                for label, count in values.items():
                    row[f"cell_type_fraction__{_slug(column)}__{_slug(label)}"] = float(
                        count / len(cell_indices)
                    )
        rows.append(row)
    contour_rna = pd.DataFrame(rows)
    if contour_rna.empty:
        return _empty_contour_rna_summary()
    return contour_rna


def _aggregate_contour_rna_by_structure(contour_rna: pd.DataFrame) -> pd.DataFrame:
    if contour_rna.empty:
        return _empty_structure_rna_summary()
    if "assigned_structure" not in contour_rna.columns:
        return _empty_structure_rna_summary()
    numeric_columns = _numeric_columns(contour_rna, exclude=set(_ID_COLUMNS) | {"n_cells"})
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


def _build_structure_rna_summary(
    *,
    sdata: XeniumSlide,
    contour_table: pd.DataFrame,
    genes: Sequence[str],
) -> pd.DataFrame:
    return _aggregate_contour_rna_by_structure(
        _build_contour_rna_summary(sdata=sdata, contour_table=contour_table, genes=genes)
    )


def _cell_indices_within_geometry(
    geometry: BaseGeometry,
    *,
    cell_points: Sequence[Point],
    point_tree: STRtree | None,
) -> np.ndarray:
    if geometry is None or geometry.is_empty or not cell_points:
        return np.asarray([], dtype=int)
    candidate_indices: np.ndarray
    if point_tree is None:
        candidate_indices = np.arange(len(cell_points), dtype=int)
    else:
        try:
            candidate_indices = np.asarray(point_tree.query(geometry, predicate="covers"), dtype=int)
            return np.sort(candidate_indices)
        except Exception:
            try:
                candidate_indices = np.asarray(point_tree.query(geometry), dtype=int)
            except Exception:
                candidate_indices = np.arange(len(cell_points), dtype=int)
    if candidate_indices.size == 0:
        return np.asarray([], dtype=int)
    inside = [
        int(index)
        for index in candidate_indices
        if geometry.covers(cell_points[int(index)])
    ]
    return np.asarray(sorted(inside), dtype=int)


def _expression_frame(adata: ad.AnnData, genes: Sequence[str]) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame(index=adata.obs_names)
    resolved = _resolve_gene_indices(adata, genes)
    if not resolved:
        return pd.DataFrame(index=adata.obs_names)
    indices = [index for _, index in resolved]
    labels = [label for label, _ in resolved]
    values = adata.X[:, indices]
    if sparse.issparse(values):
        values = values.toarray()
    return pd.DataFrame(np.asarray(values, dtype=float), columns=labels, index=adata.obs_names)


def _resolve_gene_indices(adata: ad.AnnData, genes: Sequence[str]) -> list[tuple[str, int]]:
    lookup: dict[str, int] = {}
    for index, value in enumerate(adata.var_names.astype(str)):
        lookup.setdefault(str(value).upper(), int(index))
    for column in ("name", "gene_name", "gene_symbol", "symbol", "id"):
        if column not in adata.var.columns:
            continue
        for index, value in enumerate(adata.var[column].astype(str)):
            lookup.setdefault(str(value).upper(), int(index))
    resolved = []
    seen_indices: set[int] = set()
    for gene in genes:
        index = lookup.get(str(gene).upper())
        if index is None or index in seen_indices:
            continue
        resolved.append((str(gene), int(index)))
        seen_indices.add(int(index))
    return resolved


def _resolve_program_scores(
    *,
    sdata: XeniumSlide,
    contour_table: pd.DataFrame,
    contour_key: str,
    precomputed_feature_table: Mapping[str, Any] | None,
    precomputed_program_scores: pd.DataFrame | str | Path | None,
    wta_program_library: str,
    program_library: str,
    include_wta_programs: bool,
    include_boundary_programs: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if precomputed_program_scores is not None:
        return _read_table(precomputed_program_scores), {"status": "loaded_precomputed"}
    frames: list[pd.DataFrame] = []
    status: dict[str, Any] = {
        "status": "computed" if include_wta_programs or include_boundary_programs else "disabled",
        "wta_programs": {"status": "disabled"},
        "boundary_programs": {"status": "disabled"},
    }
    if include_wta_programs:
        try:
            wta_scores, wta_status = _build_contour_wta_program_scores(
                sdata=sdata,
                contour_table=contour_table,
                program_library=wta_program_library,
            )
            if not wta_scores.empty:
                frames.append(wta_scores)
            status["wta_programs"] = wta_status
        except Exception as exc:
            status["wta_programs"] = {"status": "skipped", "reason": str(exc)}
    if include_boundary_programs:
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
            boundary_scores = pd.DataFrame(result["program_scores"]).copy()
            if not boundary_scores.empty:
                frames.append(boundary_scores)
            status["boundary_programs"] = {"status": "computed"}
        except Exception as exc:
            status["boundary_programs"] = {"status": "skipped", "reason": str(exc)}
    merged = _merge_program_score_frames(frames)
    if merged.empty and status["status"] == "computed":
        status["status"] = "skipped"
    return merged, status


def _merge_program_score_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    clean = [pd.DataFrame(frame).copy() for frame in frames if frame is not None and not frame.empty]
    clean = [frame for frame in clean if "contour_id" in frame.columns]
    if not clean:
        return pd.DataFrame(columns=["contour_id"])
    merged = clean[0].copy()
    merged["contour_id"] = merged["contour_id"].astype(str)
    for frame in clean[1:]:
        frame = frame.copy()
        frame["contour_id"] = frame["contour_id"].astype(str)
        duplicate_columns = [
            column for column in frame.columns if column != "contour_id" and column in merged.columns
        ]
        frame = frame.drop(columns=duplicate_columns, errors="ignore")
        merged = merged.merge(frame, on="contour_id", how="outer")
    return merged


def _build_contour_wta_program_scores(
    *,
    sdata: XeniumSlide,
    contour_table: pd.DataFrame,
    program_library: str,
    min_genes_per_program: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adata = sdata.table
    programs = _load_wta_gene_program_library(program_library)
    if not programs:
        return pd.DataFrame(columns=["contour_id"]), {"status": "skipped", "reason": "empty program library"}
    if "spatial" not in adata.obsm or adata.n_obs == 0:
        return pd.DataFrame(columns=["contour_id"]), {"status": "skipped", "reason": "missing spatial cells"}
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return pd.DataFrame(columns=["contour_id"]), {"status": "skipped", "reason": "invalid spatial cells"}

    requested_genes = _unique_ordered(
        gene for genes in programs.values() for gene in genes
    )
    expression = _normalized_expression_frame(adata, requested_genes)
    if expression.empty:
        return pd.DataFrame(columns=["contour_id"]), {"status": "skipped", "reason": "no program genes found"}

    cell_points = [Point(float(x), float(y)) for x, y in coords[:, :2]]
    point_tree = STRtree(cell_points) if cell_points else None
    rows: list[dict[str, Any]] = []
    for _, contour in contour_table.iterrows():
        geometry = contour["geometry"]
        cell_indices = _cell_indices_within_geometry(
            geometry,
            cell_points=cell_points,
            point_tree=point_tree,
        )
        row: dict[str, Any] = {"contour_id": str(contour["contour_id"])}
        if len(cell_indices):
            means = expression.iloc[cell_indices, :].mean(axis=0)
            row.update({str(gene): float(value) for gene, value in means.items()})
        rows.append(row)
    gene_means = pd.DataFrame(rows).set_index("contour_id")
    if gene_means.empty:
        return pd.DataFrame(columns=["contour_id"]), {"status": "skipped", "reason": "empty contour gene means"}

    gene_z = _zscore_frame(gene_means)
    scores = pd.DataFrame({"contour_id": gene_z.index.astype(str)})
    coverage: dict[str, int] = {}
    for name, genes in programs.items():
        present = [str(gene) for gene in genes if str(gene) in gene_z.columns]
        if len(present) < int(min_genes_per_program):
            continue
        column = f"wta_{_slug(name)}"
        scores[column] = gene_z.loc[:, present].mean(axis=1).to_numpy(dtype=float)
        coverage[column] = len(present)
    program_columns = [column for column in scores.columns if column != "contour_id"]
    if not program_columns:
        return pd.DataFrame(columns=["contour_id"]), {
            "status": "skipped",
            "reason": "program coverage below threshold",
            "n_found_genes": int(expression.shape[1]),
        }
    score_values = scores.loc[:, program_columns]
    scores["top_wta_program"] = [
        str(row.idxmax()).replace("wta_", "", 1) if row.notna().any() else None
        for _, row in score_values.iterrows()
    ]
    return scores, {
        "status": "computed",
        "library": str(program_library),
        "n_programs": int(len(program_columns)),
        "n_requested_genes": int(len(requested_genes)),
        "n_found_genes": int(expression.shape[1]),
        "min_genes_per_program": int(min_genes_per_program),
        "program_gene_coverage": coverage,
    }


def _load_wta_gene_program_library(program_library: str) -> dict[str, tuple[str, ...]]:
    name = str(program_library)
    if name in {"", "default", "breast_tme_wta_v1"}:
        return dict(DEFAULT_WTA_GENE_PROGRAMS)
    path = Path(name).expanduser()
    if not path.exists():
        raise ValueError(
            "`wta_program_library` must be 'breast_tme_wta_v1' or a path to a JSON/GMT file."
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("WTA program JSON must be a mapping of program name to genes.")
        return {
            str(program): tuple(str(gene) for gene in genes)
            for program, genes in payload.items()
            if isinstance(genes, Sequence) and not isinstance(genes, str)
        }
    return _read_gmt_gene_programs(path)


def _read_gmt_gene_programs(path: Path) -> dict[str, tuple[str, ...]]:
    programs: dict[str, tuple[str, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        programs[_slug(parts[0])] = tuple(gene for gene in parts[2:] if gene)
    return programs


def _normalized_expression_frame(adata: ad.AnnData, genes: Sequence[str]) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame(index=adata.obs_names)
    resolved = _resolve_gene_indices(adata, genes)
    if not resolved:
        return pd.DataFrame(index=adata.obs_names)
    indices = [index for _, index in resolved]
    labels = [label for label, _ in resolved]
    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    if sparse.issparse(matrix):
        rna = matrix.tocsr()
        values = rna[:, indices].toarray()
        library_size = np.asarray(rna.sum(axis=1)).ravel().astype(float)
    else:
        dense = np.asarray(matrix, dtype=float)
        values = dense[:, indices]
        library_size = np.asarray(dense.sum(axis=1)).ravel().astype(float)
    library_size[library_size <= 0] = 1.0
    normalized = np.log1p((np.asarray(values, dtype=float) / library_size[:, None]) * 1e4)
    return pd.DataFrame(normalized, columns=labels, index=adata.obs_names)


def _zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0, ddof=0).replace(0.0, np.nan)
    return (numeric - means) / stds


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
            "LazySlide": "Owns WSI tiling and pathology foundation model inference.",
            "pyXenium": (
                "Owns Xenium alignment, structure-level RNA/image summaries, "
                "and interpretable multimodal associations."
            ),
        },
        "prompt_metadata": {
            "prompt_set_name": config.prompt_set_name,
            "prompt_source": config.prompt_source,
            "prompt_review_status": config.prompt_review_status,
            "text_terms": list(config.text_terms),
            "requested_text_model": config.text_model,
            "effective_text_model": dict(model_result.get("model_status", {}))
            .get("text_similarity", {})
            .get("text_model"),
        },
        "inputs": {
            "n_cells": int(sdata.table.n_obs),
            "n_features": int(sdata.table.n_vars),
            "contour_key": config.contour_key,
            "he_image_key": config.he_image_key,
            "he_source_path": sdata.images[config.he_image_key].source_path
            if config.he_image_key in sdata.images
            else None,
            "wsi_reader": config.wsi_reader,
            "slide_mpp": config.slide_mpp,
            "wsi_registry": sdata.metadata.get("wsi", {}),
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
        "contour_multimodal_summary": _write_table(
            result["contour_multimodal_summary"],
            out / f"contour_multimodal_summary.{table_format}",
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
        "contour_image_molecular_associations": _write_table(
            result["contour_image_molecular_associations"],
            out / f"contour_image_molecular_associations.{table_format}",
            table_format=table_format,
        ),
        "wta_pathway_partial_correlations": _write_table(
            result["wta_pathway_partial_correlations"],
            out / f"wta_pathway_partial_correlations.{table_format}",
            table_format=table_format,
        ),
        "molecular_prediction_benchmark": _write_table(
            result["molecular_prediction_benchmark"],
            out / f"molecular_prediction_benchmark.{table_format}",
            table_format=table_format,
        ),
        "morphomolecular_hero_targets": _write_table(
            result["morphomolecular_hero_targets"],
            out / f"morphomolecular_hero_targets.{table_format}",
            table_format=table_format,
        ),
        "morphomolecular_hero_contours": _write_table(
            result["morphomolecular_hero_contours"],
            out / f"morphomolecular_hero_contours.{table_format}",
            table_format=table_format,
        ),
        "morphomolecular_concept_tests": _write_table(
            result["morphomolecular_concept_tests"],
            out / f"morphomolecular_concept_tests.{table_format}",
            table_format=table_format,
        ),
        "boundary_coupling_summary": _write_table(
            result["boundary_coupling_summary"],
            out / f"boundary_coupling_summary.{table_format}",
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
    has_prompt_similarity = "top_prompt_similarity" in frame.columns
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
            or str(column) == "top_prompt_similarity"
            or (str(column) == "top_image_label_score" and not has_prompt_similarity)
        )
    ]


def _numeric_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) not in exclude
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _contour_image_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        _EMBEDDING_PREFIX,
        _TEXT_PREFIX,
        "relative_prompt_axis__",
        "image_heterogeneity__",
        "morphology_entropy__",
        "domain_fraction__",
    )
    suffixes = ("__mean", "__std", "__rank")
    excluded = {
        "contour_id",
        "assigned_structure",
        "structure_id",
        "classification_name",
        "n_tiles",
        "n_cells",
        "area_image_px2",
        "perimeter_image_px",
        "centroid_x",
        "centroid_y",
        "shape_compactness",
        "cell_density_per_1e6_image_px2",
        "tile_density_per_1e6_image_px2",
    }
    return [
        column
        for column in _numeric_columns(frame, exclude=excluded)
        if column.startswith(prefixes)
        or column.endswith(suffixes)
        and (
            column.startswith(_EMBEDDING_PREFIX)
            or column.startswith(_TEXT_PREFIX)
            or column.startswith("top_prompt_similarity")
            or column.startswith("top_image_label_score")
        )
    ]


def _contour_molecular_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("rna__", "cell_type_fraction__", "cell_type_diversity__", "program__")
    return [
        column
        for column in _numeric_columns(frame, exclude=set(_ID_COLUMNS) | {"n_cells", "n_tiles"})
        if column.startswith(prefixes)
    ]


def _contour_control_matrix(frame: pd.DataFrame, *, controls: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if "assigned_structure" in controls and "assigned_structure" in frame.columns:
        pieces.append(
            pd.get_dummies(
                frame["assigned_structure"].fillna("unassigned").astype(str),
                prefix="structure",
                dtype=float,
            )
        )
    for control in controls:
        if control == "assigned_structure" or control not in frame.columns:
            continue
        values = pd.to_numeric(frame[control], errors="coerce")
        if values.notna().any():
            pieces.append(pd.DataFrame({control: values.fillna(values.median()).astype(float)}))
    if not pieces:
        return pd.DataFrame(index=frame.index)
    matrix = pd.concat(pieces, axis=1)
    matrix.index = frame.index
    return matrix


def _matrix_partial_spearman_associations(
    frame: pd.DataFrame,
    *,
    image_columns: Sequence[str],
    molecular_columns: Sequence[str],
    controls: Sequence[str],
    min_contours: int,
) -> pd.DataFrame:
    image_raw, image_notna = _rank_ready_matrix(frame, image_columns)
    molecular_raw, molecular_notna = _rank_ready_matrix(frame, molecular_columns)
    if image_raw.empty or molecular_raw.empty:
        return pd.DataFrame(
            columns=[
                "association_kind",
                "image_feature",
                "molecular_feature",
                "spearman_rho",
                "partial_spearman_rho",
                "abs_partial_spearman_rho",
                "p_value",
                "n_contours",
                "controls",
            ]
        )
    common = image_raw.index.intersection(molecular_raw.index)
    image_raw = image_raw.loc[common, :]
    molecular_raw = molecular_raw.loc[common, :]
    image_notna = image_notna.loc[common, :]
    molecular_notna = molecular_notna.loc[common, :]
    row_mask = image_notna.any(axis=1) & molecular_notna.any(axis=1)
    common = common[row_mask.to_numpy()]
    image_raw = image_raw.loc[common, :]
    molecular_raw = molecular_raw.loc[common, :]
    image_notna = image_notna.loc[common, :]
    molecular_notna = molecular_notna.loc[common, :]
    if len(common) < min_contours:
        return pd.DataFrame()
    control_matrix = _contour_control_matrix(frame.loc[common, :], controls=controls)
    image_rank = image_raw.rank(axis=0, method="average")
    molecular_rank = molecular_raw.rank(axis=0, method="average")
    image_resid = _residualize_matrix(image_rank.to_numpy(dtype=float), control_matrix)
    molecular_resid = _residualize_matrix(molecular_rank.to_numpy(dtype=float), control_matrix)
    partial = _column_correlation_matrix(image_resid, molecular_resid)
    raw_spearman = _column_correlation_matrix(
        image_rank.to_numpy(dtype=float),
        molecular_rank.to_numpy(dtype=float),
    )
    valid_counts = image_notna.astype(int).T.to_numpy() @ molecular_notna.astype(int).to_numpy()
    n = len(common)
    df = max(n - max(control_matrix.shape[1], 0) - 2, 1)
    clipped = np.clip(partial, -0.999999, 0.999999)
    t_stat = clipped * np.sqrt(df / np.maximum(1.0 - clipped * clipped, np.finfo(float).eps))
    p_values = 2.0 * t.sf(np.abs(t_stat), df)

    rows = []
    controls_text = ",".join([control for control in controls if control in frame.columns])
    for image_index, image_feature in enumerate(image_raw.columns):
        for molecular_index, molecular_feature in enumerate(molecular_raw.columns):
            count = int(valid_counts[image_index, molecular_index])
            if count < min_contours:
                continue
            rho = float(partial[image_index, molecular_index])
            if not np.isfinite(rho):
                continue
            rows.append(
                {
                    "association_kind": "contour_partial_image_molecular_association",
                    "image_feature": str(image_feature),
                    "molecular_feature": str(molecular_feature),
                    "spearman_rho": float(raw_spearman[image_index, molecular_index]),
                    "partial_spearman_rho": rho,
                    "abs_partial_spearman_rho": float(abs(rho)),
                    "p_value": float(p_values[image_index, molecular_index]),
                    "n_contours": count,
                    "controls": controls_text,
                }
            )
    return pd.DataFrame(rows)


def _rank_ready_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values: dict[str, pd.Series] = {}
    notna: dict[str, pd.Series] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        mask = series.notna()
        if int(mask.sum()) == 0:
            continue
        filled = series.fillna(series.median()).astype(float)
        if float(filled.std(ddof=0)) == 0.0:
            continue
        values[str(column)] = filled
        notna[str(column)] = mask
    return pd.DataFrame(values, index=frame.index), pd.DataFrame(notna, index=frame.index)


def _residualize_matrix(values: np.ndarray, controls: pd.DataFrame) -> np.ndarray:
    if controls.empty:
        return values - np.nanmean(values, axis=0, keepdims=True)
    design = np.column_stack([np.ones(values.shape[0], dtype=float), controls.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def _column_correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - np.nanmean(left, axis=0, keepdims=True)
    right_centered = right - np.nanmean(right, axis=0, keepdims=True)
    left_norm = np.sqrt(np.nansum(left_centered * left_centered, axis=0))
    right_norm = np.sqrt(np.nansum(right_centered * right_centered, axis=0))
    denom = left_norm[:, None] * right_norm[None, :]
    numerator = left_centered.T @ right_centered
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def _partial_spearman(x: np.ndarray, y: np.ndarray, controls: pd.DataFrame) -> tuple[float, float]:
    ranked_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ranked_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if controls.empty:
        rho, p_value = pearsonr(ranked_x, ranked_y)
        return float(rho), float(p_value)
    control_values = controls.to_numpy(dtype=float)
    if control_values.shape[0] != len(ranked_x):
        return np.nan, np.nan
    x_resid = _residualize(ranked_x, control_values)
    y_resid = _residualize(ranked_y, control_values)
    if np.nanstd(x_resid) == 0.0 or np.nanstd(y_resid) == 0.0:
        return np.nan, np.nan
    rho, p_value = pearsonr(x_resid, y_resid)
    return float(rho), float(p_value)


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values), dtype=float), controls])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def _finite_feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    pieces = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        filled = values.fillna(values.median()).astype(float)
        if float(filled.std(ddof=0)) == 0.0:
            continue
        pieces[column] = filled
    return pd.DataFrame(pieces, index=frame.index)


def _select_prediction_targets(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    max_targets: int,
) -> list[str]:
    scored = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        variance = float(values.var(ddof=0))
        if variance > 0:
            scored.append((column, variance))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [column for column, _ in scored[:max_targets]]


def _image_available_mask(frame: pd.DataFrame, image_columns: Sequence[str]) -> pd.Series:
    if "n_tiles" in frame.columns:
        tiles = pd.to_numeric(frame["n_tiles"], errors="coerce")
        mask = tiles.notna() & (tiles > 0)
        if bool(mask.any()):
            return mask
    available = pd.Series(False, index=frame.index)
    for column in image_columns:
        available |= pd.to_numeric(frame[column], errors="coerce").notna()
    return available


def _spatial_cv_groups(frame: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if {"centroid_x", "centroid_y"}.issubset(frame.columns):
        x = pd.to_numeric(frame["centroid_x"], errors="coerce")
        y = pd.to_numeric(frame["centroid_y"], errors="coerce")
        if x.notna().sum() >= 12 and y.notna().sum() >= 12:
            try:
                x_bin = pd.qcut(x.rank(method="first"), q=min(4, max(2, int(np.sqrt(len(frame)) // 2))), labels=False)
                y_bin = pd.qcut(y.rank(method="first"), q=min(4, max(2, int(np.sqrt(len(frame)) // 2))), labels=False)
                groups = (x_bin.astype(str) + "_" + y_bin.astype(str)).rename("spatial_block")
                if groups.nunique() >= 3:
                    return groups, "spatial_block_group_kfold"
            except Exception:
                pass
    return None, "kfold"


def _cross_validated_r2(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series | None,
    *,
    random_state: int,
) -> float:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold, KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return np.nan

    x_values = x.to_numpy(dtype=float)
    n_samples = int(len(y))
    if n_samples < 3 or x_values.shape[1] == 0:
        return np.nan
    predictions = np.full(n_samples, np.nan, dtype=float)
    if groups is not None and groups.nunique() >= 3:
        n_splits = min(5, int(groups.nunique()))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(x_values, y, groups=groups.to_numpy())
    else:
        n_splits = min(5, n_samples)
        if n_splits < 3:
            return np.nan
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(x_values, y)
    for train, test in splits:
        if len(np.unique(y[train])) < 2:
            predictions[test] = float(np.mean(y[train]))
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_values[train], y[train])
        predictions[test] = model.predict(x_values[test])
    mask = np.isfinite(predictions)
    if int(mask.sum()) < 3:
        return np.nan
    total = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    if total == 0.0:
        return np.nan
    residual = float(np.sum((y[mask] - predictions[mask]) ** 2))
    return float(1.0 - residual / total)


def _normalize_relative_prompt_axes(
    axes: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, str, str], ...]:
    if axes is None:
        return DEFAULT_RELATIVE_PROMPT_AXES
    normalized = []
    for axis in axes:
        if len(axis) != 3:
            raise ValueError("Each relative prompt axis must contain name, positive prompt, and negative prompt.")
        name, positive, negative = axis
        normalized.append((str(name), str(positive), str(negative)))
    return tuple(normalized)


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
            "top_prompt_term",
            "top_prompt_term_fraction",
            "top_image_label",
            "top_image_label_fraction",
        ]
    )


def _empty_contour_multimodal_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contour_id",
            "assigned_structure",
            "structure_id",
            "classification_name",
            "area_image_px2",
            "perimeter_image_px",
            "centroid_x",
            "centroid_y",
            "n_tiles",
            "n_cells",
        ]
    )


def _empty_contour_rna_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["contour_id", "assigned_structure", "structure_id", "n_cells"])


def _empty_structure_rna_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["assigned_structure", "structure_id", "n_contours", "n_cells"])


def _empty_molecular_prediction_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "target_feature",
            "n_contours",
            "n_image_features",
            "cv_strategy",
            "r2_structure_only",
            "r2_image_only",
            "r2_structure_image",
            "delta_r2_image_over_structure",
            "delta_r2_combined_over_structure",
        ]
    )


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


def _unique_ordered(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value)
        key = text.upper()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


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
