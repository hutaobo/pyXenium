from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyXenium.contour import add_contours_from_geojson, expand_contours
from pyXenium.contour._analysis import _prepare_contours
from pyXenium.contour._feature_table import (
    DEFAULT_CONTOUR_LR_PAIRS,
    _build_expression_frame,
    _build_pathway_activity,
    _edge_contrast_features,
    _geometry_features,
    _resolve_selected_genes,
    _slug,
)
from pyXenium.io import write_xenium
from pyXenium.io.sdata_model import XeniumSData
from pyXenium.multimodal import run_contour_boundary_ecology_pilot
from pyXenium.validation.atera_wta_cervical_end_to_end import (
    DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS,
    DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
    DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH,
    DEFAULT_ATERA_WTA_CERVICAL_DENSITY_SUBDIR,
    DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY,
    DEFAULT_ATERA_WTA_CERVICAL_MULTIMODAL_SUBDIR,
    DEFAULT_ATERA_WTA_CERVICAL_OUTPUT_DIRNAME,
    DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
    DEFAULT_ATERA_WTA_CERVICAL_SDATA_NAME,
    DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR,
    _load_atera_wta_cervical_adata,
    _load_atera_wta_cervical_sdata,
    _load_cervical_cell_groups,
    _resolve_he_pixel_size_um,
    build_atera_wta_cervical_bio6_structures,
    build_serializable_cervical_end_to_end_summary,
    render_atera_wta_cervical_end_to_end_report,
)


DATASET_ROOT = Path(DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH)
OUTPUT_ROOT = DATASET_ROOT / DEFAULT_ATERA_WTA_CERVICAL_OUTPUT_DIRNAME
CONTOUR_DIR = OUTPUT_ROOT / "contours_bio6"
DENSITY_DIR = OUTPUT_ROOT / DEFAULT_ATERA_WTA_CERVICAL_DENSITY_SUBDIR
MULTIMODAL_DIR = OUTPUT_ROOT / DEFAULT_ATERA_WTA_CERVICAL_MULTIMODAL_SUBDIR
SDATA_OUTPUT = OUTPUT_ROOT / DEFAULT_ATERA_WTA_CERVICAL_SDATA_NAME
TBC_RESULTS = DATASET_ROOT / DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR
LOG_PATH = OUTPUT_ROOT / "run_post_density_resume.log"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def existing_files_payload() -> dict[str, Any]:
    topology_dir = OUTPUT_ROOT / "topology"
    contour_geojson = CONTOUR_DIR / "xenium_explorer_annotations.geojson"
    payload: dict[str, Any] = {
        "structure_map_pdf": str(next(TBC_RESULTS.glob("StructureMap_of_*.pdf"))),
        "structure_map_table": str(next(TBC_RESULTS.glob("StructureMap_table_*.csv"))),
        "t_and_c_result": str(next(TBC_RESULTS.glob("t_and_c_result_*.csv"))),
        "contour_geojson": str(contour_geojson),
        "contour_csv": str(CONTOUR_DIR / "xenium_explorer_annotations.csv"),
        "contour_summary_csv": str(CONTOUR_DIR / "xenium_explorer_annotations_summary.csv"),
        "contour_partition_table": str(CONTOUR_DIR / "cells_with_structure_partition.parquet"),
        "contour_structure_count_csv": str(CONTOUR_DIR / "structure_contour_cell_counts.csv"),
        "contour_metrics_json": str(CONTOUR_DIR / "structure_contour_metrics.json"),
        "ring_density_csv": str(DENSITY_DIR / "ring_density_markers.csv"),
        "smooth_density_csv": str(DENSITY_DIR / "smooth_density_markers.csv"),
        "multimodal_artifact_dir": str(MULTIMODAL_DIR),
    }
    for name in (
        "ligand_to_cell.csv",
        "receptor_to_cell.csv",
        "lr_sender_receiver_scores.csv",
        "lr_component_diagnostics.csv",
        "pathway_to_cell.csv",
        "pathway_activity_to_cell.csv",
        "pathway_mode_comparison.csv",
        "pathway_structuremap.csv",
        "pathway_activity_structuremap.csv",
    ):
        path = topology_dir / name
        if path.exists():
            payload[path.stem] = str(path)
    preview = CONTOUR_DIR / "multi_structure_contour_preview.png"
    if preview.exists():
        payload["contour_preview_png"] = str(preview)
    return payload


def add_multimodal_files(payload: dict[str, Any]) -> None:
    mapping = {
        "multimodal_summary_json": "summary.json",
        "multimodal_report_md": "report.md",
        "multimodal_exemplar_montage": "exemplar_montage.png",
        "multimodal_contour_features_csv": "contour_features.csv",
        "multimodal_program_scores_csv": "program_scores.csv",
        "multimodal_ecotype_assignments_csv": "ecotype_assignments.csv",
        "multimodal_edge_gradients_csv": "edge_gradients.csv",
        "multimodal_matched_exemplars_csv": "matched_exemplars.csv",
    }
    for key, filename in mapping.items():
        path = MULTIMODAL_DIR / filename
        if path.exists():
            payload[key] = str(path)


def acceptance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    multimodal_files = [
        "summary.json",
        "report.md",
        "contour_features.csv",
        "program_scores.csv",
        "ecotype_assignments.csv",
        "edge_gradients.csv",
        "matched_exemplars.csv",
    ]
    top_level_files = ["summary.json", "report.md", DEFAULT_ATERA_WTA_CERVICAL_SDATA_NAME]
    return {
        "multimodal_files": {name: (MULTIMODAL_DIR / name).exists() for name in multimodal_files},
        "top_level_files": {name: (OUTPUT_ROOT / name).exists() for name in top_level_files},
        "contour_structure_count": payload.get("contour_structure_count"),
        "contour_key": payload.get("contour_key"),
        "expanded_contour_key": payload.get("expanded_contour_key"),
        "ring_density_rows": payload.get("ring_density_summary", {}).get("row_count"),
        "smooth_density_rows": payload.get("smooth_density_summary", {}).get("row_count"),
        "multimodal_n_contours": payload.get("multimodal_sample_summary", {}).get("n_contours"),
    }


def edge_gradients_from_smooth_density(smooth_density: pd.DataFrame) -> pd.DataFrame:
    required = {"contour_id", "signed_distance", "density"}
    missing = required.difference(smooth_density.columns)
    if missing:
        raise KeyError(f"Smooth density table is missing required columns: {sorted(missing)}")

    frame = smooth_density.copy()
    if "contour_key" not in frame.columns:
        frame["contour_key"] = DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY
    frame["signed_distance"] = pd.to_numeric(frame["signed_distance"], errors="coerce")
    frame["density"] = pd.to_numeric(frame["density"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for (contour_key, contour_id), group in frame.groupby(["contour_key", "contour_id"], sort=False, dropna=False):
        signed = group["signed_distance"].to_numpy(dtype=float)
        density = group["density"].to_numpy(dtype=float)
        finite_density = np.isfinite(density)
        inner_mask = (signed < 0.0) & finite_density
        outer_mask = (signed > 0.0) & finite_density
        positive_density = np.clip(np.nan_to_num(density, nan=0.0), a_min=0.0, a_max=None)
        if positive_density.sum() > 0:
            center_of_mass = float(np.sum(np.nan_to_num(signed, nan=0.0) * positive_density) / np.sum(positive_density))
        else:
            center_of_mass = float("nan")
        zero_index = int(np.argmin(np.abs(np.nan_to_num(signed, nan=np.inf)))) if signed.size else 0
        rows.append(
            {
                "contour_key": str(contour_key),
                "contour_id": str(contour_id),
                "gradient_key": "marker_panel",
                "inner_mean": float(np.nanmean(density[inner_mask])) if inner_mask.any() else 0.0,
                "outer_mean": float(np.nanmean(density[outer_mask])) if outer_mask.any() else 0.0,
                "outer_minus_inner": (
                    float(np.nanmean(density[outer_mask]) - np.nanmean(density[inner_mask]))
                    if inner_mask.any() and outer_mask.any()
                    else 0.0
                ),
                "boundary_peak": float(density[zero_index]) if signed.size and np.isfinite(density[zero_index]) else 0.0,
                "center_of_mass": center_of_mass,
            }
        )
    return pd.DataFrame(rows)


def build_fast_resume_feature_table(
    *,
    sdata: Any,
    adata: Any,
    precomputed_edge_gradients: pd.DataFrame,
) -> dict[str, Any]:
    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        contour_query=None,
    ).sort_values("contour_id", kind="stable").reset_index(drop=True)
    log(f"Fast feature table: loaded {len(contour_table)} contour geometries")

    partition = pd.read_parquet(
        CONTOUR_DIR / "cells_with_structure_partition.parquet",
        columns=["cell_id", "cluster", "isoline_structure_name"],
    )
    obs_names = pd.Index(adata.obs_names.astype(str), name="cell_id")
    partition = partition.drop_duplicates("cell_id").set_index("cell_id").reindex(obs_names)
    fallback_cluster = pd.Series("unassigned", index=obs_names, dtype="object")
    if "cluster" in adata.obs.columns:
        fallback_cluster = pd.Series(adata.obs["cluster"].astype(str).to_numpy(), index=obs_names, dtype="object")
    structures = partition["isoline_structure_name"].fillna("unassigned").astype(str)
    clusters = partition["cluster"].where(partition["cluster"].notna(), fallback_cluster).fillna("unassigned").astype(str)

    selected_genes = _resolve_selected_genes(adata)
    log(f"Fast feature table: extracting {len(selected_genes)} RNA/LR/pathway genes")
    expression = _build_expression_frame(adata, selected_genes)
    pathway_activity = _build_pathway_activity(expression)
    global_expression = expression.mean(axis=0)
    global_pathway = pathway_activity.mean(axis=0)
    structure_expression = expression.groupby(structures).mean()
    structure_pathway = pathway_activity.groupby(structures).mean()
    structure_counts = structures.value_counts()
    contour_counts = contour_table["assigned_structure"].astype(str).value_counts() if "assigned_structure" in contour_table.columns else pd.Series(dtype=int)

    state_categories = sorted(pd.unique(clusters).astype(str).tolist())
    state_categories.extend(["macrophage_like", "endothelial_perivascular", "emt_like_tumor", "b_plasma_like", "t_cell_exhausted_cytotoxic"])
    state_categories = sorted(set(state_categories))
    niche_categories = ["immune_rich", "mixed_low_signal"]
    global_state_fraction = clusters.value_counts(normalize=True)
    structure_state_fraction = pd.crosstab(structures, clusters, normalize="index")

    gradient_lookup = {
        str(contour_id): group
        for contour_id, group in precomputed_edge_gradients.groupby("contour_id", sort=False, dropna=False)
    }

    feature_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    rna_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []

    log("Fast feature table: assembling per-contour rows")
    for _, contour_row in contour_table.iterrows():
        contour_id = str(contour_row["contour_id"])
        structure = str(contour_row.get("assigned_structure", "unassigned"))
        inner_expression = structure_expression.loc[structure] if structure in structure_expression.index else global_expression
        outer_expression = global_expression
        inner_pathway = structure_pathway.loc[structure] if structure in structure_pathway.index else global_pathway
        outer_pathway = global_pathway
        per_contour_cells = float(structure_counts.get(structure, 0)) / max(float(contour_counts.get(structure, 1)), 1.0)

        row = {
            "sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
            "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
            "contour_id": contour_id,
        }
        for metadata_column in ("assigned_structure", "classification_name", "annotation_source", "structure_id"):
            if metadata_column in contour_row.index:
                row[metadata_column] = contour_row[metadata_column]
        geometry = contour_row["geometry"]
        row.update(_geometry_features(geometry))
        centroid = geometry.centroid
        row["context__centroid_x_um"] = float(centroid.x)
        row["context__centroid_y_um"] = float(centroid.y)
        row["context__neighbor_count"] = float(contour_counts.get(structure, 0))
        row["context__contact_degree"] = float(np.log1p(contour_counts.get(structure, 0)))
        row["context__neighbor_same_label_fraction"] = 1.0

        structure_state = structure_state_fraction.loc[structure] if structure in structure_state_fraction.index else global_state_fraction
        alias_state = _alias_state_fractions(structure)
        alias_outer = _alias_state_fractions("mixed")
        for zone_name, count_scale, expr_values, pathway_values, state_values, alias_values in (
            ("whole", 1.0, inner_expression, inner_pathway, structure_state, alias_state),
            ("inner_rim", 0.35, inner_expression, inner_pathway, structure_state, alias_state),
            ("outer_rim", 0.30, outer_expression, outer_pathway, global_state_fraction, alias_outer),
        ):
            zone_row = {
                "sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
                "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
                "contour_id": contour_id,
                "zone": zone_name,
                "area_um2": float(row.get("geometry__area_um2", 0.0)) * count_scale,
                "n_cells": int(round(per_contour_cells * count_scale)),
                "state_entropy": 0.0,
                "niche_entropy": 0.0,
            }
            for state in state_categories:
                zone_row[f"state_fraction__{_slug(state)}"] = float(state_values.get(state, 0.0))
            for alias, value in alias_values.items():
                zone_row[f"state_fraction__{alias}"] = float(value)
            zone_row["niche_fraction__immune_rich"] = float(alias_values.get("immune_rich", 0.0))
            zone_row["niche_fraction__mixed_low_signal"] = 1.0 - zone_row["niche_fraction__immune_rich"]
            zone_rows.append(zone_row)
            row.update(
                {
                    f"omics__{zone_name}__{key}": value
                    for key, value in zone_row.items()
                    if key not in {"sample_id", "contour_key", "contour_id", "zone"}
                }
            )
            row.update({f"rna__{zone_name}__{gene}": float(expr_values.get(gene, 0.0)) for gene in selected_genes})
            row.update({f"pathway__{zone_name}__{name}": float(pathway_values.get(name, 0.0)) for name in pathway_activity.columns.astype(str)})

        lr_row = {
            "sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
            "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
            "contour_id": contour_id,
        }
        for pair_name, (ligand, receptor) in DEFAULT_CONTOUR_LR_PAIRS.items():
            ligand_inner = float(inner_expression.get(ligand, 0.0))
            receptor_inner = float(inner_expression.get(receptor, 0.0))
            ligand_outer = float(outer_expression.get(ligand, 0.0))
            receptor_outer = float(outer_expression.get(receptor, 0.0))
            lr_row[f"{pair_name}__cross_zone"] = 0.5 * (
                np.sqrt(max(ligand_inner, 0.0) * max(receptor_outer, 0.0))
                + np.sqrt(max(ligand_outer, 0.0) * max(receptor_inner, 0.0))
            )
            lr_row[f"{pair_name}__outer_minus_inner"] = (ligand_outer + receptor_outer) - (ligand_inner + receptor_inner)
        row.update({f"lr__{key}": value for key, value in lr_row.items() if key not in {"sample_id", "contour_key", "contour_id"}})

        contour_gradients = gradient_lookup.get(contour_id)
        if contour_gradients is not None:
            for _, gradient_row in contour_gradients.iterrows():
                gene_set = str(gradient_row["gradient_key"])
                row[f"gradient__{gene_set}__outer_minus_inner"] = float(gradient_row["outer_minus_inner"])
                row[f"gradient__{gene_set}__boundary_peak"] = float(gradient_row["boundary_peak"])
                row[f"gradient__{gene_set}__center_of_mass"] = float(gradient_row["center_of_mass"])
                row[f"gradient__{gene_set}__outer_mean"] = float(gradient_row["outer_mean"])
                row[f"gradient__{gene_set}__inner_mean"] = float(gradient_row["inner_mean"])

        row.update(_edge_contrast_features(row))
        feature_rows.append(row)

        rna_row = {"sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID, "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY, "contour_id": contour_id}
        rna_row.update({gene: float(inner_expression.get(gene, 0.0)) for gene in selected_genes})
        rna_rows.append(rna_row)
        pathway_row = dict(rna_row)
        pathway_row = {"sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID, "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY, "contour_id": contour_id}
        pathway_row.update({name: float(inner_pathway.get(name, 0.0)) for name in pathway_activity.columns.astype(str)})
        pathway_rows.append(pathway_row)
        lr_rows.append(lr_row)
        protein_rows.append({"sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID, "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY, "contour_id": contour_id})

    contour_features = pd.DataFrame(feature_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
    log("Fast feature table: finalizing feature payload")
    return {
        "sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
        "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        "inner_rim_um": 20.0,
        "outer_rim_um": 30.0,
        "contour_features": contour_features,
        "zone_summary": pd.DataFrame(zone_rows),
        "rna_pseudobulk": pd.DataFrame(rna_rows),
        "protein_summary": pd.DataFrame(protein_rows),
        "pathway_activity": pd.DataFrame(pathway_rows),
        "ligand_receptor_summary": pd.DataFrame(lr_rows),
        "edge_gradients": precomputed_edge_gradients,
        "embedding_summary": pd.DataFrame(columns=["sample_id", "contour_key", "contour_id"]),
        "available_states": state_categories,
        "available_niches": niche_categories,
        "feature_columns": _feature_columns(contour_features),
        "context": {
            "multimodal_context": ["fast_resume_structure_level_cell_summaries"],
            "used_pathomics": False,
            "used_embeddings": False,
            "used_precomputed_edge_gradients": True,
            "fast_resume": True,
        },
    }


def _alias_state_fractions(structure: str) -> dict[str, float]:
    structure = str(structure)
    return {
        "macrophage_like": 1.0 if structure == "Myeloid" else 0.0,
        "endothelial_perivascular": 1.0 if structure in {"Vascular/Endocervical", "Stromal/Fibro/Muscle"} else 0.0,
        "emt_like_tumor": 1.0 if structure == "Tumor" else 0.0,
        "b_plasma_like": 1.0 if structure == "B/Plasma" else 0.0,
        "t_cell_exhausted_cytotoxic": 1.0 if structure == "T-cell" else 0.0,
        "immune_rich": 1.0 if structure in {"T-cell", "B/Plasma", "Myeloid"} else 0.0,
    }


def _feature_columns(contour_features: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "geometry": [column for column in contour_features.columns if column.startswith("geometry__")],
        "context": [column for column in contour_features.columns if column.startswith("context__")],
        "pathomics": [column for column in contour_features.columns if column.startswith("pathomics__")],
        "omics": [column for column in contour_features.columns if column.startswith("omics__")],
        "pathway": [column for column in contour_features.columns if column.startswith("pathway__")],
        "protein": [column for column in contour_features.columns if column.startswith("protein__")],
        "rna": [column for column in contour_features.columns if column.startswith("rna__")],
        "edge_contrast": [column for column in contour_features.columns if column.startswith("edge_contrast__")],
        "gradient": [column for column in contour_features.columns if column.startswith("gradient__")],
        "ligand_receptor": [column for column in contour_features.columns if column.startswith("lr__")],
        "embedding": [column for column in contour_features.columns if column.startswith("embedding__")],
    }


def build_lightweight_contour_sdata(sdata: Any) -> XeniumSData:
    contour_shapes = {
        key: sdata.shapes[key]
        for key in (
            DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
            DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY,
        )
        if key in sdata.shapes
    }
    metadata = dict(sdata.metadata)
    metadata["lightweight_contour_enriched_copy"] = True
    metadata["omitted_components"] = ["points", "cell_boundaries", "nucleus_boundaries", "images"]
    return XeniumSData(
        table=sdata.table,
        shapes=contour_shapes,
        images={},
        contour_images={},
        metadata=metadata,
        points={},
        point_sources={},
    )


def main() -> None:
    start = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Post-density resume started for {DATASET_ROOT}")

    require_path(TBC_RESULTS, label="sfplot results directory")
    require_path(CONTOUR_DIR / "xenium_explorer_annotations.geojson", label="contour GeoJSON")
    ring_density_csv = require_path(DENSITY_DIR / "ring_density_markers.csv", label="ring density CSV")
    smooth_density_csv = require_path(DENSITY_DIR / "smooth_density_markers.csv", label="smooth density CSV")

    log("Loading cell groups and AnnData summary table")
    group_df = _load_cervical_cell_groups(DATASET_ROOT)
    bio6_structures = build_atera_wta_cervical_bio6_structures(group_df)
    adata = _load_atera_wta_cervical_adata(DATASET_ROOT)

    log("Loading XeniumSData with streamed transcripts, boundaries, and H&E image")
    sdata = _load_atera_wta_cervical_sdata(DATASET_ROOT)

    log("Importing existing 6-structure GeoJSON contours without full H&E patch extraction")
    add_contours_from_geojson(
        sdata,
        CONTOUR_DIR / "xenium_explorer_annotations.geojson",
        key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        id_key="name",
        pixel_size_um=_resolve_he_pixel_size_um(sdata),
        extract_he_patches=False,
    )

    log("Expanding contours with 30um Voronoi mode")
    expand_contours(
        sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        distance=30.0,
        mode="voronoi",
        output_key=DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY,
    )

    log("Reading existing density tables")
    ring_density = pd.read_csv(ring_density_csv)
    smooth_density = pd.read_csv(smooth_density_csv)
    precomputed_edge_gradients = edge_gradients_from_smooth_density(smooth_density)
    log(f"Prepared {len(precomputed_edge_gradients)} precomputed marker-panel edge gradient rows")
    log("Building fast resume contour feature table from existing contours and structure-level cell summaries")
    feature_table = build_fast_resume_feature_table(
        sdata=sdata,
        adata=adata,
        precomputed_edge_gradients=precomputed_edge_gradients,
    )
    log(f"Prepared fast contour feature table with {len(feature_table['contour_features'])} contours")

    log("Running contour boundary ecology multimodal pilot in omics/geometry mode")
    multimodal = run_contour_boundary_ecology_pilot(
        sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        output_dir=MULTIMODAL_DIR,
        include_pathomics=False,
        precomputed_edge_gradients=precomputed_edge_gradients,
        precomputed_feature_table=feature_table,
    )

    log(f"Writing lightweight contour-enriched SData to {SDATA_OUTPUT}")
    sdata_to_write = build_lightweight_contour_sdata(sdata)
    sdata_write_result = write_xenium(
        sdata_to_write,
        SDATA_OUTPUT,
        format="sdata",
        overwrite=True,
    )

    files = existing_files_payload()
    add_multimodal_files(files)
    files["contour_enriched_sdata"] = str(sdata_write_result["output_path"])

    study = {
        "sample_id": DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
        "dataset_root": str(DATASET_ROOT.resolve()),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "tbc": None,
        "tbc_results": str(TBC_RESULTS.resolve()),
        "adata": adata,
        "sdata": sdata,
        "lr": {},
        "pathway": {},
        "contour_generation": {},
        "ring_density": ring_density,
        "smooth_density": smooth_density,
        "multimodal": multimodal,
        "bio6_structures": bio6_structures,
        "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        "expanded_contour_key": DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY,
        "files": files,
        "runtime_seconds": time.perf_counter() - start,
    }

    log("Writing top-level summary.json and report.md")
    payload = build_serializable_cervical_end_to_end_summary(study)
    summary_path = OUTPUT_ROOT / "summary.json"
    report_path = OUTPUT_ROOT / "report.md"
    files["summary_json"] = str(summary_path)
    files["report_md"] = str(report_path)
    study["files"] = files
    payload = build_serializable_cervical_end_to_end_summary(study)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_atera_wta_cervical_end_to_end_report(payload), encoding="utf-8")

    acceptance = acceptance_summary(payload)
    log("Acceptance summary:")
    print(json.dumps(acceptance, indent=2), flush=True)
    log("Post-density resume completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
