from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pyXenium.contour import (
    add_contours_from_geojson,
    expand_contours,
    generate_xenium_explorer_annotations,
    ring_density,
    smooth_density_by_distance,
)
from pyXenium.io import XeniumSData, read_xenium, write_xenium
from pyXenium.cci import cci_topology_analysis
from pyXenium.multimodal import run_contour_boundary_ecology_pilot
from pyXenium.pathway import pathway_topology_analysis

from .sfplot_tbc_bridge import run_sfplot_tbc_table_bundle

DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH = (
    r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Cervical_Cancer_outs"
)
DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR = r"sfplot_tbc_formal_wta\results"
DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS = "WTA_Preview_FFPE_Cervical_Cancer_cell_groups.csv"
DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID = "atera_wta_ffpe_cervical"
DEFAULT_ATERA_WTA_CERVICAL_SFPLOT_SAMPLE_NAME = "WTA_Preview_FFPE_Cervical_Cancer_formal"
DEFAULT_ATERA_WTA_CERVICAL_OUTPUT_DIRNAME = "pyxenium_cervical_end_to_end"
DEFAULT_ATERA_WTA_CERVICAL_TOPOLOGY_SUBDIR = "topology"
DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_SUBDIR = "contours_bio6"
DEFAULT_ATERA_WTA_CERVICAL_DENSITY_SUBDIR = "density_profiles"
DEFAULT_ATERA_WTA_CERVICAL_MULTIMODAL_SUBDIR = "multimodal_contour_ecology_bio6"
DEFAULT_ATERA_WTA_CERVICAL_SDATA_NAME = "cervical_with_contours.sdata"
DEFAULT_ATERA_WTA_CERVICAL_HISTOSEG_ROOT = r"D:\GitHub\HistoSeg"
DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY = "atera_cervical_bio6"
DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY = "atera_cervical_bio6_voronoi30um"
DEFAULT_ATERA_WTA_CERVICAL_PIXEL_SIZE_UM = 0.2125

DEFAULT_ATERA_WTA_CERVICAL_CCI_PANEL = pd.DataFrame(
    [
        {"ligand": "SPP1", "receptor": "CD44", "evidence_weight": 1.0},
        {"ligand": "CXCL13", "receptor": "CXCR5", "evidence_weight": 1.0},
        {"ligand": "CXCL12", "receptor": "CXCR4", "evidence_weight": 1.0},
        {"ligand": "TGFB1", "receptor": "TGFBR2", "evidence_weight": 1.0},
        {"ligand": "VEGFA", "receptor": "KDR", "evidence_weight": 1.0},
    ]
)

DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL: dict[str, list[str]] = {
    "immune_activation": ["CD3D", "TRAC", "NKG7", "CXCL13", "HLA-DRA"],
    "myeloid_activation": ["SPP1", "CD68", "HLA-DRA", "FCER1G", "LST1"],
    "vascular_stromal": ["PECAM1", "KDR", "RGS5", "COL1A1", "ACTA2"],
    "emt_invasion": ["VIM", "MMP11", "ITGB6", "KRT14", "TGFB1"],
    "stromal_matrix": ["COL1A1", "COL3A1", "DCN", "TAGLN", "THY1"],
    "tls_activation": ["CXCL13", "MS4A1", "CD79A", "JCHAIN", "TRAC"],
    "hypoxia_necrosis": ["CA9", "SLC2A1", "VEGFA", "LDHA", "HILPDA"],
}

DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL = (
    "SPP1",
    "CXCL13",
    "CXCL12",
    "TGFB1",
    "VEGFA",
    "CA9",
)

DEFAULT_ATERA_WTA_CERVICAL_BIO6_GROUPS: dict[str, tuple[str, ...]] = {
    "Tumor": (
        "Metabolic Invasive Basal Cells",
        "Hypoxic Tumor Cells",
        "Differentiating Tumor Cells",
        "Migratory Invasive Basal Cells",
        "Parabasal Tumor Cells",
        "Dyskeratotic Tumor Cells",
        "Proliferative Parabasal Cells",
        "OR4F17+ Cells",
        "Detachment",
    ),
    "T-cell": (
        "Cytotoxic T Cells",
        "Naive & Memory T Cells",
        "Regulatory T Cells",
        "Exhausted T Cells",
    ),
    "Myeloid": (
        "Macrophages",
        "Dendritic Cells",
        "Neutrophils",
        "Mast Cells",
    ),
    "B/Plasma": (
        "B Cells",
        "Plasma Cells",
    ),
    "Stromal/Fibro/Muscle": (
        "Stroma & Smooth Muscle",
        "Interstitial Fibroblasts",
        "Cancer Associated Fibroblasts",
        "Smooth Muscle",
    ),
    "Vascular/Endocervical": (
        "Endothelial Cells",
        "Endocervical Columnar Cells",
        "Endocervical Ciliated Cells",
    ),
}


def _default_tbc_results_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR


def _default_output_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DEFAULT_ATERA_WTA_CERVICAL_OUTPUT_DIRNAME


def _load_metrics_summary(dataset_root: Path) -> dict[str, Any]:
    metrics_path = dataset_root / "metrics_summary.csv"
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {}
    row = metrics.iloc[0]
    payload: dict[str, Any] = {}
    for key in ("num_cells_detected", "median_transcripts_per_cell", "median_genes_per_cell"):
        if key in row.index:
            payload[key] = int(row[key]) if pd.notna(row[key]) else None
    return payload


def _load_experiment_metadata(dataset_root: Path) -> dict[str, Any]:
    experiment_path = dataset_root / "experiment.xenium"
    if not experiment_path.exists():
        return {}
    with experiment_path.open("r", encoding="utf-8") as handle:
        experiment = json.load(handle)
    return {
        "panel_num_targets_predesigned": int(experiment.get("panel_num_targets_predesigned", 0) or 0),
        "panel_num_targets_custom": int(experiment.get("panel_num_targets_custom", 0) or 0),
        "pixel_size": float(experiment.get("pixel_size", 0.0) or 0.0),
        "panel_name": experiment.get("panel_name"),
        "analysis_sw_version": experiment.get("analysis_sw_version"),
    }


def _resolve_group_column(frame: pd.DataFrame, *candidates: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not resolve any of the required columns: {list(candidates)!r}")


def _load_cervical_cell_groups(dataset_root: str | Path) -> pd.DataFrame:
    path = Path(dataset_root) / DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS
    if not path.exists():
        raise FileNotFoundError(f"Cell-group CSV not found: {path}")

    frame = pd.read_csv(path)
    cell_id_col = _resolve_group_column(frame, "cell_id", "Barcode", "barcode")
    group_col = _resolve_group_column(frame, "group", "Cluster", "cluster")

    normalized = frame.copy()
    normalized["cell_id"] = normalized[cell_id_col].astype(str)
    normalized["group"] = normalized[group_col].astype(str)
    if "cluster" not in normalized.columns:
        normalized["cluster"] = normalized["group"]
    if "color" not in normalized.columns:
        normalized["color"] = ""
    normalized = normalized.dropna(subset=["cell_id", "group"]).drop_duplicates(subset=["cell_id"], keep="first")
    return normalized.reset_index(drop=True)


def build_atera_wta_cervical_bio6_structures(
    cell_groups: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if cell_groups is not None:
        observed = set(cell_groups["group"].astype(str))
        expected = {
            group_name
            for cluster_names in DEFAULT_ATERA_WTA_CERVICAL_BIO6_GROUPS.values()
            for group_name in cluster_names
        }
        missing = sorted(expected.difference(observed))
        if missing:
            raise ValueError(
                "The cervical cell-group table is missing required group labels for the 6-structure contour merge: "
                + ", ".join(missing)
            )

    structures: list[dict[str, Any]] = []
    for structure_id, (structure_name, cluster_ids) in enumerate(
        DEFAULT_ATERA_WTA_CERVICAL_BIO6_GROUPS.items(),
        start=1,
    ):
        structures.append(
            {
                "structure_name": structure_name,
                "structure_id": structure_id,
                "cluster_ids": list(cluster_ids),
            }
        )
    return structures


def _load_atera_wta_cervical_adata(dataset_root: str | Path):
    return read_xenium(
        str(dataset_root),
        as_="anndata",
        prefer="h5",
        include_transcripts=False,
        include_boundaries=False,
        include_images=False,
        clusters_relpath=DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS,
        cluster_column_name="cluster",
        cells_parquet="cells.parquet",
    )


def _load_atera_wta_cervical_sdata(dataset_root: str | Path) -> XeniumSData:
    return read_xenium(
        str(dataset_root),
        as_="sdata",
        prefer="h5",
        include_transcripts=True,
        include_boundaries=True,
        include_images=True,
        stream_transcripts=True,
        clusters_relpath=DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS,
        cluster_column_name="cluster",
        cells_parquet="cells.parquet",
    )


def _resolve_he_pixel_size_um(sdata: XeniumSData) -> float:
    he_image = sdata.images.get("he")
    if he_image is not None and he_image.pixel_size_um is not None:
        return float(he_image.pixel_size_um)
    return DEFAULT_ATERA_WTA_CERVICAL_PIXEL_SIZE_UM


def _collect_existing_file_payload(base_dir: Path, mapping: dict[str, str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, relative_name in mapping.items():
        path = base_dir / relative_name
        if path.exists():
            payload[key] = str(path)
    return payload


def _summarize_density_table(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "column_count": int(frame.shape[1]),
            "features": [],
        }

    feature_column = None
    for candidate in ("feature_values", "feature_value", "gene_name", "feature"):
        if candidate in frame.columns:
            feature_column = candidate
            break

    summary: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "features": [],
    }
    if feature_column is not None:
        summary["features"] = sorted(pd.unique(frame[feature_column].astype(str)).tolist())
    if "contour_id" in frame.columns:
        summary["n_contours"] = int(frame["contour_id"].astype(str).nunique())
    if {"ring_start_um", "ring_end_um"}.issubset(frame.columns):
        summary["n_rings"] = int(frame.loc[:, ["ring_start_um", "ring_end_um"]].drop_duplicates().shape[0])
    if "signed_distance_um" in frame.columns:
        summary["n_distance_points"] = int(frame["signed_distance_um"].nunique())
    return summary


def _contour_structure_labels(frame: pd.DataFrame) -> list[str]:
    for column in ("assigned_structure", "classification_name", "name"):
        if column in frame.columns:
            return sorted(pd.unique(frame[column].dropna().astype(str)).tolist())
    return []


def build_serializable_cervical_end_to_end_summary(study: dict[str, Any]) -> dict[str, Any]:
    adata = study["adata"]
    contour_key = str(study["contour_key"])
    expanded_contour_key = str(study["expanded_contour_key"])
    contour_frame = study["sdata"].shapes.get(contour_key, pd.DataFrame())
    expanded_frame = study["sdata"].shapes.get(expanded_contour_key, pd.DataFrame())
    cluster_count = int(adata.obs["cluster"].astype(str).nunique()) if "cluster" in adata.obs.columns else 0

    payload = {
        "sample_id": study["sample_id"],
        "dataset_root": study["dataset_root"],
        "output_root": study["output_root"],
        "tbc_results": study["tbc_results"],
        "n_cells": int(adata.n_obs),
        "n_rna_features": int(adata.n_vars),
        "cluster_count": cluster_count,
        "metrics_summary": _load_metrics_summary(Path(study["dataset_root"])),
        "experiment_metadata": _load_experiment_metadata(Path(study["dataset_root"])),
        "contour_key": contour_key,
        "expanded_contour_key": expanded_contour_key,
        "contour_structure_count": int(len(study["bio6_structures"])),
        "contour_structures": [
            {
                "structure_name": str(entry["structure_name"]),
                "structure_id": int(entry["structure_id"]),
                "cluster_ids": [str(value) for value in entry["cluster_ids"]],
            }
            for entry in study["bio6_structures"]
        ],
        "contour_polygon_count": (
            int(contour_frame["contour_id"].astype(str).nunique())
            if "contour_id" in contour_frame.columns
            else 0
        ),
        "expanded_contour_polygon_count": (
            int(expanded_frame["contour_id"].astype(str).nunique())
            if "contour_id" in expanded_frame.columns
            else 0
        ),
        "assigned_structures": _contour_structure_labels(contour_frame),
        "cci_panel": DEFAULT_ATERA_WTA_CERVICAL_CCI_PANEL.to_dict(orient="records"),
        "pathway_panel": [
            {"pathway": pathway, "genes": list(genes)}
            for pathway, genes in DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL.items()
        ],
        "marker_panel": list(DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL),
        "ring_density_summary": _summarize_density_table(study["ring_density"]),
        "smooth_density_summary": _summarize_density_table(study["smooth_density"]),
        "multimodal_sample_summary": study["multimodal"].get("sample_summary", {}),
        "runtime_seconds": float(study["runtime_seconds"]),
        "files": study["files"],
    }
    return payload


def render_atera_wta_cervical_end_to_end_report(payload: dict[str, Any]) -> str:
    multimodal_summary = payload.get("multimodal_sample_summary", {})
    contour_key = payload.get("contour_key", DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY)
    lines = [
        "# Atera WTA Cervical End-to-End Reproducibility Bundle",
        "",
        f"Sample ID: `{payload['sample_id']}`",
        f"Dataset root: `{payload['dataset_root']}`",
        f"Output root: `{payload['output_root']}`",
        f"t_and_c / StructureMap anchor source: `{payload['tbc_results']}`",
        "",
        "## Core Summary",
        "",
        f"- Cells loaded: `{payload['n_cells']}`",
        f"- RNA features loaded: `{payload['n_rna_features']}`",
        f"- Cluster count: `{payload['cluster_count']}`",
        f"- Contour structures: `{payload['contour_structure_count']}`",
        f"- Imported contour polygons: `{payload['contour_polygon_count']}`",
        f"- Expanded contour polygons: `{payload['expanded_contour_polygon_count']}`",
        f"- median_transcripts_per_cell: `{payload['metrics_summary'].get('median_transcripts_per_cell')}`",
        f"- Runtime (s): `{payload['runtime_seconds']:.2f}`",
        "",
        "## sfplot Anchors",
        "",
        f"- StructureMap / t_and_c directory: `{payload['tbc_results']}`",
        "",
        "## Contour Bio6 Structures",
        "",
    ]

    for structure in payload["contour_structures"]:
        lines.append(
            f"- `{structure['structure_name']}`: `{', '.join(structure['cluster_ids'])}`"
        )

    lines.extend(
        [
            "",
            "## Density Profiling",
            "",
            f"- Ring density rows: `{payload['ring_density_summary'].get('row_count', 0)}`",
                f"- Ring density features: `{', '.join(payload['ring_density_summary'].get('features', []))}`",
                f"- Smooth density rows: `{payload['smooth_density_summary'].get('row_count', 0)}`",
                f"- Smooth density features: `{', '.join(payload['smooth_density_summary'].get('features', []))}`",
                "",
                "## Multimodal Contour Ecology",
                "",
                f"- Contour layer: `{multimodal_summary.get('contour_key', contour_key)}`",
                f"- Contours analysed: `{multimodal_summary.get('n_contours')}`",
                f"- Ecotypes discovered: `{multimodal_summary.get('n_ecotypes')}`",
            "",
            "## Fixed Output Files",
            "",
        ]
    )

    for key, value in payload["files"].items():
        if isinstance(value, list):
            lines.append(f"- `{key}`: `{len(value)}` file(s)")
        else:
            lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    return "\n".join(lines)


def run_atera_wta_cervical_end_to_end(
    *,
    dataset_root: str = DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH,
    tbc_results: str | None = None,
    output_root: str | None = None,
    sample_id: str = DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID,
    sfplot_root: str | None = None,
    histoseg_root: str = DEFAULT_ATERA_WTA_CERVICAL_HISTOSEG_ROOT,
    export_figures: bool = True,
    write_sdata_copy: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else _default_output_root(dataset_root_path).resolve()
    )
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    topology_dir = resolved_output_root / DEFAULT_ATERA_WTA_CERVICAL_TOPOLOGY_SUBDIR
    contour_output_dir = resolved_output_root / DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_SUBDIR
    density_dir = resolved_output_root / DEFAULT_ATERA_WTA_CERVICAL_DENSITY_SUBDIR
    multimodal_dir = resolved_output_root / DEFAULT_ATERA_WTA_CERVICAL_MULTIMODAL_SUBDIR
    sdata_output_path = resolved_output_root / DEFAULT_ATERA_WTA_CERVICAL_SDATA_NAME

    group_df = _load_cervical_cell_groups(dataset_root_path)
    bio6_structures = build_atera_wta_cervical_bio6_structures(group_df)

    tbc_run: dict[str, Any] | None = None
    if tbc_results is None:
        tbc_run = run_sfplot_tbc_table_bundle(
            dataset_root_path,
            sample_name=DEFAULT_ATERA_WTA_CERVICAL_SFPLOT_SAMPLE_NAME,
            output_folder=dataset_root_path / DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR,
            coph_method="average",
            n_jobs=8,
            maxtasks=50,
            df=group_df.loc[:, ["cell_id", "group"]].copy(),
            gene_batch_size=128,
            sfplot_root=sfplot_root,
        )
        resolved_tbc_results = Path(tbc_run["output_dir"]).expanduser().resolve()
    else:
        resolved_tbc_results = Path(tbc_results).expanduser().resolve()
    if not resolved_tbc_results.exists():
        raise FileNotFoundError(f"Resolved t_and_c / StructureMap path does not exist: {resolved_tbc_results}")

    adata = _load_atera_wta_cervical_adata(dataset_root_path)

    cci_result = cci_topology_analysis(
        adata=adata,
        interaction_pairs=DEFAULT_ATERA_WTA_CERVICAL_CCI_PANEL,
        output_dir=topology_dir,
        tbc_results=resolved_tbc_results,
        cluster_col="cluster",
        cell_id_col="cell_id",
        x_col="x",
        y_col="y",
        anchor_mode="precomputed",
        top_n_pairs=len(DEFAULT_ATERA_WTA_CERVICAL_CCI_PANEL),
        min_cross_edges=50,
        export_figures=export_figures,
    )

    pathway_result = pathway_topology_analysis(
        adata=adata,
        pathway_definitions=DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL,
        output_dir=topology_dir,
        tbc_results=resolved_tbc_results,
        cluster_col="cluster",
        cell_id_col="cell_id",
        x_col="x",
        y_col="y",
        anchor_mode="precomputed",
        pathway_modes=("gene_topology_aggregate", "activity_point_cloud"),
        primary_pathway_mode="gene_topology_aggregate",
        pathway_aggregate="weighted_median",
        scoring_method="weighted_sum",
        activity_threshold_schedule=(0.95, 0.90, 0.80, 0.70, 0.60, 0.50),
        min_activity_cells=50,
        export_figures=export_figures,
    )

    sdata = _load_atera_wta_cervical_sdata(dataset_root_path)
    contour_artifacts = generate_xenium_explorer_annotations(
        dataset_root_path,
        structures=bio6_structures,
        output_relpath=contour_output_dir,
        clusters_relpath=DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS,
        histoseg_root=histoseg_root,
        barcode_col="cell_id",
        cluster_col="group",
    )

    add_contours_from_geojson(
        sdata,
        contour_artifacts["geojson"],
        key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        id_key="name",
        pixel_size_um=_resolve_he_pixel_size_um(sdata),
        extract_he_patches=True,
    )
    expand_contours(
        sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        distance=30.0,
        mode="voronoi",
        output_key=DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY,
    )

    density_dir.mkdir(parents=True, exist_ok=True)
    ring_density_table = ring_density(
        sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        target="transcripts",
        feature_key="gene_name",
        feature_values=list(DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL),
        inward=20.0,
        outward=30.0,
        ring_width=5.0,
    )
    ring_density_csv = density_dir / "ring_density_markers.csv"
    ring_density_table.to_csv(ring_density_csv, index=False)

    smooth_density_table = smooth_density_by_distance(
        sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        target="transcripts",
        feature_key="gene_name",
        feature_values=list(DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL),
        inward=20.0,
        outward=30.0,
        bandwidth=5.0,
    )
    smooth_density_csv = density_dir / "smooth_density_markers.csv"
    smooth_density_table.to_csv(smooth_density_csv, index=False)

    multimodal_result = run_contour_boundary_ecology_pilot(
        sdata,
        contour_key=DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        output_dir=multimodal_dir,
    )

    sdata_write_result: dict[str, Any] | None = None
    if write_sdata_copy:
        sdata_write_result = write_xenium(
            sdata,
            sdata_output_path,
            format="sdata",
            overwrite=True,
        )

    runtime_seconds = time.perf_counter() - start

    files: dict[str, Any] = {}
    if tbc_run is not None:
        for key in (
            "structure_map_pdf",
            "structure_map_table",
            "t_and_c_result",
        ):
            if key in tbc_run:
                files[key] = str(tbc_run[key])
    files.update(cci_result.get("files", {}))
    files.update(pathway_result.get("files", {}))
    files.update(
        {
            "contour_geojson": str(contour_artifacts["geojson"]),
            "contour_csv": str(contour_artifacts["csv"]),
            "contour_summary_csv": str(contour_artifacts["summary"]),
            "contour_partition_table": str(contour_artifacts["partition_table"]),
            "contour_structure_count_csv": str(contour_artifacts["structure_count_csv"]),
            "contour_metrics_json": str(contour_artifacts["metrics_json"]),
            "ring_density_csv": str(ring_density_csv),
            "smooth_density_csv": str(smooth_density_csv),
            "multimodal_artifact_dir": str(multimodal_dir),
        }
    )
    if contour_artifacts.get("preview_png"):
        files["contour_preview_png"] = str(contour_artifacts["preview_png"])
    files.update(
        _collect_existing_file_payload(
            multimodal_dir,
            {
                "multimodal_summary_json": "summary.json",
                "multimodal_report_md": "report.md",
                "multimodal_exemplar_montage": "exemplar_montage.png",
                "multimodal_contour_features_csv": "contour_features.csv",
                "multimodal_program_scores_csv": "program_scores.csv",
            },
        )
    )
    if sdata_write_result is not None:
        files["contour_enriched_sdata"] = str(sdata_write_result["output_path"])

    study = {
        "sample_id": sample_id,
        "dataset_root": str(dataset_root_path),
        "output_root": str(resolved_output_root),
        "tbc": tbc_run,
        "tbc_results": str(resolved_tbc_results),
        "adata": adata,
        "sdata": sdata,
        "cci": cci_result,
        "pathway": pathway_result,
        "contour_generation": contour_artifacts,
        "ring_density": ring_density_table,
        "smooth_density": smooth_density_table,
        "multimodal": multimodal_result,
        "bio6_structures": bio6_structures,
        "contour_key": DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY,
        "expanded_contour_key": DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY,
        "files": files,
        "runtime_seconds": runtime_seconds,
    }

    payload = build_serializable_cervical_end_to_end_summary(study)
    summary_path = resolved_output_root / "summary.json"
    report_path = resolved_output_root / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_atera_wta_cervical_end_to_end_report(payload), encoding="utf-8")

    study["files"]["summary_json"] = str(summary_path)
    study["files"]["report_md"] = str(report_path)
    payload = build_serializable_cervical_end_to_end_summary(study)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_atera_wta_cervical_end_to_end_report(payload), encoding="utf-8")
    study["payload"] = payload
    return study


__all__ = [
    "DEFAULT_ATERA_WTA_CERVICAL_CELL_GROUPS",
    "DEFAULT_ATERA_WTA_CERVICAL_CONTOUR_KEY",
    "DEFAULT_ATERA_WTA_CERVICAL_DATASET_PATH",
    "DEFAULT_ATERA_WTA_CERVICAL_EXPANDED_CONTOUR_KEY",
    "DEFAULT_ATERA_WTA_CERVICAL_CCI_PANEL",
    "DEFAULT_ATERA_WTA_CERVICAL_MARKER_PANEL",
    "DEFAULT_ATERA_WTA_CERVICAL_PATHWAY_PANEL",
    "DEFAULT_ATERA_WTA_CERVICAL_SAMPLE_ID",
    "DEFAULT_ATERA_WTA_CERVICAL_TBC_SUBDIR",
    "build_atera_wta_cervical_bio6_structures",
    "build_serializable_cervical_end_to_end_summary",
    "render_atera_wta_cervical_end_to_end_report",
    "run_atera_wta_cervical_end_to_end",
]
