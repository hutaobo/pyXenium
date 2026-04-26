from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import pi
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from shapely import intersects, points
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from pyXenium.io.sdata_model import XeniumImage, XeniumSData

from ._analysis import _cell_centroid_frame, _prepare_contours, smooth_density_by_distance
from .loading import (
    _clipped_image_bbox,
    _crop_image_level,
    _geometry_um_to_image_xy,
    _polygon_mask_for_bbox,
)

__all__ = ["DEFAULT_CONTOUR_LR_PAIRS", "DEFAULT_CONTOUR_PATHWAYS", "build_contour_feature_table"]

_DEFAULT_NEIGHBOR_K = 6
_DEFAULT_IMAGE_KEY = "he"
_DEFAULT_RIM_EPSILON = 1e-6
_DEFAULT_GRADIENT_BANDWIDTH_FRACTION = 0.25
_DEFAULT_GRADIENT_MIN_BANDWIDTH = 2.0
_DEFAULT_GRADIENT_GRID_FRACTION = 0.5

DEFAULT_CONTOUR_PATHWAYS: dict[str, list[str]] = {
    "immune_activation": ["CD3D", "CD3E", "TRAC", "PRF1", "GZMB", "CXCL13", "MS4A1"],
    "myeloid_activation": ["CD68", "CD163", "LST1", "TYROBP", "SPP1", "HLA-DRA"],
    "vascular_stromal": ["PECAM1", "EMCN", "RGS5", "ACTA2", "COL4A1", "COL4A2", "VEGFA"],
    "emt_invasion": ["VIM", "ACTA2", "TAGLN", "COL1A1", "COL1A2", "FN1", "MMP11"],
    "stromal_matrix": ["COL1A1", "COL1A2", "COL3A1", "FAP", "ACTA2", "TAGLN"],
    "tls_activation": ["CXCL13", "MS4A1", "CD79A", "JCHAIN", "TRAC", "CD3D"],
    "hypoxia_necrosis": ["CA9", "ENO1", "LDHA", "HILPDA", "SLC2A1"],
}

DEFAULT_CONTOUR_LR_PAIRS: dict[str, tuple[str, str]] = {
    "spp1_cd44": ("SPP1", "CD44"),
    "cxcl13_cxcr5": ("CXCL13", "CXCR5"),
    "cxcl12_cxcr4": ("CXCL12", "CXCR4"),
    "tgfb1_tgfbr2": ("TGFB1", "TGFBR2"),
    "vegfa_kdr": ("VEGFA", "KDR"),
}

_DEFAULT_GRADIENT_GENESETS: dict[str, list[str]] = {
    "immune": DEFAULT_CONTOUR_PATHWAYS["immune_activation"],
    "myeloid": DEFAULT_CONTOUR_PATHWAYS["myeloid_activation"],
    "vascular": DEFAULT_CONTOUR_PATHWAYS["vascular_stromal"],
    "emt": DEFAULT_CONTOUR_PATHWAYS["emt_invasion"],
    "hypoxia": DEFAULT_CONTOUR_PATHWAYS["hypoxia_necrosis"],
}


def build_contour_feature_table(
    sdata: XeniumSData,
    *,
    contour_key: str,
    he_image_key: str = _DEFAULT_IMAGE_KEY,
    inner_rim_um: float = 20.0,
    outer_rim_um: float = 30.0,
    include_pathomics: bool = True,
    embedding_backend: Any = None,
    precomputed_edge_gradients: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Build a contour-centric multimodal feature table.

    The returned payload keeps a wide contour-level table in ``contour_features`` and
    exposes supporting matrices for pseudobulk RNA/protein, pathway activity, ligand-
    receptor summaries, zone-level composition, and signed-distance gradients.
    """

    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance.")

    contour_table = _prepare_contours(sdata=sdata, contour_key=contour_key, contour_query=None)
    contour_table = contour_table.sort_values("contour_id", kind="stable").reset_index(drop=True)
    sample_id = _resolve_sample_id(sdata)
    adata = sdata.table

    if include_pathomics:
        if he_image_key not in sdata.images:
            raise ValueError(
                "Pathomics extraction requires a loaded whole-slide H&E image. "
                f"`sdata.images[{he_image_key!r}]` was not found."
            )
        whole_image = sdata.images[he_image_key]
        if whole_image.image_to_xenium_affine is None:
            raise ValueError(
                f"`sdata.images[{he_image_key!r}]` is missing image alignment metadata."
            )
        if whole_image.pixel_size_um is None:
            raise ValueError(f"`sdata.images[{he_image_key!r}]` is missing pixel_size_um.")
    else:
        whole_image = None

    if include_pathomics:
        if contour_key not in sdata.contour_images:
            raise ValueError(
                "Pathomics contour feature extraction requires contour H&E patches. "
                f"`sdata.contour_images[{contour_key!r}]` was not found."
            )
        contour_patches = sdata.contour_images[contour_key]
    else:
        contour_patches = {}

    context = _prepare_multimodal_context(adata)
    cell_table = context["cell_table"]
    cell_points = context["cell_points"]
    cell_xy = context["cell_xy"]
    state_categories = context["state_categories"]
    niche_categories = context["niche_categories"]
    expression = context["expression"]
    protein = context["protein"]
    pathway_activity = context["pathway_activity"]
    selected_genes = context["selected_genes"]
    selected_proteins = protein.columns.astype(str).tolist()

    zone_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    rna_rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []

    context_features = _compute_contour_context_features(
        contour_table=contour_table,
        sample_id=sample_id,
        contour_key=contour_key,
        neighbor_k=_DEFAULT_NEIGHBOR_K,
        outer_rim_um=outer_rim_um,
    )
    if precomputed_edge_gradients is None:
        edge_gradients = _compute_edge_gradients(
            sdata=sdata,
            contour_key=contour_key,
            inward=inner_rim_um,
            outward=outer_rim_um,
        )
    else:
        edge_gradients = _coerce_edge_gradients(precomputed_edge_gradients, contour_key=contour_key)

    for _, contour_row in contour_table.iterrows():
        contour_id = str(contour_row["contour_id"])
        contour_geometry = contour_row["geometry"]
        zones = _build_contour_zones(
            contour_geometry,
            inner_rim_um=float(inner_rim_um),
            outer_rim_um=float(outer_rim_um),
        )
        row = {
            "sample_id": sample_id,
            "contour_key": contour_key,
            "contour_id": contour_id,
        }
        for metadata_column in ("assigned_structure", "classification_name", "annotation_source", "structure_id"):
            if metadata_column in contour_row.index:
                row[metadata_column] = contour_row[metadata_column]
        row.update(_geometry_features(contour_geometry))
        row.update(context_features.get(contour_id, {}))

        zone_memberships = {
            zone_name: _geometry_membership_mask(zone_geometry, cell_points, point_xy=cell_xy)
            for zone_name, zone_geometry in zones.items()
        }
        zone_area = {
            zone_name: float(zone_geometry.area) if zone_geometry is not None else 0.0
            for zone_name, zone_geometry in zones.items()
        }

        for zone_name, membership in zone_memberships.items():
            zone_row = {
                "sample_id": sample_id,
                "contour_key": contour_key,
                "contour_id": contour_id,
                "zone": zone_name,
                "area_um2": zone_area[zone_name],
                "n_cells": int(membership.sum()),
            }
            zone_row.update(
                _composition_features(
                    cell_table.loc[membership],
                    state_categories=state_categories,
                    niche_categories=niche_categories,
                )
            )
            zone_rows.append(zone_row)
            row.update({f"omics__{zone_name}__{key}": value for key, value in zone_row.items() if key not in {"sample_id", "contour_key", "contour_id", "zone"}})

            if membership.any():
                row.update(
                    _mean_prefixed(
                        expression.loc[membership, selected_genes],
                        prefix=f"rna__{zone_name}",
                    )
                )
                row.update(
                    _mean_prefixed(
                        pathway_activity.loc[membership],
                        prefix=f"pathway__{zone_name}",
                    )
                )
                if not protein.empty:
                    row.update(
                        _mean_prefixed(
                            protein.loc[membership],
                            prefix=f"protein__{zone_name}",
                        )
                    )

        row.update(_edge_contrast_features(row))

        if include_pathomics and whole_image is not None:
            if contour_id not in contour_patches:
                raise KeyError(
                    f"`sdata.contour_images[{contour_key!r}]` does not contain contour {contour_id!r}."
                )
            contour_patch = contour_patches[contour_id]
            pathomics, embeddings = _extract_image_view_features(
                contour_geometry=contour_geometry,
                contour_patch=contour_patch,
                whole_image=whole_image,
                inner_rim_um=float(inner_rim_um),
                outer_rim_um=float(outer_rim_um),
                embedding_backend=embedding_backend,
                contour_key=contour_key,
                contour_id=contour_id,
            )
            row.update(pathomics)
            if embeddings:
                embedding_row = {
                    "sample_id": sample_id,
                    "contour_key": contour_key,
                    "contour_id": contour_id,
                }
                embedding_row.update(embeddings)
                embedding_rows.append(embedding_row)

        whole_mask = zone_memberships["whole"]
        rna_row = {
            "sample_id": sample_id,
            "contour_key": contour_key,
            "contour_id": contour_id,
        }
        protein_row = dict(rna_row)
        pathway_row = dict(rna_row)
        if whole_mask.any():
            rna_row.update(expression.loc[whole_mask, selected_genes].mean(axis=0).to_dict())
            if not protein.empty:
                protein_row.update(protein.loc[whole_mask].mean(axis=0).to_dict())
            pathway_row.update(pathway_activity.loc[whole_mask].mean(axis=0).to_dict())
        else:
            rna_row.update({gene: 0.0 for gene in selected_genes})
            protein_row.update({name: 0.0 for name in selected_proteins})
            pathway_row.update({name: 0.0 for name in pathway_activity.columns.astype(str)})
        rna_rows.append(rna_row)
        protein_rows.append(protein_row)
        pathway_rows.append(pathway_row)

        lr_row = {
            "sample_id": sample_id,
            "contour_key": contour_key,
            "contour_id": contour_id,
        }
        lr_row.update(
            _ligand_receptor_features(
                expression=expression,
                inner_mask=zone_memberships["inner_rim"],
                outer_mask=zone_memberships["outer_rim"],
            )
        )
        lr_rows.append(lr_row)
        row.update({f"lr__{key}": value for key, value in lr_row.items() if key not in {"sample_id", "contour_key", "contour_id"}})

        contour_gradients = edge_gradients.loc[edge_gradients["contour_id"] == contour_id]
        if not contour_gradients.empty:
            for _, gradient_row in contour_gradients.iterrows():
                gene_set = str(gradient_row["gradient_key"])
                row[f"gradient__{gene_set}__outer_minus_inner"] = float(gradient_row["outer_minus_inner"])
                row[f"gradient__{gene_set}__boundary_peak"] = float(gradient_row["boundary_peak"])
                row[f"gradient__{gene_set}__center_of_mass"] = float(gradient_row["center_of_mass"])
                row[f"gradient__{gene_set}__outer_mean"] = float(gradient_row["outer_mean"])
                row[f"gradient__{gene_set}__inner_mean"] = float(gradient_row["inner_mean"])

        row.update(_edge_contrast_features(row))
        feature_rows.append(row)

    contour_features = pd.DataFrame(feature_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
    zone_summary = pd.DataFrame(zone_rows).sort_values(["contour_id", "zone"], kind="stable").reset_index(drop=True)
    rna_pseudobulk = pd.DataFrame(rna_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
    protein_summary = pd.DataFrame(protein_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
    pathway_summary = pd.DataFrame(pathway_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
    ligand_receptor_summary = pd.DataFrame(lr_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
    embedding_summary = (
        pd.DataFrame(embedding_rows).sort_values("contour_id", kind="stable").reset_index(drop=True)
        if embedding_rows
        else pd.DataFrame(columns=["sample_id", "contour_key", "contour_id"])
    )

    return {
        "sample_id": sample_id,
        "contour_key": contour_key,
        "inner_rim_um": float(inner_rim_um),
        "outer_rim_um": float(outer_rim_um),
        "contour_features": contour_features,
        "zone_summary": zone_summary,
        "rna_pseudobulk": rna_pseudobulk,
        "protein_summary": protein_summary,
        "pathway_activity": pathway_summary,
        "ligand_receptor_summary": ligand_receptor_summary,
        "edge_gradients": edge_gradients,
        "embedding_summary": embedding_summary,
        "available_states": state_categories,
        "available_niches": niche_categories,
        "feature_columns": {
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
        },
        "context": {
            "multimodal_context": context["context_summary"],
            "used_pathomics": bool(include_pathomics),
            "used_embeddings": bool(include_pathomics and embedding_backend is not None),
            "used_precomputed_edge_gradients": bool(precomputed_edge_gradients is not None),
        },
    }


def _resolve_sample_id(sdata: XeniumSData) -> str:
    for key in ("sample_id", "sample", "dataset_id"):
        value = sdata.metadata.get(key)
        if value:
            return str(value)
    source_path = sdata.metadata.get("source_path")
    if source_path:
        return str(source_path).rstrip("/\\").split("\\")[-1].split("/")[-1]
    return "sample_0"


def _prepare_multimodal_context(adata) -> dict[str, Any]:
    if "spatial" not in adata.obsm:
        raise KeyError("Contour ecology workflow requires `adata.obsm['spatial']`.")

    context_messages: list[str] = []
    if "joint_cell_state" not in adata.obs.columns:
        if _has_nonempty_protein(adata):
            try:
                from pyXenium.multimodal.immune_resistance import annotate_joint_cell_states

                annotate_joint_cell_states(adata)
                context_messages.append("annotated_joint_cell_states")
            except Exception as exc:
                adata.obs["joint_cell_state"] = "unassigned"
                adata.obs["joint_cell_class"] = "unassigned"
                context_messages.append(f"joint_cell_state_fallback:{type(exc).__name__}")
        else:
            fallback = (
                adata.obs["cluster"].astype(str)
                if "cluster" in adata.obs.columns
                else pd.Series("unassigned", index=adata.obs_names, dtype="object")
            )
            adata.obs["joint_cell_state"] = fallback
            adata.obs["joint_cell_class"] = fallback
            context_messages.append("joint_cell_state_from_cluster")

    if "spatial_niche" not in adata.obs.columns:
        if _has_nonempty_protein(adata):
            try:
                from pyXenium.multimodal.immune_resistance import build_spatial_niches

                build_spatial_niches(adata)
                context_messages.append("built_spatial_niches")
            except Exception as exc:
                adata.obs["spatial_niche"] = "mixed_low_signal"
                context_messages.append(f"spatial_niche_fallback:{type(exc).__name__}")
        else:
            adata.obs["spatial_niche"] = "mixed_low_signal"
            context_messages.append("spatial_niche_fallback_no_protein")

    if _has_nonempty_protein(adata):
        discordance_columns = [column for column in adata.obs.columns if str(column).startswith("discordance__")]
        if not discordance_columns:
            try:
                from pyXenium.multimodal.immune_resistance import compute_rna_protein_discordance

                compute_rna_protein_discordance(adata)
                context_messages.append("computed_discordance")
            except Exception as exc:
                context_messages.append(f"discordance_skipped:{type(exc).__name__}")
    else:
        context_messages.append("discordance_skipped_no_protein")

    cell_obs = adata.obs.copy()
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("`adata.obsm['spatial']` must have shape (n_cells, >=2).")
    cell_table = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "x": spatial[:, 0],
            "y": spatial[:, 1],
            "joint_cell_state": cell_obs["joint_cell_state"].astype(str).to_numpy(),
            "spatial_niche": cell_obs["spatial_niche"].astype(str).to_numpy(),
        },
        index=adata.obs_names,
    )
    cell_points = np.asarray(points(cell_table["x"].to_numpy(dtype=float), cell_table["y"].to_numpy(dtype=float)), dtype=object)

    selected_genes = _resolve_selected_genes(adata)
    expression = _build_expression_frame(adata, selected_genes)
    pathway_activity = _build_pathway_activity(expression)
    protein = _protein_frame(adata)

    return {
        "cell_table": cell_table,
        "cell_points": cell_points,
        "cell_xy": spatial[:, :2].copy(),
        "state_categories": sorted(pd.unique(cell_table["joint_cell_state"].astype(str)).tolist()),
        "niche_categories": sorted(pd.unique(cell_table["spatial_niche"].astype(str)).tolist()),
        "expression": expression,
        "protein": protein,
        "pathway_activity": pathway_activity,
        "selected_genes": selected_genes,
        "context_summary": context_messages,
    }


def _has_nonempty_protein(adata) -> bool:
    protein = adata.obsm.get("protein")
    if protein is None:
        return False
    if isinstance(protein, pd.DataFrame):
        return protein.shape[1] > 0
    return np.asarray(protein).ndim == 2 and np.asarray(protein).shape[1] > 0


def _resolve_selected_genes(adata) -> list[str]:
    gene_names = (
        adata.var["name"].astype(str).tolist()
        if "name" in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )
    available = {name.casefold(): name for name in gene_names}
    selected: list[str] = []
    for genes in DEFAULT_CONTOUR_PATHWAYS.values():
        for gene in genes:
            resolved = available.get(gene.casefold())
            if resolved is not None and resolved not in selected:
                selected.append(resolved)
    for ligand, receptor in DEFAULT_CONTOUR_LR_PAIRS.values():
        for gene in (ligand, receptor):
            resolved = available.get(gene.casefold())
            if resolved is not None and resolved not in selected:
                selected.append(resolved)
    if not selected:
        selected = gene_names[: min(len(gene_names), 32)]
    return selected


def _build_expression_frame(adata, selected_genes: Sequence[str]) -> pd.DataFrame:
    if not selected_genes:
        return pd.DataFrame(index=adata.obs_names)
    gene_names = (
        adata.var["name"].astype(str).tolist()
        if "name" in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )
    lookup: dict[str, list[int]] = {}
    for index, gene in enumerate(gene_names):
        lookup.setdefault(str(gene).casefold(), []).append(index)

    matrix = adata.layers["rna"] if "rna" in adata.layers else adata.X
    if sparse.issparse(matrix):
        rna = matrix.tocsr()
    else:
        rna = sparse.csr_matrix(np.asarray(matrix, dtype=float))

    library_size = np.asarray(rna.sum(axis=1)).ravel().astype(float)
    library_size[library_size <= 0] = 1.0

    columns: dict[str, np.ndarray] = {}
    for gene in selected_genes:
        indices = lookup.get(str(gene).casefold(), [])
        if not indices:
            columns[str(gene)] = np.zeros(adata.n_obs, dtype=float)
            continue
        subset = rna[:, indices]
        values = np.asarray(subset.mean(axis=1)).ravel().astype(float)
        normalized = np.log1p((values / library_size) * 1e4)
        columns[str(gene)] = normalized
    return pd.DataFrame(columns, index=adata.obs_names)


def _protein_frame(adata) -> pd.DataFrame:
    protein = adata.obsm.get("protein")
    if protein is None:
        return pd.DataFrame(index=adata.obs_names)
    if isinstance(protein, pd.DataFrame):
        return protein.copy()
    array = np.asarray(protein, dtype=float)
    columns = [f"protein_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, index=adata.obs_names, columns=columns)


def _build_pathway_activity(expression: pd.DataFrame) -> pd.DataFrame:
    if expression.empty:
        return pd.DataFrame(index=expression.index)
    from pyXenium.pathway._analysis import compute_pathway_activity_matrix

    return compute_pathway_activity_matrix(
        expression,
        DEFAULT_CONTOUR_PATHWAYS,
        method="weighted_sum",
        normalize=True,
    )


def _compute_contour_context_features(
    *,
    contour_table: pd.DataFrame,
    sample_id: str,
    contour_key: str,
    neighbor_k: int,
    outer_rim_um: float,
) -> dict[str, dict[str, float]]:
    geometries = contour_table["geometry"].tolist()
    centroids = np.asarray(
        [[float(geometry.centroid.x), float(geometry.centroid.y)] for geometry in geometries],
        dtype=float,
    )
    labels = contour_table.get("classification_name", pd.Series("unknown", index=contour_table.index)).astype(str)
    x_span = max(float(centroids[:, 0].max() - centroids[:, 0].min()), 1.0)
    y_span = max(float(centroids[:, 1].max() - centroids[:, 1].min()), 1.0)
    pairwise = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(pairwise, np.inf)

    top_labels = labels.value_counts().head(5).index.astype(str).tolist()
    features: dict[str, dict[str, float]] = {}
    for index, contour_row in contour_table.iterrows():
        contour_id = str(contour_row["contour_id"])
        current = pairwise[index]
        finite = np.isfinite(current)
        n_neighbors = min(int(neighbor_k), int(finite.sum()))
        if n_neighbors > 0:
            neighbor_idx = np.argsort(current)[:n_neighbors]
            neighbor_distance = current[neighbor_idx]
        else:
            neighbor_idx = np.asarray([], dtype=int)
            neighbor_distance = np.asarray([], dtype=float)

        contact_degree = 0
        for other_index, other_geometry in enumerate(geometries):
            if other_index == index:
                continue
            if contour_row["geometry"].buffer(float(outer_rim_um)).intersects(other_geometry):
                contact_degree += 1

        row = {
            "sample_id": sample_id,
            "contour_key": contour_key,
            "context__centroid_x_um": float(centroids[index, 0]),
            "context__centroid_y_um": float(centroids[index, 1]),
            "context__slide_x_fraction": float((centroids[index, 0] - centroids[:, 0].min()) / x_span),
            "context__slide_y_fraction": float((centroids[index, 1] - centroids[:, 1].min()) / y_span),
            "context__neighbor_count": float(n_neighbors),
            "context__mean_neighbor_distance_um": float(np.mean(neighbor_distance)) if neighbor_distance.size else float("nan"),
            "context__contact_degree": float(contact_degree),
            "context__neighbor_same_label_fraction": float(np.mean(labels.iloc[neighbor_idx] == labels.iloc[index])) if neighbor_idx.size else float("nan"),
        }
        for label in top_labels:
            row[f"context__neighbor_label_fraction__{_slug(label)}"] = (
                float(np.mean(labels.iloc[neighbor_idx] == label)) if neighbor_idx.size else 0.0
            )
        features[contour_id] = row
    return features


def _compute_edge_gradients(
    *,
    sdata: XeniumSData,
    contour_key: str,
    inward: float,
    outward: float,
) -> pd.DataFrame:
    if "transcripts" not in sdata.points and "transcripts" not in sdata.point_sources:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "contour_key",
                "contour_id",
                "gradient_key",
                "inner_mean",
                "outer_mean",
                "outer_minus_inner",
                "boundary_peak",
                "center_of_mass",
            ]
        )

    rows: list[dict[str, Any]] = []
    bandwidth = max(float(min(inward, outward)) * _DEFAULT_GRADIENT_BANDWIDTH_FRACTION, _DEFAULT_GRADIENT_MIN_BANDWIDTH)
    grid_step = max(bandwidth * _DEFAULT_GRADIENT_GRID_FRACTION, 1.0)
    for gradient_key, genes in _DEFAULT_GRADIENT_GENESETS.items():
        try:
            profile = smooth_density_by_distance(
                sdata,
                contour_key=contour_key,
                target="transcripts",
                feature_values=genes,
                inward=inward,
                outward=outward,
                bandwidth=bandwidth,
                grid_step=grid_step,
            )
        except Exception:
            continue
        if profile.empty:
            continue
        for contour_id, group in profile.groupby("contour_id", sort=False, dropna=False):
            signed = group["signed_distance"].to_numpy(dtype=float)
            density = group["density"].to_numpy(dtype=float)
            inner_mask = (signed < 0.0) & np.isfinite(density)
            outer_mask = (signed > 0.0) & np.isfinite(density)
            positive_density = np.clip(np.nan_to_num(density, nan=0.0), a_min=0.0, a_max=None)
            if positive_density.sum() > 0:
                center_of_mass = float(np.sum(signed * positive_density) / np.sum(positive_density))
            else:
                center_of_mass = float("nan")
            zero_index = int(np.argmin(np.abs(signed))) if signed.size else 0
            rows.append(
                {
                    "contour_key": contour_key,
                    "contour_id": str(contour_id),
                    "gradient_key": gradient_key,
                    "inner_mean": float(np.nanmean(density[inner_mask])) if inner_mask.any() else 0.0,
                    "outer_mean": float(np.nanmean(density[outer_mask])) if outer_mask.any() else 0.0,
                    "outer_minus_inner": (
                        float(np.nanmean(density[outer_mask]) - np.nanmean(density[inner_mask]))
                        if inner_mask.any() and outer_mask.any()
                        else 0.0
                    ),
                    "boundary_peak": float(density[zero_index]) if density.size else 0.0,
                    "center_of_mass": center_of_mass,
                }
            )
    return pd.DataFrame(rows)


def _coerce_edge_gradients(edge_gradients: pd.DataFrame, *, contour_key: str) -> pd.DataFrame:
    required = [
        "contour_key",
        "contour_id",
        "gradient_key",
        "inner_mean",
        "outer_mean",
        "outer_minus_inner",
        "boundary_peak",
        "center_of_mass",
    ]
    frame = edge_gradients.copy()
    if frame.empty:
        return pd.DataFrame(columns=required)
    if "contour_key" not in frame.columns:
        frame["contour_key"] = contour_key
    if "contour_id" not in frame.columns:
        raise KeyError("Precomputed edge gradients must contain a `contour_id` column.")
    if "gradient_key" not in frame.columns:
        frame["gradient_key"] = "precomputed"
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["contour_key"] = frame["contour_key"].astype(str)
    frame["contour_id"] = frame["contour_id"].astype(str)
    frame["gradient_key"] = frame["gradient_key"].astype(str)
    frame = frame.loc[frame["contour_key"] == str(contour_key)].copy()
    for column in ("inner_mean", "outer_mean", "outer_minus_inner", "boundary_peak", "center_of_mass"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def _geometry_features(geometry: BaseGeometry) -> dict[str, float]:
    area = float(geometry.area)
    perimeter = float(geometry.length)
    convex_hull = geometry.convex_hull
    solidity = area / float(convex_hull.area) if convex_hull.area > 0 else float("nan")
    compactness = (4.0 * pi * area / (perimeter**2)) if perimeter > 0 else float("nan")
    boundary_roughness = (perimeter / float(convex_hull.length)) if convex_hull.length > 0 else float("nan")
    minx, miny, maxx, maxy = geometry.bounds
    width = max(float(maxx - minx), 0.0)
    height = max(float(maxy - miny), 0.0)
    bbox_aspect_ratio = width / height if height > 0 else float("nan")
    hole_area = _hole_area(geometry)
    eccentricity = _eccentricity(geometry)
    return {
        "geometry__area_um2": area,
        "geometry__perimeter_um": perimeter,
        "geometry__compactness": compactness,
        "geometry__eccentricity": eccentricity,
        "geometry__hole_burden": (hole_area / area) if area > 0 else 0.0,
        "geometry__boundary_roughness": boundary_roughness,
        "geometry__solidity": solidity,
        "geometry__bbox_width_um": width,
        "geometry__bbox_height_um": height,
        "geometry__bbox_aspect_ratio": bbox_aspect_ratio,
    }


def _hole_area(geometry: BaseGeometry) -> float:
    polygons = _polygon_parts(geometry)
    total = 0.0
    for polygon in polygons:
        for interior in polygon.interiors:
            total += abs(Polygon(interior.coords).area)
    return float(total)


def _eccentricity(geometry: BaseGeometry) -> float:
    polygons = _polygon_parts(geometry)
    coords: list[tuple[float, float]] = []
    for polygon in polygons:
        coords.extend([(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]])
    if len(coords) < 3:
        return 0.0
    array = np.asarray(coords, dtype=float)
    centered = array - array.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values = np.sort(np.linalg.eigvalsh(covariance))
    major = max(float(values[-1]), 0.0)
    minor = max(float(values[0]), 0.0)
    if major <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - minor / major)))


def _build_contour_zones(
    geometry: BaseGeometry,
    *,
    inner_rim_um: float,
    outer_rim_um: float,
) -> dict[str, BaseGeometry]:
    whole = geometry
    core = geometry.buffer(-float(inner_rim_um))
    if core.is_empty:
        core = geometry
    inner_rim = geometry.difference(core)
    outer_buffer = geometry.buffer(float(outer_rim_um))
    outer_rim = outer_buffer.difference(geometry)
    return {
        "whole": whole,
        "core": core,
        "inner_rim": inner_rim,
        "outer_rim": outer_rim,
    }


def _geometry_membership_mask(
    geometry: BaseGeometry,
    point_array: np.ndarray,
    *,
    point_xy: np.ndarray | None = None,
) -> np.ndarray:
    if geometry is None or geometry.is_empty or point_array.size == 0:
        return np.zeros(len(point_array), dtype=bool)
    buffered = geometry.buffer(_DEFAULT_RIM_EPSILON)
    if point_xy is None:
        return np.asarray(intersects(buffered, point_array), dtype=bool)

    xy = np.asarray(point_xy, dtype=float)
    minx, miny, maxx, maxy = buffered.bounds
    candidate_mask = (
        (xy[:, 0] >= float(minx))
        & (xy[:, 0] <= float(maxx))
        & (xy[:, 1] >= float(miny))
        & (xy[:, 1] <= float(maxy))
    )
    if not candidate_mask.any():
        return np.zeros(len(point_array), dtype=bool)
    membership = np.zeros(len(point_array), dtype=bool)
    membership[candidate_mask] = np.asarray(intersects(buffered, point_array[candidate_mask]), dtype=bool)
    return membership


def _composition_features(
    cell_frame: pd.DataFrame,
    *,
    state_categories: Sequence[str],
    niche_categories: Sequence[str],
) -> dict[str, float]:
    n_cells = int(len(cell_frame))
    features = {
        "state_entropy": 0.0,
        "niche_entropy": 0.0,
    }
    if n_cells == 0:
        for state in state_categories:
            features[f"state_fraction__{_slug(state)}"] = 0.0
        for niche in niche_categories:
            features[f"niche_fraction__{_slug(niche)}"] = 0.0
        return features

    state_values = cell_frame["joint_cell_state"].astype(str).value_counts(normalize=True)
    niche_values = cell_frame["spatial_niche"].astype(str).value_counts(normalize=True)
    for state in state_categories:
        features[f"state_fraction__{_slug(state)}"] = float(state_values.get(state, 0.0))
    for niche in niche_categories:
        features[f"niche_fraction__{_slug(niche)}"] = float(niche_values.get(niche, 0.0))
    features["state_entropy"] = _entropy_from_probabilities(state_values.to_numpy(dtype=float))
    features["niche_entropy"] = _entropy_from_probabilities(niche_values.to_numpy(dtype=float))
    return features


def _entropy_from_probabilities(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log2(arr)))


def _mean_prefixed(frame: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if frame.empty:
        return {}
    means = frame.mean(axis=0, numeric_only=True)
    return {f"{prefix}__{_slug(str(column))}": float(value) for column, value in means.items()}


def _edge_contrast_features(feature_row: Mapping[str, Any]) -> dict[str, float]:
    outputs: dict[str, float] = {}
    prefixes = (
        "pathomics",
        "omics",
        "pathway",
        "protein",
        "rna",
    )
    for key, value in feature_row.items():
        text = str(key)
        if not any(text.startswith(f"{prefix}__inner_rim__") for prefix in prefixes):
            continue
        parts = text.split("__")
        if len(parts) < 3:
            continue
        kind = parts[0]
        counterpart = text.replace("__inner_rim__", "__outer_rim__", 1)
        if counterpart not in feature_row:
            continue
        suffix = "__".join(parts[2:])
        outputs[f"edge_contrast__{kind}__{suffix}"] = float(feature_row[counterpart]) - float(value)
    return outputs


def _extract_image_view_features(
    *,
    contour_geometry: BaseGeometry,
    contour_patch: XeniumImage,
    whole_image: XeniumImage,
    inner_rim_um: float,
    outer_rim_um: float,
    embedding_backend: Any,
    contour_key: str,
    contour_id: str,
) -> tuple[dict[str, float], dict[str, float]]:
    pathomics: dict[str, float] = {}
    embeddings: dict[str, float] = {}
    support_geometry = contour_geometry.buffer(float(outer_rim_um))
    support_image_geometry = _geometry_um_to_image_xy(support_geometry, he_image=whole_image)
    bbox = _clipped_image_bbox(
        image_geometry=support_image_geometry,
        image_shape=tuple(int(value) for value in whole_image.levels[0].shape),
        image_axes=whole_image.axes,
        contour_key=contour_key,
        contour_id=contour_id,
    )
    crop = _crop_image_level(whole_image.levels[0], axes=whole_image.axes, bbox=bbox)
    zone_geometries = _build_contour_zones(
        contour_geometry,
        inner_rim_um=inner_rim_um,
        outer_rim_um=outer_rim_um,
    )
    masks: dict[str, np.ndarray] = {}
    for zone_name, zone_geometry in zone_geometries.items():
        if zone_geometry is None or zone_geometry.is_empty:
            y_index = whole_image.axes.index("y")
            x_index = whole_image.axes.index("x")
            masks[zone_name] = np.zeros((crop.shape[y_index], crop.shape[x_index]), dtype=bool)
            continue
        zone_image_geometry = _geometry_um_to_image_xy(zone_geometry, he_image=whole_image)
        masks[zone_name] = _polygon_mask_for_bbox(image_geometry=zone_image_geometry, bbox=bbox)

    for zone_name, mask in masks.items():
        pathomics.update(
            {
                f"pathomics__{zone_name}__{name}": value
                for name, value in _pathomics_from_mask(crop, mask=mask, axes=whole_image.axes).items()
            }
        )
        if embedding_backend is not None:
            vector = _call_embedding_backend(
                embedding_backend,
                crop,
                mask=mask,
                axes=whole_image.axes,
                contour_key=contour_key,
                contour_id=contour_id,
                zone=zone_name,
            )
            for index, value in enumerate(vector):
                embeddings[f"embedding__{zone_name}__dim_{index:03d}"] = float(value)

    patch_array = np.asarray(contour_patch.levels[0])
    patch_metrics = _pathomics_from_mask(
        patch_array,
        mask=_full_mask_for_image(contour_patch),
        axes=contour_patch.axes,
    )
    for name, value in patch_metrics.items():
        pathomics[f"pathomics__patch__{name}"] = value

    return pathomics, embeddings


def _pathomics_from_mask(
    image: Any,
    *,
    mask: np.ndarray,
    axes: str,
) -> dict[str, float]:
    yxc = _to_yxc(np.asarray(image), axes=axes)
    if mask.shape != yxc.shape[:2]:
        raise ValueError(
            "Mask shape does not match image XY dimensions: "
            f"mask={mask.shape}, image={yxc.shape[:2]}."
        )
    if not np.any(mask):
        return {
            "foreground_fraction": 0.0,
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "std_r": 0.0,
            "std_g": 0.0,
            "std_b": 0.0,
            "stain_blue_ratio": 0.0,
            "stain_pink_ratio": 0.0,
            "texture_std": 0.0,
            "texture_entropy": 0.0,
            "edge_density": 0.0,
            "nuclear_density_proxy": 0.0,
        }
    pixels = yxc[mask].astype(float)
    if pixels.ndim == 1:
        pixels = pixels[:, None]
    if pixels.shape[1] == 1:
        pixels = np.repeat(pixels, 3, axis=1)
    elif pixels.shape[1] == 2:
        pixels = np.c_[pixels, np.zeros(len(pixels), dtype=float)]
    pixels = pixels[:, :3]

    scale = 255.0 if np.nanmax(pixels) > 1.5 else 1.0
    pixels_scaled = pixels / scale
    rgb_sum = pixels_scaled.sum(axis=1) + 1e-6
    grayscale = 0.299 * pixels_scaled[:, 0] + 0.587 * pixels_scaled[:, 1] + 0.114 * pixels_scaled[:, 2]
    edge_density, _ = _edge_density(yxc, mask=mask)

    return {
        "foreground_fraction": float(mask.mean()),
        "mean_r": float(np.nanmean(pixels_scaled[:, 0])),
        "mean_g": float(np.nanmean(pixels_scaled[:, 1])),
        "mean_b": float(np.nanmean(pixels_scaled[:, 2])),
        "std_r": float(np.nanstd(pixels_scaled[:, 0])),
        "std_g": float(np.nanstd(pixels_scaled[:, 1])),
        "std_b": float(np.nanstd(pixels_scaled[:, 2])),
        "stain_blue_ratio": float(np.nanmean(pixels_scaled[:, 2] / rgb_sum)),
        "stain_pink_ratio": float(np.nanmean((pixels_scaled[:, 0] + 0.5 * pixels_scaled[:, 1]) / rgb_sum)),
        "texture_std": float(np.nanstd(grayscale)),
        "texture_entropy": _entropy_from_histogram(grayscale),
        "edge_density": edge_density,
        "nuclear_density_proxy": float(
            np.mean((pixels_scaled[:, 2] > pixels_scaled[:, 0] * 1.05) & (grayscale < np.nanquantile(grayscale, 0.65)))
        ),
    }


def _to_yxc(array: np.ndarray, *, axes: str) -> np.ndarray:
    normalized = str(axes).lower()
    if normalized == "yxc":
        return array
    order = [normalized.index("y"), normalized.index("x")]
    if "c" in normalized:
        order.append(normalized.index("c"))
    out = np.transpose(array, axes=order)
    if out.ndim == 2:
        out = out[:, :, None]
    return out


def _edge_density(image: np.ndarray, *, mask: np.ndarray) -> tuple[float, np.ndarray]:
    yxc = _to_yxc(image, axes="yxc")
    grayscale = 0.299 * yxc[:, :, 0] + 0.587 * yxc[:, :, min(1, yxc.shape[2] - 1)] + 0.114 * yxc[:, :, min(2, yxc.shape[2] - 1)]
    grad_y = np.zeros_like(grayscale, dtype=float)
    grad_x = np.zeros_like(grayscale, dtype=float)
    grad_y[1:, :] = np.diff(grayscale, axis=0)
    grad_x[:, 1:] = np.diff(grayscale, axis=1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    valid = grad[mask]
    if valid.size == 0:
        return 0.0, grad
    threshold = float(np.nanquantile(valid, 0.75))
    return float(np.mean(valid > threshold)), grad


def _entropy_from_histogram(values: np.ndarray, bins: int = 16) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, max(float(np.nanmax(values)), 1e-6)))
    hist = hist.astype(float)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


def _full_mask_for_image(image: XeniumImage) -> np.ndarray:
    level = np.asarray(image.levels[0])
    y_index = image.axes.index("y")
    x_index = image.axes.index("x")
    return np.ones((level.shape[y_index], level.shape[x_index]), dtype=bool)


def _call_embedding_backend(
    backend: Any,
    image: Any,
    *,
    mask: np.ndarray,
    axes: str,
    contour_key: str,
    contour_id: str,
    zone: str,
) -> np.ndarray:
    yxc = _to_yxc(np.asarray(image), axes=axes).copy()
    masked = yxc.copy()
    masked[~mask] = 0
    payload = _tight_crop(masked, mask)

    candidate: Callable[..., Any] | None = None
    if callable(backend):
        candidate = backend
    else:
        for name in ("embed_image", "embed_patch", "encode", "transform"):
            if hasattr(backend, name):
                candidate = getattr(backend, name)
                break
    if candidate is None:
        raise TypeError(
            "`embedding_backend` must be callable or expose one of "
            "`embed_image`, `embed_patch`, `encode`, or `transform`."
        )
    try:
        result = candidate(
            payload,
            contour_key=contour_key,
            contour_id=contour_id,
            zone=zone,
        )
    except TypeError:
        result = candidate(payload)
    if isinstance(result, Mapping):
        vector = np.asarray([result[key] for key in sorted(result)], dtype=float)
    else:
        vector = np.asarray(result, dtype=float).reshape(-1)
    return vector


def _tight_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return image[:1, :1].copy()
    yy, xx = np.nonzero(mask)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return image[y0:y1, x0:x1].copy()


def _ligand_receptor_features(
    *,
    expression: pd.DataFrame,
    inner_mask: np.ndarray,
    outer_mask: np.ndarray,
) -> dict[str, float]:
    features: dict[str, float] = {}
    inner_mean = expression.loc[inner_mask].mean(axis=0) if inner_mask.any() else pd.Series(dtype=float)
    outer_mean = expression.loc[outer_mask].mean(axis=0) if outer_mask.any() else pd.Series(dtype=float)
    for pair_name, (ligand, receptor) in DEFAULT_CONTOUR_LR_PAIRS.items():
        ligand_inner = float(inner_mean.get(ligand, 0.0))
        receptor_inner = float(inner_mean.get(receptor, 0.0))
        ligand_outer = float(outer_mean.get(ligand, 0.0))
        receptor_outer = float(outer_mean.get(receptor, 0.0))
        features[f"{pair_name}__cross_zone"] = float(
            0.5 * (
                np.sqrt(max(ligand_inner, 0.0) * max(receptor_outer, 0.0))
                + np.sqrt(max(ligand_outer, 0.0) * max(receptor_inner, 0.0))
            )
        )
        features[f"{pair_name}__outer_minus_inner"] = float((ligand_outer + receptor_outer) - (ligand_inner + receptor_inner))
    return features


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        polygons: list[Polygon] = []
        for part in geometry.geoms:
            polygons.extend(_polygon_parts(part))
        return polygons
    return []


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
