from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
from shapely import STRtree, intersects, points, union_all
from shapely.geometry.base import BaseGeometry

from ._analysis import _build_ring_intervals, _prepare_contours, _validate_distance_window
from ._geometry import geometry_table_to_contour_frame
from ._geometry import _normalize_polygonal_geometry
from ._transform import _copy_sdata
from pyXenium.io.sdata_model import XeniumSData

__all__ = [
    "compare_contour_cell_composition",
    "compare_contour_de",
    "compare_contour_transcript_de",
    "generate_barrier_contour_shells",
    "generate_contour_shells",
    "summarize_contour_composition",
]

_DEFAULT_CELL_TYPE_KEYS = (
    "cell_type",
    "cell_class",
    "joint_cell_state",
    "cluster",
    "graphclust",
)
_MEMBERSHIP_EPSILON = 1e-9
_LOG2FC_EPSILON = 1e-9
_CPM_SCALE = 1_000_000.0


def summarize_contour_composition(
    sdata: XeniumSData,
    *,
    contour_key: str,
    contour_query: str | None = None,
    cell_type_key: str | None = None,
    cell_query: str | None = None,
    genes: str | Sequence[str] | None = None,
    gene_sets: Mapping[str, Sequence[str]] | None = None,
    transcript_query: str | None = None,
    feature_key: str = "gene_name",
) -> dict[str, pd.DataFrame]:
    """
    Summarize cell-type and gene/program composition inside each contour.

    The returned dictionary contains long-form tables for cell composition,
    gene-level transcript composition, program-level transcript composition,
    and one contour-level summary table. Membership is based on cell centroids
    or transcript coordinates intersecting each contour polygon.
    """

    _validate_sdata(sdata)
    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    cell_frame = _prepare_cell_frame(sdata=sdata, cell_query=cell_query)
    resolved_cell_type_key = _resolve_cell_type_key(cell_frame, cell_type_key=cell_type_key)
    cell_memberships = _membership_masks(contour_table=contour_table, point_frame=cell_frame)
    cell_composition, cell_summary = _summarize_cell_composition(
        contour_table=contour_table,
        cell_frame=cell_frame,
        cell_type_key=resolved_cell_type_key,
        memberships=cell_memberships,
    )

    normalized_gene_sets = _normalize_gene_sets(gene_sets)
    requested_genes = _resolve_requested_genes(genes=genes, gene_sets=normalized_gene_sets)
    transcript_counts, transcript_totals, gene_order = _count_transcripts_by_contour(
        sdata=sdata,
        contour_table=contour_table,
        transcript_query=transcript_query,
        feature_key=feature_key,
        requested_genes=requested_genes,
    )
    gene_composition = _assemble_gene_composition(
        contour_table=contour_table,
        transcript_counts=transcript_counts,
        transcript_totals=transcript_totals,
        gene_order=gene_order,
        cell_summary=cell_summary,
    )
    program_composition = _assemble_program_composition(
        contour_table=contour_table,
        gene_sets=normalized_gene_sets,
        transcript_counts=transcript_counts,
        transcript_totals=transcript_totals,
        cell_summary=cell_summary,
    )
    contour_summary = _assemble_contour_summary(
        contour_key=contour_key,
        contour_table=contour_table,
        cell_summary=cell_summary,
        transcript_totals=transcript_totals,
    )
    return {
        "cell_composition": cell_composition,
        "gene_composition": gene_composition,
        "program_composition": program_composition,
        "contour_summary": contour_summary,
    }


def compare_contour_transcript_de(
    sdata: XeniumSData,
    *,
    contour_key: str,
    groupby: str,
    contour_query: str | None = None,
    cell_query: str | None = None,
    transcript_query: str | None = None,
    feature_key: str = "gene_name",
    genes: str | Sequence[str] | None = None,
    assignment: str = "coordinates",
    comparisons: str | Sequence[str] = ("global", "one_vs_rest"),
    case: str | Sequence[str] | None = None,
    reference: str | Sequence[str] | None = None,
    min_total_transcripts_per_contour: int = 1,
    min_contours_per_group: int = 2,
    include_zero_counts: bool = False,
    return_contour_gene: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compare contour groups with transcript-count pseudobulk normalization.

    Transcript coordinates are assigned directly to contour polygons, converted
    to contour x gene counts, normalized as CPM and log1p(CPM), and compared
    across contour groups. By default transcripts are assigned by their x/y
    coordinates. Set ``assignment="cell_id"`` to aggregate cell-associated
    transcripts through the contour membership of each cell centroid, which is
    useful for whole-transcriptome screening. Set ``return_contour_gene=False``
    for whole-transcriptome scans where the full contour-gene long table would
    be unnecessarily large.
    """

    _validate_sdata(sdata)
    if int(min_total_transcripts_per_contour) < 0:
        raise ValueError("`min_total_transcripts_per_contour` must be non-negative.")
    if int(min_contours_per_group) < 1:
        raise ValueError("`min_contours_per_group` must be at least 1.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    if groupby not in contour_table.columns:
        raise KeyError(f"`groupby` column {groupby!r} was not found in the contour table.")

    assignment_mode = _normalize_transcript_assignment(assignment)
    comparison_set = _normalize_comparisons(comparisons)
    requested_genes = _normalize_optional_string_sequence(genes, name="genes")
    profile = (
        _build_contour_transcript_profile_by_cell_id(
            sdata=sdata,
            contour_key=contour_key,
            contour_table=contour_table,
            groupby=groupby,
            cell_query=cell_query,
            transcript_query=transcript_query,
            feature_key=feature_key,
            genes=requested_genes,
            include_zero_counts=include_zero_counts,
            return_contour_gene=bool(return_contour_gene),
        )
        if assignment_mode == "cell_id"
        else _build_contour_transcript_profile(
            sdata=sdata,
            contour_key=contour_key,
            contour_table=contour_table,
            groupby=groupby,
            transcript_query=transcript_query,
            feature_key=feature_key,
            genes=requested_genes,
            include_zero_counts=include_zero_counts,
            return_contour_gene=bool(return_contour_gene),
        )
    )
    matrices = profile["matrices"]
    metadata = profile["contour_metadata"]
    group_values = sorted(pd.unique(metadata[groupby].astype(str)).tolist())
    eligible = metadata["total_transcripts"].to_numpy(dtype=float) >= float(min_total_transcripts_per_contour)

    outputs = {
        "contour_gene": profile["contour_gene"],
        "global_de": pd.DataFrame(),
        "one_vs_rest_de": pd.DataFrame(),
        "pairwise_de": pd.DataFrame(),
    }
    if "global" in comparison_set:
        outputs["global_de"] = _global_transcript_de(
            matrices=matrices,
            metadata=metadata,
            groupby=groupby,
            group_values=group_values,
            eligible=eligible,
            min_contours_per_group=int(min_contours_per_group),
        )
    if "one_vs_rest" in comparison_set:
        outputs["one_vs_rest_de"] = _one_vs_rest_transcript_de(
            matrices=matrices,
            metadata=metadata,
            groupby=groupby,
            group_values=group_values,
            eligible=eligible,
            min_contours_per_group=int(min_contours_per_group),
        )
    if "pairwise" in comparison_set:
        outputs["pairwise_de"] = _pairwise_transcript_de(
            matrices=matrices,
            metadata=metadata,
            groupby=groupby,
            group_values=group_values,
            eligible=eligible,
            case=case,
            reference=reference,
            min_contours_per_group=int(min_contours_per_group),
        )
    return outputs


def compare_contour_cell_composition(
    sdata: XeniumSData,
    *,
    contour_key: str,
    groupby: str,
    contour_query: str | None = None,
    cell_type_key: str | None = None,
    cell_query: str | None = None,
    min_contours_per_group: int = 2,
) -> dict[str, pd.DataFrame]:
    """
    Compare cell-type proportions and quantities across contour groups.

    Cell centroids are assigned to every contour independently. The returned
    payload contains per-contour composition, group-level summaries, global
    Kruskal-Wallis tests, and pairwise Mann-Whitney tests.
    """

    _validate_sdata(sdata)
    if int(min_contours_per_group) < 1:
        raise ValueError("`min_contours_per_group` must be at least 1.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    if groupby not in contour_table.columns:
        raise KeyError(f"`groupby` column {groupby!r} was not found in the contour table.")

    cell_frame = _prepare_cell_frame(sdata=sdata, cell_query=cell_query)
    resolved_cell_type_key = _resolve_cell_type_key(cell_frame, cell_type_key=cell_type_key)
    memberships = _membership_masks(contour_table=contour_table, point_frame=cell_frame)
    composition, _ = _summarize_cell_composition(
        contour_table=contour_table,
        cell_frame=cell_frame,
        cell_type_key=resolved_cell_type_key,
        memberships=memberships,
    )
    contour_groups = contour_table.set_index("contour_id")[groupby].astype(str).to_dict()
    composition = composition.copy()
    composition[groupby] = composition["contour_id"].map(contour_groups).astype(str)

    group_summary = _summarize_cell_composition_by_group(
        composition=composition,
        groupby=groupby,
    )
    global_stats = _cell_composition_global_stats(
        composition=composition,
        groupby=groupby,
        min_contours_per_group=int(min_contours_per_group),
    )
    pairwise_stats = _cell_composition_pairwise_stats(
        composition=composition,
        groupby=groupby,
        min_contours_per_group=int(min_contours_per_group),
    )
    return {
        "composition": composition,
        "group_summary": group_summary,
        "global_stats": global_stats,
        "pairwise_stats": pairwise_stats,
    }


def compare_contour_de(
    sdata: XeniumSData,
    *,
    contour_key: str,
    groupby: str,
    case: str | Sequence[str],
    reference: str | Sequence[str],
    contour_query: str | None = None,
    cell_query: str | None = None,
    layer: str | None = None,
    genes: str | Sequence[str] | None = None,
    min_cells_per_contour: int = 3,
    min_contours_per_group: int = 2,
) -> pd.DataFrame:
    """
    Compare contour groups with contour-level pseudobulk expression.

    Cells are assigned to each contour by centroid intersection, expression is
    averaged within every eligible contour, and the requested case/reference
    contour groups are compared with Welch t-tests over those contour means.
    """

    _validate_sdata(sdata)
    if int(min_cells_per_contour) < 1:
        raise ValueError("`min_cells_per_contour` must be at least 1.")
    if int(min_contours_per_group) < 1:
        raise ValueError("`min_contours_per_group` must be at least 1.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    if groupby not in contour_table.columns:
        raise KeyError(f"`groupby` column {groupby!r} was not found in the contour table.")

    case_values = _normalize_group_values(case, name="case")
    reference_values = _normalize_group_values(reference, name="reference")
    selected = contour_table[groupby].astype(str).isin(case_values | reference_values)
    contour_table = contour_table.loc[selected].reset_index(drop=True)
    if contour_table.empty:
        raise ValueError("No contours matched the requested `case` or `reference` groups.")

    cell_frame = _prepare_cell_frame(sdata=sdata, cell_query=cell_query)
    memberships = _membership_masks(contour_table=contour_table, point_frame=cell_frame)
    expression, gene_names = _expression_matrix_for_genes(sdata=sdata, layer=layer, genes=genes)
    pseudobulk = _build_pseudobulk_table(
        contour_table=contour_table,
        memberships=memberships,
        cell_frame=cell_frame,
        expression=expression,
        gene_names=gene_names,
        groupby=groupby,
        case_values=case_values,
        reference_values=reference_values,
        min_cells_per_contour=int(min_cells_per_contour),
    )
    return _compare_pseudobulk_groups(
        pseudobulk=pseudobulk,
        gene_names=gene_names,
        case_label=_format_group_label(case_values),
        reference_label=_format_group_label(reference_values),
        min_contours_per_group=int(min_contours_per_group),
    )


def generate_contour_shells(
    sdata: XeniumSData,
    *,
    contour_key: str,
    inward: float,
    outward: float,
    step_size: float,
    contour_query: str | None = None,
    output_key: str | None = None,
    copy: bool = False,
) -> XeniumSData | None:
    """
    Generate independent inward/outward signed-distance shells for each contour.

    Shells are produced per contour without unioning same-class annotations or
    applying Voronoi partitioning, so neighboring shell polygons may overlap.
    """

    _validate_sdata(sdata)
    source_key = str(contour_key)
    if source_key not in sdata.shapes:
        raise KeyError(f"Contour key `{source_key}` not found in `sdata.shapes`.")
    _validate_distance_window(inward=inward, outward=outward)
    step_size = float(step_size)
    if step_size <= 0:
        raise ValueError("`step_size` must be greater than 0.")

    resolved_output_key = f"{source_key}_shells" if output_key is None else str(output_key).strip()
    if not resolved_output_key:
        raise ValueError("`output_key` must be a non-empty string when provided.")
    if resolved_output_key in sdata.shapes:
        raise KeyError(f"`sdata.shapes[{resolved_output_key!r}]` already exists.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=source_key,
        contour_query=contour_query,
    )
    intervals = _build_ring_intervals(
        ring_width=step_size,
        inward=float(inward),
        outward=float(outward),
    )
    shell_table = _build_shell_geometry_table(
        contour_table=contour_table,
        intervals=intervals,
    )
    shell_frame = geometry_table_to_contour_frame(shell_table)

    target = _copy_sdata(sdata) if copy else sdata
    if not copy:
        target.metadata = deepcopy(target.metadata)
    target.shapes[resolved_output_key] = shell_frame
    target.metadata["contours"] = _updated_shell_registry(
        sdata=target,
        source_key=source_key,
        output_key=resolved_output_key,
        inward=float(inward),
        outward=float(outward),
        step_size=step_size,
        n_shells=int(shell_frame["contour_id"].nunique()),
    )
    target._validate()
    return target if copy else None


def generate_barrier_contour_shells(
    sdata: XeniumSData,
    *,
    contour_key: str,
    inward: float,
    outward: float,
    step_size: float,
    contour_query: str | None = None,
    barrier_contour_key: str | None = None,
    barrier_query: str | None = None,
    output_key: str | None = None,
    copy: bool = False,
) -> XeniumSData | None:
    """
    Generate per-contour shells whose outward rings exclude other contours.

    Inward rings are produced exactly like :func:`generate_contour_shells`.
    Outward rings are clipped by subtracting all barrier contour interiors, so
    transcript density outside one contour is not counted inside another.
    """

    _validate_sdata(sdata)
    source_key = str(contour_key)
    if source_key not in sdata.shapes:
        raise KeyError(f"Contour key `{source_key}` not found in `sdata.shapes`.")
    _validate_distance_window(inward=inward, outward=outward)
    step_size = float(step_size)
    if step_size <= 0:
        raise ValueError("`step_size` must be greater than 0.")

    resolved_output_key = (
        f"{source_key}_barrier_shells" if output_key is None else str(output_key).strip()
    )
    if not resolved_output_key:
        raise ValueError("`output_key` must be a non-empty string when provided.")
    if resolved_output_key in sdata.shapes:
        raise KeyError(f"`sdata.shapes[{resolved_output_key!r}]` already exists.")

    barrier_key = source_key if barrier_contour_key is None else str(barrier_contour_key)
    if barrier_key not in sdata.shapes:
        raise KeyError(f"Barrier contour key `{barrier_key}` not found in `sdata.shapes`.")

    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=source_key,
        contour_query=contour_query,
    )
    barrier_table = _prepare_contours(
        sdata=sdata,
        contour_key=barrier_key,
        contour_query=barrier_query,
    )
    barrier_union = _normalize_polygonal_geometry(union_all(list(barrier_table["geometry"])))
    intervals = _build_ring_intervals(
        ring_width=step_size,
        inward=float(inward),
        outward=float(outward),
    )
    shell_table = _build_shell_geometry_table(
        contour_table=contour_table,
        intervals=intervals,
        outward_barrier=barrier_union,
    )
    shell_frame = geometry_table_to_contour_frame(shell_table)

    target = _copy_sdata(sdata) if copy else sdata
    if not copy:
        target.metadata = deepcopy(target.metadata)
    target.shapes[resolved_output_key] = shell_frame
    target.metadata["contours"] = _updated_shell_registry(
        sdata=target,
        source_key=source_key,
        output_key=resolved_output_key,
        inward=float(inward),
        outward=float(outward),
        step_size=step_size,
        n_shells=int(shell_frame["contour_id"].nunique()),
        generator="generate_barrier_contour_shells",
        shell_mode="per_contour_barrier",
        barrier_contour_key=barrier_key,
    )
    target._validate()
    return target if copy else None


def _validate_sdata(sdata: XeniumSData) -> None:
    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance.")


def _prepare_cell_frame(*, sdata: XeniumSData, cell_query: str | None) -> pd.DataFrame:
    obs = sdata.table.obs.copy()
    if cell_query is not None:
        obs = obs.query(cell_query, engine="python")

    positions = sdata.table.obs_names.get_indexer(obs.index)
    valid = positions >= 0
    obs = obs.iloc[valid].copy()
    positions = positions[valid]

    for x_col, y_col in (
        ("x_centroid", "y_centroid"),
        ("cell_centroid_x", "cell_centroid_y"),
        ("x", "y"),
    ):
        if x_col in obs.columns and y_col in obs.columns:
            frame = obs.reset_index(drop=True)
            frame["_cell_id"] = obs.index.astype(str).to_numpy()
            frame["_obs_position"] = positions.astype(int)
            frame["x"] = obs[x_col].astype(float).to_numpy()
            frame["y"] = obs[y_col].astype(float).to_numpy()
            return frame

    if "spatial" in sdata.table.obsm:
        spatial = np.asarray(sdata.table.obsm["spatial"], dtype=float)
        frame = obs.reset_index(drop=True)
        frame["_cell_id"] = obs.index.astype(str).to_numpy()
        frame["_obs_position"] = positions.astype(int)
        frame["x"] = spatial[positions, 0]
        frame["y"] = spatial[positions, 1]
        return frame

    raise ValueError(
        "Unable to infer cell centroid coordinates from `sdata.table.obs`. "
        "Expected `x_centroid/y_centroid`, `cell_centroid_x/cell_centroid_y`, or `x/y`."
    )


def _resolve_cell_type_key(cell_frame: pd.DataFrame, *, cell_type_key: str | None) -> str:
    if cell_type_key is not None:
        resolved = str(cell_type_key)
        if resolved not in cell_frame.columns:
            raise KeyError(f"`cell_type_key` column {resolved!r} was not found in `sdata.table.obs`.")
        return resolved

    for candidate in _DEFAULT_CELL_TYPE_KEYS:
        if candidate in cell_frame.columns:
            return candidate
    raise KeyError(
        "`cell_type_key` was not provided and none of the default columns were found: "
        f"{list(_DEFAULT_CELL_TYPE_KEYS)}."
    )


def _membership_masks(*, contour_table: pd.DataFrame, point_frame: pd.DataFrame) -> list[np.ndarray]:
    if point_frame.empty:
        return [np.zeros(0, dtype=bool) for _ in range(len(contour_table))]

    x_values = point_frame["x"].to_numpy(dtype=float)
    y_values = point_frame["y"].to_numpy(dtype=float)
    point_geometries = points(x_values, y_values)
    masks: list[np.ndarray] = []
    for geometry in contour_table["geometry"]:
        masks.append(
            _point_membership_mask(
                geometry=geometry,
                x_values=x_values,
                y_values=y_values,
                point_geometries=point_geometries,
            )
        )
    return masks


def _point_membership_mask(
    *,
    geometry: BaseGeometry,
    x_values: np.ndarray,
    y_values: np.ndarray,
    point_geometries: np.ndarray,
) -> np.ndarray:
    if geometry is None or geometry.is_empty or point_geometries.size == 0:
        return np.zeros(len(point_geometries), dtype=bool)

    support = geometry.buffer(_MEMBERSHIP_EPSILON)
    minx, miny, maxx, maxy = support.bounds
    candidate_mask = (
        (x_values >= float(minx))
        & (x_values <= float(maxx))
        & (y_values >= float(miny))
        & (y_values <= float(maxy))
    )
    membership = np.zeros(len(point_geometries), dtype=bool)
    if candidate_mask.any():
        membership[candidate_mask] = np.asarray(
            intersects(support, point_geometries[candidate_mask]),
            dtype=bool,
        )
    return membership


def _summarize_cell_composition(
    *,
    contour_table: pd.DataFrame,
    cell_frame: pd.DataFrame,
    cell_type_key: str,
    memberships: list[np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    categories = sorted(pd.unique(cell_frame[cell_type_key].astype(str)).tolist())
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for contour_row, membership in zip(contour_table.to_dict("records"), memberships, strict=True):
        contour_id = str(contour_row["contour_id"])
        selected = cell_frame.loc[membership]
        total = int(len(selected))
        values = selected[cell_type_key].astype(str).value_counts()
        dominant_cell_type = None
        dominant_fraction = np.nan
        if total > 0 and not values.empty:
            dominant_cell_type = str(values.index[0])
            dominant_fraction = float(values.iloc[0] / total)

        summary[contour_id] = {
            "n_cells": total,
            "dominant_cell_type": dominant_cell_type,
            "dominant_cell_fraction": dominant_fraction,
        }
        for category in categories:
            count = int(values.get(category, 0))
            rows.append(
                {
                    "contour_id": contour_id,
                    "cell_type_key": cell_type_key,
                    "cell_type": category,
                    "n_cells": count,
                    "total_cells": total,
                    "fraction": np.nan if total == 0 else count / total,
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "contour_id",
            "cell_type_key",
            "cell_type",
            "n_cells",
            "total_cells",
            "fraction",
        ],
    ), summary


def _normalize_gene_sets(gene_sets: Mapping[str, Sequence[str]] | None) -> dict[str, list[str]]:
    if gene_sets is None:
        return {}
    if not isinstance(gene_sets, Mapping):
        raise TypeError("`gene_sets` must be a mapping from program name to gene names.")

    normalized: dict[str, list[str]] = {}
    for program, program_genes in gene_sets.items():
        program_name = str(program).strip()
        if not program_name:
            raise ValueError("`gene_sets` contains an empty program name.")
        normalized[program_name] = _normalize_string_sequence(program_genes, name=f"gene_sets[{program_name!r}]")
    return normalized


def _resolve_requested_genes(
    *,
    genes: str | Sequence[str] | None,
    gene_sets: Mapping[str, Sequence[str]],
) -> list[str] | None:
    requested = _normalize_optional_string_sequence(genes, name="genes")
    if not gene_sets:
        return requested

    combined = list(requested or [])
    for program_genes in gene_sets.values():
        combined.extend(str(gene) for gene in program_genes)
    return list(dict.fromkeys(gene for gene in combined if gene))


def _count_transcripts_by_contour(
    *,
    sdata: XeniumSData,
    contour_table: pd.DataFrame,
    transcript_query: str | None,
    feature_key: str,
    requested_genes: Sequence[str] | None,
) -> tuple[dict[str, Counter[str]], dict[str, int], list[str]]:
    contour_ids = [str(value) for value in contour_table["contour_id"]]
    counts = {contour_id: Counter() for contour_id in contour_ids}
    totals = {contour_id: 0 for contour_id in contour_ids}
    gene_filter = set(requested_genes) if requested_genes is not None else None
    observed_genes: set[str] = set()

    if "transcripts" not in sdata.points and "transcripts" not in sdata.point_sources:
        if requested_genes is not None:
            raise KeyError(
                "Transcript points are not available in `sdata.points` or `sdata.point_sources`."
            )
        return counts, totals, []

    for frame in _iter_transcript_frames(
        sdata=sdata,
        transcript_query=transcript_query,
        feature_key=feature_key,
    ):
        working = frame
        if gene_filter is not None:
            working = working.loc[working[feature_key].astype(str).isin(gene_filter)]
        if working.empty:
            continue

        x_values = working["x"].to_numpy(dtype=float)
        y_values = working["y"].to_numpy(dtype=float)
        point_geometries = points(x_values, y_values)
        gene_values = working[feature_key].astype(str).to_numpy()
        for _, contour_row in contour_table.iterrows():
            contour_id = str(contour_row["contour_id"])
            membership = _point_membership_mask(
                geometry=contour_row["geometry"],
                x_values=x_values,
                y_values=y_values,
                point_geometries=point_geometries,
            )
            if not membership.any():
                continue
            local_values = gene_values[membership]
            counts[contour_id].update(local_values)
            totals[contour_id] += int(len(local_values))
            observed_genes.update(str(value) for value in local_values)

    gene_order = list(requested_genes) if requested_genes is not None else sorted(observed_genes)
    return counts, totals, gene_order


def _iter_transcript_frames(
    *,
    sdata: XeniumSData,
    transcript_query: str | None,
    feature_key: str,
):
    if "transcripts" in sdata.point_sources:
        iterator = sdata.point_sources["transcripts"].iter_chunks()
    elif "transcripts" in sdata.points:
        iterator = iter([sdata.points["transcripts"]])
    else:
        raise KeyError("Transcript points are not available in `sdata.points` or `sdata.point_sources`.")

    for frame in iterator:
        working = frame.copy()
        if transcript_query is not None:
            working = working.query(transcript_query, engine="python")
        missing = {"x", "y", feature_key}.difference(working.columns)
        if missing:
            raise ValueError(f"Transcript points are missing required columns: {sorted(missing)}")
        if not working.empty:
            yield working.loc[:, ["x", "y", feature_key]].copy()


def _iter_transcript_cell_frames(
    *,
    sdata: XeniumSData,
    transcript_query: str | None,
    feature_key: str,
):
    if "transcripts" in sdata.point_sources:
        iterator = sdata.point_sources["transcripts"].iter_chunks()
    elif "transcripts" in sdata.points:
        iterator = iter([sdata.points["transcripts"]])
    else:
        raise KeyError("Transcript points are not available in `sdata.points` or `sdata.point_sources`.")

    for frame in iterator:
        working = frame.copy()
        if transcript_query is not None:
            working = working.query(transcript_query, engine="python")
        missing = {"cell_id", feature_key}.difference(working.columns)
        if missing:
            raise ValueError(
                "Cell-id transcript assignment requires transcript points with "
                f"columns: {sorted(missing)}"
            )
        if not working.empty:
            yield working.loc[:, ["cell_id", feature_key]].copy()


def _assemble_gene_composition(
    *,
    contour_table: pd.DataFrame,
    transcript_counts: dict[str, Counter[str]],
    transcript_totals: dict[str, int],
    gene_order: Sequence[str],
    cell_summary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, contour_row in contour_table.iterrows():
        contour_id = str(contour_row["contour_id"])
        total = int(transcript_totals.get(contour_id, 0))
        area = float(contour_row["geometry"].area)
        n_cells = int(cell_summary.get(contour_id, {}).get("n_cells", 0))
        for gene in gene_order:
            count = int(transcript_counts.get(contour_id, Counter()).get(str(gene), 0))
            rows.append(
                {
                    "contour_id": contour_id,
                    "gene": str(gene),
                    "count": count,
                    "total_transcripts": total,
                    "fraction": np.nan if total == 0 else count / total,
                    "transcripts_per_cell": np.nan if n_cells == 0 else count / n_cells,
                    "transcripts_per_um2": np.nan if area <= 0 else count / area,
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "contour_id",
            "gene",
            "count",
            "total_transcripts",
            "fraction",
            "transcripts_per_cell",
            "transcripts_per_um2",
        ],
    )


def _assemble_program_composition(
    *,
    contour_table: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    transcript_counts: dict[str, Counter[str]],
    transcript_totals: dict[str, int],
    cell_summary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, contour_row in contour_table.iterrows():
        contour_id = str(contour_row["contour_id"])
        total = int(transcript_totals.get(contour_id, 0))
        area = float(contour_row["geometry"].area)
        n_cells = int(cell_summary.get(contour_id, {}).get("n_cells", 0))
        counter = transcript_counts.get(contour_id, Counter())
        for program, program_genes in gene_sets.items():
            genes = [str(gene) for gene in program_genes]
            count = int(sum(counter.get(gene, 0) for gene in genes))
            rows.append(
                {
                    "contour_id": contour_id,
                    "program": str(program),
                    "genes": tuple(genes),
                    "n_genes": len(genes),
                    "count": count,
                    "mean_gene_count": np.nan if not genes else count / len(genes),
                    "total_transcripts": total,
                    "fraction": np.nan if total == 0 else count / total,
                    "transcripts_per_cell": np.nan if n_cells == 0 else count / n_cells,
                    "transcripts_per_um2": np.nan if area <= 0 else count / area,
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "contour_id",
            "program",
            "genes",
            "n_genes",
            "count",
            "mean_gene_count",
            "total_transcripts",
            "fraction",
            "transcripts_per_cell",
            "transcripts_per_um2",
        ],
    )


def _assemble_contour_summary(
    *,
    contour_key: str,
    contour_table: pd.DataFrame,
    cell_summary: dict[str, dict[str, Any]],
    transcript_totals: dict[str, int],
) -> pd.DataFrame:
    metadata_columns = [column for column in contour_table.columns if column not in {"contour_id", "geometry"}]
    rows: list[dict[str, Any]] = []
    for _, contour_row in contour_table.iterrows():
        contour_id = str(contour_row["contour_id"])
        row = {
            "contour_key": contour_key,
            "contour_id": contour_id,
            "area_um2": float(contour_row["geometry"].area),
            "n_cells": int(cell_summary.get(contour_id, {}).get("n_cells", 0)),
            "n_transcripts": int(transcript_totals.get(contour_id, 0)),
            "dominant_cell_type": cell_summary.get(contour_id, {}).get("dominant_cell_type"),
            "dominant_cell_fraction": cell_summary.get(contour_id, {}).get("dominant_cell_fraction"),
        }
        for column in metadata_columns:
            row[column] = contour_row[column]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _normalize_optional_string_sequence(values: str | Sequence[str] | None, *, name: str) -> list[str] | None:
    if values is None:
        return None
    return _normalize_string_sequence(values, name=name)


def _normalize_string_sequence(values: str | Sequence[str], *, name: str) -> list[str]:
    if isinstance(values, str):
        items = [values]
    else:
        items = [str(value) for value in values]
    normalized = list(dict.fromkeys(item.strip() for item in items if str(item).strip()))
    if not normalized:
        raise ValueError(f"`{name}` must contain at least one non-empty value.")
    return normalized


def _normalize_group_values(values: str | Sequence[str], *, name: str) -> set[str]:
    return set(_normalize_string_sequence(values, name=name))


def _format_group_label(values: set[str]) -> str:
    return "|".join(sorted(values))


def _normalize_comparisons(comparisons: str | Sequence[str]) -> set[str]:
    normalized = _normalize_string_sequence(comparisons, name="comparisons")
    valid = {"global", "one_vs_rest", "pairwise"}
    invalid = sorted(set(normalized).difference(valid))
    if invalid:
        raise ValueError(f"`comparisons` must only contain {sorted(valid)}; received {invalid}.")
    return set(normalized)


def _normalize_transcript_assignment(assignment: str) -> str:
    mode = str(assignment).strip().lower()
    valid = {"coordinates", "cell_id"}
    if mode not in valid:
        raise ValueError(f"`assignment` must be one of {sorted(valid)}; received {assignment!r}.")
    return mode


def _build_contour_transcript_profile(
    *,
    sdata: XeniumSData,
    contour_key: str,
    contour_table: pd.DataFrame,
    groupby: str,
    transcript_query: str | None,
    feature_key: str,
    genes: Sequence[str] | None,
    include_zero_counts: bool,
    return_contour_gene: bool,
) -> dict[str, Any]:
    contour_ids = [str(value) for value in contour_table["contour_id"]]
    contour_index = {contour_id: idx for idx, contour_id in enumerate(contour_ids)}
    counters = [Counter() for _ in contour_ids]
    totals = np.zeros(len(contour_ids), dtype=float)
    gene_filter = set(str(gene) for gene in genes) if genes is not None else None
    observed_genes: set[str] = set()

    geometries = np.asarray(list(contour_table["geometry"]), dtype=object)
    tree = STRtree(geometries)

    for frame in _iter_transcript_frames(
        sdata=sdata,
        transcript_query=transcript_query,
        feature_key=feature_key,
    ):
        if frame.empty:
            continue
        point_geometries = points(
            frame["x"].to_numpy(dtype=float),
            frame["y"].to_numpy(dtype=float),
        )
        pairs = np.asarray(tree.query(point_geometries, predicate="intersects"), dtype=int)
        if pairs.size == 0:
            continue
        point_indices = pairs[0]
        contour_indices = pairs[1]
        totals += np.bincount(contour_indices, minlength=len(contour_ids)).astype(float)

        gene_values = frame[feature_key].astype(str).to_numpy()[point_indices]
        if gene_filter is not None:
            selected = np.isin(gene_values, list(gene_filter))
            if not selected.any():
                continue
            gene_values = gene_values[selected]
            contour_indices = contour_indices[selected]

        observed_genes.update(str(value) for value in gene_values)
        if gene_values.size == 0:
            continue
        pair_counts = pd.DataFrame(
            {
                "contour_index": contour_indices.astype(int),
                "gene": gene_values.astype(str),
            }
        ).value_counts(["contour_index", "gene"])
        for (local_contour_index, gene), count in pair_counts.items():
            counters[int(local_contour_index)][str(gene)] += int(count)

    gene_order = list(genes) if genes is not None else sorted(observed_genes)
    if not gene_order:
        raise ValueError(
            "No transcript genes were observed inside the selected contours. "
            "Check `transcript_query`, `feature_key`, or the contour geometry."
        )

    counts = np.zeros((len(contour_ids), len(gene_order)), dtype=float)
    gene_index = {gene: idx for idx, gene in enumerate(gene_order)}
    for contour_id, local_index in contour_index.items():
        for gene, count in counters[local_index].items():
            gene_position = gene_index.get(str(gene))
            if gene_position is not None:
                counts[local_index, gene_position] = float(count)

    areas = contour_table["geometry"].map(lambda geom: float(geom.area)).to_numpy(dtype=float)
    cpm = np.divide(
        counts,
        totals[:, None],
        out=np.zeros_like(counts, dtype=float),
        where=totals[:, None] > 0,
    ) * _CPM_SCALE
    log1p_cpm = np.log1p(cpm)
    density = np.divide(
        counts,
        areas[:, None],
        out=np.full_like(counts, np.nan, dtype=float),
        where=areas[:, None] > 0,
    )

    metadata = _build_contour_transcript_metadata(
        contour_key=contour_key,
        contour_table=contour_table,
        groupby=groupby,
        totals=totals,
        areas=areas,
    )
    matrices = {
        "genes": list(gene_order),
        "counts": counts,
        "cpm": cpm,
        "log1p_cpm": log1p_cpm,
        "density_per_um2": density,
    }
    contour_gene = (
        _assemble_contour_gene_profile_table(
            metadata=metadata,
            matrices=matrices,
            include_zero_counts=include_zero_counts,
        )
        if return_contour_gene
        else _empty_contour_gene_profile_table(metadata=metadata)
    )
    return {
        "contour_metadata": metadata,
        "matrices": matrices,
        "contour_gene": contour_gene,
    }


def _build_contour_transcript_profile_by_cell_id(
    *,
    sdata: XeniumSData,
    contour_key: str,
    contour_table: pd.DataFrame,
    groupby: str,
    cell_query: str | None,
    transcript_query: str | None,
    feature_key: str,
    genes: Sequence[str] | None,
    include_zero_counts: bool,
    return_contour_gene: bool,
) -> dict[str, Any]:
    contour_ids = [str(value) for value in contour_table["contour_id"]]
    counters = [Counter() for _ in contour_ids]
    totals = np.zeros(len(contour_ids), dtype=float)
    gene_filter = set(str(gene) for gene in genes) if genes is not None else None
    observed_genes: set[str] = set()

    cell_frame = _prepare_cell_frame(sdata=sdata, cell_query=cell_query)
    memberships = _membership_masks(contour_table=contour_table, point_frame=cell_frame)
    cell_to_contours: dict[str, list[int]] = {}
    for local_index, membership in enumerate(memberships):
        if not membership.any():
            continue
        for cell_id in cell_frame.loc[membership, "_cell_id"].astype(str):
            cell_to_contours.setdefault(str(cell_id), []).append(local_index)

    candidate_cells = set(cell_to_contours)
    for frame in _iter_transcript_cell_frames(
        sdata=sdata,
        transcript_query=transcript_query,
        feature_key=feature_key,
    ):
        if frame.empty or not candidate_cells:
            continue
        working = frame.loc[frame["cell_id"].astype(str).isin(candidate_cells), ["cell_id", feature_key]].copy()
        if working.empty:
            continue
        working["cell_id"] = working["cell_id"].astype(str)
        working["contour_index"] = working["cell_id"].map(cell_to_contours)
        working = working.explode("contour_index", ignore_index=True)
        working = working.dropna(subset=["contour_index"])
        if working.empty:
            continue

        contour_indices = working["contour_index"].astype(int).to_numpy()
        totals += np.bincount(contour_indices, minlength=len(contour_ids)).astype(float)

        gene_values = working[feature_key].astype(str).to_numpy()
        if gene_filter is not None:
            selected = np.isin(gene_values, list(gene_filter))
            if not selected.any():
                continue
            gene_values = gene_values[selected]
            contour_indices = contour_indices[selected]

        observed_genes.update(str(value) for value in gene_values)
        if gene_values.size == 0:
            continue
        pair_counts = pd.DataFrame(
            {
                "contour_index": contour_indices.astype(int),
                "gene": gene_values.astype(str),
            }
        ).value_counts(["contour_index", "gene"])
        for (local_contour_index, gene), count in pair_counts.items():
            counters[int(local_contour_index)][str(gene)] += int(count)

    gene_order = list(genes) if genes is not None else sorted(observed_genes)
    if not gene_order:
        raise ValueError(
            "No cell-associated transcript genes were observed inside the selected contours. "
            "Check `transcript_query`, `feature_key`, `cell_query`, or the contour geometry."
        )

    counts = np.zeros((len(contour_ids), len(gene_order)), dtype=float)
    gene_index = {gene: idx for idx, gene in enumerate(gene_order)}
    for local_index, counter in enumerate(counters):
        for gene, count in counter.items():
            gene_position = gene_index.get(str(gene))
            if gene_position is not None:
                counts[local_index, gene_position] = float(count)

    areas = contour_table["geometry"].map(lambda geom: float(geom.area)).to_numpy(dtype=float)
    cpm = np.divide(
        counts,
        totals[:, None],
        out=np.zeros_like(counts, dtype=float),
        where=totals[:, None] > 0,
    ) * _CPM_SCALE
    log1p_cpm = np.log1p(cpm)
    density = np.divide(
        counts,
        areas[:, None],
        out=np.full_like(counts, np.nan, dtype=float),
        where=areas[:, None] > 0,
    )

    metadata = _build_contour_transcript_metadata(
        contour_key=contour_key,
        contour_table=contour_table,
        groupby=groupby,
        totals=totals,
        areas=areas,
    )
    matrices = {
        "genes": list(gene_order),
        "counts": counts,
        "cpm": cpm,
        "log1p_cpm": log1p_cpm,
        "density_per_um2": density,
    }
    contour_gene = (
        _assemble_contour_gene_profile_table(
            metadata=metadata,
            matrices=matrices,
            include_zero_counts=include_zero_counts,
        )
        if return_contour_gene
        else _empty_contour_gene_profile_table(metadata=metadata)
    )
    return {
        "contour_metadata": metadata,
        "matrices": matrices,
        "contour_gene": contour_gene,
    }


def _build_contour_transcript_metadata(
    *,
    contour_key: str,
    contour_table: pd.DataFrame,
    groupby: str,
    totals: np.ndarray,
    areas: np.ndarray,
) -> pd.DataFrame:
    metadata_columns = [column for column in contour_table.columns if column not in {"contour_id", "geometry"}]
    rows: list[dict[str, Any]] = []
    for local_index, (_, contour_row) in enumerate(contour_table.iterrows()):
        row = {
            "contour_key": str(contour_key),
            "contour_id": str(contour_row["contour_id"]),
            groupby: str(contour_row[groupby]),
            "area_um2": float(areas[local_index]),
            "total_transcripts": int(totals[local_index]),
        }
        for column in metadata_columns:
            row[column] = contour_row[column]
        row[groupby] = str(row[groupby])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _assemble_contour_gene_profile_table(
    *,
    metadata: pd.DataFrame,
    matrices: Mapping[str, Any],
    include_zero_counts: bool,
) -> pd.DataFrame:
    genes = list(matrices["genes"])
    counts = np.asarray(matrices["counts"], dtype=float)
    cpm = np.asarray(matrices["cpm"], dtype=float)
    log1p_cpm = np.asarray(matrices["log1p_cpm"], dtype=float)
    density = np.asarray(matrices["density_per_um2"], dtype=float)
    metadata_columns = metadata.columns.tolist()
    rows: list[dict[str, Any]] = []

    for contour_index, metadata_row in metadata.iterrows():
        contour_payload = metadata_row.to_dict()
        for gene_index, gene in enumerate(genes):
            count = float(counts[contour_index, gene_index])
            if not include_zero_counts and count == 0:
                continue
            row = dict(contour_payload)
            row.update(
                {
                    "gene": str(gene),
                    "count": int(count),
                    "cpm": float(cpm[contour_index, gene_index]),
                    "log1p_cpm": float(log1p_cpm[contour_index, gene_index]),
                    "density_per_um2": float(density[contour_index, gene_index]),
                }
            )
            rows.append(row)

    columns = metadata_columns + ["gene", "count", "cpm", "log1p_cpm", "density_per_um2"]
    return pd.DataFrame.from_records(rows, columns=columns)


def _empty_contour_gene_profile_table(*, metadata: pd.DataFrame) -> pd.DataFrame:
    columns = metadata.columns.tolist() + ["gene", "count", "cpm", "log1p_cpm", "density_per_um2"]
    return pd.DataFrame(columns=columns)


def _global_transcript_de(
    *,
    matrices: Mapping[str, Any],
    metadata: pd.DataFrame,
    groupby: str,
    group_values: Sequence[str],
    eligible: np.ndarray,
    min_contours_per_group: int,
) -> pd.DataFrame:
    genes = list(matrices["genes"])
    log1p_cpm = np.asarray(matrices["log1p_cpm"], dtype=float)
    cpm = np.asarray(matrices["cpm"], dtype=float)
    density = np.asarray(matrices["density_per_um2"], dtype=float)
    rows: list[dict[str, Any]] = []

    for gene_index, gene in enumerate(genes):
        group_arrays = []
        group_means: dict[str, float] = {}
        group_mean_cpm: dict[str, float] = {}
        group_mean_density: dict[str, float] = {}
        for group in group_values:
            group_mask = _group_mask(metadata=metadata, groupby=groupby, group=group, eligible=eligible)
            values = _finite_values(log1p_cpm[group_mask, gene_index])
            if values.size:
                group_means[str(group)] = _safe_mean(values)
                group_mean_cpm[str(group)] = _safe_mean(_finite_values(cpm[group_mask, gene_index]))
                group_mean_density[str(group)] = _safe_mean(_finite_values(density[group_mask, gene_index]))
            if values.size >= min_contours_per_group:
                group_arrays.append(values)

        status = "ok"
        statistic = np.nan
        p_value = np.nan
        if len(group_arrays) < 2:
            status = "insufficient_contour_replicates"
        else:
            statistic, p_value = _safe_kruskal(group_arrays)
            if not np.isfinite(p_value):
                status = "test_not_defined"

        top_group = None
        bottom_group = None
        max_mean = np.nan
        min_mean = np.nan
        top_group_mean_density = np.nan
        bottom_group_mean_density = np.nan
        if group_means:
            top_group = max(group_means, key=group_means.get)
            bottom_group = min(group_means, key=group_means.get)
            max_mean = float(group_means[top_group])
            min_mean = float(group_means[bottom_group])
            top_group_mean_density = float(group_mean_density.get(top_group, np.nan))
            bottom_group_mean_density = float(group_mean_density.get(bottom_group, np.nan))

        rows.append(
            {
                "gene": str(gene),
                "comparison": "global",
                "n_groups": int(len(group_values)),
                "n_tested_groups": int(len(group_arrays)),
                "n_contours": int(eligible.sum()),
                "top_group": top_group,
                "bottom_group": bottom_group,
                "max_mean_log1p_cpm": max_mean,
                "min_mean_log1p_cpm": min_mean,
                "delta_log1p_cpm": np.nan if not np.isfinite(max_mean) or not np.isfinite(min_mean) else max_mean - min_mean,
                "top_group_mean_cpm": np.nan if top_group is None else group_mean_cpm.get(top_group, np.nan),
                "top_group_mean_density_per_um2": top_group_mean_density,
                "bottom_group_mean_density_per_um2": bottom_group_mean_density,
                "delta_mean_density_per_um2": np.nan
                if not np.isfinite(top_group_mean_density) or not np.isfinite(bottom_group_mean_density)
                else top_group_mean_density - bottom_group_mean_density,
                "statistic": statistic,
                "p_value": p_value,
                "fdr": np.nan,
                "status": status,
            }
        )

    return _sort_de_frame(_with_fdr(pd.DataFrame.from_records(rows)))


def _one_vs_rest_transcript_de(
    *,
    matrices: Mapping[str, Any],
    metadata: pd.DataFrame,
    groupby: str,
    group_values: Sequence[str],
    eligible: np.ndarray,
    min_contours_per_group: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in group_values:
        rows.extend(
            _binary_transcript_de_rows(
                matrices=matrices,
                metadata=metadata,
                groupby=groupby,
                case_label=str(group),
                reference_label="rest",
                case_mask=_group_mask(metadata=metadata, groupby=groupby, group=group, eligible=eligible),
                reference_mask=eligible
                & ~metadata[groupby].astype(str).eq(str(group)).to_numpy(dtype=bool),
                min_contours_per_group=min_contours_per_group,
                comparison="one_vs_rest",
            )
        )
    return _sort_de_frame(_with_fdr(pd.DataFrame.from_records(rows)))


def _pairwise_transcript_de(
    *,
    matrices: Mapping[str, Any],
    metadata: pd.DataFrame,
    groupby: str,
    group_values: Sequence[str],
    eligible: np.ndarray,
    case: str | Sequence[str] | None,
    reference: str | Sequence[str] | None,
    min_contours_per_group: int,
) -> pd.DataFrame:
    pairs = _resolve_pairwise_groups(group_values=group_values, case=case, reference=reference)
    rows: list[dict[str, Any]] = []
    for case_group, reference_group in pairs:
        rows.extend(
            _binary_transcript_de_rows(
                matrices=matrices,
                metadata=metadata,
                groupby=groupby,
                case_label=str(case_group),
                reference_label=str(reference_group),
                case_mask=_group_mask(metadata=metadata, groupby=groupby, group=case_group, eligible=eligible),
                reference_mask=_group_mask(
                    metadata=metadata,
                    groupby=groupby,
                    group=reference_group,
                    eligible=eligible,
                ),
                min_contours_per_group=min_contours_per_group,
                comparison="pairwise",
            )
        )
    return _sort_de_frame(_with_fdr(pd.DataFrame.from_records(rows)))


def _resolve_pairwise_groups(
    *,
    group_values: Sequence[str],
    case: str | Sequence[str] | None,
    reference: str | Sequence[str] | None,
) -> list[tuple[str, str]]:
    available = set(str(value) for value in group_values)
    if case is None and reference is None:
        return [(str(left), str(right)) for left, right in combinations(sorted(available), 2)]
    if case is None or reference is None:
        raise ValueError("`case` and `reference` must either both be provided or both be omitted.")

    case_values = _normalize_group_values(case, name="case")
    reference_values = _normalize_group_values(reference, name="reference")
    missing = sorted((case_values | reference_values).difference(available))
    if missing:
        raise KeyError(f"Requested contour groups were not found in `{group_values}`: {missing}")
    return [
        (str(case_group), str(reference_group))
        for case_group in sorted(case_values)
        for reference_group in sorted(reference_values)
        if str(case_group) != str(reference_group)
    ]


def _binary_transcript_de_rows(
    *,
    matrices: Mapping[str, Any],
    metadata: pd.DataFrame,
    groupby: str,
    case_label: str,
    reference_label: str,
    case_mask: np.ndarray,
    reference_mask: np.ndarray,
    min_contours_per_group: int,
    comparison: str,
) -> list[dict[str, Any]]:
    genes = list(matrices["genes"])
    log1p_cpm = np.asarray(matrices["log1p_cpm"], dtype=float)
    cpm = np.asarray(matrices["cpm"], dtype=float)
    density = np.asarray(matrices["density_per_um2"], dtype=float)
    rows: list[dict[str, Any]] = []

    for gene_index, gene in enumerate(genes):
        case_values = _finite_values(log1p_cpm[case_mask, gene_index])
        reference_values = _finite_values(log1p_cpm[reference_mask, gene_index])
        case_cpm = _finite_values(cpm[case_mask, gene_index])
        reference_cpm = _finite_values(cpm[reference_mask, gene_index])
        case_density = _finite_values(density[case_mask, gene_index])
        reference_density = _finite_values(density[reference_mask, gene_index])
        mean_case_cpm = _safe_mean(case_cpm)
        mean_reference_cpm = _safe_mean(reference_cpm)
        log2fc_cpm = np.nan
        if np.isfinite(mean_case_cpm) and np.isfinite(mean_reference_cpm):
            log2fc_cpm = float(
                np.log2((mean_case_cpm + _LOG2FC_EPSILON) / (mean_reference_cpm + _LOG2FC_EPSILON))
            )

        status = "ok"
        statistic = np.nan
        p_value = np.nan
        if len(case_values) < min_contours_per_group or len(reference_values) < min_contours_per_group:
            status = "insufficient_contour_replicates"
        else:
            statistic, p_value = _welch_ttest(case_values, reference_values)
            if not np.isfinite(p_value):
                status = "test_not_defined"

        rows.append(
            {
                "gene": str(gene),
                "comparison": str(comparison),
                "groupby": str(groupby),
                "case": str(case_label),
                "reference": str(reference_label),
                "n_case_contours": int(len(case_values)),
                "n_reference_contours": int(len(reference_values)),
                "mean_case_log1p_cpm": _safe_mean(case_values),
                "mean_reference_log1p_cpm": _safe_mean(reference_values),
                "mean_case_cpm": mean_case_cpm,
                "mean_reference_cpm": mean_reference_cpm,
                "mean_case_density_per_um2": _safe_mean(case_density),
                "mean_reference_density_per_um2": _safe_mean(reference_density),
                "log2fc_cpm": log2fc_cpm,
                "statistic": statistic,
                "p_value": p_value,
                "fdr": np.nan,
                "status": status,
            }
        )
    return rows


def _group_mask(*, metadata: pd.DataFrame, groupby: str, group: str, eligible: np.ndarray) -> np.ndarray:
    return eligible & metadata[groupby].astype(str).eq(str(group)).to_numpy(dtype=bool)


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _safe_kruskal(group_arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    try:
        statistic, p_value = kruskal(*group_arrays, nan_policy="omit")
    except ValueError:
        return np.nan, np.nan
    return float(statistic), float(p_value)


def _with_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "p_value" not in frame.columns:
        return frame
    frame = frame.copy()
    if "fdr" not in frame.columns:
        frame["fdr"] = np.nan
    ok_mask = frame.get("status", pd.Series("ok", index=frame.index)).eq("ok") & np.isfinite(
        frame["p_value"].to_numpy(dtype=float)
    )
    if ok_mask.any():
        frame.loc[ok_mask, "fdr"] = _benjamini_hochberg(frame.loc[ok_mask, "p_value"].to_numpy(dtype=float))
    return frame


def _sort_de_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [column for column in ["fdr", "p_value", "gene", "case", "reference"] if column in frame.columns]
    return frame.sort_values(sort_columns, na_position="last", kind="stable").reset_index(drop=True)


def _summarize_cell_composition_by_group(
    *,
    composition: pd.DataFrame,
    groupby: str,
) -> pd.DataFrame:
    if composition.empty:
        return pd.DataFrame(
            columns=[
                groupby,
                "cell_type",
                "n_contours",
                "total_count",
                "mean_count_per_contour",
                "median_count_per_contour",
                "mean_fraction",
                "median_fraction",
                "sem_fraction",
                "mean_total_cells_per_contour",
            ]
        )

    grouped = composition.groupby([groupby, "cell_type"], dropna=False)
    summary = grouped.agg(
        n_contours=("contour_id", "nunique"),
        total_count=("n_cells", "sum"),
        mean_count_per_contour=("n_cells", "mean"),
        median_count_per_contour=("n_cells", "median"),
        mean_fraction=("fraction", "mean"),
        median_fraction=("fraction", "median"),
        sem_fraction=("fraction", lambda values: float(pd.Series(values).sem())),
        mean_total_cells_per_contour=("total_cells", "mean"),
    )
    return summary.reset_index()


def _cell_composition_global_stats(
    *,
    composition: pd.DataFrame,
    groupby: str,
    min_contours_per_group: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_values = sorted(pd.unique(composition[groupby].astype(str)).tolist())
    for cell_type in sorted(pd.unique(composition["cell_type"].astype(str)).tolist()):
        local = composition.loc[composition["cell_type"].astype(str).eq(str(cell_type))]
        for metric in ("fraction", "n_cells"):
            arrays = []
            means: dict[str, float] = {}
            for group in group_values:
                values = _finite_values(
                    local.loc[local[groupby].astype(str).eq(str(group)), metric].to_numpy(dtype=float)
                )
                if values.size:
                    means[str(group)] = _safe_mean(values)
                if values.size >= min_contours_per_group:
                    arrays.append(values)

            status = "ok"
            statistic = np.nan
            p_value = np.nan
            if len(arrays) < 2:
                status = "insufficient_contour_replicates"
            else:
                statistic, p_value = _safe_kruskal(arrays)
                if not np.isfinite(p_value):
                    status = "test_not_defined"

            top_group = max(means, key=means.get) if means else None
            bottom_group = min(means, key=means.get) if means else None
            rows.append(
                {
                    "cell_type": str(cell_type),
                    "metric": metric,
                    "comparison": "global",
                    "n_groups": int(len(group_values)),
                    "n_tested_groups": int(len(arrays)),
                    "top_group": top_group,
                    "bottom_group": bottom_group,
                    "max_mean": np.nan if top_group is None else means.get(top_group, np.nan),
                    "min_mean": np.nan if bottom_group is None else means.get(bottom_group, np.nan),
                    "statistic": statistic,
                    "p_value": p_value,
                    "fdr": np.nan,
                    "status": status,
                }
            )
    return _sort_composition_stats(_with_fdr(pd.DataFrame.from_records(rows)))


def _cell_composition_pairwise_stats(
    *,
    composition: pd.DataFrame,
    groupby: str,
    min_contours_per_group: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_values = sorted(pd.unique(composition[groupby].astype(str)).tolist())
    for cell_type in sorted(pd.unique(composition["cell_type"].astype(str)).tolist()):
        local = composition.loc[composition["cell_type"].astype(str).eq(str(cell_type))]
        for case_group, reference_group in combinations(group_values, 2):
            for metric in ("fraction", "n_cells"):
                case_values = _finite_values(
                    local.loc[local[groupby].astype(str).eq(str(case_group)), metric].to_numpy(dtype=float)
                )
                reference_values = _finite_values(
                    local.loc[local[groupby].astype(str).eq(str(reference_group)), metric].to_numpy(dtype=float)
                )
                status = "ok"
                statistic = np.nan
                p_value = np.nan
                if (
                    case_values.size < min_contours_per_group
                    or reference_values.size < min_contours_per_group
                ):
                    status = "insufficient_contour_replicates"
                else:
                    statistic, p_value = _safe_mannwhitney(case_values, reference_values)
                    if not np.isfinite(p_value):
                        status = "test_not_defined"

                rows.append(
                    {
                        "cell_type": str(cell_type),
                        "metric": metric,
                        "comparison": "pairwise",
                        "case": str(case_group),
                        "reference": str(reference_group),
                        "n_case_contours": int(case_values.size),
                        "n_reference_contours": int(reference_values.size),
                        "mean_case": _safe_mean(case_values),
                        "mean_reference": _safe_mean(reference_values),
                        "delta_mean": _safe_mean(case_values) - _safe_mean(reference_values),
                        "statistic": statistic,
                        "p_value": p_value,
                        "fdr": np.nan,
                        "status": status,
                    }
                )
    return _sort_composition_stats(_with_fdr(pd.DataFrame.from_records(rows)))


def _safe_mannwhitney(case_values: np.ndarray, reference_values: np.ndarray) -> tuple[float, float]:
    try:
        statistic, p_value = mannwhitneyu(case_values, reference_values, alternative="two-sided")
    except ValueError:
        return np.nan, np.nan
    return float(statistic), float(p_value)


def _sort_composition_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = [
        column
        for column in ["fdr", "p_value", "cell_type", "metric", "case", "reference"]
        if column in frame.columns
    ]
    return frame.sort_values(sort_columns, na_position="last", kind="stable").reset_index(drop=True)


def _expression_matrix_for_genes(
    *,
    sdata: XeniumSData,
    layer: str | None,
    genes: str | Sequence[str] | None,
):
    adata = sdata.table
    gene_names = _normalize_optional_string_sequence(genes, name="genes")
    var_names = pd.Index(adata.var_names.astype(str))
    if gene_names is None:
        gene_names = var_names.tolist()
    if not gene_names:
        raise ValueError("No genes are available for contour differential expression.")

    gene_positions = var_names.get_indexer(gene_names)
    missing = [gene for gene, position in zip(gene_names, gene_positions, strict=True) if position < 0]
    if missing:
        raise KeyError(f"Genes were not found in `sdata.table.var_names`: {missing}")

    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} was not found in `sdata.table.layers`.")
        matrix = adata.layers[layer]
    return matrix[:, gene_positions], gene_names


def _build_pseudobulk_table(
    *,
    contour_table: pd.DataFrame,
    memberships: list[np.ndarray],
    cell_frame: pd.DataFrame,
    expression: Any,
    gene_names: Sequence[str],
    groupby: str,
    case_values: set[str],
    reference_values: set[str],
    min_cells_per_contour: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contour_row, membership in zip(contour_table.to_dict("records"), memberships, strict=True):
        n_cells = int(membership.sum())
        if n_cells < min_cells_per_contour:
            continue

        group_value = str(contour_row[groupby])
        if group_value in case_values:
            group_kind = "case"
        elif group_value in reference_values:
            group_kind = "reference"
        else:
            continue

        positions = cell_frame.loc[membership, "_obs_position"].to_numpy(dtype=int)
        mean_expression = _mean_expression(expression=expression, obs_positions=positions)
        row = {
            "contour_id": str(contour_row["contour_id"]),
            "group": group_value,
            "group_kind": group_kind,
            "n_cells": n_cells,
        }
        row.update({gene: float(value) for gene, value in zip(gene_names, mean_expression, strict=True)})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _mean_expression(*, expression: Any, obs_positions: np.ndarray) -> np.ndarray:
    if obs_positions.size == 0:
        return np.full(expression.shape[1], np.nan, dtype=float)
    subset = expression[obs_positions, :]
    if sparse.issparse(subset):
        return np.asarray(subset.mean(axis=0)).ravel().astype(float)
    return np.asarray(subset, dtype=float).mean(axis=0)


def _compare_pseudobulk_groups(
    *,
    pseudobulk: pd.DataFrame,
    gene_names: Sequence[str],
    case_label: str,
    reference_label: str,
    min_contours_per_group: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    case_frame = pseudobulk.loc[pseudobulk.get("group_kind", pd.Series(dtype=str)) == "case"]
    reference_frame = pseudobulk.loc[pseudobulk.get("group_kind", pd.Series(dtype=str)) == "reference"]

    for gene in gene_names:
        case_values = _numeric_gene_values(case_frame, gene)
        reference_values = _numeric_gene_values(reference_frame, gene)
        mean_case = _safe_mean(case_values)
        mean_reference = _safe_mean(reference_values)
        log2fc = np.nan
        if np.isfinite(mean_case) and np.isfinite(mean_reference):
            log2fc = float(np.log2((mean_case + _LOG2FC_EPSILON) / (mean_reference + _LOG2FC_EPSILON)))

        status = "ok"
        statistic = np.nan
        p_value = np.nan
        if len(case_values) < min_contours_per_group or len(reference_values) < min_contours_per_group:
            status = "insufficient_contour_replicates"
        else:
            statistic, p_value = _welch_ttest(case_values, reference_values)
            if not np.isfinite(p_value):
                status = "test_not_defined"

        rows.append(
            {
                "gene": str(gene),
                "case": case_label,
                "reference": reference_label,
                "n_case_contours": int(len(case_values)),
                "n_reference_contours": int(len(reference_values)),
                "mean_case": mean_case,
                "mean_reference": mean_reference,
                "log2fc": log2fc,
                "statistic": statistic,
                "p_value": p_value,
                "fdr": np.nan,
                "status": status,
            }
        )

    result = pd.DataFrame.from_records(rows)
    ok_mask = result["status"].eq("ok") & np.isfinite(result["p_value"].to_numpy(dtype=float))
    if ok_mask.any():
        result.loc[ok_mask, "fdr"] = _benjamini_hochberg(result.loc[ok_mask, "p_value"].to_numpy(dtype=float))
    return result.sort_values(["fdr", "p_value", "gene"], na_position="last", kind="stable").reset_index(drop=True)


def _numeric_gene_values(frame: pd.DataFrame, gene: str) -> np.ndarray:
    if frame.empty or gene not in frame.columns:
        return np.asarray([], dtype=float)
    return frame[gene].to_numpy(dtype=float)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    return float(np.nanmean(values))


def _welch_ttest(case_values: np.ndarray, reference_values: np.ndarray) -> tuple[float, float]:
    statistic, p_value = ttest_ind(
        case_values,
        reference_values,
        equal_var=False,
        nan_policy="omit",
    )
    return float(statistic), float(p_value)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    finite = p_values[finite_mask]
    if finite.size == 0:
        return adjusted

    order = np.argsort(finite)
    ranked = finite[order] * finite.size / np.arange(1, finite.size + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    local_adjusted = np.empty_like(finite)
    local_adjusted[order] = np.minimum(ranked, 1.0)
    adjusted[finite_mask] = local_adjusted
    return adjusted


def _build_shell_geometry_table(
    *,
    contour_table: pd.DataFrame,
    intervals: Sequence[tuple[float, float]],
    outward_barrier: BaseGeometry | None = None,
) -> pd.DataFrame:
    metadata_columns = [column for column in contour_table.columns if column not in {"contour_id", "geometry"}]
    records: list[dict[str, Any]] = []
    for _, contour_row in contour_table.iterrows():
        source_contour_id = str(contour_row["contour_id"])
        for shell_index, (shell_start, shell_end) in enumerate(intervals):
            shell_geometry = _shell_geometry(
                contour_row["geometry"],
                shell_start=shell_start,
                shell_end=shell_end,
                outward_barrier=outward_barrier,
            )
            if shell_geometry.is_empty:
                continue

            shell_id = f"{source_contour_id}__shell_{shell_index:03d}"
            record = {
                "contour_id": shell_id,
                "geometry": shell_geometry,
                "source_contour_id": source_contour_id,
                "shell_id": shell_id,
                "shell_index": int(shell_index),
                "shell_start": float(shell_start),
                "shell_end": float(shell_end),
                "shell_mid": 0.5 * (float(shell_start) + float(shell_end)),
                "shell_direction": "inward" if shell_end <= 0 else "outward",
            }
            for column in metadata_columns:
                record[column] = contour_row[column]
            records.append(record)

    if not records:
        raise ValueError("Contour shell generation did not produce any non-empty polygon geometries.")
    return pd.DataFrame.from_records(records)


def _shell_geometry(
    geometry: BaseGeometry,
    *,
    shell_start: float,
    shell_end: float,
    outward_barrier: BaseGeometry | None = None,
) -> BaseGeometry:
    source = _normalize_polygonal_geometry(geometry)
    if source.is_empty:
        return source

    shell_start = float(shell_start)
    shell_end = float(shell_end)
    if shell_end <= 0:
        near = source if np.isclose(shell_end, 0.0) else _normalize_polygonal_geometry(source.buffer(shell_end))
        far = _normalize_polygonal_geometry(source.buffer(shell_start))
        shell = near.difference(far)
    else:
        outer = _normalize_polygonal_geometry(source.buffer(shell_end))
        inner = source if np.isclose(shell_start, 0.0) else _normalize_polygonal_geometry(source.buffer(shell_start))
        shell = outer.difference(inner)
        if outward_barrier is not None and not outward_barrier.is_empty:
            shell = shell.difference(outward_barrier)
    return _normalize_polygonal_geometry(shell)


def _updated_shell_registry(
    *,
    sdata: XeniumSData,
    source_key: str,
    output_key: str,
    inward: float,
    outward: float,
    step_size: float,
    n_shells: int,
    generator: str = "generate_contour_shells",
    shell_mode: str = "per_contour",
    barrier_contour_key: str | None = None,
) -> dict[str, Any]:
    existing_registry = sdata.metadata.get("contours", {})
    contour_registry = dict(existing_registry) if isinstance(existing_registry, dict) else {}
    source_metadata = contour_registry.get(source_key, {})
    derived_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    units = derived_metadata.get("units", sdata.metadata.get("units"))
    units_is_um = str(units).strip().lower() in {"micron", "microns", "um"}

    derived_metadata["derived_from_key"] = source_key
    derived_metadata["generator"] = str(generator)
    derived_metadata["shell_mode"] = str(shell_mode)
    if barrier_contour_key is not None:
        derived_metadata["barrier_contour_key"] = str(barrier_contour_key)
    else:
        derived_metadata.pop("barrier_contour_key", None)
    derived_metadata["inward"] = float(inward)
    derived_metadata["outward"] = float(outward)
    derived_metadata["step_size"] = float(step_size)
    derived_metadata["n_shells"] = int(n_shells)
    if units_is_um:
        derived_metadata["inward_um"] = float(inward)
        derived_metadata["outward_um"] = float(outward)
        derived_metadata["step_size_um"] = float(step_size)
    else:
        derived_metadata.pop("inward_um", None)
        derived_metadata.pop("outward_um", None)
        derived_metadata.pop("step_size_um", None)

    contour_registry[output_key] = derived_metadata
    return contour_registry
