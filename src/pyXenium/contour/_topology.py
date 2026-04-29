from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._analysis import _prepare_contours
from ._histoseg import load_histoseg_submodule
from pyXenium.io.sdata_model import XeniumSData

__all__ = ["summarize_contour_topology"]


def summarize_contour_topology(
    sdata: XeniumSData,
    *,
    contour_key: str,
    contour_query: str | None = None,
    groupby: str | None = None,
    boundary_tolerance: float = 1.0,
    min_shared_boundary: float = 0.0,
    enclosure_min_fraction: float = 0.95,
    histoseg_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Summarize contour boundary-neighbor and enclosure relationships.

    This is a thin pyXenium wrapper around HistoSeg's pure-geometry topology
    engine. It reconstructs one Shapely geometry per contour from
    ``sdata.shapes[contour_key]`` and returns plain pandas tables.
    """

    if not isinstance(sdata, XeniumSData):
        raise TypeError("`sdata` must be a XeniumSData instance.")

    histoseg_topology = _load_histoseg_topology(histoseg_root=histoseg_root)
    contour_table = _prepare_contours(
        sdata=sdata,
        contour_key=contour_key,
        contour_query=contour_query,
    )
    result = histoseg_topology(
        contour_table,
        contour_id_col="contour_id",
        geometry_col="geometry",
        groupby=groupby,
        boundary_tolerance=boundary_tolerance,
        min_shared_boundary=min_shared_boundary,
        enclosure_min_fraction=enclosure_min_fraction,
    )
    return {
        "boundary_overlap": result.boundary_overlap.copy(),
        "enclosure": result.enclosure.copy(),
        "contour_summary": result.contour_summary.copy(),
        "group_boundary_overlap": result.group_boundary_overlap.copy(),
        "group_enclosure": result.group_enclosure.copy(),
    }


def _load_histoseg_topology(*, histoseg_root: str | Path | None):
    histoseg_contour = load_histoseg_submodule(
        "histoseg.contour",
        required=("summarize_contour_topology",),
        histoseg_root=histoseg_root,
        purpose="summarize_contour_topology()",
    )
    return histoseg_contour.summarize_contour_topology
