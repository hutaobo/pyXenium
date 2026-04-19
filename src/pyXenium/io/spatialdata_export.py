from __future__ import annotations

from pathlib import Path
from typing import Any

from .api import read_sdata, read_xenium, warn_unsupported_image_export_flags, write_xenium

DEFAULT_SPATIALDATA_STORE_NAME = "spatialdata.zarr"


def _sorted_keys(mapping: Any) -> list[str]:
    if mapping is None:
        return []
    if hasattr(mapping, "keys"):
        return sorted(str(key) for key in mapping.keys())
    return sorted(str(key) for key in mapping)


def export_xenium_to_spatialdata_zarr(
    base_path: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    cells_boundaries: bool = True,
    nucleus_boundaries: bool = True,
    cells_as_circles: bool | None = None,
    cells_labels: bool = True,
    nucleus_labels: bool = True,
    transcripts: bool = True,
    morphology_mip: bool = True,
    morphology_focus: bool = True,
    aligned_images: bool = True,
    cells_table: bool = True,
) -> dict[str, Any]:
    """
    Compatibility export wrapper that writes pyXenium's own SData schema.

    The output path and summary intentionally preserve the legacy
    ``export-spatialdata`` surface so existing scripts continue to work while
    pyXenium no longer depends on ``spatialdata`` or ``spatialdata_io``.
    """

    del n_jobs
    base_path = Path(base_path).expanduser()
    if output_path is None:
        output_path = base_path / DEFAULT_SPATIALDATA_STORE_NAME
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warn_unsupported_image_export_flags(
        morphology_focus=morphology_focus,
        morphology_mip=morphology_mip,
        aligned_images=aligned_images,
    )

    sdata = read_xenium(
        str(base_path),
        as_="sdata",
        include_transcripts=transcripts,
        include_boundaries=(cells_boundaries or nucleus_boundaries),
        include_images=(morphology_focus or morphology_mip or aligned_images),
    )

    if not cells_boundaries:
        sdata.shapes.pop("cell_boundaries", None)
    if not nucleus_boundaries:
        sdata.shapes.pop("nucleus_boundaries", None)
    if not cells_table:
        sdata.table = sdata.table[:0, :].copy()
    if not cells_labels or not nucleus_labels or cells_as_circles is not None:
        sdata.metadata.setdefault("compat", {})
        sdata.metadata["compat"].update(
            {
                "cells_labels_requested": bool(cells_labels),
                "nucleus_labels_requested": bool(nucleus_labels),
                "cells_as_circles_requested": cells_as_circles,
            }
        )

    payload = write_xenium(sdata, output_path, format="sdata", overwrite=overwrite)
    reloaded = read_sdata(output_path)
    summary = reloaded.component_summary()
    return {
        "base_path": str(base_path),
        "output_path": str(output_path),
        "images": summary["images"],
        "labels": summary["labels"],
        "points": summary["points"],
        "shapes": summary["shapes"],
        "tables": summary["tables"],
        "format": payload["format"],
        "version": payload["version"],
    }
