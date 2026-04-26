from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from pyXenium.contour import generate_xenium_explorer_annotations


GRAPHCLUST_RELPATH = Path("analysis/analysis/clustering/gene_expression_graphclust/clusters.csv")
CELL_GROUPS_FILENAME = "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"
CELLS_PARQUET_RELPATH = Path("cells.parquet")
S1_S5_GEOJSON_FILENAME = "xenium_explorer_annotations.s1_s5.generated.geojson"
PIXEL_SIZE_UM = 0.2125

S1_S5_PANEL: dict[str, list[str]] = {
    "S1": [
        "CAFs Invasive Associated",
        "11q13 Invasive Tumor Cells",
        "11q13 Invasive Tumor Cells (G1/S)",
        "11q13 Invasive Tumor Cells (Mitotic)",
    ],
    "S2": [
        "Basal-like Structured DCIS Cells",
        "Dendritic Cells",
        "B Cells",
        "T Lymphocytes",
    ],
    "S3": [
        "Mast Cells",
        "Myeloid Cells",
        "Macrophages",
        "CXCL14+ Fibroblasts",
        "Endothelial Cells",
        "Pericytes",
    ],
    "S4": [
        "Myoepithelial Cells",
        "CAFs DCIS Associated",
        "Plasma Cells",
    ],
    "S5": [
        "Apocrine Cells",
        "Luminal-like Amorphous DCIS Cells",
    ],
}
LABEL_ALIASES = {
    "CAFs Invasive Associated": "CAFs, Invasive Associated",
    "CAFs DCIS Associated": "CAFs, DCIS Associated",
}
S_COLORS = {"S1": "#C43C39", "S2": "#3F7CAC", "S3": "#4A9D67", "S4": "#B9802D", "S5": "#9A5BC7"}
S_DESCRIPTIONS = {
    "S1": "Invasive tumor + CAFs",
    "S2": "Basal DCIS + lymphoid",
    "S3": "Myeloid-vascular-stromal",
    "S4": "Myoepithelial/DCIS CAF/plasma",
    "S5": "Apocrine/luminal DCIS",
}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Atera contour input: {path}")


def _build_structures(dataset_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    cluster_df = pd.read_csv(dataset_root / GRAPHCLUST_RELPATH)
    cell_groups_df = pd.read_csv(dataset_root / CELL_GROUPS_FILENAME).rename(
        columns={"cell_id": "Barcode", "group": "cell_group"}
    )
    cluster_bridge = cluster_df.merge(
        cell_groups_df[["Barcode", "cell_group", "color"]],
        on="Barcode",
        how="inner",
    )
    label_to_clusters = (
        cluster_bridge.groupby("cell_group")["Cluster"]
        .apply(lambda values: sorted(pd.Series(values).dropna().astype(int).unique().tolist()))
        .to_dict()
    )
    label_counts = cluster_bridge.groupby("cell_group").size().to_dict()

    structures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for idx, (structure_name, requested_labels) in enumerate(S1_S5_PANEL.items(), start=1):
        cluster_ids: list[int] = []
        resolved_labels: list[str] = []
        input_cell_count = 0
        for requested_label in requested_labels:
            resolved_label = LABEL_ALIASES.get(requested_label, requested_label)
            if resolved_label not in label_to_clusters:
                raise KeyError(f"Missing Atera cell group: {requested_label!r} -> {resolved_label!r}")
            cluster_ids.extend(label_to_clusters[resolved_label])
            resolved_labels.append(resolved_label)
            input_cell_count += int(label_counts[resolved_label])
        structures.append(
            {
                "structure_name": structure_name,
                "cluster_ids": sorted(set(cluster_ids)),
                "structure_color": S_COLORS[structure_name],
                "structure_id": idx,
                "member_labels": tuple(resolved_labels),
            }
        )
        rows.append(
            {
                "structure_name": structure_name,
                "description": S_DESCRIPTIONS[structure_name],
                "cluster_ids": json.dumps(sorted(set(cluster_ids))),
                "member_labels": ", ".join(resolved_labels),
                "input_cell_count": input_cell_count,
            }
        )
    return structures, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Atera S1-S5 contour GeoJSON used by contour-GMI on PDC."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pdc-root", required=True)
    parser.add_argument("--histoseg-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-cells", type=int, default=500)
    parser.add_argument("--min-component-pixels", type=int, default=180)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pdc_root = Path(args.pdc_root).expanduser().resolve()
    target_geojson = dataset_root / S1_S5_GEOJSON_FILENAME
    output_dir = pdc_root / "contour_generation" / "atera_s1_s5_contour_application"
    manifest_path = output_dir / "s1_s5_geojson_manifest.json"

    for relpath in (GRAPHCLUST_RELPATH, CELLS_PARQUET_RELPATH):
        _require_file(dataset_root / relpath)
    _require_file(dataset_root / CELL_GROUPS_FILENAME)

    output_dir.mkdir(parents=True, exist_ok=True)
    if target_geojson.exists() and not args.force:
        manifest = {
            "status": "exists",
            "geojson": str(target_geojson),
            "output_dir": str(output_dir),
            "source": "https://pyxenium.readthedocs.io/en/latest/tutorials/contour_s1_s5_breast.html",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    structures, structure_table = _build_structures(dataset_root)
    structure_table.to_csv(output_dir / "s1_s5_structure_definitions.tsv", sep="\t", index=False)

    histoseg_root = Path(args.histoseg_root).expanduser().resolve() if args.histoseg_root else None
    if histoseg_root is not None and not histoseg_root.exists():
        histoseg_root = None

    artifacts = generate_xenium_explorer_annotations(
        dataset_root,
        structures=structures,
        output_relpath=output_dir,
        clusters_relpath=GRAPHCLUST_RELPATH,
        cells_parquet_relpath=CELLS_PARQUET_RELPATH,
        histoseg_root=histoseg_root,
        min_cells=args.min_cells,
        min_component_pixels=args.min_component_pixels,
        xenium_pixel_size_um=PIXEL_SIZE_UM,
    )
    shutil.copy2(artifacts["geojson"], target_geojson)

    manifest = {
        "status": "generated",
        "geojson": str(target_geojson),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "histoseg_root": str(histoseg_root) if histoseg_root is not None else None,
        "min_cells": args.min_cells,
        "min_component_pixels": args.min_component_pixels,
        "xenium_pixel_size_um": PIXEL_SIZE_UM,
        "source": "https://pyxenium.readthedocs.io/en/latest/tutorials/contour_s1_s5_breast.html",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
