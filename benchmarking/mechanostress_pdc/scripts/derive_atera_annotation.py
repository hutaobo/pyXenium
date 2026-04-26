from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


S1_S5_PANEL = {
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
S1_TUMOR_LABELS = {
    "11q13 Invasive Tumor Cells",
    "11q13 Invasive Tumor Cells (G1/S)",
    "11q13 Invasive Tumor Cells (Mitotic)",
}


def build_annotation(dataset_root: Path) -> pd.DataFrame:
    groups = pd.read_csv(dataset_root / "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv")
    cells = pd.read_parquet(dataset_root / "cells.parquet")[["cell_id", "x_centroid", "y_centroid"]]
    resolved_to_s = {
        LABEL_ALIASES.get(label, label): s_class
        for s_class, labels in S1_S5_PANEL.items()
        for label in labels
    }
    frame = groups.merge(cells, on="cell_id", how="left")
    frame["cluster"] = frame["group"].astype(str)
    frame["s_class"] = frame["group"].map(resolved_to_s).fillna("Unassigned")
    frame["mechanostress_class"] = "Other"
    frame.loc[frame["group"].isin(S1_TUMOR_LABELS) | frame["s_class"].eq("S5"), "mechanostress_class"] = "Tumor"
    frame.loc[frame["group"].eq("CAFs, Invasive Associated"), "mechanostress_class"] = "Stromal"
    frame["axis_pool"] = "Other"
    frame.loc[frame["group"].eq("CAFs, Invasive Associated"), "axis_pool"] = "S1_CAF"
    frame.loc[frame["group"].eq("CAFs, DCIS Associated"), "axis_pool"] = "S4_CAF"
    frame.loc[frame["group"].eq("CXCL14+ Fibroblasts"), "axis_pool"] = "S3_CXCL14_fibroblast"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    annotation = build_annotation(args.dataset_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(args.output_csv, index=False)
    print(f"[mechanostress-pdc] wrote annotation table: {args.output_csv} ({len(annotation)} cells)")


if __name__ == "__main__":
    main()
