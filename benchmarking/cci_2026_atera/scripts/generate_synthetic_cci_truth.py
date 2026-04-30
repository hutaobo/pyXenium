from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_cci_truth(*, output_dir: str | Path, cells_per_type: int = 60, seed: int = 7) -> dict[str, str | int]:
    rng = np.random.default_rng(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    celltypes = ["SenderA", "ReceiverB", "BystanderC", "AbundantDecoyD"]
    centers = {
        "SenderA": (0.0, 0.0),
        "ReceiverB": (1.6, 0.0),
        "BystanderC": (7.0, 0.0),
        "AbundantDecoyD": (12.0, 0.0),
    }
    reference_rows = []
    expression_rows = []
    genes = ["PLANTED_L", "PLANTED_R", "CONTACT_L", "CONTACT_R", "DECOY_L", "DECOY_R", "TARGET_RESPONSE"]
    for celltype in celltypes:
        n_cells = cells_per_type * (3 if celltype == "AbundantDecoyD" else 1)
        center_x, center_y = centers[celltype]
        for idx in range(n_cells):
            cell_id = f"{celltype}_{idx:04d}"
            x = center_x + rng.normal(0.0, 0.55)
            y = center_y + rng.normal(0.0, 0.55)
            reference_rows.append({"cell_id": cell_id, "x": x, "y": y, "celltype": celltype})
            expr = dict.fromkeys(genes, 0.0)
            if celltype == "SenderA":
                expr["PLANTED_L"] = rng.gamma(8.0, 1.0)
                expr["CONTACT_L"] = rng.gamma(5.0, 1.0)
            if celltype == "ReceiverB":
                expr["PLANTED_R"] = rng.gamma(8.0, 1.0)
                expr["CONTACT_R"] = rng.gamma(5.0, 1.0)
                expr["TARGET_RESPONSE"] = rng.gamma(5.0, 1.0)
            if celltype == "AbundantDecoyD":
                expr["DECOY_L"] = rng.gamma(10.0, 1.0)
                expr["DECOY_R"] = rng.gamma(10.0, 1.0)
            expression_rows.append({"cell_id": cell_id, **expr})

    reference = pd.DataFrame(reference_rows)
    expression = pd.DataFrame(expression_rows).set_index("cell_id")
    t_and_c = pd.DataFrame(
        {
            "SenderA": [0.0, 1.0, 0.0, 1.0, 0.7, 0.7, 1.0],
            "ReceiverB": [1.0, 0.0, 1.0, 0.0, 0.7, 0.7, 0.0],
            "BystanderC": [0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.8],
            "AbundantDecoyD": [0.8, 0.8, 0.8, 0.8, 0.0, 0.0, 0.9],
        },
        index=genes,
    )
    structure_map = pd.DataFrame(
        [
            [0.0, 0.15, 0.75, 1.0],
            [0.15, 0.0, 0.70, 1.0],
            [0.75, 0.70, 0.0, 0.6],
            [1.0, 1.0, 0.6, 0.0],
        ],
        index=celltypes,
        columns=celltypes,
    )
    pairs = pd.DataFrame(
        [
            {"ligand": "PLANTED_L", "receptor": "PLANTED_R", "evidence_weight": 1.0, "interaction_mode": "secreted/paracrine", "truth_label": "positive"},
            {"ligand": "CONTACT_L", "receptor": "CONTACT_R", "evidence_weight": 1.0, "interaction_mode": "juxtacrine/contact", "truth_label": "positive"},
            {"ligand": "DECOY_L", "receptor": "DECOY_R", "evidence_weight": 1.0, "interaction_mode": "non-classic resource axis", "truth_label": "high_expression_decoy"},
            {"ligand": "PLANTED_L", "receptor": "DECOY_R", "evidence_weight": 0.5, "interaction_mode": "cci_resource_axis", "truth_label": "negative"},
        ]
    )
    downstream = pd.DataFrame([{"ligand": "PLANTED_L", "receptor": "PLANTED_R", "target": "TARGET_RESPONSE"}])

    files = {
        "reference_tsv": output / "reference.tsv",
        "expression_tsv": output / "expression.tsv",
        "t_and_c_tsv": output / "t_and_c.tsv",
        "structure_map_tsv": output / "structure_map.tsv",
        "interaction_pairs_tsv": output / "interaction_pairs.tsv",
        "downstream_targets_tsv": output / "downstream_targets.tsv",
        "truth_axes_tsv": output / "truth_axes.tsv",
    }
    reference.to_csv(files["reference_tsv"], sep="\t", index=False)
    expression.to_csv(files["expression_tsv"], sep="\t")
    t_and_c.to_csv(files["t_and_c_tsv"], sep="\t")
    structure_map.to_csv(files["structure_map_tsv"], sep="\t")
    pairs.to_csv(files["interaction_pairs_tsv"], sep="\t", index=False)
    downstream.to_csv(files["downstream_targets_tsv"], sep="\t", index=False)
    pairs.loc[pairs["truth_label"].eq("positive")].to_csv(files["truth_axes_tsv"], sep="\t", index=False)
    return {"n_cells": int(len(reference)), "n_pairs": int(len(pairs)), **{key: str(value) for key, value in files.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a topology-preserving synthetic CCI truth panel.")
    parser.add_argument("--output-dir", default="benchmarking/cci_2026_atera/data/synthetic_topology_truth")
    parser.add_argument("--cells-per-type", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    payload = generate_synthetic_cci_truth(output_dir=args.output_dir, cells_per_type=args.cells_per_type, seed=args.seed)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
