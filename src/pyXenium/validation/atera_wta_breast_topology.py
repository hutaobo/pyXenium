from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pyXenium.analysis import (
    ligand_receptor_topology_analysis,
    pathway_topology_analysis,
)
from pyXenium.io import read_xenium

DEFAULT_ATERA_WTA_BREAST_DATASET_PATH = (
    r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs"
)
DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR = r"sfplot_tbc_formal_wta\results"
DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS = "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"
DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID = "atera_wta_ffpe_breast"

DEFAULT_LR_SMOKE_PANEL = pd.DataFrame(
    [
        {"ligand": "CSF1", "receptor": "CSF1R", "evidence_weight": 1.0},
        {"ligand": "CXCL12", "receptor": "CXCR4", "evidence_weight": 1.0},
        {"ligand": "TGFB1", "receptor": "TGFBR2", "evidence_weight": 1.0},
        {"ligand": "JAG1", "receptor": "NOTCH1", "evidence_weight": 1.0},
        {"ligand": "DLL4", "receptor": "NOTCH3", "evidence_weight": 1.0},
    ]
)

DEFAULT_PATHWAY_PANEL: dict[str, list[str]] = {
    "MacrophageProgram": ["C1QA", "CD163", "CSF1R", "SIGLEC1", "C3"],
    "PlasmaProgram": ["IGHA1", "IGHA2", "JCHAIN", "IGHM"],
    "VascularProgram": ["CDH5", "MMRN2", "EPAS1", "KDR", "FLT1"],
    "BasalDCISProgram": ["SOSTDC1", "KLK5", "ITGB6", "DSC3", "KRT23"],
    "ApocrineProgram": ["TAT", "MYBPC1"],
    "LuminalAmorphousProgram": ["HSPB8", "CLIC6", "PIP", "ESR1", "PGR"],
}


def _resolve_output_dir(output_dir: str | None, manuscript_mode: bool, manuscript_root: str) -> Path | None:
    if output_dir:
        return Path(output_dir)
    if manuscript_mode:
        return Path(manuscript_root) / "atera_wta_breast_topology"
    return None


def _default_tbc_results_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR


def _load_atera_wta_breast_adata(dataset_root: str | Path):
    return read_xenium(
        str(dataset_root),
        as_="anndata",
        prefer="h5",
        include_transcripts=False,
        include_boundaries=False,
        include_images=False,
        clusters_relpath=DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS,
        cluster_column_name="cluster",
        cells_parquet="cells.parquet",
    )


def _load_metrics_summary(dataset_root: Path) -> dict[str, Any]:
    metrics_path = dataset_root / "metrics_summary.csv"
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {}
    row = metrics.iloc[0]
    payload = {}
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


def _cluster_counts(adata) -> pd.DataFrame:
    counts = adata.obs["cluster"].astype(str).value_counts()
    return pd.DataFrame({"cluster": counts.index.astype(str), "n_cells": counts.to_numpy(dtype=int)})


def _lr_pair_summary(scores: pd.DataFrame, ligand: str, receptor: str) -> dict[str, Any]:
    pair_scores = scores.loc[(scores["ligand"] == ligand) & (scores["receptor"] == receptor)].copy()
    if pair_scores.empty:
        return {
            "ligand": ligand,
            "receptor": receptor,
            "best_sender_celltype": None,
            "best_receiver_celltype": None,
            "best_score": None,
            "top_rows": [],
        }
    pair_scores = pair_scores.sort_values("LR_score", ascending=False).reset_index(drop=True)
    best = pair_scores.iloc[0]
    top_rows = []
    for _, row in pair_scores.head(5).iterrows():
        top_rows.append(
            {
                "sender_celltype": str(row["sender_celltype"]),
                "receiver_celltype": str(row["receiver_celltype"]),
                "LR_score": float(row["LR_score"]),
            }
        )
    return {
        "ligand": ligand,
        "receptor": receptor,
        "best_sender_celltype": str(best["sender_celltype"]),
        "best_receiver_celltype": str(best["receiver_celltype"]),
        "best_score": float(best["LR_score"]),
        "top_rows": top_rows,
    }


def _rank_of_pair(pair_scores: pd.DataFrame, sender: str, receiver: str) -> int | None:
    if pair_scores.empty:
        return None
    ranked = pair_scores.sort_values("LR_score", ascending=False).reset_index(drop=True)
    matches = ranked.index[(ranked["sender_celltype"] == sender) & (ranked["receiver_celltype"] == receiver)]
    if len(matches) == 0:
        return None
    return int(matches[0]) + 1


def _build_lr_acceptance(scores: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    csf1 = scores.loc[(scores["ligand"] == "CSF1") & (scores["receptor"] == "CSF1R")].copy()
    csf1 = csf1.sort_values("LR_score", ascending=False).reset_index(drop=True)
    csf1_sender = str(csf1.iloc[0]["sender_celltype"]) if not csf1.empty else None
    checks.append(
        {
            "check": "CSF1-CSF1R top sender should not be Mast Cells",
            "ligand": "CSF1",
            "receptor": "CSF1R",
            "observed_top_sender": csf1_sender,
            "pass": bool(csf1_sender is not None and csf1_sender != "Mast Cells"),
        }
    )

    cxcl12 = scores.loc[(scores["ligand"] == "CXCL12") & (scores["receptor"] == "CXCR4")].copy()
    cxcl12_rank = _rank_of_pair(cxcl12, "CAFs, DCIS Associated", "T Lymphocytes")
    checks.append(
        {
            "check": "CXCL12-CXCR4 should keep CAFs, DCIS Associated -> T Lymphocytes high-ranking",
            "ligand": "CXCL12",
            "receptor": "CXCR4",
            "observed_rank": cxcl12_rank,
            "pass": bool(cxcl12_rank is not None and cxcl12_rank <= 5),
        }
    )

    dll4 = scores.loc[(scores["ligand"] == "DLL4") & (scores["receptor"] == "NOTCH3")].copy()
    dll4 = dll4.sort_values("LR_score", ascending=False).reset_index(drop=True)
    if dll4.empty:
        dll4_sender = None
        dll4_receiver = None
    else:
        dll4_sender = str(dll4.iloc[0]["sender_celltype"])
        dll4_receiver = str(dll4.iloc[0]["receiver_celltype"])
    checks.append(
        {
            "check": "DLL4-NOTCH3 top hit should be Endothelial Cells -> Pericytes",
            "ligand": "DLL4",
            "receptor": "NOTCH3",
            "observed_top_sender": dll4_sender,
            "observed_top_receiver": dll4_receiver,
            "pass": bool(dll4_sender == "Endothelial Cells" and dll4_receiver == "Pericytes"),
        }
    )
    return checks


def _pathway_best_assignments(pathway_to_cell: pd.DataFrame) -> list[dict[str, Any]]:
    if pathway_to_cell.empty:
        return []
    rows: list[dict[str, Any]] = []
    for pathway in pathway_to_cell.index.astype(str):
        best_celltype = str(pathway_to_cell.loc[pathway].astype(float).idxmin())
        best_distance = float(pathway_to_cell.loc[pathway].astype(float).min())
        rows.append(
            {
                "pathway": pathway,
                "best_celltype": best_celltype,
                "best_distance": best_distance,
            }
        )
    return rows


def _build_pathway_acceptance(pathway_to_cell: pd.DataFrame) -> list[dict[str, Any]]:
    expected = {
        "MacrophageProgram": {"Macrophages"},
        "PlasmaProgram": {"Plasma Cells"},
        "VascularProgram": {"Endothelial Cells", "Pericytes"},
        "BasalDCISProgram": {"Basal-like Structured DCIS Cells"},
        "ApocrineProgram": {"Apocrine Cells"},
        "LuminalAmorphousProgram": {"Luminal-like Amorphous DCIS Cells"},
    }
    checks: list[dict[str, Any]] = []
    if pathway_to_cell.empty:
        for pathway, targets in expected.items():
            checks.append(
                {
                    "pathway": pathway,
                    "expected_best_celltypes": sorted(targets),
                    "observed_best_celltype": None,
                    "pass": False,
                }
            )
        return checks

    for pathway, targets in expected.items():
        observed = None
        if pathway in pathway_to_cell.index:
            observed = str(pathway_to_cell.loc[pathway].astype(float).idxmin())
        checks.append(
            {
                "pathway": pathway,
                "expected_best_celltypes": sorted(targets),
                "observed_best_celltype": observed,
                "pass": bool(observed in targets),
            }
        )
    return checks


def build_serializable_breast_topology_summary(study: dict) -> dict[str, Any]:
    adata = study["adata"]
    cluster_counts = _cluster_counts(adata)
    lr_scores = study["lr"]["scores"]
    pathway_to_cell = study["pathway"]["pathway_to_cell"]
    pathway_activity_to_cell = study["pathway"]["pathway_activity_to_cell"]

    metrics_summary = _load_metrics_summary(Path(study["dataset_root"]))
    experiment_metadata = _load_experiment_metadata(Path(study["dataset_root"]))
    lr_pair_summaries = [
        _lr_pair_summary(lr_scores, ligand=row["ligand"], receptor=row["receptor"])
        for _, row in DEFAULT_LR_SMOKE_PANEL.iterrows()
    ]

    payload = {
        "sample_id": study["sample_id"],
        "dataset_root": study["dataset_root"],
        "tbc_results": study["tbc_results"],
        "output_dir": study.get("artifact_dir"),
        "n_cells": int(adata.n_obs),
        "n_rna_features": int(adata.n_vars),
        "cluster_count": int(cluster_counts.shape[0]),
        "topology_celltype_count": int(pathway_to_cell.shape[1]) if not pathway_to_cell.empty else 0,
        "unassigned_cells": int(cluster_counts.loc[cluster_counts["cluster"] == "Unassigned", "n_cells"].sum()),
        "cluster_cell_counts": cluster_counts.to_dict(orient="records"),
        "metrics_summary": metrics_summary,
        "experiment_metadata": experiment_metadata,
        "lr_smoke_panel": DEFAULT_LR_SMOKE_PANEL.to_dict(orient="records"),
        "pathway_smoke_panel": [
            {"pathway": pathway, "genes": genes} for pathway, genes in DEFAULT_PATHWAY_PANEL.items()
        ],
        "lr_pair_summaries": lr_pair_summaries,
        "lr_acceptance": _build_lr_acceptance(lr_scores),
        "pathway_primary_best": _pathway_best_assignments(pathway_to_cell),
        "pathway_activity_best": _pathway_best_assignments(pathway_activity_to_cell),
        "pathway_acceptance": _build_pathway_acceptance(pathway_to_cell),
        "runtime_seconds": float(study["runtime_seconds"]),
        "files": study["files"],
    }
    return payload


def render_atera_wta_breast_topology_report(payload: dict) -> str:
    lines = [
        "# Atera WTA Breast Topology Reproducibility Bundle",
        "",
        f"Sample ID: `{payload['sample_id']}`",
        f"Dataset root: `{payload['dataset_root']}`",
        f"t_and_c / StructureMap anchor source: `{payload['tbc_results']}`",
        "",
        "## Core Summary",
        "",
        f"- Cells loaded: `{payload['n_cells']}`",
        f"- RNA features loaded: `{payload['n_rna_features']}`",
        f"- Cluster count: `{payload['cluster_count']}`",
        f"- Topology celltype count: `{payload['topology_celltype_count']}`",
        f"- Unassigned cells: `{payload['unassigned_cells']}`",
        f"- panel_num_targets_predesigned: `{payload['experiment_metadata'].get('panel_num_targets_predesigned')}`",
        f"- median_transcripts_per_cell: `{payload['metrics_summary'].get('median_transcripts_per_cell')}`",
        f"- Runtime (s): `{payload['runtime_seconds']:.2f}`",
        "",
        "## LR Smoke Panel",
        "",
    ]

    for entry in payload["lr_pair_summaries"]:
        ligand = entry["ligand"]
        receptor = entry["receptor"]
        best_sender = entry["best_sender_celltype"]
        best_receiver = entry["best_receiver_celltype"]
        best_score = entry["best_score"]
        lines.append(
            f"- `{ligand}-{receptor}`: top `{best_sender} -> {best_receiver}`"
            + (f" (`{best_score:.4f}`)" if best_score is not None else "")
        )

    lines.extend(["", "## LR Acceptance", ""])
    for check in payload["lr_acceptance"]:
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(f"- `{status}` {check['check']}")

    lines.extend(["", "## Pathway Primary Results", ""])
    for entry in payload["pathway_primary_best"]:
        lines.append(
            f"- `{entry['pathway']}` -> `{entry['best_celltype']}` (`distance={entry['best_distance']:.4f}`)"
        )

    lines.extend(["", "## Pathway Acceptance", ""])
    for check in payload["pathway_acceptance"]:
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(
            f"- `{status}` `{check['pathway']}` expected `{', '.join(check['expected_best_celltypes'])}`, observed `{check['observed_best_celltype']}`"
        )

    lines.extend(["", "## Fixed Output Files", ""])
    for key, value in payload["files"].items():
        if isinstance(value, list):
            lines.append(f"- `{key}`: `{len(value)}` file(s)")
        else:
            lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    return "\n".join(lines)


def write_atera_wta_breast_topology_artifacts(study: dict, output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_serializable_breast_topology_summary(study)
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_atera_wta_breast_topology_report(payload), encoding="utf-8")
    return out


def run_atera_wta_breast_topology(
    *,
    dataset_root: str,
    tbc_results: str | None = None,
    output_dir: str | None = None,
    manuscript_mode: bool = False,
    manuscript_root: str = "manuscript",
    sample_id: str = DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID,
    export_figures: bool = True,
    write_h5ad: str | None = None,
) -> dict:
    start = time.perf_counter()
    dataset_root_path = Path(dataset_root)
    resolved_tbc_results = Path(tbc_results) if tbc_results else _default_tbc_results_path(dataset_root_path)
    resolved_output_dir = _resolve_output_dir(output_dir, manuscript_mode, manuscript_root)

    adata = _load_atera_wta_breast_adata(dataset_root_path)

    lr_result = ligand_receptor_topology_analysis(
        adata=adata,
        lr_pairs=DEFAULT_LR_SMOKE_PANEL,
        output_dir=resolved_output_dir,
        tbc_results=resolved_tbc_results,
        cluster_col="cluster",
        cell_id_col="cell_id",
        x_col="x",
        y_col="y",
        anchor_mode="precomputed",
        top_n_pairs=len(DEFAULT_LR_SMOKE_PANEL),
        min_cross_edges=50,
        export_figures=export_figures,
    )

    pathway_result = pathway_topology_analysis(
        adata=adata,
        pathway_definitions=DEFAULT_PATHWAY_PANEL,
        output_dir=resolved_output_dir,
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

    runtime_seconds = time.perf_counter() - start
    merged_files = {}
    merged_files.update(lr_result.get("files", {}))
    merged_files.update(pathway_result.get("files", {}))

    study = {
        "sample_id": sample_id,
        "dataset_root": str(dataset_root_path),
        "tbc_results": str(resolved_tbc_results),
        "adata": adata,
        "lr": lr_result,
        "pathway": pathway_result,
        "files": merged_files,
        "runtime_seconds": runtime_seconds,
    }

    artifact_dir = write_atera_wta_breast_topology_artifacts(study, resolved_output_dir)
    if artifact_dir is not None:
        study["artifact_dir"] = str(artifact_dir)
        study["files"]["summary_json"] = str(Path(artifact_dir) / "summary.json")
        study["files"]["report_md"] = str(Path(artifact_dir) / "report.md")

    payload = build_serializable_breast_topology_summary(study)
    study["payload"] = payload

    if artifact_dir is not None:
        Path(artifact_dir, "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        Path(artifact_dir, "report.md").write_text(render_atera_wta_breast_topology_report(payload), encoding="utf-8")

    if write_h5ad:
        h5ad_path = Path(write_h5ad)
        h5ad_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(h5ad_path)

    return study


__all__ = [
    "DEFAULT_ATERA_WTA_BREAST_CELL_GROUPS",
    "DEFAULT_ATERA_WTA_BREAST_DATASET_PATH",
    "DEFAULT_ATERA_WTA_BREAST_SAMPLE_ID",
    "DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR",
    "DEFAULT_LR_SMOKE_PANEL",
    "DEFAULT_PATHWAY_PANEL",
    "build_serializable_breast_topology_summary",
    "render_atera_wta_breast_topology_report",
    "run_atera_wta_breast_topology",
    "write_atera_wta_breast_topology_artifacts",
]
