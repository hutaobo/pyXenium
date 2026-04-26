from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from pyXenium.gmi import summarize_gmi_pdc_runs


STAGE_ORDER = [
    "smoke_contour_top200_spatial50",
    "full_contour_top500_spatial100",
    "full_contour_top500_spatial100_stability",
    "validation_rna_only_qc20",
    "validation_spatial_only_qc20",
    "validation_no_coordinate_qc20",
    "sensitivity_top1000_spatial100_qc20",
    "sensitivity_all_nonempty_top500_spatial100",
]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _float(value: object) -> float | None:
    try:
        if value in (None, "", "NA", "NaN", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_metric(rows: list[dict[str, str]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _features(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("feature", "") for row in rows if row.get("feature")]


def _has_features(rows: list[dict[str, str]], names: tuple[str, ...]) -> bool:
    selected = set(_features(rows))
    return all(name in selected for name in names)


def _feature_class(feature: str) -> str:
    lowered = feature.lower()
    if "centroid" in lowered or "slide_" in lowered or lowered.endswith("_x") or lowered.endswith("_y"):
        return "coordinate"
    if "luminal_like_amorphous_dcis" in lowered or "state_fraction" in lowered or "composition" in lowered:
        return "composition"
    if "rim" in lowered or "edge" in lowered or "gradient" in lowered or "contrast" in lowered:
        return "rim_edge_gradient"
    if "caf" in lowered or "ecm" in lowered or "collagen" in lowered:
        return "caf_ecm"
    if "vascular" in lowered or "pericyte" in lowered or "endothelial" in lowered:
        return "vascular_pericyte"
    if "myeloid" in lowered or "immune" in lowered or "macrophage" in lowered:
        return "immune_myeloid"
    return "other"


def _summarize_stage(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    main = _read_tsv(run_dir / "main_effects.tsv")
    cv = _read_tsv(run_dir / "cv_metrics.tsv")
    stability = _read_tsv(run_dir / "stability.tsv")
    heterogeneity = _read_tsv(run_dir / "heterogeneity.tsv")
    figures_dir = run_dir / "figures"
    figures = sorted(str(path.name) for path in figures_dir.glob("*")) if figures_dir.exists() else []

    return {
        "stage_id": run_dir.name,
        "status": "completed" if summary else "missing_summary",
        "output_dir": str(run_dir),
        "n_contours": summary.get("n_contours"),
        "n_retained_contours": summary.get("n_retained_contours"),
        "n_dropped_contours": summary.get("n_dropped_contours"),
        "n_features": summary.get("n_features"),
        "n_rna_features": summary.get("n_rna_features"),
        "n_spatial_features": summary.get("n_spatial_features"),
        "selected_main_effects": summary.get("selected_main_effects"),
        "selected_interactions": summary.get("selected_interactions"),
        "train_metrics": summary.get("train_metrics", {}),
        "main_effects": main,
        "main_features": _features(main),
        "cv_mean": {
            "auc": _mean_metric(cv, "auc"),
            "accuracy": _mean_metric(cv, "accuracy"),
            "sensitivity": _mean_metric(cv, "sensitivity"),
            "specificity": _mean_metric(cv, "specificity"),
        },
        "cv_n_folds": len(cv),
        "stability_top": stability[:10],
        "heterogeneity_rows": len(heterogeneity),
        "figures": figures,
    }


def _build_biological_summary(stage_map: dict[str, dict[str, object]]) -> dict[str, object]:
    full = stage_map.get("full_contour_top500_spatial100", {})
    stability = stage_map.get("full_contour_top500_spatial100_stability", {})
    rna_only = stage_map.get("validation_rna_only_qc20", {})
    spatial_only = stage_map.get("validation_spatial_only_qc20", {})
    no_coordinate = stage_map.get("validation_no_coordinate_qc20", {})
    top1000 = stage_map.get("sensitivity_top1000_spatial100_qc20", {})
    all_nonempty = stage_map.get("sensitivity_all_nonempty_top500_spatial100", {})

    target = ("NIBAN1", "SORL1")
    spatial_features = list(spatial_only.get("main_features", []))
    spatial_classes = {feature: _feature_class(feature) for feature in spatial_features}

    full_features = set(full.get("main_features", []))
    top1000_features = set(top1000.get("main_features", []))
    all_nonempty_features = set(all_nonempty.get("main_features", []))

    return {
        "primary_qc20_main_features": list(full.get("main_features", [])),
        "primary_qc20_selected_niban1_sorl1": _has_features(full.get("main_effects", []), target),
        "stability_top_features": stability.get("stability_top", []),
        "rna_only_retains_niban1_sorl1": _has_features(rna_only.get("main_effects", []), target),
        "no_coordinate_retains_niban1_sorl1": _has_features(no_coordinate.get("main_effects", []), target),
        "spatial_only_features": spatial_features,
        "spatial_only_feature_classes": spatial_classes,
        "spatial_only_coordinate_driven": any(value == "coordinate" for value in spatial_classes.values()),
        "spatial_only_composition_driven": any(value == "composition" for value in spatial_classes.values()),
        "top1000_features": list(top1000_features),
        "top1000_new_features_vs_qc20": sorted(top1000_features - full_features),
        "all_nonempty_features": list(all_nonempty_features),
        "all_nonempty_changes_primary_features": sorted(all_nonempty_features.symmetric_difference(full_features)),
        "interpretation": (
            "The QC20 S1/S5 contrast is primarily driven by an S5/DCIS RNA expression program "
            "led by NIBAN1 and SORL1 when those features are retained by the full, RNA-only, "
            "and no-coordinate stages. Spatial-only signal should be read as contour context, "
            "especially composition, unless coordinate features dominate. All-nonempty sensitivity "
            "is a QC stress test and does not replace the QC20 primary result."
        ),
    }


def _render_markdown(payload: dict[str, object]) -> str:
    stages = payload["stage_summaries"]
    biological = payload["biological_summary"]
    lines = [
        "# PDC contour-GMI validation summary",
        "",
        f"- PDC root: `{payload['pdc_root']}`",
        f"- Completed stages: {payload['n_completed']}/{payload['n_expected']}",
        f"- All expected stages complete: {str(payload['all_expected_stages_completed']).lower()}",
        "",
        "## Stage results",
        "",
        "| Stage | Contours | Features | Main effects | Train AUC | CV mean AUC | Top selected features |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for stage in stages:
        metrics = stage.get("train_metrics") or {}
        top = ", ".join(stage.get("main_features", [])[:5]) or "-"
        lines.append(
            "| `{stage_id}` | {contours} | {features} | {main} | {auc} | {cv_auc} | {top} |".format(
                stage_id=stage["stage_id"],
                contours=stage.get("n_contours") or "-",
                features=stage.get("n_features") or "-",
                main=stage.get("selected_main_effects") if stage.get("selected_main_effects") is not None else "-",
                auc=metrics.get("auc", "-"),
                cv_auc=(stage.get("cv_mean") or {}).get("auc") or "-",
                top=top,
            )
        )

    lines.extend(
        [
            "",
            "## Biological readout",
            "",
            "- Primary QC20 model selected: "
            + (", ".join(biological.get("primary_qc20_main_features", [])) or "-"),
            f"- RNA-only retained NIBAN1/SORL1: {biological.get('rna_only_retains_niban1_sorl1')}",
            f"- No-coordinate retained NIBAN1/SORL1: {biological.get('no_coordinate_retains_niban1_sorl1')}",
            "- Spatial-only selected: "
            + (", ".join(biological.get("spatial_only_features", [])) or "-"),
            f"- Spatial-only coordinate-driven: {biological.get('spatial_only_coordinate_driven')}",
            f"- Spatial-only composition-driven: {biological.get('spatial_only_composition_driven')}",
            "- Top1000 new features versus QC20: "
            + (", ".join(biological.get("top1000_new_features_vs_qc20", [])) or "-"),
            "- All-nonempty feature differences versus QC20: "
            + (", ".join(biological.get("all_nonempty_changes_primary_features", [])) or "-"),
            "",
            biological.get("interpretation", ""),
            "",
            "## Caveats",
            "",
            "- QC20 remains the primary result; all-nonempty is a sensitivity analysis.",
            "- GMI is sparse and sample-size sensitive, so selected features should be interpreted with the controls.",
            "- PDC artifacts are on scratch storage and should be archived if long-term retention is needed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PDC contour-GMI stage summaries.")
    parser.add_argument("--pdc-root", default="/cfs/klemming/scratch/h/hutaobo/pyxenium_gmi_contour_2026-04")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    base_payload = summarize_gmi_pdc_runs(args.pdc_root)
    runs_dir = Path(args.pdc_root) / "runs"
    stage_dirs = [runs_dir / stage_id for stage_id in STAGE_ORDER if (runs_dir / stage_id).exists()]
    stage_summaries = [_summarize_stage(path) for path in stage_dirs]
    stage_map = {stage["stage_id"]: stage for stage in stage_summaries}
    payload = {
        **base_payload,
        "expected_stages": STAGE_ORDER,
        "n_expected": len(STAGE_ORDER),
        "n_completed_expected": sum(
            1 for stage_id in STAGE_ORDER if stage_map.get(stage_id, {}).get("status") == "completed"
        ),
        "all_expected_stages_completed": all(
            stage_map.get(stage_id, {}).get("status") == "completed" for stage_id in STAGE_ORDER
        ),
        "stage_summaries": stage_summaries,
        "biological_summary": _build_biological_summary(stage_map),
    }
    payload["n_completed"] = payload["n_completed_expected"]
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
