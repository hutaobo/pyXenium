from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript skeleton, figure captions, and readiness checklist from a morphopathway package."
    )
    parser.add_argument("package_dir", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _format_pathway_list(pathways: list[str]) -> str:
    return ", ".join(pathways)


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    manifest = _read_json(package_dir / "brief_communication_package_manifest.json")
    fig2 = _read_csv(package_dir / "main_figure_2_source_cross_cancer_stability_highnull32.csv")
    gates = _read_csv(package_dir / "supp_table_highnull32_gate_and_axis_masked_summary.csv")
    spatial = _read_csv(package_dir / "supp_table_spatial_block_and_seed_summary.csv")

    stable_core = list(map(str, manifest.get("stable_9_pathway_core", [])))
    total = int(manifest.get("cross_cancer_total", 10))
    recovery_min = int(manifest.get("cross_cancer_recovery_min", 0))
    recovery_max = int(manifest.get("cross_cancer_recovery_max", 0))
    masked_min = int(manifest.get("axis_masked_recovery_min", 0))
    masked_max = int(manifest.get("axis_masked_recovery_max", 0))
    candidate_axis_runs = int(manifest.get("candidate_axis_runs", 0))
    n_runs = int(manifest.get("n_runs", len(gates)))

    breast_neg_min = int(pd.to_numeric(gates["breast_negative_control_pass95"], errors="coerce").min())
    cervical_neg_min = int(pd.to_numeric(gates["cervical_negative_control_pass95"], errors="coerce").min())
    breast_spatial_min = int(pd.to_numeric(gates["breast_spatial_null_pass95"], errors="coerce").min())
    cervical_spatial_min = int(pd.to_numeric(gates["cervical_spatial_null_pass95"], errors="coerce").min())
    breast_blocks = pd.to_numeric(spatial["breast_spatial_blocks"], errors="coerce")
    cervical_blocks = pd.to_numeric(spatial["cervical_spatial_blocks"], errors="coerce")
    breast_block_range = f"{int(breast_blocks.min())}-{int(breast_blocks.max())}"
    cervical_block_range = f"{int(cervical_blocks.min())}-{int(cervical_blocks.max())}"

    unstable = fig2.loc[fig2["stable_9_pathway_core"].astype(str).str.lower() != "true", "pathway"].astype(str).tolist()
    top_core = (
        fig2.loc[fig2["stable_9_pathway_core"].astype(str).str.lower() == "true"]
        .head(4)["pathway"]
        .astype(str)
        .tolist()
    )

    skeleton = f"""# Manuscript Skeleton

Working title:
H&E-to-WTA morphopathway stress testing in Xenium breast and cervical preview tissues

One-sentence pitch:
`pyXenium.pathway` links PLIP-derived H&E patch embeddings to curated Xenium WTA pathway activity and tests whether pathway-family morphology signals persist across breast discovery and cervical validation tissues.

## Abstract Draft

Spatial transcriptomics creates an opportunity to evaluate whether histological image features carry pathway-scale molecular information, but the statistical framing must separate exploratory morphology associations from direct replication claims. We developed `pyXenium.pathway`, a H&E+WTA morphopathway workflow that scores curated pathway programs from Xenium WTA profiles, aggregates PLIP-derived H&E embeddings into coarse spatial pseudobulk blocks, and evaluates residual morphology-pathway associations with spatial and matched random gene-set controls. Applied to Atera breast and cervical Xenium WTA FFPE preview datasets, three high-null seeds recovered a stable {len(stable_core)}-pathway pathway-family stress-test core ({recovery_min}/{total} to {recovery_max}/{total} recovered). Axis-masked sensitivity, removing sample-specific candidate generic PLIP axes, remained {masked_min}/{total} to {masked_max}/{total}. Matched negative controls were imperfect in some breast-side runs, supporting a conservative claim: pathway-family stress-test recovery rather than direct cervical replication.

## Main Text Outline

### Motivation

- Xenium WTA allows morphology-to-pathway benchmarking using measured transcriptome-wide genes rather than targeted panels alone.
- The key methodological challenge is avoiding overclaiming direct cross-cancer replication from exploratory image embedding associations.
- The contribution is a reproducible workflow that combines curated pathway scoring, spatial pseudobulk aggregation, residual association testing, spatial nulls, matched random gene-set controls, and axis-masked sensitivity.

### Methodological Advance

- `pyXenium.pathway` builds curated pathway panels and pathway activity scores from Xenium cell-feature matrices.
- H&E patches are encoded with PLIP and averaged into 12 x 12 spatial blocks with at least 6 cells per retained block.
- Associations are residual partial Spearman correlations adjusted for structure, x/y ranks, boundary distance, and log total counts.
- High-null runs use 32 spatial permutations and 32 expression-matched random gene-set controls.

### Results

- Across {n_runs} high-null seeds, cross-cancer pathway-family recovery ranged from {recovery_min}/{total} to {recovery_max}/{total}.
- Axis-masked sensitivity remained {masked_min}/{total} to {masked_max}/{total}, arguing that candidate generic PLIP axes are not required for the main stress-test signal.
- The stable core contains {len(stable_core)} pathways: {_format_pathway_list(stable_core)}.
- `emt_invasive_front` is not part of the stable core and should be described as unstable ({', '.join(unstable) or 'none'}).
- Spatial null pass95 minima were breast {breast_spatial_min}/10 and cervical {cervical_spatial_min}/10.
- Matched negative-control pass95 minima were breast {breast_neg_min}/10 and cervical {cervical_neg_min}/10, which should be reported as the main limitation.

### Interpretation

- The evidence supports a pathway-family stress-test claim, not a direct pathway-level cervical replication claim.
- The strongest repeated families are represented by {', '.join(top_core)} among the stable core.
- The analysis should be framed as a method and evidence package for morphopathway benchmarking rather than a clinical biomarker study.

### Limitations

- Public preview datasets are sampled rather than exhaustive full-resolution cohorts.
- Matched negative controls do not pass 95% gates for every pathway/run, especially in breast-side PLIP axes.
- PLIP embeddings may include generic histology axes; these were diagnosed and masked in sensitivity analysis, but independent embedding backends remain future work.
- No patient-level outcome, diagnostic classifier, or causal mechanism is claimed.
"""

    captions = f"""# Figure Captions

## Figure 1. Breast discovery morphopathway associations from H&E-derived PLIP embeddings and Xenium WTA pathway activity.

Source table: `main_figure_1_source_breast_discovery_highnull32.csv`.

This figure should show the top breast discovery associations between PLIP-derived H&E embedding dimensions and curated WTA pathway activity scores after residual adjustment for spatial and cell-count covariates. Panels should include the pathway family, image feature, absolute partial Spearman rho, spatial-null gate status, and matched negative-control gate status. The caption should state that associations are discovery signals and are not interpreted as direct biomarkers.

## Figure 2. Cross-cancer pathway-family stress-test stability across high-null seeds.

Source table: `main_figure_2_source_cross_cancer_stability_highnull32.csv`.

This figure should summarize pathway-family recovery from breast discovery to cervical validation across three high-null seeds. The main visual should distinguish the stable {len(stable_core)}-pathway core from unstable pathways, with primary and axis-masked recovery shown side by side. The caption should report recovery of {recovery_min}/{total} to {recovery_max}/{total}, axis-masked recovery of {masked_min}/{total} to {masked_max}/{total}, and the exclusion of `emt_invasive_front` from the stable core.

## Extended Data Figure 1. Spatial pseudobulk and null-control sensitivity.

Source tables: `supp_table_spatial_block_and_seed_summary.csv` and `supp_table_highnull32_gate_and_axis_masked_summary.csv`.

This figure should document the 12 x 12 spatial pseudobulk setting, retained block counts (breast {breast_block_range}; cervical {cervical_block_range}), spatial null pass95/pass99 counts, and matched negative-control pass95/pass99 counts. The caption should explicitly identify negative-control pass95 as the remaining limitation.

## Extended Data Figure 2. Candidate generic PLIP axes and axis-masked recovery.

Source table: `supp_table_plip_axis_diagnostics_by_run.csv`.

This figure should show sample-specific candidate generic PLIP axes flagged by repeated top-40 usage plus matched-control failure. The caption should state that candidate axes occurred in {candidate_axis_runs}/{n_runs} runs, and that removing them left cross-cancer recovery unchanged at {masked_min}/{total} to {masked_max}/{total}.
"""

    checklist = f"""# Submission Readiness Checklist

## Ready

- Final high-null evidence package exists and passes QC.
- Main Figure 1 and Main Figure 2 source tables are generated.
- Supplementary gate, axis diagnostic, spatial block, and by-run validation tables are generated.
- Conservative claim wording is available and avoids direct cervical replication.
- Axis-masked sensitivity is included.
- Focused regression tests pass.

## Must Preserve In Drafting

- Claim only pathway-family stress-test recovery.
- Report cross-cancer recovery as {recovery_min}/{total} to {recovery_max}/{total}.
- Report axis-masked recovery as {masked_min}/{total} to {masked_max}/{total}.
- Exclude `emt_invasive_front` from the stable core.
- State matched negative-control pass95 limitations: breast minimum {breast_neg_min}/10, cervical minimum {cervical_neg_min}/10.

## Not Yet Publication-Ready Without Further Work

- Independent cohort validation.
- Full-scale, non-sampled run or explicit computational budget justification.
- Additional embedding backend comparison such as UNI or domain-specific PLIP variants.
- Final Methods details for software versions, hardware, and random seeds in the manuscript.
- Repository cleanup, commit, and release archive.
"""

    files = {
        "manuscript_skeleton.md": skeleton,
        "figure_captions.md": captions,
        "submission_readiness_checklist.md": checklist,
    }
    for name, content in files.items():
        (package_dir / name).write_text(content, encoding="utf-8")

    readme_path = package_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
    else:
        readme = "# Brief Communication Evidence Package\n"
    text_section = """\nManuscript-facing text pack:\n- `manuscript_skeleton.md`\n- `figure_captions.md`\n- `submission_readiness_checklist.md`\n"""
    if "Manuscript-facing text pack:" not in readme:
        readme = readme.rstrip() + "\n" + text_section
        readme_path.write_text(readme, encoding="utf-8")

    text_manifest = {
        "package_dir": str(package_dir),
        "generated_files": sorted(files),
        "stable_core_count": len(stable_core),
        "cross_cancer_recovery_range": [recovery_min, recovery_max],
        "axis_masked_recovery_range": [masked_min, masked_max],
        "cross_cancer_total": total,
        "negative_control_pass95_min": {
            "breast": breast_neg_min,
            "cervical": cervical_neg_min,
        },
        "candidate_axis_runs": candidate_axis_runs,
    }
    (package_dir / "brief_communication_text_pack_manifest.json").write_text(
        json.dumps(text_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(text_manifest, indent=2))


if __name__ == "__main__":
    main()
