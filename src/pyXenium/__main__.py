import json
from pathlib import Path

import click
import pandas as pd

from .benchmarking import (
    aggregate_standardized_results,
    DEFAULT_A100_ALL_METHODS,
    DEFAULT_A100_READONLY_XENIUM_ROOT,
    DEFAULT_A100_REMOTE_ROOT,
    build_a100_job_manifest,
    build_a100_job_matrix,
    build_a100_resource_summary,
    build_a100_sidecar_matrix,
    build_a100_stage_plan,
    build_engineering_summary,
    build_stage_manifest,
    build_method_run_plan,
    collect_a100_results,
    compute_canonical_recovery,
    compute_novelty_support,
    compute_pathway_relevance,
    compute_spatial_coherence,
    compute_robustness,
    execute_a100_stage_plan,
    finalize_a100_all,
    monitor_a100_jobs,
    prepare_a100_bundle,
    prepare_atera_lr_benchmark,
    render_atera_lr_benchmark_report,
    resolve_layout,
    run_a100_plan,
    run_registered_method,
    run_pyxenium_smoke,
    run_smoke_core,
    score_biological_performance,
    summarize_run_status,
    submit_a100_matrix,
)
from .io.io import copy_bundled_dataset, load_toy
from .io.spatialdata_export import DEFAULT_SPATIALDATA_STORE_NAME, export_xenium_to_spatialdata_zarr
from .gmi import (
    DEFAULT_GMI_A100_READONLY_XENIUM_ROOT,
    DEFAULT_GMI_A100_REMOTE_ROOT,
    DEFAULT_GMI_PDC_HISTOSEG_ROOT,
    DEFAULT_GMI_PDC_ROOT,
    DEFAULT_GMI_PDC_XENIUM_ROOT,
    ContourGmiConfig,
    build_contour_gmi_dataset,
    build_gmi_a100_plan,
    build_gmi_pdc_plan,
    load_atera_breast_for_gmi,
    run_atera_breast_contour_gmi,
    summarize_gmi_pdc_runs,
    write_gmi_a100_plan,
    write_gmi_pdc_plan,
    write_contour_gmi_dataset,
)
from .mechanostress import (
    DEFAULT_HNSCC_ROOT,
    DEFAULT_MECHANOSTRESS_RADII_UM,
    DEFAULT_SUZUKI_ER_RESULTS_ROOT,
    DEFAULT_SUZUKI_STRENGTH_ROOT,
    DEFAULT_SUZUKI_XENIUM_ROOT,
    AxisStrengthConfig,
    MechanostressConfig,
    PolarityConfig,
    TumorStromaGrowthConfig,
    classify_tumor_stroma_growth,
    compute_ane_density,
    compute_cell_polarity,
    estimate_cell_axes,
    run_mechanostress_cohort,
    run_mechanostress_workflow,
    summarize_axial_orientation,
    summarize_cell_polarity,
    summarize_tumor_growth,
    validate_hnscc_mechanostress_outputs,
    validate_suzuki_luad_mechanostress_outputs,
    write_mechanostress_artifacts,
)
from .multimodal import (
    DEFAULT_DATASET_PATH,
    run_renal_immune_resistance_pilot,
    run_validated_renal_ffpe_smoke,
)
from .validation import DEFAULT_ATERA_WTA_BREAST_DATASET_PATH, run_atera_wta_breast_topology
from .validation.atera_wta_breast_topology import DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR


@click.group()
def app():
    """pyXenium: Xenium toolkit (toy data included)"""


@app.group("multimodal")
def multimodal_group():
    """Joint RNA + protein analysis and workflow commands."""


@app.group("benchmark")
def benchmark_group():
    """Benchmark orchestration commands."""


@benchmark_group.group("atera-lr")
def benchmark_atera_lr_group():
    """Atera Xenium ligand-receptor benchmark commands."""


@app.group("gmi")
def gmi_group():
    """Contour-native GMI spatial transcriptomics commands."""


@app.group("mechanostress")
def mechanostress_group():
    """Mechanical stress state analysis from Xenium morphology and spatial context."""


def _gmi_config_from_options(**kwargs) -> ContourGmiConfig:
    return ContourGmiConfig(
        contour_key=kwargs.get("contour_key", "s1_s5_contours"),
        contour_label_col=kwargs.get("contour_label_col", "assigned_structure"),
        positive_label=kwargs.get("positive_label", "S1"),
        negative_label=kwargs.get("negative_label", "S5"),
        contour_geojson=kwargs.get("contour_geojson"),
        feature_count=kwargs.get("feature_count", 500),
        spatial_feature_count=kwargs.get("spatial_feature_count", 100),
        min_cells_per_contour=kwargs.get("min_cells_per_contour", 20),
        min_library_size=kwargs.get("min_library_size", 1.0),
        min_feature_prevalence=kwargs.get("min_feature_prevalence", 0.05),
        random_seed=kwargs.get("seed", 1),
        exclude_coordinate_spatial_features=kwargs.get("exclude_coordinate_spatial_features", False),
        rscript=kwargs.get("rscript", "Rscript"),
        r_lib_path=kwargs.get("r_lib_path"),
        install_gmi=kwargs.get("install_gmi", True),
        force_reinstall_gmi=kwargs.get("force_reinstall_gmi", False),
        spatial_cv_folds=kwargs.get("spatial_cv_folds", 0),
        bootstrap_repeats=kwargs.get("bootstrap_repeats", 0),
        run_label_permutation_control=kwargs.get("label_permutation_control", False),
        run_coordinate_shuffle_control=kwargs.get("coordinate_shuffle_control", False),
        run_spatial_feature_shuffle_control=kwargs.get("spatial_feature_shuffle_control", False),
        run_within_label_heterogeneity=kwargs.get("within_label_heterogeneity", True),
        write_spatial_visualizations=kwargs.get("write_spatial_visualizations", True),
        visualization_genes=tuple(kwargs.get("visualization_genes", ("NIBAN1", "SORL1", "CCND1"))),
    )


def _mechanostress_config_from_options(**kwargs) -> MechanostressConfig:
    radii = tuple(float(value) for value in kwargs.get("radii_um", DEFAULT_MECHANOSTRESS_RADII_UM))
    axis = AxisStrengthConfig(
        er_threshold=float(kwargs.get("er_threshold", 2.0)),
        radii_um=radii,
        groupby=tuple(kwargs.get("axis_groupby", ("cluster",))),
        local_k=int(kwargs.get("local_k", 15)),
        cell_query=kwargs.get("axis_cell_query"),
    )
    tumor_stroma = TumorStromaGrowthConfig(
        annotation_col=kwargs.get("annotation_col", "Annotation"),
        tumor_label=kwargs.get("tumor_label", "Tumor"),
        stroma_label=kwargs.get("stroma_label", "Stromal"),
        x_col=kwargs.get("x_col", "x_centroid"),
        y_col=kwargs.get("y_col", "y_centroid"),
        cell_id_col=kwargs.get("cell_id_col", "cell_id"),
        method=kwargs.get("growth_method", "delaunay_hop"),
    )
    polarity = PolarityConfig(
        offset_norm_threshold=float(kwargs.get("offset_norm_threshold", 0.30)),
        cell_id_col=kwargs.get("cell_id_col", "cell_id"),
    )
    return MechanostressConfig(
        axis=axis,
        tumor_stroma=tumor_stroma,
        polarity=polarity,
        coupling_genes=tuple(kwargs.get("coupling_genes", ())),
        sample_id=kwargs.get("sample_id"),
    )


@mechanostress_group.command("axis-strength")
@click.argument("base_path")
@click.option("--output-dir", required=True)
@click.option("--radius", "radii_um", multiple=True, type=float, default=DEFAULT_MECHANOSTRESS_RADII_UM, show_default=True)
@click.option("--er-threshold", type=float, default=2.0, show_default=True)
@click.option("--groupby", "axis_groupby", multiple=True, default=("cluster",), show_default=True)
@click.option("--local-k", type=int, default=15, show_default=True)
@click.option("--cell-query", "axis_cell_query", default=None)
@click.option("--sample-id", default=None)
def mechanostress_axis_strength(base_path, output_dir, radii_um, er_threshold, axis_groupby, local_k, axis_cell_query, sample_id):
    """Compute nucleus-axis orientation and ANE density from Xenium boundaries."""
    config = _mechanostress_config_from_options(
        radii_um=radii_um,
        er_threshold=er_threshold,
        axis_groupby=axis_groupby,
        local_k=local_k,
        axis_cell_query=axis_cell_query,
        sample_id=sample_id,
    )
    result = run_mechanostress_workflow(base_path=base_path, config=config, output_dir=output_dir)
    click.echo(json.dumps(result.summary, indent=2, default=str))


@mechanostress_group.command("tumor-stroma-growth")
@click.argument("cell_table")
@click.option("--output-dir", required=True)
@click.option("--annotation-col", default="Annotation", show_default=True)
@click.option("--tumor-label", default="Tumor", show_default=True)
@click.option("--stroma-label", default="Stromal", show_default=True)
@click.option("--x-col", default="x_centroid", show_default=True)
@click.option("--y-col", default="y_centroid", show_default=True)
@click.option("--method", "growth_method", type=click.Choice(["delaunay_hop", "nearest_distance_ratio"]), default="delaunay_hop", show_default=True)
@click.option("--sample-id", default=None)
def mechanostress_tumor_stroma_growth(
    cell_table,
    output_dir,
    annotation_col,
    tumor_label,
    stroma_label,
    x_col,
    y_col,
    growth_method,
    sample_id,
):
    """Classify tumor growth as infiltrative or expanding from tumor-stroma geometry."""
    frame = pd.read_csv(cell_table)
    growth = classify_tumor_stroma_growth(
        frame,
        annotation_col=annotation_col,
        tumor_label=tumor_label,
        stroma_label=stroma_label,
        x_col=x_col,
        y_col=y_col,
        method=growth_method,
    )
    summary = summarize_tumor_growth(growth, annotation_col=annotation_col, tumor_label=tumor_label, sample_id=sample_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells_path = out / "tumor_growth_cells.csv"
    summary_path = out / "tumor_growth_summary.csv"
    growth.to_csv(cells_path, index=False)
    summary.to_csv(summary_path, index=False)
    click.echo(json.dumps({"tumor_growth_cells": str(cells_path), "tumor_growth_summary": str(summary_path), "summary": summary.iloc[0].to_dict()}, indent=2, default=str))


@mechanostress_group.command("polarity")
@click.argument("base_path")
@click.option("--output-dir", required=True)
@click.option("--offset-norm-threshold", type=float, default=0.30, show_default=True)
@click.option("--cell-id-col", default="cell_id", show_default=True)
@click.option("--sample-id", default=None)
def mechanostress_polarity(base_path, output_dir, offset_norm_threshold, cell_id_col, sample_id):
    """Compute cell polarity from cell and nucleus centroid offsets."""
    from .io import read_xenium

    sdata = read_xenium(base_path, as_="sdata", include_transcripts=False, include_boundaries=True)
    if "cell_boundaries" not in sdata.shapes or "nucleus_boundaries" not in sdata.shapes:
        raise click.ClickException("Both cell_boundaries and nucleus_boundaries are required for polarity analysis.")
    polarity = compute_cell_polarity(
        cell_boundaries=sdata.shapes["cell_boundaries"],
        nucleus_boundaries=sdata.shapes["nucleus_boundaries"],
        cell_id_col=cell_id_col,
        offset_norm_threshold=offset_norm_threshold,
    )
    summary = summarize_cell_polarity(polarity, sample_id=sample_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    polarity_path = out / "cell_polarity.csv"
    summary_path = out / "polarity_summary.csv"
    polarity.to_csv(polarity_path, index=False)
    summary.to_csv(summary_path, index=False)
    click.echo(json.dumps({"cell_polarity": str(polarity_path), "polarity_summary": str(summary_path), "summary": summary.iloc[0].to_dict() if not summary.empty else {}}, indent=2, default=str))


@mechanostress_group.command("run-cohort")
@click.argument("cohort_root")
@click.option("--output-dir", required=True)
@click.option("--sample-glob", default="*", show_default=True)
@click.option("--annotation-glob", default="*_cell_clusters_with_annotation_and_coord.csv", show_default=True)
@click.option("--prefer", type=click.Choice(["auto", "zarr", "h5", "mex"]), default="auto", show_default=True)
@click.option("--radius", "radii_um", multiple=True, type=float, default=DEFAULT_MECHANOSTRESS_RADII_UM, show_default=True)
@click.option("--er-threshold", type=float, default=2.0, show_default=True)
@click.option("--annotation-col", default="Annotation", show_default=True)
@click.option("--growth-method", type=click.Choice(["delaunay_hop", "nearest_distance_ratio"]), default="delaunay_hop", show_default=True)
@click.option("--sample-limit", type=int, default=None)
@click.option("--fail-fast/--continue-on-error", default=False, show_default=True)
def mechanostress_run_cohort(
    cohort_root,
    output_dir,
    sample_glob,
    annotation_glob,
    prefer,
    radii_um,
    er_threshold,
    annotation_col,
    growth_method,
    sample_limit,
    fail_fast,
):
    """Run mechanostress workflow for each immediate sample directory in a cohort."""
    config = _mechanostress_config_from_options(
        radii_um=radii_um,
        er_threshold=er_threshold,
        annotation_col=annotation_col,
        growth_method=growth_method,
    )
    result = run_mechanostress_cohort(
        cohort_root,
        output_dir=output_dir,
        config=config,
        sample_glob=sample_glob,
        annotation_glob=annotation_glob,
        prefer=prefer,
        sample_limit=sample_limit,
        fail_fast=fail_fast,
    )
    click.echo(
        json.dumps(
            {
                "n_completed": int(len(result.results)),
                "n_failed": int(len(result.errors)),
                "files": result.files,
            },
            indent=2,
            default=str,
        )
    )


@mechanostress_group.command("validate-hnscc")
@click.option("--root", default=DEFAULT_HNSCC_ROOT, show_default=True)
@click.option("--recomputed-dir", default=None)
@click.option("--tolerance", type=float, default=1e-6, show_default=True)
def mechanostress_validate_hnscc(root, recomputed_dir, tolerance):
    """Validate HNSCC mechanostress reference outputs."""
    payload = validate_hnscc_mechanostress_outputs(root=root, recomputed_dir=recomputed_dir, tolerance=tolerance)
    click.echo(json.dumps(payload, indent=2, default=str))


@mechanostress_group.command("validate-suzuki-luad")
@click.option("--xenium-root", default=DEFAULT_SUZUKI_XENIUM_ROOT, show_default=True)
@click.option("--er-results-root", default=DEFAULT_SUZUKI_ER_RESULTS_ROOT, show_default=True)
@click.option("--strength-root", default=DEFAULT_SUZUKI_STRENGTH_ROOT, show_default=True)
@click.option("--recomputed-dir", default=None)
@click.option("--tolerance", type=float, default=1e-6, show_default=True)
def mechanostress_validate_suzuki_luad(xenium_root, er_results_root, strength_root, recomputed_dir, tolerance):
    """Validate Suzuki LUAD/TSU ER and ANE reference outputs."""
    payload = validate_suzuki_luad_mechanostress_outputs(
        xenium_root=xenium_root,
        er_results_root=er_results_root,
        strength_root=strength_root,
        recomputed_dir=recomputed_dir,
        tolerance=tolerance,
    )
    click.echo(json.dumps(payload, indent=2, default=str))


@gmi_group.command("prepare-atera-breast")
@click.option("--dataset-root", default=DEFAULT_ATERA_WTA_BREAST_DATASET_PATH, show_default=True)
@click.option("--output-dir", required=True)
@click.option("--contour-geojson", default=None, help="Optional S1-S5 contour GeoJSON path.")
@click.option("--contour-key", default="s1_s5_contours", show_default=True)
@click.option("--rna-feature-count", "--feature-count", "feature_count", type=int, default=500, show_default=True)
@click.option("--spatial-feature-count", type=int, default=100, show_default=True)
@click.option("--min-cells-per-contour", type=int, default=20, show_default=True)
@click.option("--exclude-coordinate-spatial-features", is_flag=True, default=False)
@click.option("--visualization-gene", "visualization_genes", multiple=True, default=("NIBAN1", "SORL1", "CCND1"))
@click.option("--spatial-visualizations/--skip-spatial-visualizations", "write_spatial_visualizations", default=True, show_default=True)
@click.option("--seed", type=int, default=1, show_default=True)
def gmi_prepare_atera_breast(
    dataset_root,
    output_dir,
    contour_geojson,
    contour_key,
    feature_count,
    spatial_feature_count,
    min_cells_per_contour,
    exclude_coordinate_spatial_features,
    visualization_genes,
    write_spatial_visualizations,
    seed,
):
    """Prepare the Atera breast S1-vs-S5 contour GMI matrix."""
    config = _gmi_config_from_options(
        contour_geojson=contour_geojson,
        contour_key=contour_key,
        feature_count=feature_count,
        spatial_feature_count=spatial_feature_count,
        min_cells_per_contour=min_cells_per_contour,
        exclude_coordinate_spatial_features=exclude_coordinate_spatial_features,
        visualization_genes=visualization_genes,
        write_spatial_visualizations=write_spatial_visualizations,
        seed=seed,
    )
    sdata = load_atera_breast_for_gmi(dataset_root)
    from .gmi._workflow import _ensure_atera_contours

    geojson = _ensure_atera_contours(sdata, dataset_root=dataset_root, config=config)
    dataset = build_contour_gmi_dataset(
        sdata,
        config=config,
        provenance={"dataset_root": str(dataset_root), "sample_id": "atera_wta_ffpe_breast", "contour_geojson": str(geojson)},
    )
    files = write_contour_gmi_dataset(dataset, output_dir)
    payload = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "contour_geojson": str(geojson),
        "n_contours": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "n_positive_contours": int((dataset.y == 1).sum()),
        "n_negative_contours": int((dataset.y == 0).sum()),
        "files": files,
    }
    click.echo(json.dumps(payload, indent=2))


@gmi_group.command("run")
@click.option("--dataset-root", default=DEFAULT_ATERA_WTA_BREAST_DATASET_PATH, show_default=True)
@click.option("--output-dir", required=True)
@click.option("--contour-geojson", default=None, help="Optional S1-S5 contour GeoJSON path.")
@click.option("--contour-key", default="s1_s5_contours", show_default=True)
@click.option("--rna-feature-count", "--feature-count", "feature_count", type=int, default=500, show_default=True)
@click.option("--spatial-feature-count", type=int, default=100, show_default=True)
@click.option("--min-cells-per-contour", type=int, default=20, show_default=True)
@click.option("--exclude-coordinate-spatial-features", is_flag=True, default=False)
@click.option("--visualization-gene", "visualization_genes", multiple=True, default=("NIBAN1", "SORL1", "CCND1"))
@click.option("--spatial-visualizations/--skip-spatial-visualizations", "write_spatial_visualizations", default=True, show_default=True)
@click.option("--seed", type=int, default=1, show_default=True)
@click.option("--rscript", default="Rscript", show_default=True)
@click.option("--r-lib-path", default=None)
@click.option("--install-gmi/--skip-install-gmi", default=True, show_default=True)
@click.option("--force-reinstall-gmi", is_flag=True, default=False)
@click.option("--spatial-cv-folds", type=int, default=0, show_default=True)
@click.option("--bootstrap-repeats", type=int, default=0, show_default=True)
@click.option("--label-permutation-control", is_flag=True, default=False)
@click.option("--coordinate-shuffle-control", is_flag=True, default=False)
@click.option("--spatial-feature-shuffle-control", is_flag=True, default=False)
@click.option("--within-label-heterogeneity/--skip-within-label-heterogeneity", default=True, show_default=True)
def gmi_run(
    dataset_root,
    output_dir,
    contour_geojson,
    contour_key,
    feature_count,
    spatial_feature_count,
    min_cells_per_contour,
    exclude_coordinate_spatial_features,
    visualization_genes,
    write_spatial_visualizations,
    seed,
    rscript,
    r_lib_path,
    install_gmi,
    force_reinstall_gmi,
    spatial_cv_folds,
    bootstrap_repeats,
    label_permutation_control,
    coordinate_shuffle_control,
    spatial_feature_shuffle_control,
    within_label_heterogeneity,
):
    """Run contour GMI on the Atera breast S1-vs-S5 task."""
    config = _gmi_config_from_options(
        contour_geojson=contour_geojson,
        contour_key=contour_key,
        feature_count=feature_count,
        spatial_feature_count=spatial_feature_count,
        min_cells_per_contour=min_cells_per_contour,
        exclude_coordinate_spatial_features=exclude_coordinate_spatial_features,
        visualization_genes=visualization_genes,
        write_spatial_visualizations=write_spatial_visualizations,
        seed=seed,
        rscript=rscript,
        r_lib_path=r_lib_path,
        install_gmi=install_gmi,
        force_reinstall_gmi=force_reinstall_gmi,
        spatial_cv_folds=spatial_cv_folds,
        bootstrap_repeats=bootstrap_repeats,
        label_permutation_control=label_permutation_control,
        coordinate_shuffle_control=coordinate_shuffle_control,
        spatial_feature_shuffle_control=spatial_feature_shuffle_control,
        within_label_heterogeneity=within_label_heterogeneity,
    )
    result = run_atera_breast_contour_gmi(dataset_root=dataset_root, output_dir=output_dir, config=config)
    click.echo(json.dumps(result.summary, indent=2, default=str))


@gmi_group.command("report")
@click.option("--summary-json", required=True)
def gmi_report(summary_json):
    """Render a saved GMI summary as markdown."""
    payload = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    report_path = payload.get("files", {}).get("report_md")
    if report_path and Path(report_path).exists():
        click.echo(Path(report_path).read_text(encoding="utf-8"))
    else:
        from .gmi import render_contour_gmi_report

        click.echo(render_contour_gmi_report(payload))


@gmi_group.command("a100-plan")
@click.option("--remote-xenium-root", default=DEFAULT_GMI_A100_READONLY_XENIUM_ROOT, show_default=True)
@click.option("--remote-root", default=DEFAULT_GMI_A100_REMOTE_ROOT, show_default=True)
@click.option("--repo-dir", default=None)
@click.option("--env-name", default="pyx-gmi", show_default=True)
@click.option("--output-json", default=None)
def gmi_a100_plan(remote_xenium_root, remote_root, repo_dir, env_name, output_json):
    """Build the A100 GMI smoke/full/stability command plan."""
    payload = build_gmi_a100_plan(
        remote_xenium_root=remote_xenium_root,
        remote_root=remote_root,
        repo_dir=repo_dir,
        env_name=env_name,
    )
    if not payload["path_policy"]["valid"]:
        raise click.ClickException("; ".join(payload["path_policy"]["issues"]))
    write_gmi_a100_plan(payload, output_json)
    click.echo(json.dumps(payload, indent=2))


@gmi_group.command("pdc-plan")
@click.option("--pdc-xenium-root", default=DEFAULT_GMI_PDC_XENIUM_ROOT, show_default=True)
@click.option("--pdc-root", default=DEFAULT_GMI_PDC_ROOT, show_default=True)
@click.option("--repo-dir", default=None)
@click.option("--conda-prefix", default=None)
@click.option("--conda-pkgs-dir", default=None)
@click.option("--histoseg-root", default=DEFAULT_GMI_PDC_HISTOSEG_ROOT, show_default=True)
@click.option("--account", default=None, help="Optional Slurm account. If omitted, PDC_PROJECT/projinfo is used by submit scripts.")
@click.option("--output-json", default=None)
def gmi_pdc_plan(pdc_xenium_root, pdc_root, repo_dir, conda_prefix, conda_pkgs_dir, histoseg_root, account, output_json):
    """Build the PDC Dardel Slurm plan for contour GMI validation."""
    payload = build_gmi_pdc_plan(
        pdc_xenium_root=pdc_xenium_root,
        pdc_root=pdc_root,
        repo_dir=repo_dir,
        conda_prefix=conda_prefix,
        conda_pkgs_dir=conda_pkgs_dir,
        histoseg_root=histoseg_root,
        account=account,
    )
    if not payload["path_policy"]["valid"]:
        raise click.ClickException("; ".join(payload["path_policy"]["issues"]))
    write_gmi_pdc_plan(payload, output_json)
    click.echo(json.dumps(payload, indent=2))


@gmi_group.command("pdc-monitor")
@click.option("--pdc-root", default=DEFAULT_GMI_PDC_ROOT, show_default=True)
@click.option("--output-json", default=None)
def gmi_pdc_monitor(pdc_root, output_json):
    """Summarize PDC contour GMI stage outputs under a PDC root."""
    payload = summarize_gmi_pdc_runs(pdc_root)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(payload, indent=2, default=str))


def _run_validate_renal_ffpe_protein(
    *,
    base_path,
    prefer,
    top_n,
    allow_mismatch,
    output_json,
    output_dir,
    write_h5ad,
):
    payload = run_validated_renal_ffpe_smoke(
        base_path=base_path,
        prefer=prefer,
        top_n=top_n,
        output_json=output_json,
        output_dir=output_dir,
        write_h5ad=write_h5ad,
    )
    click.echo(json.dumps(payload, indent=2))
    if payload["issues"] and not allow_mismatch:
        raise click.exceptions.Exit(1)


def _run_renal_immune_resistance_pilot(
    *,
    base_path,
    prefer,
    sample_id,
    n_neighbors,
    region_bins,
    top_n,
    output_json,
    output_dir,
    write_h5ad,
    manuscript_mode,
    manuscript_root,
    export_figures,
):
    study = run_renal_immune_resistance_pilot(
        base_path=base_path,
        prefer=prefer,
        sample_id=sample_id,
        n_neighbors=n_neighbors,
        region_bins=region_bins,
        output_json=output_json,
        output_dir=output_dir,
        write_h5ad=write_h5ad,
        top_n=top_n,
        manuscript_mode=manuscript_mode,
        manuscript_root=manuscript_root,
        export_figures=export_figures,
    )
    click.echo(json.dumps(study["payload"], indent=2))


@app.command()
def demo():
    ds = load_toy()
    click.echo(f"Loaded groups: {list(ds)}")


@app.command()
@click.option("--name", default="toy_slide", show_default=True)
@click.option("--url", default=None, help="Optional URL to download a dataset archive")
@click.option("--dest", default=str(Path.home() / ".cache" / "pyXenium"), show_default=True)
def datasets(name, url, dest):
    """Fetch example datasets to a local cache."""
    cache = Path(dest)
    cache.mkdir(parents=True, exist_ok=True)
    if url:
        import urllib.request

        target = cache / name
        urllib.request.urlretrieve(url, str(target))
        click.echo(f"Downloaded to {target}")
    else:
        try:
            target = copy_bundled_dataset(name=name, dest=cache)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Copied bundled toy dataset to {target}")


@app.command("export-spatialdata")
@click.argument("base_path", required=False, default=DEFAULT_DATASET_PATH)
@click.option(
    "--output-path",
    default=None,
    help=f"Optional output Zarr path. Defaults to <base_path>/{DEFAULT_SPATIALDATA_STORE_NAME}.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite an existing output store.")
@click.option("--n-jobs", type=int, default=1, show_default=True, help="Compatibility option retained for legacy scripts.")
@click.option("--no-transcripts", is_flag=True, default=False, help="Skip transcript points to reduce output size.")
@click.option("--no-morphology-focus", is_flag=True, default=False, help="Skip morphology_focus image.")
@click.option("--no-morphology-mip", is_flag=True, default=False, help="Skip morphology_mip image.")
@click.option("--no-aligned-images", is_flag=True, default=False, help="Skip aligned auxiliary images.")
def export_spatialdata(base_path, output_path, overwrite, n_jobs, no_transcripts, no_morphology_focus, no_morphology_mip, no_aligned_images):
    """Convert a Xenium export into pyXenium's SData store at the legacy path name."""
    payload = export_xenium_to_spatialdata_zarr(
        base_path=base_path,
        output_path=output_path,
        overwrite=overwrite,
        n_jobs=n_jobs,
        transcripts=not no_transcripts,
        morphology_focus=not no_morphology_focus,
        morphology_mip=not no_morphology_mip,
        aligned_images=not no_aligned_images,
    )
    click.echo(json.dumps(payload, indent=2))


@app.command("validate-renal-ffpe-protein")
@click.argument("base_path", required=False, default=DEFAULT_DATASET_PATH)
@click.option("--prefer", type=click.Choice(["auto", "zarr", "h5", "mex"]), default="auto", show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--allow-mismatch", is_flag=True, default=False)
@click.option("--output-json", default=None, help="Optional path to write the summary JSON.")
@click.option(
    "--output-dir",
    default=None,
    help="Optional directory for report.md, summary.json, and CSV summaries.",
)
@click.option("--write-h5ad", default=None, help="Optional path to export the loaded AnnData object as an .h5ad file.")
def validate_renal_ffpe_protein(base_path, prefer, top_n, allow_mismatch, output_json, output_dir, write_h5ad):
    """Deprecated alias for `pyxenium multimodal validate-renal-ffpe-protein`."""
    _run_validate_renal_ffpe_protein(
        base_path=base_path,
        prefer=prefer,
        top_n=top_n,
        allow_mismatch=allow_mismatch,
        output_json=output_json,
        output_dir=output_dir,
        write_h5ad=write_h5ad,
    )


@multimodal_group.command("validate-renal-ffpe-protein")
@click.argument("base_path", required=False, default=DEFAULT_DATASET_PATH)
@click.option("--prefer", type=click.Choice(["auto", "zarr", "h5", "mex"]), default="auto", show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--allow-mismatch", is_flag=True, default=False)
@click.option("--output-json", default=None, help="Optional path to write the summary JSON.")
@click.option(
    "--output-dir",
    default=None,
    help="Optional directory for report.md, summary.json, and CSV summaries.",
)
@click.option("--write-h5ad", default=None, help="Optional path to export the loaded AnnData object as an .h5ad file.")
def multimodal_validate_renal_ffpe_protein(base_path, prefer, top_n, allow_mismatch, output_json, output_dir, write_h5ad):
    """Validate pyXenium against the public 10x FFPE renal RNA + Protein dataset."""
    _run_validate_renal_ffpe_protein(
        base_path=base_path,
        prefer=prefer,
        top_n=top_n,
        allow_mismatch=allow_mismatch,
        output_json=output_json,
        output_dir=output_dir,
        write_h5ad=write_h5ad,
    )


@app.command("renal-immune-resistance-pilot")
@click.argument("base_path", required=False, default=DEFAULT_DATASET_PATH)
@click.option("--prefer", type=click.Choice(["auto", "zarr", "h5", "mex"]), default="auto", show_default=True)
@click.option("--sample-id", default="renal_ffpe_public_10x", show_default=True)
@click.option("--n-neighbors", type=int, default=15, show_default=True)
@click.option("--region-bins", type=int, default=24, show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--output-json", default=None, help="Optional path to write the summary JSON.")
@click.option("--output-dir", default=None, help="Optional directory for markdown and CSV artifacts.")
@click.option("--write-h5ad", default=None, help="Optional path to export the annotated AnnData object.")
@click.option("--manuscript-mode", is_flag=True, default=False, help="Write a fixed naming discovery package under the manuscript root.")
@click.option("--manuscript-root", default="manuscript", show_default=True, help="Root directory used by manuscript mode.")
@click.option("--export-figures/--no-export-figures", default=True, show_default=True)
def renal_immune_resistance_pilot(
    base_path,
    prefer,
    sample_id,
    n_neighbors,
    region_bins,
    top_n,
    output_json,
    output_dir,
    write_h5ad,
    manuscript_mode,
    manuscript_root,
    export_figures,
):
    """Deprecated alias for `pyxenium multimodal renal-immune-resistance-pilot`."""
    _run_renal_immune_resistance_pilot(
        base_path=base_path,
        prefer=prefer,
        sample_id=sample_id,
        n_neighbors=n_neighbors,
        region_bins=region_bins,
        top_n=top_n,
        output_json=output_json,
        output_dir=output_dir,
        write_h5ad=write_h5ad,
        manuscript_mode=manuscript_mode,
        manuscript_root=manuscript_root,
        export_figures=export_figures,
    )


@multimodal_group.command("renal-immune-resistance-pilot")
@click.argument("base_path", required=False, default=DEFAULT_DATASET_PATH)
@click.option("--prefer", type=click.Choice(["auto", "zarr", "h5", "mex"]), default="auto", show_default=True)
@click.option("--sample-id", default="renal_ffpe_public_10x", show_default=True)
@click.option("--n-neighbors", type=int, default=15, show_default=True)
@click.option("--region-bins", type=int, default=24, show_default=True)
@click.option("--top-n", type=int, default=10, show_default=True)
@click.option("--output-json", default=None, help="Optional path to write the summary JSON.")
@click.option("--output-dir", default=None, help="Optional directory for markdown and CSV artifacts.")
@click.option("--write-h5ad", default=None, help="Optional path to export the annotated AnnData object.")
@click.option("--manuscript-mode", is_flag=True, default=False, help="Write a fixed naming discovery package under the manuscript root.")
@click.option("--manuscript-root", default="manuscript", show_default=True, help="Root directory used by manuscript mode.")
@click.option("--export-figures/--no-export-figures", default=True, show_default=True)
def multimodal_renal_immune_resistance_pilot(
    base_path,
    prefer,
    sample_id,
    n_neighbors,
    region_bins,
    top_n,
    output_json,
    output_dir,
    write_h5ad,
    manuscript_mode,
    manuscript_root,
    export_figures,
):
    """Run the renal spatial immune-resistance pilot workflow on a Xenium RNA + Protein dataset."""
    _run_renal_immune_resistance_pilot(
        base_path=base_path,
        prefer=prefer,
        sample_id=sample_id,
        n_neighbors=n_neighbors,
        region_bins=region_bins,
        top_n=top_n,
        output_json=output_json,
        output_dir=output_dir,
        write_h5ad=write_h5ad,
        manuscript_mode=manuscript_mode,
        manuscript_root=manuscript_root,
        export_figures=export_figures,
    )


@app.command("atera-wta-breast-topology")
@click.argument("dataset_root", required=False, default=DEFAULT_ATERA_WTA_BREAST_DATASET_PATH)
@click.option(
    "--tbc-results",
    default=None,
    help="Optional path to a directory containing t_and_c_result*.csv and StructureMap_table*.csv.",
)
@click.option("--output-dir", default=None, help="Optional directory for the fixed result bundle.")
@click.option("--manuscript-mode", is_flag=True, default=False, help="Write the fixed manuscript bundle under the manuscript root.")
@click.option("--manuscript-root", default="manuscript", show_default=True)
@click.option("--sample-id", default="atera_wta_ffpe_breast", show_default=True)
@click.option("--write-h5ad", default=None, help="Optional path to export the loaded AnnData object.")
@click.option("--export-figures/--no-export-figures", default=True, show_default=True)
def atera_wta_breast_topology(
    dataset_root,
    tbc_results,
    output_dir,
    manuscript_mode,
    manuscript_root,
    sample_id,
    write_h5ad,
    export_figures,
):
    """Run the fixed Atera WTA breast LR/pathway topology reproducibility workflow."""
    study = run_atera_wta_breast_topology(
        dataset_root=dataset_root,
        tbc_results=tbc_results,
        output_dir=output_dir,
        manuscript_mode=manuscript_mode,
        manuscript_root=manuscript_root,
        sample_id=sample_id,
        export_figures=export_figures,
        write_h5ad=write_h5ad,
    )
    click.echo(json.dumps(study["payload"], indent=2))


@benchmark_atera_lr_group.command("prepare")
@click.option("--dataset-root", "--xenium-root", "dataset_root", default=DEFAULT_ATERA_WTA_BREAST_DATASET_PATH, show_default=True)
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--tbc-results", default=None, help="Optional path to the topology bundle directory.")
@click.option("--smoke-n-cells", type=int, default=20000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--prefer", type=click.Choice(["auto", "zarr", "h5", "mex"]), default="h5", show_default=True)
@click.option("--export-full-bundle/--skip-full-bundle", default=True, show_default=True)
@click.option("--write-full-h5ad/--skip-full-h5ad", default=True, show_default=True)
def benchmark_atera_lr_prepare(dataset_root, benchmark_root, tbc_results, smoke_n_cells, seed, prefer, export_full_bundle, write_full_h5ad):
    """Freeze the Atera Xenium benchmark inputs and export the cross-language bundle."""
    payload = prepare_atera_lr_benchmark(
        dataset_root=dataset_root,
        benchmark_root=benchmark_root,
        tbc_results=tbc_results,
        smoke_n_cells=smoke_n_cells,
        seed=seed,
        prefer=prefer,
        export_full_bundle=export_full_bundle,
        write_full_h5ad=write_full_h5ad,
    )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("smoke-pyxenium")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--input-h5ad", default=None, help="Optional smoke h5ad path. Defaults to <benchmark_root>/data/smoke/adata_smoke.h5ad.")
@click.option("--output-dir", default=None, help="Optional output directory. Defaults to <benchmark_root>/runs/pyxenium_smoke.")
@click.option("--tbc-results", default=None, help="Optional path to the topology bundle directory.")
@click.option("--lr-panel-path", default=None, help="Optional LR panel TSV. Defaults to <benchmark_root>/data/atera_smoke_panel.tsv.")
@click.option("--database-mode", default="smoke-panel", show_default=True)
@click.option("--export-figures/--no-export-figures", default=False, show_default=True)
def benchmark_atera_lr_smoke_pyxenium(benchmark_root, input_h5ad, output_dir, tbc_results, lr_panel_path, database_mode, export_figures):
    """Run the pyXenium smoke benchmark and standardize the result table."""
    layout = resolve_layout(relative_root=benchmark_root or Path("benchmarking") / "lr_2026_atera")
    resolved_input_h5ad = input_h5ad or layout.data_dir / "smoke" / "adata_smoke.h5ad"
    resolved_output_dir = output_dir or layout.runs_dir / "pyxenium_smoke"
    resolved_lr_panel = lr_panel_path or layout.data_dir / "atera_smoke_panel.tsv"
    resolved_tbc = tbc_results or Path(DEFAULT_ATERA_WTA_BREAST_DATASET_PATH) / DEFAULT_ATERA_WTA_BREAST_TBC_SUBDIR
    payload = run_pyxenium_smoke(
        input_h5ad=resolved_input_h5ad,
        output_dir=resolved_output_dir,
        tbc_results=resolved_tbc,
        lr_panel_path=resolved_lr_panel,
        database_mode=database_mode,
        export_figures=export_figures,
    )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("run-method")
@click.option("--method", required=True, help="Registered LR benchmark method slug, e.g. squidpy, liana, commot, cellchat.")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--input-manifest", default=None, help="Optional input manifest. Defaults to <benchmark_root>/data/input_manifest.json.")
@click.option("--output-dir", default=None, help="Optional output directory. Defaults to <benchmark_root>/runs/<method>_<phase>_<database-mode>.")
@click.option("--database-mode", default="common-db", show_default=True, help="common-db, native-db, or smoke-panel.")
@click.option("--phase", type=click.Choice(["smoke", "full"]), default="smoke", show_default=True)
@click.option("--max-lr-pairs", type=int, default=None, help="Optional cap for pilot runs.")
@click.option("--n-perms", type=int, default=100, show_default=True, help="Permutation count for adapters that expose permutations.")
@click.option("--chunk-id", type=int, default=None, help="Optional zero-based LR chunk id for chunked methods.")
@click.option("--num-chunks", type=int, default=None, help="Optional total LR chunks for chunked methods.")
@click.option("--bounded-mode", default=None, help="Optional bounded execution label, e.g. smoke_20k, pilot_50k, commot_chunk.")
@click.option("--gpu-id", default=None, help="Optional GPU id assigned by the A100 job matrix.")
@click.option("--job-id", default=None, help="Optional A100 job id/provenance label.")
@click.option("--gzip-standardized", is_flag=True, default=False, help="Write standardized output as .tsv.gz when supported.")
@click.option("--tbc-results", default=None, help="Optional topology bundle directory used by the pyXenium adapter.")
@click.option("--export-figures/--no-export-figures", default=False, show_default=True)
@click.option("--rscript", default=None, help="Optional Rscript executable for CellChat.")
@click.option("--dry-run", is_flag=True, default=False, help="Validate the run contract without executing the adapter.")
def benchmark_atera_lr_run_method(
    method,
    benchmark_root,
    input_manifest,
    output_dir,
    database_mode,
    phase,
    max_lr_pairs,
    n_perms,
    chunk_id,
    num_chunks,
    bounded_mode,
    gpu_id,
    job_id,
    gzip_standardized,
    tbc_results,
    export_figures,
    rscript,
    dry_run,
):
    """Run one real LR benchmark adapter using the unified contract."""
    if dry_run:
        payload = build_method_run_plan(
            method=method,
            input_manifest=input_manifest,
            output_dir=output_dir,
            benchmark_root=benchmark_root,
            database_mode=database_mode,
            phase=phase,
            max_lr_pairs=max_lr_pairs,
            n_perms=n_perms,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
            bounded_mode=bounded_mode,
            gpu_id=gpu_id,
            job_id=job_id,
            gzip_standardized=gzip_standardized,
        )
    else:
        payload = run_registered_method(
            method=method,
            input_manifest=input_manifest,
            output_dir=output_dir,
            benchmark_root=benchmark_root,
            database_mode=database_mode,
            phase=phase,
            max_lr_pairs=max_lr_pairs,
            n_perms=n_perms,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
            bounded_mode=bounded_mode,
            gpu_id=gpu_id,
            job_id=job_id,
            gzip_standardized=gzip_standardized,
            tbc_results=tbc_results,
            export_figures=export_figures,
            rscript=rscript,
        )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("smoke-core")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--input-manifest", default=None, help="Optional input manifest. Defaults to <benchmark_root>/data/input_manifest.json.")
@click.option("--methods", default="pyxenium,squidpy,liana,commot,cellchat", show_default=True, help="Comma-separated method slugs.")
@click.option("--database-mode", default="common-db", show_default=True, help="common-db, native-db, or smoke-panel.")
@click.option("--max-lr-pairs", type=int, default=None, help="Optional cap for pilot runs.")
@click.option("--n-perms", type=int, default=100, show_default=True, help="Permutation count for adapters that expose permutations.")
@click.option("--dry-run", is_flag=True, default=False, help="Validate all method contracts without executing adapters.")
@click.option("--continue-on-error/--strict", default=True, show_default=True)
def benchmark_atera_lr_smoke_core(benchmark_root, input_manifest, methods, database_mode, max_lr_pairs, n_perms, dry_run, continue_on_error):
    """Run or dry-run the core LR adapter smoke benchmark."""
    method_list = [item.strip() for item in methods.split(",") if item.strip()]
    payload = run_smoke_core(
        methods=method_list,
        input_manifest=input_manifest,
        benchmark_root=benchmark_root,
        database_mode=database_mode,
        max_lr_pairs=max_lr_pairs,
        n_perms=n_perms,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("aggregate")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--result-path", "result_paths", multiple=True, help="One or more standardized TSV files.")
@click.option("--output-path", default=None, help="Optional output path. Defaults to <benchmark_root>/results/combined_standardized.tsv.")
def benchmark_atera_lr_aggregate(benchmark_root, result_paths, output_path):
    """Aggregate one or more standardized LR result tables."""
    layout = resolve_layout(relative_root=benchmark_root or Path("benchmarking") / "lr_2026_atera")
    discovered = sorted([*layout.runs_dir.glob("**/*standardized*.tsv"), *layout.runs_dir.glob("**/*standardized*.tsv.gz")])
    selected = list(result_paths) if result_paths else [str(path) for path in discovered]
    if not selected:
        raise click.ClickException("No standardized result files were provided or discovered under the benchmark runs directory.")
    resolved_output = output_path or layout.results_dir / "combined_standardized.tsv"
    combined = aggregate_standardized_results(selected, output_path=resolved_output)
    click.echo(json.dumps({"output_path": str(resolved_output), "n_rows": int(len(combined)), "inputs": selected}, indent=2))


@benchmark_atera_lr_group.command("report")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--combined-results", default=None, help="Optional aggregated standardized TSV path.")
@click.option("--canonical-config", default=None, help="Optional canonical axes YAML path.")
@click.option("--pathway-config", default=None, help="Optional pathway YAML path.")
@click.option("--output-path", default=None, help="Optional markdown report path.")
def benchmark_atera_lr_report(benchmark_root, combined_results, canonical_config, pathway_config, output_path):
    """Build a markdown report from standardized benchmark outputs."""
    layout = resolve_layout(relative_root=benchmark_root or Path("benchmarking") / "lr_2026_atera")
    combined_path = Path(combined_results) if combined_results else layout.results_dir / "combined_standardized.tsv"
    if not combined_path.exists():
        raise click.ClickException(f"Combined standardized table does not exist: {combined_path}")

    canonical_path = Path(canonical_config) if canonical_config else layout.config_dir / "canonical_axes.yaml"
    pathway_path = Path(pathway_config) if pathway_config else layout.config_dir / "pathways.yaml"
    combined = pd.read_csv(combined_path, sep="\t")
    canonical_detail, canonical_summary = compute_canonical_recovery(combined, canonical_axes=canonical_path)
    _, pathway_summary = compute_pathway_relevance(combined, pathway_config=pathway_path)
    spatial_summary = compute_spatial_coherence(combined)
    _, novelty_summary = compute_novelty_support(combined)
    robustness_summary = compute_robustness(combined)
    run_status = summarize_run_status(layout.runs_dir)
    engineering_summary = build_engineering_summary(run_status)
    a100_resource_summary = pd.DataFrame()
    logs_dir = getattr(layout, "logs_dir", layout.root / "logs")
    a100_plan_path = logs_dir / "a100_bundle_plan.json"
    if a100_plan_path.exists():
        a100_payload = json.loads(a100_plan_path.read_text(encoding="utf-8"))
        a100_resource_summary = build_a100_resource_summary(a100_payload.get("job_manifest", a100_payload))
    biology_summary = score_biological_performance(
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        spatial_summary=spatial_summary,
        robustness_summary=robustness_summary,
        novelty_summary=novelty_summary,
    )
    markdown = render_atera_lr_benchmark_report(
        combined_results=combined,
        canonical_summary=canonical_summary,
        pathway_summary=pathway_summary,
        biology_summary=biology_summary,
        benchmark_root=layout.root,
        run_status=run_status,
        engineering_summary=engineering_summary,
        canonical_detail=canonical_detail,
        a100_resource_summary=a100_resource_summary,
    )
    resolved_output = Path(output_path) if output_path else layout.reports_dir / "benchmark_report.md"
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(markdown, encoding="utf-8")
    click.echo(json.dumps({"report_md": str(resolved_output), "n_methods": int(combined["method"].nunique()) if not combined.empty else 0}, indent=2))


@benchmark_atera_lr_group.command("stage-a100")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--host", default=None, help="Remote SSH host or IP.")
@click.option("--user", default=None, help="Remote SSH username.")
@click.option("--remote-root", default=DEFAULT_A100_REMOTE_ROOT, show_default=True)
@click.option("--remote-xenium-root", default=DEFAULT_A100_READONLY_XENIUM_ROOT, show_default=True, help="Read-only Xenium outs path on A100.")
@click.option("--stage-data/--skip-data", default=None, help="Whether to copy local benchmark data. Defaults to skip when a remote Xenium root is provided.")
@click.option("--transfer-mode", type=click.Choice(["auto", "rsync", "scp", "tar-scp"]), default="auto", show_default=True)
@click.option("--include-path", "include_paths", multiple=True, help="Optional paths to stage. Defaults to configs/envs/scripts/runners/data.")
@click.option("--output-json", default=None, help="Optional JSON path to persist the staging manifest.")
@click.option("--plan-only", is_flag=True, default=False, help="Generate a host-agnostic A100 stage plan without requiring host/user.")
@click.option("--dry-run/--execute", default=True, show_default=True, help="When executing, run mkdir + transfer commands immediately.")
def benchmark_atera_lr_stage_a100(benchmark_root, host, user, remote_root, remote_xenium_root, stage_data, transfer_mode, include_paths, output_json, plan_only, dry_run):
    """Generate SSH/SCP commands for staging the benchmark to A100."""
    payload = build_a100_stage_plan(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        remote_xenium_root=remote_xenium_root,
        stage_data=stage_data,
        transfer_mode=transfer_mode,
        host=host,
        user=user,
    )
    effective_dry_run = bool(plan_only or dry_run)
    response = payload
    if not effective_dry_run:
        response = {
            "stage_plan": payload,
            "stage_execution": execute_a100_stage_plan(stage_plan=payload, dry_run=False),
        }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(response, indent=2))


@benchmark_atera_lr_group.command("prepare-a100-bundle")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--remote-root", default=DEFAULT_A100_REMOTE_ROOT, show_default=True)
@click.option("--remote-xenium-root", default=DEFAULT_A100_READONLY_XENIUM_ROOT, show_default=True, help="Read-only Xenium outs path on A100.")
@click.option("--transfer-mode", type=click.Choice(["auto", "rsync", "scp", "tar-scp"]), default="auto", show_default=True)
@click.option("--methods", default="pyxenium,squidpy,liana,commot,cellchat", show_default=True)
@click.option("--database-mode", default="common-db", show_default=True)
@click.option("--phase", type=click.Choice(["smoke", "full"]), default="full", show_default=True)
@click.option("--max-lr-pairs", type=int, default=None)
@click.option("--n-perms", type=int, default=100, show_default=True)
@click.option("--include-prepare/--skip-prepare", default=True, show_default=True, help="Include the A100 job that builds full sparse bundle from the read-only Xenium root.")
@click.option("--stage-data/--skip-data", default=None, help="Whether to copy local data in the stage plan. Defaults to skip with remote Xenium root.")
@click.option("--smoke-n-cells", type=int, default=20000, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--prefer", default="h5", show_default=True)
@click.option("--host", default=None)
@click.option("--user", default=None)
@click.option("--require-full/--allow-missing-full", default=True, show_default=True)
@click.option("--output-json", default=None)
def benchmark_atera_lr_prepare_a100_bundle(
    benchmark_root,
    remote_root,
    remote_xenium_root,
    transfer_mode,
    methods,
    database_mode,
    phase,
    max_lr_pairs,
    n_perms,
    include_prepare,
    stage_data,
    smoke_n_cells,
    seed,
    prefer,
    host,
    user,
    require_full,
    output_json,
):
    """Build the A100 stage plan and per-method job manifest."""
    payload = prepare_a100_bundle(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        remote_xenium_root=remote_xenium_root,
        transfer_mode=transfer_mode,
        methods=[item.strip() for item in methods.split(",") if item.strip()],
        database_mode=database_mode,
        phase=phase,
        max_lr_pairs=max_lr_pairs,
        n_perms=n_perms,
        require_full=require_full,
        include_prepare=include_prepare,
        stage_data=stage_data,
        smoke_n_cells=smoke_n_cells,
        seed=seed,
        prefer=prefer,
        host=host,
        user=user,
        output_json=output_json,
    )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("run-a100-plan")
@click.option("--plan-json", required=True, help="A100 job manifest or bundle plan JSON.")
@click.option("--job-id", "job_ids", multiple=True, help="Optional job id filter.")
@click.option("--dry-run/--execute", default=True, show_default=True)
@click.option("--remote/--local", default=False, show_default=True, help="Execute the plan over SSH instead of locally.")
@click.option("--host", default=None, help="Remote SSH host or IP.")
@click.option("--user", default=None, help="Remote SSH username.")
def benchmark_atera_lr_run_a100_plan(plan_json, job_ids, dry_run, remote, host, user):
    """Dry-run or execute commands from an A100 job manifest."""
    payload = run_a100_plan(
        plan_json=plan_json,
        dry_run=dry_run,
        job_ids=job_ids or None,
        remote=remote,
        host=host,
        user=user,
    )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("build-a100-matrix")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--remote-root", default=DEFAULT_A100_REMOTE_ROOT, show_default=True)
@click.option("--methods", default=",".join(DEFAULT_A100_ALL_METHODS), show_default=True)
@click.option("--database-mode", default="common-db", show_default=True)
@click.option("--phase", type=click.Choice(["smoke", "full"]), default="smoke", show_default=True)
@click.option("--max-lr-pairs", type=int, default=None)
@click.option("--n-perms", type=int, default=100, show_default=True)
@click.option("--commot-chunks", type=int, default=16, show_default=True)
@click.option("--gpu-count", type=int, default=8, show_default=True)
@click.option("--gzip-edge-outputs/--plain-edge-outputs", default=True, show_default=True)
@click.option("--include-bootstrap", is_flag=True, default=False)
@click.option("--include-audit", is_flag=True, default=False)
@click.option("--repeat-id", default=None)
@click.option("--output-json", default=None)
def benchmark_atera_lr_build_a100_matrix(
    benchmark_root,
    remote_root,
    methods,
    database_mode,
    phase,
    max_lr_pairs,
    n_perms,
    commot_chunks,
    gpu_count,
    gzip_edge_outputs,
    include_bootstrap,
    include_audit,
    repeat_id,
    output_json,
):
    """Build the parallel A100 job matrix for second-wave LR methods."""
    payload = build_a100_job_matrix(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        methods=[item.strip() for item in methods.split(",") if item.strip()],
        database_mode=database_mode,
        phase=phase,
        max_lr_pairs=max_lr_pairs,
        n_perms=n_perms,
        commot_chunks=commot_chunks,
        gpu_count=gpu_count,
        gzip_edge_outputs=gzip_edge_outputs,
        include_bootstrap=include_bootstrap,
        include_audit=include_audit,
        repeat_id=repeat_id,
    )
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("build-a100-sidecar")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--remote-root", default=DEFAULT_A100_REMOTE_ROOT, show_default=True)
@click.option("--methods", default="spatialdm,stlearn,giotto,laris,cellphonedb,spatalk,niches,cellnest,cellagentchat,scild", show_default=True)
@click.option("--ready-methods", default=None, help="Comma-separated methods whose env bootstrap is already complete. Defaults to --methods.")
@click.option("--database-mode", default="common-db", show_default=True)
@click.option("--smoke-max-lr-pairs", type=int, default=25, show_default=True)
@click.option("--pilot-max-lr-pairs", type=int, default=100, show_default=True)
@click.option("--pilot-n-cells", type=int, default=50000, show_default=True)
@click.option("--pilot-seed", type=int, default=1, show_default=True)
@click.option("--include-audit/--skip-audit", default=True, show_default=True)
@click.option("--include-smoke/--skip-smoke", default=True, show_default=True)
@click.option("--include-pilot/--skip-pilot", default=True, show_default=True)
@click.option("--completed-job-id", "completed_job_ids", multiple=True, help="Sidecar job ids already completed; omitted from the plan.")
@click.option("--running-job-id", "running_job_ids", multiple=True, help="Sidecar job ids already running; omitted from the plan.")
@click.option("--gpu-count", type=int, default=8, show_default=True)
@click.option("--gpu-start", type=int, default=3, show_default=True)
@click.option("--gzip-edge-outputs/--plain-edge-outputs", default=True, show_default=True)
@click.option("--output-json", default=None)
def benchmark_atera_lr_build_a100_sidecar(
    benchmark_root,
    remote_root,
    methods,
    ready_methods,
    database_mode,
    smoke_max_lr_pairs,
    pilot_max_lr_pairs,
    pilot_n_cells,
    pilot_seed,
    include_audit,
    include_smoke,
    include_pilot,
    completed_job_ids,
    running_job_ids,
    gpu_count,
    gpu_start,
    gzip_edge_outputs,
    output_json,
):
    """Build a balanced A100 sidecar matrix for already bootstrapped LR methods."""
    payload = build_a100_sidecar_matrix(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        methods=[item.strip() for item in methods.split(",") if item.strip()],
        ready_methods=None if not ready_methods else [item.strip() for item in ready_methods.split(",") if item.strip()],
        database_mode=database_mode,
        smoke_max_lr_pairs=smoke_max_lr_pairs,
        pilot_max_lr_pairs=pilot_max_lr_pairs,
        pilot_n_cells=pilot_n_cells,
        pilot_seed=pilot_seed,
        include_audit=include_audit,
        include_smoke=include_smoke,
        include_pilot=include_pilot,
        completed_job_ids=completed_job_ids or None,
        running_job_ids=running_job_ids or None,
        gpu_count=gpu_count,
        gpu_start=gpu_start,
        gzip_edge_outputs=gzip_edge_outputs,
    )
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("submit-a100-matrix")
@click.option("--matrix-json", required=True, help="A100 job matrix JSON produced by build-a100-matrix.")
@click.option("--job-id", "job_ids", multiple=True, help="Optional job id filter.")
@click.option("--job-type", "job_types", multiple=True, help="Optional job type filter, e.g. env_bootstrap, env_audit, method_run.")
@click.option("--dry-run/--execute", default=True, show_default=True)
@click.option("--remote/--local", default=True, show_default=True)
@click.option("--host", default=None)
@click.option("--user", default=None)
def benchmark_atera_lr_submit_a100_matrix(matrix_json, job_ids, job_types, dry_run, remote, host, user):
    """Submit A100 matrix jobs with remote nohup wrappers."""
    payload = submit_a100_matrix(
        matrix_json=matrix_json,
        dry_run=dry_run,
        remote=remote,
        host=host,
        user=user,
        job_ids=job_ids or None,
        job_types=job_types or None,
    )
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("monitor-a100-jobs")
@click.option("--matrix-json", required=True, help="A100 job matrix JSON.")
@click.option("--benchmark-root", default=None)
@click.option("--output-tsv", default=None)
def benchmark_atera_lr_monitor_a100_jobs(matrix_json, benchmark_root, output_tsv):
    """Summarize collected status for planned A100 jobs."""
    table = monitor_a100_jobs(matrix_json=matrix_json, benchmark_root=benchmark_root, output_tsv=output_tsv)
    click.echo(table.to_json(orient="records", indent=2))


@benchmark_atera_lr_group.command("finalize-a100-all")
@click.option("--benchmark-root", default=None)
@click.option("--output-path", default=None)
def benchmark_atera_lr_finalize_a100_all(benchmark_root, output_path):
    """Aggregate all collected A100 standardized TSV/TSV.GZ outputs."""
    payload = finalize_a100_all(benchmark_root=benchmark_root, output_path=output_path)
    click.echo(json.dumps(payload, indent=2))


@benchmark_atera_lr_group.command("collect-a100-results")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--remote-root", default=DEFAULT_A100_REMOTE_ROOT, show_default=True)
@click.option("--host", default=None)
@click.option("--user", default=None)
@click.option("--transfer-mode", type=click.Choice(["auto", "rsync", "scp"]), default="auto", show_default=True)
@click.option("--dry-run/--execute", default=True, show_default=True)
@click.option("--since-last", is_flag=True, default=False, help="Record that this collection pass should only fetch newly finished outputs when implemented by the transfer backend.")
@click.option("--output-json", default=None)
def benchmark_atera_lr_collect_a100_results(benchmark_root, remote_root, host, user, transfer_mode, dry_run, since_last, output_json):
    """Generate or execute result recovery commands from A100."""
    payload = collect_a100_results(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        host=host,
        user=user,
        transfer_mode=transfer_mode,
        dry_run=dry_run,
        since_last=since_last,
    )
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(payload, indent=2))


def main():
    app()


if __name__ == "__main__":
    main()
