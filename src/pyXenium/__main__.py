import json
from pathlib import Path

import click
import pandas as pd

from .benchmarking import (
    aggregate_standardized_results,
    DEFAULT_A100_READONLY_XENIUM_ROOT,
    DEFAULT_A100_REMOTE_ROOT,
    build_a100_job_manifest,
    build_a100_resource_summary,
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
)
from .io.io import copy_bundled_dataset, load_toy
from .io.spatialdata_export import DEFAULT_SPATIALDATA_STORE_NAME, export_xenium_to_spatialdata_zarr
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
    discovered = sorted(layout.runs_dir.glob("**/*standardized*.tsv"))
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


@benchmark_atera_lr_group.command("collect-a100-results")
@click.option("--benchmark-root", default=None, help="Optional benchmark root. Defaults to benchmarking/lr_2026_atera under the repo root.")
@click.option("--remote-root", default=DEFAULT_A100_REMOTE_ROOT, show_default=True)
@click.option("--host", default=None)
@click.option("--user", default=None)
@click.option("--transfer-mode", type=click.Choice(["auto", "rsync", "scp"]), default="auto", show_default=True)
@click.option("--dry-run/--execute", default=True, show_default=True)
@click.option("--output-json", default=None)
def benchmark_atera_lr_collect_a100_results(benchmark_root, remote_root, host, user, transfer_mode, dry_run, output_json):
    """Generate or execute result recovery commands from A100."""
    payload = collect_a100_results(
        benchmark_root=benchmark_root,
        remote_root=remote_root,
        host=host,
        user=user,
        transfer_mode=transfer_mode,
        dry_run=dry_run,
    )
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(payload, indent=2))


def main():
    app()


if __name__ == "__main__":
    main()
