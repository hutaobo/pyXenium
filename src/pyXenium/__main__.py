import json
from pathlib import Path

import click

from .io.io import copy_bundled_dataset, load_toy
from .io.spatialdata_export import DEFAULT_SPATIALDATA_STORE_NAME, export_xenium_to_spatialdata_zarr
from .multimodal import (
    DEFAULT_DATASET_PATH,
    run_renal_immune_resistance_pilot,
    run_validated_renal_ffpe_smoke,
)
from .validation import DEFAULT_ATERA_WTA_BREAST_DATASET_PATH, run_atera_wta_breast_topology


@click.group()
def app():
    """pyXenium: Xenium toolkit (toy data included)"""


@app.group("multimodal")
def multimodal_group():
    """Joint RNA + protein analysis and workflow commands."""


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


def main():
    app()


if __name__ == "__main__":
    main()
