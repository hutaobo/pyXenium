import json
from pathlib import Path

import click

from .io.io import copy_bundled_dataset, load_toy
from .validation import DEFAULT_DATASET_PATH, run_validated_renal_ffpe_smoke


@click.group()
def app():
    """pyXenium: Xenium toolkit (toy data included)"""


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
    """Validate pyXenium against the public 10x FFPE renal RNA + Protein dataset."""
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


def main():
    app()


if __name__ == "__main__":
    main()
