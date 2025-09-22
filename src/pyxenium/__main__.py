import shutil
from pathlib import Path
import click
from .io import load_toy

@click.group()
def app():
    """pyxenium: Xenium toolkit (toy data included)"""

@app.command()
def demo():
    ds = load_toy()
    click.echo(f"Loaded groups: {list(ds)}")

@app.command()
@click.option("--name", default="toy_slide", show_default=True)
@click.option("--url", default=None, help="Optional URL to download a dataset archive")
@click.option("--dest", default=str(Path.home()/".cache"/"pyxenium"), show_default=True)
def datasets(name, url, dest):
    """Fetch example datasets to a local cache."""
    cache = Path(dest); cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if url:
        import urllib.request
        urllib.request.urlretrieve(url, str(target))
        click.echo(f"Downloaded to {target}")
    else:
        from importlib import resources
        base = resources.files("pyxenium.datasets.toy_slide")
        target.mkdir(parents=True, exist_ok=True)
        for fn in ["cells.zarr.zip", "transcripts.zarr.zip", "analysis.zarr.zip"]:
            shutil.copyfile(base/fn, target/fn)
        click.echo(f"Copied bundled toy dataset to {target}")

if __name__ == "__main__":
    app()
