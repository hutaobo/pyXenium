from pathlib import Path

import click

from .io.io import copy_bundled_dataset, load_toy


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


def main():
    app()


if __name__ == "__main__":
    main()
