from click.testing import CliRunner
from pyXenium.__main__ import app
from pyXenium.io.io import load_toy

def test_demo():
    r = CliRunner().invoke(app, ["demo"])
    assert r.exit_code == 0
    assert "Loaded groups" in r.output


def test_load_toy():
    toy = load_toy()
    assert set(toy) == {"analysis", "cells", "transcripts"}


def test_datasets_command_copies_bundled_files(tmp_path):
    r = CliRunner().invoke(app, ["datasets", "--dest", str(tmp_path)])
    assert r.exit_code == 0

    target = tmp_path / "toy_slide"
    assert target.exists()
    assert (target / "analysis.zarr.zip").exists()
    assert (target / "cells.zarr.zip").exists()
    assert (target / "transcripts.zarr.zip").exists()
