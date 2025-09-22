from click.testing import CliRunner
from pyxenium.__main__ import app

def test_demo():
    r = CliRunner().invoke(app, ["demo"])
    assert r.exit_code == 0
