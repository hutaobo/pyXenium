from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_HOST = "pdc"
DEFAULT_PDC_ROOT = "/cfs/klemming/scratch/h/hutaobo/pyxenium_mechanostress_atera_2026-04"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--pdc-root", default=DEFAULT_PDC_ROOT)
    parser.add_argument("--notebook", default="docs/tutorials/mechanostress_atera_pdc.ipynb")
    parser.add_argument("--static-dir", default="docs/_static/tutorials/mechanostress_atera_pdc")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    notebook_target = repo_root / args.notebook
    static_target = repo_root / args.static_dir
    static_target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run(
            [
                "scp",
                f"{args.host}:{args.pdc_root.rstrip('/')}/notebooks/mechanostress_atera_pdc.executed.ipynb",
                str(tmp / "mechanostress_atera_pdc.ipynb"),
            ]
        )
        notebook_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp / "mechanostress_atera_pdc.ipynb", notebook_target)

        run(["scp", "-r", f"{args.host}:{args.pdc_root.rstrip('/')}/static/mechanostress_atera_pdc/.", str(static_target)])

    print(f"[mechanostress-pdc] collected notebook to {notebook_target}")
    print(f"[mechanostress-pdc] collected static artifacts to {static_target}")


if __name__ == "__main__":
    main()
