project = "pyXenium"
author = "Taobo Hu"

extensions = [
    "myst_parser",
]

# Treat .md as MyST markdown explicitly
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Copy README.md into docs/_includes for safe include
import os
from pathlib import Path
import shutil

def setup(app):
    src_dir = Path(__file__).parent           # docs/
    project_root = src_dir.parent             # repo root
    readme_src = project_root / "README.md"
    includes_dir = src_dir / "_includes"
    includes_dir.mkdir(exist_ok=True)
    readme_dst = includes_dir / "README.md"
    if readme_src.is_file():
        shutil.copyfile(readme_src, readme_dst)
