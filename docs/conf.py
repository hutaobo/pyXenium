from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyXenium._version import __version__

project = "pyXenium"
author = "Taobo Hu"
copyright = "2025, pyXenium contributors"
release = __version__
version = __version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
add_module_names = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"pyXenium {release}"
html_static_path = ["_static"]
html_css_files = ["pyxenium.css"]
html_favicon = "_static/branding/pyxenium-favicon.png"

html_context = {
    "github_user": "hutaobo",
    "github_repo": "pyXenium",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "logo": {
        "image_light": "_static/branding/pyxenium-horizontal-dark.svg",
        "image_dark": "_static/branding/pyxenium-horizontal-light.svg",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navigation_with_keys": True,
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc"],
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/hutaobo/pyXenium",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyXenium/",
            "icon": "fa-solid fa-cube",
        },
        {
            "name": "Read the Docs",
            "url": "https://pyxenium.readthedocs.io/en/latest/",
            "icon": "fa-solid fa-book-open",
        },
    ],
}
