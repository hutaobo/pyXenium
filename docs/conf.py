# -- Project information -----------------------------------------------------
project = "pyXenium"
author = "Taobo Hu"
copyright = "2025, Taobo Hu"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Enable MyST Markdown
]

# MyST options (optional but handy)
myst_enable_extensions = [
    "colon_fence",      # ::: fences
    "deflist",          # definition lists
    "linkify",          # auto-link raw URLs
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
