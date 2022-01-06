# -*- coding: utf-8 -*-

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("emcee").version
except DistributionNotFound:
    __version__ = "unknown version"


# General stuff
extensions = [
    'sphinxcontrib.bibtex',
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
]

source_suffix = ".rst"
master_doc = "index"

# myst_nb
myst_enable_extensions = ["dollarmath", "colon_fence"]

# sphinxcontrib.bibtex
bibtex_bibfiles = ['refs.bib']

project = "l1 periodogram"
copyright = "2020-2022, Nathan C. Hara, Denis Rosset, Alessandro R. Mari & contributors"
version = __version__
release = __version__
exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "l1 periodogram"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/denisrosset/l1periodogram",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
jupyter_execute_notebooks = "off"
execution_timeout = -1
