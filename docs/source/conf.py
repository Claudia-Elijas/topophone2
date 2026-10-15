# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'topophone2'
copyright = '2026, jack-andrew smith'
author = 'jack-andrew smith'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc -------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_member_order = 'bysource'

# Mock heavy/unavailable dependencies so autodoc can import topophone2.inversion
autodoc_mock_imports = ['geopvi']

# Merge the class docstring (Attributes/Methods) with the __init__
# docstring (Input) so constructor parameters are documented too.
autoclass_content = 'both'

# -- Napoleon --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
# Docstrings here use custom "Input"/"Output" headers instead of the usual
# "Parameters"/"Returns", so map them onto the same rendering.

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Without this, each class "Attributes" entry becomes its own
# `.. attribute::` directive with its own anchor, which clutters the
# page's sidebar ToC alongside the actual methods. This renders them as
# a plain field list instead, still inline in the class docstring.
napoleon_use_ivar = True

napoleon_custom_sections = [
    ('Input', 'params_style'),
    ('Output', 'returns_style'),
    # Without this, a class "Methods" section gets turned into individual
    # `.. method::` stub directives that clash with the real automethod-
    # documented methods below. This makes it render as plain RST instead
    # (a bullet list, in this case), so it's just a summary, not a
    # duplicate definition.
    'Methods',
]

# -- nbsphinx --------------------------------------------------------------
# https://nbsphinx.readthedocs.io/en/latest/configuration.html
# Render the outputs already stored in the notebooks rather than re-running
# them at build time (they need data files and geopvi, which CI lacks).

nbsphinx_execute = 'never'

# Static gallery thumbnails for notebooks that have no plotted figures (the
# others use their last figure automatically). Keys are notebook paths relative
# to this folder, without ".ipynb"; values are image paths relative to this
# folder.
nbsphinx_thumbnails = {
    'examples/example_1/example_1.1': '_static/thumbnails/example_1.1.png',
    'examples/example_1/example_1.2': '_static/thumbnails/example_1.2.png',
    'examples/example_1/example_1.3': '_static/thumbnails/example_1.3.png',
    'examples/example_2/example_2.1': '_static/thumbnails/example_2.1.png',
    'examples/example_2/example_2.2': '_static/thumbnails/example_2.2.png',
    'examples/example_3/example_3.1': '_static/thumbnails/example_3.1.png',
    'examples/example_3/example_3.2': '_static/thumbnails/example_3.2.png',
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = 'topophone2'
html_static_path = ['_static']
html_css_files = ['custom.css']

# Show a home icon (instead of the project name) in the top-left corner of the
# navigation bar, linking back to the home page.
html_logo = '_static/home.svg'
html_theme_options = {
    'logo': {
        'alt_text': 'topophone2 home page',
    },
}
