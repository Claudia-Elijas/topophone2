# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

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



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = 'topophone2'
html_static_path = ['_static']
html_css_files = ['custom.css']
