# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys
sys.path.insert(0, "../../")
import datetime
import sphinx_rtd_theme
import doctest
import opendelta




# -- Project information -----------------------------------------------------

project = 'OpenDelta'
author = 'THUNLP OpenDelta Team'
copyright = '{}, {}, Licenced under the Apache License, Version 2.0'.format(datetime.datetime.now().year, author)


# The full version, including alpha/beta/rc tags
release = '0.3.2'
version = "0.3.2"

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {'python': ('https://docs.python.org/', None),
"torch": ("https://pytorch.org/docs/stable/", None),}

html_show_sourcelink = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.mathbase',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'sphinx_toolbox.collapse',
    'myst_parser',
]

myst_enable_extensions = [
    "html_image", 
    "colon_fence", 
    "html_admonition",
    "amsmath",
    "dollarmath",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme_options = {
    # 'collapse_navigation': False,
    # 'display_version': True,
    #'logo_only': False,
    'navigation_depth': 2,
}


html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_js_files = ['js/custom.js']
rst_context = {'opendelta': opendelta}
# rst_epilog = "\n.. include:: .special.rst\n"
add_module_names = False

def include_only_tagged(app, what, name, obj, skip, options):
    inclusion_tag_format = "[NODOC]" #can be any pattern here, choose what works for you
    for tag in app.tags.tags:
        if obj.__doc__ is not None and not obj.__doc__.startswith(inclusion_tag_format):
            return False
    return True

def skip2(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

def skip(app, what, name, obj, skip, options):
    skip = include_only_tagged(app, what, name, obj, skip, options) or\
            skip2(app, what, name, obj, skip, options)
    return skip

def setup(app):
    
    

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect('autodoc-skip-member', skip)
    app.connect("source-read", rst_jinja_render)