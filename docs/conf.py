#
# Copyright (c) 2020 Petuum, Inc. All rights reserved.
#
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

project_root = os.path.abspath('..')

try:
    import autodist
except ImportError:
    sys.path.insert(0, project_root)


# -- Project information -----------------------------------------------------

project = 'AutoDist'
project_lower = project.lower()
copyright = '2020, Petuum'
author = 'Petuum'


# The full version, including alpha/beta/rc tags
release = os.popen("cd " + project_root + " && bash GENVER").read().strip()
# The short X.Y version
# version = open(os.path.join(project_root, 'VERSION')).read().strip()
version = release


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',  # Google Docstring Format
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx_git',  # For embedding changelog
]

# Disable documentation inheritance so as to avoid inheriting
# docstrings in a different format, e.g. when the parent class
# is a PyTorch class.
autodoc_inherit_docstrings = True
add_module_names = False


autodoc_default_options = {
    'member-order': 'bysource',
    'inherited-members': True,
    'show-inheritance': True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# from recommonmark.parser import CommonMarkParser
# source_parsers = {
#     '.md': CommonMarkParser,
# }
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'

# Register the theme as an extension to generate a sitemap.xml
extensions.append("sphinx_rtd_theme")
# logo
html_logo = '_static/img/logo.png'
html_favicon = '_static/img/favicon.ico'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'style_nav_header_background': 'white',
    'display_version': True,
    'logo_only': True,
    'collapse_navigation': False,
}

html_context = {
    'css_files': [
        # 'https://fonts.googleapis.com/css?family=Roboto',
        '_static/css/customized.css'
    ],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project_lower + 'doc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project_lower + '.tex', project_lower + ' Documentation',
     'Petuum', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, project_lower, project_lower + ' Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, project_lower, project_lower + ' Documentation',
     author, project_lower, 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

def docstring(app, what, name, obj, options, lines):
    if name.startswith('autodist.runner') or name.startswith('autodist.checkpoint'):
        options['inherited-members'] = False

def setup(app):
    app.connect('autodoc-process-docstring', docstring)


# -- Options for code link ---------------------------------------------------

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    if info['module'].endswith('_pb2'):
        return None
    path = info['module'].replace('.', '/')
    return "https://gitlab.int.petuum.com/internal/scalable-ml/autodist/tree/master/" + path + '.py'

# -- Options for intersphinx and extlinks extension --------------------------

intersphinx_mapping = {'https://docs.python.org/': None}

extlinks = {
    'tf_main': ('https://www.tensorflow.org/api_docs/python/tf/%s', None)
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
