import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'btQuant'
author = 'Brayden Boyko'
release = '1.0.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
