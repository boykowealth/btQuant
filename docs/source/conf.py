import os
import sys
sys.path.insert(0, os.path.abspath('..'))

html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints'
]

project = 'btQuant'
author = 'Brayden Boyko'
release = '1.0.2'
