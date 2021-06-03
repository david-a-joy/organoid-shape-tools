#!/usr/bin/env python3

import os

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

import numpy as np

basedir = os.path.dirname(os.path.realpath(__file__))
aboutfile = os.path.join(basedir, 'organoid_shape_tools', '__about__.py')
scriptdir = os.path.join(basedir, 'scripts')

# Load the info from the about file
about = {}
with open(aboutfile) as f:
    exec(f.read(), about)

scripts = [os.path.join('scripts', p)
           for p in os.listdir(scriptdir)
           if os.path.isfile(os.path.join(scriptdir, p)) and not p.startswith('.')]

include_dirs = [np.get_include()]
# define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
define_macros = []

# Cython compile all the things
ext_modules = [
    Extension('organoid_shape_tools.utils._poly',
              sources=['organoid_shape_tools/utils/_poly.pyx'],
              include_dirs=include_dirs,
              define_macros=define_macros),
]

setup(
    name=about['__package_name__'],
    version=about['__version__'],
    url=about['__url__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    ext_modules=cythonize(ext_modules, language_level=3),
    packages=('organoid_shape_tools',
              'organoid_shape_tools.plotting',
              'organoid_shape_tools.utils'),
    scripts=scripts,
)
