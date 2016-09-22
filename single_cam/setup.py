# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import numpy as np
import os

old_pyptv = os.path.expanduser('~/pyptv-git/src_c')
inc_dirs = [np.get_include(), old_pyptv]

ext_mods = [
    Extension("calib", ["calib.pyx"],
        include_dirs=inc_dirs, libraries=['optv'],
        extra_compile_args=['-O3'], pyrex_include_dirs=['.'])
    ]

setup(
    name="calib",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_mods,
)
