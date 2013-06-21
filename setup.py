# This file is part of xrayutilities.
#
# xrayutilies is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2010-2011,2013 Dominik Kriegner <dominik.kriegner@gmail.com>

from distutils import ccompiler
from distutils.core import setup, Extension
import os.path
import numpy
import shutil

# check existence of libraries for extension module
cflags = ['-std=c99']
user_macros = []
libraries = []
compiler=ccompiler.new_compiler()
compiler.output_dir = '_config_tmp'

# check for OpenMP
if compiler.has_function('omp_set_dynamic',libraries=('gomp',)):
      cflags.append('-fopenmp')
      user_macros.append(('__OPENMP__','1'))
      libraries.append('gomp')
else:
    print('Warning: did not find openmp + header files -> using serial code')
# remove temporary directory
try: shutil.rmtree(compiler.output_dir)
except: pass

extmodul = Extension('xrayutilities.cxrayutilities',
                     sources = [os.path.join('xrayutilities','src','cxrayutilities.c'),
                                os.path.join('xrayutilities','src','gridder_utils.c'),
                                os.path.join('xrayutilities','src','gridder2d.c'),
                                os.path.join('xrayutilities','src','block_average.c'),
                                os.path.join('xrayutilities','src','qconversion.c'),
                                os.path.join('xrayutilities','src','gridder3d.c')],
                     define_macros = user_macros,
                     libraries = libraries,
                     extra_compile_args=cflags)

setup(name="xrayutilities",
      version="0.99",
      author="Eugen Wintersberger, Dominik Kriegner",
      description="package for x-ray diffraction data evaluation",
      author_email="eugen.wintersberger@desy.de, dominik.kriegner@gmail.com",
      maintainer="Dominik Kriegner",
      maintainer_email="dominik.kriegner@gmail.com",
      packages=["xrayutilities","xrayutilities.math","xrayutilities.io","xrayutilities.materials",
                "xrayutilities.analysis"],
      package_data={
          "xrayutilities":["*.conf"],
          "xrayutilities.materials":[os.path.join("data","*.db"),os.path.join("data","*.cif")]},
      requires=['numpy','scipy','matplotlib','tables'],
      include_dirs = [numpy.get_include()],
      ext_modules = [extmodul],
      url="http://xrayutilities.sourceforge.net",
      license="GPLv2"
      )
