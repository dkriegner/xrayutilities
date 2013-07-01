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

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os.path
import numpy

copt =  {'msvc' : ['/openmp',],
         'mingw32' : ['-fopenmp','-std=c99'],
         'unix' : ['-fopenmp','-std=c99','-Wall'] }
lopt =  {'mingw32' : ['-fopenmp'],
         'unix' : ['-lgomp'] }
user_macros = [('__OPENMP__',None)]

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        c = self.compiler.compiler_type
        # set custom compiler options
        if copt.has_key(c):
            for e in self.extensions:
                e.extra_compile_args = copt[ c ]
        if lopt.has_key(c):
            for e in self.extensions:
                e.extra_link_args = lopt[ c ]
        build_ext.build_extensions(self)

extmodul = Extension('xrayutilities.cxrayutilities',
                     sources = [os.path.join('xrayutilities','src','cxrayutilities.c'),
                                os.path.join('xrayutilities','src','gridder_utils.c'),
                                os.path.join('xrayutilities','src','gridder2d.c'),
                                os.path.join('xrayutilities','src','block_average.c'),
                                os.path.join('xrayutilities','src','qconversion.c'),
                                os.path.join('xrayutilities','src','gridder3d.c')],
                     define_macros = user_macros)

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
      cmdclass = {'build_ext': build_ext_subclass },
      url="http://xrayutilities.sourceforge.net",
      license="GPLv2"
      )
