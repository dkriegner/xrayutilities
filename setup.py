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
import os.path
import numpy

extmodul = Extension('cxrayutilities',
                     extra_compile_args=['-std=c99'],
                     sources = [os.path.join('src','cxrayutilities.c'),
                                os.path.join('src','gridder_utils.c'),
                                os.path.join('src','gridder2d.c')])

setup(name="xrayutilities",
      version="0.99",
      author="Eugen Wintersberger",
      description="package for x-ray diffraction data evaluation",
      author_email="eugen.wintersberger@desy.de",
      maintainer="Dominik Kriegner",
      maintainer_email="dominik.kriegner@gmail.com",
      package_dir={'':'python'},
      packages=["xrayutilities","xrayutilities.math","xrayutilities.io","xrayutilities.materials",
                "xrayutilities.analysis"],
      package_data={
          "xrayutilities":["*.conf"],
          "xrayutilities.materials":[os.path.join("data","*.db"),os.path.join("data","*.cif")]},
      requires=['numpy','scipy','tables'],
      include_dirs = [numpy.get_include()],
      ext_modules = [extmodul],
      license="GPLv2"
      )
