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
# Copyright (C) 2010-2011 Dominik Kriegner <dominik.kriegner@gmail.com>

from distutils.core import setup
import os.path

setup(name="xrutils",
      version="0.5",
      author="Eugen Wintersberger",
      description="package for x-ray diffraction data evaluation",
      author_email="eugen.wintersberger@desy.de",
      maintainer="Dominik Kriegner",
      maintainer_email="dominik.kriegner@gmail.com",
      package_dir={'':'python'},
      packages=["xrutils","xrutils.math","xrutils.io","xrutils.materials",
                "xrutils.analysis"],
      package_data={
          "xrutils":["*.conf"],
          "xrutils.materials":[os.path.join("data","*.db")]},
      requires=['numpy','scipy','tables'],
      license="GPLv2"
      )
