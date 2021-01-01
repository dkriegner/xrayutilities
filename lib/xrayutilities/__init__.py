# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
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
# Copyright (C) 2009-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrayutilities is a Python package for assisting with x-ray diffraction
experiments. Its the python package included in *xrayutilities*.

It helps with planning experiments as well as analyzing the data.

Authors:
 Dominik Kriegner <dominik.kriegner@gmail.com> and
 Eugen Wintersberger <eugen.wintersberger@desy.de>
"""
import os

# load configuration
from . import __path__, analysis, config, io, materials, math, simpack
from .experiment import (GID, GISAXS, HXRD, Experiment, FourC, NonCOP,
                         PowderExperiment, QConversion)
from .gridder import FuzzyGridder1D, Gridder1D, npyGridder1D
from .gridder2d import FuzzyGridder2D, Gridder2D, Gridder2DList
from .gridder3d import FuzzyGridder3D, Gridder3D
from .normalize import (IntensityNormalizer, blockAverage1D, blockAverage2D,
                        blockAverageCCD, blockAveragePSD)
from .q2ang_fit import Q2AngFit
from .utilities import (clear_bit, en2lam, energy, frac2str, lam2en,
                        makeNaturalName, maplog, set_bit, wavelength)

# load package version
with open(os.path.join(__path__[0], 'VERSION')) as version_file:
    __version__ = version_file.read().strip().replace('\n', '.')
