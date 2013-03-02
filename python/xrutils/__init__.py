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
# Copyright (C) 2009-2011 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrutils is a package for assisting with x-ray diffraction experiments

It helps with planning experiments as well as analyzing the data.

Authors: Dominik Kriegner and Eugen Wintersberger
"""

# load configuration
from . import config

from . import math
from . import io
from . import materials
from . import analysis

from .experiment import Experiment
from .experiment import HXRD
from .experiment import NonCOP
from .experiment import GID
from .experiment import GID_ID10B
from .experiment import GISAXS
from .experiment import Powder

from .normalize import blockAverage1D
from .normalize import blockAverage2D
from .normalize import blockAveragePSD
from .normalize import IntensityNormalizer

from .gridder import Gridder1D
from .gridder import Gridder2D
from .gridder import Gridder3D

from .utilities import maplog
from .utilities import lam2en
from .utilities import wavelength
from .utilities import energy
