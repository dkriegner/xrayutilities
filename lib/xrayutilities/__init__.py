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
# Copyright (C) 2009-2025 Dominik Kriegner <dominik.kriegner@gmail.com>

"""xrayutilities assisting with x-ray diffraction experiments.

It helps with planning experiments as well as analyzing the data.

Authors:
 Dominik Kriegner <dominik.kriegner@gmail.com> and
 Eugen Wintersberger <eugen.wintersberger@desy.de>
"""
from importlib.metadata import version

# load configuration
from . import analysis, config, io, materials, math, simpack  # noqa: F401
from .experiment import (  # noqa: F401
                         GID,
                         GISAXS,
                         HXRD,
                         Experiment,
                         FourC,
                         NonCOP,
                         PowderExperiment,
                         QConversion,
)
from .gridder import FuzzyGridder1D, Gridder1D, npyGridder1D  # noqa: F401
from .gridder2d import FuzzyGridder2D, Gridder2D, Gridder2DList  # noqa: F401
from .gridder3d import FuzzyGridder3D, Gridder3D  # noqa: F401
from .normalize import (  # noqa: F401
                         IntensityNormalizer,
                         blockAverage1D,
                         blockAverage2D,
                         blockAverageCCD,
                         blockAveragePSD,
)
from .q2ang_fit import Q2AngFit  # noqa: F401
from .utilities import (  # noqa: F401
                         en2lam,
                         energy,
                         frac2str,
                         lam2en,
                         makeNaturalName,
                         maplog,
                         wavelength,
)

# load package version
__version__ = version("xrayutilities")
