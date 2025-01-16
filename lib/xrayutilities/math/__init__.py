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
# Copyright (C) 2010-2025 Dominik Kriegner <dominik.kriegner@gmail.com>

from .algebra import solve_quartic
from .fit import (fit_peak2d, gauss_fit, linregress, multPeakFit, multPeakPlot,
                  peak_fit)
from .functions import (Debye1, Gauss1d, Gauss1d_der_p, Gauss1d_der_x,
                        Gauss1dArea, Gauss2d, Gauss2dArea, Gauss3d, Lorentz1d,
                        Lorentz1d_der_p, Lorentz1d_der_x, Lorentz1dArea,
                        Lorentz2d, NormGauss1d, NormLorentz1d, PseudoVoigt1d,
                        PseudoVoigt1d_der_p, PseudoVoigt1d_der_x,
                        PseudoVoigt1dArea, PseudoVoigt1dasym,
                        PseudoVoigt1dasym2, PseudoVoigt2d, TwoGauss2d,
                        heaviside, kill_spike, multPeak1d, multPeak2d, smooth)
from .misc import center_of_mass, derivative, fwhm_exp, gcd
from .transforms import (ArbRotation, AxisToZ, AxisToZ_keepXY,
                         CoordinateTransform, Transform, VecAngle, VecCross,
                         VecDot, VecNorm, VecUnit, XRotation, YRotation,
                         ZRotation, distance, getSyntax, getVector, rotarb)
