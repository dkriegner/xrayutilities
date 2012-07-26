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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to calculate the first Debye function as needed 
for the calculation of the thermal Debye-Waller-factor

for definition see:
http://en.wikipedia.org/wiki/Debye_function

"""

import numpy
import scipy.integrate

from .. import config

def Debye1(x):
    """
    determine Debye function by numerical integration

    D1(x) = (1/x) \int_0^x t/(exp(t)-1) dt

    Parameters:
    -----------
     x ... argument of the Debye function (float)

    Returns:
    --------
     D1(x) float value of the Debye function
     """
     
    def __int_kernel(t):
        """
        integration kernel for the numeric integration
        """
        y = t/(numpy.exp(t)-1)
        return y
    
    if x>0.:
        integral = scipy.integrate.quad(__int_kernel, 0, x)
        d1 = (1/float(x)) * integral[0]
    else:
        integral = (0,0)
        d1 = 1.

    if (config.VERBOSITY >= config.DEBUG):
        print("XU.math.Debye1: debye integral value/error estimate: %g %g"%(integral[0],integral[1]))

    return d1
