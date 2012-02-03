# This file is part of xrutils.
#
# xrutils is free software; you can redistribute it and/or modify 
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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@aol.at>

"""
xrutils utilities contains a conglomeration of useful functions
this part of utilities does not need the config class
"""

import scipy.constants

def lam2en(inp):
    #{{{
    """
    converts the input energy in eV to a wavelength in Angstrom
    or the input wavelength in Angstrom to an energy in eV

    Parameter
    ---------
     inp : either an energy in eV or an wavelength in Angstrom

    Returns
    -------
     float, energy in eV or wavlength in Angstrom

    Examples
    --------
     >>> lambda = lam2en(8048)
     >>> energy = lam2en(1.5406)
    """
    #  E(eV) = h*c/(e * lambda(A)) *1e10    
    #  lambda(A) = h*c/(e * E(eV)) *1e10
    out = scipy.constants.h*scipy.constants.speed_of_light/(scipy.constants.e* inp) * 1e10

    return out
    #}}}

