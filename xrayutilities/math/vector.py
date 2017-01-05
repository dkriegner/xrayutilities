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
# Copyright (C) 2010-2017 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module with vector operations for vectors of size 3,
since for so short vectors numpy does not give the best performance explicit
implementation of the equations is performed together with error checking to
ensure vectors of length 3.
"""

import math
import re

import numpy

from .. import config
from ..exception import InputError

circleSyntax = re.compile("[xyz][+-]")


def _checkvec(v):
    if isinstance(v, (list, tuple, numpy.ndarray)):
        vtmp = numpy.asarray(v, dtype=numpy.double)
    else:
        raise TypeError("Vector must be a list, tuple or numpy array")
    return vtmp


def VecNorm(v):
    """
    Calculate the norm of a vector.

    required input arguments:
     v .......... vector as list or numpy array

    return value:
     float holding the vector norm
    """
    if len(v) != 3:
        raise ValueError("Vector must be of size 3, but has size %d!" % len(v))

    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def VecUnit(v):
    """
    Calculate the unit vector of v.

    required input arguments:
     v ........... vector as list or numpy array

    return value:
     numpy array with the unit vector
    """
    vtmp = _checkvec(v)
    return vtmp / VecNorm(vtmp)


def VecDot(v1, v2):
    """
    Calculate the vector dot product.

    required input arguments:
     v1 .............. vector as numpy array or list
     v2 .............. vector as numpy array or list

    return value:
     float value
    """
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Vectors must be of size 3! (len(v1)=%d len(v2)=%d)"
                         % (len(v1), len(v2)))

    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def VecAngle(v1, v2, deg=False):
    """
    calculate the angle between two vectors. The following
    formula is used
    v1.v2 = norm(v1)*norm(v2)*cos(alpha)

    alpha = acos((v1.v2)/(norm(v1)*norm(v2)))

    required input arguments:
     v1 .............. vector as numpy array or list
     v2 .............. vector as numpy array or list

    optional keyword arguments:
     deg ............. (default: false) return result in degree
                       otherwise in radiants

    return value:
     float value with the angle inclined by the two vectors
    """
    u1 = VecNorm(v1)
    u2 = VecNorm(v2)

    alpha = numpy.arccos(numpy.minimum(1., VecDot(v1, v2) / u1 / u2))
    if deg:
        alpha = numpy.degrees(alpha)

    return alpha


def getVector(string):
    """
    returns unit vector along a rotation axis given in the syntax
    'x+' 'z-' or equivalents

    Parameters
    ----------
     string   [xyz][+-]

    Returns
    -------
     vector along the given direction as numpy array
    """

    if len(string) != 2:
        raise InputError("wrong length of string for conversion given")
    if not circleSyntax.search(string):
        raise InputError("getVector: incorrect string syntax (%s)" % string)

    if string[0] == 'x':
        v = [1., 0, 0]
    elif string[0] == 'y':
        v = [0, 1., 0]
    elif string[0] == 'z':
        v = [0, 0, 1.]
    else:
        raise InputError("wrong first character of string given "
                         "(needs to be one of x,y,z)")

    if string[1] == '+':
        v = numpy.asarray(v) * (+1)
    elif string[1] == '-':
        v = numpy.asarray(v) * (-1)
    else:
        raise InputError("wrong second character of string given "
                         "(needs to be + or -)")

    return v


def getSyntax(vec):
    """
    returns vector direction in the syntax
    'x+' 'z-' or equivalents
    therefore works only for principle vectors of the coordinate system
    like e.g. [1,0,0] or [0,2,0]

    Parameters
    ----------
     vec:   vector of length 3

    Returns
    -------
     [xyz][+-]
    """
    v = _checkvec(vec)
    if len(v) != 3:
        raise InputError("no valid 3D vector given")

    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    if VecNorm(numpy.cross(numpy.cross(x, y), v)) <= config.EPSILON:
        if v[2] >= 0:
            string = 'z+'
        else:
            string = 'z-'
    elif VecNorm(numpy.cross(numpy.cross(x, z), v)) <= config.EPSILON:
        if v[1] >= 0:
            string = 'y+'
        else:
            string = 'y-'
    elif VecNorm(numpy.cross(numpy.cross(y, z), v)) <= config.EPSILON:
        if v[0] >= 0:
            string = 'x+'
        else:
            string = 'x-'
    else:
        raise InputError("no valid 3D vector given")

    return string
