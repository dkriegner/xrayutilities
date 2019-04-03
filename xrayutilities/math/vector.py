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
# Copyright (C) 2010-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

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

    Parameters
    ----------
    v :     list or array-like
        input vector(s), either one vector or an array of vectors with shape
        (n, 3)

    Returns
    -------
    float or ndarray
        vector norm, either a single float or shape (n, )
    """
    if isinstance(v, numpy.ndarray):
        if len(v.shape) >= 2:
            return numpy.linalg.norm(v, axis=-1)
    if len(v) != 3:
        raise ValueError("Vector must be of length 3, but has length %d!"
                         % len(v))
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def VecUnit(v):
    """
    Calculate the unit vector of v.

    Parameters
    ----------
    v :     list or array-like
        input vector(s), either one vector or an array of vectors with shape
        (n, 3)

    Returns
    -------
    ndarray
        unit vector of `v`, either shape (3, ) or (n, 3)
    """
    vtmp = _checkvec(v)
    if len(vtmp.shape) == 1:
        return vtmp / VecNorm(vtmp)
    else:
        return vtmp / VecNorm(vtmp)[..., numpy.newaxis]


def VecDot(v1, v2):
    """
    Calculate the vector dot product.

    Parameters
    ----------
    v1, v2 :    list or array-like
        input vector(s), either one vector or an array of vectors with shape
        (n, 3)

    Returns
    -------
    float or ndarray
        innter product of the vectors, either a single float or (n, )
    """
    if isinstance(v1, numpy.ndarray):
        if len(v1.shape) >= 2:
            return numpy.einsum('...i, ...i', v1, v2)
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Vectors must be of size 3! (len(v1)=%d len(v2)=%d)"
                         % (len(v1), len(v2)))

    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def VecCross(v1, v2, out=None):
    """
    Calculate the vector cross product.

    Parameters
    ----------
    v1, v2 :    list or array-like
        input vector(s), either one vector or an array of vectors with shape
        (n, 3)
    out :       list or array-like, optional
        output vector

    Returns
    -------
    ndarray
        cross product either of shape (3, ) or (n, 3)
    """
    if isinstance(v1, numpy.ndarray):
        if len(v1.shape) >= 2 or len(v2.shape) >= 2:
            return numpy.cross(v1, v2)
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Vectors must be of size 3! (len(v1)=%d len(v2)=%d)"
                         % (len(v1), len(v2)))
    if out is None:
        out = numpy.empty(3)
    out[0] = v1[1] * v2[2] - v1[2] * v2[1]
    out[1] = v1[2] * v2[0] - v1[0] * v2[2]
    out[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return out


def VecAngle(v1, v2, deg=False):
    """
    calculate the angle between two vectors. The following
    formula is used
    v1.v2 = norm(v1)*norm(v2)*cos(alpha)

    alpha = acos((v1.v2)/(norm(v1)*norm(v2)))

    Parameters
    ----------
    v1, v2 :    list or array-like
        input vector(s), either one vector or an array of vectors with shape
        (n, 3)
    deg:        bool
        True: return result in degree, False: in radiants

    Returns
    -------
    float or ndarray
        the angle included by the two vectors `v1` and `v2`, either a single
        float or an array with shape (n, )
    """
    u1 = VecNorm(v1)
    u2 = VecNorm(v2)

    if isinstance(u1, numpy.ndarray) or isinstance(u2, numpy.ndarray):
        s = VecDot(v1, v2) / u1 / u2
        s[s > 1.0] = 1.0
        alpha = numpy.arccos(s)
        if deg:
            alpha = numpy.degrees(alpha)
    else:
        alpha = math.acos(min(1., VecDot(v1, v2) / u1 / u2))
        if deg:
            alpha = math.degrees(alpha)

    return alpha


def distance(x, y, z, point, vec):
    """
    calculate the distance between the point (x, y, z) and the line defined by
    the point and vector vec

    Parameters
    ----------
    x :     float or ndarray
        x coordinate(s) of the point(s)
    y :     float or ndarray
        y coordinate(s) of the point(s)
    z :     float or ndarray
        z coordinate(s) of the point(s)
    point : tuple, list or ndarray
        3D point on the line to which the distance should be calculated
    vec :   tuple, list or ndarray
        3D vector defining the propergation direction of the line
    """
    coords = numpy.vstack((x - point[0], y - point[1], z - point[2])).T
    return VecNorm(VecCross(coords, numpy.asarray(vec)))/VecNorm(vec)


def getVector(string):
    """
    returns unit vector along a rotation axis given in the syntax
    'x+' 'z-' or equivalents

    Parameters
    ----------
    string:     str
        vector string following the synthax [xyz][+-]

    Returns
    -------
    ndarray
        vector along the given direction
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
                         "(needs to be one of x, y, z)")

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
    like e.g. [1, 0, 0] or [0, 2, 0]

    Parameters
    ----------
    vec :   list or array-like
        vector of length 3

    Returns
    -------
    str
        vector string following the synthax [xyz][+-]
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
