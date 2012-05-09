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
# Copyright (C) 2010,2012 Dominik Kriegner <dominik.kriegner@aol.at>

"""
module with vector operations,
mostly numpy functionality is used for the vector operation itself,
however custom error checking is done to ensure vectors of length 3.
"""

import numpy

from .. import config

def VecNorm(v):
    """
    VecNorm(v):
    Calculate the norm of a vector.

    required input arguments:
    v .......... vector as list or numpy array

    return value:
    float holding the vector norm
    """
    if isinstance(v,list):
        vtmp = numpy.array(v,dtype=numpy.double)
    elif isinstance(v,numpy.ndarray):
        vtmp = v.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy array")

    if vtmp.size != 3:
        raise ValueError("Vector must be of size 3, but has size %d!"%vtmp.size)

    return numpy.linalg.norm(vtmp)

def VecUnit(v):
    """
    VecUnit(v):
    Calculate the unit vector of v.

    required input arguments:
    v ........... vector as list or numpy array

    return value:
    numpy array with the unit vector
    """
    if isinstance(v,list):
        vtmp = numpy.array(v,dtype=numpy.double)
    elif isinstance(v,numpy.ndarray):
        vtmp = v.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy arra")

    return vtmp/VecNorm(vtmp)

def VecDot(v1,v2):
    """
    VecDot(v1,v2):
    Calculate the vector dot product.

    required input arguments:
    v1 .............. vector as numpy array or list
    v2 .............. vector as numpy array or list

    return value:
    float value
    """
    if isinstance(v1,list):
        v1tmp = numpy.array(v1,dtype=numpy.double)
    elif isinstance(v1,numpy.ndarray):
        v1tmp = v1.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy array")

    if isinstance(v2,list):
        v2tmp = numpy.array(v2,dtype=numpy.double)
    elif isinstance(v2,numpy.ndarray):
        v2tmp = v2.astype(numpy.double)
    else:
        raise TypeError("Vector must be a list or numpy array")

    if v1tmp.size != 3 or v2tmp.size != 3:
        raise ValueError("Vectors must be of size 3! (len(v1)=%d len(v2)=%d)" %(v1tmp.size,v2tmp.size))

    return numpy.dot(v1tmp,v2tmp)


def VecAngle(v1,v2,deg=False):
    """
    VecAngle(v1,v2,deg=false):
    calculate the angle between two vectors. The following
    formula is used
    v1.v2 = |v1||v2|cos(alpha)

    alpha = acos((v1.v2)/|v1|/|v2|)

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
    if(config.VERBOSITY >= config.DEBUG):
        print("XU.math.VecAngle: norm of the vectors: %8.5g %8.5g" %(u1,u2))

    alpha = numpy.arccos(VecDot(v1,v2)/u1/u2)
    if deg:
        alpha = numpy.degrees(alpha)

    return alpha

