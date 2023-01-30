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
# Copyright (c) 2009-2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import math
import re

import numpy

from .. import config
from ..exception import InputError

circleSyntax = re.compile("[xyzk][+-]")


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
        raise ValueError(
            f"Vector must be of length 3, but has length {len(v)}!")
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
    deg:        bool, optional
        True: return result in degree, False: in radiants (default: False)

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
        s[numpy.abs(s) > 1.0] = numpy.sign(s[numpy.abs(s) > 1.0]) * 1.0
        alpha = numpy.arccos(s)
        if deg:
            alpha = numpy.degrees(alpha)
    else:
        alpha = math.acos(max(min(1., VecDot(v1, v2) / u1 / u2), -1.))
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
    coords = numpy.vstack((numpy.ravel(x) - point[0],
                           numpy.ravel(y) - point[1],
                           numpy.ravel(z) - point[2])).T
    ret = VecNorm(VecCross(coords, numpy.asarray(vec)))/VecNorm(vec)
    if isinstance(x, numpy.ndarray):
        ret = ret.reshape(x.shape)
    else:
        ret = ret[0]
    return ret


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
        raise InputError(f"getVector: incorrect string syntax ({string})")

    if string[0] == 'x':
        v = [1., 0, 0]
    elif string[0] == 'y':
        v = [0, 1., 0]
    elif string[0] == 'z':
        v = [0, 0, 1.]
    elif string[0] == 'k':
        # determine reference direction
        if config.KAPPA_PLANE[0] == 'x':
            v = numpy.array((1., 0, 0))
            # turn reference direction
            if config.KAPPA_PLANE[1] == 'y':
                v = ZRotation(config.KAPPA_ANGLE)(v)
            elif config.KAPPA_PLANE[1] == 'z':
                v = YRotation(-1 * config.KAPPA_ANGLE)(v)
            else:
                raise TypeError("getVector: invalid kappa_plane in config!")
        elif config.KAPPA_PLANE[0] == 'y':
            v = numpy.array((0, 1., 0))
            # turn reference direction
            if config.KAPPA_PLANE[1] == 'z':
                v = XRotation(config.KAPPA_ANGLE)(v)
            elif config.KAPPA_PLANE[1] == 'x':
                v = ZRotation(-1 * config.KAPPA_ANGLE)(v)
            else:
                raise TypeError("getVector: invalid kappa_plane in config!")
        elif config.KAPPA_PLANE[0] == 'z':
            v = numpy.array((0, 0, 1.))
            # turn reference direction
            if config.KAPPA_PLANE[1] == 'x':
                v = YRotation(config.KAPPA_ANGLE)(v)
            elif config.KAPPA_PLANE[1] == 'y':
                v = XRotation(-1 * config.KAPPA_ANGLE)(v)
            else:
                raise TypeError("getVector: invalid kappa_plane in config!")
        else:
            raise TypeError("getVector: invalid kappa_plane in config!")
    else:
        raise InputError("wrong first character of string given "
                         "(needs to be one of x, y, z, or k)")

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

    if numpy.isclose(VecNorm(numpy.cross(numpy.cross(x, y), v)), 0):
        if v[2] >= 0:
            string = 'z+'
        else:
            string = 'z-'
    elif numpy.isclose(VecNorm(numpy.cross(numpy.cross(x, z), v)), 0):
        if v[1] >= 0:
            string = 'y+'
        else:
            string = 'y-'
    elif numpy.isclose(VecNorm(numpy.cross(numpy.cross(y, z), v)), 0):
        if v[0] >= 0:
            string = 'x+'
        else:
            string = 'x-'
    else:
        raise InputError("no valid 3D vector given")

    return string


class Transform(object):

    def __init__(self, matrix):
        self.matrix = matrix
        self._imatrix = None

    @property
    def imatrix(self):
        if self._imatrix is None:
            try:
                self._imatrix = numpy.linalg.inv(self.matrix)
            except numpy.linalg.LinAlgError:
                raise Exception("XU.math.Transform: matrix cannot be inverted"
                                " - seems to be singular")
        return self._imatrix

    def inverse(self, args, rank=1):
        """
        performs inverse transformation a vector, matrix or tensor of rank 4

        Parameters
        ----------
        args :      list or array-like
            object to transform, list or numpy array of shape (..., n)
            (..., n, n), (..., n, n, n, n) where n is the size of the
            transformation matrix.
        rank :      int
            rank of the supplied object. allowed values are 1, 2, and 4
        """
        it = Transform(self.imatrix)
        return it(args, rank)

    def __call__(self, args, rank=1):
        """
        transforms a vector, matrix or tensor of rank 4
        (e.g. elasticity tensor)

        Parameters
        ----------
        args :      list or array-like
            object to transform, list or numpy array of shape (..., n)
            (..., n, n), (..., n, n, n, n) where n is the size of the
            transformation matrix.
        rank :      int
            rank of the supplied object. allowed values are 1, 2, and 4
        """

        m = self.matrix
        if rank == 1:  # argument is a vector
            # out_i = m_ij * args_j
            out = numpy.einsum('ij,...j', m, args)
        elif rank == 2:  # argument is a matrix
            # out_ij = m_ik * m_jl * args_kl
            out = numpy.einsum('ik, jl,...kl', m, m, args)
        elif rank == 4:
            # cp_ijkl = m_in * m_jo * m_kp * m_lq * args_nopq
            out = numpy.einsum('in, jo, kp, lq,...nopq', m, m, m, m, args)

        return out

    def __str__(self):
        ostr = "Transformation matrix:\n"
        ostr += str(self.matrix)
        return ostr


class CoordinateTransform(Transform):

    """
    Create a Transformation object which transforms a point into a new
    coordinate frame. The new frame is determined by the three vectors
    v1/norm(v1), v2/norm(v2) and v3/norm(v3), which need to be orthogonal!
    """

    def __init__(self, v1, v2, v3):
        """
        initialization routine for Coordinate transformation

        Parameters
        ----------
        v1, v2, v3 :     list, tuple or array-like
            new base vectors

        Returns
        -------
        Transform
            An instance of a Transform class
        """
        e1 = _checkvec(v1)
        e2 = _checkvec(v2)
        e3 = _checkvec(v3)

        # normalize base vectors
        e1 = e1 / numpy.linalg.norm(e1)
        e2 = e2 / numpy.linalg.norm(e2)
        e3 = e3 / numpy.linalg.norm(e3)

        # check that the vectors are orthogonal
        t1 = numpy.abs(numpy.dot(e1, e2))
        t2 = numpy.abs(numpy.dot(e1, e3))
        t3 = numpy.abs(numpy.dot(e2, e3))
        if not numpy.allclose((t1, t2, t3), 0):
            raise ValueError("given basis vectors need to be orthogonal!")

        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.math.CoordinateTransform: new basis set: \n"
                  " x: (%5.2f %5.2f %5.2f) \n"
                  " y: (%5.2f %5.2f %5.2f) \n"
                  " z: (%5.2f %5.2f %5.2f)"
                  % (e1[0], e1[1], e1[2], e2[0], e2[1],
                     e2[2], e3[0], e3[1], e3[2]))

        # assemble the transformation matrix
        m = numpy.array([e1, e2, e3])

        Transform.__init__(self, m)


class AxisToZ(CoordinateTransform):

    """
    Creates a coordinate transformation to move a certain axis to the z-axis.
    The rotation is done along the great circle.  The x-axis of the new
    coordinate frame is created to be normal to the new and original z-axis.
    The new y-axis is create in order to obtain a right handed coordinate
    system.
    """

    def __init__(self, newzaxis):
        """
        initialize the CoordinateTransformation to move a certain axis to the
        z-axis

        Parameters
        ----------
        newzaxis :  list or array-like
            new z-axis
        """
        newz = _checkvec(newzaxis)

        if numpy.isclose(VecAngle([0, 0, 1], newz), 0):
            newx = [1, 0, 0]
            newy = [0, 1, 0]
            newz = [0, 0, 1]
        elif numpy.isclose(VecAngle([0, 0, 1], -newz), 0):
            newx = [-1, 0, 0]
            newy = [0, 1, 0]
            newz = [0, 0, -1]
        else:
            newx = numpy.cross(newz, [0, 0, 1])
            newy = numpy.cross(newz, newx)

        CoordinateTransform.__init__(self, newx, newy, newz)


class AxisToZ_keepXY(CoordinateTransform):

    """
    Creates a coordinate transformation to move a certain axis to the z-axis.
    The rotation is done along the great circle.  The x-axis/y-axis of the new
    coordinate frame is created to be similar to the old x and y directions.
    This variant of AxisToZ assumes that the new Z-axis has its main component
    along the Z-direction
    """

    def __init__(self, newzaxis):
        """
        initialize the CoordinateTransformation to move a certain axis to the
        z-axis

        Parameters
        ----------
        newzaxis :  list or array-like
            new z-axis
        """
        newz = _checkvec(newzaxis)

        if numpy.isclose(VecAngle([0, 0, 1], newz), 0):
            newx = [1, 0, 0]
            newy = [0, 1, 0]
            newz = [0, 0, 1]
        elif numpy.isclose(VecAngle([0, 0, 1], -newz), 0):
            newx = [-1, 0, 0]
            newy = [0, 1, 0]
            newz = [0, 0, -1]
        else:
            newx = numpy.cross(newz, [0, 0, 1])
            newy = numpy.cross(newz, newx)
            # rotate newx and newy to be similar to old directions
            ang = numpy.degrees(numpy.arctan2(newz[0], newz[1]))
            newx = rotarb(newx, newz, ang)
            newy = rotarb(newy, newz, ang)

        CoordinateTransform.__init__(self, newx, newy, newz)


def _sincos(alpha, deg):
    if deg:
        a = numpy.radians(alpha)
    else:
        a = alpha
    return numpy.sin(a), numpy.cos(a)


def XRotation(alpha, deg=True):
    """
    Returns a transform that represents a rotation about the x-axis
    by an angle alpha. If deg=True the angle is assumed to be in
    degree, otherwise the function expects radiants.
    """
    sina, cosa = _sincos(alpha, deg)
    m = numpy.array(
        [[1, 0, 0], [0, cosa, -sina], [0, sina, cosa]], dtype=numpy.double)
    return Transform(m)


def YRotation(alpha, deg=True):
    """
    Returns a transform that represents a rotation about the y-axis
    by an angle alpha. If deg=True the angle is assumed to be in
    degree, otherwise the function expects radiants.
    """
    sina, cosa = _sincos(alpha, deg)
    m = numpy.array(
        [[cosa, 0, sina], [0, 1, 0], [-sina, 0, cosa]], dtype=numpy.double)
    return Transform(m)


def ZRotation(alpha, deg=True):
    """
    Returns a transform that represents a rotation about the z-axis
    by an angle alpha. If deg=True the angle is assumed to be in
    degree, otherwise the function expects radiants.
    """
    sina, cosa = _sincos(alpha, deg)
    m = numpy.array(
        [[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]], dtype=numpy.double)
    return Transform(m)


# helper scripts for rotations around arbitrary axis
def tensorprod(vec1, vec2):
    """
    function implements an elementwise multiplication of two vectors
    """
    return vec1[:, numpy.newaxis] * numpy.ones((3, 3)) * vec2[numpy.newaxis, :]


def mycross(vec, mat):
    """
    function implements the cross-product of a vector with each column of a
    matrix
    """
    out = numpy.zeros((3, 3))
    for i in range(3):
        out[:, i] = numpy.cross(vec, mat[:, i])
    return out


def ArbRotation(axis, alpha, deg=True):
    """
    Returns a transform that represents a rotation around an arbitrary axis by
    the angle alpha. positive rotation is anti-clockwise when looking from
    positive end of axis vector

    Parameters
    ----------
    axis :  list or array-like
        rotation axis
    alpha : float
        rotation angle in degree (deg=True) or in rad (deg=False)
    deg :   bool
        determines the input format of ang (default: True)

    Returns
    -------
    Transform
    """
    axis = _checkvec(axis)
    e = axis / numpy.linalg.norm(axis)
    if deg:
        rad = numpy.radians(alpha)
    else:
        rad = alpha
    get = tensorprod(e, e)
    rot = get + numpy.cos(rad) * (numpy.identity(3) - get) + \
        numpy.sin(rad) * mycross(e, numpy.identity(3))
    return Transform(rot)


def rotarb(vec, axis, ang, deg=True):
    """
    function implements the rotation around an arbitrary axis by an angle ang
    positive rotation is anti-clockwise when looking from positive end of axis
    vector

    Parameters
    ----------
    vec :   list or array-like
        vector to rotate
    axis :  list or array-like
        rotation axis
    ang :   float
        rotation angle in degree (deg=True) or in rad (deg=False)
    deg :   bool
        determines the input format of ang (default: True)

    Returns
    -------
    rotvec :  rotated vector as numpy.array

    Examples
    --------
    >>> rotarb([1, 0, 0],[0, 0, 1], 90)
    array([  6.12323400e-17,   1.00000000e+00,   0.00000000e+00])
    """
    return ArbRotation(axis, ang, deg)(vec)
