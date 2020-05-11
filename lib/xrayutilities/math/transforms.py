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
# Copyright (C) 2009-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from .. import config
from . import vector


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
        e1 = vector._checkvec(v1)
        e2 = vector._checkvec(v2)
        e3 = vector._checkvec(v3)

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
        newz = vector._checkvec(newzaxis)

        if numpy.isclose(vector.VecAngle([0, 0, 1], newz), 0):
            newx = [1, 0, 0]
            newy = [0, 1, 0]
            newz = [0, 0, 1]
        elif numpy.isclose(vector.VecAngle([0, 0, 1], -newz), 0):
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
        newz = vector._checkvec(newzaxis)

        if numpy.isclose(vector.VecAngle([0, 0, 1], newz), 0):
            newx = [1, 0, 0]
            newy = [0, 1, 0]
            newz = [0, 0, 1]
        elif numpy.isclose(vector.VecAngle([0, 0, 1], -newz), 0):
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
    axis = vector._checkvec(axis)
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
