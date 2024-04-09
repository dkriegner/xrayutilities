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
# Copyright (C) 2009-2010, 2013
#               Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009 Mario Keplinger <mario.keplinger@jku.at>
# Copyright (c) 2009-2019, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from . import cxrayutilities, exception, utilities
from .gridder import Gridder, GridderFlags, axis, delta, ones


class Gridder3D(Gridder):

    def __init__(self, nx, ny, nz):
        Gridder.__init__(self)

        # check input
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise exception.InputError('None of nx, ny and nz can be smaller '
                                       'than 1!')

        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.zmin = 0
        self.zmax = 0

        self.nx = nx
        self.nz = nz
        self.ny = ny

        self._allocate_memory()

    def _allocate_memory(self):
        """
        Class method to allocate memory for the gridder based on the nx, ny
        class attributes.
        """
        self._gdata = numpy.zeros((self.nx, self.ny, self.nz),
                                  dtype=numpy.double)
        self._gnorm = numpy.zeros((self.nx, self.ny, self.nz),
                                  dtype=numpy.double)

    def SetResolution(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self._allocate_memory()

    def __get_xaxis(self):
        return axis(self.xmin, self.xmax, self.nx)

    def __get_yaxis(self):
        return axis(self.ymin, self.ymax, self.ny)

    def __get_zaxis(self):
        return axis(self.zmin, self.zmax, self.nz)

    def __get_xmatrix(self):
        return ones(self.nx, self.ny, self.nz) *\
            self.xaxis[:, numpy.newaxis, numpy.newaxis]

    def __get_ymatrix(self):
        return ones(self.nx, self.ny, self.nz) *\
            self.yaxis[numpy.newaxis, :, numpy.newaxis]

    def __get_zmatrix(self):
        return ones(self.nx, self.ny, self.nz) *\
            self.zaxis[numpy.newaxis, numpy.newaxis, :]

    zaxis = property(__get_zaxis)
    zmatrix = property(__get_zmatrix)
    xaxis = property(__get_xaxis)
    xmatrix = property(__get_xmatrix)
    yaxis = property(__get_yaxis)
    ymatrix = property(__get_ymatrix)

    def dataRange(self, xmin, xmax, ymin, ymax, zmin, zmax, fixed=True):
        """
        define minimum and maximum data range, usually this is deduced
        from the given data automatically, however, for sequential
        gridding it is useful to set this before the first call of the
        gridder. data outside the range are simply ignored

        Parameters
        ----------
        xmin, ymin, zmin :  float
            minimum value of the gridding range in x, y, z
        xmax, ymax, zmax :  float
            maximum value of the gridding range in x, y, z
        fixed :             bool, optional
            flag to turn fixed range gridding on (True (default)) or off
            (False)
        """
        self.fixed_range = fixed
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def _checktransinput(self, x, y, z, data):
        """
        common checks and reshape commands for the input data. This function
        checks the data type and shape of the input data.
        """
        if not self.keep_data:
            self.Clear()

        x = self._prepare_array(x)
        y = self._prepare_array(y)
        z = self._prepare_array(z)
        data = self._prepare_array(data)

        if x.size != y.size or y.size != z.size or z.size != data.size:
            raise exception.InputError(
                f"XU.{self.__class__.__name__}: size of given datasets "
                "(x, y, z, data) is not equal!")

        if not self.fixed_range:
            # assume that with setting keep_data the user wants to call the
            # gridder more often and obtain a reasonable result
            self.dataRange(x.min(), x.max(),
                           y.min(), y.max(),
                           z.min(), z.max(),
                           self.keep_data)

        return x, y, z, data

    def __call__(self, x, y, z, data):
        """
        Perform gridding on a set of data. After running the gridder
        the 'data' object in the class is holding the gridded data.

        Parameters
        ----------
        x :     ndarray
            numpy array of arbitrary shape with x positions
        y :	ndarray
            numpy array of arbitrary shape with y positions
        z :	ndarray
            numpy array fo arbitrary shape with z positions
        data :	ndarray
            numpy array of arbitrary shape with data values
        """

        x, y, z, data = self._checktransinput(x, y, z, data)

        # remove normalize flag for C-code
        flags = self.flags | GridderFlags.NO_NORMALIZATION
        cxrayutilities.gridder3d(x, y, z, data, self.nx, self.ny, self.nz,
                                 self.xmin, self.xmax,
                                 self.ymin, self.ymax,
                                 self.zmin, self.zmax,
                                 self._gdata, self._gnorm, flags)


class FuzzyGridder3D(Gridder3D):
    """
    An 3D binning class considering every data point to have a finite volume.
    If necessary one data point will be split fractionally over different
    data bins. This is numerically more effort but represents better the
    typical case of a experimental data, which do not represent a mathematical
    point but have a finite size.

    Currently only a quader can be considered as volume during the gridding.
    """

    def __call__(self, x, y, z, data, **kwargs):
        """
        Perform gridding on a set of data. After running the gridder
        the 'data' object in the class is holding the gridded data.

        Parameters
        ----------
        x :	ndarray
            numpy array of arbitrary shape with x positions
        y :	ndarray
            numpy array of arbitrary shape with y positions
        z :     ndarray
            numpy array fo arbitrary shape with z positions
        data :	ndarray
            numpy array of arbitrary shape with data values
        width :	float, tuple or list, optional
            width of one data point. If not given half the bin size will be
            used. The width can be given as scalar if it is equal for all three
            dimensions, or as sequence of length 3.
        """

        valid_kwargs = {'width': 'specifiying fuzzy data size'}
        utilities.check_kwargs(kwargs, valid_kwargs,
                               self.__class__.__name__)

        x, y, z, data = self._checktransinput(x, y, z, data)

        if 'width' in kwargs:
            try:
                length = len(kwargs['width'])
            except TypeError:
                length = 1
            if length == 3:
                wx, wy, wz = kwargs['width']
            else:
                wx = kwargs['width']
                wy = wx
                wz = wx
        else:
            wx = delta(self.xmin, self.xmax, self.nx) / 2.
            wy = delta(self.ymin, self.ymax, self.ny) / 2.
            wz = delta(self.zmin, self.zmax, self.nz) / 2.

        # remove normalize flag for C-code
        flags = self.flags | GridderFlags.NO_NORMALIZATION
        cxrayutilities.fuzzygridder3d(x, y, z, data, self.nx, self.ny, self.nz,
                                      self.xmin, self.xmax,
                                      self.ymin, self.ymax,
                                      self.zmin, self.zmax,
                                      self._gdata, self._gnorm,
                                      wx, wy, wz, flags)
