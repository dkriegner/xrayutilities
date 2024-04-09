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
# Copyright (c) 2009-2021, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import abc
import enum

import numpy

from . import config, cxrayutilities, utilities
from .exception import InputError


class GridderFlags(enum.IntFlag):
    NO_DATA_INIT = 1
    NO_NORMALIZATION = 4
    VERBOSE = 16


def delta(min_value, max_value, n):
    """
    Compute the stepsize along an axis of a grid.

    Parameters
    ----------
   min_value :	 axis minimum value
   max_value :	 axis maximum value
   n :	 number of steps
    """
    if n != 1:
        return (float(max_value) - float(min_value)) / float(n - 1)
    return numpy.inf


def axis(min_value, max_value, n):
    """
    Compute the a grid axis.

    Parameters
    ----------
    min_value : float
        axis minimum value
    max_value :	float
        axis maximum value
    n :	        int
        number of steps
    """

    if n != 1:
        d = delta(min_value, max_value, n)
        a = min_value + d * numpy.arange(0, n)
    else:
        a = (min_value + max_value) / 2.

    return a


def ones(*args):
    """
    Compute ones for matrix generation. The shape is determined by the number
    of input arguments.
    """
    return numpy.ones(args, dtype=numpy.double)


class Gridder(utilities.ABC):
    """
    Basis class for gridders in xrayutilities. A gridder is a function mapping
    irregular spaced data onto a regular grid by binning the data into equally
    sized elements.

    There are different ways of defining the regular grid of a Gridder. In
    xrayutilities the data bins extend beyond the data range in the input data,
    but the given position being the center of these bins, extends from the
    minimum to the maximum of the data!  The main motivation for this was to
    create a Gridder, which when feeded with N equidistant data points and
    gridded with N bins would not change the data position (not the case with
    numpy.histogramm functions!). Of course this leads to the fact that for
    homogeneous point density the first and last bin in any direction are not
    filled as the other bins.

    A different definition is used by numpy histogram functions where the bins
    extend only to the end of the data range. (see numpy histogram,
    histrogram2d, ...)
    """

    def __init__(self):
        """
        Constructor defining default properties of any Gridder class
        """

        # flags represent a way to transmit options to the C-code
        # no data initialization necessary in c-code
        self.flags = GridderFlags.NO_DATA_INIT
        # by default every call to gridder will start a new gridding
        self.keep_data = False
        self.normalize = True
        # flag to allow for sequential gridding with fixed data range
        self.fixed_range = False

        if not hasattr(self, '_gdata'):
            self._gdata = numpy.empty(0)
        if not hasattr(self, '_gnorm'):
            self._gnorm = numpy.empty(0)

        if config.VERBOSITY >= config.INFO_ALL:
            self.flags |= GridderFlags.VERBOSE

    @abc.abstractmethod
    def __call__(self):
        """
        abstract call method which every implementation of a Gridder has to
        override
        """

    def Normalize(self, bool):
        """
        set or unset the normalization flag.  Normalization needs to be done to
        obtain proper gridding but may want to be disabled in certain cases
        when sequential gridding is performed
        """
        if bool not in [False, True]:
            raise TypeError("Normalize flag must be a boolan value "
                            "(True/False)!")
        self.normalize = bool
        if bool:  # Note this is usually done in Python anyways!
            self.flags &= ~GridderFlags.NO_NORMALIZATION
        else:
            self.flags |= GridderFlags.NO_NORMALIZATION

    def KeepData(self, bool):
        if bool not in [False, True]:
            raise TypeError("Keep Data flag must be a boolan value"
                            "(True/False)!")

        self.keep_data = bool

    def __get_data(self):
        """
        return gridded data (performs normalization if switched on)
        """
        if self.normalize:
            tmp = numpy.copy(self._gdata)
            mask = (self._gnorm != 0)
            tmp[mask] /= self._gnorm[mask].astype(numpy.double)
            return tmp
        return self._gdata.copy()

    data = property(__get_data)

    def _prepare_array(self, a):
        """
        prepare array for passing to c-code
        """
        if isinstance(a, (list, tuple, float, int)):
            a = numpy.asarray(a)
        return a.reshape(a.size)

    def Clear(self):
        """
        Clear so far gridded data to reuse this instance of the Gridder
        """
        self._gdata[...] = 0
        self._gnorm[...] = 0


class Gridder1D(Gridder):

    def __init__(self, nx):
        Gridder.__init__(self)
        if nx <= 0:
            raise InputError('nx must be a positiv integer!')

        self.nx = nx
        self.xmin = 0
        self.xmax = 0
        self._gdata = numpy.zeros(nx, dtype=numpy.double)
        self._gnorm = numpy.zeros(nx, dtype=numpy.double)

    def savetxt(self, filename, header=''):
        """
        save gridded data to a txt file with two columns. The first column is
        the data coordinate and the second the corresponding data value

        Parameters
        ----------
        filename :  str
            output filename
        header :    str, optional
            optional header for the data file.
        """
        numpy.savetxt(filename, numpy.vstack((self.xaxis, self.data)).T,
                      header=header, fmt='%.6g %.4g')

    def __get_xaxis(self):
        """
        Returns the xaxis of the gridder
        the returned values correspond to the center of the data bins used by
        the gridding algorithm
        """
        return axis(self.xmin, self.xmax, self.nx)

    xaxis = property(__get_xaxis)

    def dataRange(self, min, max, fixed=True):
        """
        define minimum and maximum data range, usually this is deduced
        from the given data automatically, however, for sequential
        gridding it is useful to set this before the first call of the
        gridder. data outside the range are simply ignored

        Parameters
        ----------
        min :   float
            minimum value of the gridding range
        max :   float
            maximum value of the gridding range
        fixed : bool, optional
            flag to turn fixed range gridding on (True (default)) or off
            (False)
        """
        self.fixed_range = fixed
        self.xmin = min
        self.xmax = max

    def _checktransinput(self, x, data):
        """
        common checks and reshape commands for the input data. This function
        checks the data type and shape of the input data.
        """
        if not self.keep_data:
            self.Clear()

        x = self._prepare_array(x)
        data = self._prepare_array(data)

        if x.size != data.size:
            raise InputError(f"XU.{self.__class__.__name__}: size of given "
                             "datasets (x, data) is not equal!")

        if not self.fixed_range:
            # assume that with setting keep_data the user wants to call the
            # gridder more often and obtain a reasonable result
            self.dataRange(x.min(), x.max(), self.keep_data)

        return x, data

    def __call__(self, x, data):
        """
        Perform gridding on a set of data. After running the gridder
        the 'data' object in the class is holding the gridded data.

        Parameters
        ----------
        x :	 ndarray
            numpy array of arbitrary shape with x positions
        data :	 ndarray
            numpy array of arbitrary shape with data values
        """
        x, data = self._checktransinput(x, data)
        # remove normalize flag for C-code, normalization is always performed
        # in python
        flags = self.flags | GridderFlags.NO_NORMALIZATION
        cxrayutilities.gridder1d(x, data, self.nx, self.xmin, self.xmax,
                                 self._gdata, self._gnorm, flags)


class FuzzyGridder1D(Gridder1D):
    """
    An 1D binning class considering every data point to have a finite width.
    If necessary one data point will be split fractionally over different
    data bins. This is numerically more effort but represents better the
    typical case of a experimental data, which do not represent a mathematical
    point but have a finite width (e.g. X-ray data from a 1D detector).
    """

    def __call__(self, x, data, width=None):
        """
        Perform gridding on a set of data. After running the gridder
        the 'data' object in the class is holding the gridded data.

        Parameters
        ----------
        x :	 ndarray
            numpy array of arbitrary shape with x positions
        data :	 ndarray
            numpy array of arbitrary shape with data values
        width :	 float, optional
            width of one data point. If not given half the bin size will be
            used.
        """
        x, data = self._checktransinput(x, data)

        if not width:
            width = delta(self.xmin, self.xmax, self.nx) / 2.

        # remove normalize flag for C-code, normalization is always performed
        # in python
        flags = self.flags | GridderFlags.NO_NORMALIZATION
        cxrayutilities.fuzzygridder1d(x, data, self.nx, self.xmin, self.xmax,
                                      self._gdata, self._gnorm, width, flags)


class npyGridder1D(Gridder1D):

    def __get_xaxis(self):
        """
        Returns the xaxis of the gridder
        the returned values correspond to the center of the data bins used by
        the numpy.histogram function
        """
        # no -1 here to be consistent with numpy.histogram
        dx = (float(self.xmax - self.xmin)) / float(self.nx)
        ax = self.xmin + dx * numpy.arange(0, self.nx) + dx / 2.
        return ax

    xaxis = property(__get_xaxis)

    def __call__(self, x, data):
        """
        Perform gridding on a set of data. After running the gridder
        the 'data' object in the class is holding the gridded data.

        Parameters
        ----------
        x :	 ndarray
            numpy array of arbitrary shape with x positions
        data :	 ndarray
            numpy array of arbitrary shape with data values
        """

        x, data = self._checktransinput(x, data)

        # use only non-NaN data values
        mask = numpy.invert(numpy.isnan(data))
        ldata = data[mask]
        lx = x[mask]

        if not self.fixed_range:
            # assume that with setting keep_data the user wants to call the
            # gridder more often and obtain a reasonable result
            self.dataRange(lx.min(), lx.max(), self.keep_data)

        # grid the data using numpy histogram
        tmpgdata, _ = numpy.histogram(lx, weights=ldata, bins=self.nx,
                                      range=(self.xmin, self.xmax))
        tmpgnorm, _ = numpy.histogram(lx, bins=self.nx,
                                      range=(self.xmin, self.xmax))
        if self.keep_data:
            self._gnorm += tmpgnorm
            self._gdata += tmpgdata
        else:
            self._gnorm = tmpgnorm
            self._gdata = tmpgdata
