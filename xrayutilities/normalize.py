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
# Copyright (C) 2010-2011 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to provide functions that perform block averaging
of intensity arrays to reduce the amount of data (mainly
for PSD and CCD measurements

and

provide functions for normalizing intensities for

* count time
* absorber (user-defined function)
* monitor
* flatfield correction
"""

import numpy

from . import cxrayutilities
from . import math
from .exception import InputError
from . import config

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str


def blockAverage1D(data, Nav):
    """
    perform block average for 1D array/list of Scalar values
    all data are used. at the end of the array a smaller cell
    may be used by the averaging algorithm

    Parameter
    ---------
     data:   data which should be contracted (length N)
     Nav:    number of values which should be averaged

    Returns
    -------
    block averaged numpy array of data type numpy.double
    (length ceil(N/Nav))
    """

    if not isinstance(data, (numpy.ndarray, list)):
        raise TypeError("first argument data must be of type list or "
                        "numpy.ndarray")

    data = numpy.array(data, dtype=numpy.double)
    block_av = cxrayutilities.block_average1d(data, Nav)

    return block_av


def blockAverage2D(data2d, Nav1, Nav2, **kwargs):
    """
    perform a block average for 2D array of Scalar values
    all data are used therefore the margin cells may differ in size

    Parameter
    ---------
     data2d:    array of 2D data shape (N,M)
     Nav1,2:    a field of (Nav1 x Nav2) values is contracted

    **kwargs:   optional keyword argument
      roi:      region of interest for the 2D array. e.g. [20,980,40,960]
                N = 980-20; M = 960-40

    Returns
    -------
    block averaged numpy array with type numpy.double with shape
    ( ceil(N/Nav1), ceil(M/Nav2) )
    """

    if not isinstance(data2d, (numpy.ndarray)):
        raise TypeError("first argument data2d must be of type numpy.ndarray")

    # kwargs
    if 'roi' in kwargs:
        if kwargs['roi']:
            roi = kwargs['roi']
        else:
            roi = [0, data2d.shape[0], 0, data2d.shape[1]]
    else:
        roi = [0, data2d.shape[0], 0, data2d.shape[1]]

    data = numpy.array(data2d[roi[0]:roi[1], roi[2]:roi[3]],
                       dtype=numpy.double)

    if config.VERBOSITY >= config.DEBUG:
        print("xu.normalize.blockAverage2D: roi: %s" % (str(roi)))
        print("xu.normalize.blockAverage2D: Nav1,2: %d,%d" % (Nav1, Nav2))
        print("xu.normalize.blockAverage2D: number of points: (%d,%d)"
              % (numpy.ceil(N / float(Nav1)), numpy.ceil(M / float(Nav2))))

    block_av = cxrayutilities.block_average2d(data, Nav1, Nav2,
                                              config.NTHREADS)

    return block_av


def blockAveragePSD(psddata, Nav, **kwargs):
    """
    perform a block average for serveral PSD spectra
    all data are used therefore the last cell used for
    averaging may differ in size

    Parameter
    ---------
     psddata:   array of 2D data shape (Nspectra,Nchannels)
     Nav:       number of channels which should be averaged

    **kwargs:   optional keyword argument
      roi:      region of interest for the 2D array. e.g. [20,980]
                Nchannels = 980-20

    Returns
    -------
    block averaged psd spectra as numpy array with type numpy.double
    of shape ( Nspectra , ceil(Nchannels/Nav) )
    """

    if not isinstance(psddata, (numpy.ndarray)):
        raise TypeError("first argument psddata must be of type numpy.ndarray")

    # kwargs
    if 'roi' in kwargs:
        roi = kwargs['roi']
    else:
        roi = [0, psddata.shape[1]]

    data = numpy.array(psddata[:, roi[0]:roi[1]], dtype=numpy.double)

    block_av = cxrayutilities.block_average_PSD(data, Nav, config.NTHREADS)

    return block_av

# #####################################
# #    Intensity correction class    ##
# #####################################


class IntensityNormalizer(object):

    """
    generic class for correction of intensity (point detector,or MCA,
    single CCD frames) for count time and absorber factors
    the class must be supplied with a absorber correction function
    and works with data structures provided by xrayutilities.io classes or the
    corresponding objects from hdf5 files read by pytables
    """

    def __init__(self, det, **keyargs):
        """
        initialization of the corrector class

        Parameter
        ---------
         det : detector field name of the data structure

        **keyargs:
          mon : monitor field name
          time: count time field name or count time as float
          av_mon: average monitor value (default: data[mon].mean())
          smoothmon: number of monitor values used to get a smooth monitor
                     signal
          absfun: absorber correction function to be used as in
                  absorber_corrected_intensity = data[det]*absfun(data)
          flatfield: flatfield of the detector; shape must be the same as
                     data[det], and is only applied for MCA detectors
          darkfield: darkfield of the detector; shape must be the same as
                     data[det], and is only applied for MCA detectors

        Example
        -------
        >>> detcorr = IntensityNormalizer(det="MCA", time="Seconds",
                absfun=lambda d: d["PSDCORR"]/d["PSD"].astype(numpy.float))
        """

        for k in keyargs.keys():
            if k not in ['mon', 'time', 'smoothmon', 'av_mon', 'absfun',
                         'flatfield', 'darkfield']:
                raise Exception("unknown keyword argument given: allowed are "
                                "'mon', 'smoothmon', 'av_mon', 'absfun'"
                                "'flatfield' and 'darkfield'")

        # check input arguments
        self._setdet(det)

        if 'mon' in keyargs:
            self._setmon(keyargs['mon'])
        else:
            self._mon = None

        if 'time' in keyargs:
            self._settime(keyargs['time'])
        else:
            self._time = None

        if 'av_mon' in keyargs:
            self._setavmon(keyargs['av_mon'])
        else:
            self._avmon = None

        if 'absfun' in keyargs:
            self._setabsfun(keyargs['absfun'])
        else:
            self._absfun = None

        if 'flatfield' in keyargs:
            self._setflatfield(keyargs['flatfield'])
        else:
            self._flatfield = None

        if 'darkfield' in keyargs:
            self._setdarkfield(keyargs['darkfield'])
        else:
            self._darkfield = None

        if 'smoothmon' in keyargs:
            self.smoothmon = keyargs['smoothmon']
        else:
            self.smoothmon = 1

    def _getdet(self):
        """
        det property handler

        returns the detector field name
        """
        return self._det

    def _setdet(self, det):
        """
        det  property handler

        sets the detector field name
        """
        if isinstance(det, basestring):
            self._det = det
        else:
            self._det = None
            raise TypeError("argument det must be of type str")

    def _gettime(self):
        """
        time property handler

        returns the count time or the field name of the count time
        or None if time is not set
        """
        return self._time

    def _settime(self, time):
        """
        time property handler

        sets the count time field or value
        """
        if isinstance(time, basestring):
            self._time = time
        elif isinstance(time, (float, int)):
            self._time = float(time)
        elif isinstance(time, type(None)):
            self._time = None
        else:
            raise TypeError("argument time must be of type str, float or None")

    def _getmon(self):
        """
        mon property handler

        returns the monitor field name or None if not set
        """
        return self._mon

    def _setmon(self, mon):
        """
        mon property handler

        sets the monitor field name
        """
        if isinstance(mon, basestring):
            self._mon = mon
        elif isinstance(mon, type(None)):
            self._mon = None
        else:
            raise TypeError("argument mon must be of type str")

    def _getavmon(self):
        """
        av_mon property handler

        returns the value of the average monitor or None
        if average is calculated from the monitor field
        """
        return self._avmon

    def _setavmon(self, avmon):
        """
        avmon property handler

        sets the average monitor field name
        """
        if isinstance(avmon, (float, int)):
            self._avmon = float(avmon)
        elif isinstance(avmon, type(None)):
            self._avmon = None
        else:
            raise TypeError("argument avmon must be of type float or None")

    def _getabsfun(self):
        """
        absfun property handler

        returns the costum correction function or None
        """
        return self._absfun

    def _setabsfun(self, absfun):
        """
        absfun property handler

        sets the costum correction function
        """
        if hasattr(absfun, '__call__'):
            self._absfun = absfun
            if self._absfun.__code__.co_argcount != 1:
                raise TypeError("argument absfun must be a function with one "
                                "argument (data object)")
        elif isinstance(absfun, type(None)):
            self._absfun = None
        else:
            raise TypeError("argument absfun must be of type function or None")

    def _getflatfield(self):
        """
        flatfield property handler

        returns the current set flatfield of the detector
        or None if not set
        """
        return self._flatfield

    def _setflatfield(self, flatf):
        """
        flatfield property handler

        sets the flatfield of the detector
        """
        if isinstance(flatf, (list, tuple, numpy.ndarray)):
            self._flatfield = numpy.array(flatf, dtype=numpy.float)
            self._flatfieldav = numpy.mean(self._flatfield[
                                           self._flatfield.nonzero()])
            self._flatfield[self.flatfield < 1.e-5] = 1.0
        elif isinstance(flatf, type(None)):
            self._flatfield = None
        else:
            raise TypeError("argument flatfield must be of type list, tuple, "
                            "numpy.ndarray or None")

    def _getdarkfield(self):
        """
        flatfield property handler

        returns the current set darkfield of the detector
        or None if not set
        """
        return self._darkfield

    def _setdarkfield(self, darkf):
        """
        flatfield property handler

        sets the darkfield of the detector
        """
        if isinstance(darkf, (list, tuple, numpy.ndarray)):
            self._darkfield = numpy.array(darkf, dtype=numpy.float)
            self._darkfieldav = numpy.mean(self._darkfield)
        elif isinstance(darkf, type(None)):
            self._darkfield = None
        else:
            raise TypeError("argument flatfield must be of type list, tuple, "
                            "numpy.ndarray or None")

    det = property(_getdet, _setdet)
    time = property(_gettime, _settime)
    mon = property(_getmon, _setmon)
    avmon = property(_getavmon, _setavmon)
    absfun = property(_getabsfun, _setabsfun)
    flatfield = property(_getflatfield, _setflatfield)
    darkfield = property(_getdarkfield, _setdarkfield)

    def __call__(self, data):
        """
        apply the correction method which was initialized to the measured data

        Parameter
        ---------
         data: data object from xrayutilities.io classes (numpy.recarray)

        Returns
        -------
         corrint: corrected intensity as numpy.ndarray of the same shape as
                  data[det]
        """
        corrint = numpy.zeros(data[self._det].shape, dtype=numpy.float)

        # set needed variables
        # monitor intensity
        if self._mon:
            if self.smoothmon == 1:
                mon = data[self._mon]
            else:
                mon = math.smooth(data[self._mon], self.smoothmon)
        else:
            mon = 1.
        # count time
        if isinstance(self._time, basestring):
            time = data[self._time]
        elif isinstance(self._time, float):
            time = self._time
        else:
            time = 1.
        # average monitor counts
        if self._avmon:
            avmon = self._avmon
        else:
            avmon = numpy.mean(mon)
        # absorber correction function
        if self._absfun:
            abscorr = self._absfun(data)
        else:
            abscorr = 1.

        c = abscorr * avmon / (mon * time)
        # correct the correction factor if it was evaluated to an incorrect
        # value
        if isinstance(c, numpy.ndarray):
            c[numpy.isnan(c)] = 1.0
            c[numpy.isinf(c)] = 1.0
            c[c == 0] = 1.0
        else:
            if numpy.isnan(c) or numpy.isinf(c) or c == 0:
                c = 1.0

        if len(data[self._det].shape) == 1:
            corrint = data[self._det] * c
        elif len(data[self._det].shape) == 2 and isinstance(c, numpy.ndarray):
            # 1D detector c.shape[0] should be data[self._det].shape[0]
            if self._darkfield is not None:
                if self._darkfield.shape[0] != data[self._det].shape[1]:
                    raise InputError("data[det] second dimension must have "
                                     "the same length as darkfield")

                if isinstance(time, numpy.ndarray):
                    # darkfield correction
                    corrint = (data[self._det] -
                               self._darkfield[numpy.newaxis, :] *
                               time[:, numpy.newaxis])
                elif isinstance(time, float):
                    # darkfield correction
                    corrint = (data[self._det] -
                               self._darkfield[numpy.newaxis, :] * time)
                else:
                    print("XU.normalize.IntensityNormalizer: check "
                          "initialization and your input")
                    return None
                corrint[corrint < 0.] = 0.

            else:
                corrint = data[self._det]

            corrint = corrint * c[:, numpy.newaxis]

            if self._flatfield is not None:
                if self._flatfield.shape[0] != data[self._det].shape[1]:
                    raise InputError("data[det] second dimension must have "
                                     "the same length as flatfield")
                # flatfield correction
                corrint = (corrint / self._flatfield[numpy.newaxis, :] *
                           self._flatfieldav)

        elif len(data[self._det].shape) == 2 and isinstance(c, numpy.float):
            # single 2D detector frame
            corrint = data[self._det] * c

        else:
            raise InputError("data[det] must be an array of dimension one "
                             "or two")

        return corrint
