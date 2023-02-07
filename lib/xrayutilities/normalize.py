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
# Copyright (c) 2010-2021, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

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

from . import config, cxrayutilities, math, utilities
from .exception import InputError


def blockAverage1D(data, Nav):
    """
    perform block average for 1D array/list of Scalar values
    all data are used. at the end of the array a smaller cell
    may be used by the averaging algorithm

    Parameters
    ----------
    data :  array-like
        data which should be contracted (length N)
    Nav :   int
        number of values which should be averaged

    Returns
    -------
    ndarray
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

    Parameters
    ----------
    data2d :        ndarray
        array of 2D data shape (N, M)
    Nav1, Nav2 :    int
        a field of (Nav1 x Nav2) values is contracted

    kwargs :        dict, optional
        optional keyword argument
    roi :           tuple or list, optional
        region of interest for the 2D array. e.g. [20, 980, 40, 960],
        reduces M, and M!

    Returns
    -------
    ndarray
        block averaged numpy array with type numpy.double with shape
        (ceil(N/Nav1), ceil(M/Nav2))
    """

    if not isinstance(data2d, (numpy.ndarray)):
        raise TypeError("first argument data2d must be of type numpy.ndarray")

    roi = kwargs.get('roi', [0, data2d.shape[0], 0, data2d.shape[1]])
    data = numpy.array(data2d[roi[0]:roi[1], roi[2]:roi[3]],
                       dtype=numpy.double)

    if config.VERBOSITY >= config.DEBUG:
        N, M = (roi[1] - roi[0], roi[3] - roi[2])
        print(f"xu.normalize.blockAverage2D: roi: {str(roi)}")
        print("xu.normalize.blockAverage2D: Nav1, 2: %d,%d" % (Nav1, Nav2))
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

    Parameters
    ----------
    psddata :   ndarray
        array of 2D data shape (Nspectra, Nchannels)
    Nav :       int
        number of channels which should be averaged

    kwargs :    dict, optional
        optional keyword argument
    roi :       tuple or list
        region of interest for the 2D array. e.g. [20, 980] Nchannels = 980-20

    Returns
    -------
    ndarray
        block averaged psd spectra as numpy array with type numpy.double of
        shape (Nspectra , ceil(Nchannels/Nav))
    """

    if not isinstance(psddata, (numpy.ndarray)):
        raise TypeError("first argument psddata must be of type numpy.ndarray")

    roi = kwargs.get('roi', [0, psddata.shape[1]])
    data = numpy.array(psddata[:, roi[0]:roi[1]], dtype=numpy.double)

    block_av = cxrayutilities.block_average_PSD(data, Nav, config.NTHREADS)

    return block_av


def blockAverageCCD(data3d, Nav1, Nav2, **kwargs):
    """
    perform a block average for 2D frames inside a 3D array.
    all data are used therefore the margin cells may differ in size

    Parameters
    ----------
    data3d :        ndarray
        array of 3D data shape (Nframes, N, M)
    Nav1, Nav2 :    int
        a field of (Nav1 x Nav2) values is contracted

    kwargs :        dict, optional
        optional keyword argument
    roi :           tuple or list, optional
        region of interest for the 2D array. e.g. [20, 980, 40, 960],
        reduces M, and M!

    Returns
    -------
    ndarray
        block averaged numpy array with type numpy.double with shape
        (Nframes, ceil(N/Nav1), ceil(M/Nav2))
    """

    if not isinstance(data3d, (numpy.ndarray)):
        raise TypeError("first argument data3d must be of type numpy.ndarray")

    roi = kwargs.get('roi', [0, data3d.shape[1], 0, data3d.shape[2]])
    data = numpy.array(data3d[:, roi[0]:roi[1], roi[2]:roi[3]],
                       dtype=numpy.double)

    if config.VERBOSITY >= config.DEBUG:
        N, M = (roi[1] - roi[0], roi[3] - roi[2])
        print(f"xu.normalize.blockAverageCCD: roi: {str(roi)}")
        print("xu.normalize.blockAverageCCD: Nav1, 2: %d,%d" % (Nav1, Nav2))
        print("xu.normalize.blockAverageCCD: number of points: (%d,%d)"
              % (numpy.ceil(N / float(Nav1)), numpy.ceil(M / float(Nav2))))

    block_av = cxrayutilities.block_average_CCD(data, Nav1, Nav2,
                                                config.NTHREADS)

    return block_av

# #####################################
# #    Intensity correction class    ##
# #####################################


class IntensityNormalizer:

    """
    generic class for correction of intensity (point detector, or MCA,
    single CCD frames) for count time and absorber factors
    the class must be supplied with a absorber correction function
    and works with data structures provided by xrayutilities.io classes or the
    corresponding objects from hdf5 files
    """

    def __init__(self, det='', **keyargs):
        """
        initialization of the corrector class

        Parameters
        ----------
        det :       str
            detector field name of the data structure

        mon :       str, optional
            monitor field name
        time:       float or str, optional
            count time field name or count time as float
        av_mon :    float, optional
            average monitor value (default: data[mon].mean())
        smoothmon : int
            number of monitor values used to get a smooth monitor signal
        absfun :    callable, optional
            absorber correction function to be used as in
            ``absorber_corrected_intensity = data[det]*absfun(data)``
        flatfield : ndarray
            flatfield of the detector; shape must be the same as data[det], and
            is only applied for MCA detectors
        darkfield : ndarray
            darkfield of the detector; shape must be the same as data[det], and
            is only applied for MCA detectors

        Examples
        --------
        >>> detcorr = IntensityNormalizer("MCA", time="Seconds",
        ... absfun=lambda d: d["PSDCORR"]/d["PSD"].astype(float))
        """
        valid_kwargs = {'mon': 'monitor field name',
                        'time': 'count time field/value',
                        'smoothmon': 'number of monitor values to average',
                        'av_mon': 'average monitor value',
                        'absfun': 'absorber correction function',
                        'flatfield': 'detector flatfield',
                        'darkfield': 'detector darkfield'}
        utilities.check_kwargs(keyargs, valid_kwargs,
                               self.__class__.__name__)

        # check input arguments
        self._setdet(det)

        self._setmon(keyargs.get('mon', None))
        self._settime(keyargs.get('time', None))
        self._setavmon(keyargs.get('av_mon', None))
        self._setabsfun(keyargs.get('absfun', None))
        self._setflatfield(keyargs.get('flatfield', None))
        self._setdarkfield(keyargs.get('darkfield', None))
        self.smoothmon = keyargs.get('smoothmon', 1)

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
        if isinstance(det, str):
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
        if isinstance(time, str):
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
        if isinstance(mon, str):
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
            self._flatfield = numpy.array(flatf, dtype=float)
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
            self._darkfield = numpy.array(darkf, dtype=float)
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

    def __call__(self, data, ccd=None):
        """
        apply the correction method which was initialized to the measured data

        Parameters
        ----------
        data :  numpy.recarray
            data object from xrayutilities.io classes
        ccd :   ndarray, optional
            optionally CCD data can be given as separate ndarray of shape
            (len(data), n1, n2), where n1, n2 is the shape of the CCD image.

        Returns
        -------
        corrint :   ndarray
            corrected intensity as numpy.ndarray of the same shape as data[det]
            (or ccd.shape)
        """
        if numpy.any(ccd):
            rawdata = ccd
        else:
            rawdata = data[self._det]

        corrint = numpy.zeros(rawdata.shape, dtype=float)

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
        if isinstance(self._time, str):
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

        if len(rawdata.shape) == 1:
            corrint = rawdata * c
        elif len(rawdata.shape) == 2 and isinstance(c, numpy.ndarray):
            # 1D detector c.shape[0] should be rawdata.shape[0]
            if self._darkfield is not None:
                if self._darkfield.shape[0] != rawdata.shape[1]:
                    raise InputError("data[det] second dimension must have "
                                     "the same length as darkfield")

                if isinstance(time, numpy.ndarray):
                    # darkfield correction
                    corrint = (rawdata -
                               self._darkfield[numpy.newaxis, :] *
                               time[:, numpy.newaxis])
                elif isinstance(time, float):
                    # darkfield correction
                    corrint = (rawdata -
                               self._darkfield[numpy.newaxis, :] * time)
                else:
                    print("XU.normalize.IntensityNormalizer: check "
                          "initialization and your input")
                    return None
                corrint[corrint < 0.] = 0.

            else:
                corrint = rawdata

            corrint = corrint * c[:, numpy.newaxis]

            if self._flatfield is not None:
                if self._flatfield.shape[0] != rawdata.shape[1]:
                    raise InputError("data[det] second dimension must have "
                                     "the same length as flatfield")
                # flatfield correction
                corrint = (corrint / self._flatfield[numpy.newaxis, :] *
                           self._flatfieldav)

        elif len(rawdata.shape) == 2 and isinstance(c, float):
            # single 2D detector frame
            corrint = rawdata * c

        elif len(rawdata.shape) == 3:
            # darkfield and flatfield correction is still missing!
            corrint = rawdata * c[:, numpy.newaxis, numpy.newaxis]

        else:
            raise InputError("data[det] must be an array of dimension one "
                             "or two or three")

        return corrint
