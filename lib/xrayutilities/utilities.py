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
# Copyright (C) 2010-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
xrayutilities utilities contains a conglomeration of useful functions
which do not fit into one of the other files
"""

import fractions
import math

import numpy

from . import config
from .utilities_noconf import *


def import_matplotlib_pyplot(funcname='XU'):
    """
    lazy import function of matplotlib.pyplot

    Parameters
    ----------
    funcname :      str
        identification string of the calling function

    Returns
    -------
    flag :  bool
        the flag is True if the loading was successful and False otherwise.
    pyplot
        On success pyplot is the matplotlib.pyplot package.
    """
    try:
        from matplotlib import pyplot as plt
        from .mpl_helper import SqrtAllowNegScale
        return True, plt
    except ImportError:
        if config.VERBOSITY >= config.INFO_LOW:
            print("%s: Warning: plot functionality not available" % funcname)
        return False, None


def import_lmfit(funcname='XU'):
    """
    lazy import function for lmfit
    """
    try:
        import lmfit
        return lmfit
    except ImportError:
        raise ImportError("%s: Fitting of models needs the lmfit package "
                          "(https://pypi.python.org/pypi/lmfit)" % funcname)


def import_mayavi_mlab(funcname='XU'):
    """
    lazy import function of mayavi.mlab

    Parameters
    ----------
    funcname :      str
        identification string of the calling function

    Returns
    -------
    flag :  bool
        the flag is True if the loading was successful and False otherwise.
    mlab
        On success mlab is the mayavi.mlab package.
    """
    try:
        from mayavi import mlab
        return True, mlab
    except ImportError:
        if config.VERBOSITY >= config.INFO_LOW:
            print("%s: Warning: plot functionality not available" % funcname)
        return False, None


def maplog(inte, dynlow="config", dynhigh="config"):
    """
    clips values smaller and larger as the given bounds and returns the log10
    of the input array. The bounds are given as exponent with base 10 with
    respect to the maximum in the input array.  The function is implemented in
    analogy to J. Stangl's matlab implementation.

    Parameters
    ----------
    inte :      ndarray
        numpy.array, values to be cut in range
    dynlow :    float, optional
        10^(-dynlow) will be the minimum cut off
    dynhigh :   float, optional
        10^(-dynhigh) will be the maximum cut off

    Returns
    -------
    ndarray
        numpy.array of the same shape as inte, where values smaller/larger than
        10^(-dynlow, dynhigh) were replaced by 10^(-dynlow, dynhigh)

    Examples
    --------
    >>> lint = maplog(int, 5, 2)
    """
    if dynlow == "config":
        dynlow = config.DYNLOW
    if dynhigh == "config":
        dynhigh = config.DYNHIGH

    mask = numpy.logical_not(numpy.isnan(inte))
    if inte[mask].max() <= 0.0:
        raise ValueError("XU.maplog: only negativ or zero values given. "
                         "Log is not defined!")
    ma = inte[mask].max() * 10 ** (-1*dynhigh)  # upper bound
    mi = inte[mask].max() * 10 ** (-1*dynlow)  # lower bound

    return numpy.log10(numpy.minimum(numpy.maximum(inte, mi), ma))


def frac2str(f, denominator_limit=25, fmt='%7.4f'):
    """
    convert a float to a string attempting to represent it as a fraction

    Parameters
    ----------
    f :    float
        floating point number to be represented as string
    denominator_limit : int
        maximal integer used as denominator. If f can't be expressed
        (within xu.config.EPSILON) by a fraction with a denominator
        up to this number a floating point string will be returned
    fmt :  str
        format string used in case a floating point representation is needed

    Returns
    -------
    str
    """
    frac = fractions.Fraction(f).limit_denominator(denominator_limit)
    if math.isclose(float(frac), f, abs_tol=config.EPSILON):
        return str(frac)
    else:
        return fmt % f
