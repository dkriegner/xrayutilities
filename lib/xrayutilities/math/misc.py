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
# Copyright (c) 2016-2025 Dominik Kriegner <dominik.kriegner@gmail.com>

import math

import numpy

from .. import config


def center_of_mass(pos, data, background="none", full_output=False):
    """
    function to determine the center of mass of an array

    Parameters
    ----------
    pos :  array-like
        position of the data points
    data :  array-like
        data values
    background : {'none', 'constant', 'linear'}
        type of background, either 'none', 'constant' or 'linear'
    full_output : bool
        return background cleaned data and background-parameters

    Returns
    -------
    float
        center of mass position
    """
    # subtract background
    slope = 0
    back = 0
    if background == "linear":
        dx = float(pos[-1] - pos[0])
        if abs(dx) > 0:
            slope = (float(data[-1]) - float(data[0])) / dx
        back = (data[0] - slope * pos[0] + data[-1] - slope * pos[-1]) / 2.0
        ld = data - (slope * pos + back)
    elif background == "constant":
        back = numpy.median(data)
        ld = data - numpy.min(data)
    else:
        ld = data

    ipos = numpy.sum(pos * ld) / numpy.sum(ld)
    if full_output:
        return ipos, ld, back, slope
    return ipos


def fwhm_exp(pos, data):
    """
    function to determine the full width at half maximum value of experimental
    data. Please check the obtained value visually (noise influences the
    result)

    Parameters
    ----------
    pos :   array-like
        position of the data points
    data :  array-like
        data values

    Returns
    -------
    float
        fwhm value
    """

    m = data.max()
    p0 = numpy.argmax(data)
    datal = data[: p0 + 1]
    datar = data[p0:]

    # determine left side half value position
    try:
        pls = pos[: p0 + 1][datal < m / 2.0][-1]
        pll = pos[: p0 + 1][datal > m / 2.0][0]
        ds = data[pos == pls][0]
        dl = data[pos == pll][0]
        pl = pls + (pll - pls) * (m / 2.0 - ds) / (dl - ds)
    except IndexError:
        if config.VERBOSITY >= config.INFO_LOW:
            print(
                "XU.math.fwhm_exp: warning: left side half value could"
                " not be determined -> returns 2*hwhm"
            )
        pl = None

    # determine right side half value position
    try:
        prs = pos[p0:][datar < m / 2.0][0]
        prl = pos[p0:][datar > m / 2.0][-1]
        ds = data[pos == prs][0]
        dl = data[pos == prl][0]
        pr = prs + (prl - prs) * (m / 2.0 - ds) / (dl - ds)
    except IndexError:
        if config.VERBOSITY >= config.INFO_LOW:
            print(
                "XU.math.fwhm_exp: warning: right side half value could"
                " not be determined -> returns 2*hwhm"
            )
        pr = None

    if pl is None:
        return numpy.abs(pr - p0) * 2
    if pr is None:
        return numpy.abs(pl - p0) * 2
    return numpy.abs(pr - pl)


def gcd(lst):
    """
    greatest common divisor function using library functions

    Parameters
    ----------
    lst:    array-like
        array of integer values for which the greatest common divisor should be
        determined

    Returns
    -------
    gcd:    int
    """
    if numpy.version.version >= "1.15.0":
        return numpy.gcd.reduce(lst)
    gcdfunc = numpy.frompyfunc(math.gcd, 2, 1)
    return numpy.ufunc.reduce(gcdfunc, lst)


def derivative(func, x0, dx=1.0):
    """
    First derivative of a function using a 3-point central difference formula.

    This code is a heavily simplified version of the deprecated
    scipy.misc.derviative.

    Parameters
    ----------
    func : function
        The function to differentiate.
    x0 : float
        The point at which to evaluate the derivative.
    dx : float, optional
        The step size for finite differences. Default is 1.0.

    Returns
    -------
    float
        The first derivative of the function at x0.

    Notes
    -----
    Uses a 3-point central difference formula for simplicity.

    Examples
    --------
    >>> def f(x):
    ...     return x**2
    >>> derivative(f, 3.0)
    6.0
    """
    # 3-point central difference weights
    weights = [-1, 0, 1]
    points = [x0 - dx, x0, x0 + dx]

    # Compute the weighted sum
    val = sum(w * func(p) for w, p in zip(weights, points))
    return val / (2 * dx)
