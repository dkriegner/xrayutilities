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
# Copyright (C) 2012-2014 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module with several common function needed in xray data analysis
"""

import numpy
import scipy.integrate
import numbers

from .. import config


def smooth(x, n):
    """
    function to smooth an array of data by averaging N adjacent data points

    Parameters
    ----------
     x:  1D data array
     n:  number of data points to average

    Returns
    -------
     xsmooth:  smoothed array with same length as x
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < n:
        raise ValueError("Input vector needs to be bigger than n.")
    if n < 2:
        return x
    # avoid boundary effects by adding mirrored signal at the boundaries
    s = numpy.r_[x[n - 1:0:-1], x, x[-1:-n - 1:-1]]
    w = numpy.ones(n, 'd')
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[n:-n + 1]


def kill_spike(data, threshold=2.):
    """
    function to smooth **single** data points which differ from the average of
    the neighboring data points by more than the threshold factor. Such spikes
    will be replaced by the mean value of the next neighbors.

    .. warning:: Use this function carefully not to manipulate your data!

    Parameters
    ----------
     data:          1d numpy array with experimental data
     threshold:     threshold factor to identify strange data points

    Returns
    -------
     1d data-array with spikes removed
    """

    dataout = data.copy()

    mean = (data[:-2] + data[2:]) / 2.
    mask = numpy.logical_or(
        numpy.abs(data[1:-1] * threshold) < numpy.abs(mean),
        numpy.abs(data[1:-1] / threshold) > numpy.abs(mean))
    # ensure that only single value are corrected and neighboring are ignored
    for i in range(1, len(mask) - 1):
        if mask[i - 1] and mask[i] and mask[i + 1]:
            mask[i - 1] = False
            mask[i + 1] = False

    dataout[1:-1][mask] = (numpy.abs(mean))[mask]

    return dataout


def Gauss1d(x, *p):
    """
    function to calculate a general one dimensional Gaussian

    Parameters
    ----------
     p:     list of parameters of the Gaussian
            [XCEN,SIGMA,AMP,BACKGROUND]
            for information: SIGMA = FWHM / (2*sqrt(2*log(2)))
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Gaussian described by the parameters p
    at position x

    Example
    -------
    Calling with a list of parameters needs a call looking as shown below
    (note the '*') or explicit listing of the parameters:
    >>> Gauss1d(x,*p)
    >>> Gauss1d(numpy.linspace(0,10,100), 5, 1, 1e3, 0)
    """
    g = p[3] + p[2] * numpy.exp(-((p[0] - x) / p[1]) ** 2 / 2.)
    return g


def Gauss1d_der_x(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect to x

    for parameter description see Gauss1d
    """

    lp = numpy.copy(p)
    lp[3] = 0
    return 2 * (p[0] - x) * Gauss1d(x, *lp)


def Gauss1d_der_p(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect the
    parameters p

    for parameter description see Gauss1d
    """
    lp = numpy.copy(p)
    lp[3] = 0
    r = numpy.concatenate((-2 * (p[0] - x) * Gauss1d(x, *lp),
                           (p[0] - x) ** 2 / (2 * p[1] ** 3) * Gauss1d(x, *lp),
                           Gauss1d(x, *lp) / p[2],
                           numpy.ones(x.shape, dtype=numpy.float)))
    r.shape = (4,) + x.shape

    return r


def Gauss2d(x, y, *p):
    """
    function to calculate a general two dimensional Gaussian

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,SIGMAX,SIGMAY,AMP,BACKGROUND,ANGLE]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
            ANGLE = rotation of the X,Y direction of the Gaussian in radians
     x,y:   coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Gaussian described by the parameters p
    at position (x,y)
    """

    rcen_x = p[0] * numpy.cos(p[6]) - p[1] * numpy.sin(p[6])
    rcen_y = p[0] * numpy.sin(p[6]) + p[1] * numpy.cos(p[6])
    xp = x * numpy.cos(p[6]) - y * numpy.sin(p[6])
    yp = x * numpy.sin(p[6]) + y * numpy.cos(p[6])

    g = p[5] + p[4] * numpy.exp(-(((rcen_x - xp) / p[2]) ** 2 +
                                  ((rcen_y - yp) / p[3]) ** 2) / 2.)
    return g


def Gauss3d(x, y, z, *p):
    """
    function to calculate a general three dimensional Gaussian

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,ZCEN,SIGMAX,SIGMAY,SIGMAZ,AMP,BACKGROUND]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
     x,y,z:   coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Gaussian described by the parameters p
    at positions (x,y,z)
    """

    g = p[7] + p[6] * numpy.exp(-(((x - p[0]) / p[3]) ** 2 +
                                  ((y - p[1]) / p[4]) ** 2 +
                                  ((z - p[2]) / p[5]) ** 2) / 2.)
    return g


def TwoGauss2d(x, y, *p):
    """
    function to calculate two general two dimensional Gaussians

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN1,YCEN1,SIGMAX1,SIGMAY1,AMP1,ANGLE1,XCEN2,YCEN2,
            SIGMAX2,SIGMAY2,AMP2,ANGLE2,BACKGROUND]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
            ANGLE = rotation of the X,Y direction of the Gaussian in radians
     x,y:   coordinate(s) where the function should be evaluated

    Return
    ------
    the value of the Gaussian described by the parameters p
    at position (x,y)
    """

    p = list(p)
    p1 = p[0:5] + [p[12], ] + [p[5], ]
    p2 = p[6:11] + [p[12], ] + [p[11], ]

    g = Gauss2d(x, y, *p1) + Gauss2d(x, y, *p2)

    return g


def Lorentz1d(x, *p):
    """
    function to calculate a general one dimensional Lorentzian

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND]
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Lorentian described by the parameters p
    at position (x,y)

    """
    g = p[3] + p[2] / (1 + (2 * (x - p[0]) / p[1]) ** 2)
    return g


def Lorentz1d_der_x(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect to x

    for parameter description see Lorentz1d
    """

    return 4 * (p[0] - x) * p[2] / p[1] / \
        (1 + (2 * (x - p[0]) / p[1]) ** 2) ** 2


def Lorentz1d_der_p(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect the
    parameters p

    for parameter description see Lorentz1d
    """
    r = numpy.concatenate((
        4 * (x - p[0]) * p[2] / p[1] / (1 + (2 * (x - p[0]) / p[1]) ** 2) ** 2,
        4 * (p[0] - x) * p[2] / p[1] ** 2 /
        (1 + (2 * (x - p[0]) / p[1]) ** 2) ** 2,
        1 / (1 + (2 * (x - p[0]) / p[1]) ** 2),
        numpy.ones(x.shape, dtype=numpy.float)))
    r.shape = (4,) + x.shape
    return r


def Lorentz2d(x, y, *p):
    """
    function to calculate a general two dimensional Lorentzian

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,YCEN,FWHMX,FWHMY,AMP,BACKGROUND,ANGLE]
            ANGLE = rotation of the X,Y direction of the Lorentzian in radians
     x,y:   coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Lorentian described by the parameters p
    at position (x,y)
    """
    rcen_x = p[0] * numpy.cos(p[6]) - p[1] * numpy.sin(p[6])
    rcen_y = p[0] * numpy.sin(p[6]) + p[1] * numpy.cos(p[6])
    xp = x * numpy.cos(p[6]) - y * numpy.sin(p[6])
    yp = x * numpy.sin(p[6]) + y * numpy.cos(p[6])

    g = p[5] + p[4] / (1 + (2 * (rcen_x - xp) / p[2]) **
                       2 + (2 * (rcen_y - yp) / p[3]) ** 2)
    return g


def PseudoVoigt1d(x, *p):
    """
    function to calculate a pseudo Voigt function as linear combination of a
    Gauss and Lorentz peak

    Parameters
    ----------
     p:     list of parameters of the pseudo Voigt-function
            [XCEN,FWHM,AMP,BACKGROUND,ETA]
            ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the PseudoVoigt described by the parameters p
    at position 'x'
    """
    if p[4] > 1.0:
        pv = 1.0
    elif p[4] < 0.:
        pv = 0.0
    else:
        pv = p[4]

    sigma = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))
    f = p[3] + pv * Lorentz1d(x, p[0], p[1], p[2], 0) + \
        (1 - pv) * Gauss1d(x, p[0], sigma, p[2], 0)
    return f

def PseudoVoigt1dasym(x, *p):
    """
    function to calculate an asymmetric pseudo Voigt function as linear
    combination of asymmetric Gauss and Lorentz peak

    Parameters
    ----------
     p:     list of parameters of the pseudo Voigt-function
            [XCEN,FWHMLEFT,FWHMRIGHT,AMP,BACKGROUND,ETA]
            ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the PseudoVoigt described by the parameters p
    at position 'x'
    """
    if p[5] > 1.0:
        pv = 1.0
    elif p[5] < 0.:
        pv = 0.0
    else:
        pv = p[5]

    sigmal = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))
    sigmar = p[2] / (2 * numpy.sqrt(2 * numpy.log(2)))

    if isinstance(x, numbers.Number):
        if x < p[0]:
            f = p[4] + pv * Lorentz1d(x, p[0], p[1], p[3], 0) + \
                (1 - pv) * Gauss1d(x, p[0], sigmal, p[3], 0)
        else:
            f = p[4] + pv * Lorentz1d(x, p[0], p[2], p[3], 0) + \
                (1 - pv) * Gauss1d(x, p[0], sigmar, p[3], 0)
    else:
        lx = numpy.asarray(x)
        f = numpy.zeros(lx.shape)
        f[lx < p[0]] = (p[4] + pv *
                        Lorentz1d(lx[x < p[0]], p[0], p[1], p[3], 0) + (1 - pv)
                        * Gauss1d(lx[x < p[0]], p[0], sigmal, p[3], 0))
        f[lx >= p[0]] = (p[4] +  pv *
                         Lorentz1d(lx[x >= p[0]], p[0], p[2], p[3], 0) +
                         (1 - pv) *
                         Gauss1d(lx[x >= p[0]], p[0], sigmar, p[3], 0))

    return f

def PseudoVoigt2d(x, y, *p):
    """
    function to calculate a pseudo Voigt function as linear combination of a
    Gauss and Lorentz peak in two dimensions

    Parameters
    ----------
     x,y:   coordinate(s) where the function should be evaluated
     p:     list of parameters of the pseudo Voigt-function
            [XCEN,YCEN,FWHMX,FWHMY,AMP,BACKGROUND,ANGLE,ETA]
            ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    the value of the PseudoVoigt described by the parameters p
    at position (x,y)
    """
    if p[7] > 1.0:
        pv = 1.0
    elif p[7] < 0.:
        pv = 0.0
    else:
        pv = p[7]
    sigmax = p[2] / (2 * numpy.sqrt(2 * numpy.log(2)))
    sigmay = p[3] / (2 * numpy.sqrt(2 * numpy.log(2)))
    f = p[5] + pv * Lorentz2d(x, y, p[0], p[1], p[2], p[3], p[4], 0, p[6]) + \
        (1 - pv) * Gauss2d(x, y, p[0], p[1], sigmax, sigmay, p[4], 0, p[6])
    return f


def Gauss1dArea(*p):
    """
    function to calculate the area of a Gauss function with neglected
    background

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,SIGMA,AMP,BACKGROUND]

    Returns
    -------
    the area of the Gaussian described by the parameters p
    """
    f = p[2] * numpy.sqrt(2 * numpy.pi) * p[1]
    return f


def Gauss2dArea(*p):
    """
    function to calculate the area of a 2D Gauss function with neglected
    background

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE,BACKGROUND]

    Returns
    -------
    the area of the Gaussian described by the parameters p
    """
    f = p[4] * numpy.sqrt(2 * numpy.pi) ** 2 * p[2] * p[3]
    return f


def Lorentz1dArea(*p):
    """
    function to calculate the area of a Lorentz function with neglected
    background

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND]

    Returns
    -------
    the area of the Lorentzian described by the parameters p
    """
    f = p[2] * numpy.pi / (2. / (p[1]))
    return f


def PseudoVoigt1dArea(*p):
    """
    function to calculate the area of a pseudo Voigt function with neglected
    background

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND,ETA]
            ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    the area of the PseudoVoigt described by the parameters p
    """
    sigma = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))
    f = p[4] * Lorentz1dArea(*p) + (1. - p[4]) * \
        Gauss1dArea(p[0], sigma, p[2], p[3])
    return f


def Debye1(x):
    """
    function to calculate the first Debye function as needed
    for the calculation of the thermal Debye-Waller-factor
    by numerical integration

    for definition see:
    http://en.wikipedia.org/wiki/Debye_function

    D1(x) = (1/x) \int_0^x t/(exp(t)-1) dt

    Parameters
    ----------
     x ... argument of the Debye function (float)

    Returns
    -------
     D1(x)  float value of the Debye function
     """

    def __int_kernel(t):
        """
        integration kernel for the numeric integration
        """
        y = t / (numpy.exp(t) - 1)
        return y

    if x > 0.:
        integral = scipy.integrate.quad(__int_kernel, 0, x)
        d1 = (1 / float(x)) * integral[0]
    else:
        integral = (0, 0)
        d1 = 1.

    if (config.VERBOSITY >= config.DEBUG):
        print(
            "XU.math.Debye1: Debye integral value/error estimate: %g %g" %
            (integral[0], integral[1]))

    return d1


def multPeak1d(x, *args):
    """
    function to calculate the sum of multiple peaks in 1D.
    the peaks can be of different type and a background function (polynom)
    can also be included.

    Parameters
    ----------
     x:         coordinate where the function should be evaluated
     args:      list of peak/function types and parameters for every function
                type two arguments need to be given first the type of function
                as string with possible values 'g': Gaussian, 'l': Lorentzian,
                'v': PseudoVoigt, 'a': asym. PseudoVoigt, 'p': polynom the
                second type of arguments is the tuple/list of parameters of the
                respective function. See documentation of math.Gauss1d,
                math.Lorentz1d, math.PseudoVoigt1d, math.PseudoVoigt1dasym, and
                numpy.polyval for details of the different function types.

    Returns
    -------
     value of the sum of functions at position x
    """
    if len(args) % 2 != 0:
        raise ValueError('number of arguments must be even!')

    if numpy.isscalar(x):
        f = 0
    else:
        lx = numpy.array(x)
        f = numpy.zeros(lx.shape)

    for i in range(int(len(args) / 2)):
        ftype = str.lower(args[2 * i])
        fparam = args[2 * i + 1]
        if ftype == 'g':
            f += Gauss1d(x, *fparam)
        elif ftype == 'l':
            f += Lorentz1d(x, *fparam)
        elif ftype == 'v':
            f += PseudoVoigt1d(x, *fparam)
        elif ftype == 'a':
            f += PseudoVoigt1dasym(x, *fparam)
        elif ftype == 'p':
            if isinstance(fparam, (tuple, list, numpy.ndarray)):
                f += numpy.polyval(fparam, x)
            else:
                f += numpy.polyval((fparam,), x)
        else:
            raise ValueError('invalid function type given!')

    return f


def multPeak2d(x, y, *args):
    """
    function to calculate the sum of multiple peaks in 2D.
    the peaks can be of different type and a background function (polynom)
    can also be included.

    Parameters
    ----------
     x,y:       coordinates where the function should be evaluated
     args:      list of peak/function types and parameters for every function
                type two arguments need to be given first the type of function
                as string with possible values 'g': Gaussian, 'l': Lorentzian,
                'v': PseudoVoigt, 'c': constant the second type of arguments is
                the tuple/list of parameters of the respective function. See
                documentation of math.Gauss2d, math.Lorentz2d,
                math.PseudoVoigt2d for details of the different function types.
                The constant accepts a single float which will be added to the
                data

    Returns
    -------
     value of the sum of functions at position (x,y)
    """
    if len(args) % 2 != 0:
        raise ValueError('number of arguments must be even!')

    if numpy.isscalar(x):
        f = 0
    else:
        lx = numpy.array(x)
        ly = numpy.array(y)
        f = numpy.zeros(lx.shape)

    for i in range(int(len(args) / 2)):
        ftype = str.lower(args[2 * i])
        fparam = args[2 * i + 1]
        if ftype == 'g':
            f += Gauss2d(lx, ly, *fparam)
        elif ftype == 'l':
            f += Lorentz2d(lx, ly, *fparam)
        elif ftype == 'v':
            f += PseudoVoigt2d(lx, ly, *fparam)
        elif ftype == 'c':
            f += fparam
        else:
            raise ValueError('invalid function type given!')

    return f
