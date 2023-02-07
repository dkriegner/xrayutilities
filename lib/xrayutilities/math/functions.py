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
# Copyright (c) 2012-2021, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module with several common function needed in xray data analysis
"""

import copy
import numbers

import numpy
import scipy.integrate

from .. import config


def smooth(x, n):
    """
    function to smooth an array of data by averaging N adjacent data points

    Parameters
    ----------
    x : array-like
        1D data array
    n : int
        number of data points to average

    Returns
    -------
    xsmooth: array-like
        smoothed array with same length as x
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


def kill_spike(data, threshold=2., offset=None):
    """
    function to smooth **single** data points which differ from the average of
    the neighboring data points by more than the threshold factor or more than
    the offset value. Such spikes will be replaced by the mean value of the
    next neighbors.

    .. warning:: Use this function carefully not to manipulate your data!

    Parameters
    ----------
    data :          array-like
        1d numpy array with experimental data
    threshold :     float or None
        threshold factor to identify outlier data points. If None it will be
        ignored.
    offset :        None or float
        offset value to identify outlier data points. If None it will be
        ignored.

    Returns
    -------
    array-like
        1d data-array with spikes removed
    """

    dataout = data.copy()

    mean = (data[:-2] + data[2:]) / 2.
    mask = numpy.zeros_like(data[1:-1], dtype=bool)
    if threshold:
        mask = numpy.logical_or(
            mask, numpy.logical_or(data[1:-1] * threshold < mean,
                                   data[1:-1] / threshold > mean))
    if offset:
        mask = numpy.logical_or(
            mask, numpy.logical_or(data[1:-1] + offset < mean,
                                   data[1:-1] - offset > mean))
    # ensure that only single value are corrected and neighboring are ignored
    for i in range(1, len(mask) - 1):
        if mask[i - 1] and mask[i] and mask[i + 1]:
            mask[i - 1] = False
            mask[i + 1] = False

    dataout[1:-1][mask] = mean[mask]

    return dataout


def Gauss1d(x, *p):
    """
    function to calculate a general one dimensional Gaussian

    Parameters
    ----------
    x :     array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gaussian [XCEN, SIGMA, AMP, BACKGROUND]
        for information: SIGMA = FWHM / (2*sqrt(2*log(2)))

    Returns
    -------
    array-like
        the value of the Gaussian described by the parameters p at position x

    Examples
    --------
    Calling with a list of parameters needs a call looking as shown below
    (note the '*') or explicit listing of the parameters

    >>> Gauss1d(x, *p)  # doctest: +SKIP

    >>> import numpy
    >>> Gauss1d(numpy.linspace(0, 10, 10), 5, 1, 1e3, 0)
    array([3.72665317e-03, 5.19975743e-01, 2.11096565e+01, 2.49352209e+02,
           8.56996891e+02, 8.56996891e+02, 2.49352209e+02, 2.11096565e+01,
           5.19975743e-01, 3.72665317e-03])
    """
    g = p[3] + p[2] * numpy.exp(-((p[0] - x) / p[1]) ** 2 / 2.)
    return g


def NormGauss1d(x, *p):
    """
    function to calculate a normalized one dimensional Gaussian

    Parameters
    ----------
    x :     array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gaussian [XCEN, SIGMA];
        for information: SIGMA = FWHM / (2*sqrt(2*log(2)))

    Returns
    -------
    array-like
        the value of the normalized Gaussian described by the parameters p at
        position x
    """
    g = numpy.exp(-((p[0] - x) / p[1]) ** 2 / 2.)
    a = numpy.sqrt(2 * numpy.pi) * p[1]
    return g / a


def Gauss1d_der_x(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect to x

    for parameter description see Gauss1d
    """

    lp = numpy.copy(p)
    lp[3] = 0
    return (p[0] - x) / p[1]**2 * Gauss1d(x, *lp)


def Gauss1d_der_p(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect the
    parameters p

    for parameter description see Gauss1d
    """
    lp = numpy.copy(p)
    lp[3] = 0
    r = numpy.vstack((- (p[0] - x) / p[1]**2 * Gauss1d(x, *lp),
                      (p[0] - x) ** 2 / (p[1] ** 3) * Gauss1d(x, *lp),
                      Gauss1d(x, *lp) / p[2],
                      numpy.ones(x.shape, dtype=float)))

    return r


def Gauss2d(x, y, *p):
    """
    function to calculate a general two dimensional Gaussian

    Parameters
    ----------
    x, y :  array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gauss-function
        [XCEN, YCEN, SIGMAX, SIGMAY, AMP, BACKGROUND, ANGLE];
        SIGMA = FWHM / (2*sqrt(2*log(2)));
        ANGLE = rotation of the X, Y direction of the Gaussian in radians

    Returns
    -------
    array-like
        the value of the Gaussian described by the parameters p at
        position (x, y)
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
    x, y, z : array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gauss-function
        [XCEN, YCEN, ZCEN, SIGMAX, SIGMAY, SIGMAZ, AMP, BACKGROUND];

        SIGMA = FWHM / (2*sqrt(2*log(2)))

    Returns
    -------
    array-like
        the value of the Gaussian described by the parameters p at
        positions (x, y, z)
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
    x, y :  array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gauss-function
        [XCEN1, YCEN1, SIGMAX1, SIGMAY1, AMP1, ANGLE1, XCEN2, YCEN2, SIGMAX2,
        SIGMAY2, AMP2, ANGLE2, BACKGROUND];
        SIGMA = FWHM / (2*sqrt(2*log(2)))
        ANGLE = rotation of the X, Y direction of the Gaussian in radians

    Returns
    -------
    array-like
        the value of the Gaussian described by the parameters p
        at position (x, y)
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
    x :     array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Lorentz-function
        [XCEN, FWHM, AMP, BACKGROUND]

    Returns
    -------
    array-like
        the value of the Lorentian described by the parameters p
        at position (x, y)

    """
    g = p[3] + p[2] / (1 + (2 * (x - p[0]) / p[1]) ** 2)
    return g


def Lorentz1d_der_x(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect to x

    for parameter description see Lorentz1d
    """
    return 8 * (p[0]-x) / p[1]**2 * p[2] / (1 + (2 * (x-p[0]) / p[1]) ** 2)**2


def Lorentz1d_der_p(x, *p):
    """
    function to calculate the derivative of a Gaussian with respect the
    parameters p

    for parameter description see Lorentz1d
    """
    r = numpy.vstack((
        8 * (x-p[0]) * p[2] / p[1]**2 / (1 + (2 * (x-p[0]) / p[1]) ** 2) ** 2,
        8 * p[2] * p[1] * (x-p[0])**2 /
        (4*p[0]**2 - 8*p[0]*x + p[1]**2 + 4*x**2) ** 2,
        1 / (1 + (2 * (x - p[0]) / p[1]) ** 2),
        numpy.ones(x.shape, dtype=float)))
    return r


def NormLorentz1d(x, *p):
    """
    function to calculate a normalized one dimensional Lorentzian

    Parameters
    ----------
    x :     array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Lorentzian [XCEN, FWHM]

    Returns
    -------
    array-like
        the value of the normalized Lorentzian described by the parameters p
        at position x
    """
    g = 1.0 / (1 + (2 * (x - p[0]) / p[1]) ** 2)
    a = numpy.pi / (2. / (p[1]))
    return g / a


def Lorentz2d(x, y, *p):
    """
    function to calculate a general two dimensional Lorentzian

    Parameters
    ----------
    x, y :   array-like
        coordinate(s) where the function should be evaluated
    p :      list
        list of parameters of the Lorentz-function
        [XCEN, YCEN, FWHMX, FWHMY, AMP, BACKGROUND, ANGLE];
        ANGLE = rotation of the X, Y direction of the Lorentzian in radians

    Returns
    -------
    array-like
        the value of the Lorentian described by the parameters p
        at position (x, y)
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
    x :     array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the pseudo Voigt-function
        [XCEN, FWHM, AMP, BACKGROUND, ETA];
        ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    array-like
        the value of the PseudoVoigt described by the parameters p
        at position `x`
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


def PseudoVoigt1d_der_x(x, *p):
    """
    function to calculate the derivative of a PseudoVoigt with respect to `x`

    for parameter description see PseudoVoigt1d
    """
    if p[4] > 1.0:
        pv = 1.0
    elif p[4] < 0.:
        pv = 0.0
    else:
        pv = p[4]

    sigma = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))

    rl = Lorentz1d_der_x(x, p[0], p[1], p[2], 0)
    rg = Gauss1d_der_x(x, p[0], sigma, p[2], 0)

    return pv * rl + (1 - pv) * rg


def PseudoVoigt1d_der_p(x, *p):
    """
    function to calculate the derivative of a PseudoVoigt with respect the
    parameters `p`

    for parameter description see PseudoVoigt1d
    """

    if p[4] > 1.0:
        pv = 1.0
    elif p[4] < 0.:
        pv = 0.0
    else:
        pv = p[4]

    sigma = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))

    lpl = [p[0], p[1], p[2], 0]
    lpg = [p[0], sigma, p[2], 0]
    rl = Lorentz1d_der_p(x, *lpl)
    rg = Gauss1d_der_p(x, *lpg)
    rg[1] /= (2 * numpy.sqrt(2 * numpy.log(2)))
    r = pv * rl + (1 - pv) * rg

    return numpy.vstack((r, Lorentz1d(x, *lpl) - Gauss1d(x, *lpg)))


def PseudoVoigt1dasym(x, *p):
    """
    function to calculate an asymmetric pseudo Voigt function as linear
    combination of asymmetric Gauss and Lorentz peak

    Parameters
    ----------
    x :     array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the pseudo Voigt-function
        [XCEN, FWHMLEFT, FWHMRIGHT, AMP, BACKGROUND, ETA];
        ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    array-like
        the value of the PseudoVoigt described by the parameters p
        at position `x`
    """
    lp = copy.copy(list(p))
    lp.insert(6, p[5])
    return PseudoVoigt1dasym2(x, *lp)


def PseudoVoigt1dasym2(x, *p):
    """
    function to calculate an asymmetric pseudo Voigt function as linear
    combination of asymmetric Gauss and Lorentz peak

    Parameters
    ----------
    x :     naddray
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the pseudo Voigt-function
        [XCEN, FWHMLEFT, FWHMRIGHT, AMP, BACKGROUND, ETALEFT, ETARIGHT];
        ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    array-like
        the value of the PseudoVoigt described by the parameters p
        at position `x`
    """
    pvl = p[5] if p[5] < 1.0 else 1.0
    pvl = pvl if p[5] > 0.0 else 0.0
    pvr = p[6] if p[6] < 1.0 else 1.0
    pvr = pvr if p[6] > 0.0 else 0.0

    sigmal = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))
    sigmar = p[2] / (2 * numpy.sqrt(2 * numpy.log(2)))

    if isinstance(x, numbers.Number):
        if x < p[0]:
            f = p[4] + pvl * Lorentz1d(x, p[0], p[1], p[3], 0) + \
                (1 - pvl) * Gauss1d(x, p[0], sigmal, p[3], 0)
        else:
            f = p[4] + pvr * Lorentz1d(x, p[0], p[2], p[3], 0) + \
                (1 - pvr) * Gauss1d(x, p[0], sigmar, p[3], 0)
    else:
        lx = numpy.asarray(x)
        f = numpy.zeros(lx.shape)
        f[lx < p[0]] = (p[4] + pvl *
                        Lorentz1d(lx[x < p[0]], p[0], p[1], p[3], 0) +
                        (1 - pvl) *
                        Gauss1d(lx[x < p[0]], p[0], sigmal, p[3], 0))
        f[lx >= p[0]] = (p[4] + pvr *
                         Lorentz1d(lx[x >= p[0]], p[0], p[2], p[3], 0) +
                         (1 - pvr) *
                         Gauss1d(lx[x >= p[0]], p[0], sigmar, p[3], 0))

    return f


def PseudoVoigt2d(x, y, *p):
    """
    function to calculate a pseudo Voigt function as linear combination of a
    Gauss and Lorentz peak in two dimensions

    Parameters
    ----------
    x, y :   array-like
        coordinate(s) where the function should be evaluated
    p :      list
        list of parameters of the pseudo Voigt-function
        [XCEN, YCEN, FWHMX, FWHMY, AMP, BACKGROUND, ANGLE, ETA];
        ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    array-like
        the value of the PseudoVoigt described by the parameters `p`
        at position `(x, y)`
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
    p :     list
        list of parameters of the Gauss-function [XCEN, SIGMA, AMP, BACKGROUND]

    Returns
    -------
    float
        the area of the Gaussian described by the parameters `p`
    """
    f = p[2] * numpy.sqrt(2 * numpy.pi) * p[1]
    return f


def Gauss2dArea(*p):
    """
    function to calculate the area of a 2D Gauss function with neglected
    background

    Parameters
    ----------
    p :     list
        list of parameters of the Gauss-function
        [XCEN, YCEN, SIGMAX, SIGMAY, AMP, ANGLE, BACKGROUND]

    Returns
    -------
    float
        the area of the Gaussian described by the parameters `p`
    """
    f = p[4] * numpy.sqrt(2 * numpy.pi) ** 2 * p[2] * p[3]
    return f


def Lorentz1dArea(*p):
    """
    function to calculate the area of a Lorentz function with neglected
    background

    Parameters
    ----------
    p :     list
        list of parameters of the Lorentz-function
        [XCEN, FWHM, AMP, BACKGROUND]

    Returns
    -------
    float
        the area of the Lorentzian described by the parameters `p`
    """
    f = p[2] * numpy.pi / (2. / (p[1]))
    return f


def PseudoVoigt1dArea(*p):
    """
    function to calculate the area of a pseudo Voigt function with neglected
    background

    Parameters
    ----------
    p :     list
        list of parameters of the Lorentz-function
        [XCEN, FWHM, AMP, BACKGROUND, ETA];
        ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz

    Returns
    -------
    float
        the area of the PseudoVoigt described by the parameters `p`
    """
    sigma = p[1] / (2 * numpy.sqrt(2 * numpy.log(2)))
    f = p[4] * Lorentz1dArea(*p) + (1. - p[4]) * \
        Gauss1dArea(p[0], sigma, p[2], p[3])
    return f


def Debye1(x):
    r"""
    function to calculate the first Debye function [1]_ as needed
    for the calculation of the thermal Debye-Waller-factor
    by numerical integration

    .. math:: D_1(x) = (1/x) \int_0^x t/(\exp(t)-1) dt

    Parameters
    ----------
    x : float
        argument of the Debye function

    Returns
    -------
    float
        D1(x)  float value of the Debye function

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Debye_function
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

    if config.VERBOSITY >= config.DEBUG:
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
    x :     array-like
        coordinate where the function should be evaluated
    args :  list
        list of peak/function types and parameters for every function type two
        arguments need to be given first the type of function as string with
        possible values 'g': Gaussian, 'l': Lorentzian, 'v': PseudoVoigt, 'a':
        asym. PseudoVoigt, 'p': polynom the second type of arguments is the
        tuple/list of parameters of the respective function. See documentation
        of math.Gauss1d, math.Lorentz1d, math.PseudoVoigt1d,
        math.PseudoVoigt1dasym, and numpy.polyval for details of the different
        function types.

    Returns
    -------
    array-like
        value of the sum of functions at position `x`
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
    x, y :  array-like
        coordinates where the function should be evaluated
    args :  list
        list of peak/function types and parameters for every function type two
        arguments need to be given first the type of function as string with
        possible values 'g': Gaussian, 'l': Lorentzian, 'v': PseudoVoigt, 'c':
        constant the second type of arguments is the tuple/list of parameters
        of the respective function. See documentation of math.Gauss2d,
        math.Lorentz2d, math.PseudoVoigt2d for details of the different
        function types.  The constant accepts a single float which will be
        added to the data

    Returns
    -------
    array-like
        value of the sum of functions at position `(x, y)`
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


def heaviside(x):
    """
    Heaviside step function for numpy arrays

    Parameters
    ----------
    x: scalar or array-like
        argument of the step function

    Returns
    -------
    int or array-like
        Heaviside step function evaluated for all values of `x` with datatype
        integer
    """
    return (numpy.sign(x)/2. + 0.5).astype(numpy.int8)
