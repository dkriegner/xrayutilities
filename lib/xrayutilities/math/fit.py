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
module with a function wrapper to scipy.optimize.leastsq
for fitting of a 2D function to a peak or a 1D Gauss fit with
the odr package
"""

import time
import warnings

import numpy
import scipy.optimize as optimize
from scipy import odr

from .. import config, utilities
from ..exception import InputError
from .functions import (Gauss1d, Gauss1d_der_p, Gauss1d_der_x, Lorentz1d,
                        Lorentz1d_der_p, Lorentz1d_der_x, PseudoVoigt1d,
                        PseudoVoigt1d_der_p, PseudoVoigt1d_der_x,
                        PseudoVoigt1dasym, PseudoVoigt1dasym2)
from .misc import center_of_mass, fwhm_exp


def linregress(x, y):
    """
    fast linregress to avoid usage of scipy.stats which is slow!
    NaN values in y are ignored by this function.

    Parameters
    ----------
    x, y :  array-like
        data coordinates and values

    Returns
    -------
    p :     tuple
        parameters of the linear fit (slope, offset)
    rsq:    float
        R^2 value

    Examples
    --------
    >>> (k, d), R2 = linregress([1, 2, 3], [3.3, 4.1, 5.05])
    """
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    mask = numpy.logical_not(numpy.isnan(y))
    lx, ly = (x[mask], y[mask])
    if numpy.all(numpy.isclose(lx-lx[0], numpy.zeros_like(lx))):
        return (0, numpy.mean(ly)), 0
    p = numpy.polyfit(lx, ly, 1)

    # calculation of r-squared
    f = numpy.polyval(p, lx)
    fbar = numpy.sum(ly) / len(ly)
    ssreg = numpy.sum((f-fbar)**2)
    sstot = numpy.sum((ly - fbar)**2)
    rsq = ssreg / sstot

    return p, rsq


def peak_fit(xdata, ydata, iparams=None, peaktype='Gauss', maxit=300,
             background='constant', plot=False, func_out=False, debug=False):
    """
    fit function using odr-pack wrapper in scipy similar to
    https://github.com/tiagopereira/python_tips/wiki/Scipy%3A-curve-fitting
    for Gauss, Lorentz or Pseudovoigt-functions

    Parameters
    ----------
    xdata :     array_like
        x-coordinates of the data to be fitted
    ydata :     array_like
        y-coordinates of the data which should be fit

    iparams :   list, optional
        initial paramters, determined automatically if not specified
    peaktype :  {'Gauss', 'Lorentz', 'PseudoVoigt',
                 'PseudoVoigtAsym', 'PseudoVoigtAsym2'}, optional
        type of peak to fit
    maxit :     int, optional
        maximal iteration number of the fit
    background : {'constant', 'linear'}, optional
        type of background function
    plot :      bool or str, optional
        flag to ask for a plot to visually judge the fit. If plot is a string
        it will be used as figure name, which makes reusing the figures easier.
    func_out :  bool, optional
        returns the fitted function, which takes the independent variables as
        only argument (f(x))

    Returns
    -------
    params :    list
        the parameters as defined in function `Gauss1d/Lorentz1d/PseudoVoigt1d/
        PseudoVoigt1dasym`. In the case of linear background one more parameter
        is included!
    sd_params : list
        For every parameter the corresponding errors are returned.
    itlim :     bool
        flag to tell if the iteration limit was reached, should be False
    fitfunc :   function, optional
        the function used in the fit can be returned (see func_out).
    """
    if plot:
        plot, plt = utilities.import_matplotlib_pyplot('XU.math.peak_fit')

    gfunc, gfunc_dx, gfunc_dp = _getfit_func(peaktype, background)

    # determine initial parameters
    _check_iparams(iparams, peaktype, background)
    if iparams is None:
        iparams = _guess_iparams(xdata, ydata, peaktype, background)
    if config.VERBOSITY >= config.DEBUG:
        print(f"XU.math.peak_fit: iparams: {str(tuple(iparams))}")

    # set up odr fitting
    peak = odr.Model(gfunc, fjacd=gfunc_dx, fjacb=gfunc_dp)

    sy = numpy.sqrt(ydata)
    sy[sy == 0] = 1
    mydata = odr.RealData(xdata, ydata, sy=sy)

    myodr = odr.ODR(mydata, peak, beta0=iparams, maxit=maxit)
    myodr.set_job(fit_type=2)  # use least-square fit

    fit = myodr.run()
    if config.VERBOSITY >= config.DEBUG:
        print('XU.math.peak_fit:')
        fit.pprint()  # prints final message from odrpack

    fparam = fit.beta
    etaidx = []
    if peaktype in ('PseudoVoigt', 'PseudoVoigtAsym'):
        if background == 'linear':
            etaidx = [-2, ]
        else:
            etaidx = [-1, ]
    elif peaktype == 'PseudoVoigtAsym2':
        etaidx = [5, 6]
    for e in etaidx:
        fparam[e] = 0 if fparam[e] < 0 else fparam[e]
        fparam[e] = 1 if fparam[e] > 1 else fparam[e]

    itlim = False
    if fit.stopreason[0] == 'Iteration limit reached':
        itlim = True
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.math.peak_fit: Iteration limit reached, "
                  "do not trust the result!")

    if plot:
        if isinstance(plot, str):
            plt.figure(plot)
        else:
            plt.figure('XU:peak_fit')
        plt.plot(xdata, ydata, 'ok', label='data', mew=2)
        if debug:
            plt.plot(xdata, gfunc(iparams, xdata), '-', color='0.5',
                     label='estimate')
        plt.plot(xdata, gfunc(fparam, xdata), '-r',
                 label=f'{peaktype}-fit')
        plt.legend()

    if func_out:
        return fparam, fit.sd_beta, itlim, lambda x: gfunc(fparam, x)
    return fparam, fit.sd_beta, itlim


def _getfit_func(peaktype, background):
    """
    internal function to prepare the model functions and derivatives for the
    peak_fit function.

    Parameters
    ----------
    peaktype :  {'Gauss', 'Lorentz', 'PseudoVoigt',
                 'PseudoVoigtAsym', 'PseudoVoigtAsym2'}
        type of peak function
    background : {'constant', 'linear'}
        type of background function

    Returns
    -------
    f, f_dx, f_dp : functions
        fit function, function of derivative regarding `x`, and functions of
        derivatives regarding the parameters
    """
    fdx = None
    fdp = None
    if peaktype == 'Gauss':
        f = Gauss1d
        fdx = Gauss1d_der_x
        fdp = Gauss1d_der_p
    elif peaktype == 'Lorentz':
        f = Lorentz1d
        fdx = Lorentz1d_der_x
        fdp = Lorentz1d_der_p
    elif peaktype == 'PseudoVoigt':
        f = PseudoVoigt1d
        fdx = PseudoVoigt1d_der_x
        fdp = PseudoVoigt1d_der_p
    elif peaktype == 'PseudoVoigtAsym':
        f = PseudoVoigt1dasym
    elif peaktype == 'PseudoVoigtAsym2':
        f = PseudoVoigt1dasym2
    else:
        raise InputError("keyword argument peaktype takes invalid value!")

    if background == 'linear':
        def gfunc(param, x):
            return f(x, *param) + x * param[-1]
    else:
        def gfunc(param, x):
            return f(x, *param)

    if peaktype in ('Gauss', 'Lorentz', 'PseudoVoigt'):
        if background == 'linear':
            def gfunc_dx(param, x):
                return fdx(x, *param) + param[-1]

            def gfunc_dp(param, x):
                return numpy.vstack((fdp(x, *param), x))
        else:
            def gfunc_dx(param, x):
                return fdx(x, *param)

            def gfunc_dp(param, x):
                return fdp(x, *param)
    else:
        gfunc_dx = None
        gfunc_dp = None

    return gfunc, gfunc_dx, gfunc_dp


def _check_iparams(iparams, peaktype, background):
    """
    internal function to check if the length of the supplied initial
    parameters is correct given the other settings of the peak_fit function.
    An InputError is raised in case of wrong shape or value.

    Parameters
    ----------
    iparams :   list
        initial paramters for the fit
    peaktype :  {'Gauss', 'Lorentz', 'PseudoVoigt',
                 'PseudoVoigtAsym', 'PseudoVoigtAsym2'}
        type of peak to fit
    background : {'constant', 'linear'}
        type of background

    """
    if iparams is None:
        return
    ptypes = {('Gauss', 'constant'): 4, ('Lorentz', 'constant'): 4,
              ('Gauss', 'linear'): 5, ('Lorentz', 'linear'): 5,
              ('PseudoVoigt', 'constant'): 5, ('PseudoVoigt', 'linear'): 6,
              ('PseudoVoigtAsym', 'constant'): 6,
              ('PseudoVoigtAsym', 'linear'): 7,
              ('PseudoVoigtAsym2', 'constant'): 7,
              ('PseudoVoigtAsym2', 'linear'): 8}
    if not all(numpy.isreal(iparams)):
        raise InputError("XU.math.peak_fit: all initial parameters need to"
                         "be real!")
    if (peaktype, background) in ptypes:
        nparams = ptypes[(peaktype, background)]
        if len(iparams) != nparams:
            raise InputError(f"XU.math.peak_fit: {nparams} initial parameters "
                             f"are needed for {peaktype}-peak with "
                             f"{background} background.")
    else:
        raise InputError(f"XU.math.peak_fit: invalid peak ({peaktype}) or "
                         f"background ({background})")


def _guess_iparams(xdata, ydata, peaktype, background):
    """
    internal function to automatically esitmate peak parameters from the data,
    considering also the background type.

    Parameters
    ----------
    xdata :     array-like
        x-coordinates of the data to be fitted
    ydata :     array-like
        y-coordinates of the data which should be fit
    peaktype :  {'Gauss', 'Lorentz', 'PseudoVoigt',
                 'PseudoVoigtAsym', 'PseudoVoigtAsym2'}
        type of peak to fit
    background : {'constant', 'linear'}
        type of background, either

    Returns
    -------
     list of initial parameters estimated from the data
    """
    ld = numpy.empty(len(ydata))
    # estimate peak position
    ipos, ld, back, slope = center_of_mass(xdata, ydata, background,
                                           full_output=True)
    maxpos = xdata[numpy.argmax(ld)]
    avx = numpy.average(xdata)
    if numpy.abs(ipos - avx) < numpy.abs(maxpos-avx):
        ipos = maxpos  # use the estimate which is further from the center

    # estimate peak width
    sigma1 = numpy.sqrt(numpy.sum(numpy.abs((xdata - ipos) ** 2 * ld)) /
                        numpy.abs(numpy.sum(ld)))
    sigma2 = fwhm_exp(xdata, ld)/(2 * numpy.sqrt(2 * numpy.log(2)))
    sigma = sigma1 if sigma1 < sigma2 else sigma2

    # build initial parameters
    iparams = [ipos, sigma, numpy.max(ld), back]
    if peaktype in ['Lorentz', 'PseudoVoigt']:
        iparams[1] *= 2 * numpy.sqrt(2 * numpy.log(2))
    if peaktype in ['PseudoVoigtAsym', 'PseudoVoigtAsym2']:
        iparams.insert(1, iparams[1])
    if peaktype in ['PseudoVoigt', 'PseudoVoigtAsym']:
        # set ETA parameter to be between Gauss and Lorentz shape
        iparams.append(0.5)
    if peaktype == 'PseudoVoigtAsym2':
        iparams.append(0.5)
        iparams.append(0.5)
    if background == 'linear':
        iparams.append(slope)
    return iparams


def gauss_fit(xdata, ydata, iparams=None, maxit=300):
    """
    Gauss fit function using odr-pack wrapper in scipy similar to
    https://github.com/tiagopereira/python_tips/wiki/Scipy%3A-curve-fitting

    Parameters
    ----------
    xdata :     array-like
        x-coordinates of the data to be fitted
    ydata :     array-like
        y-coordinates of the data which should be fit
    iparams:    list, optional
        initial paramters for the fit, determined automatically if not given
    maxit :     int, optional
        maximal iteration number of the fit

    Returns
    -------
    params :    list
        the parameters as defined in function ``Gauss1d(x, *param)``
    sd_params : list
        For every parameter the corresponding errors are returned.
    itlim :     bool
        flag to tell if the iteration limit was reached, should be False
    """
    return peak_fit(xdata, ydata, iparams=iparams,
                    peaktype='Gauss', maxit=maxit)


def fit_peak2d(x, y, data, start, drange, fit_function, maxfev=2000):
    """
    fit a two dimensional function to a two dimensional data set e.g. a
    reciprocal space map.

    Parameters
    ----------
    x : array-like
        first data coordinate (does not need to be regularly spaced)
    y : array-like
        second data coordinate (does not need to be regularly spaced)
    data : array-like
        data set used for fitting (e.g. intensity at the data coordinates)
    start : list
        set of starting parameters for the fit used as first parameter of
        function fit_function
    drange : list
        limits for the data ranges used in the fitting algorithm, e.g. it is
        clever to use only a small region around the peak which should be
        fitted, i.e. [xmin, xmax, ymin, ymax]
    fit_function : callable
        function which should be fitted. Call signature must be
        :func:`fit_function(x, y, *params) -> ndarray`

    Returns
    -------
    fitparam : list
        fitted parameters
    cov : array-like
        covariance matrix
    """
    s = time.time()
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.math.fit: Fitting started... ", end='')

    start = numpy.array(start)
    lx = x.flatten()
    ly = y.flatten()
    mask = (lx > drange[0]) * (lx < drange[1]) * \
        (ly > drange[2]) * (ly < drange[3])
    ly = ly[mask]
    lx = lx[mask]
    ldata = data.flatten()[mask]

    def errfunc(p, x, z, data):
        return fit_function(x, z, *p) - data

    p, cov, _, errmsg, success = optimize.leastsq(
        errfunc, start, args=(lx, ly, ldata), full_output=1, maxfev=maxfev)

    s = time.time() - s
    if config.VERBOSITY >= config.INFO_ALL:
        print("finished in %8.2f sec, (data length used %d)" % (s, ldata.size))
        print(f"XU.math.fit: {errmsg}")

    # calculate correct variance covariance matrix
    if cov is not None:
        s_sq = (errfunc(p, lx, ly, ldata) ** 2).sum() / \
            (len(ldata) - len(start))
        pcov = cov * s_sq
    else:
        pcov = numpy.zeros((len(start), len(start)))

    if success not in [1, 2, 3, 4]:
        print("XU.math.fit: Could not obtain fit!")
    return p, pcov


def multPeakFit(x, data, peakpos, peakwidth, dranges=None,
                peaktype='Gaussian', returnerror=False):
    """
    function to fit multiple Gaussian/Lorentzian peaks with linear background
    to a set of data

    Parameters
    ----------
    x :     array-like
        x-coordinate of the data
    data :  array-like
        data array with same length as `x`
    peakpos : list
        initial parameters for the peak positions
    peakwidth : list
        initial values for the peak width
    dranges : list of tuples
        list of tuples with (min, max) value of the data ranges to use.  does
        not need to have the same number of entries as peakpos
    peaktype : {'Gaussian', 'Lorentzian'}
        type of peaks to be used
    returnerror : bool
        decides if the fit errors of pos, sigma, and amp are returned (default:
        False)

    Returns
    -------
    pos :       list
        peak positions derived by the fit
    sigma :     list
        peak width derived by the fit
    amp :       list
        amplitudes of the peaks derived by the fit
    background :    array-like
        background values at positions `x`
    if returnerror == True:
     sd_pos :   list
        standard error of peak positions as returned by scipy.odr.Output
     sd_sigma : list
        standard error of the peak width
     sd_amp :   list
        standard error of the peak amplitude
    """
    warnings.warn("deprecated function -> use the lmfit Python packge instead",
                  DeprecationWarning)
    if peaktype == 'Gaussian':
        pfunc = Gauss1d
        pfunc_derx = Gauss1d_der_x
    elif peaktype == 'Lorentzian':
        pfunc = Lorentz1d
        pfunc_derx = Lorentz1d_der_x
    else:
        raise ValueError('wrong value for parameter peaktype was given')

    def deriv_x(p, x):
        """
        function to calculate the derivative of the signal of multiple peaks
        and background w.r.t. the x-coordinate

        Parameters
        ----------
        p :     list
            parameters, for every peak there needs to be position, sigma,
            amplitude and at the end two values for the linear background
            function (b0, b1)
        x :     array-like
            x-coordinate
        """
        derx = numpy.zeros(x.size)

        # sum up peak functions contributions
        for i in range(len(p) // 3):
            ldx = pfunc_derx(x, p[3 * i], p[3 * i + 1], p[3 * i + 2], 0)
            derx += ldx

        # background contribution
        k = p[-2]
        b = numpy.ones(x.size) * k

        return derx + b

    def deriv_p(p, x):
        """
        function to calculate the derivative of the signal of multiple peaks
        and background w.r.t. the parameters

        Parameters
        ----------
        p :     list
            parameters, for every peak there needs to be position, sigma,
            amplitude and at the end two values for the linear background
            function (b0, b1)
        x :     array-like
            x-coordinate

        returns derivative w.r.t. all the parameters with shape (len(p),x.size)
        """

        derp = numpy.empty(0)
        # peak functions contributions
        for i in range(len(p) // 3):
            lp = (p[3 * i], p[3 * i + 1], p[3 * i + 2], 0)
            if peaktype == 'Gaussian':
                derp = numpy.append(derp, -2 * (lp[0] - x) * pfunc(x, *lp))
                derp = numpy.append(
                    derp, (lp[0] - x) ** 2 / (2 * lp[1] ** 3) * pfunc(x, *lp))
                derp = numpy.append(derp, pfunc(x, *lp) / lp[2])
            else:  # Lorentzian
                derp = numpy.append(derp, 4 * (x - lp[0]) * lp[2] / lp[1] /
                                    (1 + (2 * (x - lp[0]) / lp[1]) ** 2) ** 2)
                derp = numpy.append(derp, 4 * (lp[0] - x) * lp[2] /
                                    lp[1] ** 2 / (1 + (2 * (x - lp[0]) /
                                                       lp[1]) ** 2) ** 2)
                derp = numpy.append(derp,
                                    1 / (1 + (2 * (x - p[0]) / p[1]) ** 2))

        # background contributions
        derp = numpy.append(derp, x)
        derp = numpy.append(derp, numpy.ones(x.size))

        # reshape output
        derp.shape = (len(p),) + x.shape
        return derp

    def fsignal(p, x):
        """
        function to calculate the signal of multiple peaks and background

        Parameters
        ----------
        p :     list
            list of parameters, for every peak there needs to be position,
            sigma, amplitude and at the end two values for the linear
            background function (k, d)
        x :     array-like
            x-coordinate
        """
        f = numpy.zeros(x.size)

        # sum up peak functions
        for i in range(len(p) // 3):
            lf = pfunc(x, p[3 * i], p[3 * i + 1], p[3 * i + 2], 0)
            f += lf

        # background
        k = p[-2]
        d = p[-1]
        b = numpy.polyval((k, d), x)

        return f + b

    ##########################
    # create local data set (extract data ranges)
    if dranges:
        mask = numpy.array([False] * x.size)
        for i in range(len(dranges)):
            lrange = dranges[i]
            lmask = numpy.logical_and(x > lrange[0], x < lrange[1])
            mask = numpy.logical_or(mask, lmask)
        lx = x[mask]
        ldata = data[mask]
    else:
        lx = x
        ldata = data

    # create initial parameter list
    p = []

    # background
    # exclude +/-2 peakwidth around the peaks
    bmask = numpy.ones_like(lx, dtype=bool)
    for pp, pw in zip(peakpos, peakwidth):
        bmask = numpy.logical_and(bmask, numpy.logical_or(lx < (pp-2*pw),
                                                          lx > (pp+2*pw)))
    if numpy.any(bmask):
        k, d = numpy.polyfit(lx[bmask], ldata[bmask], 1)
    else:
        if config.VERBOSITY >= config.DEBUG:
            print("XU.math.multPeakFit: no data outside peak regions!")
        k, d = (0, ldata.min())

    # peak parameters
    for i in range(len(peakpos)):
        amp = ldata[(lx - peakpos[i]) >= 0][0] - \
            numpy.polyval((k, d), lx)[(lx - peakpos[i]) >= 0][0]
        p += [peakpos[i], peakwidth[i], amp]

    # background parameters
    p += [k, d]

    if config.VERBOSITY >= config.DEBUG:
        print("XU.math.multPeakFit: intial parameters")
        print(p)

    ##########################
    # fit with odrpack
    model = odr.Model(fsignal, fjacd=deriv_x, fjacb=deriv_p)
    odata = odr.RealData(lx, ldata)
    my_odr = odr.ODR(odata, model, beta0=p)
    # fit type 2 for least squares
    my_odr.set_job(fit_type=2)
    fit = my_odr.run()

    if config.VERBOSITY >= config.DEBUG:
        print("XU.math.multPeakFit: fitted parameters")
        print(fit.beta)
    try:
        if fit.stopreason[0] not in ['Sum of squares convergence']:
            print("XU.math.multPeakFit: fit NOT converged "
                  f"({fit.stopreason[0]})")
            return None, None, None, None
    except IndexError:
        print("XU.math.multPeakFit: fit most probably NOT converged (%s)"
              % str(fit.stopreason))
        return None, None, None, None
    # prepare return values
    fpos = fit.beta[:-2:3]
    fwidth = numpy.abs(fit.beta[1:-2:3])
    famp = fit.beta[2::3]
    background = numpy.polyval((fit.beta[-2], fit.beta[-1]), x)
    if returnerror:
        sd_pos = fit.sd_beta[:-2:3]
        sd_width = fit.sd_beta[1:-2:3]
        sd_amp = fit.sd_beta[2::3]
        return fpos, fwidth, famp, background, sd_pos, sd_width, sd_amp
    return fpos, fwidth, famp, background


def multPeakPlot(x, fpos, fwidth, famp, background, dranges=None,
                 peaktype='Gaussian', fig="xu_plot", ax=None, fact=1.):
    """
    function to plot multiple Gaussian/Lorentz peaks with background values
    given by an array

    Parameters
    ----------
    x :         array-like
        x-coordinate of the data
    fpos :      list
        positions of the peaks
    fwidth :    list
        width of the peaks
    famp :      list
        amplitudes of the peaks
    background :  array-like
        background values, same shape as `x`
    dranges :   list of tuples
        list of (min, max) values of the data ranges to use.  does not need to
        have the same number of entries as fpos
    peaktype : {'Gaussian', 'Lorentzian'}
        type of peaks to be used
    fig :       int, str, or None
        matplotlib figure number or name
    ax :        matplotlib.Axes
        matplotlib axes as alternative to the figure name
    fact :      float
        factor to use as multiplicator in the plot
    """
    warnings.warn("deprecated function -> use the lmfit Python packge instead",
                  DeprecationWarning)
    success, plt = utilities.import_matplotlib_pyplot('XU.math.multPeakPlot')
    if not success:
        return

    if fig:
        plt.figure(fig)
    if ax:
        plt.sca(ax)
    # plot single peaks
    if dranges:
        mask = numpy.array([False] * x.size)
        for i in range(len(dranges)):
            lrange = dranges[i]
            lmask = numpy.logical_and(x > lrange[0], x < lrange[1])
            mask = numpy.logical_or(mask, lmask)
        lx = x[mask]
        lb = background[mask]
    else:
        lx = x
        lb = background

    f = numpy.zeros(lx.size)
    for i in range(len(fpos)):
        if peaktype == 'Gaussian':
            lf = Gauss1d(lx, fpos[i], fwidth[i], famp[i], 0)
        elif peaktype == 'Lorentzian':
            lf = Lorentz1d(lx, fpos[i], fwidth[i], famp[i], 0)
        else:
            raise ValueError('wrong value for parameter peaktype was given')
        f += lf
        plt.plot(lx, (lf + lb) * fact, ':k')

    # plot summed signal
    plt.plot(lx, (f + lb) * fact, '-r', lw=1.5)
