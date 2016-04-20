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
# Copyright (C) 2012-2016 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
module with a function wrapper to scipy.optimize.leastsq
for fitting of a 2D function to a peak or a 1D Gauss fit with
the odr package
"""

from __future__ import print_function
import numpy
import scipy.optimize as optimize
import time
from scipy.odr import odrpack as odr
from scipy.odr import models

from .. import config
from .. exception import InputError
from .misc import fwhm_exp
from .misc import center_of_mass
from .functions import Gauss1d, Gauss1d_der_x, Gauss1d_der_p
from .functions import Lorentz1d, Lorentz1d_der_x, Lorentz1d_der_p
from .functions import PseudoVoigt1d, PseudoVoigt1d_der_x, PseudoVoigt1d_der_p
from .functions import PseudoVoigt1dasym

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str


def linregress(x, y):
    """
    fast linregress to avoid usage of scipy.stats which is slow!

    Parameters
    ----------
     x,y:   data coordinates and values

    Returns
    -------
     p, rsq: parameters of the linear fit (slope, offest) and the R^2 value

    Examples
    --------
     >>> (k, d), R2 = xu.math.linregress(x, y)
    """
    p = numpy.polyfit(x, y, 1)

    # calculation of r-squared
    f = numpy.polyval(p, x)
    fbar = numpy.sum(y) / len(y)
    ssreg = numpy.sum((f-fbar)**2)
    sstot = numpy.sum((y - fbar)**2)
    rsq = ssreg / sstot

    return p, rsq


def peak_fit(xdata, ydata, iparams=[], peaktype='Gauss', maxit=300,
             background='constant', plot=False, func_out=False, debug=False):
    """
    fit function using odr-pack wrapper in scipy similar to
    https://github.com/tiagopereira/python_tips/wiki/Scipy%3A-curve-fitting
    for Gauss, Lorentz or Pseudovoigt-functions

    Parameters
    ----------
     xdata:     xcoordinates of the data to be fitted
     ydata:     ycoordinates of the data which should be fit

    keyword parameters:
     iparams:   initial paramters for the fit,
                determined automatically if not specified
     peaktype:  type of peak to fit: 'Gauss', 'Lorentz', 'PseudoVoigt',
                'PseudoVoigtAsym'
     maxit:     maximal iteration number of the fit
     background:    type of background, either 'constant' or 'linear'
     plot:      flag to ask for a plot to visually judge the fit.
                If plot is a string it will be used as figure name, which
                makes reusing the figures easier.
     func_out:  returns the fitted function, which takes the independent
                variables as only argument (f(x))

    Returns
    -------
     params,sd_params,itlim[,fitfunc]

    the parameters as defined in function Gauss1d/Lorentz1d/PseudoVoigt1d/
    PseudoVoigt1dasym(x, *param). In the case of linear background one more
    parameter is included! For every parameter the corresponding errors of the
    fit 'sd_params' are returned. A boolean flag 'itlim', which is False in
    the case of a successful fit is added by default. Further the function
    used in the fit can be returned (see func_out).
    """
    if plot:
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            if config.VERBOSITY >= config.INFO_ALL:
                print("XU.math.peak_fit: Warning: plot "
                      "functionality not available")
            plot = False

    gfunc, gfunc_dx, gfunc_dp = _getfit_func(peaktype, background)

    # determine initial parameters
    _check_iparams(iparams, peaktype, background)
    if not any(iparams):
        iparams = _guess_iparams(xdata, ydata, peaktype, background)
    if config.VERBOSITY >= config.DEBUG:
        print("XU.math.peak_fit: iparams: %s" % str(tuple(iparams)))

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
    if peaktype in ('PseudoVoigt', 'PseudoVoigtAsym'):
        if background == 'linear':
            etaidx = -2
        else:
            etaidx = -1
        fparam[etaidx] = 0 if fparam[etaidx] < 0 else fparam[etaidx]
        fparam[etaidx] = 1 if fparam[etaidx] > 1 else fparam[etaidx]

    itlim = False
    if fit.stopreason[0] == 'Iteration limit reached':
        itlim = True
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.math.peak_fit: Iteration limit reached, "
                  "do not trust the result!")

    if plot:
        if isinstance(plot, basestring):
            plt.figure(plot)
        else:
            plt.figure('XU:peak_fit')
        plt.plot(xdata, ydata, 'ko', label='data', mew=2)
        if debug:
            plt.plot(xdata, gfunc(iparams, xdata), '-', color='0.5',
                     label='estimate')
        plt.plot(xdata, gfunc(fparam, xdata), 'r-',
                 label='%s-fit' % peaktype)
        plt.legend()

    if func_out:
        return fparam, fit.sd_beta, itlim, lambda x: gfunc(fparam, x)
    else:
        return fparam, fit.sd_beta, itlim


def _getfit_func(peaktype, background):
    """
    internal function to prepare the model functions and derivatives for the
    peak_fit function.

    Parameters
    ----------
     peaktype:  type of peak to fit: 'Gauss', 'Lorentz', 'PseudoVoigt',
                'PseudoVoigtAsym'
     background:    type of background, either 'constant' or 'linear'

    Returns
    -------
     f, f_dx, f_dp: fit function, function of derivative regarding x, and
                    functions of derivatives regarding the parameters
    """
    gfunc_dx = None
    gfunc_dp = None
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
        if background == 'linear':
            def gfunc(param, x):
                return PseudoVoigt1dasym(x, *param) + x * param[-1]
        else:
            def gfunc(param, x):
                return PseudoVoigt1dasym(x, *param)
    else:
        raise InputError("keyword argument peaktype takes invalid value!")

    if peaktype in ('Gauss', 'Lorentz', 'PseudoVoigt'):
        if background == 'linear':
            def gfunc(param, x):
                return f(x, *param) + x * param[-1]

            def gfunc_dx(param, x):
                return fdx(x, *param) + param[-1]

            def gfunc_dp(param, x):
                return numpy.vstack((fdp(x, *param), x))
        else:
            def gfunc(param, x):
                return f(x, *param)

            def gfunc_dx(param, x):
                return fdx(x, *param)

            def gfunc_dp(param, x):
                return fdp(x, *param)
    return gfunc, gfunc_dx, gfunc_dp


def _check_iparams(iparams, peaktype, background):
    """
    internal function to check if the length of the supplied initial
    parameters is correct given the other settings of the peak_fit function.
    An InputError is raised in case of wrong shape or value.

    Parameters
    ----------
     iparams:   initial paramters for the fit
     peaktype:  type of peak to fit: 'Gauss', 'Lorentz', 'PseudoVoigt',
                'PseudoVoigtAsym'
     background:    type of background, either 'constant' or 'linear'

    """
    if not any(iparams):
        return
    else:
        if not all(numpy.isreal(iparams)):
            raise InputError("XU.math.peak_fit: all initial parameters need to"
                             "be real!")
        elif peaktype in ('Gauss', 'Lorentz') and background == 'constant':
            if len(iparams) != 4:
                raise InputError("XU.math.peak_fit: four initial parameters "
                                 "are needed for %s-peak with %s background."
                                 % (peaktype, background))
        elif ((peaktype in ('Gauss', 'Lorentz') and background == 'linear') or
              (peaktype == 'PseudoVoigt' and background == 'constant')):
            if len(iparams) != 5:
                raise InputError("XU.math.peak_fit: five initial parameters "
                                 "are needed for %s-peak with %s background."
                                 % (peaktype, background))
        elif ((peaktype == 'PseudoVoigt' and background == 'linear') or
              (peaktype == 'PseudoVoigtAsym' and background == 'constant')):
            if len(iparams) != 6:
                raise InputError("XU.math.peak_fit: six initial parameters are"
                                 " needed for %s-peak with %s background."
                                 % (peaktype, background))
        elif peaktype == 'PseudoVoigtAsym' and background == 'linear':
            if len(iparams) != 7:
                raise InputError("XU.math.peak_fit: seven initial parameters "
                                 "are needed for %s-peak with %s background."
                                 % (peaktype, background))


def _guess_iparams(xdata, ydata, peaktype, background):
    """
    internal function to automatically esitmate peak parameters from the data,
    considering also the background type.

    Parameters
    ----------
     xdata:     xcoordinates of the data to be fitted
     ydata:     ycoordinates of the data which should be fit
     peaktype:  type of peak to fit: 'Gauss', 'Lorentz', 'PseudoVoigt',
                'PseudoVoigtAsym'
     background:    type of background, either 'constant' or 'linear'

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
    if peaktype == 'PseudoVoigtAsym':
        iparams.insert(1, iparams[1])
    if peaktype in ['PseudoVoigt', 'PseudoVoigtAsym']:
        # set ETA parameter to be between Gauss and Lorentz shape
        iparams.append(0.5)
    if background == 'linear':
        iparams.append(slope)
    return iparams


def gauss_fit(xdata, ydata, iparams=[], maxit=300):
    """
    Gauss fit function using odr-pack wrapper in scipy similar to
    https://github.com/tiagopereira/python_tips/wiki/Scipy%3A-curve-fitting

    Parameters
    ----------
     xdata:     xcoordinates of the data to be fitted
     ydata:     ycoordinates of the data which should be fit

    keyword parameters:
     iparams:   initial paramters for the fit,
                determined automatically if not given
     maxit:     maximal iteration number of the fit

    Returns
    -------
     params,sd_params,itlim

    the Gauss parameters as defined in function Gauss1d(x, *param) and their
    errors of the fit, as well as a boolean flag which is false in the case of
    a successful fit
    """

    return peak_fit(xdata, ydata, iparams=iparams,
                    peaktype='Gauss', maxit=maxit)


def fit_peak2d(x, y, data, start, drange, fit_function, maxfev=2000):
    """
    fit a two dimensional function to a two dimensional data set
    e.g. a reciprocal space map

    Parameters
    ----------
     x,y:     data coordinates (do NOT need to be regularly spaced)
     data:    data set used for fitting (e.g. intensity at the data coords)
     start:   set of starting parameters for the fit
              used as first parameter of function fit_function
     drange:  limits for the data ranges used in the fitting algorithm,
              e.g. it is clever to use only a small region around the peak
              which should be fitted: [xmin,xmax,ymin,ymax]
     fit_function:  function which should be fitted,
                    must accept the parameters (x,y,*params)

    Returns
    -------
     (fitparam,cov)   the set of fitted parameters and covariance matrix
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

    p, cov, infodict, errmsg, success = optimize.leastsq(
        errfunc, start, args=(lx, ly, ldata), full_output=1, maxfev=maxfev)

    s = time.time() - s
    if config.VERBOSITY >= config.INFO_ALL:
        print("finished in %8.2f sec, (data length used %d)" % (s, ldata.size))
        print("XU.math.fit: %s" % errmsg)

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


def multGaussFit(*args, **kwargs):
    """
    convenience function to keep API stable
    see multPeakFit for documentation
    """
    kwargs['peaktype'] = 'Gaussian'
    return multPeakFit(*args, **kwargs)


def multPeakFit(x, data, peakpos, peakwidth, dranges=None,
                peaktype='Gaussian'):
    """
    function to fit multiple Gaussian/Lorentzian peaks with linear background
    to a set of data

    Parameters
    ----------
     x:  x-coordinate of the data
     data:  data array with same length as x
     peakpos:  initial parameters for the peak positions
     peakwidth:  initial values for the peak width
     dranges:  list of tuples with (min,max) value of the data ranges to use.
               does not need to have the same number of entries as peakpos
     peaktype: type of peaks to be used: can be either 'Gaussian' or
               'Lorentzian'

    Returns
    -------
     pos,sigma,amp,background

    pos:  list of peak positions derived by the fit
    sigma:  list of peak width derived by the fit
    amp:  list of amplitudes of the peaks derived by the fit
    background:  array of background values at positions x
    """
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

        p: list of parameters, for every peak there needs to be position,
           sigma, amplitude and at the end two values for the linear background
           function (b0,b1)
        x: x-coordinate
        """
        derx = numpy.zeros(x.size)

        # sum up peak functions contributions
        for i in range(len(p) // 3):
            ldx = pfunc_derx(x, p[3 * i], p[3 * i + 1], p[3 * i + 2], 0)
            derx += ldx

        # background contribution
        k = p[-2]
        d = p[-1]
        b = numpy.ones(x.size) * k

        return derx + b

    def deriv_p(p, x):
        """
        function to calculate the derivative of the signal of multiple peaks
        and background w.r.t. the parameters

        p: list of parameters, for every peak there needs to be position,
           sigma, amplitude and at the end two values for the linear background
           function (b0,b1)
        x: x-coordinate

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

        p: list of parameters, for every peak there needs to be position,
           sigma, amplitude and at the end two values for the linear background
           function (k,d)
        x: x-coordinate
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
    k, d = numpy.polyfit(lx, ldata, 1)

    # peak parameters
    for i in range(len(peakpos)):
        amp = ldata[(lx - peakpos[i]) >= 0][0] - \
            numpy.polyval((k, d), lx)[(lx - peakpos[i]) >= 0][0]
        p += [peakpos[i], peakwidth[i], amp]

    # background parameters
    p += [k, d]

    if(config.VERBOSITY >= config.DEBUG):
        print("XU.math.multGaussFit: intial parameters")
        print(p)

    ##########################
    # fit with odrpack
    model = odr.Model(fsignal, fjacd=deriv_x, fjacb=deriv_p)
    odata = odr.RealData(lx, ldata)
    my_odr = odr.ODR(odata, model, beta0=p)
    # fit type 2 for least squares
    my_odr.set_job(fit_type=2)
    fit = my_odr.run()

    if(config.VERBOSITY >= config.DEBUG):
        print("XU.math.multPeakFit: fitted parameters")
        print(fit.beta)
    try:
        if fit.stopreason[0] not in ['Sum of squares convergence']:
            print("XU.math.multPeakFit: fit NOT converged (%s)"
                  % fit.stopreason[0])
            return None, None, None, None
    except:
        print("XU.math.multPeakFit: fit most probably NOT converged (%s)"
              % str(fit.stopreason))
        return None, None, None, None
    # prepare return values
    fpos = fit.beta[:-2:3]
    fwidth = numpy.abs(fit.beta[1:-2:3])
    famp = fit.beta[2::3]
    background = numpy.polyval((fit.beta[-2], fit.beta[-1]), x)

    return fpos, fwidth, famp, background


def multGaussPlot(*args, **kwargs):
    """
    convenience function to keep API stable
    see multPeakPlot for documentation
    """
    kwargs['peaktype'] = 'Gaussian'
    return multPeakPlot(*args, **kwargs)


def multPeakPlot(x, fpos, fwidth, famp, background, dranges=None,
                 peaktype='Gaussian', fig="xu_plot", fact=1.):
    """
    function to plot multiple Gaussian/Lorentz peaks with background values
    given by an array

    Parameters
    ----------
     x:  x-coordinate of the data
     fpos:  list of positions of the peaks
     fwidth:  list of width of the peaks
     famp:  list of amplitudes of the peaks
     background:  array with background values
     dranges:  list of tuples with (min,max) value of the data ranges to use.
               does not need to have the same number of entries as fpos
     peaktype: type of peaks to be used: can be either 'Gaussian' or
               'Lorentzian'

     fig:  matplotlib figure number or name
     fact: factor to use as multiplicator in the plot
    """
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        if config.VERBOSITY >= config.INFO_ALL:
            print("XU.math.multPeakPlot: Warning: plot "
                  "functionality not available")
        return

    plt.figure(fig)
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
        plt.plot(lx, (lf + lb) * fact, 'k:')

    # plot summed signal
    plt.plot(lx, (f + lb) * fact, 'r-', lw=1.5)
