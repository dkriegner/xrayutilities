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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy

from . import SpecularReflectivityModel
from .. import config
from ..exception import InputError

# python 2to3 compatibility
try:
    basestring
except NameError:
    basestring = str


def fit_xrr(reflmod, params, ai, data=None, eps=None, xmin=-numpy.inf,
            xmax=numpy.inf, plot=False, verbose=False, elog=True, maxfev=500):
    """
    optimize function for a Reflectivity Model using lmfit. The fitting
    parameters must be specified as instance of lmfits Parameters class.

    Parameters
    ----------
     reflmod:   preconfigured SpecularReflectivityModel
     params:    instance of lmfits Parameters class. For every layer the
                parameters '{}_thickness', '{}_roughness', '{}_density', with
                '{}' representing the layer name are supported. In addition the
                setup parameters:
                 - 'I0' primary beam intensity
                 - 'background' background added to the simulation
                 - 'sample_width' size of the sample along the beam
                 - 'beam_width' width of the beam in the same units
                 - 'resolution_width' width of the resolution function in deg
                 - 'shift' experimental shift of the incidence angle array
     ai:        array of incidence angles for the calculation
     data:      experimental data which should be fitted
     eps:       (optional) error bar of the data
     xmin, xmax: minimum and maximum values of ai which should be used. a mask
                 is generated to cut away other data
     plot:      flag to decide wheter an plot should be created showing the
                fit's progress. If plot is a string it will be used as figure
                name, which makes reusing the figures easier.
     verbose:   flag to tell if the variation of the fitting error should be
                output during the fit.
     elog:      logarithmic error during the fit
     maxfev:    maximum number of function evaluations during the leastsq
                optimization

    Returns
    -------
     res: MinimizerResult object from lmfit, which contains the fitted
          parameters in res.params (see res.params.pretty_print) or try
          lmfit.report_fit(res)
    """
    try:
        import lmfit
    except ImportError:
        raise ImportError("XU.simpack: Fitting of models needs the lmfit "
                          "package (https://pypi.python.org/pypi/lmfit)")
        return

    if plot:
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            plot = False
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.simpack: Warning: plot "
                      "functionality not available")

    mask = numpy.logical_and(ai > xmin, ai < xmax)
    # check Parameters
    lstack = reflmod.lstack
    if not isinstance(params, lmfit.Parameters):
        raise TypeError('params argument must be of type lmfit.Parameters')
    for lname, l in zip(lstack.namelist, lstack):
        pname = '{}_thickness'.format(lname)
        if pname not in params:
            raise InputError('XU.simpack.fit_xrr: Parameter %s not defined.'
                             % pname)
        pname = '{}_roughness'.format(lname)
        if pname not in params:
            params.add(pname, value=getattr(l, 'roughness', 0), vary=False)
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.simpack.fit_xrr: adding fixed parameter %s to model"
                      % pname)
        pname = '{}_density'.format(lname)
        if pname not in params:
            params.add(pname, value=getattr(l, 'density', 1), vary=False)
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.simpack.fit_xrr: adding fixed parameter %s to model"
                      % pname)

    # residual function
    def xrr_residual(pars, ai, reflmod, **kwargs):
        """
        residual function for lmfit Minimizer routine

        Parameters
        ----------
         pars:      fit Parameters
         ai:        array of incidence angles
         reflmod:   reflectivity model object
         **kwargs:
          data:     experimental data of same shape as ai (default: None)
          eps:      experimental error bars of same shape as ai (default None)
        """
        data = kwargs.get('data', None)
        eps = kwargs.get('eps', None)
        pvals = pars.valuesdict()
        # update reflmod global parameters:
        reflmod.I0 = pvals.get('I0', reflmod.I0)
        reflmod.background = pvals.get('background', reflmod.background)
        reflmod.sample_width = pvals.get('sample_width', reflmod.sample_width)
        reflmod.beam_width = pvals.get('beam_width', reflmod.beam_width)
        reflmod.resolution_width = pvals.get('resolution_width',
                                             reflmod.resolution_width)
        shift = pvals.get('shift', 0)
        # update layer properties
        for lname, l in zip(reflmod.lstack.namelist, reflmod.lstack):
            l.thickness = pvals['{}_thickness'.format(lname)]
            l.roughness = pvals['{}_roughness'.format(lname)]
            l.density = pvals['{}_density'.format(lname)]
        # run simulation
        model = reflmod.simulate(ai - shift)
        if data is None:
            return model
        if kwargs['elog']:
            return numpy.log10(model) - numpy.log10(data)
        if eps is None:
            return (model - data)
        return (model - data)/eps

    # plot of initial values
    if plot:
        plt.ion()
        if isinstance(plot, basestring):
            plt.figure(plot)
        else:
            plt.figure('XU:fit_xrr')
        plt.clf()
        ax = plt.subplot(111)
        ax.set_yscale("log", nonposy='clip')
        if data is not None:
            if eps is not None:
                eline = plt.errorbar(ai, data, yerr=eps, ecolor='0.3',
                                     fmt='ko', errorevery=int(ai.size/80),
                                     label='data')[0]
            else:
                eline, = plt.semilogy(ai, data, 'ko', label='data')
        if verbose:
            init, = plt.semilogy(ai,
                                 xrr_residual(params, ai, reflmod, data=None),
                                 '-', color='0.5', label='initial')
        if eline:
            zord = eline.zorder+2
        else:
            zord = 1
        fline, = plt.semilogy(
                ai[mask], xrr_residual(params, ai[mask], reflmod, data=None),
                'r-', lw=2, label='fit', zorder=zord)
        plt.legend()
        plt.xlabel('incidence angle (deg)')
        plt.ylabel('Intensity (arb. u.)')
        plt.show(block=False)
    else:
        fline = None

    # create and run minimizer/minimization
    if eps is None:
        eps = numpy.ones(ai.shape)

    def cb_func(params, niter, resid, ai, reflmod, **kwargs):
        if kwargs.get('verbose', False):
            print('{:04d} {:12.3e}'.format(niter, numpy.sum(resid**2)))
        if kwargs.get('plot', False) and niter % 20 == 0:
            fl = kwargs['fline']
            plt.sca(ax)
            fl.set_ydata(xrr_residual(params, ai, reflmod, data=None))
            plt.draw()
            plt.pause(0.001)  # enable better mpl backend compatibility

    minimizer = lmfit.Minimizer(
            xrr_residual, params, fcn_args=(ai[mask], reflmod),
            fcn_kws={'data': data[mask], 'eps': eps[mask], 'fline': fline,
                     'verbose': verbose, 'plot': plot, 'elog': elog},
            iter_cb=cb_func, maxfev=maxfev)
    res = minimizer.minimize()

    # final update of plot
    if plot:
        plt.sca(ax)
        plt.semilogy(ai, xrr_residual(res.params, ai, reflmod, data=None),
                     'g-', lw=1, label='fit', zorder=fline.zorder-1)
        cb_func(res.params, 0, res.residual, ai[mask], reflmod, fline=fline,
                plot=True)

    return res
