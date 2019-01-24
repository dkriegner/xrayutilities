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
# Copyright (C) 2016-2019 Dominik Kriegner <dominik.kriegner@gmail.com>

import warnings

import numpy

from . import models
from .. import config, utilities
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
    reflmod :   SpecularReflectivityModel
        preconfigured model used for the fitting
    params :    lmfit.Parameters
        instance of lmfits Parameters class. For every layer the parameters
        '{}_thickness', '{}_roughness', '{}_density', with '{}' representing
        the layer name are supported. In addition the setup parameters:

          - 'I0' primary beam intensity
          - 'background' background added to the simulation
          - 'sample_width' size of the sample along the beam
          - 'beam_width' width of the beam in the same units
          - 'resolution_width' width of the resolution function in deg
          - 'shift' experimental shift of the incidence angle array

    ai :        array-like
        array of incidence angles for the calculation
    data :      array-like
        experimental data which should be fitted
    eps :       array-like, optional
        error bar of the data
    xmin :      float, optional
        minimum value of ai which should be used. a mask is generated to cut
        away other data
    xmax :      float, optional
        maximum value of ai which should be used. a mask is generated to cut
        away other data
    plot :      bool, optional
        flag to decide wheter an plot should be created showing the fit's
        progress. If plot is a string it will be used as figure name, which
        makes reusing the figures easier.
    verbose :   bool, optional
        flag to tell if the variation of the fitting error should be output
        during the fit.
    elog :      bool, optional
        logarithmic error during the fit
    maxfev :    int, optional
        maximum number of function evaluations during the leastsq optimization

    Returns
    -------
    res : lmfit.MinimizerResult
        object from lmfit, which contains the fitted parameters in res.params
        (see res.params.pretty_print) or try lmfit.report_fit(res)
    """
    warnings.warn("deprecated function -> change to FitModel",
                  DeprecationWarning)
    lmfit = utilities.import_lmfit('XU.simpack')

    if plot:
        plot, plt = utilities.import_matplotlib_pyplot('XU.simpack')

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
            params.add(pname, value=l.roughness, vary=False)
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.simpack.fit_xrr: adding fixed parameter %s to model"
                      % pname)
        pname = '{}_density'.format(lname)
        if pname not in params:
            params.add(pname, value=l.density, vary=False)
            if config.VERBOSITY >= config.INFO_LOW:
                print("XU.simpack.fit_xrr: adding fixed parameter %s to model"
                      % pname)

    # residual function
    def xrr_residual(pars, ai, reflmod, **kwargs):
        """
        residual function for lmfit Minimizer routine

        Parameters
        ----------
        pars :      lmfit.Parameters
            fit parameters
        ai :        array-like
            array of incidence angles
        reflmod :   SpecularReflectivityModel
            reflectivity model object
        data :      array-like, optional
            experimental data of same shape as ai (default: None)
        eps :       array-like
            experimental error bars of same shape as ai (default: None)
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
            logmodel = numpy.log10(model)
            logdata = numpy.log10(data)
            mask = numpy.logical_and(numpy.isfinite(logmodel),
                                     numpy.isfinite(logdata))
            return logmodel[mask] - logdata[mask]
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
        plt.show()
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


class FitModel(object):
    """
    Wrapper for the lmfit Model class working for instances of LayerModel

    Typically this means that after initialization of `FitModel` you want to
    use make_params to get a `lmfit.Parameters` list which one customizes for
    fitting.

    Later on you can call `fit` and `eval` methods with those parameter list.
    """
    def __init__(self, lmodel, verbose=False, plot=False, elog=True, **kwargs):
        """
        initialization of a FitModel which uses lmfit for the actual fitting,
        and generates an according lmfit.Model internally for the given
        pre-configured LayerModel, or subclasses thereof which includes models
        for reflectivity, kinematic and dynamic diffraction.

        Parameters
        ----------
        lmodel :    LayerModel
            pre-configured instance of LayerModel or any subclass
        verbose :   bool, optional
            flag to enable verbose printing during fitting
        plot :      bool or str, optional
            flag to decide wheter an plot should be created showing the fit's
            progress. If plot is a string it will be used as figure name, which
            makes reusing the figures easier.
        elog :      bool, optional
            flag to enable a logarithmic error metric between model and data.
            Since the dynamic range of data is often large its often benefitial
            to keep this enabled.
        kwargs :    dict, optional
            additional keyword arguments are forwarded to the `simulate` method
            of `lmodel`
        """
        self.verbose = verbose
        self.plot = plot
        self.elog = elog
        lmfit = utilities.import_lmfit('XU.simpack')

        assert isinstance(lmodel, models.LayerModel)
        self.lmodel = lmodel
        # generate dynamic function for model evalution
        funcstr = "def func(x, "
        # add LayerModel parameters
        for p in self.lmodel.fit_paramnames:
            funcstr += "{}, ".format(p)
        # add LayerStack parameters
        for l in self.lmodel.lstack:
            for param in self.lmodel.lstack_params:
                funcstr += '{}_{}, '.format(l.name, param)
            if self.lmodel.lstack_structural_params:
                for param in l._structural_params:
                    funcstr += '{}_{}, '.format(l.name, param)
        funcstr += "lmodel=self.lmodel, **kwargs):\n"
        # define modelfunc content
        for p in self.lmodel.fit_paramnames:
            funcstr += "    setattr(lmodel, '{}', {})\n".format(p, p)
        for i, l in enumerate(self.lmodel.lstack):
            for param in self.lmodel.lstack_params:
                varname = '{}_{}'.format(l.name, param)
                cmd = "    setattr(lmodel.lstack[{}], '{}', {})\n"
                funcstr += cmd.format(i, param, varname)
            if self.lmodel.lstack_structural_params:
                for param in l._structural_params:
                    varname = '{}_{}'.format(l.name, param)
                    cmd = "    setattr(lmodel.lstack[{}], '{}', {})\n"
                    funcstr += cmd.format(i, param, varname)
        # perform actual model calculation
        funcstr += "    return lmodel.simulate(x, **kwargs)"

        namespace = {'self': self}
        exec(funcstr, {'lmodel': self.lmodel}, namespace)
        self.func = namespace['func']
        self.emetricfunc = numpy.log10 if self.elog else lambda x: x

        def _residual(params, data, weights, **kwargs):
            """
            Return the residual. This is a (simplified, only real values)
            reimplementation of the lmfit.Model._residual function which adds
            the possibility of a logarithmic error metric.

            Default residual: (data-model)*weights.
            """
            scale = self.emetricfunc
            model = scale(self.eval(params, **kwargs))
            sdata = scale(data)
            mask = numpy.logical_and(numpy.isfinite(model),
                                     numpy.isfinite(sdata))
            diff = model[mask] - sdata[mask]
            if weights is not None and scale(1) == 1:
                diff *= weights
            return numpy.asarray(diff).ravel()

        self.lmm = lmfit.Model(self.func,
                               name=self.lmodel.__class__.__name__, **kwargs)
        self.lmm._residual = _residual
        for method in ('set_param_hint', 'print_param_hints', 'eval',
                       'make_params'):
            setattr(self, method, getattr(self.lmm, method))
        # set default parameter hints
        self._default_hints()
        self.set_fit_limits()

    def set_fit_limits(self, xmin=-numpy.inf, xmax=numpy.inf, mask=None):
        """
        set fit limits. If mask is given it must have the same size as the
        `data` and `x` variables given to fit. If mask is None it will be
        generated from xmin and xmax.

        Parameters
        ----------
        xmin :  float, optional
            minimum value of x-values to include in the fit
        xmax :  float, optional
            maximum value of x-values to include in the fit
        mask :  boolean array, optional
            mask to be used for the data given to the fit
        """
        self.mask = mask
        self.xmin = xmin
        self.xmax = xmax

    def fit(self, data, params, x, weights=None, fit_kws=None, **kwargs):
        """
        wrapper around lmfit.Model.fit which enables plotting during the
        fitting

        Parameters
        ----------
        data :      ndarray
            experimental values
        params :    lmfit.Parameters
            list of parameters for the fit, use make_params for generation
        x :         ndarray
            independent variable (incidence angle or q-position depending on
            the model)
        weights :   ndarray, optional
            values of weights for the fit, same size as data
        fit_kws :   dict, optional
            Options to pass to the minimizer being used
        kwargs :    dict, optional
            keyword arguments which are passed to lmfit.Model.fit

        Returns
        -------
        lmfit.ModelResult
        """
        if self.mask:
            mask = self.mask
        else:
            mask = numpy.logical_and(x >= self.xmin, x <= self.xmax)
        mweights = weights
        if mweights is not None:
            mweights = weights[mask]

        if self.plot:
            flag, plt = utilities.import_matplotlib_pyplot('XU.simpack')
        # plot of initial values
        if self.plot:
            plt.ion()
            if isinstance(self.plot, basestring):
                self.fig = plt.figure(self.plot)
            else:
                self.fig = plt.figure('XU:FitModel')
            plt.clf()
            self.ax = plt.subplot(111)
            if self.elog:
                self.ax.set_yscale("log", nonposy='clip')
            if weights is not None:
                eline = plt.errorbar(x, data, yerr=1/weights, ecolor='0.3',
                                     fmt='ko', errorevery=int(x.size/80),
                                     label='data')[0]
            else:
                eline, = plt.plot(x, data, 'ko', label='data')
            if self.verbose:
                init, = plt.plot(x, self.eval(params, x=x, **kwargs),
                                 '-', color='0.5', label='initial')
            if eline:
                zord = eline.zorder+2
            else:
                zord = 1
            fline, = plt.plot(x[mask], self.eval(params, x=x[mask], **kwargs),
                              'r-', lw=2, label='fit', zorder=zord)
            plt.legend()
            plt.xlabel(self.lmodel.xlabelstr)
            plt.ylabel('Intensity (arb. u.)')
            plt.show()
        else:
            fline = None

        # create callback function
        def cb_func(params, niter, resid, *args, **kwargs):
            x = kwargs['x']
            if self.verbose:
                print('{:04d} {:12.3e}'.format(niter, numpy.sum(resid**2)))
            if self.plot and niter % 20 == 0:
                plt.sca(self.ax)
                fline.set_ydata(self.eval(params, **kwargs))
                plt.draw()
                plt.pause(0.001)  # enable better mpl backend compatibility

        res = self.lmm.fit(data[mask], params, x=x[mask], weights=mweights,
                           fit_kws=fit_kws, iter_cb=cb_func, **kwargs)

        # final update of plot
        if self.plot:
            plt.sca(self.ax)
            plt.plot(x, self.eval(res.params, x=x, **kwargs),
                     'g-', lw=1, label='fit', zorder=fline.zorder-1)
            cb_func(res.params, -1, res.residual, data, mweights, x=x[mask],
                    **kwargs)
            plt.tight_layout()

        return res

    def _default_hints(self):
        """
        set useful hints for parameters all LayerModels have
        """
        # general parameters
        for pn in self.lmodel.fit_paramnames:
            self.set_param_hint(pn, value=getattr(self.lmodel, pn), vary=False)
        for pn in ('I0', 'background'):
            self.set_param_hint(pn, vary=True, min=0)
        self.set_param_hint('resolution_width', min=0, vary=False)
        self.set_param_hint('energy', min=1000, vary=False)

        # parameters of the layerstack
        for l in self.lmodel.lstack:
            for param in self.lmodel.lstack_params:
                varname = '{}_{}'.format(l.name, param)
                self.set_param_hint(varname, value=getattr(l, param), min=0)
                if param == 'density':
                    self.set_param_hint(varname, max=1.5*l.material.density)
                if param == 'thickness':
                    self.set_param_hint(varname, max=2*l.thickness)
                if param == 'roughness':
                    self.set_param_hint(varname, max=50)
            if self.lmodel.lstack_structural_params:
                for param in l._structural_params:
                    varname = '{}_{}'.format(l.name, param)
                    self.set_param_hint(varname, value=getattr(l, param),
                                        vary=False)
                    if 'occupation' in param:
                        self.set_param_hint(varname, min=0, max=1)
                    if 'biso' in param:
                        self.set_param_hint(varname, min=0, max=5)
        if self.lmodel.lstack[0].thickness == numpy.inf:
            varname = '{}_{}'.format(self.lmodel.lstack[0].name, 'thickness')
            self.set_param_hint(varname, vary=False)
