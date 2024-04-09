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
# Copyright (c) 2016-2021, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import numpy
from lmfit import Model

from .. import utilities
from . import models


class FitModel(Model):
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

        assert isinstance(lmodel, models.LayerModel)
        self.lmodel = lmodel
        # generate dynamic function for model evalution
        funcstr = "def func(x, "
        # add LayerModel parameters
        for p in self.lmodel.fit_paramnames:
            funcstr += f"{p}, "
        # add LayerStack parameters
        for layer in self.lmodel.lstack:
            for param in self.lmodel.lstack_params:
                funcstr += f'{layer.name}_{param}, '
            if self.lmodel.lstack_structural_params:
                for param in layer._structural_params:
                    funcstr += f'{layer.name}_{param}, '
        funcstr += "lmodel=self.lmodel, **kwargs):\n"
        # define modelfunc content
        for p in self.lmodel.fit_paramnames:
            funcstr += f"    setattr(lmodel, '{p}', {p})\n"
        for i, l in enumerate(self.lmodel.lstack):
            for param in self.lmodel.lstack_params:
                varname = f'{l.name}_{param}'
                cmd = "    setattr(lmodel.lstack[{}], '{}', {})\n"
                funcstr += cmd.format(i, param, varname)
            if self.lmodel.lstack_structural_params:
                for param in l._structural_params:
                    varname = f'{l.name}_{param}'
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

        super().__init__(self.func,
                         name=self.lmodel.__class__.__name__, **kwargs)
        self._residual = _residual
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

    def fit(self, data, params, x, weights=None, fit_kws=None, lmfit_kws=None,
            **kwargs):
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
        lmfit_kws : dict, optional
            keyword arguments which are passed to lmfit.Model.fit
        kwargs :    dict, optional
            keyword arguments passed to lmfit.Model.eval

        Returns
        -------
        lmfit.ModelResult
        """
        if not lmfit_kws:
            lmfit_kws = {}

        class FitPlot:
            def __init__(self, figname, logscale):
                self.figname = figname
                self.logscale = logscale
                if not self.figname:
                    self.plot = False
                else:
                    f, plt = utilities.import_matplotlib_pyplot('XU.simpack')
                    self.plt = plt
                    self.plot = f

            def plot_init(self, x, data, weights, model, mask, verbose):
                if not self.plot:
                    return
                self.plt.ion()
                if isinstance(self.figname, str):
                    self.fig = self.plt.figure(self.figname)
                else:
                    self.fig = self.plt.figure('XU:FitModel')
                self.plt.clf()
                self.ax = self.plt.subplot(111)

                if weights is not None:
                    eline = self.ax.errorbar(
                        x, data, yerr=1/weights, ecolor='0.3', fmt='ok',
                        errorevery=int(x.size/80), label='data')[0]
                else:
                    eline, = self.ax.plot(x, data, 'ok', label='data')
                if verbose:
                    self.ax.plot(x, model, '-', color='0.5', label='initial')
                if eline:
                    self.zord = eline.zorder+2
                else:
                    self.zord = 1
                if self.logscale:
                    self.ax.set_yscale("log")
                self.fline = None

            def showplot(self, xlab, ylab='Intensity (arb. u.)'):
                if not self.plot:
                    return
                self.plt.xlabel(xlab)
                self.plt.ylabel(ylab)
                self.plt.legend()
                self.fig.set_tight_layout(True)
                self.plt.show()

            def updatemodelline(self, x, newmodel):
                if not self.plot:
                    return
                try:
                    self.plt.sca(self.ax)
                except ValueError:
                    return
                if self.fline is None:
                    self.fline, = self.ax.plot(
                        x, newmodel, '-r', lw=2, label='fit', zorder=self.zord)
                else:
                    self.fline.set_data(x, newmodel)
                canvas = self.fig.canvas  # see plt.draw function (avoid show!)
                canvas.draw_idle()
                canvas.start_event_loop(0.001)

            def addfullmodelline(self, x, y):
                if not self.plot:
                    return
                self.ax.plot(x, y, '-g', lw=1, label='full model',
                             zorder=self.zord-1)

        if self.mask:
            mask = self.mask
        else:
            mask = numpy.logical_and(x >= self.xmin, x <= self.xmax)
        mweights = weights
        if mweights is not None:
            mweights = weights[mask]

        # create initial plot
        self.fitplot = FitPlot(self.plot, self.elog)
        initmodel = self.eval(params, x=x, **kwargs)
        self.fitplot.plot_init(x, data, weights, initmodel, mask, self.verbose)
        self.fitplot.showplot(xlab=self.lmodel.xlabelstr)

        # create callback function
        def cb_func(params, niter, resid, *args, **kwargs):
            if self.verbose:
                print(f'{niter:04d} {numpy.sum(resid ** 2):12.3e}')
            if self.fitplot.plot and niter % 20 == 0:
                self.fitplot.updatemodelline(kwargs['x'],
                                             self.eval(params, **kwargs))

        # perform fitting
        res = super().fit(data[mask], params, x=x[mask], weights=mweights,
                          fit_kws=fit_kws, iter_cb=cb_func, **lmfit_kws)

        # final update of plot
        if self.fitplot.plot:
            try:
                self.fitplot.plt.sca(self.fitplot.ax)
            except ValueError:
                self.fitplot.plot_init(x, data, weights, initmodel, mask,
                                       self.verbose)
            fittedmodel = self.eval(res.params, x=x, **kwargs)
            self.fitplot.addfullmodelline(x, fittedmodel)
            self.fitplot.updatemodelline(x[mask], fittedmodel[mask])
            self.fitplot.showplot(xlab=self.lmodel.xlabelstr)

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
        for lay in self.lmodel.lstack:
            for param in self.lmodel.lstack_params:
                varname = f'{lay.name}_{param}'
                self.set_param_hint(varname, value=getattr(lay, param), min=0)
                if param == 'density':
                    self.set_param_hint(varname, max=1.5*lay.material.density)
                if param == 'thickness':
                    self.set_param_hint(varname, max=2*lay.thickness)
                if param == 'roughness':
                    self.set_param_hint(varname, max=50)
            if self.lmodel.lstack_structural_params:
                for param in lay._structural_params:
                    varname = f'{lay.name}_{param}'
                    self.set_param_hint(varname, value=getattr(lay, param),
                                        vary=False)
                    if 'occupation' in param:
                        self.set_param_hint(varname, min=0, max=1)
                    if 'biso' in param:
                        self.set_param_hint(varname, min=0, max=5)
        if self.lmodel.lstack[0].thickness == numpy.inf:
            varname = f"{self.lmodel.lstack[0].name}_thickness"
            self.set_param_hint(varname, vary=False)
