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
# Copyright (C) 2016-2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import numbers
from math import sqrt

import numpy
from scipy import interpolate

from .. import materials
from .powder import PowderDiffraction
from .smaterials import Powder, PowderList


def _import_lmfit():
    global lmfit
    try:
        import lmfit
    except ImportError:
        raise ImportError("XU.simpack: Fitting of models needs the lmfit "
                          "package (https://pypi.python.org/pypi/lmfit)")


class PowderModel(object):
    """
    Class to help with powder calculations for multiple materials.  For basic
    calculations the Powder class together with the Fundamental parameters
    approach is used.
    """
    def __init__(self, *args, **kwargs):
        """
        constructor for a powder model. The arguments consist of a PowderList
        or individual Powder(s). Optional parameters are specified in the
        keyword arguments.

        Parameters
        ----------
         *args:    either one PowderList or several Powder objects can be given
         *kwargs:  optional parameters for the simulation. supported are:
           fpclass:  FP_profile derived class with possible convolver mixins.
                     (default: FP_profile)
           fpsettings: settings dictionaries for the convolvers. Default
                       settings are loaded from the config file.
           I0:     scaling factor for the simulation result

        In particular interesting in fpsettings might be:
        {'displacement': {'specimen_displacement': z-displacement of the sample
                                                   from the rotation center
                          'zero_error_deg': zero error of the 2theta angle}
         'absorption': {'sample_thickness': sample thickness (m),
                        'absorption_coefficient': sample's absorption (m^-1)}
         'axial': {'length_sample': the length of the sample in the axial
                                    direction (m)}
        }
        """
        if len(args) == 1 and isinstance(args[0], PowderList):
            self.materials = args[0]
        else:
            self.materials = PowderList('%s List' % self.__class__.__name__,
                                        *args)
        self.I0 = kwargs.pop('I0', 1.0)
        self.pdiff = []
        for mat in self.materials:
            self.pdiff.append(PowderDiffraction(mat, **kwargs))

        # default background
        self._bckg_type = 'polynomial'
        self._bckg_pol = [0, ]

    def set_parameters(self, params):
        """
        set simulation parameters of all subobjects

        Parameters
        ----------
         params:    settings dictionaries for the convolvers.
        """
        # set parameters for each convolver
        for pd in self.pdiff:
            pd.update_settings(params)

    def set_lmfit_parameters(self, lmparams):
        """
        function to update the settings of this class during an least squares
        fit

        Parameters
        ----------
         lmparams:  lmfit Parameters list of sample and instrument parameters
        """
        pv = lmparams.valuesdict()
        settings = dict()
        fp = self.pdiff[0].fp_profile[0].convolvers
        for conv in fp:
            name = conv[5:]
            settings[name] = dict()

        self.I0 = pv.pop('primary_beam_intensity', 1)
        for p in pv:
            if p.startswith('phase_'):  # sample phase parameters
                midx = 0
                for i, name in enumerate(self.materials.namelist):
                    if p.find(name) > 0:
                        midx = i
                name = self.materials.namelist[midx]
                attrname = p[p.find(name) + len(name) + 1:]
                setattr(self.materials[midx], attrname, pv[p])
            elif p.startswith('background_'):
                self._bckg_pol[int(p.split('_')[-1])] = pv[p]
            else:  # instrument parameters
                for k in settings:
                    if p.startswith(k):
                        name = p[len(k) + 1:]
                        settings[k][name] = pv[p]
                        break
        self.set_parameters(settings)

    def create_fitparameters(self):
        """
        function to create a fit model with all instrument and sample
        parameters.

        Parameters
        ----------
         pass

        Returns
        -------
         lmfit Parameters instance
        """
        _import_lmfit()

        params = lmfit.Parameters()
        # sample phase parameters
        for mat, name in zip(self.materials, self.materials.namelist):
            for k in mat.__dict__:
                attr = getattr(mat, k)
                if isinstance(attr, numbers.Number):
                    params.add('_'.join(('phase', name, k)), value=attr,
                               vary=False)

        # instrument parameters
        settings = self.pdiff[0].settings
        for pg in settings:
            for p in settings[pg]:
                val = settings[pg][p]
                if isinstance(val, numbers.Number):
                    params.add('_'.join((pg, p)), value=val, vary=False)

        # other global parameters
        params.add('primary_beam_intensity', value=self.I0, vary=False)
        if self._bckg_type == 'polynomial':
            for i, coeff in enumerate(self._bckg_pol):
                params.add('background_coeff_%d' % i, value=coeff, vary=False)
        return params

    def set_background(self, btype, **kwargs):
        """
        define background as spline or polynomial function

        Parameters
        ----------
         btype:  background type: either 'polynomial' or 'spline'. Depending on
                this value the expected keyword arguments differ.
         kwargs:
            'spline':
                x:  x-values (twotheta) of the background points
                y:  intensity values of the background
            'polynomial':
                p:  polynomial coefficients from the highest degree to the
                    constant term. len of p decides about the degree of the
                    polynomial
        """
        if btype == 'spline':
            self._bckg_spline = interpolate.InterpolatedUnivariateSpline(
                kwargs.get('x'), kwargs.get('y'), ext=0)
        elif btype == 'polynomial':
            self._bckg_pol = list(kwargs.get('p'))
        else:
            raise ValueError("btype must be either 'spline' or 'polynomial'")
        self._bckg_type = btype

    def simulate(self, twotheta, **kwargs):
        """
        calculate the powder diffraction pattern of all materials and sum the
        results based on the relative volume of the materials.

        Parameters
        ----------
         twotheta: positions at which the powder spectrum should be evaluated
         **kwargs:
            background: an array of background values (same shape as twotheta)
                        if no background is given then the background is
                        calculated as previously set by the set_background
                        function or is 0.
            further keyword arguments are passed to the Convolve
            function of of the PowderDiffraction objects

        Returns
        -------
         summed powder diffraction intensity of all materials present in the
         model

        Known issue: possibility to add a background is currently missing!
        """
        inte = numpy.zeros_like(twotheta)
        background = kwargs.pop('background', None)
        if background is None:
            if self._bckg_type == 'spline':
                background = self._bckg_spline(twotheta)
            else:
                background = numpy.polyval(self._bckg_pol, twotheta)
        totalvol = sum(pd.mat.volume for pd in self.pdiff)
        for pd in self.pdiff:
            inte += pd.Calculate(twotheta, **kwargs) * pd.mat.volume / totalvol
        return self.I0 * inte + background

    def fit(self, params, twotheta, data, std=None, maxfev=200):
        """
        make least squares fit with parameters supplied by the user

        Parameters
        ----------
         params:    lmfit Parameters object with all parameters set as intended
                    by the user
         twotheta:  angular values for the fit
         data:      experimental intensities for the fit
         std:       standard deviation of the experimental data. if 'None" the
                    sqrt of the data will be used
         maxfev:    maximal number of simulations during the least squares
                    refinement

        Returns
        -------
         lmfit MinimizerResult
        """
        _import_lmfit()

        def residual(pars, tt, data, weight):
            """
            residual function for lmfit Minimizer routine

            Parameters
            ----------
             pars:      fit Parameters
             tt:        array of twotheta angles
             reflmod:   reflectivity model object
             **kwargs:
              data:     experimental data, same shape as tt (default: None)
              eps:      experimental error bars, shape as tt (default None)
            """
            # set parameters in this instance
            self.set_lmfit_parameters(pars)

            # run simulation
            model = self.simulate(tt)
            return (model - data) * weight

        if std is None:
            weight = numpy.reciprocal(numpy.sqrt(data))
        else:
            weight = numpy.reciprocal(std)
        weight[numpy.isinf(weight)] = 1
        self.minimizer = lmfit.Minimizer(residual, params,
                                         fcn_args=(twotheta, data, weight))
        fitres = self.minimizer.minimize(maxfev=maxfev)
        self.set_lmfit_parameters(fitres.params)
        return fitres

    def __str__(self):
        """
        string representation of the PowderModel
        """
        ostr = "PowderModel {\n"
        ostr += "I0: %f\n" % self.I0
        ostr += str(self.materials)
        ostr += "}"
        return ostr


def Rietveld_error_metrics(exp, sim, weight=None, std=None,
                           Nvar=0, disp=False):
    """
    calculates common error metrics for Rietveld refinement.

    Parameters
    ----------
     exp:       experimental datapoints
     sim:       simulated data
     weight:    weight factor in the least squares sum. If it is None the
                weight is estimated from the counting statistics of 'exp'
     std:       standard deviation of the experimental data. alternative way of
                specifying the weight factor. when both are given weight
                overwrites std!
     Nvar:      number of variables in the refinement
     disp:      flag to tell if a line with the calculated values should be
                printed.

    Returns
    -------
     M, Rp, Rwp, Rwpexp, chi2
    """
    if weight is None and std is None:
        weight = numpy.reciprocal(exp)
    elif weight is None:
        weight = numpy.reciprocal(std**2)
    weight[numpy.isinf(weight)] = 1
    M = numpy.sum((exp - sim)**2 * weight)
    Rp = numpy.sum(numpy.abs(exp - sim))/numpy.sum(exp)
    Rwp = sqrt(M / numpy.sum(weight * exp**2))
    chi2 = M / (len(exp) - Nvar)
    Rwpexp = Rwp / sqrt(chi2)
    if disp:
        print('Rp=%.4f Rwp=%.4f Rwpexp=%.4f chi2=%.4f'
              % (Rp, Rwp, Rwpexp, chi2))
    return M, Rp, Rwp, Rwpexp, chi2


def plot_powder(twotheta, exp, sim, mask=None, scale='sqrt', fig='XU:powder',
                show_diff=True, show_legend=True):
    """
    Convenience function to plot the comparison between experimental and
    simulated powder diffraction data

    Parameters
    ----------
     twotheta:  angle values used for the x-axis of the plot (deg)
     exp:       experimental data (same shape as twotheta)
     sim:       simulated data
     mask:      mask to reduce the twotheta values to the be used as
                x-coordinates of sim
     scale:     string specifying the scale of the y-axis. Valid are: 'linear,
                'sqrt', and 'log'.
     fig:       matplotlib figure name (figure will be cleared!)
     show_diff: flag to specify if a difference curve should be shown
     show_legend: flag to specify if a legend should be shown
    """
    try:  # lazy import of matplotlib
        from matplotlib import pyplot as plt
        from . import mpl_helper
    except ImportError:
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.simpack: Warning: plot "
                  "functionality not available")
        return

    plt.figure(fig, figsize=(10, 7))
    plt.clf()
    ax = plt.subplot(111)
    lines = []
    lines.append(ax.plot(twotheta, exp, 'k.-', label='experiment')[0])
    if mask is None:
        mask = numpy.ones_like(twotheta, dtype=numpy.bool)
    lines.append(ax.plot(twotheta[mask], sim, 'r-', label='simulation')[0])

    if show_diff:
        # plot error between simulation and experiment
        lines.append(ax.plot(twotheta[mask], exp[mask]-sim, '.-', color='0.5',
                             label='difference')[0])

    plt.xlabel('2Theta (deg)')
    plt.ylabel('Intensity')
    leg = plt.figlegend(lines, [l.get_label() for l in lines],
                        loc='upper right', frameon=True)
    ax.set_yscale(scale)
    plt.tight_layout()
    return lines
