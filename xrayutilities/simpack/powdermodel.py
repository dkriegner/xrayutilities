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

import numpy

from .smaterials import Powder, PowderList
from .powder import PowderDiffraction
from .. import materials


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
           background: NotImplemented

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

    def set_parameters(self, params):
        """
        set simulation parameters of all subobjects

        Parameters
        ----------
         params:    settings dictionaries for the convolvers.
        """

        if 'emission' in params:
            if 'emiss_wavelength' in params['emission']:
                wl = params['emission']['emiss_wavelengths'][0]
                if 'global' in params:
                    params['global']['dominant_wavelength'] = wl
                else:
                    params['global'] = {'dominant_wavelength': wl}

                # set wavelength in base class
                for pd in self.pdiff:
                    pd._set_wavelength(wl*1e10)

        # set parameters for each convolver
        for k in params:
            for pd in self.pdiff:
                pd.fp_profile.set_parameters(convolver=k, **params[k])

    def simulate(self, twotheta, **kwargs):
        """
        calculate the powder diffraction pattern of all materials and sum the
        results based on the relative volume of the materials.

        Parameters
        ----------
         twotheta: positions at which the powder spectrum should be evaluated
         **kwargs:
            background: an array of background values (same shape as twotheta)
                        alternatively the background can be set before the
            further keyword arguments are passed to the Convolve
            function of of the PowderDiffraction objects

        Returns
        -------
         summed powder diffraction intensity of all materials present in the
         model

        Known issue: possibility to add a background is currently missing!
        """
        inte = numpy.zeros_like(twotheta)
        background = kwargs.pop('background', 0)
        totalvol = sum(pd.mat.volume for pd in self.pdiff)
        for pd in self.pdiff:
            inte += pd.Calculate(twotheta, **kwargs) * pd.mat.volume / totalvol
        return self.I0 * inte + background

    def get_fitmodel(self):
        try:
            import lmfit
        except ImportError:
            raise ImportError("XU.simpack: Fitting of models needs the lmfit "
                              "package (https://pypi.python.org/pypi/lmfit)")
            return

        def fit_residual(pars, tt, **kwargs):
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
            data = kwargs.get('data', None)
            eps = kwargs.get('eps', None)
            pvals = pars.valuesdict()
            # set parameters in model
            # ...

            # run simulation
            model = self.simulate(tt)
            if data is None:
                return model
            if kwargs['elog']:
                return numpy.log10(model) - numpy.log10(data)
            if eps is None:
                return (model - data)
            return (model - data)/eps

#        minimizer = lmfit.Minimizer(
#                fit_residual, params, fcn_args=(ai[mask], reflmod),
#                fcn_kws={'data': data[mask], 'eps': eps[mask]},
#                iter_cb=cb_func, maxfev=maxfev)
#        res = minimizer.minimize()

    def __str__(self):
        """
        string representation of the PowderModel
        """
        ostr = "PowderModel {\n"
        ostr += "I0: %f\n" % self.I0
        ostr += str(self.materials)
        ostr += "}"
        return ostr


def plot_powder(twotheta, exp, sim, scale='sqrt', fig='XU:powder',
                show_diff=True, show_legend=True):
    """
    Convenience function to plot the comparison between experimental and
    simulated powder diffraction data

    Parameters
    ----------
     twotheta:  angle values used for the x-axis of the plot (deg)
     exp:       experimental data (same shape as twotheta)
     sim:       simulated data (same shape as twotheta)
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
    lines.append(ax.plot(twotheta, sim, 'r-', label='simulation')[0])

    if show_diff:
        # plot error between simulation and experiment
        lines.append(ax.plot(twotheta, exp-sim, '.-', color='0.5',
                             label='difference')[0])

    plt.xlabel('2Theta (deg)')
    plt.ylabel('Intensity')
    leg = plt.figlegend(lines, [l.get_label() for l in lines],
                        loc='upper right', frameon=True)
    ax.set_yscale(scale)
    plt.tight_layout()
    return lines
