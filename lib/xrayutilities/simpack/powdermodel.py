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
# Copyright (c) 2017-2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import numbers
from math import sqrt

import numpy
from scipy import interpolate

from .. import utilities
from .powder import PowderDiffraction
from .smaterials import PowderList


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

        Note:
            After the end-of-use it is advisable to call the `close()` method
            to cleanup the multiprocessing calculation!

        Parameters
        ----------
        args :      PowderList or Powders
            either one PowderList or several Powder objects can be given
        kwargs :    dict
            optional parameters for the simulation. supported are:
        fpclass :   FP_profile, optional
            derived class with possible convolver mixins.  (default:
            FP_profile)
        fpsettings : dict
            settings dictionaries for the convolvers. Default settings are
            loaded from the config file.
        I0 :        float, optional
            scaling factor for the simulation result

        Notes
        -----
        In particular interesting keys in the fpsettings dictionary might be:
         'displacement':
          {'specimen_displacement': sample's z-displacement from the rotation
                                    center
           'zero_error_deg': zero error of the 2theta angle}

         'absorption':
          {'sample_thickness': sample thickness (m),
           'absorption_coefficient': sample's absorption (m^-1)}

         'axial':
          {'length_sample': sample length in the axial direction (m)}
        """
        if len(args) == 1 and isinstance(args[0], PowderList):
            self.materials = args[0]
        else:
            self.materials = PowderList(f'{self.__class__.__name__} List',
                                        *args)
        self.I0 = kwargs.pop('I0', 1.0)
        self.pdiff = []
        kwargs['enable_simulation'] = True
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
        params :    dict
            settings dictionaries for the convolvers.
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
        lmparams :  lmfit.Parameters
            lmfit Parameters list of sample and instrument parameters
        """
        pv = lmparams.valuesdict()
        settings = dict()
        h = list(self.pdiff[0].data)[0]
        fp = self.pdiff[0].data[h]['conv'].convolvers
        for conv in fp:
            name = conv[5:]
            settings[name] = dict()

        self.I0 = pv.pop('primary_beam_intensity', 1)
        set_splbkg = False
        spliney = {}
        for p in pv:
            if p.startswith('phase_'):  # sample phase parameters
                midx = 0
                for i, name in enumerate(self.materials.namelist):
                    if p.find(name) > 0:
                        midx = i
                name = self.materials.namelist[midx]
                attrname = p[p.find(name) + len(name) + 1:]
                setattr(self.materials[midx], attrname, pv[p])
            elif p.startswith('background_coeff'):
                self._bckg_pol[int(p.split('_')[-1])] = pv[p]
            elif p.startswith('background_spl_coeff'):
                set_splbkg = True
                spliney[int(p.split('_')[-1])] = pv[p]
            else:  # instrument parameters
                for k in settings:
                    if p.startswith(k):
                        slist = p[len(k) + 1:].split('_')
                        if len(slist) > 2 and slist[-2] == 'item':
                            name = '_'.join(slist[:-2])
                            if slist[-1] == '0':
                                settings[k][name] = []
                            settings[k][name].append(pv[p])
                        else:
                            name = p[len(k) + 1:]
                            settings[k][name] = pv[p]
                        break
        if set_splbkg:
            self._bckg_spline = interpolate.InterpolatedUnivariateSpline(
                self._bckg_spline._data[0],
                [spliney[k] for k in sorted(spliney)], ext=0)
        self.set_parameters(settings)

    def create_fitparameters(self):
        """
        function to create a fit model with all instrument and sample
        parameters.

        Returns
        -------
        lmfit.Parameters
        """
        lmfit = utilities.import_lmfit('XU.PowderModel')

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
                if p == 'dominant_wavelength' and pg == 'global':
                    # wavelength must be fit using emission_emiss_wavelength
                    continue
                if isinstance(val, numbers.Number):
                    params.add('_'.join((pg, p)), value=val, vary=False)
                elif isinstance(val, (numpy.ndarray, tuple, list)):
                    for j, item in enumerate(val):
                        params.add('_'.join((pg, p, 'item_%d' % j)),
                                   value=item, vary=False)

        # other global parameters
        params.add('primary_beam_intensity', value=self.I0, vary=False)
        if self._bckg_type == 'polynomial':
            for i, coeff in enumerate(self._bckg_pol):
                params.add('background_coeff_%d' % i, value=coeff, vary=False)
        elif self._bckg_type == 'spline':
            for i, coeff in enumerate(self._bckg_spline._data[1]):
                params.add('background_spl_coeff_%d' % i, value=coeff,
                           vary=False)
        return params

    def set_background(self, btype, **kwargs):
        """
        define background as spline or polynomial function

        Parameters
        ----------
        btype :     {polynomial', 'spline'}
            background type; Depending on this
            value the expected keyword arguments differ.
        kwargs :    dict
            optional keyword arguments
        x :     array-like, optional
            x-values (twotheta) of the background points (if btype='spline')
        y :     array-like, optional
            intensity values of the background (if btype='spline')
        p :     array-like, optional
            polynomial coefficients from the highest degree to the constant
            term. len of p decides about the degree of the polynomial (if
            btype='polynomial')
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
        twotheta :  array-like
            positions at which the powder pattern should be evaluated
        kwargs :    dict
            optional keyword arguments
        background : array-like
            an array of background values (same shape as twotheta) if no
            background is given then the background is calculated as previously
            set by the set_background function or is 0.


        further keyword arguments are passed to the Convolve function of of the
        PowderDiffraction objects

        Returns
        -------
        array-like
            summed powder diffraction intensity of all materials present in the
            model
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
        params :    lmfit.Parameters
            object with all parameters set as intended by the user
        twotheta :  array-like
            angular values for the fit
        data :      array-like
            experimental intensities for the fit
        std :       array-like
            standard deviation of the experimental data. if 'None' the sqrt of
            the data will be used
        maxfev:     int
            maximal number of simulations during the least squares refinement

        Returns
        -------
        lmfit.MinimizerResult
        """
        lmfit = utilities.import_lmfit('XU.PowderModel')

        def residual(pars, tt, data, weight):
            """
            residual function for lmfit Minimizer routine

            Parameters
            ----------
            pars :      lmfit.Parameters
                fit Parameters
            tt :        array-like
                array of twotheta angles
            data :      array-like
                experimental data, same shape as tt
            eps :       array-like
                experimental error bars, shape as tt
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
        fitres = self.minimizer.minimize(max_nfev=maxfev)
        self.set_lmfit_parameters(fitres.params)
        return fitres

    def plot(self, twotheta, showlines=True, label='simulation', color=None,
             formatspec='-', lcolors=[], ax=None, **kwargs):
        """
        plot the powder diffraction pattern and indicate line positions for all
        components in the model.

        Parameters
        ----------
        twotheta :  array-like
            positions at which the powder pattern should be evaluated
        showlines : bool, optional
            flag to decide if peak positions of the components will be shown on
            the top of the plot
        label :     str
            line label in the plot
        color :     matplotlib color or None
            the color used for the line plot of the simulation
        formatspec : str
            format specifier of the simulation curve
        lcolors :   list of matplotlib colors
            colors for the line indicators for the various components
        ax :        matplotlib.axes or None
            axes object to be used for plotting, if its given no axes
            decoration like labels are set.

        Further keyword arguments are passed to the simulate method.

        Returns
        -------
        matplotlib.axes object or None if matplotlib is not available
        """
        plot, plt = utilities.import_matplotlib_pyplot('XU.simpack')
        if not plot:
            return None

        if ax is None:
            fig, iax = plt.subplots()
            iax.set_xlabel(r"$2\theta$ (°)")
            iax.set_ylabel(r"Intensity")
            iax.set_xlim(twotheta.min(), twotheta.max())
        else:
            fig = ax.figure
            iax = ax

        plotkwargs = dict(label=label)
        if color is not None:
            plotkwargs['color'] = color
        iax.plot(twotheta, self.simulate(twotheta, **kwargs), formatspec,
                 **plotkwargs)

        if showlines:
            from matplotlib.colors import is_color_like
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(iax)
            taxlist = []
            lineslist = []
            annotlist = []
            settings = self.pdiff[0].settings
            wavelengths = settings['emission']['emiss_wavelengths']
            intensities = settings['emission']['emiss_intensities']
            for i, pd in enumerate(self.pdiff):
                tax = divider.append_axes("top", size="6%", pad=0.05,
                                          sharex=iax)
                if lcolors:
                    c = lcolors[i % len(lcolors)]
                elif len(self.pdiff) == 1:
                    if is_color_like(color):
                        c = color
                    elif is_color_like(formatspec[-1]):
                        c = formatspec[-1]
                    else:
                        c = 'C0'
                else:
                    c = f'C{i}'
                lw = 2
                wllist = []
                for wl, inte in zip(wavelengths, intensities):
                    q = [pd.data[h]['qpos'] for h in pd.data]
                    tt = pd.Q2Ang(q, wl=1e10*wl)*2
                    lw *= inte
                    lines = tax.vlines(tt, 0, 1, colors=c, linewidth=lw)
                    wllist.append(lines)

                plt.setp(tax.get_xticklabels(), visible=False)
                plt.setp(tax.get_yticklabels(), visible=False)
                plt.setp(tax.get_yticklines(), visible=False)
                annot = tax.annotate(
                    "", xy=(0, 0), xytext=(20, 0), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.1", fc="w", alpha=0.8),
                    arrowprops=dict(arrowstyle="->"), fontsize='x-small')
                annot.set_visible(False)
                # next line important to avoid zorder issues
                tax.figure.texts.append(tax.texts.pop())
                taxlist.append(tax)
                lineslist.append(wllist)
                annotlist.append(annot)

            def update_annot(pd, annot, lines, ind):
                h = list(pd.data)[ind]
                x = 2*pd.data[h]['ang']
                y = 0.5
                annot.xy = (x, y)
                text = f"{pd.mat.name}: {h[0]} {h[1]} {h[2]}"
                annot.set_text(text)
                annot.get_bbox_patch().set_edgecolor(lines.get_color()[0])
                annot.set_zorder(10)

            def hover(event):
                for pd, tax, annot, wllist in zip(self.pdiff, taxlist,
                                                  annotlist, lineslist):
                    vis = annot.get_visible()
                    if event.inaxes == tax:
                        for lines in wllist:
                            cont, ind = lines.contains(event)
                            if cont:
                                update_annot(pd, annot, lines, ind['ind'][0])
                                annot.set_visible(True)
                                fig.canvas.draw_idle()
                                return
                            else:
                                if vis:
                                    annot.set_visible(False)
                                    fig.canvas.draw_idle()
                                    return

            def click(event):
                for pd, tax, annot, wllist in zip(self.pdiff, taxlist,
                                                  annotlist, lineslist):
                    if event.inaxes == tax:
                        for lines in wllist:
                            cont, ind = lines.contains(event)
                            if cont:
                                h = list(pd.data)[ind['ind'][0]]
                                text = (f'{pd.mat.name}: {h[0]} {h[1]} {h[2]};'
                                        f' 2Theta = ')
                                for wl in wavelengths:
                                    tt = 2 * pd.Q2Ang(pd.data[h]['qpos'],
                                                      wl=1e10*wl)
                                    text += f'{tt:.4f}°, '
                                print(text[:-2])
                                return

            fig.canvas.mpl_connect("motion_notify_event", hover)
            fig.canvas.mpl_connect("button_press_event", click)
        if ax is None:
            fig.tight_layout()
        return iax

    def close(self):
        for pd in self.pdiff:
            pd.close()

    def __str__(self):
        """
        string representation of the PowderModel
        """
        ostr = "PowderModel {\n"
        ostr += f"I0: {self.I0:f}\n"
        ostr += str(self.materials)
        ostr += "}"
        return ostr


def Rietveld_error_metrics(exp, sim, weight=None, std=None,
                           Nvar=0, disp=False):
    """
    calculates common error metrics for Rietveld refinement.

    Parameters
    ----------
    exp :       array-like
        experimental datapoints
    sim :       array-like
        simulated data
    weight :    array-like, optional
        weight factor in the least squares sum. If it is None the weight is
        estimated from the counting statistics of 'exp'
    std :       array-like, optional
        standard deviation of the experimental data. alternative way of
        specifying the weight factor. when both are given weight overwrites
        std!
    Nvar :      int, optional
        number of variables in the refinement
    disp :      bool, optional
        flag to tell if a line with the calculated values should be printed.

    Returns
    -------
    M, Rp, Rwp, Rwpexp, chi2: float
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
        print(f"Rp={Rp:.4f} Rwp={Rwp:.4f} Rwpexp={Rwpexp:.4f} chi2={chi2:.4f}")
    return M, Rp, Rwp, Rwpexp, chi2


def plot_powder(twotheta, exp, sim, mask=None, scale='sqrt', fig='XU:powder',
                show_diff=True, show_legend=True, labelexp='experiment',
                labelsim='simulation', formatexp='.-k', formatsim='-r'):
    """
    Convenience function to plot the comparison between experimental and
    simulated powder diffraction data

    Parameters
    ----------
    twotheta :  array-like
        angle values used for the x-axis of the plot (deg)
    exp :       array-like
        experimental data (same shape as twotheta). If None only the simulation
        and no difference will be plotted
    sim :       array-like or PowederModel
        simulated data or PowderModel instance. If a PowderModel instance is
        given the plot-method of PowderModel is used.
    mask :      array-like, optional
        mask to reduce the twotheta values to the be used as x-coordinates of
        sim
    scale :     {'linear', 'sqrt', 'log'}, optional
        string specifying the scale of the y-axis.
    fig :       str or int, optional
        matplotlib figure name (figure will be cleared!)
    show_diff : bool, optional
        flag to specify if a difference curve should be shown
    show_legend: bool, optional
        flag to specify if a legend should be shown
    labelexp :  str
        plot label (legend entry) for the experimental data
    labelsim :  str
        plot label for the simulation data
    formatexp : str
        format specifier for the experimental data
    formatsim : str
        format specifier for the simulation curve

    Returns
    -------
    List of lines in the plot. Empty list in case matplotlib can't be imported
    """
    plot, plt = utilities.import_matplotlib_pyplot('XU.simpack')
    if not plot:
        return []
    if scale == 'sqrt':
        from ..mpl_helper import SqrtAllowNegScale  # noqa: F401

    f = plt.figure(fig, figsize=(10, 7))
    f.clf()
    ax = plt.subplot(111)
    if exp is not None:
        ax.plot(twotheta, exp, formatexp, label=labelexp)
    if mask is None:
        mask = numpy.ones_like(twotheta, dtype=bool)
    if isinstance(sim, PowderModel):
        simdata = sim.simulate(twotheta[mask])
        sim.plot(twotheta[mask], label=labelsim, formatspec=formatsim, ax=ax)
    else:
        simdata = sim
        ax.plot(twotheta[mask], simdata, formatsim, label=labelsim)

    if show_diff and exp is not None:
        # plot error between simulation and experiment
        ax.plot(twotheta[mask], exp[mask]-simdata, '.-', color='0.5',
                label='difference')

    ax.set_xlabel('2Theta (deg)')
    ax.set_ylabel('Intensity')
    lines = ax.get_lines()
    if show_legend:
        plt.figlegend(lines, [line.get_label() for line in lines],
                      loc='upper right', frameon=True)
    ax.set_yscale(scale)
    f.tight_layout()
    return lines
