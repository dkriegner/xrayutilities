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
# Copyright (C) 2018 Dominik Kriegner <dominik.kriegner@gmail.com>

import math
from math import pi

import numpy

from .. import config, utilities
from ..math import VecNorm


def show_reciprocal_space_plane(
        mat, exp, ttmax=None, maxqout=0.01, scalef=100, ax=None, color=None,
        show_Laue=True, show_legend=True, projection='perpendicular',
        label=None):
    """
    show a plot of the coplanar diffraction plane with peak positions for the
    respective material. the size of the spots is scaled with the strength of
    the structure factor

    Parameters
    ----------
    mat:        Crystal
        instance of Crystal for structure factor calculations
    exp:        Experiment
        instance of Experiment (likely HXRD, or FourC). defines the inplane and
        out of plane direction as well as the sample azimuth
    ttmax:      float, optional
        maximal 2Theta angle to consider, by default 180deg
    maxqout:    float, optional
        maximal out of plane q for plotted Bragg peaks as fraction of exp.k0
    scalef:     float, optional
        scale factor for the marker size
    ax:         matplotlib.Axes, optional
        matplotlib Axes to use for the plot, useful if multiple materials
        should be plotted in one plot
    color:      matplotlib color, optional
    show_Laue:  bool, optional
        flag to indicate if the Laue zones should be indicated
    show_legend:    bool, optional
        flag to indiate if a legend should be shown
    projection: 'perpendicular', 'polar', optional
        type of projection for Bragg peaks which do not fall into the
        diffraction plane. 'perpendicular' (default) uses only the inplane
        component in the scattering plane, whereas 'polar' uses the vectorial
        absolute value of the two inplane components. See also the 'maxqout'
        option.
    label:  None or str, optional
        label to be used for the legend. If 'None' the name of the material
        will be used.

    Returns
    -------
    Axes, plot_handle
    """
    def get_peaks(mat, exp, ttmax):
        """
        Parameters
        ----------
        mat:        Crystal
            instance of Crystal for structure factor calculations
        exp:        Experiment
            instance of Experiment (likely HXRD, or FourC)
        tt_cutoff:  float
            maximal 2Theta angle to consider, by default 180deg

        Returns
        -------
        ndarray
            data array with columns for 'q', 'qvec', 'hkl', 'r' for the Bragg
            peaks
        """
        # calculate maximal Bragg indices
        hma = int(math.ceil(VecNorm(mat.a1) * exp.k0 / pi *
                            math.sin(math.radians(ttmax / 2.))))
        hmi = -hma
        kma = int(math.ceil(VecNorm(mat.a2) * exp.k0 / pi *
                            math.sin(math.radians(ttmax / 2.))))
        kmi = -kma
        lma = int(math.ceil(VecNorm(mat.a3) * exp.k0 / pi *
                            math.sin(math.radians(ttmax / 2.))))
        lmi = -lma

        # calculate structure factors
        qmax = 2 * exp.k0 * math.sin(math.radians(ttmax/2.))
        hkl = numpy.mgrid[hma:hmi-1:-1,
                          kma:kmi-1:-1,
                          lma:lmi-1:-1].reshape(3, -1).T
        q = mat.Q(hkl)
        qnorm = VecNorm(q)
        m = qnorm < qmax

        data = numpy.zeros(numpy.sum(m), dtype=[('q', numpy.double),
                                                ('qvec', numpy.ndarray),
                                                ('r', numpy.double),
                                                ('hkl', numpy.ndarray)])
        data['q'] = qnorm[m]
        data['qvec'] = list(exp.Transform(q[m]))
        rref = abs(mat.StructureFactor((0, 0, 0), exp.energy)) ** 2
        data['r'] = numpy.abs(mat.StructureFactorForQ(q[m], exp.energy)) ** 2
        data['r'] /= rref
        data['hkl'] = list(hkl[m])

        return data

    plot, plt = utilities.import_matplotlib_pyplot('XU.materials')

    if not plot:
        print('matplotlib needed for show_reciprocal_space_plane')
        return

    if ttmax is None:
        ttmax = 180

    d = get_peaks(mat, exp, ttmax)
    k0 = exp.k0

    if not ax:
        fig = plt.figure(figsize=(9, 5))
        ax = plt.subplot(111)
    else:
        fig = ax.get_figure()
        plt.sca(ax)

    plt.axis('scaled')
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    plt.xlim(-2.05*k0, 2.05*k0)
    plt.ylim(-0.05*k0, 2.05*k0)

    if show_Laue:
        c = plt.matplotlib.patches.Circle((0, 0), 2*k0, facecolor='#FF9180',
                                          edgecolor='none')
        ax.add_patch(c)
        qmax = 2 * k0 * math.sin(math.radians(ttmax/2.))
        c = plt.matplotlib.patches.Circle((0, 0), qmax, facecolor='#FFFFFF',
                                          edgecolor='none')
        ax.add_patch(c)

        c = plt.matplotlib.patches.Circle((0, 0), 2*k0, facecolor='none',
                                          edgecolor='0.5')
        ax.add_patch(c)
        c = plt.matplotlib.patches.Circle((k0, 0), k0, facecolor='none',
                                          edgecolor='0.5')
        ax.add_patch(c)
        c = plt.matplotlib.patches.Circle((-k0, 0), k0, facecolor='none',
                                          edgecolor='0.5')
        ax.add_patch(c)
        plt.hlines(0, -2*k0, 2*k0, color='0.5', lw=0.5)
        plt.vlines(0, -2*k0, 2*k0, color='0.5', lw=0.5)

    # generate mask for plotting
    m = numpy.zeros_like(d, dtype=numpy.bool)
    for i, (q, r) in enumerate(zip(d['qvec'], d['r'])):
        if (abs(q[0]) < maxqout*k0 and r > config.EPSILON):
            m[i] = True

    x = numpy.empty_like(d['r'][m])
    y = numpy.empty_like(d['r'][m])
    s = numpy.empty_like(d['r'][m])
    for i, (qv, r) in enumerate(zip(d['qvec'][m], d['r'][m])):
        if projection == 'perpendicular':
            x[i] = qv[1]
        else:
            x[i] = numpy.sign(qv[1])*numpy.sqrt(qv[0]**2 + qv[1]**2)
        y[i] = qv[2]
        s[i] = r*scalef
    label = label if label else mat.name
    h = plt.scatter(x, y, s=s, zorder=2, label=label)
    if color:
        h.set_color(color)

    plt.xlabel(r'$Q$ inplane ($\AA^{-1}$)')
    plt.ylabel(r'$Q$ out of plane ($\AA^{-1}$)')

    if show_legend:
        if len(fig.legends) == 1:
            fig.legends[0].remove()
        fig.legend(*ax.get_legend_handles_labels(), loc='upper right')
    plt.tight_layout()

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = h.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}\n{}".format(mat.name,
                               str(d['hkl'][m][ind['ind'][0]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(h.get_facecolor()[0])
        annot.get_bbox_patch().set_alpha(0.2)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = h.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    def click(event):
        if event.inaxes == ax:
            cont, ind = h.contains(event)
            if cont:
                i = ind['ind'][0]
                popts = numpy.get_printoptions()
                numpy.set_printoptions(precision=4, suppress=True)
                angles = exp.Q2Ang(d['qvec'][m][ind['ind'][0]], trans=False,
                                   geometry='real')
                text = "{}\nhkl: {}\nangles: {}".format(
                    mat.name, str(d['hkl'][m][ind['ind'][0]]), str(angles))
                numpy.set_printoptions(**popts)
                print(text)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", click)

    return ax, h
