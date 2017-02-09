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
# Copyright (C) 2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import os

import lmfit
import numpy
from matplotlib.pylab import *
import xrayutilities as xu

mpl.rcParams['font.size'] = 16.0
mpl.rcParams['lines.linewidth'] = 2.0

###############################
# load example data
tt, det, sig = loadtxt(os.path.join('data', 'LaB6_d500_si_psd.xye.bz2'),
                       unpack=True)

##############################
La = xu.materials.elements.La
B = xu.materials.elements.B
LaB6 = xu.materials.Crystal(
    "LaB6", xu.materials.SGLattice(221, 4.15692, atoms=[La, B],
                                   pos=['1a', ('6f', 0.19750)],
                                   b=[0.05, 0.15]))

LaB6_powder = xu.simpack.Powder(LaB6, 1,
                                crystallite_size_gauss=1e6,
                                crystallite_size_lor=0.5e-6,
                                strain_gauss=0,
                                strain_lor=0)

settings = {'classoptions': {'oversampling': 10},
            'global': {'diffractometer_radius': 0.337,
                       'equatorial_divergence_deg': 0.40},
            'tube_tails': {'tail_left': -0.001,
                           'main_width': 0.00015,
                           'tail_right': 0.001,
                           'tail_intens': 0.0015},
            'axial': {'angI_deg': 2.0, 'angD_deg': 2.0,
                      'slit_length_target': 0.008,
                      'n_integral_points': 21,
                      'length_sample': 0.015,
                      'slit_length_source': 0.008001},
            'si_psd': {'si_psd_window_bounds': (0, 32e-3)},
            'absorption': {'sample_thickness': 500e-6,
                           'absorption_coefficient': 3e4},
            'displacement': {'specimen_displacement': -3.8e-5,
                             'zero_error_deg': 0.0},
            'emission': {'emiss_intensities': (1.0, 0.45)}}

pm = xu.simpack.PowderModel(LaB6_powder, I0=1.10e6, fpsettings=settings)

##############################
# define background
btt, bint = asarray([(15.158, 1136.452),
                     (17.886, 841.925),
                     (22.906, 645.784),
                     (26.556, 551.663),
                     (34.554, 401.219),
                     (45.764, 260.595),
                     (58.365, 171.993),
                     (81.950, 112.838),
                     (92.370, 101.276),
                     (106.441, 102.486),
                     (126.624, 112.838),
                     (139.096, 132.063),
                     (146.240, 136.500),
                     (152.022, 157.204)]).T

pm.set_background('spline', x=btt, y=bint)
mask = numpy.logical_and(tt > 18, tt < 148)

##############################
# first fit run to optimize sample displacement and zero offset
p = pm.create_fitparameters()
for pn, limit in (('primary_beam_intensity', (None, None)),
                  ('displacement_specimen_displacement', (-1e-4, 1e-4)),
                  ('displacement_zero_error_deg', (-0.01, 0.01))):
    p[pn].set(vary=True, min=limit[0], max=limit[1])

fitres1 = pm.fit(p, tt[mask], det[mask], std=sig[mask], maxfev=50)

##############################
# second fit run to optimize absorption
p = pm.create_fitparameters()
for pn, limit in (('primary_beam_intensity', (None, None)),
                  ('displacement_specimen_displacement', (-1e-4, 1e-4)),
                  ('absorption_absorption_coefficient', (1e4, 10e4)),
                  ('emission_emiss_intensities_item_1', (0.4, 0.52))):
    p[pn].set(vary=True, min=limit[0], max=limit[1])

# if needed one can also set relations between parameters:
# p['axial_angD_deg'].set(expr='axial_angI_deg')

fitres2 = pm.fit(p, tt[mask], det[mask], std=sig[mask])

##############################
# final calculation and plotting/printing of the results
sim = pm.simulate(tt[mask])

xu.simpack.plot_powder(tt, det, sim, scale='sqrt', mask=mask)
xu.simpack.Rietveld_error_metrics(det[mask], sim, std=sig[mask],
                                  Nvar=fitres2.nvarys, disp=True)

lmfit.report_fit(fitres2)
