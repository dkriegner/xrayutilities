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
# Copyright (C) 2016-2022 Dominik Kriegner <dominik.kriegner@gmail.com>

import os

import lmfit
import matplotlib.pylab as pylab
import numpy

import xrayutilities as xu

# load experimental data
ai, edata, eps = numpy.loadtxt(os.path.join('data', 'xrr_data.txt'),
                               unpack=True)
ai /= 2.0

# define layers
# SiO2 / Ru(5) / CoFe(3) / IrMn(3) / AlOx(10)
lSiO2 = xu.simpack.Layer(xu.materials.SiO2, numpy.inf, roughness=2.5)
lRu = xu.simpack.Layer(xu.materials.Ru, 47, roughness=2.8)
rho_cf = 0.5*8900 + 0.5*7874
mat_cf = xu.materials.Amorphous('CoFe', rho_cf)
lCoFe = xu.simpack.Layer(mat_cf, 27, roughness=4.6)
lIrMn = xu.simpack.Layer(xu.materials.Ir20Mn80, 21, roughness=3.0)
lAl2O3 = xu.simpack.Layer(xu.materials.Al2O3, 100, roughness=5.5)

# create model
m = xu.simpack.SpecularReflectivityModel(lSiO2, lRu, lCoFe, lIrMn, lAl2O3,
                                         energy='CuKa1', resolution_width=0.02,
                                         sample_width=6, beam_width=0.25,
                                         background=81, I0=6.35e9)

# embed model in fit code
fitm = xu.simpack.FitModel(m, plot=True, verbose=True)

# set some parameter limitations
fitm.set_param_hint('SiO2_density', vary=False)
fitm.set_param_hint('Al2O3_density', min=0.8*xu.materials.Al2O3.density,
                    max=1.2*xu.materials.Al2O3.density)
p = fitm.make_params()
fitm.set_fit_limits(xmin=0.05, xmax=8.0)

# perform the fit
res = fitm.fit(edata, p, ai, weights=1/eps)
lmfit.report_fit(res, min_correl=0.5)

m.densityprofile(500, plot=True)
pylab.show()

# export the fit result for the full data range (Note that only data between
# xmin and xmax were actually optimized)
numpy.savetxt(
    "xrrfit.dat",
    numpy.vstack((ai, res.eval(res.params, x=ai))).T,
    header="incidence angle (deg), fitted intensity (arb. u.)",
)
