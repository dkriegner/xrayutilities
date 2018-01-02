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

import os

import lmfit
import numpy
import xrayutilities as xu
from matplotlib.pylab import *

# load experimental data
ai, edata, eps = numpy.loadtxt(os.path.join('data', 'xrr_data.txt'),
                               unpack=True)
ai /= 2.0

# define layers
# SiO2 / Ru(5) / CoFe(3) / IrMn(3) / AlOx(10)
lSiO2 = xu.simpack.Layer(xu.materials.SiO2, inf)
lRu = xu.simpack.Layer(xu.materials.Ru, 50)
rho_cf = 0.5*8900 + 0.5*7874
mat_cf = xu.materials.Amorphous('CoFe', rho_cf)
lCoFe = xu.simpack.Layer(mat_cf, 30)
lIrMn = xu.simpack.Layer(xu.materials.Ir20Mn80, 30)
lAl2O3 = xu.simpack.Layer(xu.materials.Al2O3, 100)

m = xu.simpack.SpecularReflectivityModel(lSiO2, lRu, lCoFe, lIrMn, lAl2O3,
                                         energy='CuKa1')

p = lmfit.Parameters()
#          (Name,                  Value,  Vary,   Min,  Max, Expr)
p.add_many(('SiO2_thickness',  numpy.inf, False,  None, None, None),
           ('SiO2_roughness',        2.5,  True,     0,    8, None),
           ('Ru_thickness',         47.0,  True,    25,   70, None),
           ('Ru_roughness',          2.8,  True,     0,    8, None),
           ('Ru_density',            1.0,  True,   0.8,  1.0, None),
           ('CoFe_thickness',       27.0,  True,    15,   50, None),
           ('CoFe_roughness',        4.6,  True,     0,    8, None),
           ('CoFe_density',          1.0,  True,   0.8,  1.2, None),
           ('Ir20Mn80_thickness',   21.0,  True,    15,   40, None),
           ('Ir20Mn80_roughness',    3.0,  True,     0,    8, None),
           ('Ir20Mn80_density',      1.1,  True,   0.8,  1.2, None),
           ('Al2O3_thickness',     100.0,  True,    70,  130, None),
           ('Al2O3_roughness',       5.5,  True,     0,    8, None),
           ('Al2O3_density',         1.0,  True,   0.8,  1.2, None),
           # primary beam intensity
           ('I0',                 6.75e9,  True,   3e9,  8e9, None),
           # background level of the measurement
           ('background',             81,  True,    40,  100, None),
           # zero shift of the incidence angle
           ('shift',                   0,  True, -0.02, 0.02, None),
           # size of the sample along the beam propagation direction
           ('sample_width',          6.0, False,     2,    8, None),
           # width of the beam in the reflection plane
           ('beam_width',           0.25, False,   0.2,  0.4, None),
           ('resolution_width',     0.02, False,  0.01, 0.05, None))

res = xu.simpack.fit_xrr(m, p, ai, data=edata, eps=eps, xmin=0.05, xmax=8.0,
                         plot=True, verbose=True)
lmfit.report_fit(res, min_correl=0.5)

m.densityprofile(500, plot=True)
show()
