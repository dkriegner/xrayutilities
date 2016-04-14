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

from matplotlib.pylab import *
import xrayutilities as xu
import time
mpl.rcParams['font.size'] = 16.0

en = 9000  # eV
lam = xu.en2lam(en)
resol = 2*pi/5000  # resolution in q

sub = xu.simpack.Layer(xu.materials.Si, inf)
buf1 = xu.simpack.Layer(xu.materials.SiGe(0.5), 4995.10, relaxation=1.0)
buf2 = xu.simpack.Layer(xu.materials.SiGe(0.8), 4997.02, relaxation=1.0)
lay1 = xu.simpack.Layer(xu.materials.SiGe(0.6), 49.73, relaxation=0.0)
lay2 = xu.simpack.Layer(xu.materials.SiGe(1.0), 45.57, relaxation=0.0)
# create superlattice stack
# lattice param. are adjusted to the relaxation parameter of the layers
# Note that to create a superlattice you can use summation and multiplication
pls = xu.simpack.PseudomorphicStack001('SL 5/5', sub+buf1+buf2+5*(lay1+lay2))

qz = linspace(4.0, 5.0, 5e3)
ai = degrees(arcsin(lam * qz / (4 * pi)))
resolai = degrees(arcsin(lam * (mean(qz) + 2*pi/5000) / (4 * pi)) -
                  arcsin(lam * mean(qz) / (4 * pi)))
# comparison of different diffraction models
# simplest kinematical diffraction model
t0 = time.time()
mk = xu.simpack.KinematicalModel(pls, energy=en, resolution_width=resol)
Ikin = mk.simulate(qz, hkl=(0, 0, 4))
t1 = time.time()
print("%.3f sec for kinematical calculation"% (t1-t0))

# dynamical diffraction model for substrate and kinematical for the layer(s)
t0 = time.time()
mk = xu.simpack.KinematicalMultiBeamModel(pls, energy=en,
                                          surface_hkl=(0, 0, 1),
                                          resolution_width=resol)
Imult = mk.simulate(qz, hkl=(0, 0, 4), refraction=True)
t1 = time.time()
print("%.3f sec for kinematical multibeam calculation"% (t1-t0))

# general 2-beam theory based dynamical diffraction model
t0 = time.time()
md = xu.simpack.DynamicalModel(pls, energy=en, resolution_width=resolai)
Idyn = md.simulate(ai, hkl=(0, 0, 4))
t1 = time.time()
print("%.3f sec for acurate dynamical calculation"% (t1-t0))

# plot of calculated intensities
figure('XU-simpack SiGe')
clf()
semilogy(qz, Ikin, label='kinematical')
semilogy(qz, Imult, label='multibeam')
semilogy(qz, Idyn, label='full dynamical')
vlines([4*2*pi/l.material.a3[-1] for l in pls], 1e-9, 1, linestyles='dashed')
legend(fontsize='small')
xlim(qz.min(), qz.max())
xlabel('Qz ($1/\AA$)')
ylabel('Intensity (arb.u.)')
tight_layout()
show()
