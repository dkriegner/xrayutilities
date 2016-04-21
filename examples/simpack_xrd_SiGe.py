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
mpl.rcParams['font.size'] = 16.0


def alpha_i(qx, qz):
    th = arcsin(sqrt(qx**2 + qz**2) / (4 * pi) * xu.en2lam(en))
    return degrees(arctan2(qx, qz) + th)

en = 8500  # eV
resol = 0.0004  # resolution in q
h, k, l = (0, 0, 4)
qz = linspace(4.2, 5.0, 3e3)

sub = xu.simpack.Layer(xu.materials.Si, inf)
lay = xu.simpack.Layer(xu.materials.SiGe(0.6), 145.87, relaxation=0.5)
# pseudomorphic stack -> adjusts lattice parameters!
pls = xu.simpack.PseudomorphicStack001('pseudo', sub, lay)

# calculate incidence angle for dynamical diffraction models
qx = sqrt(sub.material.Q(h, k, l)[0]**2 + sub.material.Q(h, k, l)[1]**2)
ai = alpha_i(qx, qz)
resolai = abs(alpha_i(qx, mean(qz) + resol) - alpha_i(qx, mean(qz)))

# comparison of different diffraction models
# simplest kinematical diffraction model
mk = xu.simpack.KinematicalModel(pls, energy=en, resolution_width=resol)
Ikin = mk.simulate(qz, hkl=(h, k, l), refraction=True)

# kinematic multibeam diffraction model
mk = xu.simpack.KinematicalMultiBeamModel(pls, energy=en,
                                          surface_hkl=(0, 0, 1),
                                          resolution_width=resol)
Imult = mk.simulate(qz, hkl=(h, k, l), refraction=True)

# simplified dynamical diffraction model
mds = xu.simpack.SimpleDynamicalCoplanarModel(pls, energy=en,
                                              resolution_width=resolai)
Idynsub = mds.simulate(ai, hkl=(h, k, l), idxref=0)
Idynlay = mds.simulate(ai, hkl=(h, k, l), idxref=1)

# general 2-beam theory based dynamical diffraction model
md = xu.simpack.DynamicalModel(pls, energy=en, resolution_width=resolai)
Idyn = md.simulate(ai, hkl=(h, k, l))

# plot of calculated intensities
figure('XU-simpack SiGe')
clf()
semilogy(qz, Ikin, label='kinematical')
semilogy(qz, Imult, label='multibeam')
semilogy(qz, Idynsub, label='simpl. dynamical(S)')
semilogy(qz, Idynlay, label='simpl. dynamical(L)')
semilogy(qz, Idyn, label='full dynamical')
vlines([4*2*pi/l.material.a3[-1] for l in pls], 1e-9, 1, linestyles='dashed')
legend(fontsize='small')
xlim(qz.min(), qz.max())
xlabel('Qz ($1/\AA$)')
ylabel('Intensity (arb.u.)')
tight_layout()
show()
