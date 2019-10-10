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

import xrayutilities as xu
from matplotlib.pylab import *

mpl.rcParams['font.size'] = 16.0


en = 8500  # eV
resol = 0.0004  # resolution in q
H, K, L = (1, 1, 1)
qz = linspace(1.8, 2.2, 5000)
Si = xu.materials.Si
hxrd = xu.HXRD(Si.Q(1, 1, -2), Si.Q(1, 1, 1), en=en)

sub = xu.simpack.Layer(Si, inf)
lay = xu.simpack.Layer(xu.materials.SiGe(0.6), 150, relaxation=0.5)
pls = xu.simpack.PseudomorphicStack111('pseudo', sub, lay)

# calculate incidence angle for dynamical diffraction models
qx = hxrd.Transform(Si.Q(H, K, L))[1]
ai = xu.simpack.coplanar_alphai(qx, qz, en)
resolai = abs(xu.simpack.coplanar_alphai(qx, mean(qz) + resol, en) -
              xu.simpack.coplanar_alphai(qx, mean(qz), en))

# comparison of different diffraction models
# simplest kinematical diffraction model
mk = xu.simpack.KinematicalModel(pls, experiment=hxrd, resolution_width=resol)
Ikin = mk.simulate(qz, hkl=(H, K, L), refraction=True)

# simplified dynamical diffraction model
mds = xu.simpack.SimpleDynamicalCoplanarModel(pls, experiment=hxrd,
                                              resolution_width=resolai)
Idynsub = mds.simulate(ai, hkl=(H, K, L), idxref=0)
Idynlay = mds.simulate(ai, hkl=(H, K, L), idxref=1)

# general 2-beam theory based dynamical diffraction model
md = xu.simpack.DynamicalModel(pls, experiment=hxrd, resolution_width=resolai)
Idyn = md.simulate(ai, hkl=(H, K, L))

# plot of calculated intensities
figure('XU-simpack SiGe(111)')
clf()
semilogy(qz, Ikin, label='kinematical')
semilogy(xu.simpack.get_qz(qx, ai, en), Idynsub, label='simpl. dynamical(S)')
semilogy(xu.simpack.get_qz(qx, ai, en), Idynlay, label='simpl. dynamical(L)')
semilogy(xu.simpack.get_qz(qx, ai, en), Idyn, label='full dynamical')
vlines([xu.math.VecNorm(lay.material.Q(H, K, L)) for lay in pls], 1e-9, 1,
       linestyles='dashed')
legend(fontsize='small')
xlim(qz.min(), qz.max())
xlabel(r'Qz ($1/\mathrm{\AA}$)')
ylabel('Intensity (arb. u.)')
tight_layout()
show()
