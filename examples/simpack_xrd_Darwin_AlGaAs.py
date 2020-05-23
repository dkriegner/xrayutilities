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
# Copyright (C) 2016-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
from matplotlib.pylab import *
from scipy.special import erf

mpl.rcParams['font.size'] = 16.0
en = 'CuKa1'
qz = linspace(4.3, 4.6, 4000)

GaAs = xu.materials.GaAs
exp = xu.HXRD(GaAs.Q(1, 1, 0), GaAs.Q(0, 0, 1), en=en)
dm = xu.simpack.DarwinModelAlGaAs001(
    qz, experiment=exp, resolution_width=0.0005, I0=2e7, background=1e0)


def period(xavg, xwell, wellratio, thick, sigmaup, sigmadown):
    """
    chemical composition profile in the superlattice
    """
    twell = wellratio*thick
    xb = (xavg*thick - twell*xwell)/(thick - twell)
    tb = (thick-twell)/2
    print('using barrier Al-content, well thickness, barrier thickness/2: '
          '%.3f %.1f %.1f' % (xb, twell, tb))
    return lambda z: xb + (xwell-xb)*(erf((z-tb)/sigmaup) +
                                      erf(-(z-(tb+twell))/sigmadown))/2.


sideal = [{'t': 3500000, 'x': 0, 'r': 1},  # 350um substrate
          {'t': 10000, 'x': 0.5, 'r': 1},  # 1um buffer (relaxed)
          # 5 period superlattice with interdiffusion
          (5, [{'t': 75, 'x': 0.9, 'r': 0},
               {'t': 150, 'x': 0.0, 'r': 0},
               {'t': 75, 'x': 0.9, 'r': 0}]),
          {'t': 100, 'x': 0, 'r': 0}]  # 10nm cap
sample = [{'t': 3500000, 'x': 0, 'r': 1},  # 350um substrate
          {'t': 10000, 'x': 0.5, 'r': 1},  # 1um buffer (relaxed)
          # 5 period superlattice with interdiffusion
          (5, [{'t': 300, 'x': period(0.45, 0, 0.5, 300, 20, 20), 'r': 0.0}]),
          {'t': 100, 'x': 0, 'r': 0}]  # 10nm cap

# perform Darwin-theory based simulation
mlideal = dm.make_monolayers(sideal)
Iideal = dm.simulate(mlideal)
ml = dm.make_monolayers(sample)
Isim = dm.simulate(ml)

figure('XU-simpack AlGaAs (Darwin)', figsize=(10, 5))
clf()

subplot(121)
semilogy(qz, Iideal, '-m', lw=2, label='ideal')
semilogy(qz, Isim, '-r', lw=2, label='SL5')
ylim(0.5*dm.background, dm.I0)
xlim(qz.min(), qz.max())
plt.locator_params(axis='x', nbins=5)
xlabel(r'Qz ($1/\mathrm{\AA}$)')
ylabel('Intensity (arb. u.)')
legend(fontsize='small')

subplot(122)
z, xAl = dm.prop_profile(mlideal, 'x')
plot(z/10, xAl, '-m')
z, xAl = dm.prop_profile(ml, 'x')
plot(z/10, xAl, '-r')
xlabel('depth z (nm)')
ylabel('Al-content', color='m')
ylim(-0.05, 1.05)
twinx()
z, ai = dm.prop_profile(ml, 'ai')
plot(z/10, ai, '-b')
xlim(-1200, 3)
ylim(5.64, 5.68)
plt.locator_params(axis='x', nbins=5)
ylabel(r'a-inplane ($\mathrm{\AA}$)', color='b')
tight_layout()
show()
