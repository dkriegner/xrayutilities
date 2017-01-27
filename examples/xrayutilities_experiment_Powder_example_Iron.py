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
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

cryst_size = 40e-9  # meter

# create Fe BCC (space group nr. 229 Im3m) with a=2.87Angstrom
# although this is already predefined as xu.materials.Fe we will repeat here
# for educational purposes
FeBCC = xu.materials.Crystal(
    "Fe", xu.materials.SGLattice(229, 2.87, atoms=[xu.materials.elements.Fe, ],
                                 pos=['2a', ]))

print("Creating Fe powder ...")
Fe_powder = xu.simpack.Powder(FeBCC, 1, crystallite_size_gauss=cryst_size)
pd = xu.simpack.PowderDiffraction(Fe_powder)
tt = numpy.arange(5, 120, 0.01)
inte = pd.Calculate(tt)

print(pd)

# to create a mixed powder sample one would use
# Co_powder = xu.simpack.Powder(xu.materials.Co, 5)  # 5 times more Co
# pm = xu.simpack.PowderModel(Fe_powder + Co_powder, I0=100)
# inte = pm.simulate(tt)

plt.figure()
ax = plt.subplot(111)
plt.xlabel(r"$2\theta$ (deg)")
plt.ylabel(r"Intensity")
plt.plot(tt, inte, 'k-', label='Fe')
divider = make_axes_locatable(ax)

bax = divider.append_axes("top", size="10%", pad=0.05, sharex=ax)
plt.bar(pd.ang * 2, numpy.ones_like(pd.data), width=0, linewidth=2,
        color='r', align='center', orientation='vertical')
for x, hkl in zip(pd.ang*2, pd.hkl):
    h, k, l = hkl
    plt.text(x, 0.1, '%d%d%d' % (h, k, l))
plt.setp(bax.get_xticklabels(), visible=False)
plt.setp(bax.get_yticklabels(), visible=False)

ax.set_xlim(5, 120)
