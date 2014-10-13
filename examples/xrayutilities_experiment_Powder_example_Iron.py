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
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import numpy
# import matplotlib.pyplot as plt

# en = (2*8048 + 8028)/3.
en = 'CuKa1'
peak_width = 2 * numpy.pi / 100.
resolution = peak_width / 10.

# create Fe BCC with a=2.87Angstrom
FeBCC = xu.materials.Material(
    "Fe", xu.materials.BCCLattice(xu.materials.elements.Fe, 2.87))

print("Creating Fe powder ...")
Fe_powder = xu.Powder(FeBCC, en=en)
Fe_powder.PowderIntensity()
Fe_th, Fe_int = Fe_powder.Convolute(resolution, peak_width)

print(Fe_powder)

# plt.figure(1)
# plt.clf()
# ax1 = plt.subplot(111)
# plt.xlabel(r"$2\theta$ (deg)")
# plt.ylabel(r"Intensity")
# plt.semilogy(Fe_th*2,Fe_int/Fe_int.max(),'k-',linewidth=2.)
# plt.plot(Fe_th*2,Fe_int/Fe_int.max(),'k-',linewidth=2.)
# ax1.set_xlim(0,160)
# ax1.set_ylim(0.001,1.2)
# plt.grid()
