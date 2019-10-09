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
# Copyright (C) 2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import matplotlib as mpl
import matplotlib.pyplot as plt
import xrayutilities as xu
from matplotlib.colors import LogNorm

# global setting for the experiment
sample = "rsm"  # sample name used also as file name for the data file

# substrate material used for Bragg peak calculation to correct for
# experimental offsets
Si = xu.materials.Si

hxrd = xu.HXRD(Si.Q(1, 1, 0), Si.Q(0, 0, 1))

#################################
# Si/SiGe (004) reciprocal space map
omalign = 34.3051  # experimental aligned values
ttalign = 69.1446
# nominal values of the substrate peak
[omnominal, dummy, dummy, ttnominal] = hxrd.Q2Ang(Si.Q(0, 0, 4))

# read the data from the xrdml files
om, tt, psd = xu.io.getxrdml_map(sample + '_%d.xrdml.bz2', 1,
                                 path='data')

# convert angular coordinates to reciprocal space + correct for offsets
[qx, qy, qz] = hxrd.Ang2Q(om, tt, delta=[omalign - omnominal,
                                         ttalign - ttnominal])

#############################################
# three different visualization possibilities
# plot the intensity as contour plot
plt.figure(figsize=(12, 6))
ax = []
for i in range(1, 4):
    ax.append(plt.subplot(1, 3, i))

MIN = 1
MAX = 3e5
plt.sca(ax[0])
plt.title('Gridder2D')
# data on a regular grid of 200x800 points
gridder = xu.Gridder2D(200, 300)
gridder(qy, qz, psd)
cf = plt.pcolormesh(gridder.xaxis, gridder.yaxis, gridder.data.T,
                    norm=LogNorm(MIN, MAX))

plt.sca(ax[1])
plt.title('FuzzyGridder2D')
# data on a regular grid with FuzzyGridding
gridder = xu.FuzzyGridder2D(200, 300)
gridder(qy, qz, psd, width=(0.0008, 0.0003))
cf = plt.pcolormesh(gridder.xaxis, gridder.yaxis, gridder.data.T,
                    norm=LogNorm(MIN, MAX))

plt.sca(ax[2])
plt.title('pcolormesh')
# using pcolor-variants
npixel = 255
qy.shape = (qy.size//npixel, npixel)
qz.shape = qy.shape
psd.shape = qy.shape
plt.pcolormesh(qy, qz, psd, norm=LogNorm(MIN, MAX))

for a in ax:
    plt.sca(a)
    plt.xlabel(r'$Q_{[110]}$ ($\mathrm{\AA}^{-1}$)')
    plt.ylabel(r'$Q_{[001]}$ ($\mathrm{\AA}^{-1}$)')
    plt.xlim(-0.13, 0.13)
    plt.ylim(4.538, 4.654)
    plt.xticks((-0.1, 0.0, 0.1))
plt.tight_layout()
