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
# Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>

import os

import matplotlib.pyplot as plt
import xrayutilities as xu

# global setting for the experiment
sample = "testnja"  # sample name used also as file name for the data file

hxrd = xu.HXRD((1, 1, 0), (0, 0, 1))

#################################
# read the data from the Seifert NJA files
om, tt, psd = xu.io.getSeifert_map(sample + '_%02d.nja', [3, 4], path="data")

# convert angular coordinates to reciprocal space + correct for offsets
[qx, qy, qz] = hxrd.Ang2Q(om, tt)

# calculate data on a regular grid of 200x201 points
gridder = xu.Gridder2D(200, 600)
gridder(qy, qz, psd)
INT = xu.maplog(gridder.data.transpose(), 6, 0)

# plot the intensity as contour plot
plt.figure()
plt.clf()
cf = plt.contourf(gridder.xaxis, gridder.yaxis, INT, 100, extend='min')
plt.xlabel(r'$Q_{[110]}$ ($\AA^{-1}$)')
plt.ylabel(r'$Q_{[001]}$ ($\AA^{-1}$)')
cb = plt.colorbar(cf)
cb.set_label(r"$\log($Int$)$ (cps)")
