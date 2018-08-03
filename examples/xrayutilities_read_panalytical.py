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
# Copyright (C) 2012, 2018 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2012 Tanja Etzelstorfer <tanja.etzelstorfer@jku.at>

import matplotlib as mpl
import matplotlib.pyplot as plt
import xrayutilities as xu

# plot settings for matplotlib
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 20.0
mpl.rcParams['axes.labelsize'] = 'large'

# global setting for the experiment
sample = "rsm"  # sample name used also as file name for the data file

# substrate material used for Bragg peak calculation to correct for
# experimental offsets
Si = xu.materials.Si
Ge = xu.materials.Ge
SiGe = xu.materials.SiGe(1)  # parameter x_Ge = 1

hxrd = xu.HXRD(Si.Q(1, 1, 0), Si.Q(0, 0, 1))

#################################
# Si/SiGe (004) reciprocal space map
omalign = 34.3046  # experimental aligned values
ttalign = 69.1283
# nominal values of the substrate peak
[omnominal, dummy, dummy, ttnominal] = hxrd.Q2Ang(Si.Q(0, 0, 4))

# read the data from the xrdml files
om, tt, psd = xu.io.getxrdml_map(sample + '_%d.xrdml.bz2', [1, 2, 3, 4, 5],
                                 path='data')

# determine offset of substrate peak from experimental values (optional)
omalign, ttalign, p, cov = xu.analysis.fit_bragg_peak(
    om, tt, psd, omalign, ttalign, hxrd, plot=False)

# convert angular coordinates to reciprocal space + correct for offsets
qx, qy, qz = hxrd.Ang2Q(om, tt, delta=[omalign - omnominal,
                                         ttalign - ttnominal])

# calculate data on a regular grid of 200x201 points
gridder = xu.Gridder2D(200, 600)
gridder(qy, qz, psd)
INT = xu.maplog(gridder.data.transpose(), 6, 0)

# plot the intensity as contour plot
plt.figure()
cf = plt.contourf(gridder.xaxis, gridder.yaxis, INT, 100, extend='min')
plt.xlabel(r'$Q_{[110]}$ ($\AA^{-1}$)')
plt.ylabel(r'$Q_{[001]}$ ($\AA^{-1}$)')
cb = plt.colorbar(cf)
cb.set_label(r"$\log($Int$)$ (cps)")

tr = SiGe.RelaxationTriangle([0, 0, 4], Si, hxrd)
plt.plot(tr[0], tr[1], 'ko')
plt.tight_layout()

# line cut with integration along 2theta to remove beam footprint broadening
qzc, qzint, cmask = xu.analysis.get_radial_scan([qy, qz], psd, [0, 4.5],
                                                1001, 0.155, intdir='2theta')

# show used data on the reciprocal space map
plt.tricontour(qy, qz, cmask, (0.999,), colors='r')

# plot line cut
plt.figure()
plt.semilogy(qzc, qzint)
plt.xlabel(r'scattering angle (deg)')
plt.ylabel(r'intensity (arb. u.)')
plt.legend()
plt.tight_layout()
