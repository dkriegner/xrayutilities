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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import matplotlib.pyplot as plt
import numpy
import xrayutilities as xu
import os

# load powder diffraction cif data file
# here use e.g. cif-file from ICDD
pd = xu.io.pdCIF(os.path.join('data', '04-003-0996.cif'))
# plot data
plt.figure()
plt.semilogy(
    pd.data['_pd_meas_2theta_scan'],
    pd.data['_pd_meas_intensity_total'])
plt.xlim(0, 90)
plt.ylim(1e-1, 1.5e3)
plt.ylabel('Intensity Sb$_2$Te$_3$ (arb.u.)')
plt.xlabel('scattering angle (deg)')

# load materials
# structure cif from Pearson's crystal data database
st = xu.materials.Crystal.fromCIF(os.path.join('data', '1216385.cif'))
pst = xu.simpack.PowderDiffraction(st, tt_cutoff=90)

height = 0.05
# peak positions of Sb2Te3
mask = pst.data / pst.data.max() > 0.001
plt.bar(pst.ang[mask] * 2, numpy.ones(mask.sum()) * height, width=0.4,
        bottom=0.13, linewidth=0, color='r', align='center',
        orientation='vertical')
