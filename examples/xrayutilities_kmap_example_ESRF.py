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
# Copyright (C) 2014 Raphael Grifone <raphael.grifone@esrf.fr>
# Copyright (C) 2014,2018 Dominik Kriegner <dominik.kriegner@gmail.com>

import os

import xrayutilities as xu
from matplotlib.pylab import *

import xrayutilities_id01_functions as id01

# 3S+2D goniometer (ID01 goniometer, sample mu, eta, phi detector
# nu, del, mpxy, mpxz
# convention for coordinate system: x downstream; z upwards; y to the
# "outside" (righthanded)
# QConversion will set up the goniometer geometry. So the first argument
# describes the sample rotations, the second the detector rotations and the
# third the primary beam direction.
# For this consider the following right handed coordinate system (feel free to
# use your conventions):
# x: downstream (direction of primary beam)
# y: out of the ring
# z: upwards
# The outer most sample rotation (so the one mounted on the floor) is one
# which turns left-handed (-) around the z-direction -> z- (mu)
# The second sample rotation ('eta') is lefthanded (-) around y -> y-
qconv = xu.experiment.QConversion(['z-', 'y-', 'z-'],
                                  ['z-', 'y-', 'ty', 'tz'],
                                  [1, 0, 0])
hxrd = xu.HXRD([1, 1, 0], [0, 0, 1], qconv=qconv, sampleor='z+')
hxrd._A2QConversion.init_area('z-', 'y+', cch1=333.94, cch2=235.62, Nch1=516,
                              Nch2=516, pwidth1=5.5000e-02, pwidth2=5.5000e-02,
                              distance=0.53588*1000, detrot=-1.495,
                              tiltazimuth=155.0, tilt=0.745, Nav=(2, 2))

#############################################################
# load spec-file data
fnames = ('KMAP_2017_fast_00001.spec',)

scannrs = []
for fn in fnames:
    sf = xu.io.SPECFile(fn, path=datadir)
    scannrs.append(list(range(len(sf))))

nx, ny = 150, 150

# specfile, scannumbers, nx,ny, motornames, optional column names (ID01
# values are default)
fss = xu.io.FastScanSeries(fnames, scannrs, nx, ny, 'mu', 'eta', 'phi', 'nu',
                           'del', 'mpxy', 'mpxz', xmotor='adcY', ymotor='adcX',
                           ccdnr='imgnr', path=datadir)

#############################################################
# 3D RSM from summing over X,Y
# now all EDF files are parsed, this will take a while and some memory
qconv.energy = id01.getmono_energy(sf[0])
xu.config.VERBOSITY = 2
g3d = fss.get_average_RSM(81, 82, 83, qconv, datadir=id01.datadir,
                          replacedir=id01.repl_n, roi=None, nav=(4, 4),
                          filterfunc=deadpixelkill)
xu.config.VERBOSITY = 1
numpy.savez_compressed('RSM3D.npz', qx=g3d.xaxis, qy=g3d.yaxis, qz=g3d.zaxis,
                       data=g3d.data)


figure()
subplot(221)
pcolormesh(qx, qy, data.sum(axis=2).T, norm=mpl.colors.LogNorm())
xlabel(r'Qx ($\mathrm{\AA}^{-1}$)')
ylabel(r'Qy ($\mathrm{\AA}^{-1}$)')

subplot(222)
pcolormesh(qx, qz, data.sum(axis=1).T, norm=mpl.colors.LogNorm())
xlabel(r'Qx ($\mathrm{\AA}^{-1}$)')
ylabel(r'Qz ($\mathrm{\AA}^{-1}$)')

subplot(223)
pcolormesh(qy, qz, data.sum(axis=0).T, norm=mpl.colors.LogNorm())
xlabel(r'Qy ($\mathrm{\AA}^{-1}$)')
ylabel(r'Qz ($\mathrm{\AA}^{-1}$)')
tight_layout()

#############################################################
# 2D real space maps for selected region in Q-space
qr = [0.57, 0.62, -0.20, -0.16, 3.47, 3.50]  # [xmin, xmax, ymin, ..., zmax]
x, y, data = fss.get_sxrd_for_qrange(qr, qconv, datadir=id01.datadir,
                                     replacedir=id01.repl_n)
numpy.savez_compressed('output_sxrd_map.npz', x=x, y=y, data=data)


figure()
lev_exp = np.linspace(np.log10(data.min()),
                      np.log10(data.max()), 100)
levs = np.power(10, lev_exp)
tricontourf(y, x, data, levs, norm=mpl.colors.LogNorm())
axis('scaled')
ylabel('piy (um)')
xlabel('pix (um)')
tight_layout()
