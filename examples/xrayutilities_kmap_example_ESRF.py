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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import os
import numpy
import matplotlib.pyplot as plt

# define some convenience variables
en = 9000.0  # x-ray energy in eV
home = "\\homefolder"
workdir = os.path.join(home, 'work')
specdir = home  # location of spec file
sample = "sample"  # sample name -> used as spec file name
ccdfiletmp = "ccdfilename"
# region of interest on the detector; useful to reduce the amount of data
roi = [0, 516, 50, 300]

# define experimental geometry and detector parameters
# 2S+2D goniometer (simplified ID01 goniometer, sample mu,phi detector nu,del
qconv = xu.experiment.QConversion(['z+', 'y-'], ['z+', 'y-'], [1, 0, 0])
# convention for coordinate system: x downstream; z upwards; y to the
# "outside" (righthanded)
hxrd = xu.HXRD([1, 1, 0], [0, 0, 1], en=en, qconv=qconv)
hxrd.Ang2Q.init_area('z-', 'y+', cch1=200.07, cch2=297.75, Nch1=516, Nch2=516,
                     pwidth1=9.4489e-05, pwidth2=9.4452e-05, distance=1.,
                     detrot=-0.801, tiltazimuth=30.3, tilt=1.611, roi=roi)


def hotpixelkill(ccd):
    """
    function to remove hot pixels from CCD frames or apply any other filter.
    """
    # insert your own code here
    return ccd

U = numpy.identity(3)  # orientation matrix of the sample

scannr = numpy.arange(0, 100)

nx, ny = 100, 50

# specfile, scannumbers, nx,ny, motornames, optional column names (ID01
# values are default)
fss = xu.io.FastScanSeries(sample, scannr, nx, ny, 'Mu', 'Eta', 'Nu', 'Delta',
                           xmotor='adcY', ymotor='adcX', ccdnr='imgnr',
                           path=specdir)
# retrace clean all scans
fss.retrace_clean()
# align different scans (misalignment needs to be determined manually or if
# your application allows also by a 2d correllation code
# see e.g. scipy.signal.correlate2d
# fss.align(deltax,deltay)

# real space grid
g2d = xu.Gridder2D(nx, ny)
# read all motor positions from the data files
fss.read_motors()

# plot counter intensity on a grid
for idx, fs in enumerate(fss.fastscans):
    print(idx)
    g2d.Clear()
    g2d(fs.data['adcX'], fs.data['adcY'], fs.data['mpx4int'])
    plt.figure()
    plt.contourf(g2d.xaxis, g2d.yaxis, g2d.data.T, 50)
    plt.xlabel("X (um)")
    plt.ylabel("Y (um)")

posx, posy = 50, 30
# reduce data: number of pixels to average in each detector direction
nav = [2, 2]

qnx, qny, qnz = (80, 100, 101)
g = fss.gridRSM(posx, posy, qnx, qny, qnz, qconv, ccdfiletmp, path='',
                roi=roi, nav=nav, typ='index', filterfunc=hotpixelkill,
                UB=U)
# with typ='real' the position should be real space coordinates. with
# typ='index' the posx,y should specify indices within the range(nx)
# range(ny)

# g now contains a Gridder3D object which can be used for visualization
# see xrayutilities_example_plot_3D_ESRF_ID01 for example code

# Note: if you instead want to grid in two dimensions you can decrease one
# of qnx,qny or qnz to 1 and interprate the data in 2D
