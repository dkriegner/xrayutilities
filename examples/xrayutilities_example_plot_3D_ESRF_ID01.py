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
# Copyright (C) 2012-2018 Dominik Kriegner <dominik.kriegner@gmail.com>

# ALSO LOOK AT THE FILE xrayutilities_id01_functions.py

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import xrayutilities as xu

import xrayutilities_id01_functions as id01

try:
    s
except NameError:
    s = xu.io.SPECFile(sample + ".spec", path=id01.datadir)
else:
    # in ipython run with: "run -i script" to just update the spec file and
    # parse for new scans only
    s.Update()

# number of points to be used during the gridding
nx, ny, nz = 200, 201, 202

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
# all in mm since mm are used for mpxy,z in the spec-file

qx, qy, qz, gint, gridder = id01.gridmap(s, SCANNR, hxrd, nx, ny, nz)

# ################################################
# for a 3D plot using python function I sugggest
# to use mayavi's mlab package. the basic usage
# is shown below. otherwise have a look at the
# file xrayutilities_export_data2vtk.py in order learn
# how you can get your data to a vtk file for further
# processing.
# #####
# one of the following import statements is needed
# depending on the system/distribution you use
# from mayavi import mlab
# # from enthough.mayavi import mlab
# # plot 3D map using mayavi mlab
# QX,QY,QZ = numpy.mgrid[qx.min():qx.max():1j * nx,
#                        qy.min():qy.max():1j * ny,
#                        qz.min():qz.max():1j * nz]
# INT = xu.maplog(gint,4.5,0)
# mlab.figure()
# mlab.contour3d(QX, QY, QZ, INT, contours=15, opacity=0.5)
# mlab.colorbar(title="log(int)", orientation="vertical")
# mlab.axes(nb_labels=5, xlabel='Qx', ylabel='Qy', zlabel='Qz')
# # mlab.close(all=True)
############################################

# plot 2D sums using matplotlib
plt.figure()
plt.contourf(qx, qy, xu.maplog(gint.sum(axis=2), 2.8, 1.5).T, 50)
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()
# plt.savefig(os.path.join("pics","filename.png"))

# plot 2D slice using matplotlib
plt.figure()
plt.contourf(qx, qy, xu.maplog(gint[:, :, 81:89].sum(axis=2), 3.75, 0).T, 50)
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()
