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
# Copyright (C) 2012-2013 Dominik Kriegner <dominik.kriegner@gmail.com>

# ALSO LOOK AT THE FILE xrayutilities_id01_functions.py

import numpy
import xrayutilities as xu
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import xrayutilities_id01_functions as id01

home = "DATADIR" # data path (root)
datadir = os.path.join(home,"FOLDERNAME") # data path for CCD/Maxipix files
specdir = home # location of spec file

sample = "SAMPLENAME" # sample name -> used as spec file name
ccdfiletmp = os.path.join(datadir,"CCDNAME_12_%05d.edf.gz") # template for the CCD file names


h5file = os.path.join(specdir,sample+".h5")
#read spec file and save to HDF5 (needs to be done only once)
try: s
except NameError: s = xu.io.SPECFile(sample+".spec",path=specdir)
else: s.Update() # in ipython run with: "run -i script" to just update the spec file and parse for new scans only
s.Save2HDF5(h5file)

# number of points to be used during the gridding
nx, ny, nz = 200,201,202

qx,qy,qz,gint,gridder = id01.gridmap(h5file,SCANNR,ccdfiletmp,nx,ny,nz)

#################################################
# for a 3D plot using python function i sugggest
# to use mayavi's mlab package. the basic usage
# is shown below. otherwise have a look at the 
# file xrayutilities_export_data2vtk.py in order learn 
# how you can get your data to a vtk file for further 
# processing.
######
## one of the following import statements is needed
## depending on the system/distribution you use
#from mayavi import mlab
#from enthough.mayavi import mlab
## plot 3D map using mayavi mlab
#QX,QY,QZ = numpy.mgrid[qx.min():qx.max():1j*nx,qy.min():qy.max():1j*ny,qz.min():qz.max():1j*nz]
#INT = xu.maplog(gint,4.5,0)
#mlab.figure()
#mlab.contour3d(QX,QY,QZ,INT,contours=15,opacity=0.5)
#mlab.colorbar(title="log(int)",orientation="vertical")
#mlab.axes(nb_labels=5,xlabel='Qx',ylabel='Qy',zlabel='Qz')
##mlab.close(all=True)
############################################

# plot 2D sums using matplotlib
plt.figure()
plt.contourf(qx,qy,xu.maplog(gint.sum(axis=2),2.8,1.5).T,50)
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()
#plt.savefig(os.path.join("pics","filename.png"))

# plot 2D slice using matplotlib
plt.figure()
plt.contourf(qx,qy,xu.maplog(gint[:,:,81:89].sum(axis=2),3.75,0).T,50)
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()




