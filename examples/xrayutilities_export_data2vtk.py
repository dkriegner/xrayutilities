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
# Copyright (C) 2013 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (c) 2013-2023 Dominik Kriegner <dominik.kriegner@gmail.com>


import os

import numpy
import vtk
import xrayutilities_id01_functions as id01
from vtk.util import numpy_support

import xrayutilities as xu

home = "DATADIR"  # data path (root)
datadir = os.path.join(home, "FOLDERNAME")  # data path for CCD/Maxipix files
specdir = home  # location of spec file

sample = "SAMPLENAME"  # sample name -> used as spec file name
SCANNR = 95  # put your scan number here
# template for the CCD file names
ccdfiletmp = os.path.join(datadir, "CCDFILENAME_%04d.edf.gz")


h5file = os.path.join(specdir, sample + ".h5")
# read spec file and save to HDF5 (needs to be done only once)
# try: s
# except NameError: s = xu.io.SPECFile(sample+".spec",path=specdir)
# else:
#    s.Update()
# s.Save2HDF5(h5file)
# in ipython run with: "run -i script" to just update the spec file and parse
# for new scans only

# number of points to be used during the gridding
nx, ny, nz = 100, 101, 102

# read and grid data with helper function
qx, qy, qz, gint, gridder = id01.gridmap(
    h5file, SCANNR, ccdfiletmp, nx, ny, nz
)

# prepare data for export to VTK image file
INT = xu.maplog(gint, 3.0, 0)

# export variables qx, qy, qz, INT
qx0 = qx.min()
dqx = (qx.max() - qx.min()) / nx

qy0 = qy.min()
dqy = (qy.max() - qy.min()) / ny

qz0 = qz.min()
dqz = (qz.max() - qz.min()) / nz

INT = numpy.transpose(INT).reshape((INT.size))
data_array = numpy_support.numpy_to_vtk(INT)

image_data = vtk.vtkImageData()

if vtk.vtkVersion.GetVTKMajorVersion() < 6:
    image_data.SetNumberOfScalarComponents(1)
    image_data.SetScalarTypeToDouble()
else:
    vtkinfo = image_data.GetInformation()
    image_data.SetPointDataActiveScalarInfo(vtkinfo, vtk.VTK_DOUBLE, 1)

image_data.SetOrigin(qx0, qy0, qz0)
image_data.SetSpacing(dqx, dqy, dqz)
image_data.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)


pd = image_data.GetPointData()
pd.SetScalars(data_array)

# export data to file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("output.vti")
if vtk.vtkVersion.GetVTKMajorVersion() < 6:
    writer.SetInput(image_data)
else:
    writer.SetInputData(image_data)

writer.Write()
