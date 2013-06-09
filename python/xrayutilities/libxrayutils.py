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
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009-2010,2012 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
this module uses the ctypes package to provide access to the
functions implemented in the libxrayutils C library.

the functions provided by this module are low level. Users should use
the derived functions in the corresponding submodules
"""

import numpy
import ctypes

from . import config

_library = ctypes.cdll.LoadLibrary(config.CLIB_PATH)

# c library gridder functions
######################################
#     gridder 2D functions
######################################
_gridder2d = _library.gridder2d
_gridder2d.restype = ctypes.c_int
_gridder2d.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       ctypes.c_uint,
                       ctypes.c_uint,ctypes.c_uint,
                       ctypes.c_double,ctypes.c_double,
                       ctypes.c_double,ctypes.c_double,
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=2,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=2,flags="aligned,contiguous"),
                       ctypes.c_int]

_gridder2d_th = _library.gridder2d_th
_gridder2d_th.restype = ctypes.c_int
_gridder2d_th.argtypes = [ctypes.c_uint,
                          numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                          numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                          numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                          ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,
                          ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,
                          numpy.ctypeslib.ndpointer(numpy.double,ndim=2,flags="aligned,contiguous"),
                          numpy.ctypeslib.ndpointer(numpy.double,ndim=2,flags="aligned,contiguous"),
                          ctypes.c_int]

######################################
#     gridder 3D functions
######################################
_gridder3d = _library.gridder3d
_gridder3d.restype = ctypes.c_int
_gridder3d.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,
                       ctypes.c_double,ctypes.c_double,ctypes.c_double,
                       ctypes.c_double,ctypes.c_double,ctypes.c_double,
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=3,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=3,flags="aligned,contiguous"),
                       ctypes.c_int]

_gridder3d_th = _library.gridder3d

# c library qconversion functions
######################################
#     point conversion function
######################################
cang2q_point = _library.ang2q_conversion
# c declaration: int ang2q_conversion(double *sampleAngles,double *detectorAngles, double *qpos, double *ri, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double *UB, double lambda)
#define argument types
cang2q_point.restype = ctypes.c_int
cang2q_point.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_char_p,
                   ctypes.c_char_p,
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_double ]

######################################
# linear detector conversion function
######################################
cang2q_linear = _library.ang2q_conversion_linear
# c declaration: int ang2q_conversion_linear(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double cch, double dpixel, int *roi, char *dir, double tilt, double *UB, double lambda)
#define argument types
cang2q_linear.restype = ctypes.c_int
cang2q_linear.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_char_p,
                   ctypes.c_char_p,
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_double,
                   ctypes.c_double,
                   numpy.ctypeslib.ndpointer(numpy.int32,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_char_p,
                   ctypes.c_double,
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_double ]

#######################################
# area detector conversion function
#######################################
cang2q_area = _library.ang2q_conversion_area
# c declaration: int ang2q_conversion_area(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double cch1, double cch2, double dpixel1, double dpixel2, int *roi, char *dir1, char *dir2, double tiltazimuth, double tilt, double *UB, double lambda)
#define argument types
cang2q_area.restype = ctypes.c_int
cang2q_area.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_char_p,
                   ctypes.c_char_p,
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double,
                   numpy.ctypeslib.ndpointer(numpy.int32,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_char_p,
                   ctypes.c_char_p,
                   ctypes.c_double,
                   ctypes.c_double,
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_double ]

####################################################
# area detector conversion function for calibration
####################################################
cang2q_areapixel = _library.ang2q_conversion_area_pixel
# c declaration: int ang2q_conversion_area_pixel(double *detectorAngles, double *qpos, double *n1, double *n2, double *rcch, int Nd, int Npoints, char *detectorAxis, double cch1, double cch2, double dpixel1, double dpixel2, char *dir1, char *dir2, double tiltazimuth, double tilt, double lambda)
#define argument types
cang2q_areapixel.restype = ctypes.c_int
cang2q_areapixel.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_char_p,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_char_p,
                   ctypes.c_char_p,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double ]
