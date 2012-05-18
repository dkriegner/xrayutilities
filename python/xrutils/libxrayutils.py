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
# Copyright (C) 2009-2010 Dominik Kriegner <dominik.kriegner@aol.at>

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


#_ang2q_xrd2d = _library.a2q_xrd2d
#_ang2q_xrd2d.restype = ctypes.c_int

#_ang2q_xrd2d_th = _library.a2q_xrd2d_th
#_ang2q_xrd2d_th.restype = ctypes.c_int

#_ang2q_xrd3d    = _library.a2q_xrd3d
#_ang2q_xrd3d.restype = ctypes.c_int

#_ang2q_xrd3d_th = _library.a2q_xrd3d_th
#_ang2q_xrd3d_th.restype = ctypes.c_int


# c library qconversion functions
######################################
#     point conversion function
######################################
cang2q_point = _library.ang2q_conversion
# c declaration: int conversion(double *sampleAngles, double *detectorAngles, double *qpos, int Ns, int Nd, int Npoints,char *sampleAxis, char *detectorAxis, double lambda)
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
                   ctypes.c_double ]

######################################
# linear detector conversion function
######################################
cang2q_linear = _library.ang2q_conversion_linear
# c declaration: int ang2q_conversion_linear(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double cch, double dpixel, int *roi, char *dir, double tilt, double lambda)
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
                   ctypes.c_double,
                   ctypes.c_double,
                   numpy.ctypeslib.ndpointer(numpy.int32,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_char_p,
                   ctypes.c_double,
                   ctypes.c_double ]

#######################################
# area detector conversion function
#######################################
cang2q_area = _library.ang2q_conversion_area
# c declaration: int ang2q_conversion_area(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double cch1, double cch2, double dpixel1, double dpixel2, int *roi, char *dir1, char *dir2, double tilt1, double tilt2, double lambda)
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
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double,
                   numpy.ctypeslib.ndpointer(numpy.int32,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_char_p,
                   ctypes.c_char_p,
                   ctypes.c_double,
                   ctypes.c_double,
                   ctypes.c_double ]

# c library functions for block averaging
#######################################
#         1D block average
#######################################
cblockav_1d = _library.block_average1d
# c declaration: int block_average1d(double *block_av, double *input, int Nav, int N)
#define argument types
cblockav_1d.restype = ctypes.c_int
cblockav_1d.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int]
#######################################
#    block average for PSD spectra
#######################################
cblockav_psd = _library.block_average_PSD
# c declaration: int block_average_PSD(double *intensity, double *psd, int Nav, int Nch, int Nspec)
#define argument types
cblockav_psd.restype = ctypes.c_int
cblockav_psd.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int]
#######################################
#    block average for CCD spectra
#######################################
cblockav_ccd = _library.block_average2d
# c declaration: int block_average2d(double *block_av, double *ccd, int Nav1, int Nav2, int Nch1, int Nch2)
#define argument types
cblockav_ccd.restype = ctypes.c_int
cblockav_ccd.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int]


