"""
this module uses the ctypes package to provide access to the 
functions implemented in the libxrayutils C library.

the functions provided by this module are low level. Users should use
the derived functions in the corresponding submodules 
"""

import numpy
import ctypes
import config

_library = ctypes.cdll.LoadLibrary(config.clib_path)

# c library gridder functions #{{{1
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
#}}}1


#_ang2q_xrd2d = _library.a2q_xrd2d
#_ang2q_xrd2d.restype = ctypes.c_int

#_ang2q_xrd2d_th = _library.a2q_xrd2d_th
#_ang2q_xrd2d_th.restype = ctypes.c_int

#_ang2q_xrd3d    = _library.a2q_xrd3d
#_ang2q_xrd3d.restype = ctypes.c_int

#_ang2q_xrd3d_th = _library.a2q_xrd3d_th
#_ang2q_xrd3d_th.restype = ctypes.c_int


# c library qconversion functions #{{{1
######################################
#     point conversion function
######################################
cang2q_point = _library.ang2q_conversion #{{{2
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
                   ctypes.c_double ] #}}}2

######################################
# linear detector conversion function
######################################
cang2q_linear = _library.ang2q_conversion_linear #{{{2
# c declaration: int ang2q_conversion_linear(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double cch, double dpixel, int *roi, char *dir, double lambda) 
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
                   ctypes.c_double ] #}}}2

#######################################
# area detector conversion function 
#######################################
cang2q_area = _library.ang2q_conversion_area #{{{2
# c declaration: int ang2q_conversion_area(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double cch1, double cch2, double dpixel1, double dpixel2, int *roi, char *dir1, char *dir2, double lambda) 
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
                   ctypes.c_double ] #}}}2
#}}}1

# c library functions for block averaging #{{{1
#######################################
#         1D block average 
#######################################
cblockav_1d = _library.block_average1d #{{{2
# c declaration: int block_average1d(double *block_av, double *input, int Nav, int N)
#define argument types
cblockav_1d.restype = ctypes.c_int
cblockav_1d.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int] #}}}2
#######################################
#    block average for PSD spectra
#######################################
cblockav_psd = _library.block_average_PSD #{{{2
# c declaration: int block_average_PSD(double *intensity, double *psd, int Nav, int Nch, int Nspec)
#define argument types
cblockav_psd.restype = ctypes.c_int
cblockav_psd.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int] #}}}2
#######################################
#    block average for CCD spectra
#######################################
cblockav_ccd = _library.block_average2d #{{{2
# c declaration: int block_average2d(double *block_av, double *ccd, int Nav1, int Nav2, int Nch1, int Nch2)
#define argument types
cblockav_ccd.restype = ctypes.c_int
cblockav_ccd.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned, contiguous"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int,
                   ctypes.c_int] #}}}2

#}}}1


