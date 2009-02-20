#this module uses the ctypes module to provide access to the 
#functions implemented in the libxrayutils.so C library.

#the functions implemented by this module are low level. More complex 
#methods are implemented in the corresponding submodules


import numpy
import ctypes


_library = ctypes.CDLL("libxrayutils.so")

#attach functions from the gridder module
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

_gridder3d = _library.gridder3d
_gridder3d.restype = ctypes.c_int
_gridder3d.argtypes = [numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=1,flags="aligned,contiguous"),
                       ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=3,flags="aligned,contiguous"),
                       numpy.ctypeslib.ndpointer(numpy.double,ndim=3,flags="aligned,contiguous"),
                       ctypes.c_int]

_gridder3d_th = _library.gridder3d

_ang2q_xrd2d = _library.a2q_xrd2d
_ang2q_xrd2d.restype = ctypes.c_int

_ang2q_xrd2d_th = _library.a2q_xrd2d_th
_ang2q_xrd2d_th.restype = ctypes.c_int

_ang2q_xrd3d    = _library.a2q_xrd3d
_ang2q_xrd3d.restype = ctypes.c_int

_ang2q_xrd3d_th = _library.a2q_xrd3d_th
_ang2q_xrd3d_th.restype = ctypes.c_int




