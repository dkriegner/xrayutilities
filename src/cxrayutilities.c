/*
 * This file is part of xrayutilities.
 *
 * xrayutilities is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright (C) 2013-2016 Dominik Kriegner <dominik.kriegner@gmail.com>
 * Copyright (C) 2013 Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
*/
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#include <numpy/arrayobject.h>


/* functions from block_average.c */
extern PyObject* block_average1d(PyObject *self, PyObject *args);
extern PyObject* block_average2d(PyObject *self, PyObject *args);
extern PyObject* block_average_PSD(PyObject *self, PyObject *args);
extern PyObject* block_average_CCD(PyObject *self, PyObject *args);

/* functions from gridder1d.c */
extern PyObject* pygridder1d(PyObject *self, PyObject *args);
extern PyObject* pyfuzzygridder1d(PyObject *self, PyObject *args);

/* functions from gridder2d.c */
extern PyObject* pygridder2d(PyObject *self, PyObject *args);
extern PyObject* pyfuzzygridder2d(PyObject *self, PyObject *args);

/* function from gridder3d.c */
extern PyObject* pygridder3d(PyObject *self, PyObject *args);
extern PyObject* pyfuzzygridder3d(PyObject *self, PyObject *args);

/* functions from qconversion.c */
extern PyObject* py_ang2q_conversion(PyObject *self, PyObject *args);
extern PyObject* py_ang2q_conversion_linear(PyObject *self, PyObject *args);
extern PyObject* py_ang2q_conversion_area(PyObject *self, PyObject *args);
extern PyObject* ang2q_conversion_area_pixel(PyObject *self, PyObject *args);
extern PyObject* ang2q_conversion_area_pixel2(PyObject *self, PyObject *args);

extern PyObject* ang2q_detpos(PyObject *self, PyObject *args);
extern PyObject* ang2q_detpos_linear(PyObject *self, PyObject *args);
extern PyObject* ang2q_detpos_area(PyObject *self, PyObject *args);

/* functions from file_io.c */
extern PyObject* cbfread(PyObject *self, PyObject *args);

static PyMethodDef XRU_Methods[] = {
    {"block_average1d", (PyCFunction)block_average1d, METH_VARARGS,
     "block average for one-dimensional numpy double array\n\n"
     "Parameters \n"
     "----------\n"
     " input:        input array of datatype double \n"
     " Nav:          number of items to average \n\n"
     "Returns\n"
     "-------\n"
     " block_av:   block averaged output array\n"
     "             size = ceil(N / Nav) \n"
    },
    {"block_average2d", block_average2d, METH_VARARGS,
     "2D block average for one CCD frame\n\n"
     "Parameters\n"
     "----------\n"
     " ccd:     input array/CCD frame\n"
     "          size = (Nch2, Nch1) \n"
     "          Nch1 is the fast varying index\n"
     " Nav1, 2:  number of channels to average in each dimension\n"
     "          in total a block of Nav1 x Nav2 is averaged\n"
     "\n"
     "Returns\n"
     "-------\n"
     " block_av:     block averaged output array\n"
     "               size = (ceil(Nch2 / Nav2) , ceil(Nch1 / Nav1))\n"
    },
    {"block_average_PSD", block_average_PSD, METH_VARARGS,
     "1D block average for a bunch of PSD spectra in a 2D array\n"
     "\n"
     "Parameters\n"
     "----------\n"
     " psd:         input array of PSD values\n"
     "              size = (Nspec, Nch) (in)\n"
     " Nav:         number of channels to average\n"
     "\n"
     "Returns\n"
     "-------\n"
     " block_av:    block averaged output array\n"
     "              size = (Nspec , ceil(Nch/Nav)) (out)\n"
    },
    {"block_average_CCD", block_average_CCD, METH_VARARGS,
     "2D block average for a series of CCD frames\n\n"
     "Parameters\n"
     "----------\n"
     " ccd:     input array/CCD frames\n"
     "          size = (Nframes, Nch2, Nch1) \n"
     "          Nch1 is the fast varying index\n"
     " Nav1, 2:  number of channels to average in each dimension\n"
     "          in total a block of Nav1 x Nav2 is averaged\n"
     "\n"
     "Returns\n"
     "-------\n"
     " block_av:     block averaged output array\n"
     "               size = (Nframes, ceil(Nch2 / Nav2) , ceil(Nch1 / Nav1))\n"
    },
    {"gridder1d", pygridder1d, METH_VARARGS,
     "Function performs 1D gridding on 1D input data. \n\n"
     "Parameters\n"
     "----------\n"
     "  x ...... input x-values (1D numpy array - float64)\n"
     "  data ... input data (1D numpy array - float64)\n"
     "  nx ..... number of grid points in x-direction\n"
     "  xmin ... minimum x-value of the grid\n"
     "  xmax ... maximum x-value of the grid\n"
     "  out .... output data\n"
     "  norm ... normalization array\n"
     "  flags .. flags to specify behavior\n"
    },
    {"fuzzygridder1d", pyfuzzygridder1d, METH_VARARGS,
     "Function performs fuzzy 1D gridding on 1D input data. \n\n"
     "Parameters\n"
     "----------\n"
     "  x ...... input x-values (1D numpy array - float64)\n"
     "  data ... input data (1D numpy array - float64)\n"
     "  nx ..... number of grid points in x-direction\n"
     "  xmin ... minimum x-value of the grid\n"
     "  xmax ... maximum x-value of the grid\n"
     "  out .... output data\n"
     "  norm ... normalization array\n"
     "  fuzzywidth .. fuzzy-size of the input data\n"
     "  flags .. flags to specify behavior\n"
    },
    {"gridder2d", pygridder2d, METH_VARARGS,
     "Function performs 2D gridding on 1D input data. \n\n"
     "Parameters\n"
     "----------\n"
     "  x ...... input x-values (1D numpy array - float64)\n"
     "  y ...... input y-values (1D numpy array - float64)\n"
     "  data ... input data (1D numpy array - float64)\n"
     "  nx ..... number of grid points in x-direction\n"
     "  ny ..... number of grid points in y-direction\n"
     "  xmin ... minimum x-value of the grid\n"
     "  xmax ... maximum x-value of the grid\n"
     "  ymin ... minimum y-value of the grid\n"
     "  ymax ... minimum y-value of the grid\n"
     "  out .... output data\n"
     "  norm ... normalization array\n"
     "  flags .. flags to specify behavior\n"
    },
    {"fuzzygridder2d", pyfuzzygridder2d, METH_VARARGS,
     "Function performs 2D fuzzy gridding on 1D input data. \n\n"
     "Parameters\n"
     "----------\n"
     "  x ...... input x-values (1D numpy array - float64)\n"
     "  y ...... input y-values (1D numpy array - float64)\n"
     "  data ... input data (1D numpy array - float64)\n"
     "  nx ..... number of grid points in x-direction\n"
     "  ny ..... number of grid points in y-direction\n"
     "  xmin ... minimum x-value of the grid\n"
     "  xmax ... maximum x-value of the grid\n"
     "  ymin ... minimum y-value of the grid\n"
     "  ymax ... minimum y-value of the grid\n"
     "  out .... output data\n"
     "  norm ... normalization array\n"
     "  wx ..... fuzzy width of data points in x-direction\n"
     "  wy ..... fuzzy width of data points in y-direction\n"
     "  flags .. flags to specify behavior\n"
    },
    {"gridder3d", pygridder3d, METH_VARARGS,
     "Function performs 2D gridding on 1D input data. \n\n"
     "Parameters\n"
     "----------\n"
     "  x ...... input x-values (1D numpy array - float64)\n"
     "  y ...... input y-values (1D numpy array - float64)\n"
     "  z ...... input z-values (1D numpy array - float64)\n"
     "  data ... input data (1D numpy array - float64)\n"
     "  nx ..... number of grid points in x-direction\n"
     "  ny ..... number of grid points in y-direction\n"
     "  nz ..... number of grid points in z-direction\n"
     "  xmin ... minimum x-value of the grid\n"
     "  xmax ... maximum x-value of the grid\n"
     "  ymin ... minimum y-value of the grid\n"
     "  ymax ... maximum y-value of the grid\n"
     "  zmin ... minimum z-value of the grid\n"
     "  zmax ... maximum z-value of the grid\n"
     "  out .... output data\n"
     "  norm ... normalization array\n"
     "  flags .. flags to specify behavior\n"
    },
    {"fuzzygridder3d", pyfuzzygridder3d, METH_VARARGS,
     "Function performs 3D fuzzy gridding on 1D input data. \n\n"
     "Parameters\n"
     "----------\n"
     "  x ...... input x-values (1D numpy array - float64)\n"
     "  y ...... input y-values (1D numpy array - float64)\n"
     "  z ...... input z-values (1D numpy array - float64)\n"
     "  data ... input data (1D numpy array - float64)\n"
     "  nx ..... number of grid points in x-direction\n"
     "  ny ..... number of grid points in y-direction\n"
     "  nz ..... number of grid points in z-direction\n"
     "  xmin ... minimum x-value of the grid\n"
     "  xmax ... maximum x-value of the grid\n"
     "  ymin ... minimum y-value of the grid\n"
     "  ymax ... maximum y-value of the grid\n"
     "  zmin ... minimum z-value of the grid\n"
     "  zmax ... maximum z-value of the grid\n"
     "  out .... output data\n"
     "  norm ... normalization array\n"
     "  wx ..... fuzzy width of data points in x-direction\n"
     "  wy ..... fuzzy width of data points in y-direction\n"
     "  wz ..... fuzzy width of data points in z-direction\n"
     "  flags .. flags to specify behavior\n"
    },
    {"ang2q_conversion", py_ang2q_conversion, METH_VARARGS,
     "conversion of Npoints of goniometer positions to reciprocal space\n"
     "for a setup with point detector\n"
     "\n"
     "Parameters\n"
     "----------\n"
     "  sampleAngles .... angular positions of the sample goniometer\n"
     "                    (Npoints, Ns)\n"
     "  detectorAngles .. angular positions of the detector goniometer\n"
     "                    (Npoints, Nd)\n"
     "  ri .............. direction of primary beam (length irrelevant)\n"
     "                    (angles zero)\n"
     "  sampleAxis ...... string with sample axis directions\n"
     "  detectorAxis .... string with detector axis directions\n"
     "  kappadir ........ rotation axis of a possible kappa circle\n"
     "  UB .............. orientation matrix and reciprocal space conversion\n"
     "                    of investigated crystal (3, 3)\n"
     "  sampledis ....... sample displacement vector in relative units of\n"
     "                    the detector distance\n"
     "  lambda .......... wavelength of the used x-rays (Angstreom)\n"
     "  nthreads ........ number of threads to use in parallel section of\n"
     "                    the code\n"
     "  flags ........... integer flags to select sub-function\n"
     "\n"
     "Returns\n"
     "-------\n"
     " qpos .......... momentum transfer (Npoints, 3)\n"
    },
    {"ang2q_conversion_linear", py_ang2q_conversion_linear, METH_VARARGS,
     "conversion of Npoints of goniometer positions to reciprocal space\n"
     "for a linear detector with a given pixel size mounted along one of\n"
     "the coordinate axis\n"
     "\n"
     "Parameters\n"
     "----------\n"
     " sampleAngles .... angular positions of the goniometer (Npoints, Ns)\n"
     " detectorAngles .. angular positions of the detector goniometer\n"
     "                   (Npoints, Nd)\n"
     " rcch ............ direction + distance of center channel (angles\n"
     "                   zero)\n"
     " sampleAxis ...... string with sample axis directions\n"
     " detectorAxis .... string with detector axis directions\n"
     " kappadir ........ rotation axis of a possible kappa circle\n"
     " cch ............. center channel of the detector\n"
     " dpixel .......... width of one pixel, same unit as distance rcch\n"
     " roi ............. region of interest of the detector\n"
     " dir ............. direction of the detector, e.g.: 'x+'\n"
     " tilt ............ tilt of the detector direction from dir\n"
     " UB .............. orientation matrix and reciprocal space conversion\n"
     "                   of investigated crystal (9)\n"
     " sampledis ....... sample displacement vector in same unit as the\n"
     "                   detector distance\n"
     " lambda .......... wavelength of the used x-rays in Angstroem\n"
     " nthreads ........ number of threads to use in parallel section of the\n"
     "                   code\n"
     " flags ........... integer flags to select sub-function\n"
     "\n"
     "Returns\n"
     "-------\n"
     " qpos ............ momentum transfer (Npoints * Nch, 3)\n"
     " \n"
    },
    {"ang2q_conversion_area", py_ang2q_conversion_area, METH_VARARGS,
     "conversion of Npoints of goniometer positions to reciprocal space\n"
     "for an area detector with a given pixel size mounted along one of\n"
     "the coordinate axis\n"
     "\n"
     "Parameters\n"
     "----------\n"
     "  sampleAngles .... angular positions of the sample goniometer\n"
     "                    (Npoints, Ns)\n"
     "  detectorAngles .. angular positions of the detector goniometer\n"
     "                    (Npoints, Nd)\n"
     "  rcch ............ direction + distance of center pixel (angles zero)\n"
     "  sampleAxis ...... string with sample axis directions\n"
     "  detectorAxis .... string with detector axis directions\n"
     "  kappadir ...... rotation axis of a possible kappa circle\n"
     "  cch1 ............ center channel of the detector\n"
     "  cch2 ............ center channel of the detector\n"
     "  dpixel1 ......... width of one pixel in first direction, same unit\n"
     "                    as distance rcch\n"
     "  dpixel2 ......... width of one pixel in second direction, same unit\n"
     "                    as distance rcch\n"
     "  roi ............. region of interest for the area detector\n"
     "                    [dir1min, dir1max, dir2min, dir2max]\n"
     "  dir1 ............ first direction of the detector, e.g.: 'x+'\n"
     "  dir2 ............ second direction of the detector, e.g.: 'z+'\n"
     "  tiltazimuth ..... azimuth of the tilt\n"
     "  tilt ............ tilt of the detector plane (rotation around axis\n"
     "                    normal to the direction given by the tiltazimuth\n"
     "  UB .............. orientation matrix and reciprocal space conversion\n"
     "                    of investigated crystal (3, 3)\n"
     "  sampledis ....... sample displacement vector in same unit as the\n"
     "                    detector distance\n"
     "  lambda .......... wavelength of the used x-rays \n"
     "  nthreads ........ number of threads to use in parallel section of\n"
     "                    the code\n"
     "  flags ........... integer flags to select sub-function\n"
     "\n"
     "Returns\n"
     "-------\n"
     " qpos ............ momentum transfer (Npoints * Npix1 * Npix2, 3)\n"
     "\n"
    },
    {"ang2q_conversion_area_pixel", ang2q_conversion_area_pixel, METH_VARARGS,
     "conversion of Npoints of detector positions to Q\n"
     "for an area detector with a given pixel size mounted along one of\n"
     "the coordinate axis. This function only calculates the q-position for\n"
     "the pairs of pixel numbers (n1, n2) given in the input and should\n"
     "therefore be used only for detector calibration purposes.\n"
     "\n"
     "Parameters\n"
     "----------\n"
     " detectorAngles .. angular positions of the detector goniometer\n"
     "                   (Npoints, Nd)\n"
     " n1 .............. detector pixel numbers dim1 (Npoints)\n"
     " n2 .............. detector pixel numbers dim2 (Npoints)\n"
     " rcch ............ direction + distance of center pixel (angles zero)\n"
     " detectorAxis .... string with detector axis directions\n"
     " cch1 ............ center channel of the detector\n"
     " cch2 ............ center channel of the detector\n"
     " dpixel1 ......... width of one pixel in first direction, same unit as\n"
     "                   distance rcch\n"
     " dpixel2 ......... width of one pixel in second direction, same unit\n"
     "                   as distance rcch\n"
     " dir1 ............ first direction of the detector, e.g.: 'x+'\n"
     " dir2 ............ second direction of the detector, e.g.: 'z+'\n"
     " tiltazimuth ..... azimuth of the tilt\n"
     " tilt ............ tilt of the detector plane (rotation around axis\n"
     "                   normal to the direction given by the tiltazimuth\n"
     " lambda .......... wavelength of the used x-rays\n"
     " nthreads ........ number of threads to use in parallel section of the\n"
     "                   code\n"
     "\n"
     "Returns\n"
     "-------\n"
     " qpos ............ momentum transfer (Npoints, 3)\n"
    },
    {"ang2q_conversion_area_pixel2",
     ang2q_conversion_area_pixel2, METH_VARARGS,
     "conversion of Npoints of detector positions to Q.\n"
     "for an area detector with a given pixel size mounted along one of\n"
     "the coordinate axis. This function only calculates the q-position for\n"
     "the pairs of pixel numbers (n1, n2) given in the input and should\n"
     "therefore be used only for detector calibration purposes.\n"
     "\n"
     "This variant of this function also takes a sample orientation matrix\n"
     "as well as the sample goniometer as input to allow for a simultaneous\n"
     "fit of reference samples orientation\n"
     "\n"
     "Parameters\n"
     "----------\n"
     " sampleAngles .... angular positions of the sample goniometer\n"
     "                   (Npoints, Ns)\n"
     " detectorAngles .. angular positions of the detector goniometer\n"
     "                   (Npoints, Nd)\n"
     " n1 .............. detector pixel numbers dim1 (Npoints)\n"
     " n2 .............. detector pixel numbers dim2 (Npoints)\n"
     " rcch ............ direction + distance of center pixel (angles zero)\n"
     " sampleAxis ...... string with sample axis directions\n"
     " detectorAxis .... string with detector axis directions\n"
     " cch1 ............ center channel of the detector\n"
     " cch2 ............ center channel of the detector\n"
     " dpixel1 ......... width of one pixel in first direction, same unit as\n"
     "                   distance rcch\n"
     " dpixel2 ......... width of one pixel in second direction, same unit\n"
     "                   as distance rcch \n"
     " dir1 ............ first direction of the detector, e.g.: 'x+'\n"
     " dir2 ............ second direction of the detector, e.g.: 'z+'\n"
     " tiltazimuth ..... azimuth of the tilt\n"
     " tilt ............ tilt of the detector plane (rotation around axis\n"
     "                   normal to the direction given by the tiltazimuth\n"
     " UB .............. orientation matrix and reciprocal space conversion\n"
     "                   of investigated crystal (3, 3)\n"
     " lambda .......... wavelength of the used x-rays\n"
     " nthreads ........ number of threads to use in parallel section of the\n"
     "                   code\n"
     "\n"
     "Returns\n"
     "-------\n"
     " qpos ............ momentum transfer (Npoints, 3)\n"
    },
    {"ang2q_detpos", ang2q_detpos, METH_VARARGS,
     "conversion of Npoints of detector arm angles to real space position\n"
     "for a setup with point detector\n"
     "\n"
     "Parameters\n"
     "----------\n"
     "  detectorAngles .. angular positions of the detector goniometer\n"
     "                    (Npoints, Nd)\n"
     "  ri .............. direction and distance of detector at zero angles\n"
     "  detectorAxis .... string with detector axis directions\n"
     "  nthreads ........ number of threads to use in parallel section of\n"
     "                    the code\n"
     "\n"
     "Returns\n"
     "-------\n"
     " dpos .......... real space detector position (Npoints, 3)\n"
    },
    {"ang2q_detpos_linear", ang2q_detpos_linear, METH_VARARGS,
     "conversion of Npoints of detector arm angles to real space\n"
     "position for a linear detector\n"
     "\n"
     "Parameters\n"
     "----------\n"
     " detectorAngles .. angular positions of the detector goniometer\n"
     "                   (Npoints, Nd)\n"
     " rcch ............ direction + distance of center channel\n"
     "                   (angles zero)\n"
     " detectorAxis .... string with detector axis directions\n"
     " cch ............. center channel of the detector\n"
     " dpixel .......... width of one pixel, same unit as distance rcch\n"
     " roi ............. region of interest of the detector\n"
     " dir ............. direction of the detector, e.g.: 'x+'\n"
     " tilt ............ tilt of the detector direction from dir\n"
     " nthreads ........ number of threads to use in parallel section of\n"
     "                   the code\n"
     "\n"
     "Returns\n"
     "-------\n"
     " dpos ............ real space detector position (Npoints * Nch, 3)\n"
     " \n"
    },
    {"ang2q_detpos_area", ang2q_detpos_area, METH_VARARGS,
     "conversion of Npoints of detector arm angles to reciprocal space\n"
     "position for an area detector with a given pixel size\n"
     "\n"
     "Parameters\n"
     "----------\n"
     "  detectorAngles .. angular positions of the detector goniometer\n"
     "                    (Npoints, Nd)\n"
     "  rcch ............ direction + distance of center pixel (angles zero)\n"
     "  detectorAxis .... string with detector axis directions\n"
     "  cch1 ............ center channel of the detector\n"
     "  cch2 ............ center channel of the detector\n"
     "  dpixel1 ......... width of one pixel in first direction, same unit\n"
     "                    as distance rcch\n"
     "  dpixel2 ......... width of one pixel in second direction, same unit\n"
     "                    as distance rcch\n"
     "  roi ............. region of interest for the area detector\n"
     "                    [dir1min, dir1max, dir2min, dir2max]\n"
     "  dir1 ............ first direction of the detector, e.g.: 'x+'\n"
     "  dir2 ............ second direction of the detector, e.g.: 'z+'\n"
     "  tiltazimuth ..... azimuth of the tilt\n"
     "  tilt ............ tilt of the detector plane (rotation around axis\n"
     "                    normal to the direction given by the tiltazimuth\n"
     "  nthreads ........ number of threads to use in parallel section of\n"
     "                    the code\n"
     "\n"
     "Returns\n"
     "-------\n"
     " dpos ............ real space detector position\n"
     "                   (Npoints * Npix1 * Npix2, 3)\n"
     "\n"
    },
    {"cbfread", cbfread, METH_VARARGS,
     "parser for cbf data arrays from Pilatus detector images\n\n"
     " Parameters\n"
     " ----------\n"
     "  data:   data stream (character array)\n"
     "  nx, ny: number of entries of the two dimensional image\n\n"
     " Returns\n"
     " -------\n"
     "  the parsed data values as float ndarray\n"
    },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cxrayutilities",  /* m_name */
    "Python C extension including performance critical parts\n"
    "of xrayutilities (gridder, qconversion, block-averageing)\n",  /* m_doc */
    -1,                /* m_size */
    XRU_Methods,       /* m_methods */
    NULL,              /* m_reload */
    NULL,              /* m_traverse */
    NULL,              /* m_clear */
    NULL,              /* m_free */
};
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_cxrayutilities(void)
#else
initcxrayutilities(void)
#endif
{
    PyObject *m;

    #if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    #else
    m = Py_InitModule3("cxrayutilities", XRU_Methods,
        "Python C extension including performance critical parts\n"
        "of xrayutilities (gridder, qconversion, block-averageing)\n");
    #endif

    import_array();

    #if PY_MAJOR_VERSION >= 3
    return m;
    #endif
}
