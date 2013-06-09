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
 * Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>
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

/* functions from gridder2d.c */
extern PyObject* pygridder2d(PyObject *self,PyObject *args);

/* functions from qconversion.c */
extern PyObject* ang2q_conversion(PyObject *self, PyObject *args);
extern PyObject* ang2q_conversion_linear(PyObject *self, PyObject *args);
extern PyObject* ang2q_conversion_area(PyObject *self, PyObject *args);
extern PyObject* ang2q_conversion_area_pixel(PyObject *self, PyObject *args);

static PyMethodDef XRU_Methods[] = {
    {"block_average1d",  (PyCFunction)block_average1d, METH_VARARGS,
     "block average for one-dimensional numpy array"},
    {"block_average2d",  block_average2d, METH_VARARGS,
     "two dimensional block average for two-dimensional numpy array"},
    {"block_average_PSD",  block_average_PSD, METH_VARARGS,
     "one dimensional block average for two-dimensional numpy array (PSD spectra)"},
    {"gridder2d",pygridder2d,METH_VARARGS, 
     "Function performs 2D gridding on 1D input data. \n" 
     "Input arguments: \n"
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
    },
    {"ang2q_conversion", ang2q_conversion, METH_VARARGS,
     "reciprocal space conversion for apoint detectors"},
    {"ang2q_conversion_linear", ang2q_conversion_linear, METH_VARARGS,
     "reciprocal space conversion for a linear detectors"},
    {"ang2q_conversion_area", ang2q_conversion_area, METH_VARARGS,
     "reciprocal space conversion for an area detectors"},
    {"ang2q_conversion_area_pixel", ang2q_conversion_area_pixel, METH_VARARGS,
     "reciprocal space conversion for certain pixels of an area detectors"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initcxrayutilities(void) {
    PyObject *m;

    m = Py_InitModule3("cxrayutilities", XRU_Methods, 
        "Python C extension including performance critical parts\n"
        "of xrayutilities (gridder, qconversion, block-averageing)\n");
    if (m == NULL)
        return;

    import_array();
}
