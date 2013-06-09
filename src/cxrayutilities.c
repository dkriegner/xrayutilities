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
*/

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#include <numpy/arrayobject.h>

/* functions from block_average.c */
extern PyObject* block_average1d(PyObject *self, PyObject *args);
extern PyObject* block_average2d(PyObject *self, PyObject *args);
extern PyObject* block_average_PSD(PyObject *self, PyObject *args);
extern PyObject* pygridder2d(PyObject *self,PyObject *args);

static PyMethodDef XRU_Methods[] = {
    {"block_average1d",  (PyCFunction)block_average1d, METH_VARARGS,
     "block average for one-dimensional numpy array"},
    {"block_average2d",  block_average2d, METH_VARARGS,
     "two dimensional block average for two-dimensional numpy array"},
    {"block_average_PSD",  block_average_PSD, METH_VARARGS,
     "one dimensional block average for two-dimensional numpy array (PSD spectra)"},
    {"gridder2d",pygridder2d,METH_VARARGS, "2D gridder function"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initcxrayutilities(void) {
    PyObject *m;

    m = Py_InitModule("cxrayutilities", XRU_Methods);
    if (m == NULL)
        return;

    import_array();
}
