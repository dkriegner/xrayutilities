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
 * Copyright (C) 2025 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#include "array_utils.h"

int check_array(PyArrayObject** arr_ptr, int ndims, int typenum, const char* arr_name) {
    if (!arr_ptr || !*arr_ptr) {
        PyErr_Format(PyExc_ValueError, "Array pointer for '%s' is NULL", arr_name);
        return 0; // Indicate failure
    }

    PyArrayObject* arr = *arr_ptr;

    if (!PyArray_Check(arr)) {
        PyErr_Format(PyExc_TypeError, "Object '%s' is not an array", arr_name);
        return 0;
    }

    if (PyArray_NDIM(arr) != ndims) {
        PyErr_Format(PyExc_ValueError, "Array '%s' must have %d dimension(s), but has %d", arr_name, ndims, PyArray_NDIM(arr));
        return 0;
    }

    if (PyArray_TYPE(arr) != typenum) {
        PyArray_Descr *expected_descr = PyArray_DescrFromType(typenum);
        PyArray_Descr *actual_descr = PyArray_DESCR(arr);

        const char *expected_name = "unknown";
        const char *actual_name = "unknown";

        if (expected_descr && expected_descr->typeobj) {
            expected_name = expected_descr->typeobj->tp_name;
        }
        if (actual_descr && actual_descr->typeobj) {
            actual_name = actual_descr->typeobj->tp_name;
        }

        PyErr_Format(PyExc_ValueError, "Array '%s' must be of type %s, but is %s",
                     arr_name, expected_name, actual_name);

        Py_XDECREF(expected_descr); // Important: Decref the descriptor
        return 0;
    }

    if (!PyArray_ISCARRAY(arr)) {
        PyArrayObject* tmp = PyArray_GETCONTIGUOUS(arr);
        if (tmp == NULL) {
            PyErr_Format(PyExc_RuntimeError, "Could not create contiguous array for '%s'", arr_name);
            return 0;
        } else {
            Py_DECREF(*arr_ptr); // Decrement original
            *arr_ptr = tmp;      // Assign new one
            return 1; // Indicate that a new array was created
        }
    } else {
        Py_INCREF(arr); // Increment refcount of the array
        return 1; // Indicate that a new array was NOT created, but refcount was incremented
    }
}

PyArrayObject* check_and_convert_to_contiguous(PyObject* obj, int ndims, int typenum, const char* name) {
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "Object '%s' is not an array", name);
        return NULL;
    }

    PyArrayObject* arr = (PyArrayObject*)obj;

    if (PyArray_NDIM(arr) != ndims) {
        PyErr_Format(PyExc_ValueError, "Array '%s' must have %d dimension(s), but has %d", name, ndims, PyArray_NDIM(arr));
        return NULL;
    }

    if (PyArray_TYPE(arr) != typenum) {
        PyErr_Format(PyExc_ValueError, "Array '%s' must be of type %s, but is %s",
                     name, PyArray_DescrFromType(typenum)->typeobj->tp_name, PyArray_DESCR(arr)->typeobj->tp_name);
        return NULL;
    }

    if (PyArray_ISCARRAY(arr)) {
        Py_INCREF(arr); // Increment refcount if already contiguous
        return arr;
    } else {
        return PyArray_GETCONTIGUOUS(arr);
    }
}
