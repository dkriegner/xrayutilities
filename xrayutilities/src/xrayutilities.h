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
 * Copyright (C) 2014 Eugen Wintersberger <eugen.wintersberger@gmail.com>
*/
#pragma once

#include <Python.h>
#include <math.h>

/*****************************************************************************
 * NUMPY specific macros and header files
 ****************************************************************************/
/*
 * need to make some definitions before loading the arrayobject.h 
 * header file
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

/*
 * set numpy API specific macros
 */
#if NPY_FEATURE_VERSION < 0x00000007
    #define NPY_ARRAY_ALIGNED       NPY_ALIGNED
    #define NPY_ARRAY_C_CONTIGUOUS  NPY_C_CONTIGUOUS
#endif

/*
 * define a macro to check a numpy array. This should mabye go into a 
 * function in future.
 */
#define PYARRAY_CHECK(array, dims, type, msg) \
    array = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) array, \
                                               type, \
                                               NPY_ARRAY_C_CONTIGUOUS | \
                                               NPY_ARRAY_ALIGNED); \
    if (PyArray_NDIM(array) != dims ||  \
        PyArray_TYPE(array) != type) {\
        PyErr_SetString(PyExc_ValueError, msg); \
        return NULL; \
    }


/*****************************************************************************
 * Windows build related macros 
 ****************************************************************************/
/* 'extern inline' seems to work only on newer version of gcc (>4.6 tested)
 * gcc 4.1 seems to need this empty, i am not sure if there is a speed gain
 * by inlining since the calls to those functions are anyhow built dynamically
 * for compatibility keep this empty unless you can test with several compilers
 */
#define INLINE
#ifdef _WIN32
#define RESTRICT
#else
#define RESTRICT restrict
#endif


/*
 * some stuff we need for the Windows build
 */
#ifdef _WIN32
    #ifndef __MINGW32__
        #include <float.h>
        #define isnan _isnan
    #endif

#endif

/*****************************************************************************
 * general purpose macros
 ****************************************************************************/
/*
 * if M_PI is not set we do this here
 */
#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif
#define M_2PI (2 * M_PI)


/*****************************************************************************
 * OpenMP related macros 
 ****************************************************************************/
/*
 * include OpenMP header is required
 */
#ifdef __OPENMP__
#include <omp.h>
#endif

#define OMPSETNUMTHREADS(nth) \
    if (nth == 0) omp_set_num_threads(omp_get_max_threads());\
    else omp_set_num_threads(nth);
