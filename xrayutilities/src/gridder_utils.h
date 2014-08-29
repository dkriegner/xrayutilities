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
 * Copyright (C) 2013 Eugen Wintersberger <eugen.wintersberger@desy.de>
 * Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>
 *
 ******************************************************************************
 *
 * created: Jun 8,2013
 * author: Eugen Wintersberger
*/
#pragma once

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#ifdef _WIN32
    #ifndef __MINGW32__
        #include <float.h>
        #define isnan _isnan
    #endif
#endif

#define PYARRAY_CHECK(array,dims,type,msg) \
    array = (PyArrayObject *) PyArray_FROM_OTF((PyObject *)array,type,NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED); \
    if(PyArray_NDIM(array) != dims ||  \
       PyArray_TYPE(array) != type) \
    {\
        PyErr_SetString(PyExc_ValueError,\
                msg); \
        return NULL; \
    }

/*!
\brief find minimum

Finds the minimum in an array.
\param a input data
\param n number of elements
\return minimum value
*/
double get_min(double *a,unsigned int n);

//-----------------------------------------------------------------------------
/*!
\brief find maximum

Finds the maximum value in an array.
\param a input data
\param n number of elements
\return return maximum value
*/
double get_max(double *a,unsigned int n);

//-----------------------------------------------------------------------------
/*!
\brief set array values

Set all elements of an array to the same values.
\param a input array
\param n number of points
\param value the new element values
*/
void set_array(double *a,unsigned int n,double value);

//-----------------------------------------------------------------------------
/*!
\brief compute step width

Computes the stepwidth of a grid.
\param min minimum value
\param max maximum value
\param n number of steps
\return step width
*/
double delta(double min,double max,unsigned int n);

//-----------------------------------------------------------------------------
/*!
\brief compute grid index

*/
unsigned int gindex(double x,double min,double d);

#ifdef _WIN32
double rint(double x);
#endif
