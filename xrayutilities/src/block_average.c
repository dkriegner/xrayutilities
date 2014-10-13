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
 * Copyright (C) 2010-2011, 2013 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#include "xrayutilities.h"

PyObject* block_average1d(PyObject *self, PyObject *args) {
    /*    block average for one-dimensional double array
     *
     *    Parameters
     *    ----------
     *    input:        input array of datatype double
     *    Nav:          number of items to average
     *
     *    Returns
     *    -------
     *    block_av:     block averaged output array
     *                  size = ceil(N/Nav)
     *
     */

    int i, j, Nav, N;
    PyArrayObject *input=NULL, *outarr=NULL;
    double *cin, *cout;
    double buf;
    npy_intp nout;

    /* Python argument conversion code */
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &input, &Nav)) {
        return NULL;
    }

    PYARRAY_CHECK(input, 1, NPY_DOUBLE, "input must be a 1D double array!");
    N = (int) PyArray_SIZE(input);
    cin = (double *) PyArray_DATA(input);

    /* create output ndarray */
    nout = ((int) ceil(N / (float) Nav));
    outarr = (PyArrayObject *) PyArray_SimpleNew(1, &nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);

    /* c-code following is performing the block averaging */
    for (i = 0; i < N; i = i + Nav) {
        buf=0;
        /* perform one block average (j-i serves as counter
           -> last bin is therefore correct) */
        for (j = i; j < i + Nav && j < N; ++j) {
            buf += cin[j];
        }
        /* save average to output array */
        cout[i / Nav] = buf / (float) (j - i);
    }

    /* clean up */
    Py_DECREF(input);

    /* return output array */
    return PyArray_Return(outarr);
}

PyObject* block_average2d(PyObject *self, PyObject *args) {
    /*    2D block average for one CCD frame
     *
     *    Parameters
     *    ----------
     *    ccd:          input array/CCD frame
     *                  size = (Nch2, Nch1)
     *                  Nch1 is the fast varying index
     *    Nav1, 2:      number of channels to average in each dimension
     *                  in total a block of Nav1 x Nav2 is averaged
     *    nthreads:     number of threads to use in parallel section
     *
     *    Returns
     *    -------
     *    block_av:     block averaged output array
     *                  size = (ceil(Nch2/Nav2) , ceil(Nch1/Nav1))
     *
     */

    int i = 0, j = 0, k = 0, l = 0;  /* loop indices */
    int Nch1, Nch2;  /* number of values in input array */
    int Nav1, Nav2;  /* number of items to average */
    unsigned int nthreads;  /* number of threads to use */
    PyArrayObject *input = NULL, *outarr = NULL;
    double *cin, *cout;
    double buf;
    npy_intp nout[2];

    /* Python argument conversion code */
    if (!PyArg_ParseTuple(args, "O!iiI", &PyArray_Type, &input, &Nav2,
                          &Nav1, &nthreads)) {
        return NULL;
    }

    PYARRAY_CHECK(input, 2, NPY_DOUBLE, "input must be a 2D double array!");
    Nch2 = (int) PyArray_DIMS(input)[0];
    Nch1 = (int) PyArray_DIMS(input)[1];
    cin = (double *) PyArray_DATA(input);

    /* create output ndarray */
    nout[0] = ((int) ceil(Nch2 / (float) Nav2));
    nout[1] = ((int) ceil(Nch1 / (float) Nav1));
    outarr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);

    #ifdef __OPENMP__
    /* set openmp thread numbers dynamically */
    OMPSETNUMTHREADS(nthreads);
    #endif

    #pragma omp parallel for default(shared) \
     private(i, j, k, l, buf) schedule(static)
    for (i = 0; i < Nch2; i = i + Nav2) {
        for (j = 0; j < Nch1; j = j + Nav1) {
            buf = 0.;
            for (k = 0; k < Nav2 && (i + k) < Nch2; ++k) {
                for (l = 0; l < Nav1 && (j + l) < Nch1; ++l) {
                    buf += cin[(i + k) * Nch1 + (j + l)];
                }
            }
            cout[(i / Nav2) * nout[1] + j / Nav1] = buf / (float)(k * l);
        }
    }

    /* clean up */
    Py_DECREF(input);

    return PyArray_Return(outarr);
}

PyObject* block_average_PSD(PyObject *self, PyObject *args) {
    /*    block average for a bunch of PSD spectra
     *
     *    Parameters
     *    ----------
     *    psd:          input array of PSD values
     *                  size = (Nspec, Nch) (in)
     *    Nav:          number of channels to average
     *    nthreads:     number of threads to use in parallel section
     *
     *    Returns
     *    -------
     *    block_av:     block averaged output array
     *                  size = (Nspec , ceil(Nch/Nav)) (out)
     */
    int i, j, k;  /* loop indices */
    int Nspec, Nch;  /* number of items in input */
    int Nav;  /* number of items to average */
    unsigned int nthreads;  /* number of threads to use */
    PyArrayObject *input=NULL, *outarr=NULL;
    double *cin, *cout;
    double buf;
    npy_intp nout[2];

    /* Python argument conversion code */
    if (!PyArg_ParseTuple(args, "O!iI", &PyArray_Type,
                          &input, &Nav, &nthreads)) {
        return NULL;
    }
    PYARRAY_CHECK(input, 2, NPY_DOUBLE, "input must be a 2D double array!");
    Nspec = (int) PyArray_DIMS(input)[0];
    Nch = (int) PyArray_DIMS(input)[1];
    cin = (double *) PyArray_DATA(input);

    /* create output ndarray */
    nout[0] = Nspec;
    nout[1] = ((int) ceil(Nch / (float) Nav));
    outarr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);

    #ifdef __OPENMP__
    /* set openmp thread numbers dynamically */
    OMPSETNUMTHREADS(nthreads);
    #endif

    /* c-code following is performing the block averaging */
    #pragma omp parallel for default(shared) private(i, j, k, buf) \
     schedule(static)
    for (i = 0; i < Nspec; ++i) {
        for (j = 0; j < Nch; j = j + Nav) {
            buf = 0;
            /* perform one block average
               (j-i serves as counter -> last bin is therefore correct) */
            for (k = j; k < j + Nav && k < Nch; ++k) {
                buf += cin[i * Nch + k];
            }
            /* save average to output array */
            cout[j / Nav + i * nout[1]] = buf / (float) (k - j);
        }
    }

    /* clean up */
    Py_DECREF(input);

    /* return output array */
    return PyArray_Return(outarr);
}

