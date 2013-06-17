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
 * Copyright (C) 2010-2011,2013 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#include <numpy/arrayobject.h>
#include <math.h>
#ifdef __OPENMP__
#include <omp.h>
#endif

PyObject* block_average1d(PyObject *self, PyObject *args) {
    /*    block average for one-dimensional double array
     *
     *    Parameters
     *    ----------
     *    input:        input array of datatype double (in)
     *    Nav:          number of items to average
     *
     *    Returns
     *    -------
     *    block_av:     block averaged output array
     *                  size = ceil(N/Nav) (out)
     *
     */

    int i,j,Nav,N;
    PyArrayObject *input=NULL, *outarr=NULL;
    double *cin,*cout;
    double buf;

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!i",&PyArray_Type, &input, &Nav)) return NULL; 
    if (PyArray_NDIM(input) != 1 || PyArray_TYPE(input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double");
        return NULL; 
    }
    N = PyArray_DIMS(input)[0];
    cin = (double *) PyArray_DATA(input);

    // create output ndarray
    npy_intp nout;
    nout = ((int)ceil(N/(float)Nav));
    outarr = (PyArrayObject *) PyArray_SimpleNew(1, &nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);
    
    // c-code following is performing the block averaging
    for(i=0; i<N; i=i+Nav) {
        buf=0;
        //perform one block average (j-i serves as counter -> last bin is therefore correct)
        for(j=i; j<i+Nav && j<N; ++j) {
            buf += cin[j];
        }
        cout[i/Nav] = buf/(float)(j-i); //save average to output array
    }
     
    // return output array
    return PyArray_Return(outarr);
}

PyObject* block_average2d(PyObject *self, PyObject *args) {
    /*    2D block average for one CCD frame
     *
     *    Parameters
     *    ----------
     *    ccd:          input array/CCD frame
     *                  size = (Nch2, Nch1) (in)
     *                  Nch1 is the fast variing index
     *    Nav1,2:       number of channels to average in each dimension
     *                  in total a block of Nav1 x Nav2 is averaged
     *
     *    Returns
     *    -------
     *    block_av:     block averaged output array
     *                  size = (ceil(Nch2/Nav2) , ceil(Nch1/Nav1)) (out)
     *
     */

    int i=0,j=0,k=0,l=0; //loop indices
    int Nch1,Nch2;  //number of values in input array
    int Nav1,Nav2;  //number of items to average
    PyArrayObject *input=NULL, *outarr=NULL;
    double *cin,*cout;
    double buf;

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!ii",&PyArray_Type, &input, &Nav2,&Nav1)) return NULL;
    
    if (PyArray_NDIM(input) != 2 || PyArray_TYPE(input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Nch2 = PyArray_DIMS(input)[0];
    Nch1 = PyArray_DIMS(input)[1];
    cin = (double *) PyArray_DATA(input);

    // create output ndarray
    npy_intp nout[2];
    nout[0] = ((int)ceil(Nch2/(float)Nav2));
    nout[1] = ((int)ceil(Nch1/(float)Nav1));
    outarr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr);

    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    #pragma omp parallel for default(shared) private(i,j,k,l,buf) schedule(static)
    for(i=0; i<Nch2; i=i+Nav2) {
        for(j=0; j<Nch1; j=j+Nav1) {
            buf = 0.;
            for(k=0; k<Nav2 && (i+k)<Nch2; ++k) {
                for(l=0; l<Nav1 && (j+l)<Nch1; ++l) {
                    buf += cin[(i+k)*Nch1+(j+l)];
                }
            }
            cout[(i/Nav2)*nout[1]+j/Nav1] = buf/(float)(k*l);
        }
    }

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
     *
     *    Returns
     *    -------
     *    block_av:     block averaged output array
     *                  size = (Nspec , ceil(Nch/Nav)) (out)
     */
    int i,j,k; //loop indices
    int Nspec, Nch;  //number of items in input
    int Nav;  //number of items to average
    PyArrayObject *input=NULL, *outarr=NULL;
    double *cin,*cout;
    double buf; 

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!i",&PyArray_Type, &input, &Nav)) return NULL;
    
    if (PyArray_NDIM(input) != 2 || PyArray_TYPE(input) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Nspec = PyArray_DIMS(input)[0];
    Nch = PyArray_DIMS(input)[1];
    cin = (double *) PyArray_DATA(input);

    // create output ndarray
    npy_intp nout[2];
    nout[0] = Nspec;
    nout[1] = ((int)ceil(Nch/(float)Nav));
    outarr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    cout = (double *) PyArray_DATA(outarr); 
    
    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    // c-code following is performing the block averaging
    #pragma omp parallel for default(shared) private(i,j,k,buf) schedule(static)
    for(i=0; i<Nspec; ++i) {
        for(j=0; j<Nch; j=j+Nav) {
            buf=0;
            //perform one block average (j-i serves as counter -> last bin is therefore correct)
            for(k=j; k<j+Nav && k<Nch; ++k) {
                buf += cin[i*Nch+k];
            }
            cout[j/Nav+i*nout[1]] = buf/(float)(k-j); //save average to output array
        }
    }

    // return output array
    return PyArray_Return(outarr);
}
