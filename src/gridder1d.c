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
 * Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>
 *
 ******************************************************************************
 *
 * created: Sep 16, 2014
 * author: Dominik Kriegner
*/

#include "gridder.h"
#include "gridder_utils.h"

PyObject* pyfuzzygridder1d(PyObject *self, PyObject *args)
{
    PyArrayObject *py_x = NULL, *py_data = NULL,
                  *py_output = NULL, *py_norm = NULL;

    double *x = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax, fuzzywidth;
    unsigned int nx;
    int flags;
    int n, result;

    if (!PyArg_ParseTuple(args, "O!O!IddO!|O!di",
                         &PyArray_Type, &py_x,
                         &PyArray_Type, &py_data,
                         &nx, &xmin, &xmax,
                         &PyArray_Type, &py_output,
                         &PyArray_Type, &py_norm,
                         &fuzzywidth, &flags))
        return NULL;

    /* have to check input variables */
    PYARRAY_CHECK(py_x, 1, NPY_DOUBLE, "x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data, 1, NPY_DOUBLE,
                  "input data must be a 1D double array!");
    PYARRAY_CHECK(py_output, 1, NPY_DOUBLE,
                  "ouput data must be a 1D double array!");
    if (py_norm != NULL)
        PYARRAY_CHECK(py_norm, 1, NPY_DOUBLE,
                      "norm data must be a 1D double array!");

    /* get data */
    x = (double *) PyArray_DATA(py_x);
    data = (double *) PyArray_DATA(py_data);
    odata = (double *) PyArray_DATA(py_output);
    if (py_norm != NULL) {
        norm = (double *) PyArray_DATA(py_norm);
    }

    /* get the total number of points */
    n = (int) PyArray_SIZE(py_x);

    /* call the actual gridder routine */
    result = fuzzygridder1d(x, data, n, nx, xmin, xmax, odata,
                            norm, fuzzywidth, flags);

    /* clean up */
    Py_DECREF(py_x);
    Py_DECREF(py_data);
    Py_DECREF(py_output);
    if (py_norm != NULL) {
        Py_DECREF(py_norm);
    }

    return Py_BuildValue("i", &result);
}

/*---------------------------------------------------------------------------*/
int fuzzygridder1d(double *x, double *data, unsigned int n,
                   unsigned int nx,
                   double xmin, double xmax,
                   double *odata, double *norm, double fuzzywidth, int flags)
{
    double *gnorm;
    unsigned int offset1, offset2;
    unsigned int noutofbounds = 0;  /* counter for out of bounds points */

    double dx = delta(xmin, xmax, nx);
    double fraction, dwidth; /* fuzzy fraction and data width */

    unsigned int i, j; /* loop indices */

    /* initialize data if requested */
    if (!(flags & NO_DATA_INIT)) set_array(odata, nx, 0.);

    /* check if normalization array is passed */
    if (norm == NULL) {
        gnorm = malloc(sizeof(double) * nx);
        if (gnorm == NULL) {
            fprintf(stderr, "XU.FuzzyGridder1D(c): Cannot allocate memory for "
                            "normalization buffer!\n");
            return -1;
        }
        /* initialize memory for norm */
        set_array(gnorm, nx, 0.);
    }
    else {
        if (flags & VERBOSE) {
            fprintf(stdout, "XU.FuzzyGridder1D(c): use user provided buffer "
                            "for normalization data\n");
        }
        gnorm = norm;
    }

    /* the master loop over all data points */
    dwidth = fuzzywidth / dx;
    if (flags & VERBOSE) {
        fprintf(stdout, "XU.FuzzyGridder1D(c): fuzzyness: %f %f\n",
                fuzzywidth, dwidth);
    }
    for (i = 0; i < n; i++) {
        /* if data point is nan ignore it */
        if (!isnan(data[i])) {
            /* if the x value is outside the grid boundaries continue with
             * the next point */
            if ((x[i] < (xmin - fuzzywidth/2.)) || (x[i] > xmax + fuzzywidth/2.)) {
                noutofbounds++;
                continue;
            }
            /* compute the linear offset and distribute the data to the bins */
            if ((x[i] - fuzzywidth / 2.) <= xmin) {
                offset1 = 0;
            }
            else {
                offset1 = gindex(x[i] - fuzzywidth / 2., xmin, dx);
            }
            offset2 = gindex(x[i] + fuzzywidth / 2., xmin, dx);
            offset2 = offset2 < nx ? offset2 : nx - 1;
            for(j = offset1; j <= offset2; j++) {
                if (offset1 == offset2) {
                    fraction = 1.;
                }
                else if (j == offset1) {
                    fraction = (j + 1 - (x[i] - fuzzywidth / 2. - xmin + dx / 2.) / dx) / dwidth;
                }
                else if (j == offset2) {
                    fraction = ((x[i] + fuzzywidth / 2. - xmin + dx / 2.) / dx - j) / dwidth;
                }
                else {
                    fraction = 1 / dwidth;
                }
                odata[j] += data[i]*fraction;
                gnorm[j] += fraction;
            }
        }
    }

    /* perform normalization */
    if (!(flags & NO_NORMALIZATION)) {
        if (flags & VERBOSE) {
            fprintf(stdout, "XU.FuzzyGridder1D(c): perform normalization\n");
        }

        for (i = 0; i < nx; i++) {
            if (gnorm[i] > 1.e-16) {
                odata[i] = odata[i] / gnorm[i];
            }
        }
    }

    /* free the norm buffer if it has been locally allocated */
    if (norm == NULL) free(gnorm);

    /* warn the user in case more than half the data points where out
     *  of the gridding area */
    if (flags & VERBOSE) {
        if (noutofbounds > n / 2) {
            fprintf(stdout, "XU.FuzzyGridder1D(c): more than half of the "
                        "datapoints out of the data range, consider regridding"
                        " with extended range!\n");
        }
        else {
            fprintf(stdout, "XU.FuzzyGridder1D(c): %d datapoints out of the data "
                        "range!\n", noutofbounds);
        }
    }

    return 0;
}

/*---------------------------------------------------------------------------*/
PyObject* pygridder1d(PyObject *self, PyObject *args)
{
    PyArrayObject *py_x = NULL, *py_data = NULL,
                  *py_output = NULL, *py_norm = NULL;

    double *x = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax;
    unsigned int nx;
    int flags;
    int n, result;

    if (!PyArg_ParseTuple(args, "O!O!IddO!|O!i",
                         &PyArray_Type, &py_x,
                         &PyArray_Type, &py_data,
                         &nx, &xmin, &xmax,
                         &PyArray_Type, &py_output,
                         &PyArray_Type, &py_norm,
                         &flags))
        return NULL;

    /* have to check input variables */
    PYARRAY_CHECK(py_x, 1, NPY_DOUBLE, "x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data, 1, NPY_DOUBLE,
                  "input data must be a 1D double array!");
    PYARRAY_CHECK(py_output, 1, NPY_DOUBLE,
                  "ouput data must be a 1D double array!");
    if (py_norm != NULL)
        PYARRAY_CHECK(py_norm, 1, NPY_DOUBLE,
                      "norm data must be a 1D double array!");

    /* get data */
    x = (double *) PyArray_DATA(py_x);
    data = (double *) PyArray_DATA(py_data);
    odata = (double *) PyArray_DATA(py_output);
    if (py_norm != NULL) {
        norm = (double *) PyArray_DATA(py_norm);
    }

    /* get the total number of points */
    n = (int) PyArray_SIZE(py_x);

    /* call the actual gridder routine */
    result = gridder1d(x, data, n, nx, xmin, xmax, odata, norm, flags);

    /* clean up */
    Py_DECREF(py_x);
    Py_DECREF(py_data);
    Py_DECREF(py_output);
    if (py_norm != NULL) {
        Py_DECREF(py_norm);
    }

    return Py_BuildValue("i", &result);
}

/*---------------------------------------------------------------------------*/
int gridder1d(double *x, double *data, unsigned int n,
              unsigned int nx,
              double xmin, double xmax,
              double *odata, double *norm, int flags)
{
    double *gnorm;
    unsigned int offset;
    unsigned int noutofbounds = 0;  /* counter for out of bounds points */

    double dx = delta(xmin, xmax, nx);

    unsigned int i; /* loop index */

    /* initialize data if requested */
    if (!(flags & NO_DATA_INIT)) set_array(odata, nx, 0.);

    /* check if normalization array is passed */
    if (norm == NULL) {
        gnorm = malloc(sizeof(double) * nx);
        if (gnorm == NULL) {
            fprintf(stderr, "XU.Gridder1D(c): Cannot allocate memory for "
                            "normalization buffer!\n");
            return -1;
        }
        /* initialize memory for norm */
        set_array(gnorm, nx, 0.);
    }
    else {
        if (flags & VERBOSE) {
            fprintf(stdout, "XU.Gridder1D(c): use user provided buffer for "
                            "normalization data\n");
        }
        gnorm = norm;
    }

    /* the master loop over all data points */
    for (i = 0; i < n; i++) {
        /* if data point is nan ignore it */
        if (!isnan(data[i])) {
            /* if the x value is outside the grid boundaries continue with
             * the next point */
            if ((x[i] < xmin) || (x[i] > xmax)) {
                noutofbounds++;
                continue;
            }
            /* compute the linear offset and set the data */
            offset = gindex(x[i], xmin, dx);

            odata[offset] += data[i];
            gnorm[offset] += 1.;
        }
    }

    /* perform normalization */
    if (!(flags & NO_NORMALIZATION)) {
        if (flags & VERBOSE) {
            fprintf(stdout, "XU.Gridder1D(c): perform normalization ...\n");
        }

        for (i = 0; i < nx; i++) {
            if (gnorm[i] > 1.e-16) {
                odata[i] = odata[i] / gnorm[i];
            }
        }
    }

    /* free the norm buffer if it has been locally allocated */
    if (norm == NULL) free(gnorm);

    /* warn the user in case more than half the data points where out
     *  of the gridding area */
    if (flags & VERBOSE) {
        if (noutofbounds > n / 2) {
            fprintf(stdout, "XU.Gridder1D(c): more than half of the "
                        "datapoints out of the data range, consider regridding"
                        " with extended range!\n");
        }
        else {
            fprintf(stdout, "XU.Gridder1D(c): %d datapoints out of the data "
                        "range!\n", noutofbounds);
        }
    }

    return 0;
}

