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
 * Copyright (C) 2014-2025 Dominik Kriegner <dominik.kriegner@gmail.com>
 *
 ******************************************************************************
 *
 * created: Sep 16, 2014
 * author: Dominik Kriegner
*/

#include "gridder.h"
#include "gridder_utils.h"


PyObject* pyfuzzygridder1d(PyObject *self, PyObject *args) {
    PyArrayObject *px = NULL, *pdata = NULL, *poutput = NULL, *pnorm = NULL;
    PyObject *xobj = NULL, *dataobj = NULL, *outputobj = NULL, *normobj = NULL;
    double *x = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax, fuzzywidth;
    unsigned int nx;
    int flags;
    int n, result;
    PyObject *return_value = NULL;

    if (!PyArg_ParseTuple(args, "O!O!IddO!|O!di",
                         &PyArray_Type, &xobj,
                         &PyArray_Type, &dataobj,
                         &nx, &xmin, &xmax,
                         &PyArray_Type, &outputobj,
                         &PyArray_Type, &normobj,
                         &fuzzywidth, &flags)) {
        return NULL; // Return NULL directly if parsing fails
    }

    px = check_and_convert_to_contiguous(xobj, 1, NPY_DOUBLE, "x-axis");
    if (!px) goto cleanup;

    pdata = check_and_convert_to_contiguous(dataobj, 1, NPY_DOUBLE, "input data");
    if (!pdata) goto cleanup;

    poutput = check_and_convert_to_contiguous(outputobj, 1, NPY_DOUBLE, "output data");
    if (!poutput) goto cleanup;

    if (normobj != NULL) {
        pnorm = check_and_convert_to_contiguous(normobj, 1, NPY_DOUBLE, "norm");
        if (!pnorm) goto cleanup;
    }

    x = (double *)PyArray_DATA(px);
    data = (double *)PyArray_DATA(pdata);
    odata = (double *)PyArray_DATA(poutput);
    if (pnorm != NULL) {
        norm = (double *)PyArray_DATA(pnorm);
    }

    n = (int)PyArray_SIZE(px);

    result = fuzzygridder1d(x, data, n, nx, xmin, xmax, odata, norm, fuzzywidth, flags);

    return_value = Py_BuildValue("i", result);

cleanup:
    Py_XDECREF(pnorm);
    Py_XDECREF(poutput);
    Py_XDECREF(pdata);
    Py_XDECREF(px);
    return return_value;
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
    PyArrayObject *px = NULL, *pdata = NULL, *poutput = NULL, *pnorm = NULL;
    PyObject *xobj = NULL, *dataobj = NULL, *outputobj = NULL, *normobj = NULL;
    double *x = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax;
    unsigned int nx;
    int flags;
    int n, result;
    PyObject *return_value = NULL;

    if (!PyArg_ParseTuple(args, "O!O!IddO!|O!i",
                         &PyArray_Type, &xobj,
                         &PyArray_Type, &dataobj,
                         &nx, &xmin, &xmax,
                         &PyArray_Type, &outputobj,
                         &PyArray_Type, &normobj,
                         &flags)) {
        return NULL;
    }

    px = check_and_convert_to_contiguous(xobj, 1, NPY_DOUBLE, "x-axis");
    if (!px) goto cleanup;

    pdata = check_and_convert_to_contiguous(dataobj, 1, NPY_DOUBLE, "input data");
    if (!pdata) goto cleanup;

    poutput = check_and_convert_to_contiguous(outputobj, 1, NPY_DOUBLE, "output data");
    if (!poutput) goto cleanup;

    if (normobj != NULL) {
        pnorm = check_and_convert_to_contiguous(normobj, 1, NPY_DOUBLE, "norm");
        if (!pnorm) goto cleanup;
    }

    x = (double *)PyArray_DATA(px);
    data = (double *)PyArray_DATA(pdata);
    odata = (double *)PyArray_DATA(poutput);
    if (pnorm != NULL) {
        norm = (double *)PyArray_DATA(pnorm);
    }

    n = (int)PyArray_SIZE(px);

    result = gridder1d(x, data, n, nx, xmin, xmax, odata, norm, flags);

    return_value = Py_BuildValue("i", result);

cleanup:
    Py_XDECREF(pnorm);
    Py_XDECREF(poutput);
    Py_XDECREF(pdata);
    Py_XDECREF(px);
    return return_value;
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
