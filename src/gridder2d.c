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
 * Copyright (C) 2013-2025 Dominik Kriegner <dominik.kriegner@gmail.com>
 *
 ******************************************************************************
 *
 * created: Jun 8, 2013
 * author: Eugen Wintersberger
*/

#include "gridder.h"
#include "gridder_utils.h"


PyObject* pyfuzzygridder2d(PyObject *self, PyObject *args) {
    PyArrayObject *px = NULL, *py = NULL, *pdata = NULL, *poutput = NULL, *pnorm = NULL;
    PyObject *xobj = NULL, *yobj = NULL, *dataobj = NULL, *outputobj = NULL, *normobj = NULL;
    double *x = NULL, *y = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax, ymin, ymax, wx, wy;
    unsigned int nx, ny;
    int flags;
    int n, result;
    PyObject *return_value = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!IIddddO!|O!ddi",
                         &PyArray_Type, &xobj,
                         &PyArray_Type, &yobj,
                         &PyArray_Type, &dataobj,
                         &nx, &ny, &xmin, &xmax, &ymin, &ymax,
                         &PyArray_Type, &outputobj,
                         &PyArray_Type, &normobj,
                         &wx, &wy, &flags)) {
        return NULL; // Return NULL directly on parse error
    }

    px = check_and_convert_to_contiguous(xobj, 1, NPY_DOUBLE, "x-axis");
    if (!px) goto cleanup;

    py = check_and_convert_to_contiguous(yobj, 1, NPY_DOUBLE, "y-axis");
    if (!py) goto cleanup;

    pdata = check_and_convert_to_contiguous(dataobj, 1, NPY_DOUBLE, "input data");
    if (!pdata) goto cleanup;

    poutput = check_and_convert_to_contiguous(outputobj, 2, NPY_DOUBLE, "output data");
    if (!poutput) goto cleanup;

    if (normobj != NULL) {
        pnorm = check_and_convert_to_contiguous(normobj, 2, NPY_DOUBLE, "norm");
        if (!pnorm) goto cleanup;
    }

    x = (double *)PyArray_DATA(px);
    y = (double *)PyArray_DATA(py);
    data = (double *)PyArray_DATA(pdata);
    odata = (double *)PyArray_DATA(poutput);
    if (pnorm != NULL) {
        norm = (double *)PyArray_DATA(pnorm);
    }

    n = (int)PyArray_SIZE(px);

    result = fuzzygridder2d(x, y, data, n, nx, ny, xmin, xmax, ymin, ymax, odata, norm, wx, wy, flags);

    return_value = Py_BuildValue("i", result);

cleanup:
    Py_XDECREF(pnorm);
    Py_XDECREF(poutput);
    Py_XDECREF(pdata);
    Py_XDECREF(py);
    Py_XDECREF(px);

    return return_value;
}

/*--------------------------------------------------------------------------*/
int fuzzygridder2d(double *x, double *y, double *data, unsigned int n,
                   unsigned int nx, unsigned int ny,
                   double xmin, double xmax, double ymin, double ymax,
                   double *odata, double *norm, double wx, double wy,
                   int flags)
{
    double *gnorm;
    unsigned int offset, offsetx1, offsetx2, offsety1, offsety2;
    unsigned int ntot = nx * ny;  /* total number of points on the grid */
    unsigned int noutofbounds = 0;  /* number of points out of bounds */

    double fractionx, fractiony, dwx, dwy;  /* variables for the fuzzy part */
    double dx = delta(xmin, xmax, nx);
    double dy = delta(ymin, ymax, ny);

    unsigned int i, j, k;  /* loop indices */

    /* initialize data if requested */
    if (!(flags & NO_DATA_INIT)) {
        set_array(odata, ntot, 0.);
    }

    /* check if normalization array is passed */
    if (norm == NULL) {
        gnorm = malloc(sizeof(double) * (nx * ny));
        if (gnorm == NULL) {
            fprintf(stderr, "XU.FuzzyGridder2D(c): Cannot allocate memory for"
                            " normalization buffer!\n");
            return -1;
        }
        /* initialize memory for norm */
        set_array(gnorm, nx * ny, 0.);
    }
    else {
        if (flags & VERBOSE) {
            fprintf(stdout, "XU.FuzzyGridder2D(c): use user provided buffer "
                            "for normalization data\n");
        }
        gnorm = norm;
    }

    /* calculate the fuzzy spread in number of bin sizes */
    dwx = wx / dx;
    dwy = wy / dy;
    if (flags & VERBOSE) {
        fprintf(stdout, "XU.FuzzyGridder2D(c): fuzzyness: %f %f %f %f\n",
                wx, wy, dwx, dwy);
    }
    /* the master loop over all data points */
    for (i = 0; i < n; i++) {
        /* if data point is nan ignore it */
        if (!isnan(data[i])) {
            /* if the x and y values are outside the grids boundaries
             * continue with the next point */
            if ((x[i] < xmin) || (x[i] > xmax)) {
                noutofbounds++;
                continue;
            }
            if ((y[i] < ymin) || (y[i] > ymax)) {
                noutofbounds++;
                continue;
            }
            /* compute the linear offset and distribute the data to the bins */
            if ((x[i] - wx / 2.) <= xmin) {
                offsetx1 = 0;
            }
            else {
                offsetx1 = gindex(x[i] - wx / 2., xmin, dx);
            }
            offsetx2 = gindex(x[i] + wx / 2., xmin, dx);
            offsetx2 = offsetx2 < nx ? offsetx2 : nx - 1;
            if ((y[i] - wy / 2.) <= ymin) {
                offsety1 = 0;
            }
            else {
                offsety1 = gindex(y[i] - wy / 2., ymin, dy);
            }
            offsety2 = gindex(y[i] + wy / 2., ymin, dy);
            offsety2 = offsety2 < ny ? offsety2 : ny - 1;

            for(j = offsetx1; j <= offsetx2; j++) {
                if (offsetx1 == offsetx2) {
                    fractionx = 1.;
                }
                else if (j == offsetx1) {
                    fractionx = (j + 1 - (x[i] - wx / 2. - xmin + dx / 2.) / dx) / dwx;
                }
                else if (j == offsetx2) {
                    fractionx = ((x[i] + wx / 2. - xmin + dx / 2.) / dx - j) / dwx;
                }
                else {
                    fractionx = 1 / dwx;
                }

                for(k = offsety1; k <= offsety2; k++) {
                    if (offsety1 == offsety2) {
                        fractiony = 1.;
                    }
                    else if (k == offsety1) {
                        fractiony = (k + 1 - (y[i] - wy / 2. - ymin + dy / 2.) / dy) / dwy;
                    }
                    else if (k == offsety2) {
                        fractiony = ((y[i] + wy / 2. - ymin + dy / 2.) / dy - k) / dwy;
                    }
                    else {
                        fractiony = 1 / dwy;
                    }

                    offset = j * ny + k;
                    odata[offset] += data[i]*fractionx*fractiony;
                    gnorm[offset] += fractionx*fractiony;
                }
            }
        }
    }

    /* perform normalization */
    if (!(flags & NO_NORMALIZATION)) {
        if (flags & VERBOSE)
            fprintf(stdout, "XU.FuzzyGridder2D(c): perform normalization\n");

        for (i = 0; i < nx * ny; i++) {
            if (gnorm[i] > 1.e-16) {
                odata[i] = odata[i] / gnorm[i];
            }
        }
    }

    /* free the norm buffer if it has been locally allocated */
    if (norm == NULL) {
        free(gnorm);
    }

    /* warn the user in case more than half the data points where out
     * of the gridding area */
    if (flags & VERBOSE) {
        if (noutofbounds > n / 2) {
            fprintf(stdout,"XU.FuzzyGridder2D(c): more than half of the datapoints"
                        " out of the data range, consider regridding with"
                        " extended range!\n");
        }
        else {
            fprintf(stdout, "XU.FuzzyGridder2D(c): %d datapoints out of the data "
                        "range!\n", noutofbounds);
        }
    }

    return 0;
}


/*---------------------------------------------------------------------------*/
PyObject* pygridder2d(PyObject *self, PyObject *args)
{
    PyArrayObject *px = NULL, *py = NULL, *pdata = NULL, *poutput = NULL, *pnorm = NULL;
    PyObject *xobj = NULL, *yobj = NULL, *dataobj = NULL, *outputobj = NULL, *normobj = NULL;
    double *x = NULL, *y = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax, ymin, ymax;
    unsigned int nx, ny;
    int flags;
    int n, result;
    PyObject *return_value = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!IIddddO!|O!i",
                         &PyArray_Type, &xobj,
                         &PyArray_Type, &yobj,
                         &PyArray_Type, &dataobj,
                         &nx, &ny, &xmin, &xmax, &ymin, &ymax,
                         &PyArray_Type, &outputobj,
                         &PyArray_Type, &normobj,
                         &flags)) {
        return NULL;
    }

    px = check_and_convert_to_contiguous(xobj, 1, NPY_DOUBLE, "x-axis");
    if (!px) goto cleanup;

    py = check_and_convert_to_contiguous(yobj, 1, NPY_DOUBLE, "y-axis");
    if (!py) goto cleanup;

    pdata = check_and_convert_to_contiguous(dataobj, 1, NPY_DOUBLE, "input data");
    if (!pdata) goto cleanup;

    poutput = check_and_convert_to_contiguous(outputobj, 2, NPY_DOUBLE, "output data");
    if (!poutput) goto cleanup;

    if (normobj != NULL) {
        pnorm = check_and_convert_to_contiguous(normobj, 2, NPY_DOUBLE, "norm");
        if (!pnorm) goto cleanup;
    }

    x = (double *)PyArray_DATA(px);
    y = (double *)PyArray_DATA(py);
    data = (double *)PyArray_DATA(pdata);
    odata = (double *)PyArray_DATA(poutput);
    if (pnorm != NULL) {
        norm = (double *)PyArray_DATA(pnorm);
    }

    n = (int)PyArray_SIZE(px);

    result = gridder2d(x, y, data, n, nx, ny, xmin, xmax, ymin, ymax, odata,
                      norm, flags);

    return_value = Py_BuildValue("i", result);

cleanup:
    Py_XDECREF(pnorm);
    Py_XDECREF(poutput);
    Py_XDECREF(pdata);
    Py_XDECREF(py);
    Py_XDECREF(px);

    return return_value;
}

/*--------------------------------------------------------------------------*/
int gridder2d(double *x, double *y, double *data, unsigned int n,
              unsigned int nx, unsigned int ny,
              double xmin, double xmax, double ymin, double ymax,
              double *odata, double *norm, int flags)
{
    double *gnorm;
    unsigned int offset;
    unsigned int ntot = nx * ny;  /* total number of points on the grid */
    unsigned int noutofbounds = 0;  /* number of points out of bounds */

    double dx = delta(xmin, xmax, nx);
    double dy = delta(ymin, ymax, ny);

    unsigned int i;  /* loop index */

    /* initialize data if requested */
    if (!(flags & NO_DATA_INIT)) {
        set_array(odata, ntot, 0.);
    }

    /* check if normalization array is passed */
    if (norm == NULL) {
        gnorm = malloc(sizeof(double) * (nx * ny));
        if (gnorm == NULL) {
            fprintf(stderr, "XU.Gridder2D(c): Cannot allocate memory for "
                            "normalization buffer!\n");
            return -1;
        }
        /* initialize memory for norm */
        set_array(gnorm, nx * ny, 0.);
    }
    else {
        if (flags & VERBOSE) {
            fprintf(stdout, "XU.Gridder2D(c): use user provided buffer for "
                            "normalization data\n");
        }
        gnorm = norm;
    }

    /* the master loop over all data points */
    for (i = 0; i < n; i++) {
        /* if data point is nan ignore it */
        if (!isnan(data[i])) {
            /* if the x and y values are outside the grids boundaries
             * continue with the next point */
            if ((x[i] < xmin) || (x[i] > xmax)) {
                noutofbounds++;
                continue;
            }
            if ((y[i] < ymin) || (y[i] > ymax)) {
                noutofbounds++;
                continue;
            }
            /* compute the linear offset and set the data */
            offset = gindex(x[i], xmin, dx) * ny + gindex(y[i], ymin, dy);

            odata[offset] += data[i];
            gnorm[offset] += 1.;
        }
    }

    /* perform normalization */
    if (!(flags & NO_NORMALIZATION)) {
        if (flags & VERBOSE)
            fprintf(stdout, "XU.Gridder2D(c): perform normalization ...\n");

        for (i = 0; i < nx * ny; i++) {
            if (gnorm[i] > 1.e-16) {
                odata[i] = odata[i] / gnorm[i];
            }
        }
    }

    /* free the norm buffer if it has been locally allocated */
    if (norm == NULL) {
        free(gnorm);
    }

    /* warn the user in case more than half the data points where out
     * of the gridding area */
    if (flags & VERBOSE) {
        if (noutofbounds > n / 2) {
            fprintf(stdout,"XU.Gridder2D(c): more than half of the datapoints"
                        " out of the data range, consider regridding with"
                        " extended range!\n");
        }
        else {
            fprintf(stdout, "XU.Gridder2D(c): %d datapoints out of the data "
                        "range!\n", noutofbounds);
        }
    }

    return 0;
}
