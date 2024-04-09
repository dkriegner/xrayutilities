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
 * Copyright (C) 2013, 2015 Dominik Kriegner <dominik.kriegner@gmail.com>
 *
 ******************************************************************************
 *
 * created: Jun 8, 2013
 * author: Eugen Wintersberger
*/

#include "gridder.h"
#include "gridder_utils.h"


PyObject* pyfuzzygridder2d(PyObject *self, PyObject *args)
{
    PyArrayObject *py_x = NULL, *py_y = NULL, *py_data = NULL,
                  *py_output = NULL, *py_norm = NULL;

    double *x = NULL, *y = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax, ymin, ymax, wx, wy;
    unsigned int nx, ny;
    int flags;
    int n, result;

    if (!PyArg_ParseTuple(args, "O!O!O!IIddddO!|O!ddi",
                          &PyArray_Type, &py_x,
                          &PyArray_Type, &py_y,
                          &PyArray_Type, &py_data,
                          &nx, &ny, &xmin, &xmax, &ymin, &ymax,
                          &PyArray_Type, &py_output,
                          &PyArray_Type, &py_norm,
                          &wx, &wy, &flags)) {
        return NULL;
    }

    /* have to check input variables */
    PYARRAY_CHECK(py_x, 1, NPY_DOUBLE, "x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_y, 1, NPY_DOUBLE, "y-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data, 1, NPY_DOUBLE,
                  "input data must be a 1D double array!");
    PYARRAY_CHECK(py_output, 2, NPY_DOUBLE,
                  "ouput data must be a 2D double array!");
    if (py_norm != NULL) {
        PYARRAY_CHECK(py_norm, 2, NPY_DOUBLE,
                      "norm data must be a 2D double array!");
    }

    /* get data */
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    data = (double *) PyArray_DATA(py_data);
    odata = (double *) PyArray_DATA(py_output);
    if (py_norm != NULL) {
        norm = (double *) PyArray_DATA(py_norm);
    }

    /* get the total number of points */
    n = (int) PyArray_SIZE(py_x);

    /* call the actual gridder routine */
    result = fuzzygridder2d(x, y, data, n, nx, ny, xmin, xmax, ymin, ymax, odata,
                            norm, wx, wy, flags);

    /* clean up */
    Py_DECREF(py_x);
    Py_DECREF(py_y);
    Py_DECREF(py_data);
    Py_DECREF(py_output);
    if (py_norm != NULL) {
        Py_DECREF(py_norm);
    }

    return Py_BuildValue("i", &result);
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
    PyArrayObject *py_x = NULL, *py_y = NULL, *py_data = NULL,
                  *py_output = NULL, *py_norm = NULL;

    double *x = NULL, *y = NULL, *data = NULL, *odata = NULL, *norm = NULL;
    double xmin, xmax, ymin, ymax;
    unsigned int nx, ny;
    int flags;
    int n, result;

    if (!PyArg_ParseTuple(args, "O!O!O!IIddddO!|O!i",
                          &PyArray_Type, &py_x,
                          &PyArray_Type, &py_y,
                          &PyArray_Type, &py_data,
                          &nx, &ny, &xmin, &xmax, &ymin, &ymax,
                          &PyArray_Type, &py_output,
                          &PyArray_Type, &py_norm,
                          &flags)) {
        return NULL;
    }

    /* have to check input variables */
    PYARRAY_CHECK(py_x, 1, NPY_DOUBLE, "x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_y, 1, NPY_DOUBLE, "y-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data, 1, NPY_DOUBLE,
                  "input data must be a 1D double array!");
    PYARRAY_CHECK(py_output, 2, NPY_DOUBLE,
                  "ouput data must be a 2D double array!");
    if (py_norm != NULL) {
        PYARRAY_CHECK(py_norm, 2, NPY_DOUBLE,
                      "norm data must be a 2D double array!");
    }

    /* get data */
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    data = (double *) PyArray_DATA(py_data);
    odata = (double *) PyArray_DATA(py_output);
    if (py_norm != NULL) {
        norm = (double *) PyArray_DATA(py_norm);
    }

    /* get the total number of points */
    n = (int) PyArray_SIZE(py_x);

    /* call the actual gridder routine */
    result = gridder2d(x, y, data, n, nx, ny, xmin, xmax, ymin, ymax, odata,
                       norm, flags);

    /* clean up */
    Py_DECREF(py_x);
    Py_DECREF(py_y);
    Py_DECREF(py_data);
    Py_DECREF(py_output);
    if (py_norm != NULL) {
        Py_DECREF(py_norm);
    }

    return Py_BuildValue("i", &result);
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

