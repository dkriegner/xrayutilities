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
 * created: Jun 21, 2013
 * author: Eugen Wintersberger
*/

#include "gridder.h"
#include "gridder_utils.h"

PyObject* pyfuzzygridder3d(PyObject *self, PyObject *args)
{
    PyArrayObject *py_x = NULL, *py_y = NULL, *py_z = NULL, *py_data = NULL,
                  *py_output = NULL, *py_norm = NULL;

    double *x = NULL, *y = NULL, *z = NULL, *data = NULL, *odata = NULL,
           *norm = NULL;
    double xmin, xmax, ymin, ymax, zmin, zmax, wx, wy, wz;
    unsigned int nx, ny, nz;
    int flags;
    int n, result;

    if (!PyArg_ParseTuple(args, "O!O!O!O!IIIddddddO!|O!dddi",
                         &PyArray_Type, &py_x,
                         &PyArray_Type, &py_y,
                         &PyArray_Type, &py_z,
                         &PyArray_Type, &py_data,
                         &nx, &ny, &nz,
                         &xmin, &xmax, &ymin, &ymax, &zmin, &zmax,
                         &PyArray_Type, &py_output,
                         &PyArray_Type, &py_norm,
                         &wx, &wy, &wz, &flags)) {
        return NULL;
    }

    /* check input variables */
    PYARRAY_CHECK(py_x, 1, NPY_DOUBLE, "x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_y, 1, NPY_DOUBLE, "y-axis must be a 1D double array!");
    PYARRAY_CHECK(py_z, 1, NPY_DOUBLE, "z-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data, 1, NPY_DOUBLE,
                  "input data must be a 1D double array!");
    PYARRAY_CHECK(py_output, 3, NPY_DOUBLE,
                  "ouput data must be a 2D double array!");
    if (py_norm != NULL) {
        PYARRAY_CHECK(py_norm, 3, NPY_DOUBLE,
                      "norm data must be a 2D double array!");
    }

    /* get data */
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);
    data = (double *) PyArray_DATA(py_data);
    odata = (double *) PyArray_DATA(py_output);
    if (py_norm != NULL) {
        norm = (double *) PyArray_DATA(py_norm);
    }

    /* get the total number of points */
    n = (int) PyArray_SIZE(py_x);

    /* call the actual gridder routine */
    result = fuzzygridder3d(x, y, z, data, n, nx, ny, nz,
                            xmin, xmax, ymin, ymax, zmin, zmax, odata, norm,
                            wx, wy, wz, flags);

    /* clean up */
    Py_DECREF(py_x);
    Py_DECREF(py_y);
    Py_DECREF(py_z);
    Py_DECREF(py_data);
    Py_DECREF(py_output);
    if (py_norm != NULL) {
        Py_DECREF(py_norm);
    }

    return Py_BuildValue("i", &result);
}

/*---------------------------------------------------------------------------*/
int fuzzygridder3d(double *x, double *y, double *z, double *data,
                   unsigned int n, unsigned int nx, unsigned int ny,
                   unsigned int nz, double xmin, double xmax, double ymin,
                   double ymax, double zmin, double zmax,
                   double *odata, double *norm,
                   double wx, double wy, double wz, int flags)
{
    double *gnorm;                     /* pointer to normalization data */
    unsigned int offset, offsetx1, offsetx2, offsety1, offsety2, offsetz1, offsetz2;
    unsigned int ntot = nx * ny * nz;  /* total number of points on the grid */
    unsigned int i, j, k, l;          /* loop indeces variables */
    unsigned int noutofbounds = 0;     /* number of points out of bounds */

    double fractionx, fractiony, fractionz, dwx, dwy, dwz;  /* variables for
                                                               the fuzziness */
    /* compute step width for the grid */
    double dx = delta(xmin, xmax, nx);
    double dy = delta(ymin, ymax, ny);
    double dz = delta(zmin, zmax, nz);

    /* initialize data if requested */
    if (!(flags & NO_DATA_INIT)) {
        set_array(odata, ntot, 0.);
    }

    /* check if normalization array is passed */
    if (norm == NULL) {
        gnorm = malloc(sizeof(double) * ntot);
        if (gnorm == NULL) {
            fprintf(stderr, "XU.FuzzyGridder3D(c): Cannot allocate memory for "
                            "normalization buffer!\n");
            return -1;
        }
        /* initialize memory for norm */
        set_array(gnorm, ntot, 0.);
    }
    else {
        gnorm = norm;
    }

    /* calculate the fuzzy spread in number of bin sizes */
    dwx = wx / dx;
    dwy = wy / dy;
    dwz = wz / dz;
    if (flags & VERBOSE) {
        fprintf(stdout, "XU.FuzzyGridder3D(c): fuzzyness: %f %f %f %f %f %f\n",
                wx, wy, wz, dwx, dwy, dwz);
    }
    /* the master loop over all data points */
    for (i = 0; i < n; i++) {
        if (!isnan(data[i])) {
            /* check if the current point is within the bounds of the grid */
            if ((x[i] < xmin) || (x[i] > xmax)) {
                noutofbounds++;
                continue;
            }
            if ((y[i] < ymin) || (y[i] > ymax)) {
                noutofbounds++;
                continue;
            }
            if ((z[i] < zmin) || (z[i] > zmax)) {
                noutofbounds++;
                continue;
            }

            /* compute the offset value of the current input point on the
             * grid array */
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
            if ((z[i] - wz / 2.) <= zmin) {
                offsetz1 = 0;
            }
            else {
                offsetz1 = gindex(z[i] - wz / 2., zmin, dz);
            }
            offsetz2 = gindex(z[i] + wz / 2., zmin, dz);
            offsetz2 = offsetz2 < nz ? offsetz2 : nz - 1;

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
                    for(l = offsetz1; l <= offsetz2; l++) {
                        if (offsetz1 == offsetz2) {
                            fractionz = 1.;
                        }
                        else if (l == offsetz1) {
                            fractionz = (l + 1 - (z[i] - wz / 2. - zmin + dz / 2.) / dz) / dwz;
                        }
                        else if (l == offsetz2) {
                            fractionz = ((z[i] + wz / 2. - zmin + dz / 2.) / dz - l) / dwz;
                        }
                        else {
                            fractionz = 1 / dwz;
                        }

                        offset = j * ny * nz + k * nz + l;
                        odata[offset] += data[i]*fractionx*fractiony*fractionz;
                        gnorm[offset] += fractionx*fractiony*fractionz;
                    }
                }
            }
        }
    }

    /* perform normalization */
    if (!(flags & NO_NORMALIZATION)) {
        for (i = 0; i < ntot; i++) {
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
            fprintf(stdout,"XU.FuzzyGridder3D(c): more than half of the datapoints"
                        " out of the data range, consider regridding with"
                        " extended range!\n");
        }
        else {
            fprintf(stdout, "XU.FuzzyGridder3D(c): %d datapoints out of the data "
                        "range!\n", noutofbounds);
        }
    }

    return 0;
}


/*---------------------------------------------------------------------------*/
PyObject* pygridder3d(PyObject *self, PyObject *args)
{
    PyArrayObject *py_x = NULL, *py_y = NULL, *py_z = NULL, *py_data = NULL,
                  *py_output = NULL, *py_norm = NULL;

    double *x = NULL, *y = NULL, *z = NULL, *data = NULL, *odata = NULL,
           *norm = NULL;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    unsigned int nx, ny, nz;
    int flags;
    int n, result;

    if (!PyArg_ParseTuple(args, "O!O!O!O!IIIddddddO!|O!i",
                         &PyArray_Type, &py_x,
                         &PyArray_Type, &py_y,
                         &PyArray_Type, &py_z,
                         &PyArray_Type, &py_data,
                         &nx, &ny, &nz,
                         &xmin, &xmax, &ymin, &ymax, &zmin, &zmax,
                         &PyArray_Type, &py_output,
                         &PyArray_Type, &py_norm,
                         &flags)) {
        return NULL;
    }

    /* check input variables */
    PYARRAY_CHECK(py_x, 1, NPY_DOUBLE, "x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_y, 1, NPY_DOUBLE, "y-axis must be a 1D double array!");
    PYARRAY_CHECK(py_z, 1, NPY_DOUBLE, "z-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data, 1, NPY_DOUBLE,
                  "input data must be a 1D double array!");
    PYARRAY_CHECK(py_output, 3, NPY_DOUBLE,
                  "ouput data must be a 2D double array!");
    if (py_norm != NULL) {
        PYARRAY_CHECK(py_norm, 3, NPY_DOUBLE,
                      "norm data must be a 2D double array!");
    }

    /* get data */
    x = (double *) PyArray_DATA(py_x);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);
    data = (double *) PyArray_DATA(py_data);
    odata = (double *) PyArray_DATA(py_output);
    if (py_norm != NULL) {
        norm = (double *) PyArray_DATA(py_norm);
    }

    /* get the total number of points */
    n = (int) PyArray_SIZE(py_x);

    /* call the actual gridder routine */
    result = gridder3d(x, y, z, data, n, nx, ny, nz,
                       xmin, xmax, ymin, ymax, zmin, zmax, odata, norm, flags);

    /* clean up */
    Py_DECREF(py_x);
    Py_DECREF(py_y);
    Py_DECREF(py_z);
    Py_DECREF(py_data);
    Py_DECREF(py_output);
    if (py_norm != NULL) {
        Py_DECREF(py_norm);
    }

    return Py_BuildValue("i", &result);
}

/*---------------------------------------------------------------------------*/
int gridder3d(double *x, double *y, double *z, double *data, unsigned int n,
              unsigned int nx, unsigned int ny, unsigned int nz,
              double xmin, double xmax, double ymin, double ymax,
              double zmin, double zmax,
              double *odata, double *norm, int flags)
{
    double *gnorm;                     /* pointer to normalization data */
    unsigned int offset;               /* linear offset for the grid data */
    unsigned int ntot = nx * ny * nz;  /* total number of points on the grid */
    unsigned int i;                    /* loop index variable */
    unsigned int noutofbounds = 0;     /* number of points out of bounds */

    /* compute step width for the grid */
    double dx = delta(xmin, xmax, nx);
    double dy = delta(ymin, ymax, ny);
    double dz = delta(zmin, zmax, nz);

    /* initialize data if requested */
    if (!(flags & NO_DATA_INIT)) {
        set_array(odata, ntot, 0.);
    }

    /* check if normalization array is passed */
    if (norm == NULL) {
        gnorm = malloc(sizeof(double) * ntot);
        if (gnorm == NULL) {
            fprintf(stderr, "XU.Gridder3D(c): Cannot allocate memory for "
                            "normalization buffer!\n");
            return -1;
        }
        /* initialize memory for norm */
        set_array(gnorm, ntot, 0.);
    }
    else {
        gnorm = norm;
    }

    /* the master loop over all data points */
    for (i = 0; i < n; i++) {
        if (!isnan(data[i])) {
            /* check if the current point is within the bounds of the grid */
            if ((x[i] < xmin) || (x[i] > xmax)) {
                noutofbounds++;
                continue;
            }
            if ((y[i] < ymin) || (y[i] > ymax)) {
                noutofbounds++;
                continue;
            }
            if ((z[i] < zmin) || (z[i] > zmax)) {
                noutofbounds++;
                continue;
            }

            /* compute the offset value of the current input point on the
             * grid array */
            offset = gindex(x[i], xmin, dx) * ny * nz +
                     gindex(y[i], ymin, dy) * nz +
                     gindex(z[i], zmin, dz);

            odata[offset] += data[i];
            gnorm[offset] += 1.;
        }
    }

    /* perform normalization */
    if (!(flags & NO_NORMALIZATION)) {
        for (i = 0; i < ntot; i++) {
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
            fprintf(stdout,"XU.Gridder3D(c): more than half of the datapoints"
                        " out of the data range, consider regridding with"
                        " extended range!\n");
        }
        else {
            fprintf(stdout, "XU.Gridder3D(c): %d datapoints out of the data "
                        "range!\n", noutofbounds);
        }
    }

    return 0;
}

