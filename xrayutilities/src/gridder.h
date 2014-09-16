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
 * Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
 * Copyright (C) 2009-2010 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

/****h* xrayutils/gridder
 * NAME
 *   gridder - module with gridder functions
 * PURPOSE
 *
 * AUTHOR
 *   Eugen Wintersberger
 ****/

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <Python.h>

/*define flags for the gridder functions*/
#define NO_DATA_INIT 1
#define NO_NORMALIZATION 4
#define VERBOSE 16

/*!
\brief python interface function

Python interface function for gridder1d. This function is virtually doing all
the Python related stuff to run gridder1d function.
\param self reference to the module
\param args function arguments
\return return value of the function
*/
PyObject* pygridder1d(PyObject *self,PyObject *args);

//-----------------------------------------------------------------------------
/*!
\brief 1D single threaded gridder

\param x input x-values
\param data input data
\param n number of input points
\param nx number of steps in x-direction
\param xmin minimm along x-direction
\param xmax maximum along x-direction
\param odata output data
\param norm normalization data
\param flags control falgs
*/
int gridder1d(double *x,double *data,unsigned int n,
              unsigned int nx,double xmin,double xmax,
              double *odata,double *norm,int flags);
              
/*!
\brief python interface function

Python interface function for gridder2d. This function is virtually doing all
the Python related stuff to run gridder2d function.
\param self reference to the module
\param args function arguments
\return return value of the function
*/
PyObject* pygridder2d(PyObject *self,PyObject *args);

//-----------------------------------------------------------------------------
/*!
\brief 2D single threaded gridder

\param x input x-values
\param y input y-values
\param data input data
\param n number of input points
\param nx number of steps in x-direction
\param ny number of steps in y-direction
\param xmin minimm along x-direction
\param xmax maximum along x-direction
\param ymin minimum along y-direction
\param ymax maximum along y-direction
\param odata output data
\param norm normalization data
\param flags control falgs
*/
int gridder2d(double *x,double *y,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,
              double xmin,double xmax,
              double ymin,double ymax,
              double *odata,double *norm,int flags);

//-----------------------------------------------------------------------------
/*!
\brief 3D gridder python interface function

Python interface function for gridder3d. This function is virtually doing all
the Python related stuff to run gridder2d function.
\param self reference to the module
\param args function arguments
\return return value of the function
*/
PyObject* pygridder3d(PyObject *self,PyObject *args);

//-----------------------------------------------------------------------------
/*!
\brief single threaded 3d gridder

Gridder code rebinning scatterd data onto a regular grid in 3 dimensions.

\param x pointer to x-coordinates of input data
\param y pointer to y-coordinates of input data
\param z pointer to z-coordinates of input data
\param data pointer to input data
\param n number of input points
\param nx number of grid points along the x-direction
\param ny number of grid points along the y-direction
\param nz number of grid points along the z-direction
\param xmin minimum value of x-axis on the grid
\param xmax maximum value of x-axis on the grid
\param ymin minimum value of y-axis on the grid
\param ymax maximum value of y-axis on the grid
\param zmin minimum value of z-axis on the grid
\param zmax maximum value of z-axis on the grid
\param odata pointer to grid data (output data)
\param norm pointer to optional normalization from previous run
\param flags gridder flags
*/
int gridder3d(double *x,double *y,double *z,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,unsigned int nz,
              double xmin, double xmax, double ymin, double ymax,
              double zmin, double zmax,
              double *odata,double *norm,int flags);

