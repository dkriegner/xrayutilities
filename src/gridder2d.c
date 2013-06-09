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

#include "gridder.h"
#include "gridder_utils.h"

#include <numpy/arrayobject.h>

#define PYARRAY_CHECK(array,dims,type,msg) \
    if(PyArray_NDIM(array) != dims ||  \
       PyArray_TYPE(array) != type) \
    {\
        PyErr_SetString(PyExc_ValueError,\
                msg); \
        return NULL; \
    }


PyObject* pygridder2d(PyObject *self,PyObject *args)
{
    PyArrayObject *py_x=NULL,*py_y=NULL,*py_data=NULL,
                  *py_output=NULL,*py_norm=NULL;

    double *x=NULL,*y=NULL,*data=NULL,*odata=NULL,*norm=NULL;
    double xmin,xmax,ymin,ymax;
    unsigned int nx,ny;
    int flags;

    if(!PyArg_ParseTuple(args,"O!O!O!IIddddO!O!i",
                         &PyArray_Type,&py_x,
                         &PyArray_Type,&py_y,
                         &PyArray_Type,&py_data,
                         &nx,&ny,&xmin,&xmax,&ymin,&ymax,
                         &PyArray_Type,&py_output,
                         &PyArray_Type,&py_norm,
                         &flags))
        return NULL;

    //have to check input variables
    PYARRAY_CHECK(py_x,1,NPY_DOUBLE,"x-axis must be a 1D double array!");
    PYARRAY_CHECK(py_y,1,NPY_DOUBLE,"y-axis must be a 1D double array!");
    PYARRAY_CHECK(py_data,1,NPY_DOUBLE,"input data must be a 1D double array!");
    PYARRAY_CHECK(py_output,2,NPY_DOUBLE,"ouput data must be a 2D double array!");
    PYARRAY_CHECK(py_norm,2,NPY_DOUBLE,"norm data must be a 2D double array!");

    //get data
    x = (double *)PyArray_DATA(py_x);
    y = (double *)PyArray_DATA(py_y);
    data = (double *)PyArray_DATA(py_data);
    odata = (double *)PyArray_DATA(py_output);
    norm = (double *)PyArray_DATA(py_norm);

    //get the total number of points
    int n =  PyArray_SIZE(py_x);

    //call the actual gridder routine
    gridder2d(x,y,data,n,nx,ny,xmin,xmax,ymin,ymax,odata,norm,flags);

}

//-----------------------------------------------------------------------------
int gridder2d(double *x,double *y,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,
              double xmin,double xmax,double ymin,double ymax,
              double *odata,double *norm,int flags)
{
    double *gnorm;
    unsigned int offset;

    double dx = delta(xmin,xmax,nx);
    double dy = delta(ymin,ymax,ny);

    /*check if normalization array is passed*/
    if(norm==NULL)
    {
        gnorm = malloc(sizeof(double)*(nx*ny));
        if(gnorm==NULL)
        {
            fprintf(stderr,"XU.Gridder2D(c): Cannot allocate memory for normalization buffer!\n");
            return(-1);
        }
        /*initialize memory for norm*/
        set_array(gnorm,nx*ny,0.);
    }
    else
    {
        if(flags&VERBOSE) 
        {
            fprintf(stdout,"XU.Gridder2D(c): use user provided buffer for normalization data\n");
        }
        gnorm = norm;
    }

    /*the master loop over all data points*/
    for(unsigned int i=0;i<n;i++)
    {
        //if the x and y values are outside the grids boundaries continue with
        //the next point
        if ((x[i]<xmin)||(x[i]>xmax)) continue;
        if ((y[i]<ymin)||(y[i]>ymax)) continue;

        //compute the linear offset and set the data
        offset = index(x[i],xmin,dx)*ny+index(y[i],ymin,dy);
        odata[offset] += data[i];
        gnorm[offset] += 1.;
    }

    /*perform normalization*/
    if(!(flags&NO_NORMALIZATION))
    {
        if(flags&VERBOSE) 
            fprintf(stdout,"XU.Gridder2D(c): perform normalization ...\n");

        for(unsigned int i=0;i<nx*ny;i++)
            if(gnorm[i]>1.e-16) odata[i] = odata[i]/gnorm[i];

    }

    /*free the norm buffer if it has been locally allocated*/
    if(norm==NULL) free(gnorm);

    return(0);
}
