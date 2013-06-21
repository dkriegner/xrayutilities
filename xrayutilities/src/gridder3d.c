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
 * created: Jun 21,2013
 * author: Eugen Wintersberger
*/

#include "gridder.h"
#include "gridder_utils.h"


int gridder3d(double *x,double *y,double *z,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,unsigned int nz,
              double xmin, double xmax, double ymin, double ymax,
              double zmin, double zmax,
              double *odata,double *norm,int flags)
{
    double *gnorm;                //pointer to normalization data
    unsigned int offset;          //linear offset for the grid data
    unsigned int ntot = nx*ny*nz; //total number of points on the grid


    //compute step width for the grid
    double dx = delta(xmin,xmax,nx);
    double dy = delta(ymin,ymax,ny);
    double dz = delta(zmin,zmax,nz);

    /*initialize data if requested*/
    if(!(flags&NO_DATA_INIT)) set_array(odata,ntot,0.);

    /*check if normalization array is passed*/
    if(norm==NULL)
    {
        gnorm = malloc(sizeof(double)*ntot);
        if(gnorm==NULL)
        {
            fprintf(stderr,"XU.Gridder3D(c): Cannot allocate memory for normalization buffer!\n");
            return(-1);
        }
        /*initialize memory for norm*/
        set_array(gnorm,ntot,0.);
    }
    else
        gnorm = norm;

    /*the master loop over all data points*/
    for(unsigned int i=0;i<n;i++)
    {
        //check if the current point is within the bounds of the grid
        if((x[i]<xmin)||(x[i]>xmax)) continue;
        if((y[i]<ymin)||(y[i]>ymax)) continue;
        if((z[i]<zmin)||(z[i]>zmax)) continue;

        //compute the offset value of the current input point on the grid array
        offset = gindex(x[i],xmin,dx)*ny*nz +
                 gindex(y[i],ymin,dy)*nz +
                 gindex(z[i],zmin,dz);

        odata[offset] += data[i];
        gnorm[offset] += 1.;
    }

    /*perform normalization*/
    if(!(flags&NO_NORMALIZATION))
    {
        for(unsigned int i=0;i<ntot;i++)
            if(gnorm[i]>1.e-16) odata[i] = odata[i]/gnorm[i];
    }

    /*free the norm buffer if it has been locally allocated*/
    if(norm==NULL) free(gnorm);

    return(0);
}
