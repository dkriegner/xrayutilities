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
*/

/*
 * Implementation of detector functions
 */

#include<stdlib.h>
#include<math.h>
#include<stdio.h>

#include "detector.h"

/****f* Detector/det1d_get_axis
 * NAME
 *   det1d_get_axis - calculates axis values along a detector
 * SYNOPSIS
 *   det1d_get_axis(Detector1D *det,double cvalue,double *axis,ROI1D *roi)
 * DESCRIPTION
 *   Calculate the axis values along a detector. A detector might be parallel
 *   to some motor axis. Starting from a center value the "virtual" motor 
 *   positions can be calculated for every channel of the detector can be 
 *   calculated.
 * ARGUMENTS
 *   Detector1D *det ................ detector structure
 *   double cvalue .................. motor position at the center channel of
 *                                    the detector
 *   double *axis ................... data buffer holding the axis values
 *   ROI1D *roi ..................... optional region of interest
 * RETURN VALUE
 *   An integer value is returned which is either 
 *   0 in the case that everything worked well or -1.
 * EXAMPLE
 *
 * AUTHOR
 *   Eugen Wintersberger
 ****/
int det1d_get_axis(Detector1D *det,double cvalue,double *axis,ROI1D *roi){
    unsigned int i,istart,istop;

    if (roi == NULL){
        istart = 0;
        istop = det->nc;
    }else{
        istart = roi->cstart;
        istop  = roi->cstop + 1;
    }

    for(i=istart;i<istop;i++){
        axis[i] = cvalue + ((double)det->dir)*(((double)i)-nc->cchan)/nc->chdeg;
    }
    return(0);
}

int det2d_get_axis(Detector2D *det,double cxvalue,double cyvalue,double *xaxis,
				   double *yaxis,ROI2D *roi){
	printf("not implemented yet!\n");
}

/****f* Detector/det1d_create_axis
 * NAME
 *   det1d_create_axis - creates a data buffer to hold axis data
 * DESCRIPTION
 *   This function allocates memory to hold the axis values along 
 *   a detector direction. All information about the detector is provided by
 *   a Detector1D data structure. In addition a ROI structure can be passed 
 *   in which case only as much data is allocated to hold the ROI selection
 *   from the axis data.
 *
 * ARGUMENTS
 *   Detector1D *det .............. pointer to a Detector1D data structure
 *                                  holding the detector parameters
 *   double *axis ................. double pointer which can be used afterwards
 *                                  to hold axis data
 *   ROI1D *roi ................... optional region of interest
 *
 * RETURN VALUE
 *   The function regurs an integer value wich is either
 *   0 ................ everything was ok
 *   -1 ............... memory allocation problem
 *
 * EXAMPLE
 *   Detector1D det;
 *   ROI1D roi;
 *   double *axis;
 *
 *   det.cchan = 512;
 *   det.nc = 1024;
 *   det.chdeg = 200;
 *   det.dir = -1;
 *   roi.cstart = 100;
 *   det.cstop = 800;
 *
 *   //allocate axis memory for the entire detector
 *   det1d_create_axis(&det,axis,NULL);
 *   .
 *   .
 *   //allocate axis memory for a region of interest
 *   det1d_create_axis(&det,axis,&roi);
 *
 * AUTHOR
 *   Eugen Wintersberger
 ****/
int det1d_create_axis(Detector1D *det,double *axis,ROI1D *roi){
    unsigned int n;

    if (roi==NULL){
        n = det->nc;
    }else{
        n = roi->cstop-roi->cstart+1;
    }

    axis = malloc(n*sizeof(double));
    if (axis==NULL){
        printf("error allocating memory for detector axis buffer!\n");
        return(-1);
    }    

    return(0);
}

/****f* Detector/det2d_create_axis
 * NAME
 *   det2d_create_axis - creates a data buffer to hold axis data
 * DESCRIPTION
 *   This function allocates memory to hold the axis values along 
 *   a detector direction. All information about the detector is provided by
 *   a Detector1D data structure. In addition a ROI structure can be passed 
 *   in which case only as much data is allocated to hold the ROI selection
 *   from the axis data.
 *
 * ARGUMENTS
 *   Detector2D *det .............. pointer to a Detector1D data structure
 *                                  holding the detector parameters
 *   double *xaxis ................ double pointer which can be used afterwards
 *                                  to hold x-axis data
 *   double *yaxis ................ double pointer which can be used afterwards
 *                                  to hold y-axis data
 *   ROI2D *roi ................... optional region of interest
 *
 * RETURN VALUE
 *   The function regurs an integer value wich is either
 *   0 ................ everything was ok
 *   -1 ............... memory allocation problem
 *
 * EXAMPLE
 *   Detector2D det;
 *   ROI2D roi;
 *   double *xaxis;
 *   double *yaxis;
 *
 *   det.detx.cchan = 512;
 *   det.detx.nc = 1024;
 *   det.detx.chdeg = 200;
 *   det.detx.dir = -1;
 *
 *   det.dety.cchan = 512;
 *   det.dety.nc = 1024;
 *   det.dety.chdeg = 200;
 *   det.dety.dir = -1;
 *
 *   roi.roix.cstart = 100;
 *   det.roix.cstop = 800;
 *
 *   roi.roiy.cstart = 100;
 *   det.roiy.cstop = 800;
 *
 *   //allocate axis memory for the entire detector
 *   det2d_create_axis(&det,xaxis,yaxis,NULL);
 *   .
 *   .
 *   //allocate axis memory for a region of interest
 *   det2d_create_axis(&det,xaxis,yaxis,&roi);
 *
 * AUTHOR
 *   Eugen Wintersberger
 ****/
int det2d_create_axis(Detector2D *det,double *xaxis,double *yaxis,ROI2D *roi){
    unsigned int nx,ny;

    if (roi == NULL){
        nx = det->detx.nc;
        ny = det->dety.nc;
    }else{
        nx = roi->roix.cstop-roi->roix.cstart+1;
        ny = roi->roiy.cstop-roi->roiy.cstart+1;
    }

    xaxis = malloc(nx*sizeof(double));
    if (xaxis == NULL){
        printf("error allocating memory for detector x-axis buffer!\n");
        return(-1);
    }

    yaxis = malloc(ny*sizeof(double));
    if (yaxis == NULL){
        printf("error allocating memory for detector y-axis buffer!\n");
        return(-1);
    }

    return(0);
}

/****f* Detector/det1d_create_dbuffer
 * NAME
 *   det1d_create_dbuffer - allocate data buffer memory
 * SYNOPSIS
 *   det1d_create_dbuffer(Detector1D *det,double *dbuffer,ROI1D *roi)
 * DESCRIPTION
 *   Allocates memory to hold data for a given one dimensional detector setup.
 *   Optionally a region of interest (ROI) can be set. The allocated size 
 *   depends on the ROI.
 * ARGUMENTS
 *   Detector1D *det ................. a pointer to the detector structure
 *   double *dbuffer ................. pointer to the allocated memory
 *   ROI1D *roi ...................... data structure with the region of 
 *                                     interest.
 * RETURN VALUE
 *   The function returns an interger value being either
 *   0 ................. if everything was ok
 *   -1 ................ in the case of a memory allocation problem
 * EXAMPLE
 *   Detector1D det;
 *   ROI1D roi;
 *   double *dbuffer;
 *
 *   det.nc = 1024;
 *   det.cchan = 512;
 *   det.dir = -1;
 *   det.chdeg = 200;
 *
 *   roi.cstart = 100;
 *   roi.cstop = 800;
 *
 *   //allocate memory to save data of the entire detector
 *   det1d_create_dbuffer(&det,dbuffer,NULL);
 *   .
 *   .
 *   .
 *   //allocate memory to save only a ROI of the detector data
 *   det1d_create_dbuffer(&det,dbuffer,&roi);
 *
 * AUTHOR
 *   Eugen Wintersberger
 *****/
int det1d_create_dbuffer(Detector1D *det,double *dbuffer,ROI1D *roi){
    unsigned int n;

    if (roi == NULL){
        n = det->nc;
    }else{
        n = roi->cstop-roi->cstart+1;
    }

    dbuffer = malloc(n*sizeof(double));
    if (dbuffer == NULL){
        printf("error allocating data buffer memory for 1D detector\n");
        return(-1);
    }

    return(0);
}

/****f* Detector/det2d_create_dbuffer
 * NAME
 *   det2d_create_dbuffer - allocate data buffer memory
 * SYNOPSIS
 *   det2d_create_dbuffer(Detector2D *det,double *dbuffer,ROI2D *roi)
 * DESCRIPTION
 *   Allocates memory to hold data for a given tow dimensional detector setup.
 *   Optionally a region of interest (ROI) can be set. The allocated size 
 *   depends on the ROI.
 * ARGUMENTS
 *   Detector2D *det ................. a pointer to the detector structure
 *   double *dbuffer ................. pointer to the allocated memory
 *   ROI2D *roi ...................... data structure with the region of 
 *                                     interest.
 * RETURN VALUE
 *   The function returns an interger value being either
 *   0 ................. if everything was ok
 *   -1 ................ in the case of a memory allocation problem
 * EXAMPLE
 *   Detector2D det;
 *   ROI2D roi;
 *   double *dbuffer;
 *
 *   det.detx.nc = 1024;
 *   det.detx.cchan = 512;
 *   det.detx.dir = -1;
 *   det.detx.chdeg = 200;
 *
 *   det.dety.nc = 1024;
 *   det.dety.cchan = 512;
 *   det.dety.dir = -1;
 *   det.dety.chdeg = 200;
 *
 *   roi.roix.cstart = 100;
 *   roi.roix.cstop = 800;
 *
 *   roi.roiy.cstart = 100;
 *   roi.roiy.cstop = 800;
 *
 *   //allocate memory to save data of the entire detector
 *   det2d_create_dbuffer(&det,dbuffer,NULL);
 *   .
 *   .
 *   .
 *   //allocate memory to save only a ROI of the detector data
 *   det2d_create_dbuffer(&det,dbuffer,&roi);
 *
 * AUTHOR
 *   Eugen Wintersberger
 *****/
int det2d_create_dbuffer(Detector2D *det,double *dbuffer,ROI2D *roi){
    unsigned int nx,ny;

    if (roi == NULL){
        nx = det->detx.nc;
        ny = det->dety.nc;
    }else{
        nx = roi->roix.cstop-roi->roix.cstart+1;
        ny = roi->roiy.cstop-roi->roiy.cstart+1;
    }

    dbuffer = malloc(nx*ny*sizeof(double));
    if (dbuffer == NULL){
        printf("error allocating data buffer memory for 2D detector\n");
        return(-1);
    }

    return(0);
}

/****f* Detector/det1d_integrate
 * NAME
 *   det1d_integrate - integrate a 1D detector
 * SYNOPSIS
 *   det1d_integrate(Detector1D *det,double *data,ROI1D *roi)
 * DESCRIPTION
 *   Integrate the entire spectrum of a 1D detector and returns the 
 *   result. Optionally the integration can be performed over a 
 *   region of interest if the corresponding ROI structure is passed 
 *   to the function.
 * ARGUMENTS
 *   Detector1D *det ............... detector data structure
 *   double *data .................. buffer with detector data
 *   ROI1D *roi .................... region of interest
 * RETURN VALUE
 *   double s ...................... the integrated data as double value
 * EXAMPLE
 *   .
 *   .
 *   //integrate the entire PSD data
 *   det1d_integrate(&det,data,NULL);
 *   .
 *   .
 *   //integrate only a region of interest
 *   det1d_integrate(&det,data,&roi);
 *   .
 *   .
 * AUTHOR
 *   Eugen Wintersberger
 ****/
double det1d_integrate(Detector1D *det,double *data,ROI1D *roi){
	unsigned int i,istart,istop;
	double s;
	
	if (roi == NULL){
		istart = 0;
		istop = det->nc;
	}else{
		istart = roi->cstart;
		istop = roi->cstop+1;
	}
	
	s = 0.0;
	for (i=istart;i<istop;i++) s += data[i];
	
	return(s);
}

/****f* Detector/det2d_integrate
 * NAME
 *   det2d_integrate - integrate a 2D detector
 * SYNOPSIS
 *   det2d_integrate(Detector2D *det,double *data,ROI2D *roi)
 * DESCRIPTION
 *   Integrate the entire spectrum of a 2D detector and returns the 
 *   result. Optionally the integration can be performed over a 
 *   region of interest if the corresponding ROI structure is passed 
 *   to the function.
 * ARGUMENTS
 *   Detector2D *det ............... detector data structure
 *   double *data .................. buffer with detector data
 *   ROI2D *roi .................... region of interest
 * RETURN VALUE
 *   double s ...................... the integrated data as double value
 * EXAMPLE
 *   .
 *   .
 *   //integrate the entire PSD data
 *   det2d_integrate(&det,data,NULL);
 *   .
 *   .
 *   //integrate only a region of interest
 *   det2d_integrate(&det,data,&roi);
 *   .
 *   .
 * AUTHOR
 *   Eugen Wintersberger
 ****/
double det2d_integrate(Detector2D *det,double *data,ROI2D *roi){
	unsigned int i,istart,istop;
	unsigned int j,jstart,jstop;
	unsigned int ioffset;
	double s;
	
	if (roi == NULL){
		istart = 0; 
		istop = det->detx.nc;
		jstart = 0;
		jstop = det->dety.nc;
	}else{
		istart = roi->roix.cstart;
		istop  = roi->roix.cstop + 1;
		jstart = roi->roiy.cstart;
		jstop  = roi->roiy.cstop + 1;
	}
	
	s = 0.0;
	for(i=istart;i<istop;i++){
		ioffset = i*det->detx.nc;
		for(j=jstart;j<jstop;j++){
			s += data[j+ioffset];
		}
	}
	
	return(s);
}

/*functions to plot detectors*/
int det1d_plot(Detector1D *det,double *data,ROI1D *roi,int logflag){
	printf("not implemented yet!\n");
}

int det2d_plot(Detector2D *det,double *data,ROI2D *roi,int logflag){
	printf("not implemented yet!\n");
}
