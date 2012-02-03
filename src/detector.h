/*
 * This file is part of xrutils.
 * 
 * xrutils is free software; you can redistribute it and/or modify 
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

/****h* XRayUtils/Detector
 *  NAME
 *     Detector - functions and data types to handle detector data
 *  DESCRIPTION
 *     The Detector module contains functions and data types used to 
 *     handle detector data. Currently 1D detectors (like PSD) and 
 *     2D detectors (like CCD cameras) are supported. 
 *
 *     The data is stored in native double arrays - no special data 
 *     structures are used. This should keep everything as simple as 
 *     possible. 
 *
 *     A few words about how 2D data is stored in a 1D array. Consider a CCD 
 *     data set of NX x NY points. Where NX is the number of columns while
 *     NY is the number of rows. 2D data is stored in an 1D array with 
 *     column index varying fastest. To obtain the data value on pixle i,j where 
 *     i is the row index in j the column index, the following notation must 
 *     be used data[j+i*NX].
 *
 *  AUTHOR
 *     Eugen Wintersberger
 ****/

#ifndef __DETECTOR_H__
#define __DETECTOR_H__

/****s* Detector/Detector1D
 * NAME
 *   Detector1D - describes a a line detector
 * DESCRIPTION
 *   Description of a line detector (PSD). This data structure 
 *   holds all information neccessary to work with PSD data like the 
 *   number of channels and the direction.
 *
 * AUTHOR
 *   Eugen Wintersberger
 * SOURCE
 */
typedef struct{
    double cchan;    /*center channel          */
    double chdeg;    /*channels per degree     */
    int    dir;      /*direction of the PSD    */
    unsigned int nc; /*total number of channels*/
}Detector1D;
/*****/

/****s Detector/ROI1D
 * NAME
 *   ROI1D - one dimensional region of interest
 * DESCRIPTION
 *   A data type describing a one dimensional region of interest in a 
 *   PSD spectrum.
 * AUTHOR
 *   Eugen Wintersberger
 * SOURCE
 */
typedef struct{
    unsigned int cstart; /*first channel of the ROI*/
    unsigned int cstop;  /*last channel of the ROI */
}ROI1D;
/****/

/****s* Detector/Detector2D
 * NAME
 *   Detector2D - a tow dimensional detector
 * DESCRIPTION
 *   This data structure describes a tow dimensional detector like CCD.
 *   In principle a 1D detector is used in every direction to describe the 
 *   2D detector.
 * AUTHOR
 *   Eugen Wintersberger
 * SOURCE
 */
typedef struct{
    Detector1D detx; /*detector parameters in x-direction*/
    Detector1D dety; /*detector parameters in y-direction*/
}Detector2D;
/****/

/****s* Detector/ROI2D
 * NAME
 *   ROI2D - tow dimensional region of interest for a CCD
 * DESCRIPTION
 *   Description of a rectangular region of interest in a CCD camera. In
 *   principle tow one dimensional ROIs are used - one for each direction.
 * AUTHOR
 *   Eugen Wintersberger
 * SOURCE
 */
typedef struct{
    ROI1D roix; /*ROI in x-direction*/
    ROI1D roiy; /*RIO in y-direction*/
}ROI2D;
/****/

/*----------------------------functions-------------------------*/
/*calculate the detector axis*/
int det1d_get_axis(Detector1D *det,double cvalue,double *axis,ROI1D *roi);
int det2d_get_axis(Detector2D *det,double cxvalue,double cyvalue,double *xaxis,
                   double *yaxis,ROI2D *roi);
/*allocate axis memory according to detector parameters*/
int det1d_create_axis(Detector1D *det,double *axis,ROI1D *roi);
int det2d_create_axis(Detector2D *det,double *xaxis,double *yaxis,ROI2D *roi);
/*allocate buffer memory for detector data according to the detector
 * parameters
 */
int det1d_create_dbuffer(Detector1D *det,double *dbuffer,ROI1D *roi);
int det2d_create_dbuffer(Detector2D *det,double *dbuffer,ROI2D *roi);

/*functions to plot detectors*/
int det1d_plot(Detector1D *det,double *data,ROI1D *roi,int logflag);
int det2d_plot(Detector2D *det,double *data,ROI2D *roi,int logflag);

double det1d_integrate(Detector1D *det,double *data,ROI1D *roi);
double det2d_integrate(Detector2D *det,double *data,ROI2D *roi);

#endif



