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

#ifndef __GRIDDER_H__
#define __GRIDDER_H__

/*define flags for the gridder functions*/
#define NO_DATA_INIT 1
#define NO_NORMALIZATION 4
#define VERBOSE 16

/*parameters thread workers*/
typedef struct{
    unsigned int nth;
    unsigned int thid;
    unsigned int npth;
    unsigned int npres;
    double *x;
    double *y;
    double *z;
    double *data;
    unsigned int n;
    double xmin;
    double xmax;
    double dx;
    double ymin;
    double ymax;
    double dy;
    double zmin;
    double zmax;
    double dz;
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    double *odata;
    double *norm;
    int flags;
}_ThGridderArgs;

pthread_mutex_t gridder_mutex;

int gridder2d(double *x,double *y,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,
              double xmin,double xmax,
              double ymin,double ymax,
              double *odata,double *norm,int flags);

int gridder2d_th(unsigned int nth,
                 double *x,double *y,double *data,unsigned int n,
                 unsigned int nx,unsigned int ny,
                 double xmin,double xmax,double ymin,double ymax,
                 double *odata,double *norm,int flags);
void *gridder2d_th_worker(void *arg);


int gridder3d(double *x,double *y,double *z,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,unsigned int nz,
              double xmin, double xmax, double ymin, double ymax,
              double zmin, double zmax,
              double *odata,double *norm,int flags);

int gridder3d_th(unsigned int nth,
                 double *x,double *y,double *z,double *data,unsigned int n,
                 unsigned int nx,unsigned int ny,unsigned int nz,
                 double xmin, double xmax, double ymin, double ymax,
                 double zmin, double zmax,
                 double *odata,double *norm,int flags);
void *gridder3d_th_worker(void *arg);

double get_min(double *a,unsigned int n);
double get_max(double *a,unsigned int n);
#endif
