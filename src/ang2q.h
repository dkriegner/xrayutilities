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
** ang2q.h
**
** Made by Eugen Wintersberger
** Login   <eugen@ramses.lan>
**
** Started on  Tue Aug 21 23:00:13 2007 Eugen Wintersberger
** Last update Tue Aug 21 23:00:13 2007 Eugen Wintersberger
*/

#ifndef __ANG2Q_H__
#define __ANG2Q_H__

/*definition of command line options*/
#define HXRD_LOHI -1.
#define HXRD_HILO 1.

/*functions to convert x-ray data from angular to reciprocal space*/


/*simple functions to convert a bunch of angular data into q-space*/
int a2q_xrd2d(double *om,double *th2,
              double *qx,double *qz,unsigned int n,
              double lambda,double geom,double dom,double dth2);
int a2q_xrd3d(double *om,double *th2,double *delta,
              double *qx,double *qy,double *qz,
              unsigned int n,double lambda,double geom,
              double dom,double dth2,double ddelta);

/*functions and data types for xrd/xrr q-space converions using multithreading*/
typedef struct{
    unsigned int nth;
    unsigned int npth;
    unsigned int npres;
    unsigned int thid;
    double *om;
    double *th2;
    double *delta;
    double *qx;
    double *qy;
    double *qz;
    unsigned int n;
    double lambda;
    double geom;
    double dom;
    double dth2;
    double ddelta;
}a2q_xrd_thargs;

int a2q_xrd2d_th(unsigned int nth,
                double *om,double *th2,double *qx,double *qz,unsigned int n,
                double lambda,double geom,
                double dom,double dth2);
void *a2q_xrd2d_thworker(void *args);

int a2q_xrd3d_th(unsigned int nth,
                 double *om,double *th2,double *delta,
                 double *qx,double *qy,double *qz,unsigned int n,
                 double lambda,double geom,
                 double dom,double dth2,double ddelta);
void *a2q_xrd3d_th_worker(void *args);

int a2q_gid(double *th2,double *qa,double *qr,unsigned int n,double ai,
            double *af,unsigned int naf,double lambda);
int a2q_gisaxs(double *th2,double *qx,double *qy,double *qz,unsigned int n,
               double ai,double *af,unsigned int naf,double lambda);
/*functions and data types for gid q-space conversion using multithreading*/
typedef struct{
    double *th2;
    double *qa;
    double *qr;
    unsigned int n;
    double ai;
    double *af;
    unsigned int naf;
    double lambda;
}a2q_gid_thargs;

void *a2q_gid_thworker(void *args);
int a2q_gid_th(double *th2,double *qa,double *qr,unsigned int n,double ai,
               double *af,unsigned int naf,double lambda);

/*functions and data types for GISAXS q-space conversion using multithreading*/
typedef struct{
    double *th2;
    double *qx;
    double *qy;
    double *qz;
    unsigned int n;
    double ai;
    double *af;
    unsigned int naf;
    double lambda;
}a2q_gisays_thargs;

void *a2q_gisaxs_thworkder(void *args);
int a2q_gisaxs_th(double *th2,double *qx,double *qy,double *qz,unsigned int n,
                  double ai,double *af,unsigned int naf,double lambda);

#endif      /* !ANG2Q_H_ */
