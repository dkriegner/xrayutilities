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
** ang2q.c
**
** Made by (Eugen Wintersberger)
** Login   <eugen@ramses.lan>
**
** Started on  Tue Aug 21 21:02:27 2007 Eugen Wintersberger
** Last update Sun May 12 01:17:25 2002 Speed Blue
*/
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<pthread.h>

#include "ang2q.h"

/*simple functions to convert a bunch of angular data into q-space*/
int a2q_xrd2d(double *om,double *th2,double *qx,double *qz,unsigned int n,
            double lambda,double geom,double dom,double dth2)
/*{{{1*/
{
    double k02;
    unsigned int i;
    double omega;
    double tth;
    double tmp1;
    double tmp2;

    k02 = 4.*M_PI/lambda;

    for(i=0;i<n;i++){
        omega = om[i]-dom;
        tth   = th2[i]- dth2;
        tmp1  = sin(0.5*tth);
        tmp2  = 0.5*tth - omega;

        qx[i] = k02*tmp1*sin(geom*tmp2);
        qz[i] = k02*tmp1*cos(geom*tmp2);
    }

    return(0);
}
/*}}}1*/



int a2q_xrd2d_th(unsigned int nth,
                 double *om,double *th2,double *qx,double *qz,unsigned int n,
                 double lambda,double geom,
                 double dom,double dth2)
/*{{{1*/
{
    a2q_xrd_thargs *thargs;
    pthread_t *threads;
    pthread_attr_t thattr;
    unsigned int i;
    int npth,npres;
    int rc;
    void *status;
    double k02;
    double omega;
    double tth;
    double tmp1;
    double tmp2;


    /*allocate memory*/
    thargs = malloc(sizeof(a2q_xrd_thargs)*nth);
    if (thargs == NULL){
        fprintf(stderr,"Cannot allocate memory for thread arguments!\n");
        return(-1);
    }

    threads = malloc(sizeof(pthread_t)*nth);
    if (threads == NULL){
        fprintf(stderr,"Cannot allocate memory for threads!\n");
        free(thargs);
        return(-1);
    }

    /*setting up the thread parameters*/
    npres = n%nth;
    npth  = (n-npres)/nth;
    for(i=0;i<nth;i++){
        thargs[i].nth = nth;
        thargs[i].npth = npth;
        thargs[i].npres = npres;
        thargs[i].thid = i;
        thargs[i].om = om;
        thargs[i].th2 = th2;
        thargs[i].qx = qx;
        thargs[i].qz = qz;
        thargs[i].n = n;
        thargs[i].lambda = lambda;
        thargs[i].geom = geom;
        thargs[i].dom = dom;
        thargs[i].dth2 = dth2;
    }

    /*fire away the threads*/
    pthread_attr_init(&thattr);
    pthread_attr_setdetachstate(&thattr,PTHREAD_CREATE_JOINABLE);
    for(i=1;i<nth;i++){
        rc = pthread_create(&threads[i],&thattr,a2q_xrd2d_thworker,(void *)&thargs[i]);
        if(rc){
            fprintf(stderr,"Error creating thread %i\n",i);
            return(-1);
        }
    }

    /*run now the main code of this thread*/
    k02 = 4.*M_PI/thargs[0].lambda;

    for(i=0;i<thargs[i].npth+thargs[i].npres;i++){
        omega = thargs[0].om[i]-thargs[0].dom;
        tth   = thargs[0].th2[i]- thargs[0].dth2;
        tmp1  = sin(0.5*tth);
        tmp2  = 0.5*tth - omega;

        thargs[0].qx[i] = k02*tmp1*sin(thargs[0].geom*tmp2);
        thargs[0].qz[i] = k02*tmp1*cos(thargs[0].geom*tmp2);
    }

    /*join the other threads and wait for finish*/
    for(i=1;i<nth;i++){
        rc = pthread_join(threads[i],&status);
        if(rc){
            fprintf(stderr,"Error joining thread %i!\n",i);
            return(-1);
        }
    }

    /*free all memory*/
    free(thargs);
    free(threads);

    pthread_attr_destroy(&thattr);

    return(0);
}
/*}}}1*/

void *a2q_xrd2d_thworker(void *args)
/*{{{1*/
{
    double k02;
    unsigned int i,start,stop;
    double omega;
    double tth;
    double tmp1;
    double tmp2;
    a2q_xrd_thargs thargs;

    thargs = *(a2q_xrd_thargs *)args;

    k02 = 4.*M_PI/thargs.lambda;

    start = thargs.npres + thargs.thid*thargs.npth;
    stop  = start + thargs.npth;
    for(i=start;i<stop;i++){
        omega = thargs.om[i]-thargs.dom;
        tth   = thargs.th2[i]- thargs.dth2;
        tmp1  = sin(0.5*tth);
        tmp2  = 0.5*tth - omega;

        thargs.qx[i] = k02*tmp1*sin(thargs.geom*tmp2);
        thargs.qz[i] = k02*tmp1*cos(thargs.geom*tmp2);
    }

    pthread_exit(NULL);
}
/*}}}1*/



int a2q_xrd3d(double *om,double *th2,double *del,
              double *qx,double *qy,double *qz,
              unsigned int n,double lambda,double geom,
              double dom,double dth2,double ddelta)
/*{{{1*/
{

    double k02;
    unsigned int i;
    double omega;
    double tth;
    double delta;
    double tmp1;
    double tmp2;


    k02 = 4.*M_PI/lambda;

    for(i=0;i<n;i++){
        omega = om[i] - dom;
        tth   = th2[i] - dth2;
        delta = del[i] - ddelta;
        tmp1  = sin(0.5*tth);
        tmp2  = 0.5*tth - omega;

        qx[i] = k02*tmp1*sin(geom*tmp2);
        qy[i] = k02*tmp1*sin(geom*tmp2)*sin(delta);
        qz[i] = k02*tmp1*cos(geom*tmp2)*cos(delta);
    }

    return(0);
}
/*}}}1*/

int a2q_gid(double *th2,double *qa,double *qr,unsigned int n,double ai,
            double *af,unsigned int naf,double lambda){
    printf("not implemented yet!\n");
    return(0);
}
int a2q_gisaxs(double *th2,double *qx,double *qy,double *qz,unsigned int n,
               double ai,double *af,unsigned int naf,double lambda){
    printf("not implemented yet!\n");
    return(0);
}


void *a2q_gid_thworker(void *args){
    printf("not implemented yet\n");
    pthread_exit(NULL);
}

int a2q_gid_th(double *th2,double *qa,double *qr,unsigned int n,double ai,
               double *af,unsigned int naf,double lambda){
    printf("not implemented yet\n");
    return(0);
}


void *a2q_gisaxs_thworker(void *args){
    printf("not implemented yet\n");
    pthread_exit(NULL);
}

int a2q_gisaxs_th(double *th2,double *qx,double *qy,double *qz,unsigned int n,
                  double ai,double *af,unsigned int naf,double lambda){
    printf("not implemented yet\n");
    return(0);
}
