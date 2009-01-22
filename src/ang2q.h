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
              unsigned int n,double lambda,double *geom,
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

#endif 	    /* !ANG2Q_H_ */
