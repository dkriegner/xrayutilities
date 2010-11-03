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
