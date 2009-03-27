/*implementation of the gridder code*/

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<pthread.h>

#include "gridder.h"

int gridder2d(double *x,double *y,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,
              double xmin,double xmax,double ymin,double ymax,
              double *odata,double *norm,int flags)
/*{{{1*/
{
    double dx;
    double dy;
    double *gnorm;
    unsigned int i;
    unsigned int offset;
    int x_index,y_index;
    double tmpx,tmpy;

    dx = fabs(xmax-xmin)/(double)(nx-1);
    dy = fabs(ymax-ymin)/(double)(ny-1);

    /*check if normalization array is passed*/
    if(norm==NULL){
        gnorm = malloc(sizeof(double)*(nx*ny));
        if(gnorm==NULL){
            fprintf(stderr,"Cannot allocate memory for normalization buffer!\n");
            return(-1);
        }
        /*initialize memory for norm*/
        for(i=0;i<nx*ny;i++) gnorm[i] = 0.;
    }else{
        fprintf(stdout,"use user provided buffer for normalization data\n");
        gnorm = norm;
    }

    /*the master loop over all data points*/
    for(i=0;i<n;i++){
        tmpx = x[i];
        tmpy = y[i];
        if ((tmpx<xmin)||(tmpx>xmax)) continue;
        if ((tmpy<ymin)||(tmpy>ymax)) continue;

        x_index = (unsigned int)rint((tmpx-xmin)/dx);
        y_index = (unsigned int)rint((tmpy-ymin)/dy);

        offset = x_index*ny+y_index;
        odata[offset] += data[i];
        gnorm[offset] += 1.;
    }

    /*perform normalization*/
    if(!(flags&NO_NORMALIZATION)){
        fprintf(stdout,"perform normalization ...\n");
        for(i=0;i<nx*ny;i++){
            if(gnorm[i]>1.e-16){
                odata[i] = odata[i]/gnorm[i];
            }
        }
    }

    if(norm==NULL){
        /*free the norm buffer if it has been locally allocated*/
        free(gnorm);
    }

    return(0);
}
/*}}}1*/


int gridder2d_th(unsigned int nth,
                 double *x,double *y,double *data,unsigned int n,
                 unsigned int nx,unsigned int ny,
                 double xmin,double xmax,double ymin,double ymax,
                 double *odata,double *norm,int flags)
/*{{{1*/
{
    pthread_t *threads;
    pthread_attr_t thattr;
    _ThGridderArgs *thargs;
    int npth,npres;
    double dx;
    double dy;
    double *gnorm;
    unsigned int i;
    unsigned int offset;
    int x_index,y_index;
    unsigned int start,stop;
    int rc;
    void *status;


    dx = fabs(xmax-xmin)/(double)(nx-1);
    dy = fabs(ymax-ymin)/(double)(ny-1);
    
    /*check if normalization array is passed*/
    if(norm==NULL){
        gnorm = malloc(sizeof(double)*(nx*ny));
        if(gnorm==NULL){
            fprintf(stderr,"Cannot allocate memory for normalization buffer!\n");
            return(-1);
        }
        /*initialize memory for norm*/
        for(i=0;i<nx*ny;i++) gnorm[i] = 0.;
    }else{
        gnorm = norm;
    }

    /*determine the number of points to handle by each thread*/
    npres = n%nth;
    npth = (n-npres)/nth;

    /*allocate memory*/
    threads = malloc(sizeof(pthread_t)*nth);
    if(threads == NULL){
        fprintf(stderr,"Cannot allocate thread array!\n");
        if(norm==NULL) free(gnorm);
        return(-1);
    }

    thargs = malloc(sizeof(_ThGridderArgs)*nth);
    if(thargs == NULL){
        fprintf(stderr,"Cannot allocate thread argument array!\n");
        if(norm==NULL) free(gnorm);
        free(threads);
        return(-1);
    }

    /*create thread attributes and mutex lock objects*/
    pthread_attr_init(&thattr);
    pthread_attr_setdetachstate(&thattr,PTHREAD_CREATE_JOINABLE);
    pthread_mutex_init(&gridder_mutex,NULL);

    /*setup the thread parameters*/
    for(i=0;i<nth;i++){
        thargs[i].nth = nth;
        thargs[i].thid = i;
        thargs[i].npth = npth;
        thargs[i].npres = npres;
        thargs[i].x = x;
        thargs[i].y = y;
        thargs[i].data = data;
        thargs[i].n = n;
        thargs[i].xmin = xmin;
        thargs[i].xmax = xmax;
        thargs[i].ymin = ymin;
        thargs[i].ymax = ymax;
        thargs[i].dx = dx;
        thargs[i].dy = dy;
        thargs[i].nx = nx;
        thargs[i].ny = ny;
        thargs[i].odata = odata;
        thargs[i].norm = gnorm;
    }

    /*fire the threads*/
    for(i=1;i<nth;i++){
        rc = pthread_create(&threads[i],&thattr,gridder2d_th_worker,
                (void *)&thargs[i]);
        if(rc){
            fprintf(stderr,"Error creating thread %i!\n",i);
            free(threads);
            free(thargs);
            if (norm == NULL) free(gnorm);

            return(-1);
        }
    }

    /*run here the local code for this threads*/
    gridder2d_th_worker((void *)&thargs[0]);

    /*once the local code is inished all other threads are joined to 
     *until they are finished*/
    for(i=1;i<nth;i++){
        rc = pthread_join(threads[i],&status);
    }

    /*delete the thread attributes and mutex lock objects*/
    pthread_attr_destroy(&thattr);
    pthread_mutex_destroy(&gridder_mutex);


    /*perform normalization if requested*/
    if(!(flags&NO_NORMALIZATION)){
        fprintf(stdout,"perform normalization ...\n");
        for(i=0;i<nx*ny;i++){
            if(gnorm[i]>1.e-16){
                odata[i] = odata[i]/gnorm[i];
            }
        }
    }

    /*free memory*/
    free(threads);
    free(thargs);
    if (norm==NULL) free(gnorm);

    return(0);
}
/*}}}1*/

void *gridder2d_th_worker(void *arg)
/*{{{1*/
{
    unsigned int i;
    unsigned int x_index,y_index;
    unsigned int offset;
    _ThGridderArgs tharg;
    unsigned int start,stop;
    unsigned int *x_index_buffer;
    unsigned int *y_index_buffer;
    unsigned int *offset_buffer;
    unsigned int *data_index_buffer;
    unsigned int valid_point_count=0;
    unsigned int index;

    tharg = *(_ThGridderArgs *)arg;

    /*the master loop over all data points*/
    if(tharg.thid==0){
        start = 0;
        stop  = tharg.npres + tharg.npth;
    }else{
        start = tharg.npres + tharg.thid*tharg.npth;
        stop  = start + tharg.npth;
    }
    
    /*allocate memory of the offset buffer*/
    offset_buffer = malloc(sizeof(unsigned int)*tharg.npth);
    data_index_buffer = malloc(sizeof(unsigned int)*tharg.npth);

    for(i=start;i<stop;i++){
        if ((tharg.x[i]<tharg.xmin)||(tharg.x[i]>tharg.xmax)) continue;
        if ((tharg.y[i]<tharg.ymin)||(tharg.y[i]>tharg.ymax)) continue;

        x_index = (unsigned int)rint((tharg.x[i]-tharg.xmin)/tharg.dx);
        y_index = (unsigned int)rint((tharg.y[i]-tharg.ymin)/tharg.dy);

        offset = x_index*tharg.ny+y_index;
        offset_buffer[valid_point_count] = offset;
        data_index_buffer[valid_point_count] = i;
        valid_point_count++;

    }

    pthread_mutex_lock(&gridder_mutex);
    for(i=0;i<valid_point_count;i++){
        offset = offset_buffer[i];
        index  = data_index_buffer[i];
        tharg.odata[offset] += tharg.data[index];
        tharg.norm[offset] += 1.; 
    }
    pthread_mutex_unlock(&gridder_mutex);


    free(offset_buffer);
    free(data_index_buffer);

    if(tharg.thid==0){
        /*if the calling thread is the zero thread return 0 to avoid 
         *the entire program is exiting
         */
        return(0);
    }else{
        /*any other thread returns with pthread_exit()*/
        pthread_exit(NULL);
    }
}
/*}}}1*/

int gridder3d(double *x,double *y,double *z,double *data,unsigned int n,
              unsigned int nx,unsigned int ny,unsigned int nz,
              double *odata,double *norm,int flags)
/*{{{1*/
{
    double xmin,xmax,dx;
    double ymin,ymax,dy;
    double zmin,zmax,dz;
    double *gnorm;
    unsigned int i;
    unsigned int offset;
    unsigned int x_index,y_index,z_index;


    /*determine axis minimum and maximum*/
    xmin = get_min(x,n);
    xmax = get_max(x,n);
    ymin = get_min(y,n);
    ymax = get_max(y,n);
    zmin = get_min(z,n);
    zmax = get_max(z,n);

    dx = (xmax-xmin)/(double)(nx-1);
    dy = (ymax-ymin)/(double)(ny-1);
    dz = (zmax-zmin)/(double)(nz-1);

    /*initialize data if requested*/
    if(!(flags&NO_DATA_INIT)){
        for(i=0;i<nx*ny*nz;i++) odata[i] = 0.;
    }

    /*check if normalization array is passed*/
    if(norm==NULL){
        gnorm = malloc(sizeof(double)*(nx*ny*nz));
        if(gnorm==NULL){
            fprintf(stderr,"Cannot allocate memory for normalization buffer!\n");
            return(-1);
        }
        /*initialize memory for norm*/
        for(i=0;i<nx*ny*nz;i++) gnorm[i] = 0.;
    }else{
        gnorm = norm;
    }

    /*the master loop over all data points*/
    for(i=0;i<n;i++){
        x_index = (unsigned int)rint((x[i]-xmin)/dx);
        y_index = (unsigned int)rint((y[i]-ymin)/dy);
        z_index = (unsigned int)rint((z[i]-zmin)/dz);

        offset = x_index*ny*nz+y_index*nz+z_index;
        odata[offset] += data[i];
        gnorm[offset] += 1.;
    }

    /*perform normalization*/
    if(!(flags&NO_NORMALIZATION)){
        for(i=0;i<nx*ny*nz;i++){
            if(gnorm[i]>1.e-16){
                odata[i] = odata[i]/gnorm[i];
            }
        }
    }

    if(norm==NULL){
        /*free the norm buffer if it has been locally allocated*/
        free(gnorm);
    }

    return(0);
}
/*}}}1*/


int gridder3d_th(unsigned int nth,
                 double *x,double *y,double *z,double *data,unsigned int n,
                 unsigned int nx,unsigned int ny,unsigned int nz,
                 double *odata,double *norm,int flags)
/*{{{1*/
{
    pthread_t *threads;
    pthread_attr_t thattr;
    _ThGridderArgs *thargs;
    int npth,npres;
    double xmin,xmax,dx;
    double ymin,ymax,dy;
    double zmin,zmax,dz;
    double *gnorm;
    unsigned int i;
    unsigned int offset;
    int x_index,y_index,z_index;
    unsigned int start,stop;
    int rc;
    void *status;


    /*determine axis minimum and maximum*/
    xmin = get_min(x,n);
    xmax = get_max(x,n);
    ymin = get_min(y,n);
    ymax = get_max(y,n);
    zmin = get_min(z,n);
    zmax = get_max(z,n);

    dx = (xmax-xmin)/(double)(nx-1);
    dy = (ymax-ymin)/(double)(ny-1);
    dz = (zmax-zmin)/(double)(nz-1);
    
    /*initialize data if requested*/
    if(!(flags&NO_DATA_INIT)){
        for(i=0;i<nx*ny*nz;i++) odata[i] = 0.;
    }

    /*check if normalization array is passed*/
    if(norm==NULL){
        gnorm = malloc(sizeof(double)*(nx*ny*nz));
        if(gnorm==NULL){
            fprintf(stderr,"Cannot allocate memory for normalization buffer!\n");
            return(-1);
        }
        /*initialize memory for norm*/
        for(i=0;i<nx*ny*nz;i++) gnorm[i] = 0.;
    }else{
        gnorm = norm;
    }

    /*determine the number of points to handle by each thread*/
    npres = n%nth;
    npth = (n-npres)/nth;

    /*allocate memory*/
    threads = malloc(sizeof(pthread_t)*(nth-1));
    if(threads == NULL){
        fprintf(stderr,"Cannot allocate thread array!\n");
        if(norm==NULL) free(gnorm);
        return(-1);
    }

    thargs = malloc(sizeof(_ThGridderArgs)*(nth-1));
    if(thargs == NULL){
        fprintf(stderr,"Cannot allocate thread argument array!\n");
        if(norm==NULL) free(gnorm);
        free(threads);
        return(-1);
    }

    /*create thread attributes and mutex lock objects*/
    pthread_attr_init(&thattr);
    pthread_attr_setdetachstate(&thattr,PTHREAD_CREATE_JOINABLE);
    pthread_mutex_init(&gridder_mutex,NULL);

    /*setup the thread parameters*/
    for(i=0;i<nth;i++){
        thargs[i].nth = nth;
        thargs[i].thid = i;
        thargs[i].npth = npth;
        thargs[i].npres = npres;
        thargs[i].x = x;
        thargs[i].y = y;
        thargs[i].z = z;
        thargs[i].data = data;
        thargs[i].n = n;
        thargs[i].xmin = xmin;
        thargs[i].ymax = xmax;
        thargs[i].ymin = ymin;
        thargs[i].ymax = ymax;
        thargs[i].zmin = zmin;
        thargs[i].zmax = zmax;
        thargs[i].dx = dx;
        thargs[i].dy = dy;
        thargs[i].dz = dz;
        thargs[i].nx = nx;
        thargs[i].ny = ny;
        thargs[i].nz = nz;
        thargs[i].odata = odata;
        thargs[i].norm = gnorm;
    }

    /*fire the threads*/
    for(i=1;i<nth;i++){
        rc = pthread_create(&threads[i],&thattr,gridder2d_th_worker,
                (void *)&thargs[i]);
        if(rc){
            fprintf(stderr,"Error creating thread %i!\n",i);
            free(threads);
            free(thargs);
            if (norm == NULL) free(gnorm);

            return(-1);
        }
    }

    /*run here the local code for this threads*/
    start = 0; 
    stop  = thargs[0].npres + thargs[0].npth;
    for(i=start;i<stop;i++){
        x_index = rint((thargs[0].x[i]-thargs[0].xmin)/thargs[0].dx);
        y_index = rint((thargs[0].y[i]-thargs[0].ymin)/thargs[0].dy);
        z_index = rint((thargs[0].z[i]-thargs[0].zmin)/thargs[0].dz);

        offset = x_index*thargs[0].ny*thargs[0].nz+
                 y_index*thargs[0].nz + z_index;

        pthread_mutex_lock(&gridder_mutex); 
        thargs[0].odata[offset] += thargs[0].data[i];
        thargs[0].norm[offset] += 1.;
        pthread_mutex_unlock(&gridder_mutex);
    }

    /*once the local code is inished all other threads are joined to 
     *until they are finished*/
    for(i=1;i<nth;i++){
        rc = pthread_join(threads[i],&status);
    }

    /*delete the thread attributes and mutex lock objects*/
    pthread_attr_destroy(&thattr);
    pthread_mutex_destroy(&gridder_mutex);

    /*perform normalization if requested*/
    if(!(flags&NO_NORMALIZATION)){
        for(i=0;i<nx*ny;i++){
            if(gnorm[i]>1.e-16){
                odata[i] = odata[i]/gnorm[i];
            }
        }
    }

    /*free memory*/
    free(threads);
    free(thargs);
    if (norm==NULL) free(gnorm);

    return(0);
}
/*}}}1*/

void *gridder3d_th_worker(void *arg)
/*{{{1*/
{
    unsigned int i;
    unsigned int x_index,y_index,z_index;
    unsigned int offset;
    _ThGridderArgs tharg;
    unsigned int start,stop;
    unsigned int *offset_buffer;
    unsigned int *data_index_buffer;
    unsigned int valid_point_count;
    unsigned int index;

    tharg = *(_ThGridderArgs *)arg;

    /*allocate local buffer memory*/
    offset_buffer = malloc(sizeof(unsigned int)*thargs.npth);
    data_index_buffer = malloc(sizeof(unsigned int)*thargs.npth);

    /*the master loop over all data points*/
    if (tharg.thid ==0 ){
        start = 0;
        stop  = tharg.npres + tharg.npth;
    }else{
        start = tharg.npres + tharg.thid*tharg.npth;
        stop  = start + tharg.npth;
    }

    for(i=start;i<stop;i++){
        x_index = (unsigned int)rint((tharg.x[i]-tharg.xmin)/tharg.dx);
        y_index = (unsigned int)rint((tharg.y[i]-tharg.ymin)/tharg.dy);
        z_index = (unsigned int)rint((tharg.z[i]-tharg.zmin)/tharg.dz);

        offset = x_index*tharg.ny*tharg.nz+y_index*tharg.nz
                 +z_index;

        offset_buffer[valid_point_count] = offset;
        data_index_buffer[valid_point_count] = i;
        valid_point_count ++;

        pthread_mutex_lock(&gridder_mutex);
        tharg.odata[offset] += tharg.data[i];
        tharg.norm[offset] += 1.;
        pthread_mutex_unlock(&gridder_mutex);
    }

    /*copy data*/
    pthread_mutex_lock(&gridder_mutex);
    for(i=0;i<valid_point_count;i++){
        offset = offset_buffer[i];
        index  = data_index_buffer[i];
        tharg.odata[offset] += tharg.data[index];
        tharg.norm[offset]  += 1.;
    }
    pthread_mutex_unlock(&gridder_mutex);

    /*free local memory*/
    free(offset_buffer);
    free(data_index_buffer);

    if (tharg.thid==0){
        return(0);
    }else{
        pthread_exit(NULL);
    }
}
/*}}}1*/


double get_min(double *a,unsigned int n)
/*{{{1*/
{
    double m;
    unsigned int i;

    m = a[0];
    for(i=0;i<n;i++){
        if(m<a[i]){
            m = a[i];
        }
    }
    return(m);
}
/*}}}1*/

double get_max(double *a,unsigned int n)
/*{{{1*/
{
    double m;
    unsigned int i;

    m=a[0];
    for(i=0;i<n;i++){
        if(m>a[i]){
            m = a[i];
        }
    }

    return(m);
}
/*}}}1*/
