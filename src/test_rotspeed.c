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
 * Copyright (C) 2010-2012 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

/* Test file to determine the speed of a lot of rotations  
   for different implementations and optimizations */

#include "rot_matrix.h"
#include "vecmat3.h"
#include <stdio.h>
#include <cblas.h>
#define __USE_POSIX199309
#include <time.h>

#define N 1000000
#define axes 6

int main(void) {
    double x[] = {1.,0.,0.};
    double y[] = {0.,1.,0.};
    double z[] = {0.,0.,1.};
    double r[3],mx[9],my[9],mz[9],inv[9];
    double elapsed, ang = 90.;
    clock_t time1,time0;
    struct timespec wtime0;
    struct timespec wtime1;

    clock_gettime(CLOCK_REALTIME,&wtime0);
    time0 = clock();
    for(int i=0; i<N; ++i) {
        ang = deg2rad(M_PI/(double)i);
        for(int j=0; j<axes; ++j) { 
            rotation_xp(ang,mx); 
            rotation_yp(ang,my);
            rotation_zp(ang,mz);
        }
    }
    time1 = clock();
    clock_gettime(CLOCK_REALTIME,&wtime1);

    printf("rotation matrix determination\n");
    elapsed = ((double) (time1 - time0)) / CLOCKS_PER_SEC;
    printf(" CPU time:  %8.4f\n",elapsed);
    elapsed = (double) (wtime1.tv_sec - wtime0.tv_sec) + ((double)(wtime1.tv_nsec - wtime0.tv_nsec)/1.e9);
    printf(" Wall time: %8.4f\n",elapsed);


    clock_gettime(CLOCK_REALTIME,&wtime0);
    time0 = clock();
    for(int i=0; i<N; ++i) {
        ang = deg2rad(M_PI/(double)i);
        for(int j=0; j<axes; ++j) { 
           /*matvec(mx,z,r); 
           matvec(my,x,r); 
           matvec(mz,y,r); */
           cblas_dgemv(CblasRowMajor,CblasNoTrans,3,3,1,mx,3,z,1,0.,r,1); 
           cblas_dgemv(CblasRowMajor,CblasNoTrans,3,3,1,my,3,x,1,0.,r,1); 
           cblas_dgemv(CblasRowMajor,CblasNoTrans,3,3,1,mz,3,y,1,0.,r,1); 
        }
    }
    time1 = clock();
    clock_gettime(CLOCK_REALTIME,&wtime1);
    
    printf("matrix multiplication (BLAS)\n");
    elapsed = ((double) (time1 - time0)) / CLOCKS_PER_SEC;
    printf(" CPU time:  %8.4f\n",elapsed);
    elapsed = (double) (wtime1.tv_sec - wtime0.tv_sec) + ((double)(wtime1.tv_nsec - wtime0.tv_nsec)/1.e9);
    printf(" Wall time: %8.4f\n",elapsed);


    clock_gettime(CLOCK_REALTIME,&wtime0);
    time0 = clock();
    for(int i=0; i<N; ++i) {
        ang = deg2rad(M_PI/(double)i);
        for(int j=0; j<axes; ++j) { 
           matvec(mx,z,r); 
           matvec(my,x,r); 
           matvec(mz,y,r);
        }
    }
    time1 = clock();
    clock_gettime(CLOCK_REALTIME,&wtime1);
    
    printf("matrix multiplication\n");
    elapsed = ((double) (time1 - time0)) / CLOCKS_PER_SEC;
    printf(" CPU time:  %8.4f\n",elapsed);
    elapsed = (double) (wtime1.tv_sec - wtime0.tv_sec) + ((double)(wtime1.tv_nsec - wtime0.tv_nsec)/1.e9);
    printf(" Wall time: %8.4f\n",elapsed);


    clock_gettime(CLOCK_REALTIME,&wtime0);
    time0 = clock();
    for(int i=0; i<N; ++i) {
        ang = deg2rad(M_PI/(double)i);
        for(int j=0; j<axes; ++j) { 
           inversemat(mx,inv); 
        }
    }
    time1 = clock();
    clock_gettime(CLOCK_REALTIME,&wtime1);
    
    printf("matrix inversion\n");
    elapsed = ((double) (time1 - time0)) / CLOCKS_PER_SEC;
    printf(" CPU time:  %8.4f\n",elapsed);
    elapsed = (double) (wtime1.tv_sec - wtime0.tv_sec) + ((double)(wtime1.tv_nsec - wtime0.tv_nsec)/1.e9);
    printf(" Wall time: %8.4f\n",elapsed);


    return 0;
}
