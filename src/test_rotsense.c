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
 * Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

/* Test file to determine rotation sense and test 
   blas routines */

#include "rot_matrix.h"
#include "vecmat3.h"
#include <stdio.h>
#include <cblas.h>

int main(void) {
    double x[] = {1.,0.,0.};
    double y[] = {0.,1.,0.};
    double z[] = {0.,0.,1.};
    double r[3],mx[9],my[9],mz[9];
    double ang = 90.;

    ang = deg2rad(ang);
    rotation_xp(ang,mx); 
    rotation_yp(ang,my);
    rotation_zp(ang,mz);
    /* apply rotation */
    printf("z:  %5.2f %5.2f %5.2f\n",z[0],z[1],z[2]);
    matvec(mx,z,r);
    printf("mx: %5.2f %5.2f %5.2f\n",r[0],r[1],r[2]);
    matvec(my,z,r);
    printf("my: %5.2f %5.2f %5.2f\n",r[0],r[1],r[2]);

    printf("y:  %5.2f %5.2f %5.2f\n",y[0],y[1],y[2]);
    matvec(mx,y,r);
    printf("mx: %5.2f %5.2f %5.2f\n",r[0],r[1],r[2]);
    matvec(mz,y,r);
    printf("mz: %5.2f %5.2f %5.2f\n",r[0],r[1],r[2]);

    printf("x:  %5.2f %5.2f %5.2f\n",x[0],x[1],x[2]);
    matvec(my,x,r);
    printf("my: %5.2f %5.2f %5.2f\n",r[0],r[1],r[2]);
    matvec(mz,x,r);
    printf("mz: %5.2f %5.2f %5.2f\n",r[0],r[1],r[2]);
    return 0;
}
