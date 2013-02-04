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
 * Copyright (C) 2010,2012 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#ifndef _rot_matrix_h_
#define _rot_matrix_h_

/*##############################################
#   functions which implement rotation matrices
#   for all coordinate axes and rotation senses
#
#   the routines expect angles in radians
#   for conversion from degrees to radians
#   the functions and2rad and rad2ang are
#   supplied
################################################*/

#include <math.h>
#include "vecmat3.h"
#define M_PI 3.14159265358979323846
#define M_2PI (2*M_PI)

#define cdeg2rad (M_PI/180.)
#define crad2deg (180./M_PI)

/*
double deg2rad(double angle);
double rad2deg(double radian);
*/

#define deg2rad(ang) (ang*cdeg2rad)
#define rad2deg(rad) (rad*crad2deg)

typedef void (*fp_rot)(double,double *);

INLINE void rotation_xp(double a,double *mat) __attribute__((always_inline));
INLINE void rotation_yp(double a,double *mat) __attribute__((always_inline));
INLINE void rotation_zp(double a,double *mat) __attribute__((always_inline));

INLINE void rotation_xm(double a,double *mat) __attribute__((always_inline));
INLINE void rotation_ym(double a,double *mat) __attribute__((always_inline));
INLINE void rotation_zm(double a,double *mat) __attribute__((always_inline));

INLINE void rotation_kappa(double a, double *mat) __attribute__((always_inline));

INLINE void rotation_arb(double a,double *RESTRICT e,double *RESTRICT mat) __attribute__((always_inline));


INLINE void rotation_xp(double a,double *mat){
    double sa=sin(a), ca=cos(a);
    mat[0] = 1.; mat[1] = 0.; mat[2] = 0.;
    mat[3] = 0.; mat[4] = ca; mat[5] = -sa;
    mat[6] = 0.; mat[7] = sa; mat[8] = ca;
}

INLINE void rotation_xm(double a,double *mat){
    double sa=sin(a), ca=cos(a);
    mat[0] = 1.; mat[1] = 0.; mat[2] = 0.;
    mat[3] = 0.; mat[4] = ca; mat[5] = sa;
    mat[6] = 0.; mat[7] = -sa; mat[8] = ca;
}

INLINE void rotation_yp(double a,double *mat){
    double sa=sin(a), ca=cos(a);
    mat[0] = ca; mat[1] = 0.; mat[2] = sa;
    mat[3] = 0.; mat[4] = 1.; mat[5] = 0.;
    mat[6] = -sa; mat[7] = 0.; mat[8] = ca;
}

INLINE void rotation_ym(double a,double *mat){
    double sa=sin(a), ca=cos(a);
    mat[0] = ca; mat[1] = 0.; mat[2] = -sa;
    mat[3] = 0.; mat[4] = 1.; mat[5] = 0.;
    mat[6] = sa; mat[7] = 0.; mat[8] = ca;
}

INLINE void rotation_zp(double a,double *mat){
    double sa=sin(a), ca=cos(a);
    mat[0] = ca; mat[1] = -sa; mat[2] = 0.;
    mat[3] = sa; mat[4] = ca; mat[5] = 0.;
    mat[6] = 0.; mat[7] = 0.; mat[8] = 1.;
}

INLINE void rotation_zm(double a,double *mat){
    double sa=sin(a), ca=cos(a);
    mat[0] = ca; mat[1] = sa; mat[2] = 0.;
    mat[3] = -sa; mat[4] = ca; mat[5] = 0.;
    mat[6] = 0.; mat[7] = 0.; mat[8] = 1.;
}

INLINE void rotation_kappa(double a, double *mat){
    double e[3];
    e[0] = mat[0]; e[1] = mat[1]; e[2] = mat[2];
    rotation_arb(a,e,mat);
}

INLINE void rotation_arb(double a,double *RESTRICT e,double *RESTRICT mat) {
    double sa = sin(a), ca=cos(a);
    double mtemp[9],mtemp2[9];

    /* e must be normalized */

    /* ca*(ident(3) - vec(e) o vec(e))*/
    ident(mat);
    tensorprod(e,e,mtemp);
    diffmat(mat,mtemp);
    matmulc(mat,ca);

    /* tensorprod(vec(e),vec(e)) */
    summat(mat,mtemp);

    /* sa*(vec(e) cross ident(3)) */
    ident(mtemp2);
    vecmatcross(e,mtemp2,mtemp);
    matmulc(mtemp,sa);
    summat(mat,mtemp);
}


#endif
