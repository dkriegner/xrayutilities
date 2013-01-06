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
 * Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#ifndef _vecmat3_h_
#define _vecmat3_h_

/* ###################################
 * matrix vector operations for
 * 3x3 matrices and vectors of length
 * 3
 * ################################### */

#include <math.h>

#define INLINE extern inline
#define RESTRICT restrict


INLINE void ident(double *m) __attribute__((always_inline));

INLINE void sumvec(double *RESTRICT v1,double *RESTRICT v2) __attribute__((always_inline));

INLINE void diffvec(double *RESTRICT v1,double *RESTRICT v2) __attribute__((always_inline));

INLINE double norm(double *v) __attribute__((always_inline));

INLINE void normalize(double *v) __attribute__((always_inline));

INLINE void veccopy(double *RESTRICT v1, double *RESTRICT v2) __attribute__((always_inline));

INLINE void vecmul(double *RESTRICT r, double a) __attribute__((always_inline));

INLINE void cross(double *RESTRICT v1, double *RESTRICT v2, double *RESTRICT r) __attribute__((always_inline));

INLINE void vecmatcross(double *RESTRICT v, double *RESTRICT m, double *RESTRICT mr) __attribute__((always_inline));

INLINE void matmul(double *RESTRICT m1, double *RESTRICT m2) __attribute__((always_inline));

INLINE void matmulc(double *RESTRICT m, double c) __attribute__((always_inline));

INLINE void matvec(double *RESTRICT m, double *RESTRICT v, double *RESTRICT r) __attribute__((always_inline));

INLINE void tensorprod(double *RESTRICT v1, double *RESTRICT v2, double *RESTRICT m) __attribute__((always_inline));

INLINE void summat(double *RESTRICT m1,double *RESTRICT m2) __attribute__((always_inline));

INLINE void diffmat(double *RESTRICT m1,double *RESTRICT m2) __attribute__((always_inline));

INLINE void inversemat(double *RESTRICT m, double *RESTRICT i) __attribute__((always_inline));

INLINE double determinant(double *RESTRICT m) __attribute__((always_inline));


INLINE void ident(double *m) {
    m[0] = 1.; m[1] = 0.; m[2] =0.;
    m[3] = 0.; m[4] = 1.; m[5] =0.;
    m[6] = 0.; m[7] = 0.; m[8] =1.;
}

INLINE void sumvec(double *RESTRICT v1,double *RESTRICT v2) {
    for(int i=0; i<3; ++i)
        v1[i] += v2[i];
}

INLINE void diffvec(double *RESTRICT v1,double *RESTRICT v2) {
    for(int i=0; i<3; ++i)
        v1[i] -= v2[i];
}

INLINE double norm(double *v) {
    double n=0.;
    for(int i=0; i<3; ++i)
        n += v[i]*v[i];
    return sqrt(n);
}

INLINE void normalize(double *v) {
    double n=norm(v);
    for(int i=0; i<3; ++i)
        v[i] /= n;
}

INLINE void veccopy(double *RESTRICT v1, double *RESTRICT v2) {
    for(int i=0; i<3; ++i)
        v1[i] = v2[i];
}

INLINE void vecmul(double *RESTRICT r, double a) {
    for(int i=0; i<3; ++i)
        r[i] *= a;
}

INLINE void cross(double *RESTRICT v1, double *RESTRICT v2, double *RESTRICT r) {
    r[0] =  v1[1]*v2[2] - v1[2]*v2[1];
    r[1] = -v1[0]*v2[2] + v1[2]*v2[0];
    r[2] =  v1[0]*v2[1] - v1[1]*v2[0];
}

INLINE void vecmatcross(double *RESTRICT v, double *RESTRICT m, double *RESTRICT mr) {
    for (int i=0; i<9; i=i+3) {
        mr[0+i] =  v[1]*m[2+i] - v[2]*m[1+i];
        mr[1+i] = -v[0]*m[2+i] + v[2]*m[0+i];
        mr[2+i] =  v[0]*m[1+i] - v[1]*m[0+i];
    }
}

INLINE void matmulc(double *RESTRICT m, double c) {
    for (int i=0; i<9; i=i+1) {
        m[i] *= c;
    }
}

INLINE void matvec(double *RESTRICT m, double *RESTRICT v, double *RESTRICT r) {
    r[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2];
    r[1] = m[3]*v[0] + m[4]*v[1] + m[5]*v[2];
    r[2] = m[6]*v[0] + m[7]*v[1] + m[8]*v[2];
}

INLINE void matmul(double *RESTRICT m1, double *RESTRICT m2) {
    double a,b,c;
    for(int i=0; i<9; i=i+3) {
        a = m1[i]*m2[0] + m1[i+1]*m2[3] + m1[i+2]*m2[6];
        b = m1[i]*m2[1] + m1[i+1]*m2[4] + m1[i+2]*m2[7];
        c = m1[i]*m2[2] + m1[i+1]*m2[5] + m1[i+2]*m2[8];
        m1[i] = a;
        m1[i+1] = b;
        m1[i+2] = c;
    }
}

INLINE void tensorprod(double *RESTRICT v1, double *RESTRICT v2, double *RESTRICT m) {
    for(int i=0; i<3; i=i+1) {
        for(int j=0; j<3; j=j+1) {
            m[i*3+j] = v1[i]*v2[j];
        }
    }
}

INLINE void summat(double *RESTRICT m1,double *RESTRICT m2) {
    for(int i=0; i<9; ++i)
        m1[i] += m2[i];
}

INLINE void diffmat(double *RESTRICT m1,double *RESTRICT m2) {
    for(int i=0; i<9; ++i)
        m1[i] -= m2[i];
}

INLINE void inversemat(double *RESTRICT m, double *RESTRICT i) {
    double det;
    double h1,h2,h3,h4,h5,h6;

    h1 = m[4]*m[8]; // m11*m22
    h2 = m[5]*m[6]; // m12*m20
    h3 = m[3]*m[7]; // m10*m21
    h4 = m[4]*m[6]; // m11*m20
    h5 = m[3]*m[8]; // m10*m22
    h6 = m[5]*m[7]; // m12*m21
    det = m[0]*h1 + m[1]*h2 + m[2]*h3 - m[2]*h4 - m[1]*h5 - m[0]*h6;

    i[0] = (h1 - h6);
    i[1] = (m[2]*m[7] - m[1]*m[8]);
    i[2] = (m[1]*m[5] - m[2]*m[4]);
    i[3] = (h2 - h5);
    i[4] = (m[0]*m[8] - m[2]*m[6]);
    i[5] = (m[2]*m[3] - m[0]*m[5]);
    i[6] = (h3 - h4);
    i[7] = (m[1]*m[6] - m[0]*m[7]);
    i[8] = (m[0]*m[4] - m[1]*m[3]);

    for(int j=0; j<9; ++j)
        i[j] /= det;
}

INLINE double determinant(double *RESTRICT m) {
    double h1,h2,h3,h4,h5,h6;
    double det=0;

    h1 = m[4]*m[8]; // m11*m22
    h2 = m[5]*m[6]; // m12*m20
    h3 = m[3]*m[7]; // m10*m21
    h4 = m[4]*m[6]; // m11*m20
    h5 = m[3]*m[8]; // m10*m22
    h6 = m[5]*m[7]; // m12*m21

    det = m[0]*h1 + m[1]*h2 + m[2]*h3 - m[2]*h4 - m[1]*h5 - m[0]*h6;
    return det;
}

#endif
