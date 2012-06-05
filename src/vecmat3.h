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

#include <math.h>

#define INLINE
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

#endif
