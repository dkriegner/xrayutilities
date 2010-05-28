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

INLINE void matmul(double *RESTRICT m1, double *RESTRICT m2) __attribute__((always_inline));

INLINE void matvec(double *RESTRICT m, double *RESTRICT v, double *RESTRICT r) __attribute__((always_inline));

INLINE void summat(double *RESTRICT m1,double *RESTRICT m2) __attribute__((always_inline));

INLINE void diffmat(double *RESTRICT m1,double *RESTRICT m2) __attribute__((always_inline));

INLINE void inversemat(double *RESTRICT m, double *RESTRICT i) __attribute__((always_inline));

#endif
