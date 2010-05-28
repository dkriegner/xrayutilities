#ifndef _rot_matrix_h_
#define _rot_matrix_h_

#include <math.h>
#define M_PI 3.14159265358979323846264338327
#define M_2PI (2*M_PI)
#define INLINE  

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

#endif
