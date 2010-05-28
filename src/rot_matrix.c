/*##############################################
#   functions which implement rotation matrices 
#   for all coordinate axes and rotation senses
#
#   the routines expect angles in radians
#   for conversion from degrees to radians
#   the functions and2rad and rad2ang are 
#   supplied
################################################*/

#include "rot_matrix.h"

/*
double deg2rad(double angle) {
    return (angle*cdeg2rad);    
}

double rad2deg(double radian){
    return (radian*crad2deg);
}
*/

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
