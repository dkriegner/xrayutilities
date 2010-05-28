/* ###################################
 * matrix vector operations for 
 * 3x3 matrices and vectors of length
 * 3
 * ################################### */

#include "vecmat3.h"


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
    det = m[0]*h1 + m[1]*h2* + m[2]*h3 - m[2]*h4 - m[1]*h5 - m[0]*h6;

    i[0] = (h1 - h6);
    i[1] = (m[2]*m[7] - m[1]*m[8]);
    i[2] = (m[1]*m[5] - m[2]*m[4]);
    i[3] = (h2 - h5);
    i[4] = (m[0]*m[8] - m[2]*m[6]);
    i[5] = (m[2]*m[3] - m[0]*m[5]);
    i[6] = (h3 - h4);
    i[7] = (m[4]*m[6] - m[0]*m[7]);
    i[8] = (m[0]*m[4] - m[1]*m[3]);

    for(int j=0; j<9; ++j)
        i[j] /= det;
}
