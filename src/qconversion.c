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
 * Copyright (C) 2010-2013 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

/* ######################################
 *   conversion of angular coordinates
 *   to reciprocal space
 *   using general algorithms to work
 *   for different types of geometries
 *   and detectors
 * ######################################*/

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#include <numpy/arrayobject.h>

#include "qconversion.h"
#include <ctype.h>
#include <math.h>
#ifdef __OPENMP__
#include <omp.h>
#endif

/* ###################################
 * matrix vector operations for
 * 3x3 matrices and vectors of length
 * 3
 * ################################### */

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


/*##############################################
#   functions which implement rotation matrices
#   for all coordinate axes and rotation senses
#
#   the routines expect angles in radians
#   for conversion from degrees to radians
#   the functions and2rad and rad2ang are
#   supplied
################################################*/

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


/* #######################################
 *  conversion helper functions
 * #######################################*/

int print_matrix(double *m) {
    for(int i=0;i<9;i+=3) {
        printf("%8.5g %8.5g %8.5g\n",m[i],m[i+1],m[i+2]);
    }
    printf("\n");
    return 0;
}

int print_vector(double *m) {
    printf("\n%8.5g %8.5g %8.5g\n",m[0],m[1],m[2]);
    return 0;
}

int determine_detector_pixel(double *rpixel,char *dir, double dpixel, double *r_i, double tilt) {
    /* determine the direction of linear direction or one of the directions
     * of an area detector.
     * the function returns the vector containing the distance from one to
     * the next pixel
     * a tilt of the detector axis with respect to the coordinate axis can
     * be considered as well! rotation of pixel direction around the
     * crossproduct of primary beam and detector axis.
     * this is mainly usefull for linear detectors, since the tilt of area
     * detectors is handled different.
     * */

    double tiltaxis[3], tiltmat[9];

    for(int i=0; i<3; ++i)
        rpixel[i] = 0.;

    switch(tolower(dir[0])) {
        case 'x':
            switch(dir[1]) {
                case '+':
                    rpixel[0] = dpixel;
                break;
                case '-':
                    rpixel[0] = -dpixel;
                break;
                default:
                    printf("XU.Qconversion(c): detector determination: no valid direction sign given\n");
                    return 1;
            }
        break;
        case 'y':
            switch(dir[1]) {
                case '+':
                    rpixel[1] = dpixel;
                break;
                case '-':
                    rpixel[1] = -dpixel;
                break;
                default:
                    printf("XU.Qconversion(c): detector determination: no valid direction sign given\n");
                    return 1;
            }
        break;
        case 'z':
            switch(dir[1]) {
                case '+':
                    rpixel[2] = dpixel;
                break;
                case '-':
                    rpixel[2] = -dpixel;
                break;
                default:
                    printf("XU.Qconversion(c): detector determination: no valid direction sign given\n");
                    return 1;
            }
        break;
        default:
            printf("XU.Qconversion(c): detector determination: no valid detector direction given\n");
            return 2;
    }

    /* include possible tilt of detector axis with respect to its direction */
    cross(r_i,rpixel,tiltaxis);
    normalize(tiltaxis);
    //print_vector(tiltaxis);
    /* create needed rotation matrix */
    rotation_arb(tilt,tiltaxis,tiltmat);
    //print_matrix(tiltmat);
    /* rotate rpixel */
    matvec(tiltmat,rpixel,tiltaxis);
    //print_vector(rpixel);
    veccopy(rpixel,tiltaxis);
    //print_vector(rpixel);
    return 0;
}

int determine_axes_directions(fp_rot *fp_circles,char *stringAxis,int n) {
    /* feed the function pointer array with the correct
     * rotation matrix generating functions
     * */

    for(int i=0; i<n; ++i) {
        switch(tolower(stringAxis[2*i])) {
            case 'x':
                switch(stringAxis[2*i+1]) {
                    case '+':
                        fp_circles[i] = &rotation_xp;
                    break;
                    case '-':
                        fp_circles[i] = &rotation_xm;
                    break;
                    default:
                        printf("XU.Qconversion(c): axis determination: no valid rotation sense found\n");
                        return 1;
                }
            break;
            case 'y':
                switch(stringAxis[2*i+1]) {
                    case '+':
                        fp_circles[i] = &rotation_yp;
                    break;
                    case '-':
                        fp_circles[i] = &rotation_ym;
                    break;
                    default:
                        printf("XU.Qconversion(c): axis determination: no valid rotation sense found\n");
                        return 1;
                }
            break;
            case 'z':
                switch(stringAxis[2*i+1]) {
                    case '+':
                        fp_circles[i] = &rotation_zp;
                    break;
                    case '-':
                        fp_circles[i] = &rotation_zm;
                    break;
                    default:
                        printf("XU.Qconversion(c): axis determination: no valid rotation sense found\n");
                        return 1;
                }
            break;
            case 'k':
                fp_circles[i] = &rotation_kappa;
            break;
            default:
                printf("XU.Qconversion(c): axis determination: no valid axis direction found!\n");
                return 2;
        }
    }

    return 0;
}


/* #######################################
 *  conversion functions
 * #######################################*/

PyObject* ang2q_conversion(PyObject *self, PyObject *args)
//int ang2q_conversion(double *sampleAngles,double *detectorAngles, double *qpos, double *ri, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double *UB, double lambda)
   /* conversion of Npoints of goniometer positions to reciprocal space
    * for a setup with point detector
    *
    *   Parameters
    *   ----------
    *    sampleAngles .. angular positions of the sample goniometer (Npoints,Ns) (in)
    *    detectorAngles. angular positions of the detector goniometer (Npoints,Nd) (in)
    *    ri ............ direction of primary beam (length irrelevant) (angles zero) (in)
    *    sampleAxis .... string with sample axis directions (in)
    *    detectorAxis .. string with detector axis directions (in)
    *    kappadir ...... rotation axis of a possible kappa circle (in)
    *    UB ............ orientation matrix and reciprocal space conversion of investigated crystal (9) (in)
    *    lambda ........ wavelength of the used x-rays (Angstreom) (in)
    *
    *   Returns
    *   -------
    *    qpos .......... momentum transfer (Npoints,3) (out)
    *    
    *   */
{
    double mtemp[9],mtemp2[9], ms[9], md[9]; //matrices
    double local_ri[3]; // copy of primary beam direction
    int i,j; // needed indices
    int Ns,Nd; // number of sample and detector circles
    int Npoints; // number of angular positions
    double lambda; // x-ray wavelength
    char *sampleAxis,*detectorAxis; // string with sample and detector axis
    double *sampleAngles,*detectorAngles, *ri, *kappadir, *UB, *qpos; // c-arrays for further usage
    
    PyArrayObject *sampleAnglesArr=NULL, *detectorAnglesArr=NULL, *riArr=NULL, *kappadirArr=NULL, 
                  *UBArr=NULL, *qposArr=NULL; // numpy arrays

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!O!O!ssO!O!d",&PyArray_Type, &sampleAnglesArr, 
                                      &PyArray_Type, &detectorAnglesArr,
                                      &PyArray_Type, &riArr,
                                      &sampleAxis, &detectorAxis,
                                      &PyArray_Type, &kappadirArr,
                                      &PyArray_Type, &UBArr,
                                      &lambda)) { 
        return NULL;
    }
    
    // check Python array dimensions and types
    if (PyArray_NDIM(sampleAnglesArr) != 2 || PyArray_TYPE(sampleAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Npoints = PyArray_DIMS(sampleAnglesArr)[0];
    Ns = PyArray_DIMS(sampleAnglesArr)[1];
    sampleAngles = (double *) PyArray_DATA(sampleAnglesArr);
    
    if (PyArray_NDIM(detectorAnglesArr) != 2 || PyArray_TYPE(detectorAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Nd = PyArray_DIMS(detectorAnglesArr)[1];
    detectorAngles = (double *) PyArray_DATA(detectorAnglesArr);

    if (PyArray_NDIM(riArr) != 1 || PyArray_TYPE(riArr) != NPY_DOUBLE || PyArray_DIMS(riArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    ri = (double *) PyArray_DATA(riArr);
    
    if (PyArray_NDIM(kappadirArr) != 1 || PyArray_TYPE(kappadirArr) != NPY_DOUBLE || PyArray_DIMS(kappadirArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    kappadir = (double *) PyArray_DATA(kappadirArr);

    if (PyArray_NDIM(UBArr) != 2 || PyArray_TYPE(UBArr) != NPY_DOUBLE || PyArray_DIMS(UBArr)[0] != 3 || PyArray_DIMS(UBArr)[1] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double and shape (3,3)");
        return NULL; }
    UB = (double *) PyArray_DATA(UBArr);

    // create output ndarray
    npy_intp nout[2];
    nout[0] = Npoints;
    nout[1] = 3;
    qposArr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    qpos = (double *) PyArray_DATA(qposArr); 

    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    // arrays with function pointers to rotation matrix functions
    fp_rot sampleRotation[Ns];
    fp_rot detectorRotation[Nd];

    //printf("general conversion ang2q\n");
    // determine axes directions
    if(determine_axes_directions(sampleRotation,sampleAxis,Ns) != 0) {
        PyErr_SetString(PyExc_ValueError,"sampleRotation axis could not be identified");
        return NULL;
    }
    if(determine_axes_directions(detectorRotation,detectorAxis,Nd) != 0) {
        PyErr_SetString(PyExc_ValueError,"detectorRotation axis could not be identified");
        return NULL;
    }

    // give ri correct length
    veccopy(local_ri,ri);
    normalize(local_ri);
    vecmul(local_ri,M_2PI/lambda);

    //debug
    //print_matrix(UB);

    // calculate rotation matices and perform rotations
    #pragma omp parallel for default(shared) \
            private(i,j,mtemp,mtemp2,ms,md) \
            schedule(static)
    for(i=0; i<Npoints; ++i) {
        // determine sample rotations
        ident(mtemp);
        for(j=0; j<Ns; ++j) {
            // load kappa direction into matrix (just needed for kappa goniometer)
            mtemp2[0] = kappadir[0]; mtemp2[1] = kappadir[1]; mtemp2[2] = kappadir[2];
            sampleRotation[j](sampleAngles[Ns*i+j],mtemp2);
            matmul(mtemp,mtemp2);
        }
        // apply rotation of orientation matrix
        matmul(mtemp,UB);
        // determine inverse matrix
        inversemat(mtemp,ms);

        // determine detector rotations
        ident(md);
        for (j=0; j<Nd; ++j) {
            detectorRotation[j](detectorAngles[Nd*i+j],mtemp);
            matmul(md,mtemp);
        }
        ident(mtemp);
        diffmat(md,mtemp);

        matmul(ms,md);
        // ms contains now the rotation matrix to determine the momentum transfer
        // calculate the momentum transfer
        matvec(ms, local_ri, &qpos[3*i]);
    }

    // return output array
    return PyArray_Return(qposArr);
}

PyObject* ang2q_conversion_linear(PyObject *self, PyObject *args)
//int ang2q_conversion_linear(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double cch, double dpixel, int *roi, char *dir, double tilt, double *UB, double lambda)
   /* conversion of Npoints of goniometer positions to reciprocal space
    * for a linear detector with a given pixel size mounted along one of
    * the coordinate axis
    *
    *   Parameters
    *   ----------
    *   sampleAngles .... angular positions of the goniometer (Npoints,Ns) (in)
    *   detectorAngles .. angular positions of the detector goniometer (Npoints,Nd) (in)
    *   rcch ............ direction + distance of center channel (angles zero) (in)
    *   sampleAxis ...... string with sample axis directions (in)
    *   detectorAxis .... string with detector axis directions (in)
    *   kappadir ........ rotation axis of a possible kappa circle (in)
    *   cch ............. center channel of the detector (in)
    *   dpixel .......... width of one pixel, same unit as distance rcch (in)
    *   roi ............. region of interest of the detector (in)
    *   dir ............. direction of the detector, e.g.: "x+" (in)
    *   tilt ............ tilt of the detector direction from dir (in)
    *   UB .............. orientation matrix and reciprocal space conversion of investigated crystal (9) (in)
    *   lambda .......... wavelength of the used x-rays in Angstroem (in)
    *   
    *   Returns
    *   -------
    *   qpos ............ momentum transfer (Npoints*Nch,3) (out)
    *   */
{
    double mtemp[9],mtemp2[9], ms[9], md[9]; //matrices
    double rd[3],rpixel[3],rcchp[3]; // detector position
    double r_i[3],rtemp[3]; //center channel direction
    int i,j,k; // needed indices
    int Ns,Nd; // number of sample and detector circles
    int Npoints; // number of angular positions
    int Nch; // number of channels in region of interest
    double f,lambda,cch,dpixel,tilt; // x-ray wavelength, f=M_2PI/lambda and detector parameters
    char *sampleAxis,*detectorAxis,*dir; // string with sample and detector axis, and detector direction
    double *sampleAngles,*detectorAngles, *rcch, *kappadir, *UB, *qpos; // c-arrays for further usage
    int *roi; //  region of interest integer array
    
    PyArrayObject *sampleAnglesArr=NULL, *detectorAnglesArr=NULL, *rcchArr=NULL,
                  *kappadirArr=NULL, *roiArr=NULL, *UBArr=NULL, *qposArr=NULL; // numpy arrays

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!O!O!ssO!ddO!sdO!d",&PyArray_Type, &sampleAnglesArr, 
                                      &PyArray_Type, &detectorAnglesArr,
                                      &PyArray_Type, &rcchArr,
                                      &sampleAxis, &detectorAxis,
                                      &PyArray_Type, &kappadirArr,
                                      &cch, &dpixel, &PyArray_Type, &roiArr, 
                                      &dir, &tilt,
                                      &PyArray_Type, &UBArr,
                                      &lambda)) { 
        return NULL;
    }
    
    // check Python array dimensions and types
    if (PyArray_NDIM(sampleAnglesArr) != 2 || PyArray_TYPE(sampleAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Npoints = PyArray_DIMS(sampleAnglesArr)[0];
    Ns = PyArray_DIMS(sampleAnglesArr)[1];
    sampleAngles = (double *) PyArray_DATA(sampleAnglesArr);
    
    if (PyArray_NDIM(detectorAnglesArr) != 2 || PyArray_TYPE(detectorAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Nd = PyArray_DIMS(detectorAnglesArr)[1];
    detectorAngles = (double *) PyArray_DATA(detectorAnglesArr);

    if (PyArray_NDIM(rcchArr) != 1 || PyArray_TYPE(rcchArr) != NPY_DOUBLE || PyArray_DIMS(rcchArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    rcch = (double *) PyArray_DATA(rcchArr);
    
    if (PyArray_NDIM(kappadirArr) != 1 || PyArray_TYPE(kappadirArr) != NPY_DOUBLE || PyArray_DIMS(kappadirArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    kappadir = (double *) PyArray_DATA(kappadirArr);

    if (PyArray_NDIM(UBArr) != 2 || PyArray_TYPE(UBArr) != NPY_DOUBLE || PyArray_DIMS(UBArr)[0] != 3 || PyArray_DIMS(UBArr)[1] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double and shape (3,3)");
        return NULL; }
    UB = (double *) PyArray_DATA(UBArr);

    if (PyArray_NDIM(roiArr) != 1 || PyArray_TYPE(roiArr) != NPY_INT || PyArray_DIMS(roiArr)[0] != 2) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type int and size 2");
        return NULL; }
    roi = (int *) PyArray_DATA(roiArr);
    
    // derived values from input parameters
    Nch = roi[1]-roi[0]; // number of channels
    f=M_2PI/lambda;
    
    // create output ndarray
    npy_intp nout[2];
    nout[0] = Npoints*Nch;
    nout[1] = 3;
    qposArr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    qpos = (double *) PyArray_DATA(qposArr); 


    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    // arrays with function pointers to rotation matrix functions
    fp_rot sampleRotation[Ns];
    fp_rot detectorRotation[Nd];

    //printf("general conversion ang2q (linear detector)\n");
    // determine axes directions
    if(determine_axes_directions(sampleRotation,sampleAxis,Ns) != 0) {
        PyErr_SetString(PyExc_ValueError,"sampleRotation axis could not be identified");
        return NULL;
    }
    if(determine_axes_directions(detectorRotation,detectorAxis,Nd) != 0) {
        PyErr_SetString(PyExc_ValueError,"detectorRotation axis could not be identified");
        return NULL;
    }

    veccopy(r_i,rcch);
    normalize(r_i);
    // determine detector pixel vector
    if(determine_detector_pixel(rpixel, dir, dpixel, r_i, tilt) != 0) {
        PyErr_SetString(PyExc_ValueError,"detector direction could not be identified");
        return NULL;
    }
    for(int k=0; k<3; ++k)
        rcchp[k] = rpixel[k]*cch;

    // calculate rotation matices and perform rotations
    #pragma omp parallel for default(shared) \
            private(i,j,k,mtemp,mtemp2,ms,md,rd,rtemp) \
            schedule(static)
    for(i=0; i<Npoints; ++i) {
        // determine sample rotations
        ident(mtemp);
        for(j=0; j<Ns; ++j) {
            // load kappa direction into matrix (just needed for kappa goniometer)
            mtemp2[0] = kappadir[0]; mtemp2[1] = kappadir[1]; mtemp2[2] = kappadir[2];
            sampleRotation[j](sampleAngles[Ns*i+j],mtemp2);
            matmul(mtemp,mtemp2);
        }
        // apply rotation of orientation matrix
        matmul(mtemp,UB);
        // determine inverse matrix
        inversemat(mtemp,ms);

        // determine detector rotations
        ident(md);
        for (j=0; j<Nd; ++j) {
            detectorRotation[j](detectorAngles[Nd*i+j],mtemp);
            matmul(md,mtemp);
        }

        // ms contains now the inverse rotation matrix for the sample circles
        // md contains the detector rotation matrix
        // calculate the momentum transfer for each detector pixel
        for (j=roi[0]; j<roi[1]; ++j) {
            for (k=0; k<3; ++k)
                rd[k] = j*rpixel[k] - rcchp[k];
            sumvec(rd,rcch);
            normalize(rd);
            // rd contains detector pixel direction, r_i contains primary beam direction
            matvec(md,rd,rtemp);
            diffvec(rtemp,r_i);
            vecmul(rtemp,f);
            // determine momentum transfer
            matvec(ms, rtemp, &qpos[3*(i*Nch+j-roi[0])]);
        }
    }

    // return output array
    return PyArray_Return(qposArr);
}

PyObject* ang2q_conversion_area(PyObject *self, PyObject *args)
//int ang2q_conversion_area(double *sampleAngles, double *detectorAngles, double *qpos, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double cch1, double cch2, double dpixel1, double dpixel2, int *roi, char *dir1, char *dir2, double tiltazimuth, double tilt, double *UB, double lambda)
   /* conversion of Npoints of goniometer positions to reciprocal space
    * for a area detector with a given pixel size mounted along one of
    * the coordinate axis
    *
    *   Parameters
    *   ----------
    *   sampleAngles .... angular positions of the sample goniometer (Npoints,Ns) (in)
    *   detectorAngles .. angular positions of the detector goniometer (Npoints,Nd) (in)
    *   rcch ............ direction + distance of center pixel (angles zero) (in)
    *   sampleAxis ...... string with sample axis directions (in)
    *   detectorAxis .... string with detector axis directions (in)
    *   kappadir ...... rotation axis of a possible kappa circle (in)
    *   cch1 ............ center channel of the detector (in)
    *   cch2 ............ center channel of the detector (in)
    *   dpixel1 ......... width of one pixel in first direction, same unit as distance rcch (in)
    *   dpixel2 ......... width of one pixel in second direction, same unit as distance rcch (in)
    *   roi ............. region of interest for the area detector [dir1min,dir1max,dir2min,dir2max]
    *   dir1 ............ first direction of the detector, e.g.: "x+" (in)
    *   dir2 ............ second direction of the detector, e.g.: "z+" (in)
    *   tiltazimuth ..... azimuth of the tilt (in)
    *   tilt ............ tilt of the detector plane (rotation around axis normal to the direction
    *                     given by the tiltazimuth (in)
    *   UB .............. orientation matrix and reciprocal space conversion of investigated crystal (9) (in)
    *   lambda .......... wavelength of the used x-rays (in)
    *
    *   Returns
    *   -------
    *   qpos ............ momentum transfer (Npoints*Npix1*Npix2,3) (out)
    *   */
{
    double mtemp[9],mtemp2[9], ms[9], md[9]; //matrices
    double rd[3],rpixel1[3],rpixel2[3],rcchp[3]; // detector position
    double r_i[3],rtemp[3],rtemp2[3]; //r_i: center channel direction
    int i,j,j1,j2,k; // loop indices
    int idxh1,idxh2; // temporary index helper
    int Ns,Nd; // number of sample and detector circles
    int Npoints; // number of angular positions
    double f,lambda,cch1,cch2,dpixel1,dpixel2,tilt,tiltazimuth; // x-ray wavelength, f=M_2PI/lambda and detector parameters
    char *sampleAxis,*detectorAxis,*dir1,*dir2; // string with sample and detector axis, and detector direction
    double *sampleAngles,*detectorAngles, *rcch, *kappadir, *UB, *qpos; // c-arrays for further usage
    int *roi; //  region of interest integer array
    
    PyArrayObject *sampleAnglesArr=NULL, *detectorAnglesArr=NULL, *rcchArr=NULL,
                  *kappadirArr=NULL, *roiArr=NULL, *UBArr=NULL, *qposArr=NULL; // numpy arrays

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!O!O!ssO!ddddO!ssddO!d",&PyArray_Type, &sampleAnglesArr, 
                                      &PyArray_Type, &detectorAnglesArr,
                                      &PyArray_Type, &rcchArr,
                                      &sampleAxis, &detectorAxis,
                                      &PyArray_Type, &kappadirArr,
                                      &cch1, &cch2, &dpixel1, &dpixel2, 
                                      &PyArray_Type, &roiArr, 
                                      &dir1, &dir2, &tiltazimuth, &tilt,
                                      &PyArray_Type, &UBArr,
                                      &lambda)) { 
        return NULL;
    }
    
    // check Python array dimensions and types
    if (PyArray_NDIM(sampleAnglesArr) != 2 || PyArray_TYPE(sampleAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Npoints = PyArray_DIMS(sampleAnglesArr)[0];
    Ns = PyArray_DIMS(sampleAnglesArr)[1];
    sampleAngles = (double *) PyArray_DATA(sampleAnglesArr);
    
    if (PyArray_NDIM(detectorAnglesArr) != 2 || PyArray_TYPE(detectorAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Nd = PyArray_DIMS(detectorAnglesArr)[1];
    detectorAngles = (double *) PyArray_DATA(detectorAnglesArr);

    if (PyArray_NDIM(rcchArr) != 1 || PyArray_TYPE(rcchArr) != NPY_DOUBLE || PyArray_DIMS(rcchArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    rcch = (double *) PyArray_DATA(rcchArr);
    
    if (PyArray_NDIM(kappadirArr) != 1 || PyArray_TYPE(kappadirArr) != NPY_DOUBLE || PyArray_DIMS(kappadirArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    kappadir = (double *) PyArray_DATA(kappadirArr);

    if (PyArray_NDIM(UBArr) != 2 || PyArray_TYPE(UBArr) != NPY_DOUBLE || PyArray_DIMS(UBArr)[0] != 3 || PyArray_DIMS(UBArr)[1] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double and shape (3,3)");
        return NULL; }
    UB = (double *) PyArray_DATA(UBArr);

    if (PyArray_NDIM(roiArr) != 1 || PyArray_TYPE(roiArr) != NPY_INT || PyArray_DIMS(roiArr)[0] != 4) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type int and size 2");
        return NULL; }
    roi = (int *) PyArray_DATA(roiArr);
    
    // derived values from input parameters
    f=M_2PI/lambda;
    // calculate some index shortcuts
    idxh1 = (roi[1]-roi[0])*(roi[3]-roi[2]);
    idxh2 = roi[3]-roi[2];
    
    // create output ndarray
    npy_intp nout[2];
    nout[0] = Npoints*idxh1;
    nout[1] = 3;
    qposArr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    qpos = (double *) PyArray_DATA(qposArr); 

    
    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    // arrays with function pointers to rotation matrix functions
    fp_rot sampleRotation[Ns];
    fp_rot detectorRotation[Nd];

    //printf("general conversion ang2q (area detector)\n");
    // determine axes directions
    if(determine_axes_directions(sampleRotation,sampleAxis,Ns) != 0) {
        PyErr_SetString(PyExc_ValueError,"sampleRotation axis could not be identified");
        return NULL;
    }
    if(determine_axes_directions(detectorRotation,detectorAxis,Nd) != 0) {
        PyErr_SetString(PyExc_ValueError,"detectorRotation axis could not be identified");
        return NULL;
    }

    veccopy(r_i,rcch);
    normalize(r_i);

    // determine detector pixel vector
    if(determine_detector_pixel(rpixel1, dir1, dpixel1, r_i, 0.) != 0) {
        PyErr_SetString(PyExc_ValueError,"first detector direction could not be identified");
        return NULL;
    }
    if(determine_detector_pixel(rpixel2, dir2, dpixel2, r_i, 0.) != 0) {
        PyErr_SetString(PyExc_ValueError,"second detector direction could not be identified");
        return NULL;
    }

    /*print_vector(rpixel1);
    print_vector(rpixel2);*/
    // rotate detector pixel vectors according to tilt
    veccopy(rtemp,rpixel1);
    normalize(rtemp);
    vecmul(rtemp,cos(tiltazimuth+M_PI/2.));

    veccopy(rtemp2,rpixel2);
    normalize(rtemp2);
    vecmul(rtemp2,sin(tiltazimuth+M_PI/2.));

    sumvec(rtemp,rtemp2); // tiltaxis (rotation axis) now stored in rtemp

    rotation_arb(tilt,rtemp,mtemp); // rotation matrix now in mtemp

    // rotate detector pixel directions
    veccopy(rtemp,rpixel1);
    matvec(mtemp,rtemp,rpixel1);
    veccopy(rtemp,rpixel2);
    matvec(mtemp,rtemp,rpixel2);

    /*print_vector(rpixel1);
    print_vector(rpixel2);*/

    // calculate center channel position in detector plane
    for(int k=0; k<3; ++k)
        rcchp[k] = rpixel1[k]*cch1 + rpixel2[k]*cch2;

    // calculate rotation matices and perform rotations
    #pragma omp parallel for default(shared) \
            private(i,j,j1,j2,k,mtemp,mtemp2,ms,md,rd,rtemp) \
            schedule(static)
    for(i=0; i<Npoints; ++i) {
        // determine sample rotations
        ident(mtemp);
        for(j=0; j<Ns; ++j) {
            // load kappa direction into matrix (just needed for kappa goniometer)
            mtemp2[0] = kappadir[0]; mtemp2[1] = kappadir[1]; mtemp2[2] = kappadir[2];
            sampleRotation[j](sampleAngles[Ns*i+j],mtemp2);
            matmul(mtemp,mtemp2);
        }
        // apply rotation of orientation matrix
        matmul(mtemp,UB);
        // determine inverse matrix
        inversemat(mtemp,ms);

        // determine detector rotations
        ident(md);
        for (j=0; j<Nd; ++j) {
            detectorRotation[j](detectorAngles[Nd*i+j],mtemp);
            matmul(md,mtemp);
        }

        // ms contains now the inverse rotation matrix for the sample circles
        // md contains the detector rotation matrix
        // calculate the momentum transfer for each detector pixel
        for (j1=roi[0]; j1<roi[1]; ++j1) {
            for (j2=roi[2]; j2<roi[3]; ++j2) {
                for (k=0; k<3; ++k)
                    rd[k] = j1*rpixel1[k] + j2*rpixel2[k] - rcchp[k];
                sumvec(rd,rcch);
                normalize(rd);
                // rd contains detector pixel direction, r_i contains primary beam direction
                matvec(md,rd,rtemp);
                diffvec(rtemp,r_i);
                vecmul(rtemp,f);
                // determine momentum transfer
                matvec(ms, rtemp, &qpos[3*(i*idxh1+idxh2*(j1-roi[0])+(j2-roi[2]))]);
                //print_matrix(ms);
                //print_vector(rtemp);
                //print_vector(&qpos[3*(i*idxh1+idxh2*(j1-roi[0])+(j2-roi[2]))]);
            }
        }
    }

    // return output array
    return PyArray_Return(qposArr);
}

PyObject* ang2q_conversion_area_pixel(PyObject *self, PyObject *args)
//int ang2q_conversion_area_pixel(double *detectorAngles, double *qpos, double *n1, double *n2, double *rcch, int Nd, int Npoints, char *detectorAxis, double cch1, double cch2, double dpixel1, double dpixel2, char *dir1, char *dir2, double tiltazimuth, double tilt, double lambda)
   /* conversion of Npoints of detector positions to Q
    * for a area detector with a given pixel size mounted along one of
    * the coordinate axis. This function only calculates the q-position for the
    * pairs of pixel numbers (n1,n2) given in the input and should therefore be
    * used only for detector calibration purposes.
    *
    *  Parameters
    *  ----------
    *   detectorAngles .. angular positions of the detector goniometer (Npoints,Nd) (in)
    *   n1 .............. detector pixel numbers dim1 (Npoints) (in)
    *   n2 .............. detector pixel numbers dim2 (Npoints) (in)
    *   rcch ............ direction + distance of center pixel (angles zero) (in)
    *   detectorAxis .... string with detector axis directions (in)
    *   cch1 ............ center channel of the detector (in)
    *   cch2 ............ center channel of the detector (in)
    *   dpixel1 ......... width of one pixel in first direction, same unit as distance rcch (in)
    *   dpixel2 ......... width of one pixel in second direction, same unit as distance rcch (in)
    *   dir1 ............ first direction of the detector, e.g.: "x+" (in)
    *   dir2 ............ second direction of the detector, e.g.: "z+" (in)
    *   tiltazimuth ..... azimuth of the tilt (in)
    *   tilt ............ tilt of the detector plane (rotation around axis normal to the direction
    *                     given by the tiltazimuth (in)
    *   lambda .......... wavelength of the used x-rays (in)
    *
    *   Returns
    *   -------
    *   qpos ............ momentum transfer (Npoints,3) (out)
    *   */
{
    double mtemp[9], md[9]; //matrices
    double rd[3],rpixel1[3],rpixel2[3],rcchp[3]; // detector position
    double r_i[3],rtemp[3],rtemp2[3]; //r_i: center channel direction
    int i,j,k; // loop indices
    int Nd; // number of detector circles
    int Npoints; // number of angular positions
    double f,lambda,cch1,cch2,dpixel1,dpixel2,tilt,tiltazimuth; // x-ray wavelength, f=M_2PI/lambda and detector parameters
    char *sampleAxis,*detectorAxis,*dir1,*dir2; // string with sample and detector axis, and detector direction
    double *detectorAngles, *n1, *n2, *rcch, *qpos; // c-arrays for further usage
    
    PyArrayObject *detectorAnglesArr=NULL, *n1Arr=NULL, *n2Arr=NULL, *rcchArr=NULL, *qposArr=NULL; // numpy arrays

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!O!O!O!sddddssddd", &PyArray_Type, &detectorAnglesArr,
                                      &PyArray_Type, &n1Arr, &PyArray_Type, &n2Arr, &PyArray_Type, &rcchArr,
                                      &detectorAxis, &cch1, &cch2, &dpixel1, &dpixel2, 
                                      &dir1, &dir2, &tiltazimuth, &tilt,
                                      &lambda)) { 
        return NULL;
    }
    
    // check Python array dimensions and types 
    if (PyArray_NDIM(detectorAnglesArr) != 2 || PyArray_TYPE(detectorAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Npoints = PyArray_DIMS(detectorAnglesArr)[0];
    Nd = PyArray_DIMS(detectorAnglesArr)[1];
    detectorAngles = (double *) PyArray_DATA(detectorAnglesArr);
    
    if (PyArray_NDIM(n1Arr) != 1 || PyArray_TYPE(n1Arr) != NPY_DOUBLE || PyArray_DIMS(n1Arr)[0] != Npoints) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length Npoints");
        return NULL; }
    n1 = (double *) PyArray_DATA(n1Arr); 
    
    if (PyArray_NDIM(n2Arr) != 1 || PyArray_TYPE(n2Arr) != NPY_DOUBLE || PyArray_DIMS(n2Arr)[0] != Npoints) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length Npoints");
        return NULL; }
    n2 = (double *) PyArray_DATA(n2Arr); 

    if (PyArray_NDIM(rcchArr) != 1 || PyArray_TYPE(rcchArr) != NPY_DOUBLE || PyArray_DIMS(rcchArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    rcch = (double *) PyArray_DATA(rcchArr); 
    
    // derived values from input parameters
    f=M_2PI/lambda;
    
    // create output ndarray
    npy_intp nout[2];
    nout[0] = Npoints;
    nout[1] = 3;
    qposArr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    qpos = (double *) PyArray_DATA(qposArr); 

    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    // arrays with function pointers to rotation matrix functions
    fp_rot detectorRotation[Nd];

    //printf("general conversion ang2q (area detector)\n");
    // determine axes directions
    if(determine_axes_directions(detectorRotation,detectorAxis,Nd) != 0) {
        PyErr_SetString(PyExc_ValueError,"detectorRotation axis could not be identified");
        return NULL;
    }

    veccopy(r_i,rcch);
    normalize(r_i);

    // determine detector pixel vector
    if(determine_detector_pixel(rpixel1, dir1, dpixel1, r_i, 0.) != 0) {
        PyErr_SetString(PyExc_ValueError,"first detector direction could not be identified");
        return NULL;
    }
    if(determine_detector_pixel(rpixel2, dir2, dpixel2, r_i, 0.) != 0) {
        PyErr_SetString(PyExc_ValueError,"second detector direction could not be identified");
        return NULL;
    }

    // rotate detector pixel vectors according to tilt
    veccopy(rtemp,rpixel1);
    normalize(rtemp);
    vecmul(rtemp,cos(tiltazimuth+M_PI/2.));

    veccopy(rtemp2,rpixel2);
    normalize(rtemp2);
    vecmul(rtemp2,sin(tiltazimuth+M_PI/2.));

    sumvec(rtemp,rtemp2); // tiltaxis (rotation axis) now stored in rtemp

    rotation_arb(tilt,rtemp,mtemp); // rotation matrix now in mtemp

    // rotate detector pixel directions
    veccopy(rtemp,rpixel1);
    matvec(mtemp,rtemp,rpixel1);
    veccopy(rtemp,rpixel2);
    matvec(mtemp,rtemp,rpixel2);

    // calculate center channel position in detector plane
    for(int k=0; k<3; ++k)
        rcchp[k] = rpixel1[k]*cch1 + rpixel2[k]*cch2;

    // calculate rotation matices and perform rotations
    #pragma omp parallel for default(shared) \
            private(i,j,k,mtemp,md,rd,rtemp) \
            schedule(static)
    for(i=0; i<Npoints; ++i) {
        // determine detector rotations
        ident(md);
        for (j=0; j<Nd; ++j) {
            detectorRotation[j](detectorAngles[Nd*i+j],mtemp);
            matmul(md,mtemp);
        }
        // md contains the detector rotation matrix
        // calculate momentum transfer for the detector pixel n1[i],n2[i]
        for (k=0; k<3; ++k)
            rd[k] = n1[i]*rpixel1[k] + n2[i]*rpixel2[k] - rcchp[k];
        sumvec(rd,rcch);
        normalize(rd);
        // rd contains detector pixel direction, r_i contains primary beam direction
        matvec(md,rd,rtemp);
        diffvec(rtemp,r_i);
        vecmul(rtemp,f);
        // save momentum transfer to output
        veccopy(&qpos[3*i], rtemp);
    }

    // return output array
    return PyArray_Return(qposArr);
}

PyObject* ang2q_conversion_area_pixel2(PyObject *self, PyObject *args)
//int ang2q_conversion_area_pixel2(double *sampleAngles, double *detectorAngles, double *qpos, double *n1, double *n2, double *rcch, int Ns, int Nd, int Npoints, char *sampleAxis, char *detectorAxis, double *kappadir, double cch1, double cch2, double dpixel1, double dpixel2, char *dir1, char *dir2, double tiltazimuth, double tilt, double *UB, double lambda)
   /* conversion of Npoints of detector positions to Q
    * for a area detector with a given pixel size mounted along one of
    * the coordinate axis. This function only calculates the q-position for the
    * pairs of pixel numbers (n1,n2) given in the input and should therefore be
    * used only for detector calibration purposes.
    *
    * This variant of this function also takes a sample orientation matrix as well as the sample goniometer
    * as input to allow for a simultaneous fit of reference samples orientation
    *
    * Interface:
    *   sampleAngles .... angular positions of the sample goniometer (Npoints,Ns) (in)
    *   detectorAngles .. angular positions of the detector goniometer (Npoints,Nd) (in)
    *   n1 .............. detector pixel numbers dim1 (Npoints) (in)
    *   n2 .............. detector pixel numbers dim2 (Npoints) (in)
    *   rcch ............ direction + distance of center pixel (angles zero) (in)
    *   sampleAxis ...... string with sample axis directions (in)
    *   detectorAxis .... string with detector axis directions (in)
    *   cch1 ............ center channel of the detector (in)
    *   cch2 ............ center channel of the detector (in)
    *   dpixel1 ......... width of one pixel in first direction, same unit as distance rcch (in)
    *   dpixel2 ......... width of one pixel in second direction, same unit as distance rcch (in)
    *   dir1 ............ first direction of the detector, e.g.: "x+" (in)
    *   dir2 ............ second direction of the detector, e.g.: "z+" (in)
    *   tiltazimuth ..... azimuth of the tilt (in)
    *   tilt ............ tilt of the detector plane (rotation around axis normal to the direction
    *                     given by the tiltazimuth (in)
    *   UB .............. orientation matrix and reciprocal space conversion of investigated crystal (9) (in)
    *   lambda .......... wavelength of the used x-rays (in)
    *
    *   Returns
    *   -------
    *   qpos ............ momentum transfer (Npoints,3) (out)
    *   */
{
    double mtemp[9],mtemp2[9], ms[9], md[9]; //matrices
    double rd[3],rpixel1[3],rpixel2[3],rcchp[3]; // detector position
    double r_i[3],rtemp[3],rtemp2[3]; //r_i: center channel direction
    int i,j,j1,j2,k; // loop indices
    int Ns,Nd; // number of sample/detector circles
    int Npoints; // number of angular positions
    double f,lambda,cch1,cch2,dpixel1,dpixel2,tilt,tiltazimuth; // x-ray wavelength, f=M_2PI/lambda and detector parameters
    char *sampleAxis,*detectorAxis,*dir1,*dir2; // string with sample and detector axis, and detector direction
    double *sampleAngles, *detectorAngles, *n1, *n2, *rcch, *UB, *qpos; // c-arrays for further usage

    PyArrayObject *sampleAnglesArr=NULL, *detectorAnglesArr=NULL, *n1Arr=NULL, *n2Arr=NULL, *rcchArr=NULL, *UBArr=NULL, *qposArr=NULL; // numpy arrays

    printf("test");
    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ssddddssddO!d", &PyArray_Type, &sampleAnglesArr, &PyArray_Type, &detectorAnglesArr,
                                      &PyArray_Type, &n1Arr, &PyArray_Type, &n2Arr, &PyArray_Type, &rcchArr,
                                      &sampleAxis, &detectorAxis, &cch1, &cch2, &dpixel1, &dpixel2, 
                                      &dir1, &dir2, &tiltazimuth, &tilt, &PyArray_Type, &UB,
                                      &lambda)) { 
        return NULL;
    }

    // check Python array dimensions and types 
    if (PyArray_NDIM(sampleAnglesArr) != 2 || PyArray_TYPE(sampleAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Npoints = PyArray_DIMS(sampleAnglesArr)[0];
    Ns = PyArray_DIMS(sampleAnglesArr)[1];
    sampleAngles = (double *) PyArray_DATA(sampleAnglesArr);

    if (PyArray_NDIM(detectorAnglesArr) != 2 || PyArray_TYPE(detectorAnglesArr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double");
        return NULL; }
    Nd = PyArray_DIMS(detectorAnglesArr)[1];
    detectorAngles = (double *) PyArray_DATA(detectorAnglesArr);
    
    if (PyArray_NDIM(n1Arr) != 1 || PyArray_TYPE(n1Arr) != NPY_DOUBLE || PyArray_DIMS(n1Arr)[0] != Npoints) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length Npoints");
        return NULL; }
    n1 = (double *) PyArray_DATA(n1Arr); 
    
    if (PyArray_NDIM(n2Arr) != 1 || PyArray_TYPE(n2Arr) != NPY_DOUBLE || PyArray_DIMS(n2Arr)[0] != Npoints) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length Npoints");
        return NULL; }
    n2 = (double *) PyArray_DATA(n2Arr); 

    if (PyArray_NDIM(rcchArr) != 1 || PyArray_TYPE(rcchArr) != NPY_DOUBLE || PyArray_DIMS(rcchArr)[0] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double and length 3");
        return NULL; }
    rcch = (double *) PyArray_DATA(rcchArr); 
    
    if (PyArray_NDIM(UBArr) != 2 || PyArray_TYPE(UBArr) != NPY_DOUBLE || PyArray_DIMS(UBArr)[0] != 3 || PyArray_DIMS(UBArr)[1] != 3) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type double and shape (3,3)");
        return NULL; }
    UB = (double *) PyArray_DATA(UBArr);

    // derived values from input parameters
    f=M_2PI/lambda;

    // create output ndarray
    npy_intp nout[2];
    nout[0] = Npoints;
    nout[1] = 3;
    qposArr = (PyArrayObject *) PyArray_SimpleNew(2, nout, NPY_DOUBLE);
    qpos = (double *) PyArray_DATA(qposArr); 

    #ifdef __OPENMP__
    //set openmp thread numbers dynamically
    omp_set_dynamic(1);
    #endif

    // arrays with function pointers to rotation matrix functions
    fp_rot sampleRotation[Ns];
    fp_rot detectorRotation[Nd];

    //printf("general conversion ang2q (area detector)\n");
    // determine axes directions
    if(determine_axes_directions(sampleRotation,sampleAxis,Ns) != 0) {
        PyErr_SetString(PyExc_ValueError,"sampleRotation axis could not be identified");
        return NULL;
    }
    if(determine_axes_directions(detectorRotation,detectorAxis,Nd) != 0) {
        PyErr_SetString(PyExc_ValueError,"detectorRotation axis could not be identified");
        return NULL;
    }

    veccopy(r_i,rcch);
    normalize(r_i);

    // determine detector pixel vector
    if(determine_detector_pixel(rpixel1, dir1, dpixel1, r_i, 0.) != 0) {
        PyErr_SetString(PyExc_ValueError,"first detector direction could not be identified");
        return NULL;
    }
    if(determine_detector_pixel(rpixel2, dir2, dpixel2, r_i, 0.) != 0) {
        PyErr_SetString(PyExc_ValueError,"second detector direction could not be identified");
        return NULL;
    }

    // rotate detector pixel vectors according to tilt
    veccopy(rtemp,rpixel1);
    normalize(rtemp);
    vecmul(rtemp,cos(tiltazimuth+M_PI/2.));

    veccopy(rtemp2,rpixel2);
    normalize(rtemp2);
    vecmul(rtemp2,sin(tiltazimuth+M_PI/2.));

    sumvec(rtemp,rtemp2); // tiltaxis (rotation axis) now stored in rtemp

    rotation_arb(tilt,rtemp,mtemp); // rotation matrix now in mtemp

    // rotate detector pixel directions
    veccopy(rtemp,rpixel1);
    matvec(mtemp,rtemp,rpixel1);
    veccopy(rtemp,rpixel2);
    matvec(mtemp,rtemp,rpixel2);

    // calculate center channel position in detector plane
    for(int k=0; k<3; ++k)
        rcchp[k] = rpixel1[k]*cch1 + rpixel2[k]*cch2;

    // calculate rotation matices and perform rotations
    #pragma omp parallel for default(shared) \
            private(i,j,k,mtemp,mtemp2,ms,md,rd,rtemp) \
            schedule(static)
    for(i=0; i<Npoints; ++i) {
        // determine sample rotations
        ident(mtemp);
        for(j=0; j<Ns; ++j) {
            sampleRotation[j](sampleAngles[Ns*i+j],mtemp2);
            matmul(mtemp,mtemp2);
        }
        // apply rotation of orientation matrix
        matmul(mtemp,UB);
        // determine inverse matrix
        inversemat(mtemp,ms);

        // determine detector rotations
        ident(md);
        for (j=0; j<Nd; ++j) {
            detectorRotation[j](detectorAngles[Nd*i+j],mtemp);
            matmul(md,mtemp);
        }

        // ms contains now the inverse rotation matrix for the sample circles
        // md contains the detector rotation matrix
        // calculate the momentum transfer for a certain detector pixel
        for (k=0; k<3; ++k)
            rd[k] = n1[i]*rpixel1[k] + n2[i]*rpixel2[k] - rcchp[k];
        sumvec(rd,rcch);
        normalize(rd);
        // rd contains detector pixel direction, r_i contains primary beam direction
        matvec(md,rd,rtemp);
        diffvec(rtemp,r_i);
        vecmul(rtemp,f);
        // determine momentum transfer
        matvec(ms, rtemp, &qpos[3*i]);
    }

    // return output array
    return PyArray_Return(qposArr);
}

