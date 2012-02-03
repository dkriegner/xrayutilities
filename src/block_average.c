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
 * Copyright (C) 2010-2011 Dominik Kriegner <dominik.kriegner@aol.at>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef __OPENMP__
#include <omp.h>
#endif

int block_average1d(double *block_av, double *input, int Nav, int N) {
    /*    block average for one-dimensional double array
     *
     *    Parameters
     *    ----------
     *    block_av:     block averaged output array 
     *                  size = ceil(N/Nav) (out)
     *    input:        input array of double (in)
     *    Nav:          number of double to average
     *    N:            total number of input values
     */
    
    int i,j; //loop indices
    double buf;

    for(i=0; i<N; i=i+Nav) {
        buf=0;
        //perform one block average (j serves as counter -> last bin is therefore correct)
        for(j=0; j<Nav && (i+j)<N; ++j) {
            buf += input[i+j];
        }
        block_av[i/Nav] = buf/(float)j; //save average to output array       
    }

    return 1;
}

int block_average_PSD(double *intensity, double *psd, int Nav, int Nch, int Nspec) {
    /*    block average for a bunch of PSD spectra
     *
     *    Parameters
     *    ----------
     *    intensity:    block averaged output array 
     *                  size = (Nspec , ceil(Nch/Nav)) (out)
     *    psd:          input array of PSD values  
     *                  size = (Nspec, Nch) (in)
     *    Nav:          number of channels to average
     *    Nch:          number of channels per spectrum
     *    Nspec:        number of spectra
     */
    int i; //loop indices
    int Nout = (int) ceil(Nch/(float)Nav);

    #ifdef __OPENMP__
	//set openmp thread numbers dynamically
	omp_set_dynamic(1);
	#endif

    #pragma omp parallel for default(shared) private(i) schedule(static)
    for(i=0; i<Nspec; ++i) {
        block_average1d(&intensity[i*Nout], &psd[i*Nch], Nav, Nch);
    }
    
    return 1;

}

int block_average2d(double *block_av, double *ccd, int Nav2, int Nav1, int Nch2, int Nch1) {
    /*    2D block average for one CCD frame
     *
     *    Parameters
     *    ----------
     *    block_av:     block averaged output array 
     *                  size = (ceil(Nch2/Nav2) , ceil(Nch1/Nav1)) (out)
     *    ccd:          input array/CCD frame  
     *                  size = (Nch2, Nch1) (in)
     *                  Nch1 is the fast variing index
     *    Nav1,2:       number of channels to average in each dimension
     *                  in total a block of Nav1 x Nav2 is averaged
     *    Nch1,2:       number of channels of the CCD frame
     */

    int i=0,j=0,k=0,l=0; //loop indices
    double buf;
    int Nout1,Nout2;

    #ifdef __OPENMP__
	//set openmp thread numbers dynamically
	omp_set_dynamic(1);
	#endif

    Nout1 = ceil(Nch1/(float)Nav1);
    Nout2 = ceil(Nch2/(float)Nav2);

    #pragma omp parallel for default(shared) private(i,j,k,l,buf) schedule(static)
    for(i=0; i<Nch2; i=i+Nav2) {
        for(j=0; j<Nch1; j=j+Nav1) {
            buf = 0.;
            for(k=0; k<Nav2 && (i+k)<Nch2; ++k) {
                for(l=0; l<Nav1 && (j+l)<Nch1; ++l) {
                    buf += ccd[(i+k)*Nch1+(j+l)];
                }
            }
            block_av[(i/Nav2)*Nout1+j/Nav1] = buf/(float)(k*l);
        }
    }

	return 1;
}



