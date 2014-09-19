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
 * Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <stdio.h>

PyObject* cbfread(PyObject *self, PyObject *args) {
    /* parser for cbf data arrays from Pilatus detector images
     *
     * Parameters
     * ----------
     *  data:   data stream (character array)
     *  nx,ny:  number of entries of the two dimensional image
     *
     * Returns
     * -------
     *  the parsed data values as float ndarray
     */

    unsigned int i,start=0,nx,ny,len;
    unsigned int parsed = 0;
    PyArrayObject *outarr=NULL;
    unsigned char *cin;
    float *cout;
    npy_intp nout;

    int cur  = 0;
    int diff = 0;
    unsigned int np = 0;

    union {
        const unsigned char*  uint8;
        const unsigned short* uint16;
        const unsigned int*   uint32;
        const          char*   int8;
        const          short*  int16;
        const          int*    int32;
    } parser;

    // Python argument conversion code
    if (!PyArg_ParseTuple(args, "s#ii", &cin, &len, &nx, &ny)) return NULL;
    /*printf("stream length: %d\n",len);
    printf("entries: %d %d\n",nx,ny);*/

    // create output ndarray
    nout = nx*ny;
    outarr = (PyArrayObject *) PyArray_SimpleNew(1, &nout, NPY_FLOAT);
    cout = (float *) PyArray_DATA(outarr);

    i = 0;
    while (i<len-10) {   // find the start of the array
        if ((cin[i]==0x0c)&&(cin[i+1]==0x1a)&&(cin[i+2]==0x04)&&(cin[i+3]==0xd5)) {
            start = i+4;
            i = len+10;
        }
        i++;
    }
    if(i==len-10) {
        PyErr_SetString(PyExc_ValueError,"start of data in stream not found!\n");
        return NULL;
    }
    /*else {
        printf("found start at %d\n",start);
    }*/

    // next while part was taken from pilatus code and adapted by O. Seeck and D. Kriegner
    parser.uint8 = (const unsigned char*) cin+start;

    while (parsed<(len-start)) {
        //printf("%d ",parsed);
        if (*parser.uint8 != 0x80) {	
	        diff = (int) *parser.int8;
	        parser.int8++;
	        parsed += 1;
	    }
        else {
	        parser.uint8++;
	        parsed += 1;
            if (*parser.uint16 != 0x8000) {
	            diff = (int) *parser.int16;
	            parser.int16++;
	            parsed += 2;
            }
            else {
	            parser.uint16++;
	            parsed += 2;
                diff = (int) *parser.int32;
	            parser.int32++;
	            parsed += 4;
	        }
	    }
	    cur += diff;
	    *cout++ = (float) cur;
        np++;
        // check if we already have all data (file might be longer)
        if(np==nout) {
            //printf("all data read (%d,%d)\n",np,parsed);
            break;
        }
    }

    // return output array
    return PyArray_Return(outarr);
}

#undef PY_ARRAY_UNIQUE_SYMBOL
