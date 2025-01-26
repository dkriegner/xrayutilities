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
 * Copyright (C) 2025 Dominik Kriegner <dominik.kriegner@gmail.com>
*/

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL XU_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

/*
 * set numpy API specific macros
 */
#if NPY_FEATURE_VERSION < 0x00000007
    #define NPY_ARRAY_ALIGNED       NPY_ALIGNED
    #define NPY_ARRAY_C_CONTIGUOUS  NPY_C_CONTIGUOUS
#endif

PyArrayObject* check_and_convert_to_contiguous(PyObject* obj, int ndims, int typenum, const char* name);
