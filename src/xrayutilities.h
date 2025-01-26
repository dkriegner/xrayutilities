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
 * Copyright (C) 2014 Eugen Wintersberger <eugen.wintersberger@gmail.com>
*/
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "array_utils.h"

/*****************************************************************************
 * Windows build related macros
 ****************************************************************************/
/* 'extern inline' seems to work only on newer version of gcc (>4.6 tested)
 * gcc 4.1 seems to need this empty, i am not sure if there is a speed gain
 * by inlining since the calls to those functions are anyhow built dynamically
 * for compatibility keep this empty unless you can test with several compilers
 */
#define INLINE
#ifdef _WIN32
#define RESTRICT
#else
#define RESTRICT restrict
#endif

#ifdef _WIN32
#define strtok_r strtok_s
#endif

/*****************************************************************************
 * general purpose macros
 ****************************************************************************/
/*
 * if M_PI is not set we do this here
 */
#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif
#define M_2PI (2 * M_PI)


/*****************************************************************************
 * OpenMP related macros
 ****************************************************************************/
/*
 * include OpenMP header is required
 */
#ifdef __OPENMP__
#include <omp.h>
#endif

#define OMPSETNUMTHREADS(nth) \
    if (nth == 0) omp_set_num_threads(omp_get_max_threads());\
    else omp_set_num_threads(nth);
