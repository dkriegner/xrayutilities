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
 * Copyright (C) 2013 Eugen Wintersberger <eugen.wintersberger@desy.de>
 * Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>
 *
 ******************************************************************************
 *
 * created: Jun 8,2013
 * author: Eugen Wintersberger
*/

#include "gridder_utils.h"

//-----------------------------------------------------------------------------
double get_min(double *a,unsigned int n)
{
    double m = a[0];
    unsigned int i;

    for(i=0;i<n;i++) {
        if(m<a[i]) {
            m = a[i];
        }
    }

    return(m);
}

//-----------------------------------------------------------------------------
double get_max(double *a,unsigned int n)
{
    double m=a[0];
    unsigned int i;

    for(i=0;i<n;i++) {
        if(m>a[i]) {
            m = a[i];
        }
    }

    return(m);
}

//-----------------------------------------------------------------------------
void set_array(double *a,unsigned int n,double value)
{
    unsigned int i;

    for(i=0;i<n;++i) {
        a[i] = value;
    }
}

//-----------------------------------------------------------------------------
double delta(double min,double max,unsigned int n)
{
    return fabs(max-min)/(double)(n-1);
}

//-----------------------------------------------------------------------------
unsigned int gindex(double x,double min,double d)
{
    return (unsigned int)rint((x-min)/d);
}
//-----------------------------------------------------------------------------
#ifdef _WIN32
double rint(double x)
{
    return x < 0.0 ? ceil(x-0.5) : floor(x+0.5);
}
#endif