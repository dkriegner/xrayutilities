# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012 Dominik Kriegner <dominik.kriegner@gmail.com>
"""
module with a function wrapper to scipy.optimize.leastsq
for fitting of a 2D function to a peak
"""

from __future__ import print_function
import numpy
import scipy.optimize as optimize
import time

from .. import config


def fit_peak2d(x,y,data,start,drange,fit_function,maxfev=2000):
    """
    fit a two dimensional function to a two dimensional data set
    e.g. a reciprocal space map

    Parameters
    ----------
     x,y:     data coordinates (do NOT need to be regularly spaced)
     data:    data set used for fitting (e.g. intensity at the data coords)
     start:   set of starting parameters for the fit
              used as first parameter of function fit_function
     drange:  limits for the data ranges used in the fitting algorithm
              e.g. it is clever to use only a small region around the peak which 
              should be fitted: [xmin,xmax,ymin,ymax]
     fit_function:  function which should be fitted 
                    must accept the parameters (x,y,params)

    Returns
    -------
     (fitparam,cov)   the set of fitted parameters and covariance matrix
    """
    s = time.time()
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.math.fit: Fitting started... ",end='')

    start = numpy.array(start)
    lx = x.flatten()
    ly = y.flatten()
    mask = (lx>drange[0])*(lx<drange[1])*(ly>drange[2])*(ly<drange[3])
    ly = ly[mask]
    lx = lx[mask]
    ldata = data.flatten()[mask]
    errfunc = lambda p,x,z,data: fit_function(x,z,p) - data
    p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, start, args=(lx,ly,ldata), full_output=1,maxfev=maxfev)

    s = time.time() - s
    if config.VERBOSITY >= config.INFO_ALL:
        print("finished in %8.2f sec, (data length used %d)"%(s,ldata.size))
        print("XU.math.fit: %s"%errmsg)
        
    # calculate correct variance covariance matrix
    if cov != None:
        s_sq = (errfunc(p,lx,ly,ldata)**2).sum()/(len(ldata)-len(start))
        pcov = cov * s_sq
    else: pcov = numpy.zeros((len(start),len(start)))

    if success not in [1,2,3,4]:
        print("XU.math.fit: Could not obtain fit!")
    return p,pcov
