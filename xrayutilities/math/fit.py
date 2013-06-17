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
for fitting of a 2D function to a peak or a 1D Gauss fit with
the odr package
"""

from __future__ import print_function
import numpy
import scipy.optimize as optimize
import time
from scipy.odr import odrpack as odr
from scipy.odr import models

from .. import config
from .functions import Gauss1d,Gauss1d_der_x,Gauss1d_der_p

def gauss_fit(xdata,ydata,iparams=[],maxit=200):
    """
    Gauss fit function using odr-pack wrapper in scipy similar to
    https://github.com/tiagopereira/python_tips/wiki/Scipy%3A-curve-fitting

    Parameters
    ----------
     xdata:     xcoordinates of the data to be fitted
     ydata:     ycoordinates of the data which should be fit

    keyword parameters:
     iparams:   initial paramters for the fit (determined automatically if nothing is given
     maxit:     maximal iteration number of the fit
    
    Returns
    -------
     params,sd_params,itlim

    the Gauss parameters as defined in function Gauss1d(x, *param)
    and their errors of the fit, as well as a boolean flag which is false in the case of a 
    successful fit
    """

    gfunc = lambda param,x: Gauss1d(x, *param)
    gfunc_dx = lambda param,x: Gauss1d_der_x(x, *param)
    gfunc_dp = lambda param,x: Gauss1d_der_p(x, *param)

    if not any(iparams):
        cen = numpy.sum(xdata*ydata)/numpy.sum(ydata)
        iparams = numpy.array([cen,\
            numpy.sqrt(numpy.abs(numpy.sum((xdata-cen)**2*ydata)/numpy.sum(ydata))),\
            numpy.max(ydata),\
            numpy.min(ydata)])

    if config.VERBOSITY >= config.DEBUG:
        print("XU.math.gauss_fit: iparams: [%f %f %f %f]" %tuple(iparams))

    gauss  = odr.Model(gfunc, fjacd=gfunc_dx, fjacb=gfunc_dp)

    sy = numpy.sqrt(ydata)
    sy[sy==0] = 1
    mydata = odr.RealData(xdata, ydata, sy=sy)

    myodr  = odr.ODR(mydata, gauss, beta0=iparams,maxit=maxit)

    # use least-square fit
    myodr.set_job(fit_type=2)

#    # DK comment out because this command triggers a synthax error with new scipy version 2013/5/7
#    if config.VERBOSITY >= config.DEBUG:
#        myodr.set_iprint(final=1)

    fit = myodr.run()

    #fit.pprint() # prints final message from odrpack

    if config.VERBOSITY >= config.DEBUG:
        print("XU.math.gauss_fit: params: [%f %f %f %f]" %tuple(fit.beta))
        print("XU.math.gauss_fit: params std: [%f %f %f %f]" %tuple(fit.sd_beta))
        print("XU.math.gauss_fit: %s" %fit.stopreason[0])

    itlim = False
    if fit.stopreason[0] == 'Iteration limit reached':
        itlim = True
        if config.VERBOSITY >= config.INFO_LOW:
            print("XU.math.gauss_fit: Iteration limit reached, do not trust the result!")

    return fit.beta, fit.sd_beta, itlim


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
                    must accept the parameters (x,y,*params)

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
    errfunc = lambda p,x,z,data: (fit_function(x,z,*p) - data)#/(numpy.abs(numpy.sqrt(data))+numpy.abs(numpy.sqrt(data[data!=0].min())))
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
