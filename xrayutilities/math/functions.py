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
module with several common function needed in xray data analysis
"""

import numpy
import scipy.integrate

from .. import config

def smooth(x,n):
    """
    function to smooth an array of data by averaging N adjacent data points

    Parameters
    ----------
     x:  1D data array
     n:  number of data points to average

    Returns
    -------
     xsmooth:  smoothed array with same length as x
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < n:
        raise ValueError("Input vector needs to be bigger than n.")
    if n<2:
        return x
    # avoid boundary effects by adding mirrored signal at the boundaries
    s=numpy.r_[x[n-1:0:-1],x,x[-1:-n-1:-1]]
    w=numpy.ones(n,'d')
    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[n:-n+1]


def kill_spike(data,threshold=2.):
    """
    function to smooth **single** data points which differ from the average of the neighboring data 
    points by more than the threshold factor. Such spikes will be replaced by the mean value of the 
    next neighbors.

    .. warning:: Use this function carefully not to manipulate your data!

    Parameters
    ----------
     data:          1d numpy array with experimental data
     threshold:     threshold factor to identify strange data points

    Returns
    -------
     1d data-array with spikes removed 
    """

    dataout = data.copy()

    mean = (data[:-2] + data[2:])/2.
    mask = numpy.logical_or( numpy.abs(data[1:-1]*threshold) < numpy.abs(mean) , numpy.abs(data[1:-1]/threshold) > numpy.abs(mean) )
    #ensure that only single value are corrected and neighboring are ignored
    for i in range(1,len(mask)-1):
        if mask[i-1] and mask[i] and mask[i+1]:
            mask[i-1]=False
            mask[i+1]=False

    dataout[1:-1][mask] = (numpy.abs(mean))[mask]

    return dataout


def Gauss1d(x,*p):
    """
    function to calculate a general one dimensional Gaussian

    Parameters
    ----------
     p:     list of parameters of the Gaussian
            [XCEN,SIGMA,AMP,BACKGROUND]
            for information: SIGMA = FWHM / (2*sqrt(2*log(2)))
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Gaussian described by the parameters p
    at position x
    """

    g = p[3]+p[2]*numpy.exp(-((p[0]-x)/p[1])**2/2.)
    return g


def Gauss1d_der_x(x,*p):
    """
    function to calculate the derivative of a Gaussian with respect to x

    for parameter description see Gauss1d
    """

    lp = numpy.copy(p)
    lp[3] = 0
    return 2*(p[0]-x)*Gauss1d(x,*lp)


def Gauss1d_der_p(x,*p):
    """
    function to calculate the derivative of a Gaussian with respect the
    parameters p

    for parameter description see Gauss1d
    """
    lp = numpy.copy(p)
    lp[3] = 0
    r = numpy.concatenate(( -2*(p[0]-x)*Gauss1d(x,*lp),\
                            (p[0]-x)**2/(2*p[1]**3)*Gauss1d(x,*lp),\
                            Gauss1d(x,*lp)/p[2],\
                            numpy.ones(x.shape,dtype=numpy.float) ))
    r.shape = (4,) + x.shape

    return r


def Gauss2d(x,y,*p):
    """
    function to calculate a general two dimensional Gaussian

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,SIGMAX,SIGMAY,AMP,BACKGROUND,ANGLE]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
            ANGLE = rotation of the X,Y direction of the Gaussian in radians
     x,y:   coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Gaussian described by the parameters p
    at position (x,y)
    """

    rcen_x = p[0] * numpy.cos(p[6]) - p[1] * numpy.sin(p[6])
    rcen_y = p[0] * numpy.sin(p[6]) + p[1] * numpy.cos(p[6])
    xp = x * numpy.cos(p[6]) - y * numpy.sin(p[6])
    yp = x * numpy.sin(p[6]) + y * numpy.cos(p[6])

    g = p[5]+p[4]*numpy.exp(-(((rcen_x-xp)/p[2])**2+
                                     ((rcen_y-yp)/p[3])**2)/2.)
    return g

def Gauss3d(x,y,z,*p):
    """
    function to calculate a general three dimensional Gaussian

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,ZCEN,SIGMAX,SIGMAY,SIGMAZ,AMP,BACKGROUND]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
     x,y,z:   coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Gaussian described by the parameters p
    at positions (x,y,z)
    """

    g = p[7]+p[6]*numpy.exp(-(((x-p[0])/p[3])**2+((y-p[1])/p[4])**2+((z-p[2])/p[5])**2)/2.)
    return g

def TwoGauss2d(x,y,*p):
    """
    function to calculate two general two dimensional Gaussians

    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN1,YCEN1,SIGMAX1,SIGMAY1,AMP1,ANGLE1,XCEN2,YCEN2,SIGMAX2,SIGMAY2,AMP2,ANGLE2,BACKGROUND]
            SIGMA = FWHM / (2*sqrt(2*log(2)))
            ANGLE = rotation of the X,Y direction of the Gaussian in radians
     x,y:   coordinate(s) where the function should be evaluated

    Return
    ------
    the value of the Gaussian described by the parameters p
    at position (x,y)
    """

    p = list(p)
    p1 = p[0:5] + [p[12],] + [p[6],]
    p2 = p[6:11] + [p[12],] +[p[11],]

    g = Gauss2d(x,y,*p1) + Gauss2d(x,y,*p2)

    return g

def Lorentz1d(x,*p):
    """
    function to calculate a general one dimensional Lorentzian

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND]
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Lorentian described by the parameters p
    at position (x,y)

    """

    g = p[3]+p[2]/(1+(2*(x-p[0])/p[1])**2)

    return g

def Lorentz1d_der_x(x,*p):
    """
    function to calculate the derivative of a Gaussian with respect to x

    for parameter description see Lorentz1d
    """

    return 4*(p[0]-x)* p[2]/p[1]/(1+(2*(x-p[0])/p[1])**2)**2

def Lorentz1d_der_p(x,*p):
    """
    function to calculate the derivative of a Gaussian with respect the
    parameters p

    for parameter description see Lorentz1d
    """

    r = numpy.concatenate(( 4*(x-p[0])* p[2]/p[1]/(1+(2*(x-p[0])/p[1])**2)**2,\
                            4*(p[0]-x)* p[2]/p[1]**2/(1+(2*(x-p[0])/p[1])**2)**2,\
                            1/(1+(2*(x-p[0])/p[1])**2),\
                            numpy.ones(x.shape,dtype=numpy.float) ))
    r.shape = (4,) + x.shape

    return r

def Lorentz2d(x,y,*p):
    """
    function to calculate a general two dimensional Lorentzian

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,YCEN,FWHMX,FWHMY,AMP,BACKGROUND,ANGLE]
            ANGLE = rotation of the X,Y direction of the Lorentzian in radians
     x,y:   coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the Lorentian described by the parameters p
    at position (x,y)
    """

    rcen_x = p[0] * numpy.cos(p[6]) - p[1] * numpy.sin(p[6])
    rcen_y = p[0] * numpy.sin(p[6]) + p[1] * numpy.cos(p[6])
    xp = x * numpy.cos(p[6]) - y * numpy.sin(p[6])
    yp = x * numpy.sin(p[6]) + y * numpy.cos(p[6])

    g = p[5]+p[4]/(1+(2*(rcen_x-xp)/p[2])**2+(2*(rcen_y-yp)/p[3])**2)
    return g


def PseudoVoigt1d(x,*p):
    """
    function to calculate a pseudo Voigt function as linear combination of a Gauss and Lorentz peak

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND,ETA]
            ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz 
     x:     coordinate(s) where the function should be evaluated

    Returns
    -------
    the value of the PseudoVoigt described by the parameters p
    at position (x,y)

    """

    f = p[3]+p[2] * ( p[4]*(1/(1+(2*(x-p[0])/(p[1]/2.))**2)) + (1-p[4])*numpy.exp(-numpy.log(2)*((p[0]-x)/(p[1]/2.))**2) ) 

    return f

def PseudoVoigt1dArea(*p):
    """
    function to calculate the area of a pseudo Voigt function with neglected background

    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND,ETA]
            ETA: 0 ...1  0 means pure Gauss and 1 means pure Lorentz 

    Returns
    -------
    the value of the PseudoVoigt described by the parameters p
    at position (x,y)

    """

    f = p[2] * ( p[4]*numpy.pi/(4/(p[1])) + (1-p[4])*numpy.sqrt(numpy.pi)/(numpy.log(2)*(2./(p[1]))) ) 

    return f

def Debye1(x):
    """
    function to calculate the first Debye function as needed
    for the calculation of the thermal Debye-Waller-factor
    by numerical integration

    for definition see:
    http://en.wikipedia.org/wiki/Debye_function

    D1(x) = (1/x) \int_0^x t/(exp(t)-1) dt

    Parameters
    ----------
     x ... argument of the Debye function (float)

    Returns
    -------
     D1(x)  float value of the Debye function
     """

    def __int_kernel(t):
        """
        integration kernel for the numeric integration
        """
        y = t/(numpy.exp(t)-1)
        return y

    if x>0.:
        integral = scipy.integrate.quad(__int_kernel, 0, x)
        d1 = (1/float(x)) * integral[0]
    else:
        integral = (0,0)
        d1 = 1.

    if (config.VERBOSITY >= config.DEBUG):
        print("XU.math.Debye1: Debye integral value/error estimate: %g %g"%(integral[0],integral[1]))

    return d1
