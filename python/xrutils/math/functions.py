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

def Gauss1d(x,*p):
    """ 
    function to calculate a general one dimensional Gaussian
    
    Parameters
    ----------
     p:     list of parameters of the Gaussian
            [XCEN,SIGMA,AMP,BACKGROUND]
            for information:
                SIGMA = FWHM / (2*sqrt(2*log(2)))
     x:     coordinate(s) where the function should be evoluated
    
    Returns
    -------
    the value of the Gaussian described by the parameters p 
    at position x
    """
    
    g = p[3]+p[2]*numpy.exp(-((p[0]-x)/p[1])**2/2.)
    return g

def Gauss2d(x,y,*p):
    """ 
    function to calculate a general two dimensional Gaussian
    
    Parameters
    ----------
     p:     list of parameters of the Gauss-function
            [XCEN,YCEN,SIGMAX,SIGMAY,AMP,BACKGROUND,ANGLE]
                SIGMA = FWHM / (2*sqrt(2*log(2)))
                ANGLE = rotation of the X,Y direction of the Gaussian    
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

def Lorentz1d(x,*p):
    """ 
    function to calculate a general one dimensional Lorentzian
    
    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,FWHM,AMP,BACKGROUND]
     x,y:   coordinate(s) where the function should be evaluated
    
    Returns
    -------
    the value of the Lorentian described by the parameters p 
    at position (x,y)
    
    """

    g = p[3]+p[2]/(1+(2*(x-p[0])/p[1])**2)

    return g

def Lorentz2d(x,y,*p):
    """ 
    function to calculate a general two dimensional Lorentzian
    
    Parameters
    ----------
     p:     list of parameters of the Lorentz-function
            [XCEN,YCEN,FWHMX,FWHMY,AMP,BACKGROUND,ANGLE]
                ANGLE = rotation of the X,Y direction of the Lorentzian    
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


def Debye1(x):
    """
    function to calculate the first Debye function as needed
    for the calculation of the thermal Debye-Waller-factor
    by numerical integration

    for definition see:
    http://en.wikipedia.org/wiki/Debye_function

    D1(x) = (1/x) \int_0^x t/(exp(t)-1) dt

    Parameters:
    -----------
     x ... argument of the Debye function (float)

    Returns:
    --------
     D1(x) float value of the Debye function
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
        print("XU.math.Debye1: debye integral value/error estimate: %g %g"%(integral[0],integral[1]))

    return d1
