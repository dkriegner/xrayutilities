"""
xrutils utilities contains a conglomeration of useful functions
which do not fit into one of the other files
"""

import numpy
import scipy.constants

def lam2en(inp):
    #{{{
    """
    converts the input energy in eV to a wavelength in Angstrom
    or the input wavelength in Angstrom to an energy in eV

    Parameter
    ---------
     inp : either an energy in eV or an wavelength in Angstrom

    Returns
    -------
     float, energy in eV or wavlength in Angstrom

    Examples
    --------
     >>> lambda = lam2en(8048)
     >>> energy = lam2en(1.5406)
    """
    #  E(eV) = h*c/(e * lambda(A)) *1e10    
    #  lambda(A) = h*c/(e * E(eV)) *1e10
    out = scipy.constants.h*scipy.constants.speed_of_light/(scipy.constants.e* inp) * 1e10

    return out
    #}}}


def maplog(inte,dynlow = 6,dynhigh =0):
    #{{{
    """
    clips values smaller and larger as the given bounds and returns the log10
    of the input array. The bounds are given as exponent with base 10 with respect 
    to the maximum in the input array.
    The function is implemented in analogy to J. Stangl's matlab implementation.

    Parameters
    ----------
     inte : numpy.array, values to be cut in range
     dynlow : 10^(-dynlow) will be the minimum cut off
     dynhigh : 10^(-dynhigh) will be the maximum cut off
                               
    Returns
    -------
     numpy.array of the same shape as inte, where values smaller/larger then 
     10^(-dynlow,dynhigh) were replaced by 10^(-dynlow,dynhigh) 

    Example
    -------
     >>> lint = maplog(int,5,2)
    """
    
    ma = inte.max()*10**(-dynhigh) # upper bound
    mi = inte.max()*10**(-dynlow)  # lower bound
 
    return numpy.log10(numpy.minimum(numpy.maximum(inte,mi),ma))
    #}}}

