"""
xrutils utilities contains a conglomeration of useful functions
which do not fit into one of the other files
"""

import numpy

from . import config
from .utilities_noconf import *

def maplog(inte,dynlow = "config",dynhigh = "config",**keyargs):
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

    optional keyword arguments (NOT IMPLEMENTED):
     abslow: 10^(abslow) will be taken as lower boundary
     abshigh: 10^(abshigh) will be taken as higher boundary
                               
    Returns
    -------
     numpy.array of the same shape as inte, where values smaller/larger then 
     10^(-dynlow,dynhigh) were replaced by 10^(-dynlow,dynhigh) 

    Example
    -------
     >>> lint = maplog(int,5,2)
    """
    
    if dynlow == "config":
        dynlow = config.DYNLOW
    if dynhigh == "config":
        dynhigh = config.DYNHIGH
    
    if inte.max() <= 0.0:
        raise ValueError("XU.maplog: only negativ or zero values given. Log is not defined!")
    ma = inte.max()*10**(-dynhigh) # upper bound
    mi = inte.max()*10**(-dynlow)  # lower bound
 
    return numpy.log10(numpy.minimum(numpy.maximum(inte,mi),ma))
    #}}}

