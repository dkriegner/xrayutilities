"""
xrutils utilities contains a conglomeration of useful functions
this part of utilities does not need the config class
"""

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

