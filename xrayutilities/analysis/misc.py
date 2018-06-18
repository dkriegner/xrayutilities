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
# Copyright (C) 2012-2018 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
miscellaneous functions helpful in the analysis and experiment
"""

import numpy

from .. import config
from .. import math


def getangles(peak, sur, inp):
    """
    calculates the chi and phi angles for a given peak

    Parameters
    ----------

     peak:  array which gives hkl for the peak of interest
     sur:   hkl of the surface
     inp:   inplane reference peak or direction

    Returns
    -------
     [chi,phi] for the given peak on surface sur with inplane direction inp
               as reference

    Examples
    --------
     To get the angles for the -224 peak on a 111 surface type
      [chi,phi] = getangles([-2,2,4],[1,1,1],[2,2,4])

    """

    # transform input to numpy.arrays
    peak = numpy.array(peak)
    sur = numpy.array(sur)
    inp = numpy.array(inp)

    peak = peak / numpy.linalg.norm(peak)
    sur = sur / numpy.linalg.norm(sur)
    inp = inp / numpy.linalg.norm(inp)

    # calculate reference inplane direction
    inplane = numpy.cross(numpy.cross(sur, inp), sur)
    inplane = inplane / numpy.linalg.norm(inplane)
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analyis.getangles: reference inplane direction: ", inplane)

    # calculate inplane direction of peak
    pinp = numpy.cross(numpy.cross(sur, peak), sur)
    pinp = pinp / numpy.linalg.norm(pinp)
    if(numpy.linalg.norm(numpy.cross(sur, peak)) <= config.EPSILON):
        pinp = inplane
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analyis.getangles: peaks inplane direction: ", pinp)

    # calculate angles
    r2d = 180. / numpy.pi
    chi = numpy.arccos(numpy.dot(sur, peak)) * r2d
    if(numpy.dot(sur, peak) >= 1. - config.EPSILON):
        chi = 0.
        phi = 0.
    elif(numpy.dot(sur, peak) <= -1. + config.EPSILON):
        chi = 180.
        phi = 0.
    elif(numpy.dot(sur, numpy.cross(inplane, pinp)) <= config.EPSILON):
        if numpy.dot(pinp, inplane) >= 1.0:
            phi = 0.
        elif numpy.dot(pinp, inplane) <= -1.0:
            phi = 180.
        else:
            phi = numpy.sign(numpy.dot(sur, numpy.cross(inplane, pinp))) *\
                numpy.arccos(numpy.dot(pinp, inplane)) * r2d
    else:
        phi = numpy.sign(numpy.dot(sur, numpy.cross(inplane, pinp))) * \
            numpy.arccos(numpy.dot(pinp, inplane)) * r2d
    phi = phi - round(phi / 360.) * 360

    return (chi, phi)


def getunitvector(chi, phi, ndir=(0, 0, 1), idir=(1, 0, 0)):
    """
    return unit vector determined by spherical angles and definition of the
    polar axis and inplane reference direction (phi=0)

    Parameters
    ----------
     chi, phi: spherical angles (polar and azimuthal) in degree
     ndir:     polar/z-axis (determines chi=0)
     idir:     azimuthal axis (determines phi=0)
    """
    chi_axis = numpy.cross(ndir, idir)
    v = math.rotarb(ndir, chi_axis, chi)
    v = math.rotarb(v, ndir, phi)
    return v / numpy.linalg.norm(v)


def coplanar_intensity(mat, exp, hkl, thickness, thMono, sample_width=10,
                       beam_width=1):
    """
    Calculates the expected intensity of a Bragg peak from an epitaxial thin
    film measured in coplanar geometry (integration over omega and 2theta in
    angular space!)

    Parameters
    ----------
    mat:          Crystal instance for structure factor calculation
    exp:          Experimental(HXRD) class for the angle calculation
    hkl:          Miller indices of the peak to calculate
    thickness:    film thickness in nm
    thMono:       Bragg angle of the monochromator (deg)
    sample_width: width of the sample along the beam
    beam_width:   width of the beam in the same units as the sample size

    Returns
    -------
     float: intensity of the peak
    """
    # angle calculation for geometrical factors
    om, chi, phi, tt = exp.Q2Ang(mat.Q(hkl))

    # structure factor calculation
    r = abs(mat.StructureFactor(mat.Q(hkl)))**2

    # polarization factor
    Cmono = numpy.cos(2 * numpy.radians(thMono))
    P = (1 + Cmono * numpy.cos(numpy.radians(tt))**2) / ((1 + Cmono))
    # Lorentz factor to be used when integrating in angular space
    L = 1 / numpy.sin(numpy.radians(tt))
    # shape factor: changing illumination with the incidence angle
    shapef = beam_width / (numpy.sin(numpy.radians(om)) * sample_width)
    if shapef > 1:
        shapef = 1

    # absorption correction
    mu = 1 / (mat.absorption_length() * 1e3)
    mu_eff = mu * (abs(1 / numpy.sin(numpy.radians(om))) +
                   abs(1 / numpy.sin(numpy.radians(tt - om))))
    Nblocks = (1 - numpy.exp(-mu_eff * thickness)) / mu_eff

    return r * P * L * shapef * Nblocks
