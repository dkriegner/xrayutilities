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
miscellaneous functions helpful in the analysis and experiment
"""

import numpy

from .. import config
from .. import math

def getangles(peak,sur,inp):
    """     
    calculates the chi and phi angles for a given peak 
    
    Parameter 
    ---------

     peak:  array which gives hkl for the peak of interest
     sur:   hkl of the surface
     inp:   inplane reference peak or direction

    Returns
    -------
     [chi,phi] for the given peak on surface sur with inplane direction inp as reference
    
    Example
    -------
     To get the angles for the -224 peak on a 111 surface type
      [chi,phi] = getangles([-2,2,4],[1,1,1],[2,2,4])
    
    """
    
    # transform input to numpy.arrays
    peak = numpy.array(peak)
    sur = numpy.array(sur)
    inp = numpy.array(inp)
    
    peak = peak/numpy.linalg.norm(peak)
    sur = sur/numpy.linalg.norm(sur)
    inp = inp/numpy.linalg.norm(inp)
    
    # calculate reference inplane direction
    inplane = numpy.cross(numpy.cross(sur,inp),sur)
    inplane = inplane/numpy.linalg.norm(inplane)
    if config.VERBOSITY >= config.INFO_ALL: 
        print("XU.analyis.getangles: reference inplane direction: ", inplane)

    # calculate inplane direction of peak
    pinp = numpy.cross(numpy.cross(sur,peak),sur)
    pinp = pinp/numpy.linalg.norm(pinp)
    if(numpy.linalg.norm(numpy.cross(sur,peak))<=config.EPSILON): pinp = inplane
    if config.VERBOSITY >= config.INFO_ALL:
        print("XU.analyis.getangles: peaks inplane direction: ", pinp)

    # calculate angles
    r2d = 180./numpy.pi
    chi = numpy.arccos(numpy.dot(sur,peak))*r2d
    #print numpy.dot(sur,peak),numpy.dot(sur,numpy.cross(inplane,pinp)),numpy.sign(numpy.dot(sur,numpy.cross(inplane,pinp))),numpy.dot(pinp,inplane)
    if(numpy.dot(sur,peak)>=1.-config.EPSILON): 
        chi = 0. 
        phi = 0.
    elif(numpy.dot(sur,peak)<=-1.+config.EPSILON):
        chi=180.
        phi=0.
    elif(numpy.dot(sur,numpy.cross(inplane,pinp))<=config.EPSILON):
        if numpy.dot(pinp,inplane) >= 1.0:
            phi = 0.
        elif numpy.dot(pinp,inplane) <= -1.0:
            phi =180.
        else: 
            phi = numpy.sign(numpy.dot(sur,numpy.cross(inplane,pinp)))*numpy.arccos(numpy.dot(pinp,inplane))*r2d
    else: 
        phi = numpy.sign(numpy.dot(sur,numpy.cross(inplane,pinp)))*numpy.arccos(numpy.dot(pinp,inplane))*r2d
    phi = phi - round(phi/360.)*360
   
    return (chi,phi)


