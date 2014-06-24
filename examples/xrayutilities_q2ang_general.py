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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
This example shows the use of the Q2AngFit function to calculate experimental
angles which can not be calculated by the analytic functions coded in the 
HXRD, NonCOP and GID experimental classes, which use a fixed geometry. 

Here an arbitrary goniometer together with some restrictions can be defined and
experimental angles can be calculated for this geometry
"""

import xrayutilities as xu
import numpy
import time

energy = 15000

###########################
# definition of goniometer
###########################
qconv = xu.experiment.QConversion(['z+','y-','z-'],['z+','y-'],[1,0,0]) # 3S+2D goniometer (simplified ID01 goniometer, sample mu,eta,phi detector nu,del
    # convention for coordinate system: x downstream; z upwards; y to the "outside" (righthanded)
    # QConversion will set up the goniometer geometry. So the first argument describes the sample rotations, the second the detector rotations and the third the primary beam direction.
    # For this consider the following coordinate system (at least this is what i use at ID01, feel free to use your conventions):
    # x: downstream (direction of primary beam)
    # y: out of the ring
    # z: upwards
    # these three axis form a right handed coordinate system.
    # The outer most sample rotation (so the one mounted on the floor) is one which turns righthanded (+) around the z-direction -> z+ (at the moment this rotation is called 'mu' in the spec-session)
    # The second sample rotation ('eta') is lefthanded (-) around y -> y-

# define experimental geometry with respect to the crystalline directions of the substrate
hxrd = xu.HXRD((1,0,0),(0,0,1),en=energy,qconv=qconv)

# tell bounds of angles / (min,max) pair for all motors
# mu,eta,phi detector nu,del
bounds = (0,(0,90),0,(-1,90),(0,90))


#############################
# call angle fit function 
#############################

ang = None
tbegin = time.time()
for i in range(1000):
    
    qvec = numpy.array((0,0,i*0.001))
    t0 = time.time()
    ang,qerror,errcode = xu.Q2AngFit(qvec,hxrd,bounds,startvalues=ang)
    t1 = time.time()

    print("%.4f: err %d qvec %s angles %s"%(t1-t0,errcode,str(qvec),str(ang)))

tend = time.time()
print("Total time needed: %.2fsec"%(tend-tbegin))
