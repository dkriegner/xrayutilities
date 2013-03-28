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

import xrayutilities as xu
import matplotlib.pyplot as plt
import numpy

matA = xu.materials.InAs
matB = xu.materials.InP
substrate = xu.materials.Si

alloy = xu.materials.Alloy(matA,matB,0)

exp001 = xu.HXRD([1,1,0],[0,0,1])

# note
# copy.deepcopy fails on matA.lattice; I think this is because it is not possible to copy the latticebase stuff because it is not a standard list and the deepcopy command is not capable of this?


# draw two relaxation triangles for the given Alloy in the substrate
[qxt0,qzt0] = alloy.RelaxationTriangle([2,2,4],substrate,exp001)
alloy.x = 1.
[qxt1,qzt1] = alloy.RelaxationTriangle([2,2,4],substrate,exp001)

plt.figure(1)
plt.clf()
plt.plot(qxt0,qzt0,'r-')
plt.plot(qxt1,qzt1,'b-')

# print concentration of alloy B calculated from a reciprocal space point
print(alloy.ContentBasym(3.02829203,4.28265165,[2,2,4],[0,0,1]))


