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
# Copyright (C) 2012, 2016-2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import matplotlib.pyplot as plt
import numpy
import xrayutilities as xu

matA = xu.materials.InAs
matB = xu.materials.InP
substrate = xu.materials.Si

alloy = xu.materials.CubicAlloy(matA, matB, 0)

hxrd001 = xu.HXRD([1, 1, 0], [0, 0, 1])
qinp, qout = (3.02829203, 4.28265165)

# draw two relaxation triangles for the given Alloy in the substrate
[qxt0, qzt0] = alloy.RelaxationTriangle([2, 2, 4], substrate, hxrd001)
alloy.x = 1.
[qxt1, qzt1] = alloy.RelaxationTriangle([2, 2, 4], substrate, hxrd001)

plt.figure()
plt.plot(qxt0, qzt0, 'r-')
plt.plot(qxt1, qzt1, 'b-')
plt.plot(qinp, qout, 'ko')

# print concentration of alloy B calculated from a reciprocal space point
print(alloy.ContentBasym(qinp, qout, [2, 2, 4], [0, 0, 1]))
