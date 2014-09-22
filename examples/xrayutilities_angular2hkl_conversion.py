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
# Copyright (C) 2013,2014 Dominik Kriegner <dominik.kriegner@gmail.com>


"""
This example shows the use of the Ang2HKL function
used to convert angles to HKL coordinates of a certain material

The orientation of the crystal is specified by the two directions given to the
HXRD class.

The same example is shown twice using different goniometer definitions with
different incidence beam directions. The different definition changes the
momentum transfer Q (only its orientation), however both yields the same HKL
upon back and force conversion to angular and HKL-space.
"""

import xrayutilities as xu

# material used in this example
mat = xu.materials.Si

H = 3
K = 3
L = 1

# define experiment geometry
hxrd = xu.HXRD(mat.Q(1, 1, -2), mat.Q(1, 1, 1))

# calculate angles
[om, dummy1, dummy2, tt] = hxrd.Q2Ang(mat.Q(H, K, L))

# perform conversion to reciprocal space
print(hxrd.Transform(mat.Q(H, K, L)))
print(hxrd.Ang2Q(om, tt))

# perform conversion ro HKL coordinates
if dummy1 != 0 or dummy2 != 0:
    print("Geometry not enough for full description -> HKL will be wrong")
print(hxrd.Ang2HKL(om, tt, mat=mat))

print("--------------")
# example with custom qconv
qconv = xu.experiment.QConversion(['z+', 'x+', 'y-'], 'z+', [1, 0, 0])

# material used in this example
mat = xu.materials.Si

# define experiment geometry
hxrd = xu.HXRD(mat.Q(1, 1, -2), mat.Q(1, 1, 1), qconv=qconv)

# calculate angles
# IMPORTANT: the Q2Ang function is not influenced by the qconv argument. A
# proper calculation of the angles needs the Q2AngFit function described in
# the example file 'xrayutilities_q2ang_general.py'
[om, chi, phi, tt] = hxrd.Q2Ang(mat.Q(H, K, L))

# perform conversion to reciprocal space
print(hxrd.Transform(mat.Q(H, K, L)))
print(hxrd.Ang2Q(om, chi, phi, tt))

# perform conversion ro HKL coordinates
print(hxrd.Ang2HKL(om, chi, phi, tt, mat=mat))
