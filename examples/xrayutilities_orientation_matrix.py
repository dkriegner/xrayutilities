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
# Copyright (C) 2013 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2013 Raphael Grifone <raphael.grifone@esrf.fr>

"""
example to show how to calculate the orientation matrix U from one symmetric
and one asymmetric diffraction peak. The orientation matrix can be used in the
reciprocal space conversion to correct the experimental alignment of the
crystal.

The procedure is as follows:

 1) determine the reciprocal space position without setting a custom U matrix
 2) calculate the orientation matrix
 3) recalculate the reciprocal space positions corrected for a misalignment of
    the lattice from the ideal diffraction condition

In this example only step number 2 is shown. look into other examples to see
how steps 1 and 3 are performed (this depends on the source of data you have)
"""

import numpy
import xrayutilities as xu

# lets assume in step 1) we found the reciprocal space positions of a GaAs
# crystal as

qsym = (0.01822137, -0.013112, 4.45427)  # (004) peak
qasym = (0.0173859, 1.56126512, 3.3450612)  # (113) peak

# the goal is to correct to the diffraction positions in a way that the
# symmetric peak is located on the z-axis and the asymmetric peak is located in
# either the x/z or y/z plane
# we do this by using the coordinate transformations included in xrayutilities

tz = xu.math.AxisToZ(qsym)  # correction of qsym to z-axis
# tz is a coordinate transformation objetct
# calling
tz(qsym)
# should return (~0, ~0, qz)

# now lets correct the sample azimuth in order to have the asymmetric peak in
# the x/z plane, again we make use of coordinate transformations in
# xrayutilities

qasym_tiltcorr = tz(qasym)
phicorr = numpy.arctan2(qasym_tiltcorr[1], qasym_tiltcorr[0])  # in radians
ta = xu.math.ZRotation(-phicorr, deg=False)

# to correctly transform the asymmetric peak to the to x/z plane use
ta(tz(qasym))
# should return (qx, ~0, qz)

# the orientation matrix is then given by

U = numpy.linalg.inv(numpy.dot(ta.matrix, tz.matrix))  # orientation matrix
# the matrix inversion is needed here because of the definition of U,
# which is usually used to calculate the misaligned q from the ideal one

# in case you just want the corrected peak positions you can use again a
# coordinate transformation
t = xu.math.Transform(numpy.linalg.inv(U))

t(qasym)  # will transform qasym to the x/z plane

# for correctly transforming a set of data us U in the call to the
# transformation function for point, linear or area detector
hxrd = xu.HXRD([1, 0, 0], [0, 0, 1])
# qx,qy,qz = hxrd.Ang2Q(om, tt, UB=U)
# or
# qx,qy,qz = hxrd.Ang2Q.linear(om, tt, UB=U)
# or
# qx,qy,qz = hxrd.Ang2Q.area(om, tt, UB=U)
