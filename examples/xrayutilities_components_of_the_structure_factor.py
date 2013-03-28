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

# f = f0(|Q|) + f1(en) + j * f2(en)
import xrayutilities as xu
import numpy

Fe = xu.materials.elements.Fe # iron atom
Q = numpy.array([0,0,1.9],dtype=numpy.double)
en = 10000 # energy in eV

print("Iron (Fe): E: %9.1f eV" % en)
print("f0: %8.4g" % Fe.f0(numpy.linalg.norm(Q)))
print("f1: %8.4g" % Fe.f1(en))
print("f2: %8.4g" % Fe.f2(en))
