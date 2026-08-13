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

import numpy

# f = f0(|Q|) + f1(en) + j * f2(en)
import xrayutilities as xu

Fe = xu.materials.elements.Fe  # iron atom
Q = numpy.array([0, 0, 1.9], dtype=numpy.double)
en = 10000  # energy in eV

print(f"Iron (Fe): E: {en:9.1f} eV")
print(f"f0: {Fe.f0(numpy.linalg.norm(Q)):8.4g}")
print(f"f1: {Fe.f1(en):8.4g}")
print(f"f2: {Fe.f2(en):8.4g}")
