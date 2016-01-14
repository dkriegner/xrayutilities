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
import numpy

# defining a ZincBlendeLattice with two types of atoms and lattice constant 'a'


def ZincBlendeLattice(aa, ab, a):
    # create lattice base
    lb = xu.materials.LatticeBase()
    lb.append(aa, [0, 0, 0])
    lb.append(aa, [0.5, 0.5, 0])
    lb.append(aa, [0.5, 0, 0.5])
    lb.append(aa, [0, 0.5, 0.5])
    lb.append(ab, [0.25, 0.25, 0.25])
    lb.append(ab, [0.75, 0.75, 0.25])
    lb.append(ab, [0.75, 0.25, 0.75])
    lb.append(ab, [0.25, 0.75, 0.75])

    # create lattice vectors
    a1 = [a, 0, 0]
    a2 = [0, a, 0]
    a3 = [0, 0, a]

    l = xu.materials.Lattice(a1, a2, a3, base=lb)
    return l

# defining InP, no elastic properties are given,
# helper functions exist to create the (6,6) elastic tensor for cubic materials
# which can be given as optional argument to the material class
InP = xu.materials.Crystal("InP",
                           ZincBlendeLattice(xu.materials.elements.In,
                                             xu.materials.elements.P, 5.8687))
# InP is of course already included in the xu.materials module
print(InP)
