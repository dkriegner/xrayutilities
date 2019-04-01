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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy

import xrayutilities as xu


class TestAlloyContentCalc(unittest.TestCase):
    matA = xu.materials.InAs
    matB = xu.materials.InP
    substrate = xu.materials.InAs
    x = numpy.random.rand()
    alloy = xu.materials.CubicAlloy(matA, matB, x)
    hxrd001 = xu.HXRD([1, 1, 0], [0, 0, 1])
    [qxa, qza] = alloy.RelaxationTriangle([2, 2, 4], substrate, hxrd001)
    [qxs, qzs] = alloy.RelaxationTriangle([0, 0, 4], substrate, hxrd001)

    def test_ContentAsym(self):
        content, [ainp, aperp, abulk, eps_inp, eps_perp] = \
            self.alloy.ContentBasym(self.qxa[0], self.qza[0],
                                    [2, 2, 4], [0, 0, 1])
        self.assertAlmostEqual(content, self.x, places=6)
        content, [ainp, aperp, abulk, eps_inp, eps_perp] = \
            self.alloy.ContentBasym(self.qxa[1], self.qza[1],
                                    [2, 2, 4], [0, 0, 1])
        self.assertAlmostEqual(content, self.x, places=6)

    def test_ContentSym(self):
        content = self.alloy.ContentBsym(
            self.qzs[0], [0, 0, 4], [1, 1, 0], self.matA.lattice.a, 1.0)
        self.assertAlmostEqual(content, self.x, places=6)
        content = self.alloy.ContentBsym(
            self.qzs[1], [0, 0, 4], [1, 1, 0], self.matA.lattice.a, 0.0)
        self.assertAlmostEqual(content, self.x, places=6)


if __name__ == '__main__':
    unittest.main()
