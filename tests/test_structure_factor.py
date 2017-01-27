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
# Copyright (C) 2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import xrayutilities as xu
import numpy


class TestStructureFactor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = xu.materials.elements.dummy
        cls.mat = xu.materials.Crystal(
            'test', xu.materials.SGLattice('227:1', 4, atoms=[cls.at, ],
                                           pos=['8a', ]))

    def test_StructureFactor(self):
        f = self.mat.StructureFactor(self.mat.Q(1, 3, 1))
        self.assertAlmostEqual(f, 4 - 4j, places=10)
        f = self.mat.StructureFactor(self.mat.Q(0, 4, 0))
        self.assertAlmostEqual(f, 8, places=10)
        f = self.mat.StructureFactor(self.mat.Q(1, 2, 1))
        self.assertAlmostEqual(f, 0, places=10)

    def test_StructureFactorQ(self):
        q = (self.mat.Q(1, 1, 1), self.mat.Q(0, 4, 0), self.mat.Q(1, 2, 1))
        f = self.mat.StructureFactorForQ(q)
        for i in range(3):
            self.assertAlmostEqual(f[i], (4 + 4j, 8, 0)[i], places=10)

    def test_StructureFactorE(self):
        q = self.mat.Q(1, 1, 1)
        f = self.mat.StructureFactorForEnergy(q, (1000, 2000, 3000))
        for i in range(3):
            self.assertAlmostEqual(f[i], 4 + 4j, places=10)


if __name__ == '__main__':
    unittest.main()
