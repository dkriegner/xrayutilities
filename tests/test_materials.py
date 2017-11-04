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
# Copyright (C) 2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu


class TestMaterialsTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a, b, c = numpy.random.rand(3)*2 + 4
        alpha, beta, gamma = numpy.random.rand(3) * 60 + 60
        cls.p1mat = xu.materials.Crystal('P1',
                                         xu.materials.SGLattice(1, a, b, c,
                                                                alpha, beta,
                                                                gamma))

    def test_q2hkl_hkl2q(self):
        for i in range(3):
            hkls = numpy.random.randint(-5, 6, 3)
            qvec = self.p1mat.Q(hkls)
            backhkl = self.p1mat.HKL(qvec)
            for j in range(3):
                self.assertAlmostEqual(hkls[j], backhkl[j], places=10)
