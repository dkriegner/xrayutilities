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
from numpy import arccos, cos, radians, sin, sqrt


class TestMaterialsTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b, cls.c = numpy.random.rand(3)*2 + 4
        cls.alpha, cls.beta, cls.gamma = numpy.random.rand(3) * 60 + 60
        cls.p1mat = xu.materials.Crystal(
            'P1', xu.materials.SGLattice(1, cls.a, cls.b, cls.c,
                                         cls.alpha, cls.beta, cls.gamma))

    def test_q2hkl_hkl2q(self):
        for i in range(3):
            hkls = numpy.random.randint(-5, 6, 3)
            qvec = self.p1mat.Q(hkls)
            backhkl = self.p1mat.HKL(qvec)
            for j in range(3):
                self.assertAlmostEqual(hkls[j], backhkl[j], places=10)

    def test_Bmatrix(self):
        """
        check if our B matrix is compatible with the one from
        Busing&Levy Acta Cryst. 22, 457 (1967)
        """
        ca = cos(radians(self.alpha))
        cb = cos(radians(self.beta))
        cg = cos(radians(self.gamma))
        sa = sin(radians(self.alpha))
        sb = sin(radians(self.beta))
        sg = sin(radians(self.gamma))
        vh = sqrt(1 - ca**2-cb**2-cg**2 + 2*ca*cb*cg)
        pi2 = numpy.pi * 2
        ra, rb, rc = pi2*sa/(self.a*vh), pi2*sb/(self.b*vh), pi2*sg/(self.c*vh)
        cralpha = (cb*cg - ca)/(sb*sg)
        crbeta = (ca*cg - cb)/(sa*sg)
        crgamma = (ca*cb - cg)/(sa*sb)

        b = numpy.zeros((3, 3))
        b[0, 0] = ra
        b[0, 1] = rb * crgamma
        b[1, 1] = rb * sin(arccos(crgamma))
        b[0, 2] = rc * crbeta
        b[1, 2] = -rc * sin(arccos(crbeta))*cos(radians(self.alpha))
        b[2, 2] = pi2 / self.c

        for j in range(9):
            self.assertAlmostEqual(b.flat[j], self.p1mat.B.flat[j], places=10)
