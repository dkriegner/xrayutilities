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
# Copyright (C) 2014-2020 Dominik Kriegner <dominik.kriegner@gmail.com>
# Copyright (C) 2022 Vinícius Frehse <vinifrehse@gmail.com>

import math
import unittest

import numpy
import xrayutilities as xu
from numpy import arccos, cos, radians, sin, sqrt


class TestMaterialsTransform(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.a, cls.b, cls.c = numpy.random.rand(3) * 2 + 4
        cls.alpha, cls.beta, cls.gamma = numpy.random.rand(3) * 60 + 60
        cls.c11, cls.c12, cls.c44 = numpy.random.rand(3) * 1e10
        cls.p1mat = xu.materials.Crystal(
            'P1', xu.materials.SGLattice(1, cls.a, cls.b, cls.c,
                                         cls.alpha, cls.beta, cls.gamma), 
                                         xu.materials.CubicElasticTensor(
                                         cls.c11, cls.c12, cls.c44))

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
        # cralpha = (cb*cg - ca)/(sb*sg)
        crbeta = (ca*cg - cb)/(sa*sg)
        crgamma = (ca*cb - cg)/(sa*sb)

        b = numpy.zeros((3, 3))
        b[0, 0] = ra
        b[0, 1] = rb * crgamma
        b[1, 1] = rb * sin(arccos(crgamma))
        b[0, 2] = rc * crbeta
        b[1, 2] = -rc * sin(arccos(crbeta)) * ca
        b[2, 2] = pi2 / self.c

        for j in range(9):
            self.assertAlmostEqual(b.flat[j], self.p1mat.B.flat[j], places=10)

    def test_Bmatrix_after_a_setter(self):
        self.a = numpy.random.rand() * 2 + 4
        self.p1mat.lattice.a = self.a
        self.test_Bmatrix()

    def test_Bmatrix_after_alpha_setter(self):
        self.alpha = numpy.random.rand() * 60 + 60
        self.p1mat.lattice.alpha = self.alpha
        self.test_Bmatrix()

    def test_Bmatrix_after_all_setters(self):
        # change materials unit cell parameters to test setters
        self.a, self.b, self.c = numpy.random.rand(3) * 2 + 4
        self.alpha, self.beta, self.gamma = numpy.random.rand(3) * 60 + 60
        self.p1mat.lattice.a = self.a
        self.p1mat.lattice.b = self.b
        self.p1mat.lattice.c = self.c
        self.p1mat.lattice.alpha = self.alpha
        self.p1mat.lattice.beta = self.beta
        self.p1mat.lattice.gamma = self.gamma
        self.test_Bmatrix()

    def test_environment(self):
        maxdist = max(self.a, self.b, self.c) + 0.01
        e = self.p1mat.environment((0, 0, 0), maxdist=maxdist)

        self.assertTrue(len(e) >= 4,
                        f"Length of environment must be >= 4, is {len(e)}")
        for dis in (0.0, self.a, self.b, self.c):
            found = False
            for d, at, mult in e:
                if numpy.isclose(d, dis):
                    found = True
            self.assertTrue(found, "expected atomic distance not found")

    def test_environment_Si(self):
        a = xu.materials.Si.a
        e = xu.materials.Si.environment(0.125, 0.125, 0.125)

        self.assertAlmostEqual(e[0][0], a*math.sqrt(3)/8, places=10)
        self.assertEqual(e[0][1], xu.materials.elements.Si)
        self.assertAlmostEqual(e[0][2], 2.0)

    def test_isequivalent(self):
        hkl1 = (1, 2, 3)
        materials = ['C', 'C_HOPG', 'TiO2', 'GeTe', 'Ag2Se']
        hkl2lst = [((2, 1, -3), (2, 2, 3)),
                   ((1, -3, 3), (1, -2, 3)),
                   ((-2, 1, 3), (3, 2, 1)),
                   ((1, 3, 2), (1, 3, -2)),
                   ((1, -2, -3), (1, 3, 2))]
        for mname, hkl2s in zip(materials, hkl2lst):
            mat = getattr(xu.materials, mname)
            self.assertTrue(mat.lattice.isequivalent(hkl1, hkl2s[0]))
            self.assertFalse(mat.lattice.isequivalent(hkl1, hkl2s[1]))

    def test_Strain(self):
        strain = numpy.zeros((3,3), dtype=numpy.double)
        strain[0,0:3] = numpy.random.rand(3)
        strain[1,1:3] = numpy.random.rand(2)
        strain[2,2] = numpy.random.rand(1)
        strain[0:3,0] = strain[0,0:3]
        strain[1:3,1] = strain[1,1:3]

        stress = elf.p1mat.GetStress(strain)
        strain_rev = elf.p1mat.GetStrain(stress)
        numpy.testing.assert_almost_equal(strain, strain_rev)

    def test_Stress(self):
        stress = numpy.zeros((3,3), dtype=numpy.double)
        stress[0,0:3] = numpy.random.rand(3)
        stress[1,1:3] = numpy.random.rand(2)
        stress[2,2] = numpy.random.rand(1)
        stress[0:3,0] = stress[0,0:3]
        stress[1:3,1] = stress[1,1:3]
        
        strain = self.p1mat.GetStrain(stress)
        stress_rev = self.p1mat.GetStress(strain)
        numpy.testing.assert_almost_equal(stress, stress_rev)


if __name__ == '__main__':
    unittest.main()
