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
# Copyright (C) 2014-2015 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu


class TestQConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat = xu.materials.Si
        cls.hxrd = xu.HXRD(cls.mat.Q(1, 1, 0), cls.mat.Q(0, 0, 1))
        cls.hklsym = (0, 0, 4)
        cls.hklasym = (2, 2, 4)

    def test_qconversion_point(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        qout = self.hxrd.Ang2HKL(ang[0], ang[3], mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hklsym[i], places=10)

    def test_qconversion_point_asym(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklasym))
        qout = self.hxrd.Ang2HKL(ang[0], ang[3], mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i], self.hklasym[i], places=10)

    def test_qconversion_energy(self):
        ang1 = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        ang2 = self.hxrd.Q2Ang(self.mat.Q(self.hklsym)/2.)
        qout = self.hxrd.Ang2HKL((ang1[0], ang2[0]), (ang1[3], ang2[3]),
                                 en=(self.hxrd.energy, 2 * self.hxrd.energy),
                                 mat=self.mat)
        for i in range(3):
            self.assertAlmostEqual(qout[i][0], self.hklsym[i], places=10)
            self.assertAlmostEqual(qout[i][1], self.hklsym[i], places=10)

    def test_qconversion_detpos(self):
        tt = numpy.random.rand() * 90
        dpos = self.hxrd.Ang2Q.getDetectorPos(tt, dim=0)
        dpos = numpy.asarray(dpos)
        kf = dpos / numpy.linalg.norm(dpos) * self.hxrd.k0
        ki = self.hxrd._A2QConversion.r_i / \
            numpy.linalg.norm(self.hxrd._A2QConversion.r_i) * self.hxrd.k0
        qout = self.hxrd.Ang2Q(0, tt)
        for i in range(3):
            self.assertAlmostEqual(qout[i], kf[i]-ki[i], places=10)


if __name__ == '__main__':
    unittest.main()
