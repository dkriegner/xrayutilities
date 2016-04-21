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

import xrayutilities as xu
import numpy


class TestQConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat = xu.materials.Si
        cls.hxrd = xu.HXRD(cls.mat.Q(1, 1, 0), cls.mat.Q(0, 0, 1))
        cls.nch = 9
        cls.ncch = 4
        cls.hxrd.Ang2Q.init_linear('z+', cls.ncch, cls.nch, 1.0, 50e-6, 0.)
        cls.hklsym = (0, 0, 4)
        cls.hklasym = (2, 2, 4)

    def test_qconversion_linear(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        qout = self.hxrd.Ang2HKL(ang[0], ang[3], mat=self.mat,
                                 dettype='linear')
        self.assertEqual(qout[0].size, self.nch)
        for i in range(3):
            q = qout[i]
            self.assertAlmostEqual(numpy.average(q), self.hklsym[i], places=6)

    def test_qconversion_linear_asym(self):
        ang = self.hxrd.Q2Ang(self.mat.Q(self.hklasym))
        qout = self.hxrd.Ang2HKL(ang[0], ang[3], mat=self.mat,
                                 dettype='linear')
        self.assertEqual(qout[0].size, self.nch)
        for i in range(3):
            q = qout[i]
            self.assertAlmostEqual(numpy.average(q), self.hklasym[i], places=6)

    def test_qconversion_linear_energy(self):
        ang1 = self.hxrd.Q2Ang(self.mat.Q(self.hklsym))
        ang2 = self.hxrd.Q2Ang(self.mat.Q(self.hklsym) / 2.)
        qout = self.hxrd.Ang2HKL((ang1[0], ang2[0]), (ang1[3], ang2[3]),
                                 en=(self.hxrd.energy, 2 * self.hxrd.energy),
                                 mat=self.mat, dettype='linear')
        for i in range(3):
            q = qout[i]
            self.assertAlmostEqual(q[0, self.ncch], self.hklsym[i], places=10)
            self.assertAlmostEqual(q[1, self.ncch], self.hklsym[i], places=10)

    def test_qconversion_linear_detpos(self):
        tt = numpy.random.rand() * 90
        dpos = self.hxrd.Ang2Q.getDetectorPos(tt, dim=1)
        ki = self.hxrd._A2QConversion.r_i / \
            numpy.linalg.norm(self.hxrd._A2QConversion.r_i) * self.hxrd.k0
        qout = self.hxrd.Ang2Q.linear(0, tt)
        for j, x, y, z in zip(range(self.nch), dpos[0], dpos[1], dpos[2]):
            vpos = numpy.asarray((x, y, z))
            kf = vpos / numpy.linalg.norm(vpos) * self.hxrd.k0
            for i in range(3):
                self.assertAlmostEqual(qout[i][j], kf[i]-ki[i], places=10)


if __name__ == '__main__':
    unittest.main()
