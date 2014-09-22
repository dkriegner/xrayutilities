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
# Copyright (C) 2014 Dominik Kriegner <dominik.kriegner@gmail.com>

import xrayutilities as xu
import numpy
import unittest


class TestQConversion(unittest.TestCase):

    def setUp(self):
        self.mat = xu.materials.Si
        self.hxrd = xu.HXRD(self.mat.Q(1, 1, 0), self.mat.Q(0, 0, 1))
        self.nch = 9
        self.hxrd.Ang2Q.init_linear('z+', 4, self.nch, 1.0, 50e-6, 0.)
        self.hklsym = (0, 0, 4)
        self.hklasym = (2, 2, 4)

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

if __name__ == '__main__':
    unittest.main()
