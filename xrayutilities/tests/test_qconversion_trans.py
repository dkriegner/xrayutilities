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


class TestQConversionTrans(unittest.TestCase):

    def setUp(self):
        self.nch = 9
        self.ncch = 4
        self.nch2d = (9, 13)
        self.ncch1 = 4
        self.ncch2 = 6
        # standard 1S+1D goniometer
        qconv = xu.QConversion('x+', 'x+', (0, 1, 0))
        self.hxrd = xu.HXRD((1., 1., 0.), (0., 0., 1.), qconv=qconv)
        # comparable goniometer with translations
        qconv = xu.QConversion('x+', ['ty', 'tz'], (0, 1e-15, 0))
        self.hxrdtrans = xu.HXRD((1., 1., 0.), (0., 0., 1.), qconv=qconv,
                                 sampleor='z+')
        self.hxrdtrans.Ang2Q.init_linear('z+', self.ncch, self.nch,
                                         1e-15, 50e-6)
        self.hxrdtrans.Ang2Q.init_area('z+', 'x+', self.ncch1, self.ncch2,
                                       self.nch2d[0], self.nch2d[1],
                                       1e-15, 50e-6, 50e-6)

        self.angle = numpy.random.rand() * 45

    def test_qtrans0(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 0)
        qvec2 = self.hxrdtrans.Ang2Q(self.angle, 1, 0)
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans45(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 45)
        qvec2 = self.hxrdtrans.Ang2Q(self.angle, 1, 1)

        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans90(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 90)
        qvec2 = self.hxrdtrans.Ang2Q(self.angle, 0, 1)
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans0_linear(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 0)
        qx, qy, qz = self.hxrdtrans.Ang2Q.linear(self.angle, 1, 0)
        qvec2 = (qx[self.ncch], qy[self.ncch], qz[self.ncch])
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans45_linear(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 45)
        qx, qy, qz = self.hxrdtrans.Ang2Q.linear(self.angle, 1, 1)
        qvec2 = (qx[self.ncch], qy[self.ncch], qz[self.ncch])
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans90_linear(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 90)
        qx, qy, qz = self.hxrdtrans.Ang2Q.linear(self.angle, 0, 1)
        qvec2 = (qx[self.ncch], qy[self.ncch], qz[self.ncch])
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans0_area(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 0)
        qx, qy, qz = self.hxrdtrans.Ang2Q.area(self.angle, 1, 0)
        qvec2 = (qx[self.ncch1, self.ncch2],
                 qy[self.ncch1, self.ncch2],
                 qz[self.ncch1, self.ncch2])
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans45_area(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 45)
        qx, qy, qz = self.hxrdtrans.Ang2Q.area(self.angle, 1, 1)
        qvec2 = (qx[self.ncch1, self.ncch2],
                 qy[self.ncch1, self.ncch2],
                 qz[self.ncch1, self.ncch2])
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)

    def test_qtrans90_area(self):
        qvec1 = self.hxrd.Ang2Q(self.angle, 90)
        qx, qy, qz = self.hxrdtrans.Ang2Q.area(self.angle, 0, 1)
        qvec2 = (qx[self.ncch1, self.ncch2],
                 qy[self.ncch1, self.ncch2],
                 qz[self.ncch1, self.ncch2])
        for i in range(3):
            self.assertAlmostEqual(qvec1[i], qvec2[i], places=10)


if __name__ == '__main__':
    unittest.main()
