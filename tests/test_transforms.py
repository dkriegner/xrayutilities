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
# Copyright (C) 2015-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy
import xrayutilities as xu


class TestTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = (1.0, 0.0, 0.0)
        cls.y = (0.0, 1.0, 0.0)
        cls.z = (0.0, 0.0, 1.0)

    def test_Xrot(self):
        xr = xu.math.XRotation(90)
        for i in range(3):
            self.assertAlmostEqual(xr(self.y)[i], self.z[i], places=10)
        for i in range(3):
            self.assertAlmostEqual(xr(self.z)[i], -self.y[i], places=10)

    def test_Yrot(self):
        yr = xu.math.YRotation(90)
        for i in range(3):
            self.assertAlmostEqual(yr(self.z)[i], self.x[i], places=10)
        for i in range(3):
            self.assertAlmostEqual(yr(self.x)[i], -self.z[i], places=10)

    def test_Zrot(self):
        zr = xu.math.ZRotation(90)
        for i in range(3):
            self.assertAlmostEqual(zr(self.x)[i], self.y[i], places=10)
        for i in range(3):
            self.assertAlmostEqual(zr(self.y)[i], -self.x[i], places=10)

    def test_arbrot(self):
        r = xu.math.rotarb
        a = (1, 1, 1)
        for i in range(3):
            self.assertAlmostEqual(r(self.x, a, 120)[i], self.y[i], places=10)
        for i in range(3):
            self.assertAlmostEqual(r(self.z, a, 120)[i], self.x[i], places=10)

    def test_Axis2Z(self):
        a = numpy.random.rand(3)
        a[2] = 2  # ensure non-zero vector
        r = xu.math.AxisToZ(a)
        for i in range(3):
            self.assertAlmostEqual(r(a)[i],
                                   self.z[i] * numpy.linalg.norm(a),
                                   places=10)
        r = xu.math.AxisToZ_keepXY(a)
        for i in range(3):
            self.assertAlmostEqual(r(a)[i],
                                   self.z[i] * numpy.linalg.norm(a),
                                   places=10)
        # test inverse
        for i in range(3):
            self.assertAlmostEqual(r.inverse(self.z)[i],
                                   a[i] / numpy.linalg.norm(a),
                                   places=10)


if __name__ == '__main__':
    unittest.main()
