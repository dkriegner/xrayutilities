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

import unittest

import numpy
import xrayutilities as xu


class TestGridder2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nx = 10
        # do not change this here unless you fix also the tests cases
        cls.ny = 19
        cls.xmin = 1
        cls.xmax = 10
        cls.x = numpy.linspace(cls.xmin, cls.xmax, num=cls.nx)
        cls.y = cls.x.copy()
        cls.data = numpy.random.rand(cls.nx)
        cls.gridder = xu.Gridder2D(cls.nx, cls.ny)
        cls.gridder(cls.x, cls.y, cls.data)

    def test_gridder2d_xaxis(self):
        # test length of xaxis
        self.assertEqual(len(self.gridder.xaxis), self.nx)
        # test values of xaxis
        for i in range(self.nx):
            self.assertAlmostEqual(self.gridder.xaxis[i], self.x[i], places=12)

    def test_gridder2d_yaxis(self):
        # test length of yaxis
        self.assertEqual(len(self.gridder.yaxis), self.ny)
        # test end values of yaxis
        self.assertAlmostEqual(self.gridder.yaxis[0], self.y[0], places=12)
        self.assertAlmostEqual(self.gridder.yaxis[-1], self.y[-1], places=12)
        self.assertAlmostEqual(
            self.gridder.yaxis[1] - self.gridder.yaxis[0],
            (self.xmax - self.xmin) / float(self.ny - 1),
            places=12)

    def test_gridder2d_data(self):
        # test shape of data
        self.assertEqual(self.gridder.data.shape[0], self.nx)
        self.assertEqual(self.gridder.data.shape[1], self.ny)
        # test values of data
        aj, ak = numpy.indices((self.nx, self.ny))
        aj, ak = numpy.ravel(aj), numpy.ravel(ak)
        for i in range(self.nx * self.ny):
            j, k = (aj[i], ak[i])
            if k == 2 * j:
                self.assertAlmostEqual(
                    self.gridder.data[j, 2 * j],
                    self.data[j],
                    places=12)
            else:
                self.assertEqual(self.gridder.data[j, k], 0.)


if __name__ == '__main__':
    unittest.main()
