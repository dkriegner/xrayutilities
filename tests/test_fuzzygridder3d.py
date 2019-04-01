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

import numpy

import xrayutilities as xu


class TestFuzzyGridder3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 10
        cls.min = 1
        cls.max = 10
        cls.axis = numpy.linspace(cls.min, cls.max, cls.n)
        cls.x = int(numpy.random.rand()*10) + 1
        cls.y = int(numpy.random.rand()*10) + 1
        cls.z = int(numpy.random.rand()*10) + 1
        cls.data = numpy.random.rand()
        cls.gridder = xu.FuzzyGridder3D(cls.n, cls.n, cls.n)
        cls.gridder.dataRange(cls.min, cls.max,
                              cls.min, cls.max,
                              cls.min, cls.max)
        cls.gridder(cls.x, cls.y, cls.z, cls.data, width=2)

    def test_gridder3d_xaxis(self):
        # test length of xaxis
        self.assertEqual(len(self.gridder.xaxis), self.n)
        # test values of xaxis
        for i in range(self.n):
            self.assertAlmostEqual(self.gridder.xaxis[i], self.axis[i],
                                   places=12)

    def test_gridder3d_yaxis(self):
        # test length of yaxis
        self.assertEqual(len(self.gridder.yaxis), self.n)
        # test end values of yaxis
        for i in range(self.n):
            self.assertAlmostEqual(self.gridder.yaxis[i], self.axis[i],
                                   places=12)

    def test_gridder3d_zaxis(self):
        # test length of zaxis
        self.assertEqual(len(self.gridder.zaxis), self.n)
        # test end values of yaxis
        for i in range(self.n):
            self.assertAlmostEqual(self.gridder.zaxis[i], self.axis[i],
                                   places=12)

    def test_fuzzygridder3d_data(self):
        # test shape of data
        self.assertEqual(self.gridder.data.shape[0], self.n)
        self.assertEqual(self.gridder.data.shape[1], self.n)
        self.assertEqual(self.gridder.data.shape[2], self.n)
        # test values of data
        vg = numpy.zeros((self.gridder.data.shape))
        norm = numpy.copy(vg)
        ix, iy, iz = self.x - 1, self.y - 1, self.z - 1
        for i in range(ix-1, ix+2):
            for j in range(iy-1, iy+2):
                for k in range(iz-1, iz+2):
                    idx1 = i
                    idx2 = j
                    idx3 = k
                    n = 1/8.
                    if abs(i - ix) > 0:
                        n /= 2.
                    if abs(j - iy) > 0:
                        n /= 2.
                    if abs(k - iz) > 0:
                        n /= 2.
                    if i < 0:
                        idx1 = 0
                    elif i >= self.n:
                        idx1 = -1
                    if j < 0:
                        idx2 = 0
                    elif j >= self.n:
                        idx2 = -1
                    if k < 0:
                        idx3 = 0
                    elif k >= self.n:
                        idx3 = -1
                    norm[idx1, idx2, idx3] += n
                    vg[idx1, idx2, idx3] = self.data

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    self.assertAlmostEqual(self.gridder.data[i, j, k],
                                           vg[i, j, k],
                                           places=12)
                    self.assertAlmostEqual(self.gridder._gnorm[i, j, k],
                                           norm[i, j, k],
                                           places=12)


if __name__ == '__main__':
    unittest.main()
