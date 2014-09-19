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

class TestNpyGridder1D(unittest.TestCase):

    def setUp(self):
        self.num = numpy.random.randint(10,99)
        self.xmin = 1
        self.xmax = self.num
        self.x = numpy.linspace(self.xmin,self.xmax,num=self.num)
        self.data = numpy.random.rand(self.num)
        self.gridder = xu.npyGridder1D(self.num)
        self.gridder(self.x,self.data)

    def test_npygridder1d_axis(self):
        hist,bins = numpy.histogram(self.x,bins=self.num)
        x = (bins[0:-1]+bins[1:])/2.
        # test length of xaxis
        self.assertEqual(len(self.gridder.xaxis), self.num)
        # test values of xaxis
        for i in range(self.num):
            self.assertAlmostEqual(self.gridder.xaxis[i], x[i], places=12)

    def test_npygridder1d_data(self):
        # test length of data
        self.assertEqual(len(self.gridder.data), self.num)
        # test values of data
        for i in range(self.num):
            self.assertAlmostEqual(self.gridder.data[i], self.data[i], places=12)

if __name__ == '__main__':
        unittest.main()
