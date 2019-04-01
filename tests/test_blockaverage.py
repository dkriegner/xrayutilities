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


class TestBlockAverageFunctions(unittest.TestCase):

    def setUp(self):
        self.seq = numpy.random.rand(11)
        self.n = 3
        self.seq2d = numpy.random.rand(10, 15)
        self.n2d = (3, 4)

    def test_blockav1d(self):
        out = xu.blockAverage1D(self.seq, self.n)
        self.assertAlmostEqual(out[0], numpy.average(self.seq[0:self.n]))
        self.assertEqual(out.size, numpy.ceil(self.seq.size / float(self.n)))

    def test_blockav2d(self):
        out = xu.blockAverage2D(self.seq2d, self.n2d[0], self.n2d[1])
        self.assertAlmostEqual(
            out[0, 0],
            numpy.average(self.seq2d[0:self.n2d[0], 0:self.n2d[1]]))
        self.assertEqual(
            out.shape,
            (numpy.ceil(self.seq2d.shape[0] / float(self.n2d[0])),
             numpy.ceil(self.seq2d.shape[1] / float(self.n2d[1]))))

    def test_blockav_psd(self):
        out = xu.blockAveragePSD(self.seq2d, self.n)
        self.assertAlmostEqual(out[0, 0],
                               numpy.average(self.seq2d[0, 0:self.n]))
        self.assertEqual(
            out.shape,
            (self.seq2d.shape[0],
             numpy.ceil(self.seq2d.shape[1] / float(self.n))))


if __name__ == '__main__':
    unittest.main()
