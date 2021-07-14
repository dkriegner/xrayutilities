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

import os.path
import unittest

import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'fastscan.spec.gz'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.io)")
class TestIO_FastScan(unittest.TestCase):
    dshape = (62500,)
    dshape_retrace = (61992,)
    dmax = 154795.0
    dmin = 3797.0
    xmax = 80.038498
    xmin = 40.283199
    ymin = 138.61099
    ymax = 180.02901
    cname = 'mpx4int'
    timer = 'timer'
    timermax = 1285.15

    @classmethod
    def setUp(cls):
        cls.fs = xu.io.FastScan(testfile, 3, path=datadir)

    def test_grid2D(self):
        N = 200
        self.fs.retrace_clean()
        g2d = self.fs.grid2D(N, N+1, gridrange=((self.xmin, self.xmax),
                                                (self.ymin, self.ymax)))
        self.assertEqual(g2d.data.shape, (N, N+1))
        self.assertAlmostEqual(g2d.xaxis.min(), self.xmin, places=4)
        self.assertAlmostEqual(g2d.xaxis.max(), self.xmax, places=4)
        self.assertAlmostEqual(g2d.yaxis.min(), self.ymin, places=4)
        self.assertAlmostEqual(g2d.yaxis.max(), self.ymax, places=4)

    def test_datashape(self):
        self.assertEqual(self.fs.data.shape, self.dshape)
        self.assertEqual(self.fs.xvalues.shape, self.dshape)
        self.assertEqual(self.fs.yvalues.shape, self.dshape)
        self.fs.retrace_clean()
        self.assertEqual(self.fs.data.shape, self.dshape_retrace)
        self.assertEqual(self.fs.xvalues.shape, self.dshape_retrace)
        self.assertEqual(self.fs.yvalues.shape, self.dshape_retrace)

    def test_datavalues(self):
        self.assertAlmostEqual(self.fs.motorposition(self.timer).max(),
                               self.timermax, places=2)
        self.assertAlmostEqual(self.fs.xvalues.min(), self.xmin, places=5)
        self.assertAlmostEqual(self.fs.xvalues.max(), self.xmax, places=5)
        self.assertAlmostEqual(self.fs.yvalues.min(), self.ymin, places=5)
        self.assertAlmostEqual(self.fs.yvalues.max(), self.ymax, places=5)
        self.assertAlmostEqual(self.fs.data[self.cname].min(), self.dmin,
                               places=5)
        self.assertAlmostEqual(self.fs.data[self.cname].max(), self.dmax,
                               places=5)


if __name__ == '__main__':
    unittest.main()
