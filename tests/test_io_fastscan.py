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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import numpy
import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'fastscan.spec.gz'
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_FastScan(unittest.TestCase):
    dshape = (62500,)
    dmax = 154795.0
    dmin = 3797.0
    xmax = 80.038498
    xmin = 40.283199
    ymin = 138.61099
    ymax = 180.02901
    cname = 'mpx4int'

    @classmethod
    def setUpClass(cls):
        cls.fs = xu.io.FastScan(testfile, 3, path=datadir)

    def test_datashape(self):
        self.assertEqual(self.fs.data.shape, self.dshape)
        self.assertEqual(self.fs.xvalues.shape, self.dshape)
        self.assertEqual(self.fs.yvalues.shape, self.dshape)

    def test_datavalues(self):
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
