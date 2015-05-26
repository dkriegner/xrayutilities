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

import os.path
import unittest

import xrayutilities as xu
import numpy

testfile = 'omega_mm.xrdml'
datadir = 'data'
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_XRDML(unittest.TestCase):
    dshape = (499,)
    dmax = 75052800.0
    dmin = 54.0
    motmax = 16.7872
    motmin = 11.8072
    tpos = 240
    dtpos = 1502.0

    @classmethod
    def setUpClass(cls):
        cls.xrdmlfile = xu.io.XRDMLFile(fullfilename)
        cls.data1 = cls.xrdmlfile.scan['detector']
        cls.motor, cls.data2 = xu.io.getxrdml_scan(fullfilename)

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data1.shape)
        self.assertEqual(self.dshape, self.data2.shape)
        self.assertEqual(self.dshape, self.motor.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.motmax, self.motor.max(), places=10)
        self.assertAlmostEqual(self.motmin, self.motor.min(), places=10)
        self.assertAlmostEqual(self.dmax, self.data1.max(), places=10)
        self.assertAlmostEqual(self.dmin, self.data1.min(), places=10)
        self.assertAlmostEqual(self.dtpos, self.data1[self.tpos], places=10)

    def test_datamethods(self):
        self.assertTrue(numpy.all(self.data1 == self.data2))


if __name__ == '__main__':
    unittest.main()
