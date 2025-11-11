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

import numpy
import xrayutilities as xu

testfile = "413396"
datadir = os.path.join(os.path.dirname(__file__), "data")
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(
    not os.path.isfile(fullfilename),
    "additional test data needed (http://xrayutilities.sf.io)",
)
class TestIO_numor(unittest.TestCase):
    dshape = (31,)
    dmax = 32823.0
    dmin = 64.0
    motmax = 18.34
    motmin = 16.26
    tpos = 12
    dtpos = 17480.0
    motorname = "omega"
    countername = "detector"

    @classmethod
    def setUpClass(cls):
        cls.motor, cls.data = xu.io.numor_scan(
            testfile, cls.motorname, path=datadir
        )
        cls.inte = cls.data[cls.countername]

    def test_datashape(self):
        self.assertEqual(self.dshape, self.data.shape)
        self.assertEqual(self.dshape, self.motor.shape)

    def test_datavalues(self):
        self.assertAlmostEqual(self.motmax, self.motor.max(), places=6)
        self.assertAlmostEqual(self.motmin, self.motor.min(), places=6)
        self.assertAlmostEqual(self.dmax, self.inte.max(), places=6)
        self.assertAlmostEqual(self.dmin, self.inte.min(), places=6)
        self.assertAlmostEqual(self.dtpos, self.inte[self.tpos], places=6)

    def test_equaldata(self):
        self.assertTrue(numpy.all(self.motor == self.data[self.motorname]))


if __name__ == "__main__":
    unittest.main()
