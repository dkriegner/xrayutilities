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

testfiletmp = 'p08tty_%05d.dat'
testfile = testfiletmp % 29
datadir = os.path.join(os.path.dirname(__file__), 'data')
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_TTY(unittest.TestCase):
    dshape = (102,)
    dmax = 1444999.0
    dmin = 314133.0
    motmax = 26.15
    motmin = 13.0
    tpos = 53
    dtpos = 342586.0
    motorname = 'om'
    countername = 'EigerInt'

    @classmethod
    def setUpClass(cls):
        cls.motor, cls.data = xu.io.gettty08_scan(testfiletmp, (29, 30),
                                                  cls.motorname, path=datadir)
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


if __name__ == '__main__':
    unittest.main()
