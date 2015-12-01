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

xu.config.VERBOSITY = 0  # make no output during test
testfile = 'rigaku_rsm.ras.gz'
datadir = 'data'
fullfilename = os.path.join(datadir, testfile)


@unittest.skipIf(not os.path.isfile(fullfilename),
                 "additional test data needed (http://xrayutilities.sf.net)")
class TestIO_Rigaku(unittest.TestCase):
    nscans = 401
    dshape = (301,)
    scannr = 21
    dmax = 187.0
    dmin = 101.0
    motmax = 29.90
    motmin = 26.90
    tpos = 34
    dtpos = 164.0
    motorname = 'TwoTheta'
    countername = 'int'

    @classmethod
    def setUpClass(cls):
        cls.rasfile = xu.io.RASFile(testfile, path=datadir)
        cls.sdata = cls.rasfile.scans[cls.scannr].data
        cls.motor, data = xu.io.getras_scan(testfile+'%s', '',
                                               cls.motorname, path=datadir)
        cls.inte = data[cls.countername]

    def test_datashape_ras(self):
        self.assertEqual(self.nscans, len(self.rasfile.scans))
        self.assertEqual(self.dshape, self.sdata.shape)

    def test_datashape_all(self):
        self.assertEqual(self.dshape[0]*self.nscans, self.inte.size)
        self.assertEqual(self.dshape[0]*self.nscans, self.motor.size)

    def test_datavalues(self):
        mot = self.sdata[self.motorname]
        inte = self.sdata[self.countername]
        self.assertAlmostEqual(self.motmax, mot.max(), places=6)
        self.assertAlmostEqual(self.motmin, mot.min(), places=6)
        self.assertAlmostEqual(self.dmax, inte.max(), places=6)
        self.assertAlmostEqual(self.dmin, inte.min(), places=6)
        self.assertAlmostEqual(self.dtpos, inte[self.tpos], places=6)

    def test_datamethods(self):
        self.assertTrue(numpy.all(self.sdata[self.countername] ==
                                  self.inte[self.scannr*self.dshape[0]:
                                            (self.scannr+1)*self.dshape[0]]))


if __name__ == '__main__':
    unittest.main()
